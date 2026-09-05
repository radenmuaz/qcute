"""Equinox port of run_ar_clockwork_v1.py (plain-dict-pytree JAX). Same ClockworkRNN-style
sandwich-strided AR baseline, dual RGB heads (parallel/sequential-MTP), NTP anchor head --
only the parameter representation changed, from a manually-threaded params dict to eqx.Module
classes (hyperparameters become static module fields instead of a `cfg` argument threaded
through every function). Shared primitives (RMSNorm/SwiGLU/Attention/Block) come from
eqx_common.py. Data loading, CLI/config plumbing, and the pure cfg-only helpers
(collector_of/reads_of/level_order) are imported unchanged from run_ar_clockwork_v1 -- no
reason to duplicate logic that doesn't touch params. Adds checkpoint save/resume (v1 had none).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from image_gen_cifar_jax.eqx_common import Attention, Block, RMSNorm, load_checkpoint, save_checkpoint
from image_gen_cifar_jax.run_ar_clockwork_v1 import (
    BatchIterator, Config, Logger, MODULE_DIR, REPO_ROOT, collector_of, level_order,
    load_cifar10, load_config_module, reads_of, save_sample_grid, warmup_schedule,
    write_resolved_config, _tuple_arg,
)


# ---------------------------------------------------------------------------
# Level (a clockwork level = stack of Blocks + final norm, unchanged semantics from v1)
# ---------------------------------------------------------------------------

class Level(eqx.Module):
    blocks: list
    ln_f: RMSNorm

    def __init__(self, key, d_model: int, n_layers: int, n_heads: int, n_kv_heads: int,
                 mlp_mult: int, rope_base: float):
        keys = jax.random.split(key, n_layers)
        self.blocks = [Block(k, d_model, n_heads, n_kv_heads, mlp_mult, rope_base) for k in keys]
        self.ln_f = RMSNorm(d_model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, tick_pos, T_max: int) -> tuple:
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(self.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], tick_pos, T_max)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)


# ---------------------------------------------------------------------------
# RGB output heads (parallel baseline + DeepSeek-MTP-style sequential chain)
# ---------------------------------------------------------------------------

class ParallelRGBHead(eqx.Module):
    head_r: jnp.ndarray
    head_g: jnp.ndarray
    head_b: jnp.ndarray
    img_size: int = eqx.field(static=True)

    def __init__(self, key, d_model: int, img_size: int):
        kr, kg, kb = jax.random.split(key, 3)
        self.head_r = jax.random.normal(kr, (d_model, img_size * 256)) * 0.02
        self.head_g = jax.random.normal(kg, (d_model, img_size * 256)) * 0.02
        self.head_b = jax.random.normal(kb, (d_model, img_size * 256)) * 0.02
        self.img_size = img_size

    def forward(self, h_out: jnp.ndarray) -> tuple:
        B, img, _ = h_out.shape
        logits_r = (h_out @ self.head_r).reshape(B, img, self.img_size, 256)
        logits_g = (h_out @ self.head_g).reshape(B, img, self.img_size, 256)
        logits_b = (h_out @ self.head_b).reshape(B, img, self.img_size, 256)
        return logits_r, logits_g, logits_b

    def forward_row(self, h_out_row: jnp.ndarray) -> tuple:
        n = h_out_row.shape[0]
        logits_r = (h_out_row @ self.head_r).reshape(n, self.img_size, 256)
        logits_g = (h_out_row @ self.head_g).reshape(n, self.img_size, 256)
        logits_b = (h_out_row @ self.head_b).reshape(n, self.img_size, 256)
        return logits_r, logits_g, logits_b


class SequentialRGBHead(eqx.Module):
    """DeepSeek-MTP-style: a tiny 1-layer causal decoder chains R->G->B per column via real
    byte embeddings (shared table, tied as the output head). Columns stay independent/parallel
    (a learned per-column embedding stands in for the parallel head's per-column weight row);
    only the R/G/B channel axis becomes a genuine 3-step causal chain instead of independent."""
    in_proj: jnp.ndarray
    col_embed: jnp.ndarray
    byte_embed: jnp.ndarray
    block: Block
    ln_f: RMSNorm
    img_size: int = eqx.field(static=True)
    mtp_dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, img_size: int, mtp_dim: int, mtp_n_heads: int,
                 mtp_mlp_mult: int, rope_base: float):
        k_in, k_col, k_byte, k_blk = jax.random.split(key, 4)
        self.in_proj = jax.random.normal(k_in, (d_model, mtp_dim)) * 0.02
        self.col_embed = jax.random.normal(k_col, (img_size, mtp_dim)) * 0.02
        self.byte_embed = jax.random.normal(k_byte, (256, mtp_dim)) * 0.02
        self.block = Block(k_blk, mtp_dim, mtp_n_heads, mtp_n_heads, mtp_mlp_mult, rope_base)
        self.ln_f = RMSNorm(mtp_dim)
        self.img_size, self.mtp_dim, self.n_heads, self.rope_base = img_size, mtp_dim, mtp_n_heads, rope_base

    def _run(self, seq: jnp.ndarray) -> jnp.ndarray:
        return self.ln_f(self.block(seq))

    def forward(self, h_out: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray) -> tuple:
        """Teacher-forced training pass, vectorized over B*img*img_size."""
        B, img, _ = h_out.shape
        ctx = (h_out @ self.in_proj)[:, :, None, :] + self.col_embed[None, None, :, :]
        embed_r, embed_g = self.byte_embed[r], self.byte_embed[g]
        seq_in = jnp.stack([ctx, embed_r, embed_g], axis=-2).reshape(B * img * self.img_size, 3, self.mtp_dim)
        out = self._run(seq_in)
        logits = (out @ self.byte_embed.T).reshape(B, img, self.img_size, 3, 256)
        return logits[..., 0, :], logits[..., 1, :], logits[..., 2, :]

    def generate(self, h_out_row: jnp.ndarray, sample_fn, rng) -> tuple:
        """Free-running per-row generation: recomputes fresh at T=1,2,3 (no KV cache needed --
        max length 3, tiny dims, cheap to just recompute)."""
        n = h_out_row.shape[0]
        ctx = (h_out_row @ self.in_proj)[:, None, :] + self.col_embed[None, :, :]
        ctx_flat = ctx.reshape(n * self.img_size, self.mtp_dim)

        out1 = self._run(ctx_flat[:, None, :])
        logits_r = (out1[:, 0] @ self.byte_embed.T).reshape(n, self.img_size, 256)
        rng, kr = jax.random.split(rng)
        r_col = sample_fn(logits_r, kr)
        embed_r = self.byte_embed[r_col.reshape(-1)]

        out2 = self._run(jnp.stack([ctx_flat, embed_r], axis=1))
        logits_g = (out2[:, 1] @ self.byte_embed.T).reshape(n, self.img_size, 256)
        rng, kg = jax.random.split(rng)
        g_col = sample_fn(logits_g, kg)
        embed_g = self.byte_embed[g_col.reshape(-1)]

        out3 = self._run(jnp.stack([ctx_flat, embed_r, embed_g], axis=1))
        logits_b = (out3[:, 2] @ self.byte_embed.T).reshape(n, self.img_size, 256)
        rng, kb = jax.random.split(rng)
        b_col = sample_fn(logits_b, kb)
        return r_col, g_col, b_col, rng


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def pool_row(r_row, g_row, b_row, r_embed, g_embed, b_embed) -> jnp.ndarray:
    e = r_embed[r_row] + g_embed[g_row] + b_embed[b_row]
    return jnp.mean(e, axis=-2)


class ClockworkModel(eqx.Module):
    r_embed: jnp.ndarray
    g_embed: jnp.ndarray
    b_embed: jnp.ndarray
    bootstrap_row: jnp.ndarray
    input_proj: list
    levels: list
    cond_proj: list
    ntp_head_r: jnp.ndarray
    ntp_head_g: jnp.ndarray
    ntp_head_b: jnp.ndarray
    rgb_head: eqx.Module
    class_embed: jnp.ndarray | None
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        keys = jax.random.split(key, 8 + 2 * n)
        self.r_embed = jax.random.normal(keys[0], (256, cfg.embed_dim)) * 0.02
        self.g_embed = jax.random.normal(keys[1], (256, cfg.embed_dim)) * 0.02
        self.b_embed = jax.random.normal(keys[2], (256, cfg.embed_dim)) * 0.02
        self.bootstrap_row = jax.random.normal(keys[3], (cfg.img_size, cfg.embed_dim)) * 0.02
        self.input_proj = [jax.random.normal(keys[4 + i], (cfg.embed_dim, cfg.d_model[i])) * 0.02
                            for i in range(n)]
        self.levels = [Level(keys[4 + n + i], cfg.d_model[i], cfg.n_layers[i], cfg.n_heads[i],
                              cfg.n_kv_heads[i], cfg.mlp_mult, cfg.rope_base) for i in range(n)]
        if cfg.head_type == "sequential":
            self.rgb_head = SequentialRGBHead(keys[4 + 2 * n], cfg.d_model[-1], cfg.img_size,
                                               cfg.mtp_dim, cfg.mtp_n_heads, cfg.mtp_mlp_mult, cfg.rope_base)
        else:
            self.rgb_head = ParallelRGBHead(keys[4 + 2 * n], cfg.d_model[-1], cfg.img_size)
        cond_key = keys[5 + 2 * n]
        cond_proj = []
        for i in range(n):
            row = {}
            for j in reads_of(cfg, i):
                cond_key, k = jax.random.split(cond_key)
                row[j] = jax.random.normal(k, (cfg.d_model[j], cfg.d_model[i])) * 0.02
            cond_proj.append(row)
        self.cond_proj = cond_proj
        cond_key, ntp_kr, ntp_kg, ntp_kb = jax.random.split(cond_key, 4)
        self.ntp_head_r = jax.random.normal(ntp_kr, (cfg.d_model[-1], 256)) * 0.02
        self.ntp_head_g = jax.random.normal(ntp_kg, (cfg.d_model[-1], 256)) * 0.02
        self.ntp_head_b = jax.random.normal(ntp_kb, (cfg.d_model[-1], 256)) * 0.02
        self.class_embed = (jax.random.normal(keys[6 + 2 * n], (cfg.n_classes, cfg.embed_dim)) * 0.02
                             if cfg.class_conditional else None)

    def __call__(self, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray) -> tuple:
        cfg = self.cfg
        B, img, _ = r.shape
        row_e = pool_row(r, g, b, self.r_embed, self.g_embed, self.b_embed)
        boot = jnp.mean(self.bootstrap_row, axis=0).reshape(1, 1, -1)
        boot = jnp.broadcast_to(boot, (B, 1, boot.shape[-1]))
        y_embed = self.class_embed[y] if cfg.class_conditional else None
        if y_embed is not None:
            row_e = row_e + y_embed[:, None, :]
            boot = boot + y_embed[:, None, :]
        x_in = jnp.concatenate([boot, row_e[:, :-1]], axis=1)

        held = [None] * len(cfg.strides)
        for i in level_order(cfg):
            stride_i = cfg.strides[i]
            idx = jnp.arange(0, img, stride_i)
            xi = x_in[:, idx] @ self.input_proj[i]
            for j in reads_of(cfg, i):
                xi = xi + held[j][:, idx] @ self.cond_proj[i][j]
            hi = self.levels[i](xi)
            held[i] = jnp.repeat(hi, stride_i, axis=1)[:, :img]

        h_out = held[collector_of(cfg)]
        if cfg.head_type == "sequential":
            logits_r, logits_g, logits_b = self.rgb_head.forward(h_out, r, g)
        else:
            logits_r, logits_g, logits_b = self.rgb_head.forward(h_out)

        def ce(logits, target):
            logp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

        loss_r, loss_g, loss_b = ce(logits_r, r), ce(logits_g, g), ce(logits_b, b)
        acc_main = (jnp.mean(jnp.argmax(logits_r, -1) == r) + jnp.mean(jnp.argmax(logits_g, -1) == g)
                    + jnp.mean(jnp.argmax(logits_b, -1) == b)) / 3
        loss_main = (loss_r + loss_g + loss_b) / 3

        ntp_logits_r = h_out @ self.ntp_head_r
        ntp_logits_g = h_out @ self.ntp_head_g
        ntp_logits_b = h_out @ self.ntp_head_b
        r0, g0, b0 = r[:, :, 0], g[:, :, 0], b[:, :, 0]
        ntp_loss_r, ntp_loss_g, ntp_loss_b = ce(ntp_logits_r, r0), ce(ntp_logits_g, g0), ce(ntp_logits_b, b0)
        acc_ntp = (jnp.mean(jnp.argmax(ntp_logits_r, -1) == r0) + jnp.mean(jnp.argmax(ntp_logits_g, -1) == g0)
                   + jnp.mean(jnp.argmax(ntp_logits_b, -1) == b0)) / 3
        loss_ntp = (ntp_loss_r + ntp_loss_g + ntp_loss_b) / 3

        loss = cfg.row_weight * loss_main + cfg.ntp_weight * loss_ntp
        return loss, (loss_main / jnp.log(2.0), acc_main, loss_ntp / jnp.log(2.0), acc_ntp)

    def generate(self, n: int, greedy: bool = False, temperature: float = 1.0, y: jnp.ndarray = None,
                 prompt_r: jnp.ndarray = None, prompt_g: jnp.ndarray = None, prompt_b: jnp.ndarray = None,
                 seed: int = 0) -> jnp.ndarray:
        cfg = self.cfg
        img = cfg.img_size
        n_levels = len(cfg.strides)
        n_prompt = prompt_r.shape[1] if prompt_r is not None else 0
        y_embed = self.class_embed[y] if (cfg.class_conditional and y is not None) else None
        order = level_order(cfg)
        collector = collector_of(cfg)
        rng = jax.random.PRNGKey(seed)

        def new_caches(i):
            hd = cfg.d_model[i] // cfg.n_heads[i]
            n_ticks = math.ceil(img / cfg.strides[i])
            shape = (cfg.n_layers[i], n, cfg.n_kv_heads[i], n_ticks, hd)
            return jnp.zeros(shape), jnp.zeros(shape)

        caches = [new_caches(i) for i in range(n_levels)]
        held = [None] * n_levels
        tick_pos = [0] * n_levels

        step_fns = {}
        for i in range(n_levels):
            n_ticks = math.ceil(img / cfg.strides[i])
            step_fns[i] = jax.jit(lambda x, ck, cv, pos, i=i, T=n_ticks: self.levels[i].step(x, ck, cv, pos, T))

        def sample(logits, key):
            if greedy:
                return jnp.argmax(logits, axis=-1)
            return jax.random.categorical(key, logits / temperature, axis=-1)

        x_input = jnp.mean(self.bootstrap_row, axis=0).reshape(1, -1)
        x_input = jnp.broadcast_to(x_input, (n, x_input.shape[-1]))
        if y_embed is not None:
            x_input = x_input + y_embed

        r_out = jnp.zeros((n, img, img), dtype=jnp.int32)
        g_out = jnp.zeros((n, img, img), dtype=jnp.int32)
        b_out = jnp.zeros((n, img, img), dtype=jnp.int32)

        for t in range(img):
            for i in order:
                if t % cfg.strides[i] == 0:
                    xi = x_input @ self.input_proj[i]
                    for j in reads_of(cfg, i):
                        xi = xi + held[j] @ self.cond_proj[i][j]
                    ck, cv = caches[i]
                    hi, ck, cv = step_fns[i](xi, ck, cv, tick_pos[i])
                    caches[i] = (ck, cv)
                    tick_pos[i] += 1
                    held[i] = hi

            if t < n_prompt:
                row_r, row_g, row_b = prompt_r[:, t, :], prompt_g[:, t, :], prompt_b[:, t, :]
            else:
                h_out = held[collector]
                if cfg.head_type == "sequential":
                    row_r, row_g, row_b, rng = self.rgb_head.generate(h_out, sample, rng)
                else:
                    logits_r, logits_g, logits_b = self.rgb_head.forward_row(h_out)
                    rng, kr, kg, kb = jax.random.split(rng, 4)
                    row_r, row_g, row_b = sample(logits_r, kr), sample(logits_g, kg), sample(logits_b, kb)
            r_out = r_out.at[:, t, :].set(row_r)
            g_out = g_out.at[:, t, :].set(row_g)
            b_out = b_out.at[:, t, :].set(row_b)

            x_input = pool_row(row_r, row_g, row_b, self.r_embed, self.g_embed, self.b_embed)
            if y_embed is not None:
                x_input = x_input + y_embed

        return jnp.stack([r_out, g_out, b_out], axis=-1).clip(0, 255).astype(jnp.uint8)


def save_compare_grid(gen: np.ndarray, gt: np.ndarray, path: Path, pad: int = 2) -> None:
    """gen/gt: (n,H,W,3) uint8 -- side-by-side [generated | ground truth] pairs, one row per
    example, for visually confirming overfitting (generated should closely match gt when the
    model has memorized a small training subset)."""
    from PIL import Image
    n, h, w, c = gen.shape
    grid = np.full((n * (h + pad) + pad, 2 * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        y = pad + i * (h + pad)
        grid[y:y + h, pad:pad + w] = gen[i]
        grid[y:y + h, 2 * pad + w:2 * pad + 2 * w] = gt[i]
    Image.fromarray(grid).save(path)


def count_params(model: ClockworkModel) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train_step(optimizer):
    def loss_fn(model, r, g, b, y):
        return model(r, g, b, y)

    def train_step(model, opt_state, r, g, b, y):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, r, g, b, y)
        grads = jax.lax.pmean(grads, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step():
    def eval_step(model, r, g, b, y):
        _, aux = model(r, g, b, y)
        return jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)

    return jax.pmap(eval_step, axis_name="d")


CONFIG_FIELDS = ("embed_dim", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "mlp_mult", "rope_base", "class_conditional", "n_classes", "row_weight", "ntp_weight",
                  "head_type", "mtp_dim", "mtp_n_heads", "mtp_mlp_mult")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                    help="Python config file (image_gen_cifar_jax/configs/*.py) -- every run must "
                         "have one, no bare-CLI-flags-only runs")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_ar_clockwork_jax")
    p.add_argument("--batch_size", type=int, default=8, help="per-device batch size")
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--checkpoint_every_epochs", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--train_subset_n", type=int, default=None,
                    help="truncate the train split to the first N images -- for overfit sanity checks")
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--qual_gen_greedy", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--qual_gen_temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=Config.embed_dim)
    p.add_argument("--d_model", type=_tuple_arg, default=Config.d_model)
    p.add_argument("--n_layers", type=_tuple_arg, default=Config.n_layers)
    p.add_argument("--n_heads", type=_tuple_arg, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=_tuple_arg, default=Config.n_kv_heads)
    p.add_argument("--strides", type=_tuple_arg, default=Config.strides)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
    p.add_argument("--class_conditional", type=lambda x: x.lower() != "false", default=Config.class_conditional)
    p.add_argument("--n_classes", type=int, default=Config.n_classes)
    p.add_argument("--row_weight", type=float, default=Config.row_weight)
    p.add_argument("--ntp_weight", type=float, default=Config.ntp_weight)
    p.add_argument("--head_type", type=str, default=Config.head_type, choices=["parallel", "sequential"])
    p.add_argument("--mtp_dim", type=int, default=Config.mtp_dim)
    p.add_argument("--mtp_n_heads", type=int, default=Config.mtp_n_heads)
    p.add_argument("--mtp_mlp_mult", type=int, default=Config.mtp_mlp_mult)

    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    if unknown:
        p.error(f"--config {pre_args.config} sets unknown field(s): {sorted(unknown)}")
    p.set_defaults(**config_vars)
    args = p.parse_args()

    n_devices = args.n_devices or jax.local_device_count()
    print(f"jax devices ({n_devices} used of {jax.local_device_count()} local): {jax.devices()}")

    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})

    (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np, train_labels = train_np[:args.train_subset_n], train_labels[:args.train_subset_n]
    train_iter = BatchIterator(train_np, train_labels, args.batch_size, n_devices, shuffle=True, seed=args.seed)
    val_iter = BatchIterator(val_np, val_labels, args.batch_size, n_devices, shuffle=False, seed=args.seed + 1)

    rng = jax.random.PRNGKey(args.seed)
    model = ClockworkModel(rng, cfg)
    n_params = count_params(model)

    lr_schedule = warmup_schedule(args.lr, args.warmup_steps)
    optimizer = optax.adamw(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 1
    step = 0
    if args.resume_from:
        model, opt_state, step, start_epoch = load_checkpoint(Path(args.resume_from), model, opt_state)
        start_epoch += 1

    def replicate(pytree):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape)
                                       if eqx.is_array(x) else x, pytree)

    p_model = replicate(model)
    p_opt_state = replicate(opt_state)

    train_step = make_train_step(optimizer)
    eval_step = make_eval_step()

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"run args: epochs={args.epochs} lr={args.lr} warmup_steps={args.warmup_steps} "
           f"batch_size={args.batch_size} n_devices={n_devices} resume_from={args.resume_from}")
    logger(f"params: {n_params / 1e6:.2f}M, devices={jax.devices()}")

    def run_eval() -> float:
        bpbs, accs, ntp_bpbs, ntp_accs = [], [], [], []
        for i, (r, g, b, y) in enumerate(val_iter):
            bpb, acc, ntp_bpb, ntp_acc = eval_step(p_model, r, g, b, y)
            bpbs.append(float(bpb[0]))
            accs.append(float(acc[0]))
            ntp_bpbs.append(float(ntp_bpb[0]))
            ntp_accs.append(float(ntp_acc[0]))
            if i >= 20:
                break
        bpb, acc = sum(bpbs) / len(bpbs), sum(accs) / len(accs)
        ntp_bpb, ntp_acc = sum(ntp_bpbs) / len(ntp_bpbs), sum(ntp_accs) / len(ntp_accs)
        logger(f"val bpb_main(32ahead)={bpb:.4f} acc_main={acc:.4f} bpb_ntp={ntp_bpb:.4f} acc_ntp={ntp_acc:.4f}",
               val_bpb_main=bpb, val_acc_main=acc, val_bpb_ntp=ntp_bpb, val_acc_ntp=ntp_acc)
        return bpb

    train_prompt = train_np[:args.qual_gen_n, 0:1, :, :]
    train_prompt_full = train_np[:args.qual_gen_n]  # (qual_gen_n,img,img,3) full images for the compare grid
    val_prompt = val_np[:args.qual_gen_n, 0:1, :, :]

    def run_qual_gen(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        gkw = dict(greedy=args.qual_gen_greedy, temperature=args.qual_gen_temperature, seed=epoch)

        modes = {
            "free": {},
            "trainprompt": dict(prompt_r=jnp.array(train_prompt[..., 0]), prompt_g=jnp.array(train_prompt[..., 1]),
                                 prompt_b=jnp.array(train_prompt[..., 2])),
            "valprompt": dict(prompt_r=jnp.array(val_prompt[..., 0]), prompt_g=jnp.array(val_prompt[..., 1]),
                               prompt_b=jnp.array(val_prompt[..., 2])),
        }
        for mode_name, extra in modes.items():
            samples = single_model.generate(args.qual_gen_n, **gkw, **extra)
            out_path = run_dir / f"samples_epoch{epoch}_{mode_name}.png"
            save_sample_grid(np.asarray(samples), out_path)
            if mode_name == "trainprompt":
                save_compare_grid(np.asarray(samples), np.asarray(train_prompt_full),
                                   run_dir / f"samples_epoch{epoch}_traincompare.png")
        logger(f"saved qual-gen samples (free/trainprompt/valprompt/traincompare) for epoch {epoch}")

    def run_checkpoint(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        single_opt_state = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_opt_state)
        ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch}"
        save_checkpoint(ckpt_dir, single_model, single_opt_state, step, epoch)
        logger(f"saved checkpoint at epoch {epoch} -> {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(train_iter, desc=f"epoch {epoch}/{args.epochs}")
        for r, g, b, y in pbar:
            p_model, p_opt_state, (bpb, acc, ntp_bpb, ntp_acc) = train_step(p_model, p_opt_state, r, g, b, y)
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb_main(32ahead)={float(bpb[0]):.4f} acc_main={float(acc[0]):.4f} "
                       f"bpb_ntp={float(ntp_bpb[0]):.4f} acc_ntp={float(ntp_acc[0]):.4f}",
                       epoch=epoch, step=step, train_bpb_main=float(bpb[0]), train_acc_main=float(acc[0]),
                       train_bpb_ntp=float(ntp_bpb[0]), train_acc_ntp=float(ntp_acc[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            run_eval()
            run_qual_gen(epoch)
        if epoch % args.checkpoint_every_epochs == 0 or epoch == args.epochs:
            run_checkpoint(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
