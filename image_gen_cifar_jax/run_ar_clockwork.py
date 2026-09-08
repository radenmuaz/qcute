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
import json
import math
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from image_gen_cifar_jax.eqx_common import Attention, Block, RMSNorm, load_checkpoint, save_checkpoint, sinkgd
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


class DiffusionRGBHead(eqx.Module):
    """Discrete-diffusion-style masked head: each of the 3 R/G/B tokens for a column is
    independently replaced with a learned MASK embedding with probability mask_prob during
    training (else it keeps its real byte value), and a single BIDIRECTIONAL (non-causal)
    block attends over the 3 tokens (each already carrying the shared row/column context) to
    predict the ORIGINAL value at every masked position -- loss/accuracy computed only there,
    standard MLM convention (an unmasked position's "prediction" is trivial, it was just handed
    the answer). At generation time all three start fully masked (no real byte info at all),
    so the single-shot default pass reduces to independent-per-channel prediction from shared
    context only -- same behavior as the parallel head. True iterative multi-step remasking/
    refinement is NOT implemented here, only this single-shot default."""
    in_proj: jnp.ndarray
    col_embed: jnp.ndarray
    byte_embed: jnp.ndarray
    mask_embed: jnp.ndarray
    channel_embed: jnp.ndarray
    block: Block
    ln_f: RMSNorm
    img_size: int = eqx.field(static=True)
    mtp_dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    mask_prob: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, img_size: int, mtp_dim: int, mtp_n_heads: int,
                 mtp_mlp_mult: int, rope_base: float, mask_prob: float):
        k_in, k_col, k_byte, k_mask, k_chan, k_blk = jax.random.split(key, 6)
        self.in_proj = jax.random.normal(k_in, (d_model, mtp_dim)) * 0.02
        self.col_embed = jax.random.normal(k_col, (img_size, mtp_dim)) * 0.02
        self.byte_embed = jax.random.normal(k_byte, (256, mtp_dim)) * 0.02
        self.mask_embed = jax.random.normal(k_mask, (mtp_dim,)) * 0.02
        self.channel_embed = jax.random.normal(k_chan, (3, mtp_dim)) * 0.02
        self.block = Block(k_blk, mtp_dim, mtp_n_heads, mtp_n_heads, mtp_mlp_mult, rope_base)
        self.ln_f = RMSNorm(mtp_dim)
        self.img_size, self.mtp_dim, self.n_heads, self.mask_prob = img_size, mtp_dim, mtp_n_heads, mask_prob

    def _run(self, seq: jnp.ndarray) -> jnp.ndarray:
        return self.ln_f(self.block(seq, causal=False))

    def forward(self, h_out: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, rng) -> tuple:
        """Teacher-forced training pass, vectorized over B*img*img_size. Returns
        (logits_r,g,b each (B,img,img_size,256), mask_r,g,b each (B,img,img_size) bool --
        True where that channel's real byte was replaced by the MASK embedding)."""
        B, img, _ = h_out.shape
        ctx = (h_out @ self.in_proj)[:, :, None, :] + self.col_embed[None, None, :, :]
        mask = jax.random.bernoulli(rng, self.mask_prob, (B, img, self.img_size, 3))
        mask_r, mask_g, mask_b = mask[..., 0], mask[..., 1], mask[..., 2]

        def tok(byte_val, is_masked, chan_idx):
            real = self.byte_embed[byte_val]
            return jnp.where(is_masked[..., None], self.mask_embed, real) + self.channel_embed[chan_idx]

        seq_in = jnp.stack([ctx + tok(r, mask_r, 0), ctx + tok(g, mask_g, 1), ctx + tok(b, mask_b, 2)], axis=-2)
        seq_in = seq_in.reshape(B * img * self.img_size, 3, self.mtp_dim)
        out = self._run(seq_in)
        logits = (out @ self.byte_embed.T).reshape(B, img, self.img_size, 3, 256)
        return logits[..., 0, :], logits[..., 1, :], logits[..., 2, :], mask_r, mask_g, mask_b

    def generate(self, h_out_row: jnp.ndarray, sample_fn, rng) -> tuple:
        """Default single-shot inference: all three channels fully masked (no real byte info
        at all) -- reduces to independent-per-channel prediction from shared context, same
        behavior as the parallel head's forward_row."""
        n = h_out_row.shape[0]
        ctx = (h_out_row @ self.in_proj)[:, None, :] + self.col_embed[None, :, :]
        seq_in = jnp.stack([ctx + self.mask_embed + self.channel_embed[0],
                             ctx + self.mask_embed + self.channel_embed[1],
                             ctx + self.mask_embed + self.channel_embed[2]], axis=-2)
        seq_in = seq_in.reshape(n * self.img_size, 3, self.mtp_dim)
        out = self._run(seq_in)
        logits = (out @ self.byte_embed.T).reshape(n, self.img_size, 3, 256)
        rng, kr, kg, kb = jax.random.split(rng, 4)
        r_col = sample_fn(logits[:, :, 0, :], kr)
        g_col = sample_fn(logits[:, :, 1, :], kg)
        b_col = sample_fn(logits[:, :, 2, :], kb)
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
        elif cfg.head_type == "diffusion":
            self.rgb_head = DiffusionRGBHead(keys[4 + 2 * n], cfg.d_model[-1], cfg.img_size,
                                              cfg.mtp_dim, cfg.mtp_n_heads, cfg.mtp_mlp_mult, cfg.rope_base,
                                              cfg.mask_prob)
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

    def __call__(self, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, rng=None) -> tuple:
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

        def ce(logits, target):
            logp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

        def masked_ce_acc(logits, target, mask):
            logp = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
            correct = (jnp.argmax(logits, -1) == target).astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(nll * mask) / denom, jnp.sum(correct * mask) / denom

        if cfg.head_type == "sequential":
            logits_r, logits_g, logits_b = self.rgb_head.forward(h_out, r, g)
            loss_r, loss_g, loss_b = ce(logits_r, r), ce(logits_g, g), ce(logits_b, b)
            acc_main = (jnp.mean(jnp.argmax(logits_r, -1) == r) + jnp.mean(jnp.argmax(logits_g, -1) == g)
                        + jnp.mean(jnp.argmax(logits_b, -1) == b)) / 3
        elif cfg.head_type == "diffusion":
            logits_r, logits_g, logits_b, mask_r, mask_g, mask_b = self.rgb_head.forward(h_out, r, g, b, rng)
            mr, mg, mb = mask_r.astype(jnp.float32), mask_g.astype(jnp.float32), mask_b.astype(jnp.float32)
            loss_r, acc_r = masked_ce_acc(logits_r, r, mr)
            loss_g, acc_g = masked_ce_acc(logits_g, g, mg)
            loss_b, acc_b = masked_ce_acc(logits_b, b, mb)
            acc_main = (acc_r + acc_g + acc_b) / 3
        else:
            logits_r, logits_g, logits_b = self.rgb_head.forward(h_out)
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
                if cfg.head_type in ("sequential", "diffusion"):
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
    def loss_fn(model, r, g, b, y, rng):
        return model(r, g, b, y, rng)

    def train_step(model, opt_state, r, g, b, y, rng):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, r, g, b, y, rng)
        grads = jax.lax.pmean(grads, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step():
    def eval_step(model, r, g, b, y, rng):
        _, aux = model(r, g, b, y, rng)
        return jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)

    return jax.pmap(eval_step, axis_name="d")


CONFIG_FIELDS = ("embed_dim", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "mlp_mult", "rope_base", "class_conditional", "n_classes", "row_weight", "ntp_weight",
                  "head_type", "mtp_dim", "mtp_n_heads", "mtp_mlp_mult", "mask_prob")


def warmup_const_decay_schedule(peak_lr: float, warmup_steps: int, constant_steps: int, total_steps: int,
                                 min_lr_ratio: float = 0.01):
    """Linear warmup -> flat at peak_lr for constant_steps -> cosine decay down to
    peak_lr*min_lr_ratio over the remaining steps. For long "overfit as hard as possible"
    runs: flat-forever LR (the old warmup_schedule) oscillates once near a sharp minimum
    instead of settling into it -- decaying late in training takes smaller, more precise
    steps toward the memorized optimum."""
    decay_start = warmup_steps + constant_steps

    def schedule(step):
        warmup_lr = jnp.minimum(1.0, (step + 1) / max(warmup_steps, 1)) * peak_lr
        decay_total = max(total_steps - decay_start, 1)
        decay_frac = jnp.clip((step - decay_start) / decay_total, 0.0, 1.0)
        cosine = 0.5 * (1 + jnp.cos(jnp.pi * decay_frac))
        decayed_lr = (min_lr_ratio + (1 - min_lr_ratio) * cosine) * peak_lr
        return jnp.where(step < warmup_steps, warmup_lr, jnp.where(step < decay_start, peak_lr, decayed_lr))

    return schedule


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
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={},
                    help="extra kwargs forwarded to the optimizer constructor (optax.adamw or "
                         "eqx_common.sinkgd) -- set as a plain dict literal in the config file, "
                         "or a JSON string on the CLI, e.g. sinkgd's linear_lr_scale/sinkhorn_iters")
    p.add_argument("--lr_decay", type=lambda x: x.lower() != "false", default=False,
                    help="if true, use warmup + constant + cosine-decay (warmup_epochs/constant_epochs/"
                         "epochs, computed in steps from the actual per-epoch step count) instead of "
                         "the plain flat-after-warmup schedule")
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--constant_epochs", type=float, default=10.0)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
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
    p.add_argument("--head_type", type=str, default=Config.head_type,
                    choices=["parallel", "sequential", "diffusion"])
    p.add_argument("--mask_prob", type=float, default=Config.mask_prob)
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
    data_rng = jax.random.PRNGKey(args.seed + 1000)  # drives diffusion-head masking only

    if args.lr_decay:
        steps_per_epoch = len(train_np) // (args.batch_size * n_devices)
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = round(args.warmup_epochs * steps_per_epoch)
        constant_steps = round(args.constant_epochs * steps_per_epoch)
        lr_schedule = warmup_const_decay_schedule(args.lr, warmup_steps, constant_steps, total_steps,
                                                   args.min_lr_ratio)
        print(f"lr schedule: warmup {warmup_steps} steps, constant {constant_steps} steps, "
              f"cosine decay to {args.lr * args.min_lr_ratio:.2e} over remaining "
              f"{total_steps - warmup_steps - constant_steps} steps (total {total_steps})")
    else:
        lr_schedule = warmup_schedule(args.lr, args.warmup_steps)
    if args.optimizer == "sinkgd":
        optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
    else:
        optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay, **args.optimizer_kwargs)
    print(f"optimizer: {args.optimizer} kwargs={args.optimizer_kwargs}")
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
        nonlocal data_rng
        bpbs, accs, ntp_bpbs, ntp_accs = [], [], [], []
        for i, (r, g, b, y) in enumerate(val_iter):
            data_rng, step_rng = jax.random.split(data_rng)
            step_rngs = jax.random.split(step_rng, n_devices)
            bpb, acc, ntp_bpb, ntp_acc = eval_step(p_model, r, g, b, y, step_rngs)
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
            data_rng, step_rng = jax.random.split(data_rng)
            step_rngs = jax.random.split(step_rng, n_devices)
            p_model, p_opt_state, (bpb, acc, ntp_bpb, ntp_acc) = train_step(p_model, p_opt_state, r, g, b, y, step_rngs)
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
