"""JAX/pmap port of image_gen_cifar/run_ar_clockwork.py. Plain functional JAX (params as a
nested dict/list pytree, pure forward functions), optax for the optimizer, jax.pmap for
data-parallel training across every local device.

ClockworkRNN-style decoder-only AR baseline: no encoder, no latent codes. The sequence runs
at ROW granularity (32 macro-steps for a 32x32 image): each row's 32 PQ-embedded pixels
(3x256 R/G/B tables, summed then mean-pooled) become one row-embedding, and the model
predicts the ENTIRE next row's 32x3 bytes in one parallel shot from the fastest level's
hidden state -- NTP "32 positions ahead", not per-byte. Row 0 is conditioned on a trainable
pixel line (small-noise init, NOT zeros -- see run_ar_clockwork.py's comment: an exact-zero
input hits RMSNorm's rsqrt(mean(x^2)+eps) pathology and blows up gradients through depth).

`strides` sets each level's CLOCK PERIOD over the row-sequence: level i only computes when
`row_idx % strides[i] == 0`; every other row costs zero compute, its state just carries
forward unchanged (true ClockworkRNN, not an approximation). strides[0] must be 1 (fastest,
drives the row prediction). Faster levels read every slower level's current held state as
additive conditioning (fast reads slow, never reverse). `d_model`/`n_layers`/`n_heads` are
per-level tuples.

Training computes each level's own strided subsequence in one parallel teacher-forced pass
(real rows all known upfront). Generation is a genuinely clocked, KV-cached per-row Python
loop: a level's step function is simply never called on its off-tick rows (real skipped
compute, not a masked no-op) -- each level's own cache only grows on its own ticks.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 32
    embed_dim: int = 256
    d_model: tuple = (256, 256, 128, 256)
    n_layers: tuple = (2, 2, 2, 2)
    n_heads: tuple = (4, 4, 4, 4)
    n_kv_heads: tuple = (None, None, None, None)
    strides: tuple = (1, 2, 4, 1)  # SANDWICH: strides[0]==1 (fast input level), non-decreasing
    # filling in between (slower/coarser summary levels), strides[-1]==1 (fast COLLECTOR level,
    # forced to tick every row -- it drives the prediction, reads every other level
    # unconditionally including same-speed level0, since it's the one place that must integrate
    # fast detail and every slow summary). Non-collector levels keep the plain clockwork rule
    # (read only strictly-slower levels among themselves).
    mlp_mult: int = 4
    rope_base: float = 10000.0
    class_conditional: bool = False
    n_classes: int = 10
    row_weight: float = 1.0    # weight on the main 32-ahead (full next-row) head's loss
    ntp_weight: float = 1.0    # weight on the auxiliary NTP (next single pixel) head's loss --
    # anchor task only, never used at generation time, dropped for sampling
    head_type: str = "parallel"  # "parallel" (independent R/G/B linear heads, columns+channels
    # all-at-once) or "sequential" (DeepSeek-MTP-style: tiny causal decoder chains R->G->B per
    # column via real byte embeddings; columns stay independent/parallel, only channels chain)
    mtp_dim: int = 64          # sequential head's internal width (unused if head_type=parallel)
    mtp_n_heads: int = 2       # plain MHA (no GQA -- already tiny, no KV cache used anyway)
    mtp_mlp_mult: int = 4

    def __post_init__(self):
        n = len(self.strides)
        assert n >= 2
        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n
        assert self.strides[0] == 1
        assert self.strides[-1] == 1
        assert all(self.strides[i] <= self.strides[i + 1] for i in range(n - 2))
        resolved_kv = []
        for i in range(n):
            kv = self.n_kv_heads[i] if self.n_kv_heads[i] is not None else max(1, self.n_heads[i] // 4)
            assert self.n_heads[i] % kv == 0
            assert self.d_model[i] % self.n_heads[i] == 0
            resolved_kv.append(kv)
        self.n_kv_heads = tuple(resolved_kv)
        assert self.head_type in ("parallel", "sequential")
        assert self.mtp_dim % self.mtp_n_heads == 0


# ---------------------------------------------------------------------------
# CIFAR-10 data (numpy, no torch)
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def load_cifar10(data_root: Path) -> tuple:
    data_root.mkdir(parents=True, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        import urllib.request
        print(f"downloading {CIFAR10_URL} -> {tar_path}")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    extract_dir = data_root / "cifar-10-batches-py"
    if not extract_dir.exists():
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_root)

    def load_batch(fname: str) -> tuple:
        with open(extract_dir / fname, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"labels"], dtype=np.int32)
        return images, labels

    train_batches = [load_batch(f"data_batch_{i}") for i in range(1, 6)]
    train = np.concatenate([b[0] for b in train_batches], axis=0)
    train_labels = np.concatenate([b[1] for b in train_batches], axis=0)
    test, test_labels = load_batch("test_batch")
    return (train, train_labels), (test, test_labels)


class BatchIterator:
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, n_devices: int,
                 shuffle: bool, seed: int = 0):
        self.images, self.labels = images, labels
        self.batch_size, self.n_devices = batch_size, n_devices
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.total = batch_size * n_devices

    def __iter__(self):
        n = len(self.images)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n - self.total + 1, self.total):
            sel = idx[start:start + self.total]
            img = self.images[sel].astype(np.int32)
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            y = self.labels[sel].astype(np.int32)

            def shard(x):
                return x.reshape(self.n_devices, self.batch_size, *x.shape[1:])

            yield shard(r), shard(g), shard(b), shard(y)


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------

def rmsnorm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x * weight


def swiglu(x: jnp.ndarray, p: dict) -> jnp.ndarray:
    return (jax.nn.silu(x @ p["gate"]) * (x @ p["up"])) @ p["down"]


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rope_cos_sin_pos(pos, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = pos * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def apply_rope_single(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos[None, :] + rotate_half(x) * sin[None, :]


def attention(x: jnp.ndarray, p: dict, n_heads: int, n_kv_heads: int, rope_base: float) -> jnp.ndarray:
    B, T, D = x.shape
    hd = D // n_heads
    qkv = x @ p["qkv"]
    q, k, v = jnp.split(qkv, [D, D + n_kv_heads * hd], axis=-1)
    q = q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_kv_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_kv_heads, hd).transpose(0, 2, 1, 3)
    q, k = rmsnorm(q, p["q_norm"]), rmsnorm(k, p["k_norm"])
    cos, sin = rope_cos_sin(T, hd, rope_base)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k, v = jnp.repeat(k, n_rep, axis=1), jnp.repeat(v, n_rep, axis=1)
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    logits = jnp.where(mask[None, None], logits, -1e9)
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ p["out"]


def block_forward(x: jnp.ndarray, p: dict, n_heads: int, n_kv_heads: int, rope_base: float) -> jnp.ndarray:
    x = x + attention(rmsnorm(x, p["norm1"]), p["attn"], n_heads, n_kv_heads, rope_base)
    x = x + swiglu(rmsnorm(x, p["norm2"]), p["mlp"])
    return x


def attention_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos,
                    n_heads: int, n_kv_heads: int, rope_base: float, T_max: int) -> tuple:
    Bc, D = x_new.shape
    hd = D // n_heads
    qkv = x_new @ p["qkv"]
    q, k, v = jnp.split(qkv, [D, D + n_kv_heads * hd], axis=-1)
    q, k, v = q.reshape(Bc, n_heads, hd), k.reshape(Bc, n_kv_heads, hd), v.reshape(Bc, n_kv_heads, hd)
    q, k = rmsnorm(q, p["q_norm"]), rmsnorm(k, p["k_norm"])
    cos, sin = rope_cos_sin_pos(pos, hd, rope_base)
    q, k = apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
    cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :], (0, 0, pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :], (0, 0, pos, 0))
    n_rep = n_heads // n_kv_heads
    k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
    v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
    valid = jnp.arange(T_max) <= pos
    logits = jnp.where(valid[None, None, :], logits, -1e9)
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bht,bhtd->bhd", attn, v_full).reshape(Bc, D)
    return y @ p["out"], cache_k, cache_v


def block_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos,
               n_heads: int, n_kv_heads: int, rope_base: float, T_max: int) -> tuple:
    attn_out, ck, cv = attention_step(rmsnorm(x_new, p["norm1"]), p["attn"], cache_k, cache_v, pos,
                                       n_heads, n_kv_heads, rope_base, T_max)
    x = x_new + attn_out
    x = x + swiglu(rmsnorm(x, p["norm2"]), p["mlp"])
    return x, ck, cv


def level_run(x: jnp.ndarray, p: dict, n_heads: int, n_kv_heads: int, rope_base: float) -> jnp.ndarray:
    for blk in p["blocks"]:
        x = block_forward(x, blk, n_heads, n_kv_heads, rope_base)
    return rmsnorm(x, p["ln_f"])


def level_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, tick_pos,
               n_heads: int, n_kv_heads: int, rope_base: float, T_max: int) -> tuple:
    new_ck, new_cv = [], []
    x = x_new
    for i, blk in enumerate(p["blocks"]):
        x, ck_i, cv_i = block_step(x, blk, cache_k[i], cache_v[i], tick_pos, n_heads, n_kv_heads, rope_base, T_max)
        new_ck.append(ck_i)
        new_cv.append(cv_i)
    return rmsnorm(x, p["ln_f"]), jnp.stack(new_ck), jnp.stack(new_cv)


# ---------------------------------------------------------------------------
# RGB output heads -- two swappable modules, both consume the trunk's per-row `h_out`
# and (during training) the real r/g/b targets; JAX functional style (init/apply pair
# instead of a class, matching every other component in this file).
# ---------------------------------------------------------------------------

def init_parallel_rgb_head(rng, d_model: int, img_size: int) -> dict:
    """Independent linear heads: h_out -> (img_size, 256) logits per channel, all columns
    and all channels produced in one shot from the same shared vector, no cross-conditioning."""
    kr, kg, kb = jax.random.split(rng, 3)
    return {
        "head_r": jax.random.normal(kr, (d_model, img_size * 256)) * 0.02,
        "head_g": jax.random.normal(kg, (d_model, img_size * 256)) * 0.02,
        "head_b": jax.random.normal(kb, (d_model, img_size * 256)) * 0.02,
    }


def parallel_rgb_head_forward(h_out: jnp.ndarray, p: dict, img_size: int) -> tuple:
    B, img, _ = h_out.shape
    logits_r = (h_out @ p["head_r"]).reshape(B, img, img_size, 256)
    logits_g = (h_out @ p["head_g"]).reshape(B, img, img_size, 256)
    logits_b = (h_out @ p["head_b"]).reshape(B, img, img_size, 256)
    return logits_r, logits_g, logits_b


def parallel_rgb_head_forward_row(h_out_row: jnp.ndarray, p: dict, img_size: int) -> tuple:
    """Same as above but for a single row (n, d_model) -> (n, img_size, 256) each, used
    during free-running generation."""
    n = h_out_row.shape[0]
    logits_r = (h_out_row @ p["head_r"]).reshape(n, img_size, 256)
    logits_g = (h_out_row @ p["head_g"]).reshape(n, img_size, 256)
    logits_b = (h_out_row @ p["head_b"]).reshape(n, img_size, 256)
    return logits_r, logits_g, logits_b


def init_sequential_rgb_head(rng, d_model: int, cfg: "Config") -> dict:
    """DeepSeek-MTP-style: a tiny 1-layer causal decoder chains R->G->B per column via real
    byte embeddings (shared table, tied as the output head). Columns stay independent/parallel
    (a learned per-column embedding stands in for the parallel head's per-column weight row);
    only the R/G/B channel axis becomes a genuine 3-step causal chain instead of independent."""
    k_in, k_col, k_byte, k_blk = jax.random.split(rng, 4)
    return {
        "mtp_in_proj": jax.random.normal(k_in, (d_model, cfg.mtp_dim)) * 0.02,
        "mtp_col_embed": jax.random.normal(k_col, (cfg.img_size, cfg.mtp_dim)) * 0.02,
        "mtp_byte_embed": jax.random.normal(k_byte, (256, cfg.mtp_dim)) * 0.02,
        "mtp_block": init_block(k_blk, cfg.mtp_dim, cfg.mtp_n_heads, cfg.mtp_n_heads, cfg.mtp_mlp_mult),
        "mtp_ln_f": jnp.ones((cfg.mtp_dim,)),
    }


def _mtp_run(seq: jnp.ndarray, p: dict, cfg: "Config") -> jnp.ndarray:
    x = block_forward(seq, p["mtp_block"], cfg.mtp_n_heads, cfg.mtp_n_heads, cfg.rope_base)
    return rmsnorm(x, p["mtp_ln_f"])


def sequential_rgb_head_forward(h_out: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, p: dict,
                                 cfg: "Config") -> tuple:
    """Teacher-forced training pass: one parallel length-3 causal sequence [ctx, embed(R), embed(G)]
    per (row, column), predicting [R, G, B] respectively -- vectorized over B*img*img_size."""
    B, img, _ = h_out.shape
    img_size = cfg.img_size
    ctx = (h_out @ p["mtp_in_proj"])[:, :, None, :] + p["mtp_col_embed"][None, None, :, :]
    embed_r = p["mtp_byte_embed"][r]
    embed_g = p["mtp_byte_embed"][g]
    seq_in = jnp.stack([ctx, embed_r, embed_g], axis=-2).reshape(B * img * img_size, 3, cfg.mtp_dim)
    out = _mtp_run(seq_in, p, cfg)
    logits = (out @ p["mtp_byte_embed"].T).reshape(B, img, img_size, 3, 256)
    return logits[..., 0, :], logits[..., 1, :], logits[..., 2, :]


def sequential_rgb_head_generate(h_out_row: jnp.ndarray, p: dict, cfg: "Config", sample_fn, rng) -> tuple:
    """Free-running per-row generation: recomputes the tiny decoder fresh at T=1,2,3 (no KV
    cache -- cheap given mtp_dim/n_heads are tiny), sampling R then G then B in sequence."""
    n = h_out_row.shape[0]
    img_size = cfg.img_size
    ctx = (h_out_row @ p["mtp_in_proj"])[:, None, :] + p["mtp_col_embed"][None, :, :]
    ctx_flat = ctx.reshape(n * img_size, cfg.mtp_dim)

    out1 = _mtp_run(ctx_flat[:, None, :], p, cfg)
    logits_r = (out1[:, 0] @ p["mtp_byte_embed"].T).reshape(n, img_size, 256)
    rng, kr = jax.random.split(rng)
    r_col = sample_fn(logits_r, kr)
    embed_r = p["mtp_byte_embed"][r_col.reshape(-1)]

    out2 = _mtp_run(jnp.stack([ctx_flat, embed_r], axis=1), p, cfg)
    logits_g = (out2[:, 1] @ p["mtp_byte_embed"].T).reshape(n, img_size, 256)
    rng, kg = jax.random.split(rng)
    g_col = sample_fn(logits_g, kg)
    embed_g = p["mtp_byte_embed"][g_col.reshape(-1)]

    out3 = _mtp_run(jnp.stack([ctx_flat, embed_r, embed_g], axis=1), p, cfg)
    logits_b = (out3[:, 2] @ p["mtp_byte_embed"].T).reshape(n, img_size, 256)
    rng, kb = jax.random.split(rng)
    b_col = sample_fn(logits_b, kb)

    return r_col, g_col, b_col, rng


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def pool_row(r_row: jnp.ndarray, g_row: jnp.ndarray, b_row: jnp.ndarray, p: dict) -> jnp.ndarray:
    e = p["r_embed"][r_row] + p["g_embed"][g_row] + p["b_embed"][b_row]
    return jnp.mean(e, axis=-2)


def collector_of(cfg: Config) -> int:
    return len(cfg.strides) - 1


def reads_of(cfg: Config, i: int) -> list:
    """Which levels level i reads as additive conditioning. The collector (last level)
    reads everyone else unconditionally; every other level keeps the plain clockwork
    rule (read only strictly-slower levels, among non-collector levels)."""
    n = len(cfg.strides)
    c = collector_of(cfg)
    if i == c:
        return [j for j in range(n) if j != i]
    return [j for j in range(n) if j != c and cfg.strides[j] > cfg.strides[i]]


def level_order(cfg: Config) -> list:
    """Non-collector levels slowest-to-fastest first, collector strictly last (it needs
    everyone else computed first)."""
    n = len(cfg.strides)
    non_collector = sorted(range(n - 1), key=lambda i: -cfg.strides[i])
    return non_collector + [collector_of(cfg)]


def model_forward(params: dict, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, cfg: Config) -> tuple:
    B, img, _ = r.shape
    row_e = pool_row(r, g, b, params)  # (B,img,embed_dim)
    boot = jnp.mean(params["bootstrap_row"], axis=0).reshape(1, 1, -1)
    boot = jnp.broadcast_to(boot, (B, 1, boot.shape[-1]))
    y_embed = params["class_embed"][y] if cfg.class_conditional else None
    if y_embed is not None:
        row_e = row_e + y_embed[:, None, :]
        boot = boot + y_embed[:, None, :]
    x_in = jnp.concatenate([boot, row_e[:, :-1]], axis=1)  # (B,img,embed_dim)

    held = [None] * len(cfg.strides)
    for i in level_order(cfg):
        stride_i = cfg.strides[i]
        idx = jnp.arange(0, img, stride_i)
        xi = x_in[:, idx] @ params["input_proj"][i]
        for j in reads_of(cfg, i):
            xi = xi + held[j][:, idx] @ params["cond_proj"][i][j]
        hi = level_run(xi, params["levels"][i], cfg.n_heads[i], cfg.n_kv_heads[i], cfg.rope_base)
        held[i] = jnp.repeat(hi, stride_i, axis=1)[:, :img]

    h_out = held[collector_of(cfg)]
    if cfg.head_type == "sequential":
        logits_r, logits_g, logits_b = sequential_rgb_head_forward(h_out, r, g, params["rgb_head"], cfg)
    else:
        logits_r, logits_g, logits_b = parallel_rgb_head_forward(h_out, params["rgb_head"], cfg.img_size)

    def ce(logits, target):
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

    loss_r, loss_g, loss_b = ce(logits_r, r), ce(logits_g, g), ce(logits_b, b)
    acc_main = (jnp.mean(jnp.argmax(logits_r, -1) == r) + jnp.mean(jnp.argmax(logits_g, -1) == g)
                + jnp.mean(jnp.argmax(logits_b, -1) == b)) / 3
    loss_main = (loss_r + loss_g + loss_b) / 3

    # NTP anchor head: same h_out (conditions on rows < t only, still causal), but predicts
    # just the immediate next pixel (row t, column 0) instead of the whole 32-ahead row --
    # an easier auxiliary task to help learning. Never used for generation/sampling.
    ntp_logits_r = h_out @ params["ntp_head_r"]
    ntp_logits_g = h_out @ params["ntp_head_g"]
    ntp_logits_b = h_out @ params["ntp_head_b"]
    r0, g0, b0 = r[:, :, 0], g[:, :, 0], b[:, :, 0]
    ntp_loss_r, ntp_loss_g, ntp_loss_b = ce(ntp_logits_r, r0), ce(ntp_logits_g, g0), ce(ntp_logits_b, b0)
    acc_ntp = (jnp.mean(jnp.argmax(ntp_logits_r, -1) == r0) + jnp.mean(jnp.argmax(ntp_logits_g, -1) == g0)
               + jnp.mean(jnp.argmax(ntp_logits_b, -1) == b0)) / 3
    loss_ntp = (ntp_loss_r + ntp_loss_g + ntp_loss_b) / 3

    loss = cfg.row_weight * loss_main + cfg.ntp_weight * loss_ntp
    return loss, (loss_main / jnp.log(2.0), acc_main, loss_ntp / jnp.log(2.0), acc_ntp)


def generate(params: dict, cfg: Config, n: int, greedy: bool = False, temperature: float = 1.0,
             y: jnp.ndarray = None, prompt_r: jnp.ndarray = None, prompt_g: jnp.ndarray = None,
             prompt_b: jnp.ndarray = None, seed: int = 0) -> jnp.ndarray:
    """greedy=False (default) samples categorically at `temperature` (1.0 = unscaled
    softmax); greedy=True argmaxes instead. prompt_r/g/b: (n, n_prompt, img_size) real
    bytes for the first n_prompt rows (or None -> fully free/unconditional from the
    trainable bootstrap row) -- prompted rows are emitted as-is (not sampled) and used
    to condition every later row exactly like the model's own output would be."""
    img = cfg.img_size
    n_levels = len(cfg.strides)
    n_prompt = prompt_r.shape[1] if prompt_r is not None else 0
    y_embed = params["class_embed"][y] if (cfg.class_conditional and y is not None) else None
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
        step_fns[i] = jax.jit(lambda x, ck, cv, pos, i=i, T=n_ticks: level_step(
            x, params["levels"][i], ck, cv, pos, cfg.n_heads[i], cfg.n_kv_heads[i], cfg.rope_base, T))

    def sample(logits, key):
        if greedy:
            return jnp.argmax(logits, axis=-1)
        return jax.random.categorical(key, logits / temperature, axis=-1)

    x_input = jnp.mean(params["bootstrap_row"], axis=0).reshape(1, -1)
    x_input = jnp.broadcast_to(x_input, (n, x_input.shape[-1]))
    if y_embed is not None:
        x_input = x_input + y_embed

    r_out = jnp.zeros((n, img, img), dtype=jnp.int32)
    g_out = jnp.zeros((n, img, img), dtype=jnp.int32)
    b_out = jnp.zeros((n, img, img), dtype=jnp.int32)

    for t in range(img):
        for i in order:
            if t % cfg.strides[i] == 0:
                xi = x_input @ params["input_proj"][i]
                for j in reads_of(cfg, i):
                    xi = xi + held[j] @ params["cond_proj"][i][j]
                ck, cv = caches[i]
                hi, ck, cv = step_fns[i](xi, ck, cv, tick_pos[i])
                caches[i] = (ck, cv)
                tick_pos[i] += 1
                held[i] = hi
            # else: held[i] unchanged, this level's step_fn is simply never called this row

        if t < n_prompt:
            row_r, row_g, row_b = prompt_r[:, t, :], prompt_g[:, t, :], prompt_b[:, t, :]
        else:
            h_out = held[collector]
            if cfg.head_type == "sequential":
                row_r, row_g, row_b, rng = sequential_rgb_head_generate(h_out, params["rgb_head"], cfg, sample, rng)
            else:
                logits_r, logits_g, logits_b = parallel_rgb_head_forward_row(h_out, params["rgb_head"], img)
                rng, kr, kg, kb = jax.random.split(rng, 4)
                row_r, row_g, row_b = sample(logits_r, kr), sample(logits_g, kg), sample(logits_b, kb)
        r_out = r_out.at[:, t, :].set(row_r)
        g_out = g_out.at[:, t, :].set(row_g)
        b_out = b_out.at[:, t, :].set(row_b)

        x_input = pool_row(row_r, row_g, row_b, params)
        if y_embed is not None:
            x_input = x_input + y_embed

    return jnp.stack([r_out, g_out, b_out], axis=-1).clip(0, 255).astype(jnp.uint8)


def save_sample_grid(samples: np.ndarray, path: Path, pad: int = 2) -> None:
    from PIL import Image
    n, h, w, c = samples.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i, img in enumerate(samples):
        row, col = divmod(i, cols)
        y, x = pad + row * (h + pad), pad + col * (w + pad)
        grid[y:y + h, x:x + w] = img
    Image.fromarray(grid).save(path)


# ---------------------------------------------------------------------------
# Param init
# ---------------------------------------------------------------------------

def init_block(rng, d_model, n_heads, n_kv_heads, mlp_mult) -> dict:
    hd = d_model // n_heads
    hidden = d_model * mlp_mult
    k = jax.random.split(rng, 5)
    return {
        "norm1": jnp.ones((d_model,)),
        "attn": {
            "qkv": jax.random.normal(k[0], (d_model, d_model + 2 * n_kv_heads * hd)) * 0.02,
            "out": jax.random.normal(k[1], (d_model, d_model)) * 0.02,
            "q_norm": jnp.ones((hd,)),
            "k_norm": jnp.ones((hd,)),
        },
        "norm2": jnp.ones((d_model,)),
        "mlp": {
            "gate": jax.random.normal(k[2], (d_model, hidden)) * 0.02,
            "up": jax.random.normal(k[3], (d_model, hidden)) * 0.02,
            "down": jax.random.normal(k[4], (hidden, d_model)) * 0.02,
        },
    }


def init_level(rng, d_model, n_layers, n_heads, n_kv_heads, mlp_mult) -> dict:
    keys = jax.random.split(rng, n_layers)
    return {"blocks": [init_block(k, d_model, n_heads, n_kv_heads, mlp_mult) for k in keys],
            "ln_f": jnp.ones((d_model,))}


def init_params(rng, cfg: Config) -> dict:
    n_levels = len(cfg.strides)
    keys = jax.random.split(rng, 10 + 2 * n_levels)
    params = {
        "r_embed": jax.random.normal(keys[0], (256, cfg.embed_dim)) * 0.02,
        "g_embed": jax.random.normal(keys[1], (256, cfg.embed_dim)) * 0.02,
        "b_embed": jax.random.normal(keys[2], (256, cfg.embed_dim)) * 0.02,
        "bootstrap_row": jax.random.normal(keys[3], (cfg.img_size, cfg.embed_dim)) * 0.02,
        "input_proj": [jax.random.normal(keys[4 + i], (cfg.embed_dim, cfg.d_model[i])) * 0.02
                        for i in range(n_levels)],
        "levels": [init_level(keys[4 + n_levels + i], cfg.d_model[i], cfg.n_layers[i], cfg.n_heads[i],
                               cfg.n_kv_heads[i], cfg.mlp_mult) for i in range(n_levels)],
    }
    if cfg.head_type == "sequential":
        params["rgb_head"] = init_sequential_rgb_head(keys[4 + 2 * n_levels], cfg.d_model[-1], cfg)
    else:
        params["rgb_head"] = init_parallel_rgb_head(keys[4 + 2 * n_levels], cfg.d_model[-1], cfg.img_size)
    cond_key = keys[7 + 2 * n_levels]
    cond_proj = []
    for i in range(n_levels):
        row = {}
        for j in reads_of(cfg, i):
            cond_key, k = jax.random.split(cond_key)
            row[j] = jax.random.normal(k, (cfg.d_model[j], cfg.d_model[i])) * 0.02
        cond_proj.append(row)
    params["cond_proj"] = cond_proj
    cond_key, ntp_kr, ntp_kg, ntp_kb = jax.random.split(cond_key, 4)
    params["ntp_head_r"] = jax.random.normal(ntp_kr, (cfg.d_model[-1], 256)) * 0.02
    params["ntp_head_g"] = jax.random.normal(ntp_kg, (cfg.d_model[-1], 256)) * 0.02
    params["ntp_head_b"] = jax.random.normal(ntp_kb, (cfg.d_model[-1], 256)) * 0.02
    if cfg.class_conditional:
        params["class_embed"] = jax.random.normal(keys[8 + 2 * n_levels], (cfg.n_classes, cfg.embed_dim)) * 0.02
    return params


def count_params(p) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train_step(cfg: Config, optimizer):
    def loss_fn(params, r, g, b, y):
        return model_forward(params, r, g, b, y, cfg)

    def train_step(state, r, g, b, y):
        params, opt_state = state
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, r, g, b, y)
        grads = jax.lax.pmean(grads, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step(cfg: Config):
    def eval_step(params, r, g, b, y):
        _, aux = model_forward(params, r, g, b, y, cfg)
        return jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)

    return jax.pmap(eval_step, axis_name="d")


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.text_f = open(run_dir / "run.log", "a")
        self.json_f = open(run_dir / "run.jsonl", "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        h, rem = divmod(elapsed_s, 3600)
        m, s = divmod(rem, 60)
        line = f"[{h:02d}:{m:02d}:{s:02d}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        rec = {"elapsed_s": elapsed_s, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(rec) + "\n")
        self.json_f.flush()


def _tuple_arg(s: str) -> tuple:
    return tuple(None if x.strip().lower() == "none" else int(x) for x in s.split(","))


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def write_resolved_config(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items()) if k != "config"]
    (run_dir / "resolved_config.py").write_text("\n".join(lines) + "\n")


CONFIG_FIELDS = ("embed_dim", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "mlp_mult", "rope_base", "class_conditional", "n_classes", "row_weight", "ntp_weight",
                  "head_type", "mtp_dim", "mtp_n_heads", "mtp_mlp_mult")


def warmup_schedule(peak_lr: float, warmup_steps: int):
    def schedule(step):
        return jnp.minimum(1.0, (step + 1) / max(warmup_steps, 1)) * peak_lr
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
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=1)
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
    train_iter = BatchIterator(train_np, train_labels, args.batch_size, n_devices, shuffle=True, seed=args.seed)
    val_iter = BatchIterator(val_np, val_labels, args.batch_size, n_devices, shuffle=False, seed=args.seed + 1)

    rng = jax.random.PRNGKey(args.seed)
    params = init_params(rng, cfg)
    n_params = count_params(params)

    lr_schedule = warmup_schedule(args.lr, args.warmup_steps)
    optimizer = optax.adamw(lr_schedule)
    opt_state = optimizer.init(params)

    def replicate(pytree):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape), pytree)

    p_params = replicate(params)
    p_opt_state = replicate(opt_state)

    train_step = make_train_step(cfg, optimizer)
    eval_step = make_eval_step(cfg)

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"run args: epochs={args.epochs} lr={args.lr} warmup_steps={args.warmup_steps} "
           f"batch_size={args.batch_size} n_devices={n_devices}")
    logger(f"params: {n_params / 1e6:.2f}M, devices={jax.devices()}")

    def run_eval() -> float:
        bpbs, accs, ntp_bpbs, ntp_accs = [], [], [], []
        for i, (r, g, b, y) in enumerate(val_iter):
            bpb, acc, ntp_bpb, ntp_acc = eval_step(p_params, r, g, b, y)
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

    # fixed single-row prompts for the two prompted qual-gen modes, picked once up front
    # so the same train/val image is used for every epoch's prompted sample (comparable over time)
    train_prompt = train_np[:args.qual_gen_n, 0:1, :, :]  # (qual_gen_n, 1, img, 3)
    val_prompt = val_np[:args.qual_gen_n, 0:1, :, :]

    def run_qual_gen(epoch: int) -> None:
        single_params = jax.tree_util.tree_map(lambda x: x[0], p_params)
        gkw = dict(greedy=args.qual_gen_greedy, temperature=args.qual_gen_temperature, seed=epoch)

        modes = {
            "free": {},
            "trainprompt": dict(prompt_r=jnp.array(train_prompt[..., 0]), prompt_g=jnp.array(train_prompt[..., 1]),
                                 prompt_b=jnp.array(train_prompt[..., 2])),
            "valprompt": dict(prompt_r=jnp.array(val_prompt[..., 0]), prompt_g=jnp.array(val_prompt[..., 1]),
                               prompt_b=jnp.array(val_prompt[..., 2])),
        }
        for mode_name, extra in modes.items():
            samples = generate(single_params, cfg, args.qual_gen_n, **gkw, **extra)
            out_path = run_dir / f"samples_epoch{epoch}_{mode_name}.png"
            save_sample_grid(np.asarray(samples), out_path)
        logger(f"saved qual-gen samples (free/trainprompt/valprompt) for epoch {epoch}")

    step = 0
    state = (p_params, p_opt_state)
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_iter, desc=f"epoch {epoch}/{args.epochs}")
        for r, g, b, y in pbar:
            state, (bpb, acc, ntp_bpb, ntp_acc) = train_step(state, r, g, b, y)
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb_main(32ahead)={float(bpb[0]):.4f} acc_main={float(acc[0]):.4f} "
                       f"bpb_ntp={float(ntp_bpb[0]):.4f} acc_ntp={float(ntp_acc[0]):.4f}",
                       epoch=epoch, step=step, train_bpb_main=float(bpb[0]), train_acc_main=float(acc[0]),
                       train_bpb_ntp=float(ntp_bpb[0]), train_acc_ntp=float(ntp_acc[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            p_params, p_opt_state = state
            run_eval()
            run_qual_gen(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
