"""JAX/pmap port of image_gen_cifar/run_fullattn_noquant.py. Plain functional JAX (params
as a nested dict pytree, pure forward functions), optax for the optimizer, jax.pmap for
data-parallel training across every local device (default -- pass --n_devices to cap it).

Same architecture as the PyTorch original: encoder is a 3-level per-row FULL (bidirectional,
non-causal) attention hierarchy with a plain Linear(D,D) passthrough at each level instead
of any quantization (no code_vocab/pq_chunks, no codebook) -- rows are folded into the batch
dim and never attend across each other, which is what keeps row r's code depending only on
row r's own pixels. Decoder is a causal, lag-1-row (row r's conditioning = row r-1's real
code; row 0 uses a learned BOS), column-batched (fold columns into batch) GPT-style
transformer with GQA and optional column-mix (col_group_size: SISO/MIMO/grouped).

Generation (`generate()`) is a genuinely interleaved per-row loop (encode row r's real/
just-realized bytes -> condition row r+1's decode -> repeat), KV-cached: a fixed-size
preallocated cache written via jax.lax.dynamic_update_slice + a valid-length mask (the
standard JAX pattern), not PyTorch's growing torch.cat -- one XLA compilation covers every
decode step since the cache shape never changes, only a traced `pos` scalar does. Not
ported: the prompting path (real given rows) -- unconditional generation only for now.
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
    d_model: int = 256
    n_layers: int = 1
    n_heads: int = 4
    n_kv_heads: int = None  # None = max(1, n_heads//4) GQA-by-default
    strides: tuple = (2, 4, 4)
    code_extract_mode: str = "mean"  # "mean" or "last_idx"
    decoder_mode: str = "seq"        # "seq" or "mtp"
    rope_base: float = 10000.0
    mlp_mult: int = 4
    col_group_size: int = 1
    class_conditional: bool = False
    n_classes: int = 10

    def __post_init__(self):
        assert math.prod(self.strides) == self.img_size
        assert self.d_model % self.n_heads == 0
        assert self.img_size % self.col_group_size == 0
        if self.n_kv_heads is None:
            self.n_kv_heads = max(1, self.n_heads // 4)
        assert self.n_heads % self.n_kv_heads == 0


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
    """Plain numpy batcher, no torch DataLoader -- yields (r,g,b,y) int32 arrays shaped
    for pmap: (n_devices, per_device_batch, ...)."""

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
            img = self.images[sel].astype(np.int32)  # (total,32,32,3)
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


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def attention(x: jnp.ndarray, p: dict, n_heads: int, n_kv_heads: int, rope_base: float,
              causal: bool) -> jnp.ndarray:
    """x: (B,T,D). GQA (n_kv_heads<n_heads repeats each KV head), Qwen3-style QK-norm+RoPE."""
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
        k = jnp.repeat(k, n_rep, axis=1)
        v = jnp.repeat(v, n_rep, axis=1)
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if causal:
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        logits = jnp.where(mask[None, None], logits, -1e9)
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ p["out"]


def block_forward(x: jnp.ndarray, p: dict, cfg: Config, causal: bool) -> jnp.ndarray:
    x = x + attention(rmsnorm(x, p["norm1"]), p["attn"], cfg.n_heads, cfg.n_kv_heads, cfg.rope_base, causal)
    x = x + swiglu(rmsnorm(x, p["norm2"]), p["mlp"])
    return x


def col_mix_forward(x: jnp.ndarray, p: dict, n_heads: int, group_size: int) -> jnp.ndarray:
    """x: (B,row,col,D) -- non-causal self-attention across COLUMNS, grouped. group_size<=1
    is a no-op (SISO); group_size==col count is full entanglement (MIMO)."""
    if group_size <= 1:
        return x
    B, R, C, D = x.shape
    g = group_size
    hd = D // n_heads
    xn = rmsnorm(x, p["norm"])
    qkv = xn @ p["qkv"]  # (B,R,C,3D)
    qkv = qkv.reshape(B, R, C // g, g, 3, n_heads, hd)
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]  # (B,R,C//g,g,H,hd)

    def fold(t):
        t = jnp.moveaxis(t, -2, 3)  # (B,R,C//g,H,g,hd)
        return t.reshape(B * R * (C // g), n_heads, g, hd)

    qf, kf, vf = fold(q), fold(k), fold(v)
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", attn, vf)  # (B*R*(C//g),H,g,hd)
    y = y.reshape(B, R, C // g, n_heads, g, hd)
    y = jnp.moveaxis(y, 3, -2).reshape(B, R, C, D)
    return x + y @ p["out"]


# ---------------------------------------------------------------------------
# Encoder: 3-level FULL-ATTENTION hierarchy, no NTP, no quantization, batched over ROWS
# ---------------------------------------------------------------------------

def encoder_level_forward(x: jnp.ndarray, p: dict, cfg: Config, stride: int) -> jnp.ndarray:
    """x: (M,L,D), M = independent row-instances (never attend across each other).
    Full (non-causal) attention within each row, then pool by stride, then a plain
    Linear(D,D) passthrough -- no quantization anywhere."""
    for blk in p["blocks"]:
        x = block_forward(x, blk, cfg, causal=False)
    h = rmsnorm(x, p["ln_f"])
    M, L, D = h.shape
    h = h.reshape(M, L // stride, stride, D)
    pooled = jnp.mean(h, axis=2) if cfg.code_extract_mode == "mean" else h[:, :, -1, :]
    return pooled @ p["code_head"]


def encode_rows(r_rows: jnp.ndarray, g_rows: jnp.ndarray, b_rows: jnp.ndarray, p: dict, cfg: Config) -> tuple:
    x0 = p["r_embed"][r_rows] + p["g_embed"][g_rows] + p["b_embed"][b_rows]
    code0 = encoder_level_forward(x0, p["level0"], cfg, cfg.strides[0])
    code1 = encoder_level_forward(code0, p["level1"], cfg, cfg.strides[1])
    code2 = encoder_level_forward(code1, p["level2"], cfg, cfg.strides[2])
    return code0, code1, code2.squeeze(1)


def image_encoder_forward(r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, p: dict, cfg: Config) -> tuple:
    B, img, _ = r.shape
    r_rows, g_rows, b_rows = r.reshape(B * img, img), g.reshape(B * img, img), b.reshape(B * img, img)
    code0, code1, code2 = encode_rows(r_rows, g_rows, b_rows, p, cfg)
    code0 = code0.reshape(B, img * code0.shape[1], -1)
    code1 = code1.reshape(B, img * code1.shape[1], -1)
    code2 = code2.reshape(B, img, -1)
    return code0, code1, code2


# ---------------------------------------------------------------------------
# Decoder: causal, lag-1-row, GQA, column-batched (SISO/MIMO/grouped), BOS-bootstrapped
# ---------------------------------------------------------------------------

SLOT_L2, SLOT_L1, SLOT_L0, SLOT_R, SLOT_G, SLOT_B = range(6)
SLOT_L0_MTP, SLOT_RGB_MTP = 2, 3


def lagged_code_embeds(code2: jnp.ndarray, code1: jnp.ndarray, code0: jnp.ndarray, p: dict, cfg: Config,
                        y_embed: jnp.ndarray = None) -> tuple:
    """Shift by one row (row0 -> learned BOS) so row r's conditioning only ever depends
    on rows < r -- valid chain-rule NLL, not a reconstruction bound (row r's own encoder
    code causally includes row r itself, via the full-attention pass over that row)."""
    img = cfg.img_size
    D = cfg.d_model
    l2e = code2  # (B,img,D) -- already continuous, no embedding lookup
    B = l2e.shape[0]
    l1e = code1.reshape(B, img, code1.shape[1] // img, D)
    l0e = code0.reshape(B, img, code0.shape[1] // img, D)
    bos_l2 = jnp.broadcast_to(p["bos_l2"], (B, 1, D))
    bos_l1 = jnp.broadcast_to(p["bos_l1"], (B, 1, l1e.shape[2], D))
    bos_l0 = jnp.broadcast_to(p["bos_l0"], (B, 1, l0e.shape[2], D))
    l2e_lag = jnp.concatenate([bos_l2, l2e[:, :-1]], axis=1)
    l1e_lag = jnp.concatenate([bos_l1, l1e[:, :-1]], axis=1)
    l0e_lag = jnp.concatenate([bos_l0, l0e[:, :-1]], axis=1)
    if y_embed is not None:
        l2e_lag = l2e_lag + y_embed[:, None, :]
        l1e_lag = l1e_lag + y_embed[:, None, None, :]
        l0e_lag = l0e_lag + y_embed[:, None, None, :]
    return l2e_lag, l1e_lag, l0e_lag


def per_column_cond(code2: jnp.ndarray, code1: jnp.ndarray, code0: jnp.ndarray, p: dict, cfg: Config,
                     y_embed: jnp.ndarray = None) -> tuple:
    img = cfg.img_size
    l2e_lag, l1e_lag, l0e_lag = lagged_code_embeds(code2, code1, code0, p, cfg, y_embed)
    B, _, D = l2e_lag.shape
    n_l1, n_l0 = l1e_lag.shape[2], l0e_lag.shape[2]
    cols = jnp.arange(img)
    l1_g = cols // (img // n_l1)
    l0_g = cols // (img // n_l0)
    l2e_col = jnp.broadcast_to(l2e_lag[:, :, None, :], (B, img, img, D))
    l1e_col = l1e_lag[:, :, l1_g, :]
    l0e_col = l0e_lag[:, :, l0_g, :]
    l2e_col = col_mix_forward(l2e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    l1e_col = col_mix_forward(l1e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    l0e_col = col_mix_forward(l0e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    return l2e_col, l1e_col, l0e_col


def decoder_forward(code2: jnp.ndarray, code1: jnp.ndarray, code0: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray,
                     b: jnp.ndarray, p: dict, cfg: Config, y_embed: jnp.ndarray = None) -> tuple:
    """Teacher-forced training pass. r,g,b: (B,img,img) ground-truth bytes, [row,col]."""
    img = cfg.img_size
    D = cfg.d_model
    B = r.shape[0]
    l2e_col, l1e_col, l0e_col = per_column_cond(code2, code1, code0, p, cfg, y_embed)
    r_e, g_e, b_e = p["byte_embed"][r], p["byte_embed"][g], p["byte_embed"][b]

    if cfg.decoder_mode == "mtp":
        slots = jnp.stack([l2e_col, l1e_col, l0e_col, r_e + g_e + b_e], axis=3)
    else:
        slots = jnp.stack([l2e_col, l1e_col, l0e_col, r_e, g_e, b_e], axis=3)
    n_slots = slots.shape[3]
    slots = slots + p["slot_embed"][None, None, None, :, :]

    x = jnp.transpose(slots, (0, 2, 1, 3, 4)).reshape(B * img, img * n_slots, D)
    for blk in p["blocks"]:
        x = block_forward(x, blk, cfg, causal=True)
    h = rmsnorm(x, p["ln_f"])
    h = h.reshape(B, img, img, n_slots, D)
    h = jnp.transpose(h, (0, 2, 1, 3, 4))  # (B,row,col,slot,D)

    if cfg.decoder_mode == "mtp":
        h_seed = h[:, :, :, SLOT_L0_MTP, :]
        logits_r, logits_g, logits_b = h_seed @ p["head_r"], h_seed @ p["head_g"], h_seed @ p["head_b"]
    else:
        logits_r = h[:, :, :, SLOT_L0, :] @ p["head_r"]
        logits_g = h[:, :, :, SLOT_R, :] @ p["head_g"]
        logits_b = h[:, :, :, SLOT_G, :] @ p["head_b"]

    def ce(logits, target):
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

    loss_r, loss_g, loss_b = ce(logits_r, r), ce(logits_g, g), ce(logits_b, b)
    acc = (jnp.mean(jnp.argmax(logits_r, -1) == r) + jnp.mean(jnp.argmax(logits_g, -1) == g)
           + jnp.mean(jnp.argmax(logits_b, -1) == b)) / 3
    return (loss_r + loss_g + loss_b) / 3, acc


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def model_forward(params: dict, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, cfg: Config) -> tuple:
    y_embed = params["class_embed"][y] if cfg.class_conditional else None
    code0, code1, code2 = image_encoder_forward(r, g, b, params["encoder"], cfg)
    loss, acc = decoder_forward(code2, code1, code0, r, g, b, params["decoder"], cfg, y_embed)
    bpb = loss / jnp.log(2.0)
    return loss, (bpb, acc)


# ---------------------------------------------------------------------------
# KV-cached generation: genuinely interleaved per-row loop (encode row r's REAL/just-
# realized bytes -> condition row r+1's decode -> repeat), matching the PyTorch
# version's row_cond_from_codes/bos_row_cond/step exactly, ported to a fixed-size
# preallocated cache (jax.lax.dynamic_update_slice + a valid-length mask) instead of
# PyTorch's growing torch.cat -- the standard JAX KV-cache pattern. This is what saves
# the cost the naive path pays: a real KV cache never re-runs the block stack over the
# whole prefix at every step, it only ever projects the ONE new token's q/k/v and reuses
# cached k/v for everything before it.
# ---------------------------------------------------------------------------

def rope_cos_sin_pos(pos, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = pos * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope_single(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """x: (Bc,H,hd); cos/sin: (hd,)."""
    return x * cos[None, None, :] + rotate_half(x) * sin[None, None, :]


def attention_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos,
                    n_heads: int, n_kv_heads: int, rope_base: float, T_max: int) -> tuple:
    """x_new: (Bc,D) one new token. cache_k/cache_v: (Bc,n_kv_heads,T_max,hd) for this
    layer, written in place at index `pos` (a traced scalar -> one XLA compilation
    covers every step, not one per position)."""
    Bc, D = x_new.shape
    hd = D // n_heads
    qkv = x_new @ p["qkv"]
    q, k, v = jnp.split(qkv, [D, D + n_kv_heads * hd], axis=-1)
    q = q.reshape(Bc, n_heads, hd)
    k = k.reshape(Bc, n_kv_heads, hd)
    v = v.reshape(Bc, n_kv_heads, hd)
    q, k = rmsnorm(q, p["q_norm"]), rmsnorm(k, p["k_norm"])
    cos, sin = rope_cos_sin_pos(pos, hd, rope_base)
    q, k = apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
    cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :], (0, 0, pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :], (0, 0, pos, 0))
    n_rep = n_heads // n_kv_heads
    k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
    v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale  # (Bc,n_heads,T_max)
    valid = jnp.arange(T_max) <= pos
    logits = jnp.where(valid[None, None, :], logits, -1e9)
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bht,bhtd->bhd", attn, v_full).reshape(Bc, D)
    return y @ p["out"], cache_k, cache_v


def block_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, cfg: Config,
               T_max: int) -> tuple:
    attn_out, ck, cv = attention_step(rmsnorm(x_new, p["norm1"]), p["attn"], cache_k, cache_v, pos,
                                       cfg.n_heads, cfg.n_kv_heads, cfg.rope_base, T_max)
    x = x_new + attn_out
    x = x + swiglu(rmsnorm(x, p["norm2"]), p["mlp"])
    return x, ck, cv


def decoder_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, cfg: Config,
                  T_max: int) -> tuple:
    """cache_k/v: (n_layers,Bc,n_kv_heads,T_max,hd). Returns (h, new_cache_k, new_cache_v)."""
    new_ck, new_cv = [], []
    x = x_new
    for i, blk in enumerate(p["blocks"]):
        x, ck_i, cv_i = block_step(x, blk, cache_k[i], cache_v[i], pos, cfg, T_max)
        new_ck.append(ck_i)
        new_cv.append(cv_i)
    return rmsnorm(x, p["ln_f"]), jnp.stack(new_ck), jnp.stack(new_cv)


def row_cond_from_codes(code2_row: jnp.ndarray, code1_row: jnp.ndarray, code0_row: jnp.ndarray, p: dict,
                         cfg: Config, y_embed: jnp.ndarray = None) -> tuple:
    """ONE row's own (real or just-realized) codes -> (l2e,l1e,l0e) each (B*img,D), ready
    to condition the NEXT row's decode -- same per-column broadcast/grouping/col_mix as
    per_column_cond, just for a single row."""
    img = cfg.img_size
    D = cfg.d_model
    B = code2_row.shape[0]
    n_l1, n_l0 = code1_row.shape[1], code0_row.shape[1]
    cols = jnp.arange(img)
    l1_g, l0_g = cols // (img // n_l1), cols // (img // n_l0)
    l2e_col = jnp.broadcast_to(code2_row[:, None, :], (B, img, D))
    l1e_col = code1_row[:, l1_g, :]
    l0e_col = code0_row[:, l0_g, :]
    if y_embed is not None:
        l2e_col, l1e_col, l0e_col = (e + y_embed[:, None, :] for e in (l2e_col, l1e_col, l0e_col))
    l2e_col = col_mix_forward(l2e_col[:, None], p["col_mix"], cfg.n_heads, cfg.col_group_size)[:, 0]
    l1e_col = col_mix_forward(l1e_col[:, None], p["col_mix"], cfg.n_heads, cfg.col_group_size)[:, 0]
    l0e_col = col_mix_forward(l0e_col[:, None], p["col_mix"], cfg.n_heads, cfg.col_group_size)[:, 0]
    return l2e_col.reshape(B * img, D), l1e_col.reshape(B * img, D), l0e_col.reshape(B * img, D)


def bos_row_cond(p: dict, cfg: Config, B: int, y_embed: jnp.ndarray = None) -> tuple:
    """Row-0 bootstrap: learned BOS in place of a previous row's codes -- matches
    lagged_code_embeds's row-0 handling (one vector broadcast to every group, col_mixed)."""
    img = cfg.img_size
    D = cfg.d_model
    outs = []
    for key in ("bos_l2", "bos_l1", "bos_l0"):
        e = jnp.broadcast_to(p[key], (B, img, D))
        if y_embed is not None:
            e = e + y_embed[:, None, :]
        e = col_mix_forward(e[:, None], p["col_mix"], cfg.n_heads, cfg.col_group_size)[:, 0]
        outs.append(e.reshape(B * img, D))
    return tuple(outs)


def generate(params: dict, cfg: Config, n: int, greedy: bool = False, temperature: float = 1.0,
             y: jnp.ndarray = None, prompt_r: jnp.ndarray = None, prompt_g: jnp.ndarray = None,
             prompt_b: jnp.ndarray = None, seed: int = 0) -> jnp.ndarray:
    """Genuinely interleaved per-row generation, KV-cached. greedy=False (default)
    samples categorically at `temperature`; greedy=True argmaxes. prompt_r/g/b:
    (n, n_prompt, img) real bytes for the first n_prompt rows -- given rows are
    emitted as-is (not sampled) and encoded to condition every later row exactly like
    the model's own output would be; None (default) -> fully unconditional from BOS."""
    cfg_img = cfg.img_size
    D = cfg.d_model
    n_slots = 4 if cfg.decoder_mode == "mtp" else 6
    T_max = cfg_img * n_slots
    Bc = n * cfg_img
    dec = params["decoder"]
    hd = D // cfg.n_heads
    n_prompt = prompt_r.shape[1] if prompt_r is not None else 0
    rng = jax.random.PRNGKey(seed)

    y_embed = params["class_embed"][y] if (cfg.class_conditional and y is not None) else None
    cache_k = jnp.zeros((cfg.n_layers, Bc, cfg.n_kv_heads, T_max, hd))
    cache_v = jnp.zeros_like(cache_k)
    l2e_prev, l1e_prev, l0e_prev = bos_row_cond(dec, cfg, n, y_embed)
    slot_w = dec["slot_embed"]

    step_fn = jax.jit(lambda x, ck, cv, pos: decoder_step(x, dec, ck, cv, pos, cfg, T_max))

    def sample(logits, key):
        if greedy:
            return jnp.argmax(logits, axis=-1)
        return jax.random.categorical(key, logits / temperature, axis=-1)

    r_out = jnp.zeros((n, cfg_img, cfg_img), dtype=jnp.int32)
    g_out = jnp.zeros((n, cfg_img, cfg_img), dtype=jnp.int32)
    b_out = jnp.zeros((n, cfg_img, cfg_img), dtype=jnp.int32)
    pos = 0

    for row in range(cfg_img):
        h, cache_k, cache_v = step_fn(l2e_prev + slot_w[SLOT_L2], cache_k, cache_v, pos); pos += 1
        h, cache_k, cache_v = step_fn(l1e_prev + slot_w[SLOT_L1], cache_k, cache_v, pos); pos += 1
        l0_slot = slot_w[SLOT_L0_MTP if cfg.decoder_mode == "mtp" else SLOT_L0]
        h_seed, cache_k, cache_v = step_fn(l0e_prev + l0_slot, cache_k, cache_v, pos); pos += 1

        if row < n_prompt:
            r_row, g_row, b_row = prompt_r[:, row, :].reshape(-1), prompt_g[:, row, :].reshape(-1), prompt_b[:, row, :].reshape(-1)
            if cfg.decoder_mode == "mtp":
                rgb_e = dec["byte_embed"][r_row] + dec["byte_embed"][g_row] + dec["byte_embed"][b_row] + slot_w[SLOT_RGB_MTP]
                _, cache_k, cache_v = step_fn(rgb_e, cache_k, cache_v, pos); pos += 1
            else:
                x = dec["byte_embed"][r_row] + slot_w[SLOT_R]
                _, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1
                x = dec["byte_embed"][g_row] + slot_w[SLOT_G]
                _, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1
                x = dec["byte_embed"][b_row] + slot_w[SLOT_B]
                _, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1
        elif cfg.decoder_mode == "mtp":
            rng, kr, kg, kb = jax.random.split(rng, 4)
            r_row = sample(h_seed @ dec["head_r"], kr)
            g_row = sample(h_seed @ dec["head_g"], kg)
            b_row = sample(h_seed @ dec["head_b"], kb)
            rgb_e = dec["byte_embed"][r_row] + dec["byte_embed"][g_row] + dec["byte_embed"][b_row] + slot_w[SLOT_RGB_MTP]
            _, cache_k, cache_v = step_fn(rgb_e, cache_k, cache_v, pos); pos += 1
        else:
            rng, kr, kg, kb = jax.random.split(rng, 4)
            r_row = sample(h_seed @ dec["head_r"], kr)
            x = dec["byte_embed"][r_row] + slot_w[SLOT_R]
            h_r, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1
            g_row = sample(h_r @ dec["head_g"], kg)
            x = dec["byte_embed"][g_row] + slot_w[SLOT_G]
            h_g, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1
            b_row = sample(h_g @ dec["head_b"], kb)
            x = dec["byte_embed"][b_row] + slot_w[SLOT_B]
            _, cache_k, cache_v = step_fn(x, cache_k, cache_v, pos); pos += 1

        r_out = r_out.at[:, row, :].set(r_row.reshape(n, cfg_img))
        g_out = g_out.at[:, row, :].set(g_row.reshape(n, cfg_img))
        b_out = b_out.at[:, row, :].set(b_row.reshape(n, cfg_img))

        if row < cfg_img - 1:
            code0_r, code1_r, code2_r = encode_rows(r_row.reshape(n, cfg_img), g_row.reshape(n, cfg_img),
                                                      b_row.reshape(n, cfg_img), params["encoder"], cfg)
            l2e_prev, l1e_prev, l0e_prev = row_cond_from_codes(code2_r, code1_r, code0_r, dec, cfg, y_embed)

    return jnp.stack([r_out, g_out, b_out], axis=-1).clip(0, 255).astype(jnp.uint8)  # (n,img,img,3)


def save_sample_grid(samples: np.ndarray, path: Path, pad: int = 2) -> None:
    """samples: (n,H,W,3) uint8 -- tile into a near-square grid on a white background."""
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

def init_block(rng, cfg: Config) -> dict:
    D, Hkv, hd = cfg.d_model, cfg.n_kv_heads, cfg.d_model // cfg.n_heads
    hidden = D * cfg.mlp_mult
    k = jax.random.split(rng, 5)
    return {
        "norm1": jnp.ones((D,)),
        "attn": {
            "qkv": jax.random.normal(k[0], (D, D + 2 * Hkv * hd)) * 0.02,
            "out": jax.random.normal(k[1], (D, D)) * 0.02,
            "q_norm": jnp.ones((hd,)),
            "k_norm": jnp.ones((hd,)),
        },
        "norm2": jnp.ones((D,)),
        "mlp": {
            "gate": jax.random.normal(k[2], (D, hidden)) * 0.02,
            "up": jax.random.normal(k[3], (D, hidden)) * 0.02,
            "down": jax.random.normal(k[4], (hidden, D)) * 0.02,
        },
    }


def init_encoder_level(rng, cfg: Config) -> dict:
    D = cfg.d_model
    k_blocks, k_head = jax.random.split(rng)
    block_keys = jax.random.split(k_blocks, cfg.n_layers)
    return {
        "blocks": [init_block(bk, cfg) for bk in block_keys],
        "ln_f": jnp.ones((D,)),
        "code_head": jax.random.normal(k_head, (D, D)) * 0.02,
    }


def init_col_mix(rng, cfg: Config) -> dict:
    D = cfg.d_model
    k1, k2 = jax.random.split(rng)
    return {
        "norm": jnp.ones((D,)),
        "qkv": jax.random.normal(k1, (D, 3 * D)) * 0.02,
        "out": jax.random.normal(k2, (D, D)) * 0.02,
    }


def init_params(rng, cfg: Config) -> dict:
    D = cfg.d_model
    keys = jax.random.split(rng, 16)
    encoder = {
        "r_embed": jax.random.normal(keys[0], (256, D)) * 0.02,
        "g_embed": jax.random.normal(keys[1], (256, D)) * 0.02,
        "b_embed": jax.random.normal(keys[2], (256, D)) * 0.02,
        "level0": init_encoder_level(keys[3], cfg),
        "level1": init_encoder_level(keys[4], cfg),
        "level2": init_encoder_level(keys[5], cfg),
    }
    n_slots = 4 if cfg.decoder_mode == "mtp" else 6
    dec_block_keys = jax.random.split(keys[6], cfg.n_layers)
    decoder = {
        "byte_embed": jax.random.normal(keys[7], (256, D)) * 0.02,
        "slot_embed": jax.random.normal(keys[8], (n_slots, D)) * 0.02,
        "bos_l2": jnp.zeros((D,)),
        "bos_l1": jnp.zeros((D,)),
        "bos_l0": jnp.zeros((D,)),
        "col_mix": init_col_mix(keys[9], cfg),
        "blocks": [init_block(bk, cfg) for bk in dec_block_keys],
        "ln_f": jnp.ones((D,)),
        "head_r": jax.random.normal(keys[10], (D, 256)) * 0.02,
        "head_g": jax.random.normal(keys[11], (D, 256)) * 0.02,
        "head_b": jax.random.normal(keys[12], (D, 256)) * 0.02,
    }
    params = {"encoder": encoder, "decoder": decoder}
    if cfg.class_conditional:
        params["class_embed"] = jax.random.normal(keys[13], (cfg.n_classes, D)) * 0.02
    return params


def count_params(p) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train_step(cfg: Config, optimizer):
    def loss_fn(params, r, g, b, y):
        loss, (bpb, acc) = model_forward(params, r, g, b, y, cfg)
        return loss, (bpb, acc)

    def train_step(state, r, g, b, y):
        params, opt_state = state
        (loss, (bpb, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, r, g, b, y)
        grads = jax.lax.pmean(grads, axis_name="d")
        bpb, acc = jax.lax.pmean(bpb, axis_name="d"), jax.lax.pmean(acc, axis_name="d")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (bpb, acc)

    return jax.pmap(train_step, axis_name="d")


def make_eval_step(cfg: Config):
    def eval_step(params, r, g, b, y):
        _, (bpb, acc) = model_forward(params, r, g, b, y, cfg)
        return jax.lax.pmean(bpb, axis_name="d"), jax.lax.pmean(acc, axis_name="d")

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


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def write_resolved_config(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items()) if k != "config"]
    (run_dir / "resolved_config.py").write_text("\n".join(lines) + "\n")


def warmup_schedule(peak_lr: float, warmup_steps: int):
    def schedule(step):
        return jnp.minimum(1.0, (step + 1) / max(warmup_steps, 1)) * peak_lr
    return schedule


CONFIG_FIELDS = ("d_model", "n_layers", "n_heads", "decoder_mode", "col_group_size",
                  "class_conditional", "n_classes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                    help="Python config file (image_gen_cifar_jax/configs/*.py) -- every run must have one")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_fullattn_noquant_jax")
    p.add_argument("--batch_size", type=int, default=8, help="per-device batch size")
    p.add_argument("--n_devices", type=int, default=None, help="default: all local devices")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--qual_gen_greedy", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--qual_gen_temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--decoder_mode", type=str, default="seq", choices=["seq", "mtp"])
    p.add_argument("--col_group_size", type=int, default=1)
    p.add_argument("--class_conditional", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--n_classes", type=int, default=10)

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
        bpbs, accs = [], []
        for i, (r, g, b, y) in enumerate(val_iter):
            bpb, acc = eval_step(p_params, r, g, b, y)
            bpbs.append(float(bpb[0]))
            accs.append(float(acc[0]))
            if i >= 20:
                break
        bpb, acc = sum(bpbs) / len(bpbs), sum(accs) / len(accs)
        logger(f"val bpb={bpb:.4f} acc={acc:.4f}", val_bpb=bpb, val_acc=acc)
        return bpb

    train_prompt = train_np[:args.qual_gen_n, 0:1, :, :]
    val_prompt = val_np[:args.qual_gen_n, 0:1, :, :]

    def run_qual_gen(epoch: int) -> None:
        single_params = jax.tree_util.tree_map(lambda x: x[0], p_params)  # all devices hold identical params
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
            state, (bpb, acc) = train_step(state, r, g, b, y)
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb={float(bpb[0]):.4f} acc={float(acc[0]):.4f}",
                       epoch=epoch, step=step, train_bpb=float(bpb[0]), train_acc=float(acc[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            p_params, p_opt_state = state
            run_eval()
            run_qual_gen(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
