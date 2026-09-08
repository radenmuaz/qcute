"""JAX/pmap port of image_gen_cifar/run_causalattn.py. Plain functional JAX (params as a
nested dict pytree, pure forward functions), optax for the optimizer, jax.pmap for
data-parallel training across every local device (default -- pass --n_devices to cap it).

Same architecture as the PyTorch original: encoder is a 3-level CAUSAL LM hierarchy over
the whole row-major-flattened image at once (strides=(2,4,4), product 32 == image width,
so level2's 32 codes land one-per-row). Each level's code head mean-pools/last-idx-pools
its stride-window before a product-quantized (PQ) categorical code (code_vocab=16 per
chunk, pq_chunks=4, gumbel-hard STE). Levels 1/2 also carry an NTP head so they can act as
free-running KV-cached generative priors (sample_codes()) -- level0 has none, its
generative role is the Decoder. Decoder is the same causal, lag-1-row, column-batched
(SISO/MIMO/grouped via col_group_size), GQA, BOS-bootstrapped GPT-style transformer as the
no-quant/full-attention variants, except it now looks codes up through embedding tables
(dec_l2_embed/dec_l1_embed/dec_l0_embed) instead of using a continuous vector directly.

Generation is the PyTorch original's two-phase structure (no per-row interleaving needed
here, since the encoder is itself a free-running generative prior): sample_codes() KV-cache
free-runs level2 then level1 to synthesize a whole code hierarchy from scratch, then the
decoder's own KV-cache generate() autoregresses bytes row by row from those codes. Both
caches use the same fixed-size preallocated jax.lax.dynamic_update_slice pattern as the
no-quant port. Not ported: prompting (real given rows/codes) -- unconditional only.
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
    n_kv_heads: int = None
    code_vocab: int = 16
    pq_chunks: int = 4
    strides: tuple = (2, 4, 4)
    code_extract_mode: str = "mean"
    decoder_mode: str = "seq"
    rope_base: float = 10000.0
    mlp_mult: int = 4
    ntp_aux_weight: float = 1.0
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
    return x * cos[None, None, :] + rotate_half(x) * sin[None, None, :]


def attention(x: jnp.ndarray, p: dict, n_heads: int, n_kv_heads: int, rope_base: float, causal: bool) -> jnp.ndarray:
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


def block_step(x_new: jnp.ndarray, p: dict, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, cfg: Config,
               T_max: int) -> tuple:
    attn_out, ck, cv = attention_step(rmsnorm(x_new, p["norm1"]), p["attn"], cache_k, cache_v, pos,
                                       cfg.n_heads, cfg.n_kv_heads, cfg.rope_base, T_max)
    x = x_new + attn_out
    x = x + swiglu(rmsnorm(x, p["norm2"]), p["mlp"])
    return x, ck, cv


def stack_step(x_new: jnp.ndarray, blocks: list, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, cfg: Config,
               T_max: int) -> tuple:
    """cache_k/v: (n_layers,Bc,n_kv_heads,T_max,hd)."""
    new_ck, new_cv = [], []
    x = x_new
    for i, blk in enumerate(blocks):
        x, ck_i, cv_i = block_step(x, blk, cache_k[i], cache_v[i], pos, cfg, T_max)
        new_ck.append(ck_i)
        new_cv.append(cv_i)
    return x, jnp.stack(new_ck), jnp.stack(new_cv)


def col_mix_forward(x: jnp.ndarray, p: dict, n_heads: int, group_size: int) -> jnp.ndarray:
    if group_size <= 1:
        return x
    B, R, C, D = x.shape
    g = group_size
    hd = D // n_heads
    xn = rmsnorm(x, p["norm"])
    qkv = xn @ p["qkv"]
    qkv = qkv.reshape(B, R, C // g, g, 3, n_heads, hd)
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

    def fold(t):
        t = jnp.moveaxis(t, -2, 3)
        return t.reshape(B * R * (C // g), n_heads, g, hd)

    qf, kf, vf = fold(q), fold(k), fold(v)
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    attn = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", attn, vf)
    y = y.reshape(B, R, C // g, n_heads, g, hd)
    y = jnp.moveaxis(y, 3, -2).reshape(B, R, C, D)
    return x + y @ p["out"]


def quantize_hard(logits: jnp.ndarray, rng=None) -> tuple:
    """Deterministic hard-argmax categorical with straight-through gradient."""
    soft = jax.nn.softmax(logits, axis=-1)
    idx = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=soft.dtype)
    code_soft = soft + jax.lax.stop_gradient(hard - soft)
    return code_soft, idx


def codebook_utilization(idx: jnp.ndarray, vocab: int) -> jnp.ndarray:
    flat = idx.reshape(-1, idx.shape[-1])
    utils = []
    for c in range(flat.shape[-1]):
        counts = jax.nn.one_hot(flat[:, c], vocab).sum(0)
        probs = counts / jnp.maximum(counts.sum(), 1)
        ent = -(probs * jnp.log(jnp.maximum(probs, 1e-9))).sum()
        utils.append(jnp.exp(ent) / vocab)
    return jnp.stack(utils).mean()


def code_embed(code: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
    """code: STE soft per-chunk one-hot (...,pq_chunks,V) or realized ids (...,pq_chunks)."""
    if jnp.issubdtype(code.dtype, jnp.integer):
        return table[code].sum(-2)
    return (code @ table).sum(-2)


# ---------------------------------------------------------------------------
# Encoder: 3-level CAUSAL hierarchy, PQ-quantized, with NTP heads
# ---------------------------------------------------------------------------

def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


def encoder_level_run(x: jnp.ndarray, p: dict, cfg: Config) -> jnp.ndarray:
    for blk in p["blocks"]:
        x = block_forward(x, blk, cfg, causal=True)
    return rmsnorm(x, p["ln_f"])


def encoder_level_pool(h: jnp.ndarray, stride: int, cfg: Config) -> jnp.ndarray:
    M, L, D = h.shape
    h = h.reshape(M, L // stride, stride, D)
    return jnp.mean(h, axis=2) if cfg.code_extract_mode == "mean" else h[:, :, -1, :]


def encoder_level_encode(x: jnp.ndarray, p: dict, stride: int, cfg: Config) -> tuple:
    h = encoder_level_run(x, p, cfg)
    pooled = encoder_level_pool(h, stride, cfg)
    logits = reshape_pq(pooled @ p["code_head"], cfg.pq_chunks, cfg.code_vocab)
    return quantize_hard(logits)


def encoder_level_ntp_logits(h: jnp.ndarray, p: dict, cfg: Config) -> jnp.ndarray:
    return reshape_pq(h @ p["ntp_head"], cfg.pq_chunks, cfg.code_vocab)


def ntp_loss_fn(x: jnp.ndarray, target_idx: jnp.ndarray, p: dict, cfg: Config, cond: jnp.ndarray = None,
                 y_embed: jnp.ndarray = None) -> jnp.ndarray:
    if cond is not None:
        x = x + cond
    if y_embed is not None:
        x = x + y_embed[:, None, :]
    h = encoder_level_run(x, p, cfg)
    logits = encoder_level_ntp_logits(h[:, :-1, :], p, cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    target = target_idx[:, 1:]
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


def image_encoder_forward(r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, p: dict, cfg: Config,
                           y_embed: jnp.ndarray = None) -> dict:
    B = r.shape[0]
    r, g, b = r.reshape(B, -1), g.reshape(B, -1), b.reshape(B, -1)
    x0 = p["r_embed"][r] + p["g_embed"][g] + p["b_embed"][b]
    code0_soft, code0_idx = encoder_level_encode(x0, p["level0"], cfg.strides[0], cfg)

    x1 = code_embed(code0_soft, p["code0_embed"])
    code1_soft, code1_idx = encoder_level_encode(x1, p["level1"], cfg.strides[1], cfg)
    cond1 = jnp.repeat(code_embed(code1_idx, p["code1_embed"]), cfg.strides[1], axis=1)
    ntp1 = ntp_loss_fn(x1, code0_idx, p["level1"], cfg, cond=cond1, y_embed=y_embed)

    x2 = code_embed(code1_soft, p["code1_embed"])
    code2_soft, code2_idx = encoder_level_encode(x2, p["level2"], cfg.strides[2], cfg)
    ntp2 = ntp_loss_fn(x2, code1_idx, p["level2"], cfg, y_embed=y_embed)

    vocab = cfg.code_vocab
    return dict(code0_soft=code0_soft, code1_soft=code1_soft, code2_soft=code2_soft, ntp_loss=ntp1 + ntp2,
                util0=codebook_utilization(code0_idx, vocab), util1=codebook_utilization(code1_idx, vocab),
                util2=codebook_utilization(code2_idx, vocab))


# ---------------------------------------------------------------------------
# Decoder: causal, lag-1-row, GQA, column-batched, BOS-bootstrapped, code-embedding tables
# ---------------------------------------------------------------------------

SLOT_L2, SLOT_L1, SLOT_L0, SLOT_R, SLOT_G, SLOT_B = range(6)
SLOT_L0_MTP, SLOT_RGB_MTP = 2, 3


def lagged_code_embeds(code2, code1, code0, p: dict, cfg: Config, y_embed: jnp.ndarray = None) -> tuple:
    img = cfg.img_size
    D = cfg.d_model
    l2e = code_embed(code2, p["dec_l2_embed"])
    l1e = code_embed(code1, p["dec_l1_embed"]).reshape(-1, img, code1.shape[1] // img, D)
    l0e = code_embed(code0, p["dec_l0_embed"]).reshape(-1, img, code0.shape[1] // img, D)
    B = l2e.shape[0]
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


def per_column_cond(code2, code1, code0, p: dict, cfg: Config, y_embed: jnp.ndarray = None) -> tuple:
    img = cfg.img_size
    l2e_lag, l1e_lag, l0e_lag = lagged_code_embeds(code2, code1, code0, p, cfg, y_embed)
    B, _, D = l2e_lag.shape
    n_l1, n_l0 = l1e_lag.shape[2], l0e_lag.shape[2]
    cols = jnp.arange(img)
    l1_g, l0_g = cols // (img // n_l1), cols // (img // n_l0)
    l2e_col = jnp.broadcast_to(l2e_lag[:, :, None, :], (B, img, img, D))
    l1e_col = l1e_lag[:, :, l1_g, :]
    l0e_col = l0e_lag[:, :, l0_g, :]
    l2e_col = col_mix_forward(l2e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    l1e_col = col_mix_forward(l1e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    l0e_col = col_mix_forward(l0e_col, p["col_mix"], cfg.n_heads, cfg.col_group_size)
    return l2e_col, l1e_col, l0e_col


def decoder_forward(code2, code1, code0, r, g, b, p: dict, cfg: Config, y_embed: jnp.ndarray = None) -> tuple:
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
    h = jnp.transpose(h, (0, 2, 1, 3, 4))

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
# Full model (training)
# ---------------------------------------------------------------------------

def model_forward(params: dict, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray, cfg: Config) -> tuple:
    y_embed = params["class_embed"][y] if cfg.class_conditional else None
    enc = image_encoder_forward(r, g, b, params["encoder"], cfg, y_embed)
    loss, acc = decoder_forward(enc["code2_soft"], enc["code1_soft"], enc["code0_soft"], r, g, b,
                                 params["decoder"], cfg, y_embed)
    total = loss + cfg.ntp_aux_weight * enc["ntp_loss"]
    bpb = loss / jnp.log(2.0)
    return total, (bpb, acc, enc["util0"], enc["util1"], enc["util2"])


# ---------------------------------------------------------------------------
# KV-cached generation: two-phase (sample_codes free-runs the code hierarchy, then the
# decoder autoregresses bytes from it), matching the PyTorch original.
# ---------------------------------------------------------------------------

def sample_codes(params: dict, cfg: Config, B: int, greedy: bool = False, temperature: float = 1.0,
                  y_embed: jnp.ndarray = None, prompt_code1: jnp.ndarray = None,
                  prompt_code0: jnp.ndarray = None, seed: int = 0) -> tuple:
    """prompt_code1/prompt_code0: real (teacher-forced) PQ code ids for a prefix of the
    level2/level1 AR loops (from encoding a real prompt image -- see encode_prompt_codes),
    or None for fully free-running. Teacher-forcing a prefix here is what lets a "prompt
    the first row" request produce genuinely real codes for that row instead of sampled
    ones, consistent with the decoder side also getting the real prompted bytes."""
    enc = params["encoder"]
    code0_len = cfg.img_size * cfg.img_size // cfg.strides[0]
    code1_len = code0_len // cfg.strides[1]
    D = cfg.d_model
    hd = D // cfg.n_heads
    pqc = cfg.pq_chunks
    n_prompt1 = prompt_code1.shape[1] if prompt_code1 is not None else 0
    n_prompt0 = prompt_code0.shape[1] if prompt_code0 is not None else 0
    rng = jax.random.PRNGKey(seed)

    def sample(logits, key):
        if greedy:
            return jnp.argmax(logits, axis=-1)
        return jax.random.categorical(key, logits / temperature, axis=-1)

    Hkv = cfg.n_kv_heads

    # level2 free-runs unconditionally over the code1 alphabet (teacher-forced for the
    # first n_prompt1 steps if a real prompt was given)
    cache_k = jnp.zeros((cfg.n_layers, B, Hkv, code1_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    step2 = jax.jit(lambda x, ck, cv, pos: stack_step(x, enc["level2"]["blocks"], ck, cv, pos, cfg, code1_len))
    x_new = jnp.broadcast_to(enc["level2_bos"], (B, D))
    if y_embed is not None:
        x_new = x_new + y_embed
    code1_idx = []
    for t in range(code1_len):
        h, cache_k, cache_v = step2(x_new, cache_k, cache_v, t)
        if t < n_prompt1:
            nxt = prompt_code1[:, t, :]
        else:
            h_n = rmsnorm(h, enc["level2"]["ln_f"])
            logits = reshape_pq(h_n @ enc["level2"]["ntp_head"], pqc, cfg.code_vocab)
            rng, k = jax.random.split(rng)
            nxt = sample(logits, k)
        code1_idx.append(nxt)
        x_new = code_embed(nxt, enc["code1_embed"])
        if y_embed is not None:
            x_new = x_new + y_embed
    code1_idx = jnp.stack(code1_idx, axis=1)  # (B,code1_len,pqc)

    # level1 free-runs over the code0 alphabet, conditioned on the just-sampled code1
    # (teacher-forced for the first n_prompt0 steps if a real prompt was given)
    cache_k = jnp.zeros((cfg.n_layers, B, Hkv, code0_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    step1 = jax.jit(lambda x, ck, cv, pos: stack_step(x, enc["level1"]["blocks"], ck, cv, pos, cfg, code0_len))
    x_new = jnp.broadcast_to(enc["level1_bos"], (B, D))
    if y_embed is not None:
        x_new = x_new + y_embed
    code0_idx = []
    for t in range(code0_len):
        h, cache_k, cache_v = step1(x_new, cache_k, cache_v, t)
        if t < n_prompt0:
            nxt = prompt_code0[:, t, :]
        else:
            h_n = rmsnorm(h, enc["level1"]["ln_f"])
            logits = reshape_pq(h_n @ enc["level1"]["ntp_head"], pqc, cfg.code_vocab)
            rng, k = jax.random.split(rng)
            nxt = sample(logits, k)
        code0_idx.append(nxt)
        cond = code_embed(code1_idx[:, t // cfg.strides[1]], enc["code1_embed"])
        x_new = code_embed(nxt, enc["code0_embed"]) + cond
        if y_embed is not None:
            x_new = x_new + y_embed
    code0_idx = jnp.stack(code0_idx, axis=1)  # (B,code0_len,pqc)

    x2 = code_embed(code1_idx, enc["code1_embed"])
    _, code2_idx = encoder_level_encode(x2, enc["level2"], cfg.strides[2], cfg)
    return code0_idx, code1_idx, code2_idx


def encode_prompt_codes(params: dict, cfg: Config, prompt_r: jnp.ndarray, prompt_g: jnp.ndarray,
                         prompt_b: jnp.ndarray, n_prompt_rows: int, y_embed: jnp.ndarray = None) -> tuple:
    """prompt_r/g/b: (B,img,img) a FULL real image (the causal encoder needs the whole
    flattened sequence, unlike the per-row full-attention forks) -- teacher-force-encodes
    it for real, then slices out just the n_prompt_rows worth of code1/code0 positions to
    hand to sample_codes() as its teacher-forced prefix."""
    img = cfg.img_size
    enc = image_encoder_forward(prompt_r, prompt_g, prompt_b, params["encoder"], cfg, y_embed)
    code1_idx_full = jnp.argmax(enc["code1_soft"], axis=-1)  # (B,code1_len,pqc)
    code0_idx_full = jnp.argmax(enc["code0_soft"], axis=-1)  # (B,code0_len,pqc)
    n1_per_row = code1_idx_full.shape[1] // img
    n0_per_row = code0_idx_full.shape[1] // img
    return code1_idx_full[:, :n_prompt_rows * n1_per_row, :], code0_idx_full[:, :n_prompt_rows * n0_per_row, :]


def decoder_generate(code0_idx, code1_idx, code2_idx, params: dict, cfg: Config, n: int, greedy: bool = False,
                      temperature: float = 1.0, y_embed: jnp.ndarray = None, prompt_r: jnp.ndarray = None,
                      prompt_g: jnp.ndarray = None, prompt_b: jnp.ndarray = None, seed: int = 0) -> jnp.ndarray:
    """prompt_r/g/b: (n,n_prompt,img) real bytes for the first n_prompt rows -- emitted
    as-is (not sampled). code0_idx/code1_idx/code2_idx must already reflect the real
    prompt for those rows too (see encode_prompt_codes + sample_codes' prompt_code*)."""
    dec = params["decoder"]
    img = cfg.img_size
    D = cfg.d_model
    n_slots = 4 if cfg.decoder_mode == "mtp" else 6
    T_max = img * n_slots
    Bc = n * img
    hd = D // cfg.n_heads
    n_prompt = prompt_r.shape[1] if prompt_r is not None else 0
    rng = jax.random.PRNGKey(seed)

    l2e_col, l1e_col, l0e_col = per_column_cond(code2_idx, code1_idx, code0_idx, dec, cfg, y_embed)
    l2e = jnp.transpose(l2e_col, (0, 2, 1, 3)).reshape(Bc, img, D)
    l1e = jnp.transpose(l1e_col, (0, 2, 1, 3)).reshape(Bc, img, D)
    l0e = jnp.transpose(l0e_col, (0, 2, 1, 3)).reshape(Bc, img, D)

    cache_k = jnp.zeros((cfg.n_layers, Bc, cfg.n_kv_heads, T_max, hd))
    cache_v = jnp.zeros_like(cache_k)
    slot_w = dec["slot_embed"]
    step_fn = jax.jit(lambda x, ck, cv, pos: stack_step(x, dec["blocks"], ck, cv, pos, cfg, T_max))

    def sample(logits, key):
        if greedy:
            return jnp.argmax(logits, axis=-1)
        return jax.random.categorical(key, logits / temperature, axis=-1)

    r_out = jnp.zeros((n, img, img), dtype=jnp.int32)
    g_out = jnp.zeros((n, img, img), dtype=jnp.int32)
    b_out = jnp.zeros((n, img, img), dtype=jnp.int32)
    pos = 0

    for row in range(img):
        x, cache_k, cache_v = step_fn(l2e[:, row] + slot_w[SLOT_L2], cache_k, cache_v, pos); pos += 1
        x, cache_k, cache_v = step_fn(l1e[:, row] + slot_w[SLOT_L1], cache_k, cache_v, pos); pos += 1
        l0_slot = slot_w[SLOT_L0_MTP if cfg.decoder_mode == "mtp" else SLOT_L0]
        x, cache_k, cache_v = step_fn(l0e[:, row] + l0_slot, cache_k, cache_v, pos); pos += 1
        h_seed = rmsnorm(x, dec["ln_f"])

        if row < n_prompt:
            r_row = prompt_r[:, row, :].reshape(-1)
            g_row = prompt_g[:, row, :].reshape(-1)
            b_row = prompt_b[:, row, :].reshape(-1)
            if cfg.decoder_mode == "mtp":
                rgb_e = dec["byte_embed"][r_row] + dec["byte_embed"][g_row] + dec["byte_embed"][b_row] + slot_w[SLOT_RGB_MTP]
                _, cache_k, cache_v = step_fn(rgb_e, cache_k, cache_v, pos); pos += 1
            else:
                _, cache_k, cache_v = step_fn(dec["byte_embed"][r_row] + slot_w[SLOT_R], cache_k, cache_v, pos); pos += 1
                _, cache_k, cache_v = step_fn(dec["byte_embed"][g_row] + slot_w[SLOT_G], cache_k, cache_v, pos); pos += 1
                _, cache_k, cache_v = step_fn(dec["byte_embed"][b_row] + slot_w[SLOT_B], cache_k, cache_v, pos); pos += 1
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
            x, cache_k, cache_v = step_fn(dec["byte_embed"][r_row] + slot_w[SLOT_R], cache_k, cache_v, pos); pos += 1
            h_r = rmsnorm(x, dec["ln_f"])
            g_row = sample(h_r @ dec["head_g"], kg)
            x, cache_k, cache_v = step_fn(dec["byte_embed"][g_row] + slot_w[SLOT_G], cache_k, cache_v, pos); pos += 1
            h_g = rmsnorm(x, dec["ln_f"])
            b_row = sample(h_g @ dec["head_b"], kb)
            _, cache_k, cache_v = step_fn(dec["byte_embed"][b_row] + slot_w[SLOT_B], cache_k, cache_v, pos); pos += 1

        r_out = r_out.at[:, row, :].set(r_row.reshape(n, img))
        g_out = g_out.at[:, row, :].set(g_row.reshape(n, img))
        b_out = b_out.at[:, row, :].set(b_row.reshape(n, img))

    return jnp.stack([r_out, g_out, b_out], axis=-1).clip(0, 255).astype(jnp.uint8)


def generate(params: dict, cfg: Config, n: int, greedy: bool = False, temperature: float = 1.0,
             y: jnp.ndarray = None, full_prompt_r: jnp.ndarray = None, full_prompt_g: jnp.ndarray = None,
             full_prompt_b: jnp.ndarray = None, n_prompt: int = 0, seed: int = 0) -> jnp.ndarray:
    """full_prompt_r/g/b: a FULL (n,img,img) real image -- the causal encoder needs the
    whole flattened sequence to encode anything, unlike the per-row full-attention forks
    -- of which the first `n_prompt` rows are used as the actual prompt (encoded for real
    to seed the code hierarchy, and their real bytes forced into the decoder); the rest
    of that image is ignored. n_prompt=0 (default) -> fully unconditional, no encoder
    call needed at all."""
    y_embed = params["class_embed"][y] if (cfg.class_conditional and y is not None) else None
    prompt_code1 = prompt_code0 = None
    prompt_r = prompt_g = prompt_b = None
    if n_prompt > 0:
        prompt_code1, prompt_code0 = encode_prompt_codes(params, cfg, full_prompt_r, full_prompt_g, full_prompt_b,
                                                           n_prompt, y_embed)
        prompt_r, prompt_g, prompt_b = full_prompt_r[:, :n_prompt, :], full_prompt_g[:, :n_prompt, :], full_prompt_b[:, :n_prompt, :]
    code0_idx, code1_idx, code2_idx = sample_codes(params, cfg, n, greedy=greedy, temperature=temperature,
                                                     y_embed=y_embed, prompt_code1=prompt_code1,
                                                     prompt_code0=prompt_code0, seed=seed)
    return decoder_generate(code0_idx, code1_idx, code2_idx, params, cfg, n, greedy=greedy, temperature=temperature,
                             y_embed=y_embed, prompt_r=prompt_r, prompt_g=prompt_g, prompt_b=prompt_b, seed=seed)


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


def init_encoder_level(rng, cfg: Config, has_ntp: bool) -> dict:
    D = cfg.d_model
    k_blocks, k_head, k_ntp = jax.random.split(rng, 3)
    block_keys = jax.random.split(k_blocks, cfg.n_layers)
    p = {
        "blocks": [init_block(bk, cfg) for bk in block_keys],
        "ln_f": jnp.ones((D,)),
        "code_head": jax.random.normal(k_head, (D, cfg.pq_chunks * cfg.code_vocab)) * 0.02,
    }
    if has_ntp:
        p["ntp_head"] = jax.random.normal(k_ntp, (D, cfg.pq_chunks * cfg.code_vocab)) * 0.02
    return p


def init_col_mix(rng, cfg: Config) -> dict:
    D = cfg.d_model
    k1, k2 = jax.random.split(rng)
    return {"norm": jnp.ones((D,)), "qkv": jax.random.normal(k1, (D, 3 * D)) * 0.02,
            "out": jax.random.normal(k2, (D, D)) * 0.02}


def init_params(rng, cfg: Config) -> dict:
    D = cfg.d_model
    keys = jax.random.split(rng, 20)
    encoder = {
        "r_embed": jax.random.normal(keys[0], (256, D)) * 0.02,
        "g_embed": jax.random.normal(keys[1], (256, D)) * 0.02,
        "b_embed": jax.random.normal(keys[2], (256, D)) * 0.02,
        "level0": init_encoder_level(keys[3], cfg, has_ntp=False),
        "code0_embed": jax.random.normal(keys[4], (cfg.code_vocab, D)) * 0.02,
        "level1": init_encoder_level(keys[5], cfg, has_ntp=True),
        "code1_embed": jax.random.normal(keys[6], (cfg.code_vocab, D)) * 0.02,
        "level2": init_encoder_level(keys[7], cfg, has_ntp=True),
        "level1_bos": jnp.zeros((D,)),
        "level2_bos": jnp.zeros((D,)),
    }
    n_slots = 4 if cfg.decoder_mode == "mtp" else 6
    dec_block_keys = jax.random.split(keys[8], cfg.n_layers)
    decoder = {
        "dec_l2_embed": jax.random.normal(keys[9], (cfg.code_vocab, D)) * 0.02,
        "dec_l1_embed": jax.random.normal(keys[10], (cfg.code_vocab, D)) * 0.02,
        "dec_l0_embed": jax.random.normal(keys[11], (cfg.code_vocab, D)) * 0.02,
        "byte_embed": jax.random.normal(keys[12], (256, D)) * 0.02,
        "slot_embed": jax.random.normal(keys[13], (n_slots, D)) * 0.02,
        "bos_l2": jnp.zeros((D,)),
        "bos_l1": jnp.zeros((D,)),
        "bos_l0": jnp.zeros((D,)),
        "col_mix": init_col_mix(keys[14], cfg),
        "blocks": [init_block(bk, cfg) for bk in dec_block_keys],
        "ln_f": jnp.ones((D,)),
        "head_r": jax.random.normal(keys[15], (D, 256)) * 0.02,
        "head_g": jax.random.normal(keys[16], (D, 256)) * 0.02,
        "head_b": jax.random.normal(keys[17], (D, 256)) * 0.02,
    }
    params = {"encoder": encoder, "decoder": decoder}
    if cfg.class_conditional:
        params["class_embed"] = jax.random.normal(keys[18], (cfg.n_classes, D)) * 0.02
    return params


def count_params(p) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train_step(cfg: Config, optimizer):
    def loss_fn(params, r, g, b, y):
        total, aux = model_forward(params, r, g, b, y, cfg)
        return total, aux

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


CONFIG_FIELDS = ("d_model", "n_layers", "n_heads", "code_vocab", "pq_chunks", "decoder_mode",
                  "col_group_size", "class_conditional", "n_classes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                    help="Python config file (image_gen_cifar_jax/configs/*.py) -- every run must have one")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_causalattn_jax")
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
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--code_vocab", type=int, default=16)
    p.add_argument("--pq_chunks", type=int, default=4)
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
            bpb, acc, u0, u1, u2 = eval_step(p_params, r, g, b, y)
            bpbs.append(float(bpb[0]))
            accs.append(float(acc[0]))
            if i >= 20:
                break
        bpb, acc = sum(bpbs) / len(bpbs), sum(accs) / len(accs)
        logger(f"val bpb={bpb:.4f} acc={acc:.4f}", val_bpb=bpb, val_acc=acc)
        return bpb

    train_prompt = train_np[:args.qual_gen_n]  # (qual_gen_n,img,img,3) FULL images
    val_prompt = val_np[:args.qual_gen_n]

    def run_qual_gen(epoch: int) -> None:
        single_params = jax.tree_util.tree_map(lambda x: x[0], p_params)
        gkw = dict(greedy=args.qual_gen_greedy, temperature=args.qual_gen_temperature, seed=epoch)

        modes = {
            "free": dict(n_prompt=0),
            "trainprompt": dict(n_prompt=1, full_prompt_r=jnp.array(train_prompt[..., 0]),
                                 full_prompt_g=jnp.array(train_prompt[..., 1]), full_prompt_b=jnp.array(train_prompt[..., 2])),
            "valprompt": dict(n_prompt=1, full_prompt_r=jnp.array(val_prompt[..., 0]),
                               full_prompt_g=jnp.array(val_prompt[..., 1]), full_prompt_b=jnp.array(val_prompt[..., 2])),
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
            state, aux = train_step(state, r, g, b, y)
            bpb, acc, u0, u1, u2 = aux
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb={float(bpb[0]):.4f} acc={float(acc[0]):.4f} "
                       f"util(l0/l1/l2)={float(u0[0]):.2f}/{float(u1[0]):.2f}/{float(u2[0]):.2f}",
                       epoch=epoch, step=step, train_bpb=float(bpb[0]), train_acc=float(acc[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            p_params, p_opt_state = state
            run_eval()
            run_qual_gen(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
