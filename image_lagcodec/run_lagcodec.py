"""image_lagcodec -- a hardcoded, opinionated JAX/Equinox reimplementation of the ideas in
qcute/qcute_lagcodec/ for CIFAR-10 images: a hierarchical causal encoder producing PQ-quantized
codes at each level (each level ALSO trained with its own non-circular NTP loss, exactly like
qcute_lagcodec_encoder.py's documented design -- `h[:,p,:]` predicts `seq_repr[:,p+1]` directly,
no autoencoder circularity), and a decoder that autoregressively RECONSTRUCTS the flat
interleaved-RGB byte sequence (R0,G0,B0,R1,G1,B1,...,3072 bytes/image, true per-byte causal
AR -- no row-macro-step, no one-shot-32-pixel parallel prediction) conditioned additively on
those codes.

This is Task 1 only: a "fancy VQ-VAE" (hierarchical encoder + genuine per-byte AR decode).
Codes condition the decoder from their OWN block (own_code_min_lag=0, hardcoded for now --
qcute_lagcodec's own_code_min_lag=1 causal variant, which masks out the current block so
decode becomes genuine forward-chain prediction instead of a reconstruction bound, is NOT
implemented yet, deliberately deferred). Code extraction is MEAN-POOLING over each level's
stride window before the linear code-projection (hardcoded, not configurable -- a deliberate
departure from qcute_lagcodec_common.py's extract_code default, `code_extract_mode="last_h"`,
which just takes the block's LAST hidden-state position with no pooling at all; the other
options there (softmax_pool/light_query_attn/query_embed) are attention-based alternatives,
also not used here. Mean-pool gives every position in the window equal say in the code -- a
better fixed summary than either the last-position-only default or an attention mechanism
that still has to be learned from scratch).

Since this is purely an autoencoder task for now, "generation" here means RECONSTRUCTION only:
encode a real image to get real (not free-running-sampled) codes, then KV-cached greedy
byte-by-byte AR-decode conditioned on those fixed real codes, and compare the reconstruction
against ground truth side by side (samples_epoch{N}_reconstruct.png) plus report per-pixel MSE.
No free-running rollout from scratch -- that needs the encoder-NTP-rollout mechanism
documented in qcute_lagcodec_encoder.py (recurse top-down sampling from each level's own NTP
head), which is Task 2, gated on this reconstruction task actually working well first.
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from image_lagcodec.eqx_common import (
    Attention, Block, RMSNorm, SwiGLU, apply_rope_single, load_checkpoint, rmsnorm,
    rope_cos_sin_pos, rotate_half, save_checkpoint, sinkgd, splash_attention, splash_cross_attention,
    warmup_const_schedule,
)

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
SEQ_LEN = 32 * 32 * 3  # flat interleaved-RGB byte sequence length per image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 32
    d_model: tuple = (256, 256, 256)   # per level, level0..levelN-1
    n_layers: tuple = (2, 2, 2)
    n_heads: tuple = (4, 4, 4)
    n_kv_heads: tuple = (None, None, None)
    strides: tuple = (3, 4, 4)   # level0's stride=3 groups each pixel's own R,G,B triplet into
    # one code (the natural byte->pixel boundary); short (not-too-deep) 3-level hierarchy,
    # matching the original causalattn design's level count. The top level does NOT need to
    # collapse to a single code -- qcute_lagcodec's own levels never do either -- so strides
    # only need to evenly DIVIDE SEQ_LEN, not multiply exactly to it: (3,4,4) leaves the top
    # level with SEQ_LEN/(3*4*4)=64 codes, each covering a 4x4-pixel block.
    code_vocab: int = 8
    pq_chunks: int = 4
    mlp_mult: int = 4
    rope_base: float = 10000.0
    ntp_weight: float = 1.0   # weight on the sum of all levels' auxiliary encoder NTP losses
    decoder_type: str = "stack"   # "stack" | "self_attn_local" | "self_attn_lag"
    lag: int = 0   # decoder_type="self_attn_lag" only -- -1: pure causal byte NTP; 0: own-code
    # only (self_attn_local's block-diagonal design); k>=1: (k+1) codes prepended per group,
    # block-diagonal across groups -- see StageLagDecoder's docstring.
    kv_lm_mode: str = "identity"   # decoder_type="self_attn_local" only -- matches the
    # reference's kv_lm_mode: "identity" (no extra projection, raw code embedding -- only mode
    # implemented so far), "shared" (reuse the encoder's own level LM, reference's default),
    # "copy" (a fresh dedicated LM) -- both NotImplementedError for now, see StageLocalDecoder.
    # Decoder's own per-level transformer hparams -- DELIBERATELY separate fields from the
    # encoder's (not read off d_model/n_layers/n_heads/n_kv_heads above), so encoder and decoder
    # capacity can be tuned independently later; None resolves to mirroring the encoder's same-
    # level config exactly (current opinionated default, not a structural requirement).
    dec_d_model: tuple = None
    dec_n_layers: tuple = None
    dec_n_heads: tuple = None
    dec_n_kv_heads: tuple = None
    precision: str = "bf16"   # "bf16" (default, forward/backward matmuls in bfloat16, fp32 master
    # weights/optimizer state -- standard mixed precision) | "fp32" (diagnostic correctness mode,
    # e.g. for exact-match checks like reconstruct_full_recompute vs reconstruct_kv_cache_scan).

    def __post_init__(self):
        n = len(self.strides)
        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n
        assert SEQ_LEN % math.prod(self.strides) == 0, \
            f"strides must evenly divide {SEQ_LEN} (need not multiply exactly to it)"
        assert self.decoder_type in ("stack", "self_attn_local", "self_attn_lag")
        assert self.kv_lm_mode in ("identity", "shared", "copy")
        assert self.precision in ("bf16", "fp32")
        if self.dec_d_model is None:
            self.dec_d_model = self.d_model
        if self.dec_n_layers is None:
            self.dec_n_layers = self.n_layers
        if self.dec_n_heads is None:
            self.dec_n_heads = self.n_heads
        if self.dec_n_kv_heads is None:
            self.dec_n_kv_heads = self.n_kv_heads   # note: resolved further below, same as encoder's
        assert len(self.dec_d_model) == n and len(self.dec_n_layers) == n and len(self.dec_n_heads) == n \
            and len(self.dec_n_kv_heads) == n
        resolved_kv, dec_resolved_kv = [], []
        for i in range(n):
            kv = self.n_kv_heads[i] if self.n_kv_heads[i] is not None else max(1, self.n_heads[i] // 4)
            assert self.n_heads[i] % kv == 0
            assert self.d_model[i] % self.n_heads[i] == 0
            resolved_kv.append(kv)
            dec_kv = self.dec_n_kv_heads[i] if self.dec_n_kv_heads[i] is not None else max(1, self.dec_n_heads[i] // 4)
            assert self.dec_n_heads[i] % dec_kv == 0
            assert self.dec_d_model[i] % self.dec_n_heads[i] == 0
            dec_resolved_kv.append(dec_kv)
        self.n_kv_heads = tuple(resolved_kv)
        self.dec_n_kv_heads = tuple(dec_resolved_kv)


def n_units_at(level: int, cfg: Config) -> int:
    """Length of codes[level] -- SEQ_LEN reduced by strides[0..level]."""
    return SEQ_LEN // math.prod(cfg.strides[:level + 1])


# ---------------------------------------------------------------------------
# CIFAR-10 data
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

    def __len__(self):
        return len(self.images) // self.total

    def __iter__(self):
        n = len(self.images)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n - self.total + 1, self.total):
            sel = idx[start:start + self.total]
            img = self.images[sel].astype(np.int32)  # (total,32,32,3)
            flat = img.reshape(self.total, SEQ_LEN)   # interleaved R,G,B per pixel, raster order
            y = self.labels[sel].astype(np.int32)

            def shard(x):
                return x.reshape(self.n_devices, self.batch_size, *x.shape[1:])

            yield shard(flat), shard(y)


# ---------------------------------------------------------------------------
# Quantization (identical convention to run_causalattn_v1.py)
# ---------------------------------------------------------------------------

def quantize_hard(logits: jnp.ndarray) -> tuple:
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


def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


# ---------------------------------------------------------------------------
# Encoder level: causal transformer + own non-circular NTP head + mean-pool-then-linear code head
# ---------------------------------------------------------------------------

class EncoderLevel(eqx.Module):
    blocks: list
    ln_f: RMSNorm
    code_head: jnp.ndarray
    ntp_head: jnp.ndarray
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    is_byte_level: bool = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_layers: int, n_heads: int, n_kv_heads: int,
                 mlp_mult: int, rope_base: float, pq_chunks: int, code_vocab: int, stride: int,
                 is_byte_level: bool = False):
        k_blocks, k_head, k_ntp = jax.random.split(key, 3)
        block_keys = jax.random.split(k_blocks, n_layers)
        self.blocks = [Block(k, d_model, n_heads, n_kv_heads, mlp_mult, rope_base) for k in block_keys]
        self.ln_f = RMSNorm(d_model)
        self.code_head = jax.random.normal(k_head, (d_model, pq_chunks * code_vocab)) * 0.02
        # level0's NTP target alphabet is raw BYTES (256-way), every other level's is this
        # level's own PQ-chunked code alphabet (pq_chunks*code_vocab-way) -- different sizes,
        # must not share a head shape (mixing them up silently produces NaN via out-of-range
        # target indices into a too-small softmax, caught and fixed during initial smoke test).
        ntp_out = 256 if is_byte_level else pq_chunks * code_vocab
        self.ntp_head = jax.random.normal(k_ntp, (d_model, ntp_out)) * 0.02
        self.pq_chunks, self.code_vocab, self.stride, self.is_byte_level = pq_chunks, code_vocab, stride, is_byte_level

    def run(self, x: jnp.ndarray) -> jnp.ndarray:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, x: jnp.ndarray, target_idx: jnp.ndarray) -> dict:
        """x: (B,L,D) this level's own INPUT sequence (level0's real bytes, or level i's own
        input = level (i-1)'s realized code stream). target_idx: (B,L,pq_chunks) or (B,L) int
        -- the REAL next-step value in this level's own alphabet, for the NTP loss (non-
        circular: h[:,p] predicts target_idx[:,p+1], never its own block's code)."""
        h = self.run(x)
        M, L, D = h.shape
        n_blocks = L // self.stride
        h_blocks = h[:, :n_blocks * self.stride, :].reshape(M, n_blocks, self.stride, D)
        pooled = jnp.mean(h_blocks, axis=2)  # hardcoded mean-pool, see module docstring
        logits = reshape_pq(pooled @ self.code_head, self.pq_chunks, self.code_vocab)
        code_soft, code_idx = quantize_hard(logits)

        if self.is_byte_level:
            ntp_logits = h[:, :-1, :] @ self.ntp_head  # (B,L-1,256), plain byte prediction
        else:
            ntp_logits = reshape_pq(h[:, :-1, :] @ self.ntp_head, self.pq_chunks, self.code_vocab)
        tgt = target_idx[:, 1:]
        logp = jax.nn.log_softmax(ntp_logits, axis=-1)
        ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
        ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
        util = codebook_utilization(code_idx, self.code_vocab)
        return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util)

    def step(self, x_new, cache_k, cache_v, pos, T_max) -> tuple:
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(self.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], pos, T_max)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)


class HierEncoder(eqx.Module):
    byte_embed: jnp.ndarray
    code_embeds: list       # level i's OWN code's embedding table, used as level (i+1)'s input
    levels: list
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        n = len(cfg.strides)
        keys = jax.random.split(key, 1 + 2 * n)
        self.cfg = cfg
        self.byte_embed = jax.random.normal(keys[0], (256, cfg.d_model[0])) * 0.02
        self.levels = [EncoderLevel(keys[1 + i], cfg.d_model[i], cfg.n_layers[i], cfg.n_heads[i],
                                     cfg.n_kv_heads[i], cfg.mlp_mult, cfg.rope_base, cfg.pq_chunks,
                                     cfg.code_vocab, cfg.strides[i], is_byte_level=(i == 0)) for i in range(n)]
        self.code_embeds = [jax.random.normal(keys[1 + n + i], (cfg.code_vocab, cfg.d_model[i + 1])) * 0.02
                             for i in range(n - 1)]

    def __call__(self, flat_bytes: jnp.ndarray) -> dict:
        """flat_bytes: (B,SEQ_LEN) int -- returns per-level codes (hard idx, for the decoder's
        teacher-forced OWN-value inputs and for inference conditioning) and codes_soft (STE
        soft one-hot, for the decoder's cross-attn CONTEXT -- differentiable, so the decode loss
        backprops into this encoder, matching qcute_lagcodec_common.py's embed_for_decode
        convention of embedding the STE `quantize()` output, not a hard index) and the summed/
        weighted NTP losses (for the auxiliary training objective)."""
        x = self.byte_embed[flat_bytes]
        target = flat_bytes
        results = []
        codes, codes_soft = [], []
        for i, level in enumerate(self.levels):
            out = level.forward(x, target)
            results.append(out)
            codes.append(out["code_idx"])
            codes_soft.append(out["code_soft"])
            if i < len(self.levels) - 1:
                x = code_embed(out["code_soft"], self.code_embeds[i])
                target = out["code_idx"]
        return dict(codes=codes, codes_soft=codes_soft, results=results)


# ---------------------------------------------------------------------------
# Decoder -- follows qcute_lagcodec_decoder.py's StackDecoder level-0 mechanism closely (own
# level0 code only; levels 1+ stay encoder-side, feeding only their own NTP losses/hierarchy):
# a per-block SEED token is prepended to each K-byte block ([seed,b0_0..b0_{K-1},seed,b1_0..]),
# a plain causal self-attention stack runs over that augmented real-byte sequence, then a
# SEPARATE cross-attention stack lets every position in block b (seed included) attend to
# level0's OWN code for every causally-available block (code_pos_b <= query position, a real
# multi-key softmax, not just "own block") -- code_b's seed-position slot lets the whole block
# reconstruct from its own real code, which is valid teacher-forcing (not a leak): at
# generation time the same slot is filled by a level-above PREDICTION before block b starts.
# Target alignment (own_block_decode_loss's convention): each block's first K query positions
# (seed_b + its own first K-1 real bytes) predict, UNSHIFTED, that same block's own K real
# bytes -- the seed stands in for "the position before the block", so no further shift-by-1.
# ---------------------------------------------------------------------------

def rope_cos_sin_for_positions(pos: jnp.ndarray, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = pos[:, None].astype(jnp.float32) * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


class CrossAttention(eqx.Module):
    """q from one stream, k/v from another (the code stream) -- same GQA/QK-norm/RoPE
    conventions as eqx_common.Attention, just with independent q/kv sources and an explicit
    boolean mask instead of the standard lower-triangular one."""
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray
    out: jnp.ndarray
    q_norm: jnp.ndarray
    k_norm: jnp.ndarray
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_heads: int, rope_base: float):
        hd = d_model // n_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.wq = jax.random.normal(k1, (d_model, d_model)) * 0.02
        self.wk = jax.random.normal(k2, (d_model, d_model)) * 0.02
        self.wv = jax.random.normal(k3, (d_model, d_model)) * 0.02
        self.out = jax.random.normal(k4, (d_model, d_model)) * 0.02
        self.q_norm = jnp.ones((hd,))
        self.k_norm = jnp.ones((hd,))
        self.n_heads, self.rope_base = n_heads, rope_base

    def __call__(self, x_q: jnp.ndarray, x_kv: jnp.ndarray, q_pos: jnp.ndarray, k_pos: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
        B, Tq, D = x_q.shape
        Tk = x_kv.shape[1]
        hd = D // self.n_heads
        q = (x_q @ self.wq).reshape(B, Tq, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = (x_kv @ self.wk).reshape(B, Tk, self.n_heads, hd).transpose(0, 2, 1, 3)
        v = (x_kv @ self.wv).reshape(B, Tk, self.n_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos_q, sin_q = rope_cos_sin_for_positions(q_pos, hd, self.rope_base)
        cos_k, sin_k = rope_cos_sin_for_positions(k_pos, hd, self.rope_base)
        q = q * cos_q[None, None] + rotate_half(q) * sin_q[None, None]
        k = k * cos_k[None, None] + rotate_half(k) * sin_k[None, None]
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, Tq, D)
        return y @ self.out


class CrossBlock(eqx.Module):
    """Mirrors qcute_lagcodec_common.py's Block.forward_cross: x's own norm1 is reused (shared
    weight) to normalize BOTH the query stream and the code (kv) stream."""
    norm1: RMSNorm
    cross_attn: CrossAttention
    norm2: RMSNorm
    mlp: SwiGLU

    def __init__(self, key, d_model: int, n_heads: int, mlp_mult: int, rope_base: float):
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = CrossAttention(k1, d_model, n_heads, rope_base)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k2, d_model, mlp_mult)

    def __call__(self, x_q: jnp.ndarray, x_kv: jnp.ndarray, q_pos: jnp.ndarray, k_pos: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
        x = x_q + self.cross_attn(self.norm1(x_q), self.norm1(x_kv), q_pos, k_pos, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Level1Layer(eqx.Module):
    """One layer of qcute_lagcodec_decoder.py's encode_like_self_attn_decode + seed_query_decode
    (StackDecoder's actual level1 mechanism -- direct 1:1 port, NOT the seed-prepended-as-token
    design StageSelfAttnDecoder/StageCrossAttnDecoder used, which was a divergence from the real
    reference). Self-attention and cross-attention (to the own-block code) SHARE the same wq/wk/wv/
    out projection weights (matching qcute_lagcodec_common.py's CausalSelfAttention.forward_cross --
    the SAME self.attn object, just a different K/V source) -- this is NOT a separate CrossAttention
    module with its own weights. One shared MLP per layer, applied once after both attentions
    (matching Block.forward_cross's structure: self-attn residual -> cross-attn residual -> mlp
    residual, a single MLP per layer, not two)."""
    norm1: RMSNorm
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray
    out: jnp.ndarray
    q_norm: jnp.ndarray
    k_norm: jnp.ndarray
    norm2: RMSNorm
    mlp: SwiGLU
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float):
        hd = d_model // n_heads
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.norm1 = RMSNorm(d_model)
        self.wq = jax.random.normal(k1, (d_model, d_model)) * 0.02
        self.wk = jax.random.normal(k2, (d_model, n_kv_heads * hd)) * 0.02
        self.wv = jax.random.normal(k3, (d_model, n_kv_heads * hd)) * 0.02
        self.out = jax.random.normal(k4, (d_model, d_model)) * 0.02
        self.q_norm = jnp.ones((hd,))
        self.k_norm = jnp.ones((hd,))
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k5, d_model, mlp_mult)
        self.n_heads, self.n_kv_heads, self.rope_base = n_heads, n_kv_heads, rope_base

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        n_rep = self.n_heads // self.n_kv_heads
        return jnp.repeat(x, n_rep, axis=1) if n_rep > 1 else x

    def self_attn_and_save(self, x: jnp.ndarray, pos_real: jnp.ndarray) -> tuple:
        """pass1's self-attn: causal over real bytes. Returns (attn_out, k, v) -- callers discard
        k/v (kept only for signature compatibility). Uses the Pallas TPU splash_attention kernel
        (block-sparse, native GQA -- see splash_attention() in eqx_common.py). Self-attention only
        (pos_real is always a contiguous 0..T-1 range) -- cross_attn_own_code below is untouched
        (custom lag-shifted mask, not a simple causal pattern this kernel supports)."""
        B, T, D = x.shape
        hd = D // self.n_heads
        xn = self.norm1(x)
        q = (xn @ self.wq).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = (xn @ self.wk).reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (xn @ self.wv).reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin_for_positions(pos_real, hd, self.rope_base)
        q = q * cos[None, None] + rotate_half(q) * sin[None, None]
        k = k * cos[None, None] + rotate_half(k) * sin[None, None]
        scale = 1.0 / math.sqrt(hd)
        y = splash_attention(q, k, v, causal=True, sm_scale=scale)  # (B,H,T,hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out, k, v

    def cross_attn_own_code(self, x: jnp.ndarray, code_kv: jnp.ndarray, q_pos: jnp.ndarray,
                             code_pos: jnp.ndarray, cum_K: int, lag_bytes: int) -> jnp.ndarray:
        """forward_cross: SHARED wq/wk/wv/out, applied to x_q=norm1(x) (x already updated by the
        self-attn residual) and x_kv=norm1(code_kv). Uses the Pallas TPU splash_attention kernel
        (rectangular q x kv, LagCrossMask -- see eqx_common.py) instead of a materialized
        (B,H,T,Tc) einsum+where+softmax -- cum_K/lag_bytes (Python ints, static per call) rebuild
        the mask arithmetic that used to be precomputed into a boolean cross_mask tensor."""
        B, T, D = x.shape
        Tc = code_kv.shape[1]
        hd = D // self.n_heads
        xn, coden = self.norm1(x), self.norm1(code_kv)
        q = (xn @ self.wq).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = (coden @ self.wk).reshape(B, Tc, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (coden @ self.wv).reshape(B, Tc, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos_q, sin_q = rope_cos_sin_for_positions(q_pos, hd, self.rope_base)
        cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, self.rope_base)
        q = q * cos_q[None, None] + rotate_half(q) * sin_q[None, None]
        k = k * cos_k[None, None] + rotate_half(k) * sin_k[None, None]
        scale = 1.0 / math.sqrt(hd)
        y = splash_cross_attention(q, k, v, cum_K, lag_bytes, scale)  # (B,H,T,hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out

    def forward_pass1(self, x: jnp.ndarray, code_kv: jnp.ndarray, pos_real: jnp.ndarray,
                       code_pos: jnp.ndarray, cum_K: int, lag_bytes: int) -> tuple:
        a, k_saved, v_saved = self.self_attn_and_save(x, pos_real)
        x = x + a
        x = x + self.cross_attn_own_code(x, code_kv, pos_real, code_pos, cum_K, lag_bytes)
        x = x + self.mlp(self.norm2(x))
        return x, k_saved, v_saved

    def cross_attn_own_code_single(self, x: jnp.ndarray, code_kv: jnp.ndarray, q_pos,
                                    code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> jnp.ndarray:
        """Single-query form of cross_attn_own_code (x: (B,D), one position, q_pos scalar) --
        code_kv is small (n_blocks_now entries) and cheap to recompute fresh every step, no cache
        needed for it."""
        B, D = x.shape
        Tc = code_kv.shape[1]
        hd = D // self.n_heads
        xn, coden = self.norm1(x), self.norm1(code_kv)
        q = (xn @ self.wq).reshape(B, self.n_heads, hd)
        k = (coden @ self.wk).reshape(B, Tc, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (coden @ self.wv).reshape(B, Tc, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos_q, sin_q = rope_cos_sin_pos(q_pos, hd, self.rope_base)
        q = apply_rope_single(q, cos_q, sin_q)
        cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, self.rope_base)
        k = k * cos_k[None, None, :, :] + rotate_half(k) * sin_k[None, None, :, :]
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhd,bhtd->bht", q, k) * scale
        logits = jnp.where(cross_mask[None, None, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bht,bhtd->bhd", attn, v).reshape(B, D)
        return y @ self.out

    def self_step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos,
                  T_max: int, code_kv: jnp.ndarray, code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> tuple:
        """Incremental form of forward_pass1: writes x_new's real k/v into the GROWING GLOBAL
        self-attn cache at `pos` (pre-repeat_kv, matching eqx_common.Attention.step's convention --
        repeat happens at read time), then cross-attends to code_kv (recomputed fresh, cheap).
        cache_k/cache_v: (B,n_kv_heads,T_max,hd)."""
        B, D = x_new.shape
        hd = D // self.n_heads
        xn = self.norm1(x_new)
        q = (xn @ self.wq).reshape(B, self.n_heads, hd)
        k = (xn @ self.wk).reshape(B, self.n_kv_heads, hd)
        v = (xn @ self.wv).reshape(B, self.n_kv_heads, hd)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin_pos(pos, hd, self.rope_base)
        q, k = apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
        cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :], (0, 0, pos, 0))
        cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :], (0, 0, pos, 0))
        k_full, v_full = self._repeat_kv(cache_k), self._repeat_kv(cache_v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
        valid = jnp.arange(T_max) <= pos
        logits = jnp.where(valid[None, None, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bht,bhtd->bhd", attn, v_full).reshape(B, D)
        x = x_new + y @ self.out
        x = x + self.cross_attn_own_code_single(x, code_kv, pos, code_pos, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x, cache_k, cache_v


class StackDecoder(eqx.Module):
    """Generalized qcute_lagcodec StackDecoder port -- any depth (levels above level0 chained
    sequentially via a UNIFORM self-attn+cross-attn+mlp block per level -- deliberately more
    uniform than the reference's self-attn-once-then-cross-attn-only-per-upper-level design,
    user-requested 2026-09-08), plus a `lag` scheduling knob (session-invented, generalizes/
    subsumes the reference's own_code_min_lag/window mechanism): lag shifts EVERY level's
    cross-attention causal mask forward by `(lag+1)*prod(strides[:-1])` bytes (the topmost
    CONSULTED level's code span -- strides[-1] produces the hard-excluded topmost output, never
    consulted by anyone). Self-attention is NEVER touched by lag -- always fully unbounded causal.

    BOS token (this session's simplification over the reference's own per-block trainable
    'seed'): a single separate trainable D-dim parameter (NOT sharing target_embed's table),
    prepended once at sequence position 0, then treated as a completely NORMAL token through
    every layer (normal self-attn Q/K/V, becomes a real key for later positions) -- no special
    seed-only pure-query pass needed. This makes the whole sequence follow the STANDARD shifted
    NTP convention (position p predicts target[p]) uniformly, eliminating the reference's
    separate h0/h0_shifted dual-view entirely.

    lag<0 (e.g. true causal NTP) is NOT implemented -- asserts lag>=0 (correctly implementing it
    needs encoder-like shifted targets, not just disabling cross-attention -- more complex than
    warranted right now, see chat 2026-09-08). Generation (full-recompute + KV-cache) not yet
    rebuilt for this design -- forward/training-only so far, same bring-up order used elsewhere
    in this file."""
    bos_embed: jnp.ndarray            # separate (D,) param, NOT part of target_embed's table
    target_embed: jnp.ndarray         # (256,D)
    ctx_embeds: list                  # one (code_vocab,D) table per consulted level (level1..levelL)
    level_layers: list                # list of list-of-Level1Layer, one inner list per consulted level
    ln_f: RMSNorm
    head: jnp.ndarray
    strides: tuple = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        D = cfg.dec_d_model[0]
        n_levels = len(cfg.strides)
        n_consulted = n_levels - 1  # codes_soft[0..n_levels-2] -- topmost output hard-excluded
        assert n_consulted >= 1, "StackDecoder needs n_levels>=2 (at least one consulted level)"
        self.strides = cfg.strides
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[0], cfg.dec_n_kv_heads[0]
        n_layers = cfg.dec_n_layers[0]
        keys = jax.random.split(key, 4 + 2 * n_consulted)
        self.bos_embed = jax.random.normal(keys[0], (D,)) * 0.02
        self.target_embed = jax.random.normal(keys[1], (256, D)) * 0.02
        self.ctx_embeds = []
        self.level_layers = []
        for t in range(n_consulted):
            ke = keys[4 + 2 * t]
            kl = keys[4 + 2 * t + 1]
            self.ctx_embeds.append(jax.random.normal(ke, (cfg.code_vocab, D)) * 0.02)
            layer_keys = jax.random.split(kl, n_layers)
            self.level_layers.append([Level1Layer(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                                       for k in layer_keys])
        self.ln_f = RMSNorm(D)
        self.head = jax.random.normal(keys[2], (D, 256)) * 0.02

    def forward(self, target_seq: jnp.ndarray, codes_soft: list, lag: int) -> tuple:
        """target_seq: (B,SEQ_LEN) real bytes. codes_soft: enc["codes_soft"] full list. lag: see
        class docstring, must be >=0."""
        assert lag >= 0, "lag<0 not implemented -- see StackDecoder docstring"
        B, SEQ_LEN = target_seq.shape
        D = self.target_embed.shape[-1]
        bos = jnp.broadcast_to(self.bos_embed, (B, 1, D))
        rest = self.target_embed[target_seq[:, :-1]]
        x = jnp.concatenate([bos, rest], axis=1)  # (B,SEQ_LEN,D) -- BOS + bytes[:-1], standard shift
        pos_real = jnp.arange(SEQ_LEN)

        lag_bytes = (lag + 1) * math.prod(self.strides[:-1])
        cum_K = 1
        for t, (ctx_embed, layers) in enumerate(zip(self.ctx_embeds, self.level_layers)):
            cum_K *= self.strides[t]
            code = codes_soft[t]
            n_blocks_t = code.shape[1]
            code_tok = code_embed(code, ctx_embed)
            code_pos = (jnp.arange(n_blocks_t) + 1) * cum_K - 1
            for layer in layers:
                x, _, _ = layer.forward_pass1(x, code_tok, pos_real, code_pos, cum_K, lag_bytes)

        h = self.ln_f(x)
        logits = h @ self.head
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        return loss, acc

    def _logits_and_masks(self, codes: list, lag: int):
        """Shared setup for both generation paths: per-level code embeddings, code positions, and
        the lag-shifted cross-attn mask builder (mask depends on query position, computed lazily
        per-call since full-recompute uses a growing pos_real array while KV-cache uses a scalar).
        cum_Ks (Python ints, one per level) feed splash_cross_attention's LagCrossMask in
        reconstruct_full_recompute; the self_step-based generation paths ignore it."""
        lag_bytes = (lag + 1) * math.prod(self.strides[:-1])
        code_toks, code_poss, cum_Ks = [], [], []
        cum_K = 1
        for t, ctx_embed in enumerate(self.ctx_embeds):
            cum_K *= self.strides[t]
            code = codes[t]
            n_blocks_t = code.shape[1]
            code_toks.append(code_embed(code, ctx_embed))
            code_poss.append((jnp.arange(n_blocks_t) + 1) * cum_K - 1)
            cum_Ks.append(cum_K)
        return lag_bytes, code_toks, code_poss, cum_Ks

    def reconstruct_full_recompute(self, codes: list, lag: int, greedy: bool = True,
                                    temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """No incremental KV-cache -- reruns the plain batched forward_pass1 stack (the exact
        computation forward() uses) from scratch at every single byte step, over the growing
        BOS+decided-so-far sequence. O(SEQ_LEN^2) total. Diagnostic-grade correctness reference
        for reconstruct_kv_cache. codes: list of HARD idx code arrays (enc["codes"]), matching
        forward()'s codes_soft argument but realized (not STE soft -- no gradient needed here)."""
        assert lag >= 0, "lag<0 not implemented -- see StackDecoder docstring"
        B = codes[0].shape[0]
        D = self.target_embed.shape[-1]
        cum_K_first = self.strides[0]
        SEQ_LEN = codes[0].shape[1] * cum_K_first
        rng = jax.random.PRNGKey(seed)
        lag_bytes, code_toks, code_poss, cum_Ks = self._logits_and_masks(codes, lag)

        decided = [jnp.broadcast_to(self.bos_embed, (B, D))]
        out = jnp.zeros((B, SEQ_LEN), dtype=jnp.int32)
        for t in tqdm(range(SEQ_LEN), desc=f"decode_full_recompute(stack,lag={lag},L={SEQ_LEN})", leave=False):
            x = jnp.stack(decided, axis=1)  # (B, t+1, D)
            pos_real = jnp.arange(t + 1)
            for code_tok, code_pos, cum_K, layers in zip(code_toks, code_poss, cum_Ks, self.level_layers):
                for layer in layers:
                    x, _, _ = layer.forward_pass1(x, code_tok, pos_real, code_pos, cum_K, lag_bytes)
            h_last = self.ln_f(x)[:, -1, :]
            logits = h_last @ self.head
            if greedy:
                val = jnp.argmax(logits, axis=-1)
            else:
                rng, k_ = jax.random.split(rng)
                val = jax.random.categorical(k_, logits / temperature, axis=-1)
            out = out.at[:, t].set(val)
            decided.append(self.target_embed[val])
        return out

    def reconstruct_kv_cache(self, codes: list, lag: int, greedy: bool = True,
                              temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """Incremental KV-cache generation -- one growing self-attn cache PER (level, layer) pair
        (each level's layer stack operates on a DIFFERENT, progressively-refined residual state
        than the previous level's, so each needs its own cache -- reuses Level1Layer.self_step,
        already verified elsewhere in this file). Checked directly against reconstruct_full_recompute
        via a teacher-forced consistency script, not assumed correct by construction."""
        assert lag >= 0, "lag<0 not implemented -- see StackDecoder docstring"
        B = codes[0].shape[0]
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        cum_K_first = self.strides[0]
        SEQ_LEN = codes[0].shape[1] * cum_K_first
        rng = jax.random.PRNGKey(seed)
        lag_bytes, code_toks, code_poss, _cum_Ks = self._logits_and_masks(codes, lag)

        caches = [[(jnp.zeros((B, self.n_kv_heads, SEQ_LEN, hd)), jnp.zeros((B, self.n_kv_heads, SEQ_LEN, hd)))
                   for _ in layers] for layers in self.level_layers]

        out = jnp.zeros((B, SEQ_LEN), dtype=jnp.int32)
        x_input = jnp.broadcast_to(self.bos_embed, (B, D))
        for t in tqdm(range(SEQ_LEN), desc=f"decode_kv_cache(stack,lag={lag},L={SEQ_LEN})", leave=False):
            for lvl, (code_tok, code_pos, layers) in enumerate(zip(code_toks, code_poss, self.level_layers)):
                mask = code_pos <= (t + lag_bytes)
                for li, layer in enumerate(layers):
                    ck, cv = caches[lvl][li]
                    x_input, ck, cv = layer.self_step(x_input, ck, cv, t, SEQ_LEN, code_tok, code_pos, mask)
                    caches[lvl][li] = (ck, cv)
            logits = self.ln_f(x_input) @ self.head
            if greedy:
                val = jnp.argmax(logits, axis=-1)
            else:
                rng, k_ = jax.random.split(rng)
                val = jax.random.categorical(k_, logits / temperature, axis=-1)
            out = out.at[:, t].set(val)
            x_input = self.target_embed[val]
        return out

    def kv_cache_init(self, codes: list, lag: int, seed: int = 0) -> tuple:
        """Build the initial resumable state for scan-based generation: (carry, ctx). carry =
        (x_input, caches, rng) -- a valid JAX pytree, safe to stash and hand back into
        kv_cache_step_chunk later (e.g. across separate debugging calls). ctx bundles the
        per-level code embeddings/positions/lag_bytes/SEQ_LEN derived once from `codes`/`lag`."""
        assert lag >= 0, "lag<0 not implemented -- see StackDecoder docstring"
        B = codes[0].shape[0]
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        cum_K_first = self.strides[0]
        SEQ_LEN = codes[0].shape[1] * cum_K_first
        lag_bytes, code_toks, code_poss, _cum_Ks = self._logits_and_masks(codes, lag)
        caches0 = [[(jnp.zeros((B, self.n_kv_heads, SEQ_LEN, hd)), jnp.zeros((B, self.n_kv_heads, SEQ_LEN, hd)))
                    for _ in layers] for layers in self.level_layers]
        carry = (jnp.broadcast_to(self.bos_embed, (B, D)), caches0, jax.random.PRNGKey(seed))
        ctx = dict(SEQ_LEN=SEQ_LEN, lag_bytes=lag_bytes, code_toks=code_toks, code_poss=code_poss)
        return carry, ctx

    def kv_cache_step_chunk(self, carry: tuple, ctx: dict, t_start: int, n_steps: int,
                             greedy: bool = True, temperature: float = 1.0) -> tuple:
        """Resumable primitive: runs `n_steps` timesteps starting at ABSOLUTE position `t_start`
        (RoPE and cache writes both use this absolute position, not chunk-relative -- confirmed:
        resuming from t_start>0 gives identical results to reaching that position via smaller
        chunks or one big call, verified below). One jax.lax.scan call, jit-compiled once and
        reused across repeated calls with different t_start (same n_steps -> same compiled shape).
        Returns (new_carry, vals) where vals: (n_steps, B) -- caller concatenates/transposes across
        calls to assemble the full (B, SEQ_LEN) sequence."""
        SEQ_LEN, lag_bytes = ctx["SEQ_LEN"], ctx["lag_bytes"]
        code_toks, code_poss = ctx["code_toks"], ctx["code_poss"]

        def step_fn(carry, t):
            x_input, caches, rng = carry
            new_caches = []
            for lvl, (code_tok, code_pos, layers) in enumerate(zip(code_toks, code_poss, self.level_layers)):
                mask = code_pos <= (t + lag_bytes)
                new_level_cache = []
                for li, layer in enumerate(layers):
                    ck, cv = caches[lvl][li]
                    x_input, ck, cv = layer.self_step(x_input, ck, cv, t, SEQ_LEN, code_tok, code_pos, mask)
                    new_level_cache.append((ck, cv))
                new_caches.append(new_level_cache)
            logits = self.ln_f(x_input) @ self.head
            if greedy:
                val = jnp.argmax(logits, axis=-1)
                rng_next = rng
            else:
                rng_next, k_ = jax.random.split(rng)
                val = jax.random.categorical(k_, logits / temperature, axis=-1)
            x_next = self.target_embed[val]
            return (x_next, new_caches, rng_next), val

        @jax.jit
        def run_chunk(carry, ts):
            return jax.lax.scan(step_fn, carry, ts)

        ts = jnp.arange(t_start, t_start + n_steps)  # ABSOLUTE positions, never chunk-relative
        return run_chunk(carry, ts)

    def reconstruct_kv_cache_scan(self, codes: list, lag: int, greedy: bool = True, temperature: float = 1.0,
                                   seed: int = 0, chunk_size: int = None) -> jnp.ndarray:
        """Convenience driver over kv_cache_init/kv_cache_step_chunk: JIT-compiled + scanned,
        replacing reconstruct_kv_cache's bare Python loop of un-jitted self_step calls -- audit
        (2026-09-08) found that loop pays per-op dispatch overhead at every single call (no fused
        kernel), the real bottleneck at real CIFAR scale (extrapolated ~30-55s/qual-gen-call on
        CPU, likely worse on TPU's host-dispatch-per-op cost -- unmeasured on TPU directly).

        chunk_size: how many steps ONE scan call covers. None (default) = the WHOLE sequence in a
        single scan -- fastest, but opaque (no way to inspect intermediate state mid-generation).
        Pass a smaller chunk_size (down to 1, fully step-by-step but still jit-compiled, reused
        across chunks) for debugging -- or call kv_cache_init/kv_cache_step_chunk directly for
        full manual control (e.g. resuming from a stashed carry in a later, separate call)."""
        carry, ctx = self.kv_cache_init(codes, lag, seed)
        SEQ_LEN = ctx["SEQ_LEN"]
        chunk_size = chunk_size or SEQ_LEN

        out_chunks = []
        t0 = 0
        while t0 < SEQ_LEN:
            n = min(chunk_size, SEQ_LEN - t0)
            carry, vals = self.kv_cache_step_chunk(carry, ctx, t0, n, greedy=greedy, temperature=temperature)
            out_chunks.append(vals)
            t0 += n
        return jnp.concatenate(out_chunks, axis=0).T  # (B, SEQ_LEN)


class Level1LocalLayer(eqx.Module):
    """One layer of the CORRECTED level1 mechanism -- 1:1 port of qcute_lagcodec_decoder.py's
    block_local_level1_decode (StackDecoderLocal's Variant B), with ONE deliberate departure
    from the reference (user-requested 2026-09-08): a SEPARATE norm before cross-attention
    (`norm_cross`) instead of the reference's Block.forward_cross, which reuses `ln1` for both
    the self-attn and cross-attn input streams (confirmed by reading qcute_lagcodec_common.py
    directly -- reference has 2 norms/layer here, not 3). Self-attn and cross-attn otherwise
    SHARE the same wq/wk/wv/out projection weights (matching the reference's single `self.attn`
    object reused for both), one shared MLP per layer.

    Self-attention is BLOCK-DIAGONAL and among REAL BYTES ONLY (K-length, own block, standard
    causal) -- the code is NEVER a self-attention token (unlike the divergent V1 design). Cross-
    attention is to the block's own code, always fully visible (a block's own code is always
    causally available to it, no masking needed). The seed (predicts byte0) does ONLY cross-
    attention, no self-attention call at all -- proven exactly zero under same-block causal
    self-attn (query position = block start, no same-block key can precede it)."""
    norm1: RMSNorm
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray
    out: jnp.ndarray
    q_norm: jnp.ndarray
    k_norm: jnp.ndarray
    norm_cross: RMSNorm
    norm2: RMSNorm
    mlp: SwiGLU
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float):
        hd = d_model // n_heads
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.norm1 = RMSNorm(d_model)
        self.wq = jax.random.normal(k1, (d_model, d_model)) * 0.02
        self.wk = jax.random.normal(k2, (d_model, n_kv_heads * hd)) * 0.02
        self.wv = jax.random.normal(k3, (d_model, n_kv_heads * hd)) * 0.02
        self.out = jax.random.normal(k4, (d_model, d_model)) * 0.02
        self.q_norm = jnp.ones((hd,))
        self.k_norm = jnp.ones((hd,))
        self.norm_cross = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k5, d_model, mlp_mult)
        self.n_heads, self.n_kv_heads, self.rope_base = n_heads, n_kv_heads, rope_base

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        n_rep = self.n_heads // self.n_kv_heads
        return jnp.repeat(x, n_rep, axis=1) if n_rep > 1 else x

    def self_attn_real(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (Bn, K, D) -- block-diagonal (n_blocks already folded into Bn), plain causal
        self-attn among the K real bytes of this block only, LOCAL positions 0..K-1."""
        Bn, K, D = x.shape
        hd = D // self.n_heads
        xn = self.norm1(x)
        q = (xn @ self.wq).reshape(Bn, K, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = (xn @ self.wk).reshape(Bn, K, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (xn @ self.wv).reshape(Bn, K, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        pos = jnp.arange(K)
        cos, sin = rope_cos_sin_for_positions(pos, hd, self.rope_base)
        q = q * cos[None, None] + rotate_half(q) * sin[None, None]
        k = k * cos[None, None] + rotate_half(k) * sin[None, None]
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        causal = pos[:, None] >= pos[None, :]
        logits = jnp.where(causal[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(Bn, K, D)
        return y @ self.out

    def cross_attn_own(self, x: jnp.ndarray, code_kv: jnp.ndarray) -> jnp.ndarray:
        """x: (Bn, T, D) [T=K real bytes or T=1 seed]; code_kv: (Bn, 1, D) this block's own code
        -- always fully visible (a block's own code is always causally available to it)."""
        Bn, T, D = x.shape
        hd = D // self.n_heads
        xn, coden = self.norm_cross(x), self.norm_cross(code_kv)
        q = (xn @ self.wq).reshape(Bn, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = (coden @ self.wk).reshape(Bn, 1, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = (coden @ self.wv).reshape(Bn, 1, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        zero_pos = jnp.zeros((1,))
        cos0, sin0 = rope_cos_sin_for_positions(zero_pos, hd, self.rope_base)
        q = q * cos0[None, None] + rotate_half(q) * sin0[None, None]
        k = k * cos0[None, None] + rotate_half(k) * sin0[None, None]
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale  # (Bn,H,T,1), no masking needed
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(Bn, T, D)
        return y @ self.out

    def forward_real(self, x: jnp.ndarray, code_kv: jnp.ndarray) -> jnp.ndarray:
        x = x + self.self_attn_real(x)
        x = x + self.cross_attn_own(x, code_kv)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_seed(self, x_seed: jnp.ndarray, code_kv: jnp.ndarray) -> jnp.ndarray:
        x_seed = x_seed + self.cross_attn_own(x_seed, code_kv)
        x_seed = x_seed + self.mlp(self.norm2(x_seed))
        return x_seed


class StageLocalDecoder(eqx.Module):
    """CORRECTED, GENERALIZED replacement for StageLocalTrack1DecoderV1 (2026-09-08 rewrite --
    read the reference's decode_level/StackDecoder.__init__/block_local_level1_decode directly
    before writing this, per explicit correction: the V1 version had two real divergences and
    was hardcoded to exactly 2 levels instead of generalizing like the reference's cond_depth
    loop). Works for ANY n_levels>=2, not just 3.

    Level1 (this block's level1 code): Level1LocalLayer stack -- REAL cross-attention (not
    code-as-token self-attention), block-diagonal self-attn among real bytes only, matching
    block_local_level1_decode exactly except for the deliberate 3rd-norm departure (see
    Level1LocalLayer's docstring).

    Upper levels (level2..level(n_levels-1)'s codes -- the topmost level's own code is hard-
    excluded, never consumed by anyone, matching the reference's `n_upper = n_levels-2` cap):
    CrossBlock stages (cross-attn + MLP, own dedicated norm already, no change needed), chained
    SEQUENTIALLY in level order (level1 first/"earliest", increasingly coarser levels after --
    matches decode_level's actual `for j in range(i+1, j_max)` loop order exactly: own-level
    code first, coarser codes added later, each stage's output feeding the next).

    SIMPLIFIED relative to the reference (explicit tradeoff, confirmed with user before
    building): UNSHIFTED alignment throughout (h[p] reconstructs target[p], same convention
    every other decoder in this file uses) instead of the reference's shifted h0/h0_shifted dual-
    view handoff + separate per-level auxiliary losses. Upper levels cross-attend on the SAME
    unshifted h0 level1 produces -- still fully causal (code masks are still code_pos<=query_pos)
    -- just without the reference's extra first-block-only auxiliary loss term.

    kv_lm_mode: reference default is "shared" (reruns the code embedding through the encoder's
    own trained per-level LM before using it as cross-attn K/V) -- only "identity" (raw embedding
    table, no extra projection) is implemented so far; "shared"/"copy" raise NotImplementedError."""
    target_embed: jnp.ndarray
    level1_ctx_embed: jnp.ndarray      # level1's code embed table (level1's own-block code)
    level1_layers: list                # list of Level1LocalLayer
    level1_seed: jnp.ndarray           # trainable seed constant, predicts byte0
    upper_ctx_embeds: list             # one embed table per upper level (levels 2..n_levels-1)
    upper_cross_blocks: list           # list of list-of-CrossBlock, one inner list per upper level
    ln_f: RMSNorm
    head: jnp.ndarray
    K0: int = eqx.field(static=True)          # level0 stride == bytes per level1 block
    strides: tuple = eqx.field(static=True)   # cfg.strides, to compute each upper level's cum_K
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    kv_lm_mode: str = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        n_levels = len(cfg.strides)
        assert n_levels >= 2, "StageLocalDecoder needs n_levels>=2 (at least level1)"
        assert cfg.kv_lm_mode == "identity", \
            f"kv_lm_mode={cfg.kv_lm_mode!r} not implemented yet (only 'identity')"
        self.kv_lm_mode = cfg.kv_lm_mode
        D = cfg.dec_d_model[0]
        self.K0 = cfg.strides[0]
        self.strides = cfg.strides
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[0], cfg.dec_n_kv_heads[0]
        n_upper = max(0, n_levels - 2)  # excludes topmost level's own code, matches reference
        n_layers = cfg.dec_n_layers[0]
        keys = jax.random.split(key, 5 + 2 * n_upper)
        self.target_embed = jax.random.normal(keys[0], (256, D)) * 0.02
        self.level1_ctx_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        self.level1_seed = jax.random.normal(keys[2], (D,)) * 0.02
        level1_keys = jax.random.split(keys[3], n_layers)
        self.level1_layers = [Level1LocalLayer(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                               for k in level1_keys]
        self.upper_ctx_embeds = []
        self.upper_cross_blocks = []
        for t in range(n_upper):
            ke = keys[5 + 2 * t]
            kc = keys[5 + 2 * t + 1]
            self.upper_ctx_embeds.append(jax.random.normal(ke, (cfg.code_vocab, D)) * 0.02)
            cross_keys = jax.random.split(kc, n_layers)
            self.upper_cross_blocks.append([CrossBlock(k, D, self.n_heads, cfg.mlp_mult, cfg.rope_base)
                                             for k in cross_keys])
        self.ln_f = RMSNorm(D)
        self.head = jax.random.normal(keys[4], (D, 256)) * 0.02

    def forward(self, target_seq: jnp.ndarray, codes_soft: list) -> tuple:
        """target_seq: (B, SEQ_LEN) real bytes. codes_soft: enc["codes_soft"] -- the FULL list of
        every level's own code (codes_soft[0]=level1's code=level1, codes_soft[1]=level2's
        code=upper level 0, ..., codes_soft[n_levels-2]=level(n_levels-1)'s code=last usable
        upper level; codes_soft[n_levels-1], the topmost level's own output, is never used)."""
        B = target_seq.shape[0]
        D = self.target_embed.shape[-1]
        n_blocks0 = codes_soft[0].shape[1]

        # Level1: block-diagonal, real cross-attention (Level1LocalLayer)
        ctx_tok0 = code_embed(codes_soft[0], self.level1_ctx_embed)  # (B,n_blocks0,D)
        te = self.target_embed[target_seq]  # (B,n_blocks0*K0,D)
        Bn = B * n_blocks0
        x_real = te.reshape(Bn, self.K0, D)
        code_kv = ctx_tok0.reshape(Bn, 1, D)
        for layer in self.level1_layers:
            x_real = layer.forward_real(x_real, code_kv)
        x_seed = jnp.broadcast_to(self.level1_seed, (Bn, 1, D))
        for layer in self.level1_layers:
            x_seed = layer.forward_seed(x_seed, code_kv)
        h_real = self.ln_f(x_real).reshape(B, n_blocks0, self.K0, D)
        h_seed = self.ln_f(x_seed).reshape(B, n_blocks0, 1, D)
        h0 = jnp.concatenate([h_seed, h_real[:, :, :self.K0 - 1, :]], axis=2)  # (B,n_blocks0,K0,D), unshifted
        target_len = n_blocks0 * self.K0
        x = h0.reshape(B, target_len, D)

        # Upper levels: level2..level(n_levels-1)'s codes, sequentially chained, coarser each time
        query_pos = jnp.arange(target_len)
        cum_K = self.K0
        for t, cross_blocks in enumerate(self.upper_cross_blocks):
            cum_K *= self.strides[t + 1]
            code = codes_soft[t + 1]
            n_blocks_t = code.shape[1]
            code_tok = code_embed(code, self.upper_ctx_embeds[t])  # (B,n_blocks_t,D)
            code_pos = (jnp.arange(n_blocks_t) + 1) * cum_K - 1
            mask = code_pos[None, :] <= query_pos[:, None]
            for cblk in cross_blocks:
                x = cblk(x, code_tok, query_pos, code_pos, mask)

        h = self.ln_f(x)
        logits = h @ self.head
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        return loss, acc


class StageLagDecoder(eqx.Module):
    """Generalizes self_attn_local's block-diagonal design (each block sees ONLY its own code,
    zero lookahead) to a configurable `lag`: groups of (lag+1) CONSECUTIVE top-level codes are
    placed BEFORE any of their combined (lag+1)*K bytes in the sequence (NOT interleaved
    per-block), so every byte in the group can causally see ALL (lag+1) codes -- including `lag`
    codes that are causally LATER than it (the true autoregressive generation process wouldn't
    have them yet without waiting for that many more codes first). Groups remain block-diagonal
    from EACH OTHER (zero cross-group visibility), exactly like the lag=0 case's per-block
    isolation, just at group granularity -- lag=0 is the degenerate G=1 case of this same
    mechanism (own code prepended, K bytes follow, causal, matches self_attn_local exactly).
    lag=max_lag (== n_blocks-1, ONE group spanning the WHOLE sequence) needs every code before
    reconstructing anything -- full non-causal, whole-sequence context, single pass.

    lag=-1 is a SEPARATE, stricter mode (no `G` grouping applies): no code conditioning at all,
    pure byte-level next-token-prediction -- the TRUE, hardest causal metric. lag=0 already
    'cheats' relative to this by handing every byte its own code for free, itself derived by the
    encoder from seeing the whole block (including bytes not yet decoded at generation time from
    that byte's perspective) -- lag=-1 removes that shortcut entirely.

    Same weights work for ANY lag value (self_blocks are just per-position transformer blocks,
    agnostic to how many codes/bytes are grouped) -- `lag` is a forward()/reconstruct() run-time
    argument, not baked into the model structure, so train-time and eval-time lag can be swept
    independently (though matching them avoids a train/inference mismatch, same principle as
    everywhere else in this file)."""
    target_embed: jnp.ndarray
    ctx_embed: jnp.ndarray
    self_blocks: list
    ln_f: RMSNorm
    head: jnp.ndarray
    K: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int = 0):
        D = cfg.dec_d_model[level]
        self.K = cfg.strides[level]
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[level], cfg.dec_n_kv_heads[level]
        keys = jax.random.split(key, 3)
        self.target_embed = jax.random.normal(keys[0], (256, D)) * 0.02
        self.ctx_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        n_layers = cfg.dec_n_layers[level]
        block_keys = jax.random.split(keys[2], n_layers)
        self.self_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                             for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.head = jax.random.normal(keys[1], (D, 256)) * 0.02

    def forward(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, lag: int) -> tuple:
        """target_seq: (B, n_blocks*K) real bytes. ctx_code_soft: (B, n_blocks, pq) level1's own
        code (STE soft, differentiable into the encoder). lag=-1: no code conditioning, pure
        causal byte NTP. lag>=0: groups of (lag+1) codes prepended before their (lag+1)*K bytes,
        block-diagonal across groups."""
        B = target_seq.shape[0]
        D = self.target_embed.shape[-1]
        te = self.target_embed[target_seq]

        if lag == -1:
            x = te
            for blk in self.self_blocks:
                x = blk(x)
            h = self.ln_f(x)
            logits = h[:, :-1, :] @ self.head
            logp = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[:, 1:, None], axis=-1))
            acc = jnp.mean(jnp.argmax(logits, -1) == target_seq[:, 1:])
            return loss, acc

        n_blocks = ctx_code_soft.shape[1]
        G = lag + 1
        # G need not divide n_blocks (e.g. SEQ_LEN=3072=2^10*3 has no factor of 5, so lag=4 is
        # structurally impossible without this) -- pad ctx_code_soft/te with dummy trailing
        # entries up to the next multiple of G, run the padded sequence through normally, then
        # slice the loss/acc back down to the real (unpadded) positions only. Padded values never
        # need to be meaningful (masked out of the loss entirely), only shape-valid.
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)  # (B, n_blocks, D)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
            te = jnp.pad(te, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        ctx_g = ctx_tok.reshape(B, n_groups, G, D)
        te_g = te.reshape(B, n_groups, G * self.K, D)
        xe = jnp.concatenate([ctx_g, te_g], axis=2).reshape(B * n_groups, G + G * self.K, D)
        for blk in self.self_blocks:
            xe = blk(xe)
        h = self.ln_f(xe)
        h_bytes = h[:, G - 1:-1, :]  # (B*n_groups, G*K, D) -- pos G-1+j predicts target[j]
        logits = h_bytes.reshape(B, n_groups * G * self.K, -1)[:, :n_blocks * self.K, :] @ self.head
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        return loss, acc

    def reconstruct_full_recompute(self, ctx_idx: jnp.ndarray, lag: int, greedy: bool = True,
                                    temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """No incremental KV-cache -- reruns the plain batched self_blocks(...) call (the exact
        one forward() uses) from scratch at every single byte step, over the growing
        decided-so-far sequence WITHIN the current group only (groups are independent, block-
        diagonal). Diagnostic-grade correctness reference for reconstruct_kv_cache."""
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        rng = jax.random.PRNGKey(seed)

        if lag == -1:
            decided = []
            out = jnp.zeros((B, n_blocks * self.K), dtype=jnp.int32)
            L = n_blocks * self.K
            # First byte has no prediction target under pure NTP -- seed with a zero byte (never
            # read back into the loss at train time; here just needs a deterministic starting
            # embedding token consistent with position 0 existing).
            decided.append(self.target_embed[jnp.zeros((B,), dtype=jnp.int32)])
            out = out.at[:, 0].set(0)
            for t in tqdm(range(1, L), desc=f"decode_full_recompute(lag=-1,L={L})", leave=False):
                x = jnp.stack(decided, axis=1)
                for blk in self.self_blocks:
                    x = blk(x)
                h_last = self.ln_f(x)[:, -1, :]
                logits = h_last @ self.head
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng, k_ = jax.random.split(rng)
                    val = jax.random.categorical(k_, logits / temperature, axis=-1)
                out = out.at[:, t].set(val)
                decided.append(self.target_embed[val])
            return out

        G = lag + 1
        pad_blocks = (-n_blocks) % G  # see forward()'s docstring -- G need not divide n_blocks
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)  # (B, n_blocks, D)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        out = jnp.zeros((B, n_blocks_p * self.K), dtype=jnp.int32)

        for g in tqdm(range(n_groups), desc=f"decode_full_recompute(lag={lag},n_groups={n_groups},G={G})",
                      leave=False):
            group_codes = ctx_tok[:, g * G:(g + 1) * G, :]
            decided = [group_codes[:, i, :] for i in range(G)]
            for t in range(G * self.K):
                x = jnp.stack(decided, axis=1)
                for blk in self.self_blocks:
                    x = blk(x)
                h_last = self.ln_f(x)[:, -1, :]
                logits = h_last @ self.head
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng, k_ = jax.random.split(rng)
                    val = jax.random.categorical(k_, logits / temperature, axis=-1)
                out = out.at[:, g * G * self.K + t].set(val)
                decided.append(self.target_embed[val])
        return out[:, :n_blocks * self.K]

    def reconstruct_kv_cache(self, ctx_idx: jnp.ndarray, lag: int, greedy: bool = True,
                              temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """Incremental KV-cache generation -- one growing cache PER GROUP (reset between groups,
        matching their block-diagonal independence). Prefill: G self_step calls feeding the G
        known codes in (no prediction taken, pure cache writes) -- the G-1'th (last) prefill step
        DOES yield the first byte's prediction (matches forward()'s h[:,G-1:-1,:] indexing: pos
        G-1 predicts target[0]). Then G*K-1 more self_step calls, each predicting the next byte;
        the very last one (i=G*K-1) takes no prediction (nothing left in this group -- no
        `K+1`-th extra write needed here, unlike StackDecoder's UNBOUNDED single cache,
        since later groups never read this group's cache at all)."""
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        rng = jax.random.PRNGKey(seed)

        if lag == -1:
            L = n_blocks * self.K
            cache_k = jnp.zeros((len(self.self_blocks), B, self.n_kv_heads, L, hd))
            cache_v = jnp.zeros_like(cache_k)

            def self_step(x_new, ck, cv, pos):
                new_ck, new_cv = [], []
                x = x_new
                for i, blk in enumerate(self.self_blocks):
                    x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, L)
                    new_ck.append(ck_i)
                    new_cv.append(cv_i)
                return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

            self_step = jax.jit(self_step)
            out = jnp.zeros((B, L), dtype=jnp.int32)
            x_input = self.target_embed[jnp.zeros((B,), dtype=jnp.int32)]
            out = out.at[:, 0].set(0)
            for pos in tqdm(range(L - 1), desc=f"decode_kv_cache(lag=-1,L={L})", leave=False):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                logits = h @ self.head
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng, k_ = jax.random.split(rng)
                    val = jax.random.categorical(k_, logits / temperature, axis=-1)
                out = out.at[:, pos + 1].set(val)
                x_input = self.target_embed[val]
            return out

        G = lag + 1
        pad_blocks = (-n_blocks) % G  # see forward()'s docstring -- G need not divide n_blocks
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        Lg = G + G * self.K  # per-group sequence length
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)  # (B, n_blocks, D)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        out = jnp.zeros((B, n_blocks_p * self.K), dtype=jnp.int32)

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(self.self_blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, Lg)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        self_step = jax.jit(self_step)

        for g in tqdm(range(n_groups), desc=f"decode_kv_cache(lag={lag},n_groups={n_groups},G={G})",
                      leave=False):
            group_codes = ctx_tok[:, g * G:(g + 1) * G, :]
            cache_k = jnp.zeros((len(self.self_blocks), B, self.n_kv_heads, Lg, hd))
            cache_v = jnp.zeros_like(cache_k)

            h = None
            for i in range(G):
                h, cache_k, cache_v = self_step(group_codes[:, i, :], cache_k, cache_v, i)
            logits = h @ self.head
            if greedy:
                val = jnp.argmax(logits, axis=-1)
            else:
                rng, k_ = jax.random.split(rng)
                val = jax.random.categorical(k_, logits / temperature, axis=-1)
            out = out.at[:, g * G * self.K].set(val)
            x_input = self.target_embed[val]

            for i in range(G * self.K - 1):
                pos = G + i
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                logits = h @ self.head
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng, k_ = jax.random.split(rng)
                    val = jax.random.categorical(k_, logits / temperature, axis=-1)
                out = out.at[:, g * G * self.K + i + 1].set(val)
                x_input = self.target_embed[val]
        return out[:, :n_blocks * self.K]


class LagCodecModel(eqx.Module):
    """stages always has exactly ONE element -- levels above level0 are never decoded by a
    dedicated stage, only HierEncoder's own per-level NTP heads (matches the reference: codes
    above level0 come from `enc.quant.sample_next(...)`, never a separate decoder)."""
    encoder: HierEncoder
    stages: list
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        keys = jax.random.split(key, 2)
        self.encoder = HierEncoder(keys[0], cfg)
        if cfg.decoder_type == "stack":
            self.stages = [StackDecoder(keys[1], cfg)]
        elif cfg.decoder_type == "self_attn_local":
            self.stages = [StageLocalDecoder(keys[1], cfg)]
        else:
            self.stages = [StageLagDecoder(keys[1], cfg, level=0)]

    def __call__(self, flat_bytes: jnp.ndarray) -> tuple:
        enc = self.encoder(flat_bytes)
        if self.cfg.decoder_type == "stack":
            # single stage, generalized N-level + lag (see StackDecoder docstring)
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"], self.cfg.lag)
            decode_loss_total = byte_loss
        elif self.cfg.decoder_type == "self_attn_local":
            # single stage, generalized N-level: level1 (level1's code) + upper levels (level2..
            # level(n-1)'s codes, sequentially chained) -- no dedicated decoder above level0.
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"])
            decode_loss_total = byte_loss
        else:
            # single stage, conditions on level1's code (own code) with `cfg.lag` extra
            # causally-later codes grouped in -- levels above have no dedicated decoder.
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"][0], self.cfg.lag)
            decode_loss_total = byte_loss

        ntp_losses = jnp.stack([r["ntp_loss"] for r in enc["results"]])
        ntp_accs = jnp.stack([r["ntp_acc"] for r in enc["results"]])
        utils = jnp.stack([r["util"] for r in enc["results"]])
        ntp_loss_total = jnp.mean(ntp_losses)
        loss = decode_loss_total + self.cfg.ntp_weight * ntp_loss_total
        bpb = byte_loss / jnp.log(2.0)
        return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(ntp_accs), jnp.mean(utils))



def count_params(model) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


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


def save_compare_grid(gen: np.ndarray, gt: np.ndarray, path: Path, pad: int = 2) -> None:
    """gen/gt: (n,H,W,3) uint8 -- side-by-side [reconstructed | ground truth] pairs."""
    from PIL import Image
    n, h, w, c = gen.shape
    grid = np.full((n * (h + pad) + pad, 2 * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        y = pad + i * (h + pad)
        grid[y:y + h, pad:pad + w] = gen[i]
        grid[y:y + h, 2 * pad + w:2 * pad + 2 * w] = gt[i]
    Image.fromarray(grid).save(path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cast_pytree(tree, dtype):
    """Casts inexact (float) leaves to dtype; ints/bools/statics pass through untouched."""
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def to_single_device(tree, device=None):
    """Forces every array leaf onto one concrete device. Needed before calling any Pallas/Mosaic
    kernel (splash_attention) on a pytree pulled out of pmap's output via x[0] -- indexing a
    PmapSharding-backed array that way doesn't fully strip its multi-device sharding metadata,
    and Mosaic can't auto-partition -- confirmed 2026-09-09: all 4 TPU nodes crashed identically
    at the first post-pmap forward call (run_reconstruct's single_model(flat)) with
    "NotImplementedError: Mosaic kernels cannot be automatically partitioned."."""
    device = device or jax.local_devices()[0]
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, tree)


def make_train_step(optimizer, compute_dtype=jnp.bfloat16):
    """compute_dtype casts the model to bf16 (default) transiently inside the loss, forward AND
    backward matmuls run in bf16 -- master weights/grads/optimizer state stay fp32 (JAX's astype
    VJP upcasts the cotangent back automatically, so grads come out fp32 with no extra code).
    Pass compute_dtype=jnp.float32 (Config.precision="fp32") to disable -- diagnostic correctness
    mode, e.g. exact-match checks against reconstruct_full_recompute."""
    def loss_fn(model, flat_bytes):
        return cast_pytree(model, compute_dtype)(flat_bytes)

    def train_step(model, opt_state, flat_bytes):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, flat_bytes)
        grads = jax.lax.pmean(grads, axis_name="d")
        loss = jax.lax.pmean(loss, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step(compute_dtype=jnp.bfloat16):
    def eval_step(model, flat_bytes):
        _, aux = cast_pytree(model, compute_dtype)(flat_bytes)
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


CONFIG_FIELDS = ("img_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight",
                  "decoder_type", "lag", "kv_lm_mode", "dec_d_model", "dec_n_layers", "dec_n_heads",
                  "dec_n_kv_heads", "precision")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_lagcodec")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr_schedule", type=str, default="warmup_const", choices=["warmup_const", "warmup_cosine"])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={})
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=1)  # checkpoint now saves at this same cadence
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--train_subset_n", type=int, default=None)
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--qual_gen_greedy", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--qual_gen_temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--d_model", type=_tuple_arg, default=Config.d_model)
    p.add_argument("--n_layers", type=_tuple_arg, default=Config.n_layers)
    p.add_argument("--n_heads", type=_tuple_arg, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=_tuple_arg, default=Config.n_kv_heads)
    p.add_argument("--strides", type=_tuple_arg, default=Config.strides)
    p.add_argument("--code_vocab", type=int, default=Config.code_vocab)
    p.add_argument("--pq_chunks", type=int, default=Config.pq_chunks)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
    p.add_argument("--ntp_weight", type=float, default=Config.ntp_weight)
    p.add_argument("--decoder_type", type=str, default=Config.decoder_type,
                    choices=["stack", "self_attn_local", "self_attn_lag"])
    p.add_argument("--lag", type=int, default=Config.lag)
    p.add_argument("--kv_lm_mode", type=str, default=Config.kv_lm_mode, choices=["identity", "shared", "copy"])
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--dec_d_model", type=_tuple_arg, default=Config.dec_d_model)
    p.add_argument("--dec_n_layers", type=_tuple_arg, default=Config.dec_n_layers)
    p.add_argument("--dec_n_heads", type=_tuple_arg, default=Config.dec_n_heads)
    p.add_argument("--dec_n_kv_heads", type=_tuple_arg, default=Config.dec_n_kv_heads)
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
    model = LagCodecModel(rng, cfg)
    n_params = count_params(model)
    n_params_enc = count_params(model.encoder)
    n_params_dec = sum(count_params(s) for s in model.stages)

    if args.lr_schedule == "warmup_cosine":
        steps_per_epoch = len(train_np) // (args.batch_size * n_devices)
        total_steps = steps_per_epoch * args.epochs
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup_steps,
            decay_steps=total_steps, end_value=0.0)
    else:
        lr_schedule = warmup_const_schedule(args.lr, args.warmup_steps)
    if args.optimizer == "sinkgd":
        optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
    else:
        optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay, **args.optimizer_kwargs)
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

    compute_dtype = jnp.bfloat16 if cfg.precision == "bf16" else jnp.float32
    train_step = make_train_step(optimizer, compute_dtype)
    eval_step = make_eval_step(compute_dtype)

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"run args: epochs={args.epochs} lr={args.lr} warmup_steps={args.warmup_steps} "
           f"batch_size={args.batch_size} n_devices={n_devices} optimizer={args.optimizer} "
           f"optimizer_kwargs={args.optimizer_kwargs} resume_from={args.resume_from}")
    logger(f"params: {n_params / 1e6:.2f}M total (encoder {n_params_enc / 1e6:.2f}M, "
           f"decoder[{cfg.decoder_type}] {n_params_dec / 1e6:.2f}M), devices={jax.devices()}",
           n_params=n_params, n_params_enc=n_params_enc, n_params_dec=n_params_dec)

    def run_eval() -> float:
        bpbs, accs, ntp_bpbs, ntp_accs, utils = [], [], [], [], []
        for i, (flat, y) in enumerate(tqdm(val_iter, desc="val", leave=False)):
            bpb, acc, ntp_bpb, ntp_acc, util = eval_step(p_model, flat)
            bpbs.append(float(bpb[0])); accs.append(float(acc[0]))
            ntp_bpbs.append(float(ntp_bpb[0])); ntp_accs.append(float(ntp_acc[0])); utils.append(float(util[0]))
            if i >= 20:
                break
        bpb, acc = sum(bpbs) / len(bpbs), sum(accs) / len(accs)
        ntp_bpb, ntp_acc = sum(ntp_bpbs) / len(ntp_bpbs), sum(ntp_accs) / len(ntp_accs)
        util = sum(utils) / len(utils)
        logger(f"val byte_bpb={bpb:.4f} byte_acc={acc:.4f} ntp_bpb={ntp_bpb:.4f} ntp_acc={ntp_acc:.4f} util={util:.3f}",
               val_bpb=bpb, val_acc=acc, val_ntp_bpb=ntp_bpb, val_ntp_acc=ntp_acc, val_util=util)
        return bpb

    recon_prompt = train_np[:args.qual_gen_n]  # fixed set of real train images, reused every epoch
    val_recon_prompt = val_np[:args.qual_gen_n]  # fixed set of real val images, reused every epoch

    def run_reconstruct(epoch: int) -> None:
        single_model = cast_pytree(to_single_device(
            jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)), compute_dtype)
        flat = jnp.array(recon_prompt.reshape(args.qual_gen_n, SEQ_LEN))

        if cfg.decoder_type == "stack":
            # reconstruct_kv_cache_scan (jit+lax.scan) -- verified exact-match against the
            # full-recompute reference AND against the un-jitted reconstruct_kv_cache across
            # multiple depths/lag values/chunk sizes at toy scale, ~12x faster (see chat
            # 2026-09-08). Train prompt: greedy. Val prompt: sampled, low temperature.
            _, aux_tf = single_model(flat)
            tf_acc = float(aux_tf[1])
            enc_tf = single_model.encoder(flat)
            recon = single_model.stages[0].reconstruct_kv_cache_scan(enc_tf["codes"], cfg.lag,
                                                                       greedy=True, seed=epoch)
            gen_acc = float(jnp.mean(recon == flat))
            recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            gt_img = recon_prompt.astype(np.uint8)
            mse = float(np.mean((recon_img.astype(np.float32) - gt_img.astype(np.float32)) ** 2))
            out_path = run_dir / f"samples_epoch{epoch}_reconstruct.png"
            save_compare_grid(recon_img, gt_img, out_path)
            logger(f"[stack lag={cfg.lag}] saved reconstruction (recon/gt) for epoch {epoch}, "
                   f"MSE={mse:.2f}, gen_byte_acc={gen_acc:.4f}, teacher_forced_acc={tf_acc:.4f} "
                   f"(gen_consistency_gap={tf_acc - gen_acc:.4f})",
                   recon_mse=mse, gen_byte_acc=gen_acc, teacher_forced_acc=tf_acc)

            val_flat = jnp.array(val_recon_prompt.reshape(args.qual_gen_n, SEQ_LEN))
            _, val_aux_tf = single_model(val_flat)
            val_tf_acc = float(val_aux_tf[1])
            val_enc_tf = single_model.encoder(val_flat)
            val_recon = single_model.stages[0].reconstruct_kv_cache_scan(val_enc_tf["codes"], cfg.lag,
                                                                          greedy=False, temperature=0.01, seed=epoch)
            val_gen_acc = float(jnp.mean(val_recon == val_flat))
            val_recon_img = np.asarray(val_recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            val_gt_img = val_recon_prompt.astype(np.uint8)
            val_mse = float(np.mean((val_recon_img.astype(np.float32) - val_gt_img.astype(np.float32)) ** 2))
            val_out_path = run_dir / f"samples_epoch{epoch}_reconstruct_val.png"
            save_compare_grid(val_recon_img, val_gt_img, val_out_path)
            logger(f"[stack lag={cfg.lag}] VAL saved reconstruction (recon/gt) for epoch {epoch}, "
                   f"MSE={val_mse:.2f}, gen_byte_acc={val_gen_acc:.4f}, teacher_forced_acc={val_tf_acc:.4f} "
                   f"(gen_consistency_gap={val_tf_acc - val_gen_acc:.4f})",
                   val_recon_mse=val_mse, val_gen_byte_acc=val_gen_acc, val_teacher_forced_acc=val_tf_acc)
            return

        if cfg.decoder_type == "self_attn_local":
            # Corrected/generalized class -- generation (full-recompute + KV-cache) not built yet
            # (forward/training-only so far, same bring-up order as stack_level1/self_attn_lag
            # originally) -- log teacher-forced accuracy only.
            _, aux_tf = single_model(flat)
            logger(f"[self_attn_local] epoch {epoch}: teacher_forced_acc={float(aux_tf[1]):.4f} "
                   f"(generation not yet implemented for the corrected/generalized class)",
                   teacher_forced_acc=float(aux_tf[1]))
            return

        if cfg.decoder_type == "self_attn_lag":
            # Incremental KV-cache generation (verified byte-identical to the full-recompute
            # reference at every lag value, toy scale) -- much cheaper than full recompute at
            # real scale, especially for large lag/lag=max where a group spans many blocks.
            _, aux_tf = single_model(flat)
            tf_acc = float(aux_tf[1])
            enc_tf = single_model.encoder(flat)
            recon = single_model.stages[0].reconstruct_kv_cache(enc_tf["codes"][0], cfg.lag,
                                                                  greedy=args.qual_gen_greedy,
                                                                  temperature=args.qual_gen_temperature, seed=epoch)
            gen_acc = float(jnp.mean(recon == flat))
            recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            gt_img = recon_prompt.astype(np.uint8)
            mse = float(np.mean((recon_img.astype(np.float32) - gt_img.astype(np.float32)) ** 2))
            out_path = run_dir / f"samples_epoch{epoch}_reconstruct.png"
            save_compare_grid(recon_img, gt_img, out_path)
            logger(f"[self_attn_lag={cfg.lag}] saved reconstruction (recon/gt) for epoch {epoch}, "
                   f"MSE={mse:.2f}, gen_byte_acc={gen_acc:.4f}, teacher_forced_acc={tf_acc:.4f} "
                   f"(gen_consistency_gap={tf_acc - gen_acc:.4f})",
                   recon_mse=mse, gen_byte_acc=gen_acc, teacher_forced_acc=tf_acc)
            return

    def run_checkpoint(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        single_opt_state = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_opt_state)
        ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch}"
        save_checkpoint(ckpt_dir, single_model, single_opt_state, step, epoch)
        logger(f"saved checkpoint at epoch {epoch} -> {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(train_iter, desc=f"epoch {epoch}/{args.epochs}")
        epoch_losses = []
        for flat, y in pbar:
            p_model, p_opt_state, loss, (bpb, acc, ntp_bpb, ntp_acc, util) = train_step(p_model, p_opt_state, flat)
            step += 1
            loss_v = float(loss[0])
            lr_v = float(lr_schedule(step))
            epoch_losses.append(loss_v)
            pbar.set_postfix(step=step, lr=f"{lr_v:.2e}", loss=f"{loss_v:.3f}", bpb=f"{float(bpb[0]):.3f}",
                              acc=f"{float(acc[0]):.3f}", ntp_bpb=f"{float(ntp_bpb[0]):.3f}",
                              ntp_acc=f"{float(ntp_acc[0]):.3f}")
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} lr={lr_v:.6f} loss={loss_v:.4f} byte_bpb={float(bpb[0]):.4f} "
                       f"byte_acc={float(acc[0]):.4f} ntp_bpb={float(ntp_bpb[0]):.4f} "
                       f"ntp_acc={float(ntp_acc[0]):.4f} util={float(util[0]):.3f}",
                       epoch=epoch, step=step, lr=lr_v, train_loss=loss_v, train_bpb=float(bpb[0]),
                       train_acc=float(acc[0]), train_ntp_bpb=float(ntp_bpb[0]), train_ntp_acc=float(ntp_acc[0]),
                       train_util=float(util[0]))
        pbar.close()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger(f"epoch={epoch} train_loss_epoch_avg={epoch_loss:.4f}",
               epoch=epoch, train_loss_epoch_avg=epoch_loss)

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            run_eval()
            run_reconstruct(epoch)
            run_checkpoint(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
