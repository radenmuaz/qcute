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
    Attention, Block, RMSNorm, SwiGLU, apply_rope_single, load_checkpoint, rmsnorm, rope_cos_sin_pos,
    rotate_half, save_checkpoint, sinkgd, warmup_const_schedule,
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
    recon_ncode: tuple = None   # per-level (len==len(strides)) sibling-block AR-chain group size at
    # reconstruction time; None resolves to all-1s (fully block-parallel, see StageDecoder docstring).
    decoder_type: str = "cross_attn"   # "cross_attn" (StageCrossAttnDecoder) or "self_attn" (StageSelfAttnDecoder)
    lag: int = 0   # decoder_type="self_attn_lag" only -- -1: pure causal byte NTP; 0: own-code
    # only (self_attn_local's block-diagonal design); k>=1: (k+1) codes prepended per group,
    # block-diagonal across groups -- see StageLagDecoder's docstring.
    kv_lm_mode: str = "identity"   # decoder_type="self_attn_local_track1" only -- matches the
    # reference's kv_lm_mode: "identity" (no extra projection, raw code embedding -- only mode
    # implemented so far), "shared" (reuse the encoder's own level LM, reference's default),
    # "copy" (a fresh dedicated LM) -- both NotImplementedError for now, see StageLocalTrack1Decoder.
    # Decoder's own per-level transformer hparams -- DELIBERATELY separate fields from the
    # encoder's (not read off d_model/n_layers/n_heads/n_kv_heads above), so encoder and decoder
    # capacity can be tuned independently later; None resolves to mirroring the encoder's same-
    # level config exactly (current opinionated default, not a structural requirement).
    dec_d_model: tuple = None
    dec_n_layers: tuple = None
    dec_n_heads: tuple = None
    dec_n_kv_heads: tuple = None

    def __post_init__(self):
        n = len(self.strides)
        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n
        assert SEQ_LEN % math.prod(self.strides) == 0, \
            f"strides must evenly divide {SEQ_LEN} (need not multiply exactly to it)"
        if self.recon_ncode is None:
            self.recon_ncode = (1,) * n
        assert len(self.recon_ncode) == n
        assert self.decoder_type in ("cross_attn", "self_attn", "self_attn_local", "stack_track0",
                                      "self_attn_local_track1", "self_attn_local_track1_v1", "self_attn_lag")
        assert self.kv_lm_mode in ("identity", "shared", "copy")
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


class StageCrossAttnDecoder(eqx.Module):
    """Reduces level `level`'s own codes -> level `(level-1)`'s sequence (raw bytes if
    level==0, else level (level-1)'s own PQ codes). Per-block seed token prepended to each
    K-byte/code block, causal self-attn over that augmented REAL-value (teacher-forced)
    sequence, then a separate causal cross-attn stack where every position attends to this
    level's OWN code for every causally-available block (own_code_min_lag=0). Gradient: the
    cross-attn CONTEXT is embedded from the STE soft code (code_embed's differentiable float
    path, matching qcute_lagcodec_common.py's embed_for_decode(quantize()) convention) -- decode
    loss backprops into the encoder's level-`level` code (and from there into its trunk); the
    self-attn's own teacher-forced input uses the hard idx (plain embedding lookup, like byte
    inputs -- an input, not a learned target). Width/layer-count are the decoder's OWN
    cfg.dec_d_model[level]/dec_n_layers[level]/... (separate fields from the encoder's, default-
    resolved to mirror the ENCODER's same-level config exactly -- current opinionated choice for
    now, not a structural requirement).

    Training always teacher-forces the FULL per-image sequence (all of that level's blocks at
    once, ground truth everywhere -- no chaining needed, every stage's loss is independent given
    the encoder's real codes). Reconstruction (see LagCodecModel.reconstruct_tree) instead groups
    `recon_ncode` sibling blocks into one shared local AR chain per call -- ncode=1 (default): all
    blocks decode in parallel, each an isolated K-step chain; ncode=g>1: g adjacent blocks share
    one g*K-step chain (each later block genuinely attends to the earlier ones' now-known codes,
    since the existing cross-mask already allows attending to all preceding blocks) -- more
    chain-conditioned/accurate, less parallel. Purely an inference-time batching knob; training
    is unaffected."""
    target_embed: jnp.ndarray     # (256,D) if is_byte_target else (code_vocab,D)
    seed: jnp.ndarray
    dec_code_embed: jnp.ndarray   # embeds THIS stage's context code (level `level`'s own code)
    self_blocks: list
    cross_blocks: list
    ln_f: RMSNorm
    head: jnp.ndarray             # (D,256) if is_byte_target else (D,pq_chunks*code_vocab)
    K: int = eqx.field(static=True)              # = strides[level], target block size
    is_byte_target: bool = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int):
        is_byte_target = level == 0
        D = cfg.dec_d_model[level]
        self.K = cfg.strides[level]
        self.is_byte_target = is_byte_target
        self.pq_chunks, self.code_vocab = cfg.pq_chunks, cfg.code_vocab
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[level], cfg.dec_n_kv_heads[level]
        keys = jax.random.split(key, 4)
        target_vocab = 256 if is_byte_target else cfg.code_vocab
        self.target_embed = jax.random.normal(keys[0], (target_vocab, D)) * 0.02
        self.seed = jax.random.normal(keys[1], (D,)) * 0.02
        self.dec_code_embed = jax.random.normal(keys[2], (cfg.code_vocab, D)) * 0.02
        n_layers = cfg.dec_n_layers[level]
        self_keys = jax.random.split(keys[3], n_layers)
        self.self_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                             for k in self_keys]
        cross_keys = jax.random.split(keys[3], n_layers)  # separate stack, same layer count
        self.cross_blocks = [CrossBlock(k, D, self.n_heads, cfg.mlp_mult, cfg.rope_base) for k in cross_keys]
        self.ln_f = RMSNorm(D)
        head_out = 256 if is_byte_target else cfg.pq_chunks * cfg.code_vocab
        self.head = jax.random.normal(keys[2], (D, head_out)) * 0.02

    def _target_embed_of(self, target_seq: jnp.ndarray) -> jnp.ndarray:
        return self.target_embed[target_seq] if self.is_byte_target else code_embed(target_seq, self.target_embed)

    def _augment(self, te: jnp.ndarray, n_blocks: int) -> jnp.ndarray:
        """(B,n_blocks*K,D) teacher-forced target embeddings -> (B,n_blocks*(K+1),D) with a seed
        token prepended before each K-sized block."""
        B, L, D = te.shape
        xb = te.reshape(B, n_blocks, self.K, D)
        seed = jnp.broadcast_to(self.seed, (B, n_blocks, 1, D))
        return jnp.concatenate([seed, xb], axis=2).reshape(B, n_blocks * (self.K + 1), D)

    def _cross_mask(self, n_blocks: int) -> jnp.ndarray:
        Le = n_blocks * (self.K + 1)
        code_pos = jnp.arange(n_blocks) * (self.K + 1)
        query_pos = jnp.arange(Le)
        return code_pos[None, :] <= query_pos[:, None]  # (Le, n_blocks)

    def forward(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray) -> tuple:
        """Teacher-forced training pass over the FULL sequence. target_seq: (B,L) bytes if
        is_byte_target else (B,n_blocks_ctx,pq_chunks) real codes of level (level-1) -- ground
        truth own-value input. ctx_code_soft: (B,n_blocks_ctx,pq_chunks,code_vocab) STE soft code
        of level `level` (this stage's context, differentiable). Returns (loss, acc, pred_code_soft)
        -- pred_code_soft is this stage's own STE soft prediction, for the stage below to chain on."""
        B = target_seq.shape[0]
        n_blocks = ctx_code_soft.shape[1]
        te = self._target_embed_of(target_seq)
        xe = self._augment(te, n_blocks)
        for blk in self.self_blocks:
            xe = blk(xe)

        code_kv = code_embed(ctx_code_soft, self.dec_code_embed)  # (B,n_blocks,D), differentiable
        Le = xe.shape[1]
        q_pos = jnp.arange(Le)
        k_pos = jnp.arange(n_blocks) * (self.K + 1)
        mask = self._cross_mask(n_blocks)
        for blk in self.cross_blocks:
            xe = blk(xe, code_kv, q_pos, k_pos, mask)
        h = self.ln_f(xe)

        h_blocks = h.reshape(B, n_blocks, self.K + 1, -1)[:, :, :self.K, :]  # drop each block's
        # last augmented position (its "next slot" is the next block's seed, not a real target)
        target_len = n_blocks * self.K
        h_q = h_blocks.reshape(B, target_len, -1)
        logits = h_q @ self.head
        if self.is_byte_target:
            logp = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
            acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        else:
            logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
            logp = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
            acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        # STE soft prediction of THIS stage's own target values (same construction the encoder
        # uses, quantize_hard) -- lets the NEXT stage down condition on this stage's PREDICTION
        # (differentiable) instead of the encoder's real code, matching generation's actual
        # top-down chain and keeping gradient flowing end-to-end from the top stage into the
        # encoder (symmetric with the encoder's own STE end-to-end chaining).
        pred_code_soft, _ = quantize_hard(logits)
        return loss, acc, pred_code_soft

    def reconstruct_group(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                           seed: int = 0) -> jnp.ndarray:
        """KV-cached local AR reconstruction of ONE group of `g` sibling blocks at once (the
        `recon_ncode` grouping -- g==1 is a single isolated block, g>1 a merged chain), conditioned
        on FIXED context codes (real, from encoding, or predicted by the stage above). ctx_idx:
        (Bg,g,pq_chunks) int -- Bg is (batch * n_groups) flattened together, run as one batch.
        Returns (Bg, g*K) bytes if is_byte_target else (Bg, g*K, pq_chunks) int codes."""
        Bg, g, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        Le = g * (self.K + 1)
        code_kv = code_embed(ctx_idx, self.dec_code_embed)  # (Bg,g,D), fixed
        k_pos = jnp.arange(g) * (self.K + 1)
        rng = jax.random.PRNGKey(seed)

        cache_k = jnp.zeros((len(self.self_blocks), Bg, self.n_kv_heads, Le, hd))
        cache_v = jnp.zeros_like(cache_k)

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(self.self_blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, Le)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return x, jnp.stack(new_ck), jnp.stack(new_cv)

        def cross_apply(x_q, pos):
            q_pos = jnp.full((1,), pos)
            allow = (k_pos <= pos)[None, :]  # (1,g)
            x = x_q[:, None, :]
            for blk in self.cross_blocks:
                x = blk(x, code_kv, q_pos, k_pos, allow)
            return self.ln_f(x)[:, 0, :]

        self_step = jax.jit(self_step)
        cross_apply = jax.jit(cross_apply)

        if self.is_byte_target:
            out = jnp.zeros((Bg, g * self.K), dtype=jnp.int32)
        else:
            out = jnp.zeros((Bg, g * self.K, self.pq_chunks), dtype=jnp.int32)
        x_input = jnp.broadcast_to(self.seed, (Bg, D))
        pos = 0
        for b in tqdm(range(g), desc=f"decode(cross_attn,Bg={Bg},g={g},K={self.K})", leave=False):
            # K+1 self_step calls per block, matching forward()'s (K+1)-slot augmented sequence --
            # see StageSelfAttnDecoder.reconstruct_group's identical fix for the full rationale
            # (same underlying self-attn augmentation/bug, cross-attention here is unaffected).
            for t in range(self.K + 1):
                h_self, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos += 1
                if t < self.K:
                    h = cross_apply(h_self, pos - 1)
                    logits = h @ self.head
                    if not self.is_byte_target:
                        logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
                    if greedy:
                        val = jnp.argmax(logits, axis=-1)
                    else:
                        rng_local, k = jax.random.split(rng)
                        rng = rng_local
                        val = jax.random.categorical(k, logits / temperature, axis=-1)
                    flat_pos = b * self.K + t
                    out = out.at[:, flat_pos].set(val)
                    x_input = self._target_embed_of(val)
                elif b < g - 1:
                    x_input = jnp.broadcast_to(self.seed, (Bg, D))
        return out


class StageSelfAttnDecoder(eqx.Module):
    """Same level-reduction contract as StageCrossAttnDecoder (level -> level-1, bytes if
    level==0 else PQ codes) but with NO separate cross-attention stack: this level's own code
    embedding is placed directly INTO the augmented self-attn sequence as each block's start
    token (replacing the old learned-constant `seed`), so plain causal self-attention already
    gives every position the "attend to all causally-available blocks' own codes" property that
    StageCrossAttnDecoder's cross-attn mask provided separately -- cross-attn there ran once per
    AR step against a context that's fixed for the whole call anyway, so folding it into the
    single causal sequence is strictly simpler (one attention stack instead of two) for the same
    conditioning power. Gradient path unchanged: `ctx_embed` is fed the STE soft code during
    training (differentiable into the encoder), the hard idx at inference."""
    target_embed: jnp.ndarray     # (256,D) if is_byte_target else (code_vocab,D)
    ctx_embed: jnp.ndarray        # embeds THIS stage's context code (level `level`'s own code) -- the block-start token
    self_blocks: list
    ln_f: RMSNorm
    head: jnp.ndarray             # (D,256) if is_byte_target else (D,pq_chunks*code_vocab)
    K: int = eqx.field(static=True)
    is_byte_target: bool = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    local: bool = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, local: bool = False):
        """local=True: block-diagonal self-attn (StackDecoderLocal-inspired) -- every block's
        (K+1)-token augmented sequence [ctx_b, t0,...,t_{K-1}] is folded into the BATCH dimension
        instead of concatenated into one long causal sequence, so a block gets zero direct
        visibility into any other block's target bytes. Still causality-preserving overall: ctx_b
        (this level's own code for block b) is itself produced by the HierEncoder's causal
        self-attention over all bytes/codes up to and including block b, so it already carries
        everything upstream a global causal chain would otherwise need to re-derive from raw
        target bytes -- block-local decode is causally sufficient, not merely an approximation.
        Consequence: forward() and generation become the IDENTICAL computation by construction
        (n_blocks folded into batch either way), eliminating the whole cache/group-boundary/cold-
        start bug class that afflicts the global-causal-chain default (local=False) at generation
        time -- no recon_ncode grouping question even arises. Local RoPE positions 0..K per block
        (Block.__call__/step already compute cos/sin fresh from the passed array's own length --
        no code changes needed there, only how forward()/reconstruct fold n_blocks)."""
        is_byte_target = level == 0
        D = cfg.dec_d_model[level]
        self.K = cfg.strides[level]
        self.is_byte_target = is_byte_target
        self.pq_chunks, self.code_vocab = cfg.pq_chunks, cfg.code_vocab
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[level], cfg.dec_n_kv_heads[level]
        self.local = local
        keys = jax.random.split(key, 3)
        target_vocab = 256 if is_byte_target else cfg.code_vocab
        self.target_embed = jax.random.normal(keys[0], (target_vocab, D)) * 0.02
        self.ctx_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        n_layers = cfg.dec_n_layers[level]
        self_keys = jax.random.split(keys[2], n_layers)
        self.self_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                             for k in self_keys]
        self.ln_f = RMSNorm(D)
        head_out = 256 if is_byte_target else cfg.pq_chunks * cfg.code_vocab
        self.head = jax.random.normal(keys[1], (D, head_out)) * 0.02

    def _target_embed_of(self, target_seq: jnp.ndarray) -> jnp.ndarray:
        return self.target_embed[target_seq] if self.is_byte_target else code_embed(target_seq, self.target_embed)

    def _augment(self, te: jnp.ndarray, ctx_tok: jnp.ndarray, n_blocks: int) -> jnp.ndarray:
        """te: (B,n_blocks*K,D) teacher-forced target embeddings; ctx_tok: (B,n_blocks,D) this
        level's own per-block code embedding, used as the block-start token in place of a seed."""
        B, L, D = te.shape
        xb = te.reshape(B, n_blocks, self.K, D)
        return jnp.concatenate([ctx_tok[:, :, None, :], xb], axis=2).reshape(B, n_blocks * (self.K + 1), D)

    def _run_self_blocks(self, xe: jnp.ndarray, B: int, n_blocks: int) -> jnp.ndarray:
        """xe: (B, n_blocks*(K+1), D). local=False (default): one global causal chain, exactly as
        before. local=True: fold n_blocks into the batch dim so each block's (K+1)-token sequence
        gets its own independent causal self-attn call (local RoPE positions 0..K) -- block-diagonal,
        zero cross-block visibility, see __init__ docstring. Returns (B, n_blocks, K+1, D)."""
        D = xe.shape[-1]
        if self.local:
            xe = xe.reshape(B * n_blocks, self.K + 1, D)
            for blk in self.self_blocks:
                xe = blk(xe)
            h = self.ln_f(xe)
            return h.reshape(B, n_blocks, self.K + 1, D)
        else:
            for blk in self.self_blocks:
                xe = blk(xe)
            h = self.ln_f(xe)
            return h.reshape(B, n_blocks, self.K + 1, D)

    def forward(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray) -> tuple:
        """Same contract as StageCrossAttnDecoder.forward."""
        B = target_seq.shape[0]
        n_blocks = ctx_code_soft.shape[1]
        te = self._target_embed_of(target_seq)
        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)  # (B,n_blocks,D), differentiable into encoder
        xe = self._augment(te, ctx_tok, n_blocks)
        h = self._run_self_blocks(xe, B, n_blocks)

        h_blocks = h[:, :, :self.K, :]
        target_len = n_blocks * self.K
        h_q = h_blocks.reshape(B, target_len, -1)
        logits = h_q @ self.head
        if self.is_byte_target:
            logp = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
            acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        else:
            logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
            logp = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
            acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        # STE soft prediction of THIS stage's own target values (same construction the encoder
        # uses, quantize_hard) -- lets the NEXT stage down condition on this stage's PREDICTION
        # (differentiable) instead of the encoder's real code, matching generation's actual
        # top-down chain and keeping gradient flowing end-to-end from the top stage into the
        # encoder (symmetric with the encoder's own STE end-to-end chaining).
        pred_code_soft, _ = quantize_hard(logits)
        return loss, acc, pred_code_soft

    def predict_teacher_forced(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray) -> jnp.ndarray:
        """Same computation as forward() up to the argmax -- per-position predicted values (bytes
        if is_byte_target else pq codes), for gen-consistency diagnostics (compare against
        reconstruct_group's free-running output byte-for-byte, not just aggregate accuracy)."""
        B = target_seq.shape[0]
        n_blocks = ctx_code_soft.shape[1]
        te = self._target_embed_of(target_seq)
        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)
        xe = self._augment(te, ctx_tok, n_blocks)
        h = self._run_self_blocks(xe, B, n_blocks)
        h_blocks = h[:, :, :self.K, :]
        h_q = h_blocks.reshape(B, n_blocks * self.K, -1)
        logits = h_q @ self.head
        if not self.is_byte_target:
            logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
        return jnp.argmax(logits, -1)

    def reconstruct_group(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                           seed: int = 0) -> jnp.ndarray:
        """Same contract as StageCrossAttnDecoder.reconstruct_group -- KV-cached, no cross-attn
        call at all: each block's start token is that block's own (fixed) context-code embedding."""
        Bg, g, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        Le = g * (self.K + 1)
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)  # (Bg,g,D), fixed per block
        rng = jax.random.PRNGKey(seed)

        cache_k = jnp.zeros((len(self.self_blocks), Bg, self.n_kv_heads, Le, hd))
        cache_v = jnp.zeros_like(cache_k)

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(self.self_blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, Le)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        self_step = jax.jit(self_step)

        if self.is_byte_target:
            out = jnp.zeros((Bg, g * self.K), dtype=jnp.int32)
        else:
            out = jnp.zeros((Bg, g * self.K, self.pq_chunks), dtype=jnp.int32)
        x_input = ctx_tok[:, 0, :]
        pos = 0
        for b in tqdm(range(g), desc=f"decode(self_attn,Bg={Bg},g={g},K={self.K})", leave=False):
            # K+1 self_step calls per block, matching forward()'s (K+1)-slot augmented sequence
            # ([ctx_b, t0,...,t_{K-1}]) exactly -- the first K calls predict this block's own K
            # real bytes (unshifted: ctx_b's output predicts t0, t0's output predicts t1, ...);
            # the extra (K+1-th) call feeds t_{K-1} itself through self-attention with NO
            # prediction taken -- forward() does the same (that position is a real KV entry every
            # later block's causal attention can see, even though its own query output is dropped
            # from the loss). Skipping this extra call (as the previous version did) silently
            # drops t_{K-1} from the cache entirely and shifts every later block's positions by
            # one -- confirmed via a teacher-forced step-vs-forward consistency check: block 0
            # matched forward() to float32 noise, every block after it diverged by up to ~34 in
            # logit space (~75% argmax mismatch) despite ~99.7% teacher-forced accuracy.
            for t in range(self.K + 1):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos += 1
                if t < self.K:
                    logits = h @ self.head
                    if not self.is_byte_target:
                        logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
                    if greedy:
                        val = jnp.argmax(logits, axis=-1)
                    else:
                        rng_local, k = jax.random.split(rng)
                        rng = rng_local
                        val = jax.random.categorical(k, logits / temperature, axis=-1)
                    flat_pos = b * self.K + t
                    out = out.at[:, flat_pos].set(val)
                    x_input = self._target_embed_of(val)
                elif b < g - 1:
                    x_input = ctx_tok[:, b + 1, :]
        return out

    def reconstruct_block_local(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                                 seed: int = 0) -> jnp.ndarray:
        """Generation for local=True: folds n_blocks into the batch dim (like forward()) so every
        block's (K+1)-token chain runs in lockstep -- K+1 TOTAL sequential self_step calls
        regardless of n_blocks (all blocks decoded in parallel at each within-block offset t),
        vs. reconstruct_group's n_blocks*(K+1) sequential calls. This is the identical computation
        forward() does (n_blocks folded into batch, local positions/cache per block) run
        incrementally -- no group-boundary cold starts are even possible since there's no grouping
        construct, every block already has zero cross-block visibility by construction (see
        __init__ docstring)."""
        assert self.local, "reconstruct_block_local requires local=True (use reconstruct_group otherwise)"
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        Le = self.K + 1
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)  # (B, n_blocks, D)
        Bn = B * n_blocks
        x_input = ctx_tok.reshape(Bn, D)
        rng = jax.random.PRNGKey(seed)

        cache_k = jnp.zeros((len(self.self_blocks), Bn, self.n_kv_heads, Le, hd))
        cache_v = jnp.zeros_like(cache_k)

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(self.self_blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, Le)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        self_step = jax.jit(self_step)

        if self.is_byte_target:
            out = jnp.zeros((Bn, self.K), dtype=jnp.int32)
        else:
            out = jnp.zeros((Bn, self.K, self.pq_chunks), dtype=jnp.int32)

        pos = 0
        for t in tqdm(range(self.K + 1),
                      desc=f"decode(self_attn_local,B={B},n_blocks={n_blocks},K={self.K})", leave=False):
            h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
            pos += 1
            if t < self.K:
                logits = h @ self.head
                if not self.is_byte_target:
                    logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng, k = jax.random.split(rng)
                    val = jax.random.categorical(k, logits / temperature, axis=-1)
                out = out.at[:, t].set(val)
                x_input = self._target_embed_of(val)
        if self.is_byte_target:
            return out.reshape(B, n_blocks * self.K)
        else:
            return out.reshape(B, n_blocks * self.K, self.pq_chunks)

    def reconstruct_sequential(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                                seed: int = 0) -> jnp.ndarray:
        """Diagnostic/confirmation mode: decode the ENTIRE sequence (however many blocks ctx_idx
        holds -- works at any level, not hardcoded to level0) as ONE continuous causal chain
        starting from the very first block, with a single growing KV cache spanning all of it --
        no group-boundary resets. This exactly matches forward()'s single-pass causal structure
        (every block genuinely sees every real/generated block before it), unlike
        reconstruct_group's grouped/isolated-per-group decoding (recon_ncode>1 fixed, independent
        islands with zero cross-group visibility). Just reconstruct_group with the WHOLE sequence
        as one group (g=n_blocks, Bg=B) -- no separate mechanism needed, reconstruct_group was
        already general enough. Used to confirm whether the group-boundary cold-start problem
        (each group's first block having no visibility into other groups' bytes) is what's
        actually suppressing recon_ncode>1 reconstruction quality -- much slower (n_blocks*(K+1)
        sequential steps, no batch-parallelism across groups), not meant for routine use."""
        return self.reconstruct_group(ctx_idx, greedy=greedy, temperature=temperature, seed=seed)

    def reconstruct_full_recompute(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                                    seed: int = 0) -> jnp.ndarray:
        """Mirrors qcute_lagcodec_decoder.py's actual _stack_generate_blockwise generation
        mechanism EXACTLY: no incremental KV-cache at all (unlike reconstruct_group/
        reconstruct_sequential, which use Block.step's cache -- proven equivalent to this via a
        teacher-forced consistency check, but this method removes even the possibility of a cache
        bug by construction). At every single byte step, re-embeds the ENTIRE decided-so-far
        augmented sequence and reruns the plain BATCHED self_blocks(...) call (the exact same one
        forward() uses) from scratch -- one continuous, unbounded, ungrouped causal chain (the
        reference's documented default, window=None -- "sync... one continuous causal chain across
        every block"; their windowed/grouped variant is explicitly labeled an "async ablation", not
        the default). O(n_blocks^2) total cost (recomputes the growing prefix every step) --
        diagnostic-grade confirmation only, not for routine/large-scale use."""
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)  # (B, n_blocks, D), fixed
        rng = jax.random.PRNGKey(seed)

        decided = []  # python list of (B,D) embeddings, in augmented-sequence order
        if self.is_byte_target:
            out = jnp.zeros((B, n_blocks * self.K), dtype=jnp.int32)
        else:
            out = jnp.zeros((B, n_blocks * self.K, self.pq_chunks), dtype=jnp.int32)
        for b in tqdm(range(n_blocks), desc=f"decode_full_recompute(self_attn,n_blocks={n_blocks},K={self.K})",
                      leave=False):
            decided.append(ctx_tok[:, b, :])
            for t in range(self.K):
                xe = jnp.stack(decided, axis=1)  # (B, L, D) -- L grows by 1 every step
                for blk in self.self_blocks:
                    xe = blk(xe)  # plain batched causal call, identical to forward()'s
                h_last = self.ln_f(xe)[:, -1, :]
                logits = h_last @ self.head
                if not self.is_byte_target:
                    logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
                if greedy:
                    val = jnp.argmax(logits, axis=-1)
                else:
                    rng_local, k = jax.random.split(rng)
                    rng = rng_local
                    val = jax.random.categorical(k, logits / temperature, axis=-1)
                out = out.at[:, b * self.K + t].set(val)
                decided.append(self._target_embed_of(val))
        return out


class Track0Layer(eqx.Module):
    """One layer of qcute_lagcodec_decoder.py's encode_like_self_attn_decode + seed_query_decode
    (StackDecoder's actual track0 mechanism -- direct 1:1 port, NOT the seed-prepended-as-token
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
        """pass1's self-attn: causal over real bytes. Returns (attn_out, k_full, v_full) -- k/v
        POST repeat_kv (full n_heads), exactly what encode_like_self_attn_decode saves for reuse."""
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
        k_full, v_full = self._repeat_kv(k), self._repeat_kv(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k_full) * scale
        causal = pos_real[:, None] >= pos_real[None, :]
        logits = jnp.where(causal[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v_full).transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out, k_full, v_full

    def cross_attn_own_code(self, x: jnp.ndarray, code_kv: jnp.ndarray, q_pos: jnp.ndarray,
                             code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> jnp.ndarray:
        """forward_cross: SHARED wq/wk/wv/out, applied to x_q=norm1(x) (x already updated by the
        self-attn residual) and x_kv=norm1(code_kv)."""
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
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        logits = jnp.where(cross_mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out

    def forward_pass1(self, x: jnp.ndarray, code_kv: jnp.ndarray, pos_real: jnp.ndarray,
                       code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> tuple:
        a, k_saved, v_saved = self.self_attn_and_save(x, pos_real)
        x = x + a
        x = x + self.cross_attn_own_code(x, code_kv, pos_real, code_pos, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x, k_saved, v_saved

    def seed_pass2(self, x_seed: jnp.ndarray, saved_k: jnp.ndarray, saved_v: jnp.ndarray,
                    code_kv: jnp.ndarray, seed_pos: jnp.ndarray, self_mask: jnp.ndarray,
                    code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> jnp.ndarray:
        """pass2: seed as PURE query against pass1's saved real-byte k/v (never recomputed, never
        itself a key), then cross-attn to the own-block code. self_mask rows that are entirely
        False (block 0 -- no real bytes precede it) get an explicit zero self-attn contribution,
        matching seed_step's `if cache_k is None: a = zeros(...)` special case (softmax over an
        all -1e9 row would otherwise give bogus uniform attention, not the reference's true zero)."""
        B, Tn, D = x_seed.shape
        hd = D // self.n_heads
        xn = self.norm1(x_seed)
        q = (xn @ self.wq).reshape(B, Tn, self.n_heads, hd).transpose(0, 2, 1, 3)
        q = rmsnorm(q, self.q_norm)
        cos_q, sin_q = rope_cos_sin_for_positions(seed_pos, hd, self.rope_base)
        q = q * cos_q[None, None] + rotate_half(q) * sin_q[None, None]
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, saved_k) * scale
        logits = jnp.where(self_mask[None, None], logits, -1e9)
        row_has_valid = jnp.any(self_mask, axis=-1)  # (Tn,)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, saved_v)
        y = jnp.where(row_has_valid[None, None, :, None], y, 0.0)
        y = y.transpose(0, 2, 1, 3).reshape(B, Tn, D)
        x = x_seed + y @ self.out
        x = x + self.cross_attn_own_code(x, code_kv, seed_pos, code_pos, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x

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

    def seed_step(self, x_seed: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, block_start,
                  T_max: int, code_kv: jnp.ndarray, code_pos: jnp.ndarray, cross_mask: jnp.ndarray) -> jnp.ndarray:
        """Incremental form of seed_pass2: seed queries the growing global cache with a STRICT
        `< block_start` mask (never sees this new block's own not-yet-written bytes -- there are
        none yet regardless, but strict for clarity/symmetry with self_mask's semantics). block 0
        (block_start=0, nothing valid) gets the same explicit zero self-attn override as
        seed_pass2 -- a `row_has_valid` scalar here (single query, not per-row) rather than a
        per-position vector."""
        B, D = x_seed.shape
        hd = D // self.n_heads
        xn = self.norm1(x_seed)
        q = (xn @ self.wq).reshape(B, self.n_heads, hd)
        q = rmsnorm(q, self.q_norm)
        cos, sin = rope_cos_sin_pos(block_start, hd, self.rope_base)
        q = apply_rope_single(q, cos, sin)
        k_full, v_full = self._repeat_kv(cache_k), self._repeat_kv(cache_v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
        valid = jnp.arange(T_max) < block_start
        logits = jnp.where(valid[None, None, :], logits, -1e9)
        row_has_valid = jnp.any(valid)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bht,bhtd->bhd", attn, v_full).reshape(B, D)
        y = jnp.where(row_has_valid, y, 0.0)
        x = x_seed + y @ self.out
        x = x + self.cross_attn_own_code_single(x, code_kv, block_start, code_pos, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class StackDecoderTrack0(eqx.Module):
    """1:1 port of qcute_lagcodec_decoder.py's StackDecoder track0 (level i's byte/code decode
    conditioned on level (i+1)'s own-block code, own_code_min_lag=0 hardcoded -- track1+ (level
    (i+2) and above) NOT YET implemented, this is track0 only). Two-pass mechanism:
    encode_like_self_attn_decode (plain causal self-attn over real target values, own-code cross-
    attn spliced in per layer, saves each layer's real-token k/v) then seed_query_decode (a
    per-block trainable seed token as pure query against those saved k/v, plus its own cross-attn
    to the code) -- see Track0Layer's docstring for exactly how this differs from
    StageSelfAttnDecoder/StageCrossAttnDecoder's (divergent, seed-prepended-as-a-token) design.
    h0 = [seed's output (predicts each block's own byte 0), pass1's real-token hidden states at
    LOCAL positions 0..K-2 (predict bytes 1..K-1, standard next-token shift WITHIN the block)] --
    same own_block_decode_loss UNSHIFTED target alignment as the other decoders, only the h0
    computation mechanism differs."""
    seed: jnp.ndarray             # bb.self_code_const -- per-level trainable seed, NOT per-block
    target_embed: jnp.ndarray     # (256,D) if is_byte_target else (code_vocab,D)
    dec_code_embed: jnp.ndarray   # embeds this stage's context code (level `level`'s own code)
    layers: list                  # list of Track0Layer
    ln_f: RMSNorm
    head: jnp.ndarray
    K: int = eqx.field(static=True)
    is_byte_target: bool = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int):
        is_byte_target = level == 0
        D = cfg.dec_d_model[level]
        self.K = cfg.strides[level]
        self.is_byte_target = is_byte_target
        self.pq_chunks, self.code_vocab = cfg.pq_chunks, cfg.code_vocab
        self.n_heads, self.n_kv_heads = cfg.dec_n_heads[level], cfg.dec_n_kv_heads[level]
        keys = jax.random.split(key, 4)
        target_vocab = 256 if is_byte_target else cfg.code_vocab
        self.target_embed = jax.random.normal(keys[0], (target_vocab, D)) * 0.02
        self.seed = jax.random.normal(keys[1], (D,)) * 0.02
        self.dec_code_embed = jax.random.normal(keys[2], (cfg.code_vocab, D)) * 0.02
        n_layers = cfg.dec_n_layers[level]
        layer_keys = jax.random.split(keys[3], n_layers)
        self.layers = [Track0Layer(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                       for k in layer_keys]
        self.ln_f = RMSNorm(D)
        head_out = 256 if is_byte_target else cfg.pq_chunks * cfg.code_vocab
        self.head = jax.random.normal(keys[2], (D, head_out)) * 0.02

    def _target_embed_of(self, target_seq: jnp.ndarray) -> jnp.ndarray:
        return self.target_embed[target_seq] if self.is_byte_target else code_embed(target_seq, self.target_embed)

    def forward(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray) -> tuple:
        """Teacher-forced training pass, full sequence (no grouping -- track0 is inherently one
        continuous causal chain, matching the reference exactly). target_seq: (B,L) real own-value
        sequence. ctx_code_soft: (B,n_blocks,pq_chunks,code_vocab) STE soft code of level `level+1`
        this level's context (differentiable into the encoder)."""
        B = target_seq.shape[0]
        n_blocks = ctx_code_soft.shape[1]
        L = n_blocks * self.K
        D = self.target_embed.shape[-1]

        x0 = self._target_embed_of(target_seq)[:, :L, :]
        code_kv = code_embed(ctx_code_soft, self.dec_code_embed)  # (B,n_blocks,D), differentiable

        pos_real = jnp.arange(L)
        code_pos = jnp.arange(n_blocks) * self.K
        block_lag = (pos_real[:, None] // self.K) - code_pos[None, :] // self.K  # (L,n_blocks)
        cross_mask_pass1 = block_lag >= 0  # own_code_min_lag=0, unbounded window

        x = x0
        saved_k, saved_v = [], []
        for layer in self.layers:
            x, k_i, v_i = layer.forward_pass1(x, code_kv, pos_real, code_pos, cross_mask_pass1)
            saved_k.append(k_i)
            saved_v.append(v_i)
        h_real = self.ln_f(x)  # ln_f applied once, after the full layer stack (saved_k/v above are pre-ln_f)

        seed_pos = jnp.arange(n_blocks) * self.K
        byte_pos = jnp.arange(L)
        self_mask = byte_pos[None, :] < seed_pos[:, None]  # (n_blocks,L) -- strictly before this block
        seed_block_lag = jnp.arange(n_blocks)[:, None] - jnp.arange(n_blocks)[None, :]
        cross_mask_pass2 = seed_block_lag >= 0  # (n_blocks,n_blocks)

        x_seed = jnp.broadcast_to(self.seed, (B, n_blocks, D))
        for layer, k_i, v_i in zip(self.layers, saved_k, saved_v):
            x_seed = layer.seed_pass2(x_seed, k_i, v_i, code_kv, seed_pos, self_mask, code_pos, cross_mask_pass2)
        h_seed = self.ln_f(x_seed)  # (B,n_blocks,D)

        h_real_b = h_real.reshape(B, n_blocks, self.K, D)
        h0 = jnp.concatenate([h_seed[:, :, None, :], h_real_b[:, :, :self.K - 1, :]], axis=2)
        h0 = h0.reshape(B, L, D)

        logits = h0 @ self.head
        if not self.is_byte_target:
            logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[:, :L][..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target_seq[:, :L])
        return loss, acc

    def reconstruct(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                     seed: int = 0) -> jnp.ndarray:
        """1:1 port of _stack_generate_blockwise -- the reference's ACTUAL, fixed generation path
        ("generation fix chat 2026-08-20"), NOT the separate encode_like_step/seed_step incremental
        primitives (StackDecoder.generate_kv_cache -- an older/parallel path the reference itself
        guards with an equality assertion against known divergence risk from
        _stack_generate_blockwise; a from-scratch JAX port of THAT path produced a real divergence
        here too, starting block 2, so this uses the reference's own definitively-correct method
        instead). FULL RECOMPUTE: reruns forward_pass1+seed_pass2 (the SAME methods forward() uses,
        already verified byte-identical to the batched training path) over the growing buffer of
        bytes-so-far, once per NEW byte (not incremental -- O(n_blocks^2) total, matching the
        reference's own documented cost), teacher-forcing each predicted byte back in before the
        next step. track0-only (no upper tracks)."""
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        code_kv_all = code_embed(ctx_idx, self.dec_code_embed)  # (B,n_blocks,D), fixed
        rng = jax.random.PRNGKey(seed)

        if self.is_byte_target:
            buf = jnp.zeros((B, n_blocks * self.K), dtype=jnp.int32)
        else:
            buf = jnp.zeros((B, n_blocks * self.K, self.pq_chunks), dtype=jnp.int32)

        for b in tqdm(range(n_blocks), desc=f"decode(stack_track0,n_blocks={n_blocks},K={self.K})", leave=False):
            n_blocks_now = b + 1
            L_now = n_blocks_now * self.K
            code_kv_now = code_kv_all[:, :n_blocks_now, :]
            for t in range(self.K):
                x0 = self._target_embed_of(buf[:, :L_now])
                pos_real = jnp.arange(L_now)
                code_pos = jnp.arange(n_blocks_now) * self.K
                block_lag = (pos_real[:, None] // self.K) - code_pos[None, :] // self.K
                cross_mask_pass1 = block_lag >= 0
                x = x0
                saved_k, saved_v = [], []
                for layer in self.layers:
                    x, k_i, v_i = layer.forward_pass1(x, code_kv_now, pos_real, code_pos, cross_mask_pass1)
                    saved_k.append(k_i)
                    saved_v.append(v_i)
                h_real = self.ln_f(x)

                if t == 0:
                    seed_pos = jnp.arange(n_blocks_now) * self.K
                    byte_pos = jnp.arange(L_now)
                    self_mask = byte_pos[None, :] < seed_pos[:, None]
                    seed_block_lag = jnp.arange(n_blocks_now)[:, None] - jnp.arange(n_blocks_now)[None, :]
                    cross_mask_pass2 = seed_block_lag >= 0
                    x_seed = jnp.broadcast_to(self.seed, (B, n_blocks_now, D))
                    for layer, k_i, v_i in zip(self.layers, saved_k, saved_v):
                        x_seed = layer.seed_pass2(x_seed, k_i, v_i, code_kv_now, seed_pos, self_mask,
                                                   code_pos, cross_mask_pass2)
                    h_seed = self.ln_f(x_seed)
                    h_query = h_seed[:, -1, :]
                else:
                    h_real_b = h_real.reshape(B, n_blocks_now, self.K, D)
                    h_query = h_real_b[:, -1, t - 1, :]

                logits = self._logits_from_h(h_query)
                val = self._sample(logits, greedy, temperature, rng, seed + b * self.K + t)
                buf = self._write_out(buf, b * self.K + t, val)
        return buf

    def reconstruct_kv_cache(self, ctx_idx: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                              seed: int = 0) -> jnp.ndarray:
        """NEW, correctly-derived incremental KV-cache generation -- re-attempted after the
        previous encode_like_step/seed_step port was found to diverge (~75% argmax match, wrong
        from block 2 onward) and deleted; that failure is why reconstruct() (full recompute) was
        built as the verified-correct reference in the first place. This version is checked
        directly against that reference via a teacher-forced consistency script, not assumed
        correct by construction.

        One GROWING GLOBAL self-attn cache (size n_blocks*K, matching the reference's genuinely
        unbounded track0 self-attention -- unlike StageLocalTrack1Decoder's block-diagonal cache,
        this one spans every block). Per block b: seed_step queries the cache accumulated from
        ALL STRICTLY EARLIER blocks (block_start=b*K) to predict byte0; then K self_step calls (one
        per byte 0..K-1, each writing that REAL byte's k/v into the cache at its own position)
        produce predictions for bytes 1..K-1 from steps t=0..K-2 -- the K-1'th (t=K-1) self_step
        call takes NO prediction, existing purely to write the block's own LAST byte into the cache
        so later blocks' self-attention/seed can see it (same K+1-total-calls-per-block pattern as
        the StageSelfAttnDecoder/StageLocalTrack1Decoder K+1 cache-write fix earlier this session --
        omitting it would silently drop each block's last byte from every later block's causal
        view). code_kv (own-block code, small) is recomputed fresh every step, not cached -- cheap,
        and cross_mask is always all-True since code_kv is always sliced to exactly the causally-
        available blocks already."""
        B, n_blocks, _ = ctx_idx.shape
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        L_max = n_blocks * self.K
        code_kv_all = code_embed(ctx_idx, self.dec_code_embed)  # (B, n_blocks, D), fixed
        rng = jax.random.PRNGKey(seed)

        if self.is_byte_target:
            buf = jnp.zeros((B, L_max), dtype=jnp.int32)
        else:
            buf = jnp.zeros((B, L_max, self.pq_chunks), dtype=jnp.int32)

        cache_k = jnp.zeros((len(self.layers), B, self.n_kv_heads, L_max, hd))
        cache_v = jnp.zeros_like(cache_k)

        def seed_step_all(ck, cv, block_start, code_kv_now, cross_mask, code_pos_now):
            x = jnp.broadcast_to(self.seed, (B, D))
            for i, layer in enumerate(self.layers):
                x = layer.seed_step(x, ck[i], cv[i], block_start, L_max, code_kv_now, code_pos_now, cross_mask)
            return self.ln_f(x)

        def self_step_all(x_new, ck, cv, pos, code_kv_now, cross_mask, code_pos_now):
            new_ck, new_cv = [], []
            x = x_new
            for i, layer in enumerate(self.layers):
                x, ck_i, cv_i = layer.self_step(x, ck[i], cv[i], pos, L_max, code_kv_now, code_pos_now, cross_mask)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return self.ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        seed_step_all = jax.jit(seed_step_all)
        self_step_all = jax.jit(self_step_all)

        for b in tqdm(range(n_blocks), desc=f"decode_kv_cache(stack_track0,n_blocks={n_blocks},K={self.K})",
                      leave=False):
            n_blocks_now = b + 1
            block_start = b * self.K
            code_kv_now = code_kv_all[:, :n_blocks_now, :]
            code_pos_now = jnp.arange(n_blocks_now) * self.K
            cross_mask = jnp.ones((n_blocks_now,), dtype=bool)

            h_seed = seed_step_all(cache_k, cache_v, block_start, code_kv_now, cross_mask, code_pos_now)
            logits = self._logits_from_h(h_seed)
            val = self._sample(logits, greedy, temperature, rng, seed + block_start)
            buf = self._write_out(buf, block_start, val)
            x_input = self._target_embed_of(val)

            for t in range(self.K):
                pos = block_start + t
                h, cache_k, cache_v = self_step_all(x_input, cache_k, cache_v, pos, code_kv_now,
                                                     cross_mask, code_pos_now)
                if t < self.K - 1:
                    logits = self._logits_from_h(h)
                    val = self._sample(logits, greedy, temperature, rng, seed + pos + 1)
                    buf = self._write_out(buf, pos + 1, val)
                    x_input = self._target_embed_of(val)
        return buf

    def _logits_from_h(self, h: jnp.ndarray) -> jnp.ndarray:
        logits = h @ self.head
        if not self.is_byte_target:
            logits = reshape_pq(logits, self.pq_chunks, self.code_vocab)
        return logits

    def _sample(self, logits: jnp.ndarray, greedy: bool, temperature: float, rng, seed: int) -> jnp.ndarray:
        if greedy:
            return jnp.argmax(logits, axis=-1)
        return jax.random.categorical(jax.random.fold_in(rng, seed), logits / temperature, axis=-1)

    def _write_out(self, out: jnp.ndarray, pos: int, val: jnp.ndarray) -> jnp.ndarray:
        return out.at[:, pos].set(val)


class StageLocalTrack1DecoderV1(eqx.Module):
    """FORKED (kept for reference/comparison, see StageLocalTrack1Decoder for the corrected,
    generalized replacement) -- this version has TWO known divergences from the reference,
    caught 2026-09-08: (1) track0's own-code conditioning is code-as-token self-attention, NOT
    the reference's actual self-attn+cross-attn mechanism (block_local_track0_decode); (2)
    hardcoded to exactly track0+track1 (3 levels), not generalized to arbitrary depth like the
    reference's cond_depth loop. decoder_type="self_attn_local_track1_v1".

    Single decoder for level0's bytes -- NO per-level cascade of stages (unlike
    self_attn/cross_attn/self_attn_local, which each have one stage per level chaining
    codes[i]->codes[i-1]). Requires n_levels>=3 (e.g. strides=(2,2,1)): levels 1 and 2 have no
    dedicated decoder at all -- their own codes come solely from HierEncoder's own per-level NTP
    head, exactly like StackDecoder's reference design (see qcute_lagcodec_decoder.py's NAMING
    docstring: "level i's decoder never has anything of its own to condition on beyond the
    bytes/values it's reconstructing").

    Track0 (own code = level1's code): block-diagonal self-attn, StackDecoderLocal-inspired (see
    StageSelfAttnDecoder's local=True docstring) -- causally sufficient on its own in principle
    (level1's code already summarizes everything upstream), but empirically capacity-starved
    (self_attn_local shallow2 ablation plateaued at ~7% teacher-forced acc). Track1 (level2's
    code) adds cross-attention on top -- reference's cross_attn_stage, global causal (code
    becomes visible to a query once its own covered span has fully completed, unbounded window)
    -- giving level0 strictly more conditioning information without falling back to the
    self_attn/cross_attn architectures' raw-cross-block-byte-copying shortcut (this decoder never
    sees any other block's real bytes, only codes)."""
    target_embed: jnp.ndarray
    ctx_embed: jnp.ndarray        # embeds level1's code (track0 own-code, block-start token)
    self_blocks: list             # track0: block-diagonal self-attn
    track1_embed: jnp.ndarray     # embeds level2's code (track1)
    cross_blocks: list            # track1: CrossBlock stack (global causal cross-attn)
    ln_f: RMSNorm
    head: jnp.ndarray
    K0: int = eqx.field(static=True)   # level0 stride == bytes per track0 block
    K1: int = eqx.field(static=True)   # level1 stride == level0-blocks per level1-code == per level2-code (stride[2]=1)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        assert len(cfg.strides) >= 3, "StageLocalTrack1Decoder needs n_levels>=3 (track0+track1)"
        D = cfg.dec_d_model[0]
        self.K0 = cfg.strides[0]
        self.K1 = cfg.strides[1]
        self.code_vocab = cfg.code_vocab
        self.n_heads = cfg.dec_n_heads[0]
        self.n_kv_heads = cfg.dec_n_kv_heads[0]
        n_layers = cfg.dec_n_layers[0]
        keys = jax.random.split(key, 5)
        self.target_embed = jax.random.normal(keys[0], (256, D)) * 0.02
        self.ctx_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        self.track1_embed = jax.random.normal(keys[2], (cfg.code_vocab, D)) * 0.02
        self_keys = jax.random.split(keys[3], n_layers)
        self.self_blocks = [Block(k, D, self.n_heads, cfg.dec_n_kv_heads[0], cfg.mlp_mult, cfg.rope_base)
                             for k in self_keys]
        cross_keys = jax.random.split(keys[4], n_layers)
        self.cross_blocks = [CrossBlock(k, D, self.n_heads, cfg.mlp_mult, cfg.rope_base) for k in cross_keys]
        self.ln_f = RMSNorm(D)
        self.head = jax.random.normal(keys[2], (D, 256)) * 0.02

    def _augment(self, te: jnp.ndarray, ctx_tok: jnp.ndarray, n_blocks: int) -> jnp.ndarray:
        B, L, D = te.shape
        xb = te.reshape(B, n_blocks, self.K0, D)
        return jnp.concatenate([ctx_tok[:, :, None, :], xb], axis=2).reshape(B, n_blocks * (self.K0 + 1), D)

    def forward(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray,
                track1_code_soft: jnp.ndarray) -> tuple:
        """target_seq: (B, SEQ_LEN) real bytes. ctx_code_soft: (B, n_blocks0, pq) level1's own
        code (track0). track1_code_soft: (B, n_blocks1, pq) level2's own code (track1)."""
        B = target_seq.shape[0]
        n_blocks0 = ctx_code_soft.shape[1]
        D = self.target_embed.shape[-1]
        te = self.target_embed[target_seq]
        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)  # (B,n_blocks0,D), differentiable into encoder
        xe = self._augment(te, ctx_tok, n_blocks0)

        # track0: block-diagonal self-attn (n_blocks0 folded into batch, local RoPE positions)
        x = xe.reshape(B * n_blocks0, self.K0 + 1, D)
        for blk in self.self_blocks:
            x = blk(x)
        x = x.reshape(B, n_blocks0, self.K0 + 1, D)[:, :, :self.K0, :]
        target_len = n_blocks0 * self.K0
        x = x.reshape(B, target_len, D)  # pre-final-norm residual stream, unshifted (x[p] predicts target[p])

        # track1: global causal cross-attn to level2's code (own code chain never needed -- the
        # topmost level's own code is genuinely produced by its own NTP head, not this decoder).
        n_blocks1 = track1_code_soft.shape[1]
        track1_tok = code_embed(track1_code_soft, self.track1_embed)  # (B,n_blocks1,D)
        cum_K1 = self.K0 * self.K1  # bytes covered by one level2-code entry
        code_pos = (jnp.arange(n_blocks1) + 1) * cum_K1 - 1
        query_pos = jnp.arange(target_len)
        mask = code_pos[None, :] <= query_pos[:, None]  # (target_len, n_blocks1), causal
        for cblk in self.cross_blocks:
            x = cblk(x, track1_tok, query_pos, code_pos, mask)
        h = self.ln_f(x)

        logits = h @ self.head
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target_seq[..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target_seq)
        return loss, acc

    def reconstruct(self, ctx_code_soft: jnp.ndarray, track1_code_soft: jnp.ndarray, group: int = None,
                     greedy: bool = True, temperature: float = 1.0, seed: int = 0,
                     cache_track1: bool = True) -> jnp.ndarray:
        """Given ALL level1 codes (ctx_code_soft, track0 own-code) and ALL level2 codes
        (track1_code_soft) already known -- like reconstruct_tree's start_level, this decodes
        bytes given a FIXED, already-complete set of top-level codes, never generating codes one
        at a time -- decode every level0 block's own K0 bytes. `group` (default None ->
        n_blocks0, fully parallel) folds `group` blocks at a time into the batch dim for track0's
        incremental self-attn; group=1 regresses to fully sequential, block-by-block (matching
        StackDecoderTrack0's reconstruct loop) -- track0 is block-diagonal BY CONSTRUCTION so
        `group` is purely a compute/memory tradeoff here, never a correctness one (unlike
        StageSelfAttnDecoder's recon_ncode, where group size changes actual cross-block
        conditioning/cache visibility).

        Track1's cross-attention K/V only depends on track1_code_soft (fixed for the whole call,
        identical across every group and every within-block step t) -- cache_track1=True (default)
        precomputes each cross_block's K/V ONCE up front and reuses it for every step, instead of
        recomputing the wk/wv projections from scratch at every single byte. cache_track1=False
        keeps the original recompute-every-step path (kept as a correctness reference -- verified
        byte-identical to the cached path via a teacher-forced consistency check)."""
        B, n_blocks0 = ctx_code_soft.shape[:2]
        n_blocks1 = track1_code_soft.shape[1]
        D = self.target_embed.shape[-1]
        hd = D // self.n_heads
        K0e = self.K0 + 1
        group = group or n_blocks0
        assert n_blocks0 % group == 0, f"group={group} must divide n_blocks0={n_blocks0}"
        n_groups = n_blocks0 // group

        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)  # (B, n_blocks0, D)
        track1_tok = code_embed(track1_code_soft, self.track1_embed)  # (B, n_blocks1, D)
        cum_K1 = self.K0 * self.K1
        code_pos1 = (jnp.arange(n_blocks1) + 1) * cum_K1 - 1
        rng = jax.random.PRNGKey(seed)
        Bg = B * group

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(self.self_blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, K0e)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return x, jnp.stack(new_ck), jnp.stack(new_cv)

        self_step = jax.jit(self_step)

        def build_track1_kv():
            # K/V built at batch size B (NOT Bg=B*group) -- every block within an image shares the
            # IDENTICAL track1 code, so replicating K/V per-block (the original design) wasted
            # group-times more memory than needed (measured: 9.66GB captured constants at real
            # scale, B=8/n_blocks0=1536/group=1536 -- exactly B*n_heads*n_blocks1*hd*4bytes*2
            # matches). Query gets reshaped to (B,group,...) per step instead, broadcasting
            # against this small (B,H,n_blocks1,hd) K/V.
            kv = []
            for cblk in self.cross_blocks:
                xkv = cblk.norm1(track1_tok)  # (B, n_blocks1, D)
                ca = cblk.cross_attn
                k = (xkv @ ca.wk).reshape(B, n_blocks1, self.n_heads, hd).transpose(0, 2, 1, 3)
                v = (xkv @ ca.wv).reshape(B, n_blocks1, self.n_heads, hd).transpose(0, 2, 1, 3)
                k = rmsnorm(k, ca.k_norm)
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos1, hd, ca.rope_base)
                k = k * cos_k[None, None, :, :] + rotate_half(k) * sin_k[None, None, :, :]
                kv.append((k, v))  # each (B, n_heads, n_blocks1, hd)
            return kv

        def track1_step_cached(x_q, global_pos, kv_cache):
            # x_q: (Bg,D); global_pos: (Bg,) -- reshape to (B,group,...) for the actual attention,
            # flatten back to (Bg,D) only for the residual add (matches the original API exactly).
            x = x_q
            global_pos_r = global_pos.reshape(B, group)
            mask = code_pos1[None, None, :] <= global_pos_r[:, :, None]  # (B, group, n_blocks1)
            for cblk, (k, v) in zip(self.cross_blocks, kv_cache):
                xn = cblk.norm1(x).reshape(B, group, D)
                ca = cblk.cross_attn
                q = (xn @ ca.wq).reshape(B, group, self.n_heads, hd).transpose(0, 2, 1, 3)  # (B,H,group,hd)
                q = rmsnorm(q, ca.q_norm)
                cos_q, sin_q = rope_cos_sin_for_positions(global_pos, hd, ca.rope_base)
                cos_q = cos_q.reshape(B, group, hd)[:, None, :, :]
                sin_q = sin_q.reshape(B, group, hd)[:, None, :, :]
                q = q * cos_q + rotate_half(q) * sin_q
                scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
                logits = jnp.einsum("bhgd,bhkd->bhgk", q, k) * scale  # (B,H,group,n_blocks1)
                logits = jnp.where(mask[:, None, :, :], logits, -1e9)
                attn = jax.nn.softmax(logits, axis=-1)
                y = jnp.einsum("bhgk,bhkd->bhgd", attn, v)  # (B,H,group,hd)
                y = y.transpose(0, 2, 1, 3).reshape(Bg, D)
                x = x + y @ ca.out
                x = x + cblk.mlp(cblk.norm2(x))
            return x

        def track1_step_uncached(x_q, global_pos):
            # Reference path (no cache) -- same B-batched shape discipline as the cached path
            # (avoids the same Bg-replication blowup), recomputes wk/wv from track1_tok every step.
            x = x_q
            global_pos_r = global_pos.reshape(B, group)
            mask = code_pos1[None, None, :] <= global_pos_r[:, :, None]  # (B, group, n_blocks1)
            for cblk in self.cross_blocks:
                xn = cblk.norm1(x).reshape(B, group, D)
                xkv = cblk.norm1(track1_tok)  # (B, n_blocks1, D)
                ca = cblk.cross_attn
                q = (xn @ ca.wq).reshape(B, group, self.n_heads, hd).transpose(0, 2, 1, 3)
                k = (xkv @ ca.wk).reshape(B, n_blocks1, self.n_heads, hd).transpose(0, 2, 1, 3)
                v = (xkv @ ca.wv).reshape(B, n_blocks1, self.n_heads, hd).transpose(0, 2, 1, 3)
                q, k = rmsnorm(q, ca.q_norm), rmsnorm(k, ca.k_norm)
                cos_q, sin_q = rope_cos_sin_for_positions(global_pos, hd, ca.rope_base)
                cos_q = cos_q.reshape(B, group, hd)[:, None, :, :]
                sin_q = sin_q.reshape(B, group, hd)[:, None, :, :]
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos1, hd, ca.rope_base)
                q = q * cos_q + rotate_half(q) * sin_q
                k = k * cos_k[None, None, :, :] + rotate_half(k) * sin_k[None, None, :, :]
                scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
                logits = jnp.einsum("bhgd,bhkd->bhgk", q, k) * scale
                logits = jnp.where(mask[:, None, :, :], logits, -1e9)
                attn = jax.nn.softmax(logits, axis=-1)
                y = jnp.einsum("bhgk,bhkd->bhgd", attn, v)
                y = y.transpose(0, 2, 1, 3).reshape(Bg, D)
                x = x + y @ ca.out
                x = x + cblk.mlp(cblk.norm2(x))
            return x

        track1_kv = jax.jit(build_track1_kv)() if cache_track1 else None
        track1_step_cached = jax.jit(track1_step_cached)
        track1_step_uncached = jax.jit(track1_step_uncached)

        out = jnp.zeros((B, n_blocks0 * self.K0), dtype=jnp.int32)
        b_idx = jnp.arange(Bg) // group  # batch index per row, matches (B,group,D)->(Bg,D) reshape order
        block_idx_local = jnp.tile(jnp.arange(group), B)  # local block index within this group

        for g in tqdm(range(n_groups), desc=f"decode(local_track1,Bg={Bg},group={group},K0={self.K0})",
                      leave=False):
            blk_lo = g * group
            ctx_tok_g = ctx_tok[:, blk_lo:blk_lo + group, :]
            x_input = ctx_tok_g.reshape(Bg, D)
            global_block_idx = block_idx_local + blk_lo

            cache_k = jnp.zeros((len(self.self_blocks), Bg, self.n_kv_heads, K0e, hd))
            cache_v = jnp.zeros_like(cache_k)
            pos = 0
            for t in range(K0e):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                global_pos = global_block_idx * self.K0 + t
                if cache_track1:
                    h2 = self.ln_f(track1_step_cached(h, global_pos, track1_kv))
                else:
                    h2 = self.ln_f(track1_step_uncached(h, global_pos))
                pos += 1
                if t < self.K0:
                    logits = h2 @ self.head
                    if greedy:
                        val = jnp.argmax(logits, axis=-1)
                    else:
                        rng, k_ = jax.random.split(rng)
                        val = jax.random.categorical(k_, logits / temperature, axis=-1)
                    flat_pos = global_block_idx * self.K0 + t
                    out = out.at[b_idx, flat_pos].set(val)
                    x_input = self.target_embed[val]
        return out


class Track0LocalLayer(eqx.Module):
    """One layer of the CORRECTED track0 mechanism -- 1:1 port of qcute_lagcodec_decoder.py's
    block_local_track0_decode (StackDecoderLocal's Variant B), with ONE deliberate departure
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


class StageLocalTrack1Decoder(eqx.Module):
    """CORRECTED, GENERALIZED replacement for StageLocalTrack1DecoderV1 (2026-09-08 rewrite --
    read the reference's decode_level/StackDecoder.__init__/block_local_track0_decode directly
    before writing this, per explicit correction: the V1 version had two real divergences and
    was hardcoded to exactly 2 tracks instead of generalizing like the reference's cond_depth
    loop). Works for ANY n_levels>=2, not just 3.

    Track0 (level1's own-block code): Track0LocalLayer stack -- REAL cross-attention (not
    code-as-token self-attention), block-diagonal self-attn among real bytes only, matching
    block_local_track0_decode exactly except for the deliberate 3rd-norm departure (see
    Track0LocalLayer's docstring).

    Upper tracks (level2..level(n_levels-1)'s codes -- the topmost level's own code is hard-
    excluded, never consumed by anyone, matching the reference's `n_upper = n_levels-2` cap):
    CrossBlock stages (cross-attn + MLP, own dedicated norm already, no change needed), chained
    SEQUENTIALLY in level order (track0 first/"earliest", increasingly coarser levels after --
    matches decode_level's actual `for j in range(i+1, j_max)` loop order exactly: own-level
    code first, coarser codes added later, each stage's output feeding the next).

    SIMPLIFIED relative to the reference (explicit tradeoff, confirmed with user before
    building): UNSHIFTED alignment throughout (h[p] reconstructs target[p], same convention
    every other decoder in this file uses) instead of the reference's shifted h0/h0_shifted dual-
    view handoff + separate per-track auxiliary losses. Upper tracks cross-attend on the SAME
    unshifted h0 track0 produces -- still fully causal (code masks are still code_pos<=query_pos)
    -- just without the reference's extra first-block-only auxiliary loss term.

    kv_lm_mode: reference default is "shared" (reruns the code embedding through the encoder's
    own trained per-level LM before using it as cross-attn K/V) -- only "identity" (raw embedding
    table, no extra projection) is implemented so far; "shared"/"copy" raise NotImplementedError."""
    target_embed: jnp.ndarray
    track0_ctx_embed: jnp.ndarray      # level1's code embed table (track0's own-block code)
    track0_layers: list                # list of Track0LocalLayer
    track0_seed: jnp.ndarray           # trainable seed constant, predicts byte0
    upper_ctx_embeds: list             # one embed table per upper track (levels 2..n_levels-1)
    upper_cross_blocks: list           # list of list-of-CrossBlock, one inner list per upper track
    ln_f: RMSNorm
    head: jnp.ndarray
    K0: int = eqx.field(static=True)          # level0 stride == bytes per track0 block
    strides: tuple = eqx.field(static=True)   # cfg.strides, to compute each upper track's cum_K
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    kv_lm_mode: str = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        n_levels = len(cfg.strides)
        assert n_levels >= 2, "StageLocalTrack1Decoder needs n_levels>=2 (at least track0)"
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
        self.track0_ctx_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        self.track0_seed = jax.random.normal(keys[2], (D,)) * 0.02
        track0_keys = jax.random.split(keys[3], n_layers)
        self.track0_layers = [Track0LocalLayer(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                               for k in track0_keys]
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
        every level's own code (codes_soft[0]=level1's code=track0, codes_soft[1]=level2's
        code=upper track 0, ..., codes_soft[n_levels-2]=level(n_levels-1)'s code=last usable
        upper track; codes_soft[n_levels-1], the topmost level's own output, is never used)."""
        B = target_seq.shape[0]
        D = self.target_embed.shape[-1]
        n_blocks0 = codes_soft[0].shape[1]

        # Track0: block-diagonal, real cross-attention (Track0LocalLayer)
        ctx_tok0 = code_embed(codes_soft[0], self.track0_ctx_embed)  # (B,n_blocks0,D)
        te = self.target_embed[target_seq]  # (B,n_blocks0*K0,D)
        Bn = B * n_blocks0
        x_real = te.reshape(Bn, self.K0, D)
        code_kv = ctx_tok0.reshape(Bn, 1, D)
        for layer in self.track0_layers:
            x_real = layer.forward_real(x_real, code_kv)
        x_seed = jnp.broadcast_to(self.track0_seed, (Bn, 1, D))
        for layer in self.track0_layers:
            x_seed = layer.forward_seed(x_seed, code_kv)
        h_real = self.ln_f(x_real).reshape(B, n_blocks0, self.K0, D)
        h_seed = self.ln_f(x_seed).reshape(B, n_blocks0, 1, D)
        h0 = jnp.concatenate([h_seed, h_real[:, :, :self.K0 - 1, :]], axis=2)  # (B,n_blocks0,K0,D), unshifted
        target_len = n_blocks0 * self.K0
        x = h0.reshape(B, target_len, D)

        # Upper tracks: level2..level(n_levels-1)'s codes, sequentially chained, coarser each time
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
        `K+1`-th extra write needed here, unlike StackDecoderTrack0's UNBOUNDED single cache,
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
    encoder: HierEncoder
    stages: list   # stages[i] = a StageCrossAttnDecoder/StageSelfAttnDecoder reducing level i's codes -> level (i-1)'s sequence
    # decoder_type="stack_track0": stages has exactly ONE element, a StackDecoderTrack0 for level0's
    # own bytes conditioned on codes[0] (track0 only, cfg.cond_depth=1 in the reference's terms) --
    # NOT a chain of separate per-level decoders (that cascade was this file's own invention, not
    # part of the reference: the reference generates codes above level0 via each level's OWN NTP
    # head, already computed by HierEncoder, never a dedicated decoder).
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        keys = jax.random.split(key, 1 + n)
        self.encoder = HierEncoder(keys[0], cfg)
        if cfg.decoder_type == "stack_track0":
            self.stages = [StackDecoderTrack0(keys[1], cfg, level=0)]
        elif cfg.decoder_type == "self_attn_local_track1":
            self.stages = [StageLocalTrack1Decoder(keys[1], cfg)]
        elif cfg.decoder_type == "self_attn_local_track1_v1":
            self.stages = [StageLocalTrack1DecoderV1(keys[1], cfg)]
        elif cfg.decoder_type == "self_attn_lag":
            self.stages = [StageLagDecoder(keys[1], cfg, level=0)]
        elif cfg.decoder_type == "cross_attn":
            self.stages = [StageCrossAttnDecoder(keys[1 + i], cfg, level=i) for i in range(n)]
        else:
            local = cfg.decoder_type == "self_attn_local"
            self.stages = [StageSelfAttnDecoder(keys[1 + i], cfg, level=i, local=local) for i in range(n)]

    def __call__(self, flat_bytes: jnp.ndarray) -> tuple:
        enc = self.encoder(flat_bytes)
        n = len(self.stages)
        if self.cfg.decoder_type == "stack_track0":
            # single stage, no chain to build -- context is always the encoder's real code
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"][0])
            decode_loss_total = byte_loss
        elif self.cfg.decoder_type == "self_attn_local_track1":
            # single stage, generalized N-level: track0 (level1's code) + upper tracks (level2..
            # level(n-1)'s codes, sequentially chained) -- no dedicated decoder above level0.
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"])
            decode_loss_total = byte_loss
        elif self.cfg.decoder_type == "self_attn_local_track1_v1":
            # single stage, conditions directly on level1's code (track0) AND level2's code
            # (track1) -- levels 1/2 have no dedicated decoder, only their own NTP heads.
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"][0], enc["codes_soft"][1])
            decode_loss_total = byte_loss
        elif self.cfg.decoder_type == "self_attn_lag":
            # single stage, conditions on level1's code (own code) with `cfg.lag` extra
            # causally-later codes grouped in -- levels above have no dedicated decoder.
            byte_loss, byte_acc = self.stages[0].forward(flat_bytes, enc["codes_soft"][0], self.cfg.lag)
            decode_loss_total = byte_loss
        else:
            # Chain TOP-DOWN (matches generation's actual top-down chain, eliminates the earlier
            # train/inference mismatch where every stage trained independently against the
            # encoder's real code regardless of level): only the TOPMOST stage conditions on the
            # encoder's real top code (nothing above it to predict from -- same as generation's
            # entry point); every stage below conditions on the stage ABOVE's own STE soft
            # PREDICTION instead, so gradient flows end-to-end from the bottom stage's loss up
            # through every stage into the encoder (symmetric with the encoder's own STE chain).
            stage_losses, stage_accs = [], []
            context = enc["codes_soft"][n - 1]
            for i in range(n - 1, -1, -1):
                target_seq = flat_bytes if i == 0 else enc["codes"][i - 1]
                loss_i, acc_i, pred_code_soft = self.stages[i].forward(target_seq, context)
                stage_losses.append(loss_i)
                stage_accs.append(acc_i)
                context = pred_code_soft
            byte_loss, byte_acc = stage_losses[-1], stage_accs[-1]  # loop ran top->0, level0 is LAST appended
            decode_loss_total = jnp.mean(jnp.stack(stage_losses))

        ntp_losses = jnp.stack([r["ntp_loss"] for r in enc["results"]])
        ntp_accs = jnp.stack([r["ntp_acc"] for r in enc["results"]])
        utils = jnp.stack([r["util"] for r in enc["results"]])
        ntp_loss_total = jnp.mean(ntp_losses)
        loss = decode_loss_total + self.cfg.ntp_weight * ntp_loss_total
        bpb = byte_loss / jnp.log(2.0)
        return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(ntp_accs), jnp.mean(utils))

    def reconstruct_tree(self, flat_bytes: jnp.ndarray, start_level: int = 0, recon_ncode: tuple = None,
                          greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """Encode once, then chain StageDecoders from `start_level` down to 0, grouping each
        level's blocks into `recon_ncode[level]`-sized local AR chains (see StageDecoder
        docstring). start_level==0, recon_ncode all-1 reproduces the original single-stage,
        fully-block-parallel byte reconstruction exactly."""
        recon_ncode = recon_ncode or self.cfg.recon_ncode
        enc = self.encoder(flat_bytes)
        ctx = enc["codes"][start_level]  # (B, n_units(start_level), pq_chunks) -- real, from encoding
        for level in range(start_level, -1, -1):
            stage = self.stages[level]
            B, n_blk, pq = ctx.shape
            if getattr(stage, "local", False):
                # local=True: no recon_ncode grouping construct at all -- reconstruct_block_local
                # already decodes every block in parallel (n_blocks folded into batch), matching
                # forward()'s block-diagonal computation exactly, and already returns (B,...) shaped.
                ctx = stage.reconstruct_block_local(ctx, greedy=greedy, temperature=temperature, seed=seed + level)
            else:
                g = recon_ncode[level]
                assert n_blk % g == 0, f"recon_ncode[{level}]={g} must divide n_blocks={n_blk}"
                n_groups = n_blk // g
                ctx_grouped = ctx.reshape(B * n_groups, g, pq)
                out = stage.reconstruct_group(ctx_grouped, greedy=greedy, temperature=temperature, seed=seed + level)
                target_len = g * stage.K
                if stage.is_byte_target:
                    ctx = out.reshape(B, n_groups * target_len)
                else:
                    ctx = out.reshape(B, n_groups * target_len, pq)
        return ctx

    def reconstruct_tree_sequential(self, flat_bytes: jnp.ndarray, start_level: int = 0,
                                     greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """reconstruct_tree's diagnostic counterpart: chains StageSelfAttnDecoder.reconstruct_sequential
        (no recon_ncode grouping, one continuous causal chain per level, no group-boundary resets)
        from `start_level` down to 0 instead of the grouped/isolated-island reconstruct_group.
        self_attn decoder_type only for now (StageCrossAttnDecoder has no reconstruct_sequential --
        this diagnostic hasn't been extended there). Much slower than reconstruct_tree (no batch-
        parallelism across groups -- n_units(level)*(K+1) sequential steps per level) -- for
        confirming whether recon_ncode grouping's cold-start islands are what's suppressing
        reconstruction quality, not for routine use."""
        if self.cfg.decoder_type != "self_attn":
            raise NotImplementedError("reconstruct_tree_sequential only supports decoder_type='self_attn' so far")
        enc = self.encoder(flat_bytes)
        ctx = enc["codes"][start_level]
        for level in range(start_level, -1, -1):
            stage = self.stages[level]
            B, n_blk, pq = ctx.shape
            out = stage.reconstruct_sequential(ctx, greedy=greedy, temperature=temperature, seed=seed + level)
            if stage.is_byte_target:
                ctx = out.reshape(B, n_blk * stage.K)
            else:
                ctx = out.reshape(B, n_blk * stage.K, pq)
        return ctx  # (B, SEQ_LEN) bytes

    def reconstruct(self, flat_bytes: jnp.ndarray, greedy: bool = True, temperature: float = 1.0,
                     seed: int = 0) -> jnp.ndarray:
        """DEBUG SHORTCUT, not a real reconstruction test: hands level0's stage its own REAL code
        directly (start_level=0), skipping levels 1+ entirely. Genuine end-to-end reconstruction
        (from the compressed TOP-level code, the actual point of the hierarchy) needs
        reconstruct_tree(start_level=len(cfg.strides)-1) -- main()'s run_reconstruct defaults to
        that; this method is kept only for quick ad hoc checks of level0 in isolation."""
        return self.reconstruct_tree(flat_bytes, start_level=0, greedy=greedy, temperature=temperature, seed=seed)


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

def make_train_step(optimizer):
    def loss_fn(model, flat_bytes):
        return model(flat_bytes)

    def train_step(model, opt_state, flat_bytes):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, flat_bytes)
        grads = jax.lax.pmean(grads, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step():
    def eval_step(model, flat_bytes):
        _, aux = model(flat_bytes)
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
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight", "recon_ncode",
                  "decoder_type", "lag", "kv_lm_mode", "dec_d_model", "dec_n_layers", "dec_n_heads", "dec_n_kv_heads")


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
    p.add_argument("--sequential_decode", type=lambda x: x.lower() != "false", default=False,
                    help="use reconstruct_tree_sequential (dedicated one-continuous-chain method) for "
                         "qualitative reconstruction instead of reconstruct_tree's recon_ncode grouping")
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
    p.add_argument("--recon_ncode", type=_tuple_arg, default=Config.recon_ncode)
    p.add_argument("--decoder_type", type=str, default=Config.decoder_type,
                    choices=["cross_attn", "self_attn", "self_attn_local", "stack_track0",
                             "self_attn_local_track1", "self_attn_local_track1_v1", "self_attn_lag"])
    p.add_argument("--lag", type=int, default=Config.lag)
    p.add_argument("--kv_lm_mode", type=str, default=Config.kv_lm_mode, choices=["identity", "shared", "copy"])
    p.add_argument("--dec_d_model", type=_tuple_arg, default=Config.dec_d_model)
    p.add_argument("--dec_n_layers", type=_tuple_arg, default=Config.dec_n_layers)
    p.add_argument("--dec_n_heads", type=_tuple_arg, default=Config.dec_n_heads)
    p.add_argument("--dec_n_kv_heads", type=_tuple_arg, default=Config.dec_n_kv_heads)
    p.add_argument("--start_level", type=int, default=None,
                    help="reconstruct by chaining StageDecoders down from this level. None (default) "
                         "resolves to the TOP level (len(strides)-1) -- genuine end-to-end reconstruction "
                         "from the compressed top-level code, the actual point of the hierarchy. Passing "
                         "0 explicitly bypasses levels 1+ entirely and hands level0 its own real code "
                         "directly -- a debug shortcut, not a real reconstruction test.")

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
    if args.start_level is None:
        args.start_level = len(cfg.strides) - 1  # top level -- genuine end-to-end reconstruction by default

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

    train_step = make_train_step(optimizer)
    eval_step = make_eval_step()

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
        for i, (flat, y) in enumerate(val_iter):
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

    def run_reconstruct(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        flat = jnp.array(recon_prompt.reshape(args.qual_gen_n, SEQ_LEN))

        if cfg.decoder_type == "stack_track0":
            # reconstruct_kv_cache (verified byte-identical to the full-recompute reference at
            # toy scale) -- reconstruct() (full recompute, O(n_blocks^2)) is diagnostic-grade
            # only and pathologically slow at real scale (measured: ~6h/eval at n_blocks=1024).
            _, aux_tf = single_model(flat)
            tf_acc = float(aux_tf[1])
            enc_tf = single_model.encoder(flat)
            recon = single_model.stages[0].reconstruct_kv_cache(enc_tf["codes"][0], greedy=args.qual_gen_greedy,
                                                                  temperature=args.qual_gen_temperature, seed=epoch)
            gen_acc = float(jnp.mean(recon == flat))
            recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            gt_img = recon_prompt.astype(np.uint8)
            mse = float(np.mean((recon_img.astype(np.float32) - gt_img.astype(np.float32)) ** 2))
            out_path = run_dir / f"samples_epoch{epoch}_reconstruct.png"
            save_compare_grid(recon_img, gt_img, out_path)
            logger(f"[stack_track0] saved reconstruction (recon/gt) for epoch {epoch}, "
                   f"MSE={mse:.2f}, gen_byte_acc={gen_acc:.4f}, teacher_forced_acc={tf_acc:.4f} "
                   f"(gen_consistency_gap={tf_acc - gen_acc:.4f})",
                   recon_mse=mse, gen_byte_acc=gen_acc, teacher_forced_acc=tf_acc)
            return

        if cfg.decoder_type == "self_attn_local_track1":
            # Corrected/generalized class -- generation (full-recompute + KV-cache) not built yet
            # (forward/training-only so far, same bring-up order as stack_track0/self_attn_lag
            # originally) -- log teacher-forced accuracy only.
            _, aux_tf = single_model(flat)
            logger(f"[self_attn_local_track1] epoch {epoch}: teacher_forced_acc={float(aux_tf[1]):.4f} "
                   f"(generation not yet implemented for the corrected/generalized class)",
                   teacher_forced_acc=float(aux_tf[1]))
            return

        if cfg.decoder_type == "self_attn_local_track1_v1":
            # Start decode from ALL top-level codes already known (like reconstruct_tree's
            # start_level -- real encoder codes, not generated) -- both track0 (level1) and
            # track1 (level2). group=None (fully parallel) is exact by construction, see
            # StageLocalTrack1DecoderV1.reconstruct's docstring.
            _, aux_tf = single_model(flat)
            tf_acc = float(aux_tf[1])
            enc_tf = single_model.encoder(flat)
            recon = single_model.stages[0].reconstruct(enc_tf["codes_soft"][0], enc_tf["codes_soft"][1],
                                                         group=None, greedy=args.qual_gen_greedy,
                                                         temperature=args.qual_gen_temperature, seed=epoch)
            gen_acc = float(jnp.mean(recon == flat))
            recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            gt_img = recon_prompt.astype(np.uint8)
            mse = float(np.mean((recon_img.astype(np.float32) - gt_img.astype(np.float32)) ** 2))
            out_path = run_dir / f"samples_epoch{epoch}_reconstruct.png"
            save_compare_grid(recon_img, gt_img, out_path)
            logger(f"[self_attn_local_track1_v1] saved reconstruction (recon/gt) for epoch {epoch}, "
                   f"MSE={mse:.2f}, gen_byte_acc={gen_acc:.4f}, teacher_forced_acc={tf_acc:.4f} "
                   f"(gen_consistency_gap={tf_acc - gen_acc:.4f})",
                   recon_mse=mse, gen_byte_acc=gen_acc, teacher_forced_acc=tf_acc)
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

        # teacher-forced accuracy AND per-position predictions on this SAME prompt subset --
        # model(flat)'s aux[1] is exactly stage0.forward's byte_acc (real previous bytes fed in
        # at every step, matches training); predict_teacher_forced gives the byte-for-byte
        # predictions needed for the gen-consistency divergence print below.
        _, aux_tf = single_model(flat)
        tf_acc = float(aux_tf[1])
        enc_tf = single_model.encoder(flat)
        tf_pred = single_model.stages[0].predict_teacher_forced(flat, enc_tf["codes_soft"][0]) \
            if cfg.decoder_type in ("self_attn", "self_attn_local") else None

        if args.sequential_decode:
            recon = single_model.reconstruct_tree_sequential(flat, start_level=args.start_level,
                                                               greedy=args.qual_gen_greedy,
                                                               temperature=args.qual_gen_temperature, seed=epoch)
        else:
            recon = single_model.reconstruct_tree(flat, start_level=args.start_level, recon_ncode=cfg.recon_ncode,
                                                   greedy=args.qual_gen_greedy, temperature=args.qual_gen_temperature,
                                                   seed=epoch)
        gen_acc = float(jnp.mean(recon == flat))  # free-running greedy accuracy, same prompt subset

        if tf_pred is not None:
            fr_match_full = np.asarray(recon == flat)
            for i in range(args.qual_gen_n):
                wrong_idx = np.where(~fr_match_full[i])[0]
                first_div = int(wrong_idx[0]) if len(wrong_idx) else -1
                N = 60
                logger(f"gen_consistency img{i} ep{epoch}: first_divergence_byte={first_div} "
                       f"(of {SEQ_LEN})\n"
                       f"  ground_truth  [:{N}]: {np.asarray(flat[i, :N]).tolist()}\n"
                       f"  teacher_forced[:{N}]: {np.asarray(tf_pred[i, :N]).tolist()}\n"
                       f"  free_rollout  [:{N}]: {np.asarray(recon[i, :N]).tolist()}",
                       gen_consistency_img=i, first_divergence_byte=first_div)
        recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
        gt_img = recon_prompt.astype(np.uint8)
        mse = float(np.mean((recon_img.astype(np.float32) - gt_img.astype(np.float32)) ** 2))
        out_path = run_dir / f"samples_epoch{epoch}_reconstruct.png"
        save_compare_grid(recon_img, gt_img, out_path)
        logger(f"saved reconstruction (recon/gt) for epoch {epoch}, MSE={mse:.2f}, "
               f"gen_byte_acc={gen_acc:.4f}, teacher_forced_acc={tf_acc:.4f} "
               f"(gen_consistency_gap={tf_acc - gen_acc:.4f})",
               recon_mse=mse, gen_byte_acc=gen_acc, teacher_forced_acc=tf_acc)

    def run_checkpoint(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        single_opt_state = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_opt_state)
        ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch}"
        save_checkpoint(ckpt_dir, single_model, single_opt_state, step, epoch)
        logger(f"saved checkpoint at epoch {epoch} -> {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(train_iter, desc=f"epoch {epoch}/{args.epochs}")
        for flat, y in pbar:
            p_model, p_opt_state, (bpb, acc, ntp_bpb, ntp_acc, util) = train_step(p_model, p_opt_state, flat)
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} byte_bpb={float(bpb[0]):.4f} byte_acc={float(acc[0]):.4f} "
                       f"ntp_bpb={float(ntp_bpb[0]):.4f} ntp_acc={float(ntp_acc[0]):.4f} util={float(util[0]):.3f}",
                       epoch=epoch, step=step, train_bpb=float(bpb[0]), train_acc=float(acc[0]),
                       train_ntp_bpb=float(ntp_bpb[0]), train_ntp_acc=float(ntp_acc[0]), train_util=float(util[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            run_eval()
            run_reconstruct(epoch)
            run_checkpoint(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
