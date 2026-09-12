"""Self-contained curriculum-training fork of run_lagcodec.py -- NOT imported from
run_lagcodec.py, no StackDecoder/StageLocalDecoder/StageLagDecoder/Level1Layer/cross-attn
leftovers. Single design: per-level EncDecLevel, optionally weight-shared between its encoder
role (own unconditioned NTP + code_head, like HierEncoder.EncoderLevel) and decoder role
(ctx-conditioned, BOS-seeded generation of the level below, like StageLagDecoder) -- literally
the SAME transformer block weights serve both roles when weight_sharing=True ("encoderlevel_i =
decoderlevel_i"), with ONLY the topmost level (its own code never consumed by anyone) staying
encoder-only, unshared.

Training is still plain teacher-forcing throughout (forward() always sees the REAL target).
CURRICULUM (chat 2026-09-11, NO lead-in phase): n_phases = n_levels-1, phase p (1-indexed)
trains ONLY levels[p-1] -- its encoder (own NTP + code_head) and decoder (conditioned on its own
just-produced code) trained JOINTLY in one phase, directly like a plain autoencoder. Levels below
p-1 still run forward (frozen, no gradient) to produce the code that feeds levels[p-1]'s encoder
input. Once phase p ends, levels[p-1] is frozen and phase p+1 moves up to train levels[p] the
same way -- no level ever gets a separate encoder-only warm-up phase.

pmap-parallel across all local devices.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curriculum_shallow_lag0_sharing_on_thin.py
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

from image_lagcodec.eqx_common import Block, RMSNorm, sinkgd, warmup_const_schedule

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
SEQ_LEN = 32 * 32 * 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 32
    d_model: tuple = (256, 256, 256, 256)
    n_layers: tuple = (2, 2, 2, 2)
    n_heads: tuple = (4, 4, 4, 4)
    n_kv_heads: tuple = (None, None, None, None)
    strides: tuple = (3, 16, 16, -1)
    code_vocab: tuple = (4, 4, 4, 4)   # per level, level0..levelN-1 (ported from run_lagcodec.py,
    pq_chunks: tuple = (5, 5, 5, 5)    # chat 2026-09-11) -- effective per-level vocab is
    # code_vocab[i]**pq_chunks[i]; level[-1] don't-care when its level is never built (see
    # EncDecLevel -- topmost level here always has has_decoder=False and is always built, unlike
    # run_lagcodec.py's train_last_encoder, so no level is actually don't-care in this file today).
    mlp_mult: int = 4
    rope_base: float = 10000.0
    ntp_weight: float = 1.0
    lag: int = 0                    # decode-mode lag, same value used by every level's decoder
    weight_sharing: bool = True     # encoderlevel_i = decoderlevel_i for every level with a decoder
    precision: str = "bf16"
    curriculum_mode: str = "freeze"   # "freeze" (default): phase p trains ONLY levels[p-1], every
    # earlier level stays frozen forever once its own phase ends (current/original behavior).
    # "no_freeze" (chat 2026-09-11): phase p trains ALL of levels[0..p-1] jointly -- levels never
    # freeze, each new phase just adds one more level to the still-training set. Note: opt_state
    # is still reinitialized fresh every phase either way (existing simplification, momentum
    # resets at phase boundaries even for levels that stay trainable across phases in this mode).
    quantize_mode: str = "argmax"     # "argmax" (default): quantize_hard, deterministic. "gumbel"
    # (chat 2026-09-11): quantize_gumbel -- stochastic, same raw input can land on different
    # similar-probability codes across calls, giving decode() a more diverse teacher-forced ctx/
    # target distribution during TRAINING (regularization against brittle exact-code overfitting).
    gumbel_temperature: float = 1.0
    gumbel_at_inference: bool = False   # "gumbel" quantize_mode is TRAINING-only by default (see
    # phase_forward's rng threading) -- run_gen_eval never samples gumbel noise unless this is
    # explicitly set True (still off by default even then unless quantize_mode="gumbel" too).

    def __post_init__(self):
        n = len(self.strides)
        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert self.strides[-1] == -1, "top level's stride is unused -- use -1 as the don't-care convention"
        assert all(s >= 1 for s in self.strides[:-1])
        assert SEQ_LEN % math.prod(self.strides[:-1]) == 0
        assert self.precision in ("bf16", "fp32")
        assert self.curriculum_mode in ("freeze", "no_freeze")
        assert self.quantize_mode in ("argmax", "gumbel")
        resolved_kv = []
        for i in range(n):
            kv = self.n_kv_heads[i] if self.n_kv_heads[i] is not None else max(1, self.n_heads[i] // 4)
            assert self.n_heads[i] % kv == 0
            assert self.d_model[i] % self.n_heads[i] == 0
            resolved_kv.append(kv)
        self.n_kv_heads = tuple(resolved_kv)


# ---------------------------------------------------------------------------
# CIFAR-10 data (identical to run_lagcodec.py)
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
            img = self.images[sel].astype(np.int32)
            flat = img.reshape(self.total, SEQ_LEN)
            yield flat.reshape(self.n_devices, self.batch_size, SEQ_LEN)


# ---------------------------------------------------------------------------
# Quantization (identical convention to run_lagcodec.py)
# ---------------------------------------------------------------------------

def quantize_hard(logits: jnp.ndarray) -> tuple:
    soft = jax.nn.softmax(logits, axis=-1)
    idx = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=soft.dtype)
    code_soft = soft + jax.lax.stop_gradient(hard - soft)
    return code_soft, idx


def quantize_gumbel(logits: jnp.ndarray, rng, temperature: float = 1.0) -> tuple:
    """Gumbel-softmax relaxed quantization (chat 2026-09-11): adds Gumbel(0,1) noise to logits
    before the softmax/argmax, so the SAME raw input can land on different (similar-probability)
    codes across calls -- unlike quantize_hard's fixed deterministic argmax. Intent: give
    teacher-forced decode() a more diverse ctx/target distribution during TRAINING (regularizes
    against the decoder overfitting to one exact code per input), reducing exactly the kind of
    brittle exposure-bias gap seen in the freeze-mode cascade collapse. Still straight-through:
    forward value is the hard one-hot of the noisy argmax, gradient flows through the soft
    (noisy) softmax. temperature=1.0 default -- higher softens/smooths the noisy softmax (more
    diffuse gradient, code_soft less peaked), lower sharpens it (closer to quantize_hard as T->0).
    NOT used at inference/eval time (run_gen_eval never passes an rng into encode() -- see
    EncDecLevel.encode's rng=None fallback to quantize_hard)."""
    u = jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
    gumbel_noise = -jnp.log(-jnp.log(u))
    noisy_logits = (logits + gumbel_noise) / temperature
    soft = jax.nn.softmax(noisy_logits, axis=-1)
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
    if jnp.issubdtype(code.dtype, jnp.integer):
        return table[code].sum(-2)
    return (code @ table).sum(-2)


def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


# ---------------------------------------------------------------------------
# EncDecLevel -- one module per level, optionally shared between encoder/decoder roles
# ---------------------------------------------------------------------------

class EncDecLevel(eqx.Module):
    blocks: list
    ln_f: RMSNorm
    own_input_embed: jnp.ndarray     # embeds THIS level's own input (bytes if level==0, else
                                       # code_vocab-sized) -- doubles as decode-mode's target
                                       # embedding when weight_sharing=True (same domain).
    ntp_head: jnp.ndarray             # own-input NTP logits; doubles as decode-mode's prediction
                                       # head when weight_sharing=True (same domain/target).
    code_head: jnp.ndarray            # pooled hidden -> this level's own output code
    bos_embed: jnp.ndarray            # decode-mode only, never shared with the encoder role
    ctx_embed: jnp.ndarray            # decode-mode only, embeds ctx=this level's own code
    dec_blocks: list                  # weight_sharing=False only (else None, decode() uses blocks)
    dec_ln_f: RMSNorm                 # weight_sharing=False only (else None)
    dec_target_embed: jnp.ndarray     # weight_sharing=False only (else None)
    dec_head: jnp.ndarray             # weight_sharing=False only (else None)
    has_decoder: bool = eqx.field(static=True)
    weight_sharing: bool = eqx.field(static=True)
    is_byte_level: bool = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)       # THIS level's own output code (code_head/ctx_embed)
    code_vocab: int = eqx.field(static=True)
    in_pq_chunks: int = eqx.field(static=True)    # THIS level's own INPUT stream alphabet, i.e.
    in_code_vocab: int = eqx.field(static=True)   # level(i-1)'s code (or bytes for level0) --
    # DISTINCT from pq_chunks/code_vocab above once per-level values diverge (chat 2026-09-11,
    # ported from run_lagcodec.py's EncoderLevel.ntp_pq_chunks fix): own_input_embed/ntp_head
    # (and decode()'s target embed/head, same domain) live in this alphabet, not this level's own
    # output alphabet -- coincidentally identical only when code_vocab/pq_chunks were uniform.
    K: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    quantize_mode: str = eqx.field(static=True)
    gumbel_temperature: float = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, has_decoder: bool, weight_sharing: bool):
        D = cfg.d_model[level]
        self.K = cfg.strides[level] if cfg.strides[level] != -1 else 1
        self.n_heads, self.n_kv_heads = cfg.n_heads[level], cfg.n_kv_heads[level]
        self.quantize_mode = cfg.quantize_mode
        self.gumbel_temperature = cfg.gumbel_temperature
        self.is_byte_level = (level == 0)
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[level], cfg.code_vocab[level]
        self.in_pq_chunks = None if self.is_byte_level else cfg.pq_chunks[level - 1]
        self.in_code_vocab = None if self.is_byte_level else cfg.code_vocab[level - 1]
        self.has_decoder = has_decoder
        self.weight_sharing = weight_sharing
        own_vocab = 256 if self.is_byte_level else self.in_code_vocab
        ntp_out = 256 if self.is_byte_level else self.in_pq_chunks * self.in_code_vocab
        keys = jax.random.split(key, 9)

        self.own_input_embed = jax.random.normal(keys[0], (own_vocab, D)) * 0.02
        n_layers = cfg.n_layers[level]
        block_keys = jax.random.split(keys[1], n_layers)
        self.blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                       for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.code_head = jax.random.normal(keys[2], (D, self.pq_chunks * self.code_vocab)) * 0.02
        self.ntp_head = jax.random.normal(keys[3], (D, ntp_out)) * 0.02
        self.bos_embed = jax.random.normal(keys[4], (D,)) * 0.02
        self.ctx_embed = jax.random.normal(keys[5], (self.code_vocab, D)) * 0.02

        if has_decoder and not weight_sharing:
            dec_block_keys = jax.random.split(keys[6], n_layers)
            self.dec_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult, cfg.rope_base)
                               for k in dec_block_keys]
            self.dec_ln_f = RMSNorm(D)
            self.dec_target_embed = jax.random.normal(keys[7], (own_vocab, D)) * 0.02
            self.dec_head = jax.random.normal(keys[8], (D, ntp_out)) * 0.02
        else:
            self.dec_blocks, self.dec_ln_f, self.dec_target_embed, self.dec_head = None, None, None, None

    # --- encoder role (mirrors HierEncoder.EncoderLevel.forward) ---

    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, rng=None) -> dict:
        """rng=None (default, always the case at inference/eval -- run_gen_eval never passes
        one): quantize_hard, deterministic argmax. rng given (training only) AND
        self.quantize_mode=="gumbel": quantize_gumbel instead -- stochastic, diverse codes for
        the same input across calls (chat 2026-09-11)."""
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        M, L, D = h.shape
        n_blocks = L // self.K
        h_blocks = h[:, :n_blocks * self.K, :].reshape(M, n_blocks, self.K, D)
        pooled = h_blocks[:, :, self.K - 1, :]
        logits = reshape_pq(pooled @ self.code_head, self.pq_chunks, self.code_vocab)
        if rng is not None and self.quantize_mode == "gumbel":
            code_soft, code_idx = quantize_gumbel(logits, rng, self.gumbel_temperature)
        else:
            code_soft, code_idx = quantize_hard(logits)

        if self.is_byte_level:
            ntp_logits = h[:, :-1, :] @ self.ntp_head
        else:
            ntp_logits = reshape_pq(h[:, :-1, :] @ self.ntp_head, self.in_pq_chunks, self.in_code_vocab)
        tgt = target_idx[:, 1:]
        logp = jax.nn.log_softmax(ntp_logits, axis=-1)
        ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
        ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
        util = codebook_utilization(code_idx, self.code_vocab)
        return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util)

    # --- decoder role (mirrors StageLagDecoder.forward/reconstruct_kv_cache) ---

    def _dec_blocks(self):
        return self.blocks if self.weight_sharing else self.dec_blocks

    def _dec_ln_f(self):
        return self.ln_f if self.weight_sharing else self.dec_ln_f

    def _dec_embed_target(self, idx: jnp.ndarray) -> jnp.ndarray:
        table = self.own_input_embed if self.weight_sharing else self.dec_target_embed
        return table[idx] if self.is_byte_level else code_embed(idx, table)

    def _dec_logits(self, h: jnp.ndarray) -> jnp.ndarray:
        head = self.ntp_head if self.weight_sharing else self.dec_head
        logits = h @ head
        return logits if self.is_byte_level else reshape_pq(logits, self.in_pq_chunks, self.in_code_vocab)

    def _dec_loss_acc(self, logits: jnp.ndarray, target: jnp.ndarray) -> tuple:
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))
        acc = jnp.mean(jnp.argmax(logits, -1) == target)
        return loss, acc

    def _sample(self, logits: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
        if greedy:
            return jnp.argmax(logits, axis=-1), rng
        rng, k_ = jax.random.split(rng)
        return jax.random.categorical(k_, logits / temperature, axis=-1), rng

    def decode(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, lag: int) -> tuple:
        """target_seq: (B,n_blocks*K[,pq_chunks]) real target (bytes if level==0, else codes
        below). ctx_code_soft: (B,n_blocks,pq) this level's own code (STE soft). Only lag>=0
        supported (this fork's configs always use lag=0)."""
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B = target_seq.shape[0]
        D = self.bos_embed.shape[-1]
        te = self._dec_embed_target(target_seq)
        n_blocks = ctx_code_soft.shape[1]
        G = lag + 1
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed(ctx_code_soft, self.ctx_embed)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
            te = jnp.pad(te, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        ctx_g = ctx_tok.reshape(B, n_groups, G, D)
        te_g = te.reshape(B, n_groups, G * self.K, D)
        bos_g = jnp.broadcast_to(self.bos_embed, (B, n_groups, 1, D))
        per_group_len = G + 1 + G * self.K
        xe = jnp.concatenate([ctx_g, bos_g, te_g], axis=2).reshape(B, n_groups * per_group_len, D)
        for blk in blocks:
            xe = blk(xe)
        h = ln_f(xe)
        pred_pos = (jnp.arange(n_groups)[:, None] * per_group_len + G
                    + jnp.arange(G * self.K)[None, :]).reshape(-1)
        h_t = h[:, pred_pos, :]
        logits = self._dec_logits(h_t[:, :n_blocks * self.K, :])
        return self._dec_loss_acc(logits, target_seq)

    def decode_generate(self, ctx_idx: jnp.ndarray, lag: int, greedy: bool = True,
                         temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """Incremental KV-cache generation, chunk-prefilled ctx, SCANNED across groups (chat
        2026-09-11, ported from run_lagcodec.py's kv_cache_init/kv_cache_step_chunk design --
        the earlier bare Python for-loop over n_groups paid per-group host dispatch overhead,
        the real bottleneck at real n_groups (up to 1024); one jax.lax.scan over the group axis
        compiles once and runs entirely on-device, no per-group Python round-trip. The G*K-1
        sequential self_step calls WITHIN one group stay a plain (small, static-length) Python
        loop inside the scan body -- fine, unrolled once at trace time, not a per-call cost)."""
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = () if self.is_byte_level else (self.in_pq_chunks,)
        G = lag + 1
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        ctx_tok_g = jnp.swapaxes(ctx_tok.reshape(B, n_groups, G, D), 0, 1)  # (n_groups,B,G,D), scan axis first

        def self_step(x_new, ck, cv, pos):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, L_total)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        def self_chunk_step(x_chunk, ck, cv, pos_start):
            new_ck, new_cv = [], []
            x = x_chunk
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = blk.chunk_step(x, ck[i], cv[i], pos_start, L_total)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        def group_step(carry, group_codes):
            cache_k, cache_v, pos, rng = carry
            bos_in = jnp.broadcast_to(self.bos_embed, (B, 1, D))
            chunk = jnp.concatenate([group_codes, bos_in], axis=1)
            h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, pos)
            pos = pos + (G + 1)
            h = h_chunk[:, -1, :]
            logits = self._dec_logits(h)
            val, rng = self._sample(logits, rng, greedy, temperature)
            vals = [val]
            x_input = self._dec_embed_target(val)

            for _ in range(G * self.K - 1):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos = pos + 1
                logits = self._dec_logits(h)
                val, rng = self._sample(logits, rng, greedy, temperature)
                vals.append(val)
                x_input = self._dec_embed_target(val)

            _, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
            pos = pos + 1
            return (cache_k, cache_v, pos, rng), jnp.stack(vals, axis=1)  # vals: (B,G*K[,pq])

        cache_k0 = jnp.zeros((len(blocks), B, self.n_kv_heads, L_total, hd))
        cache_v0 = jnp.zeros_like(cache_k0)
        init_carry = (cache_k0, cache_v0, jnp.array(0), jax.random.PRNGKey(seed))

        @jax.jit
        def run_scan(carry, xs):
            return jax.lax.scan(group_step, carry, xs)

        _, vals_all = run_scan(init_carry, ctx_tok_g)   # (n_groups,B,G*K[,pq])
        vals_all = jnp.moveaxis(vals_all, 0, 1)          # (B,n_groups,G*K[,pq])
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        return out[:, :n_blocks * self.K]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HierEncDec(eqx.Module):
    levels: list   # length N; levels[N-1] has_decoder=False (top, own code never consumed)
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        keys = jax.random.split(key, n)
        self.levels = [EncDecLevel(keys[i], cfg, level=i, has_decoder=(i < n - 1),
                                    weight_sharing=cfg.weight_sharing) for i in range(n)]


def phase_forward(model: HierEncDec, flat_bytes: jnp.ndarray, phase: int, rng=None) -> tuple:
    """phase in 1..N-1, active_level = phase-1. Runs encode() for levels 0..active_level (levels
    below active_level are frozen but still run forward, unconditioned NTP + producing the code
    that feeds the next level up -- range(phase), NOT range(phase+1): no lead-in level above
    active_level is touched), plus decode() for levels[active_level], conditioned on its OWN
    just-computed code -- encoder and decoder of the SAME level trained jointly this phase,
    directly like an autoencoder. Which leaves actually receive gradient is controlled OUTSIDE
    this function (eqx.partition on the caller's trainable filter) -- this just computes the
    value for whatever model it's given. rng=None (default, eval/inference): every level's
    encode() uses quantize_hard. rng given (training): split one sub-key per level so each
    level's gumbel noise (when quantize_mode="gumbel") is independent."""
    levels = model.levels
    x = levels[0].own_input_embed[flat_bytes]
    target = flat_bytes
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils = [], [], []
    level_rngs = [None] * phase if rng is None else list(jax.random.split(rng, phase))
    for i in range(phase):
        out = levels[i].encode(x, target, rng=level_rngs[i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        enc_losses.append(out["ntp_loss"])
        enc_accs.append(out["ntp_acc"])
        utils.append(out["util"])
        if i < phase - 1:
            x = code_embed(out["code_soft"], levels[i + 1].own_input_embed)
            target = out["code_idx"]

    dec_target = flat_bytes if phase - 1 == 0 else codes[phase - 2]
    dec_loss, dec_acc = levels[phase - 1].decode(dec_target, codes_soft[phase - 1], model.cfg.lag)

    ntp_loss_total = jnp.mean(jnp.stack(enc_losses))
    loss = dec_loss + model.cfg.ntp_weight * ntp_loss_total
    bpb = dec_loss / jnp.log(2.0)
    return loss, (bpb, dec_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)))


def phase_trainable_filter(model: HierEncDec, phase: int):
    """Boolean pytree (matching eqx.filter(model, eqx.is_array)'s structure). model.cfg.
    curriculum_mode="freeze" (default): True only for levels[phase-1]'s array leaves (every
    earlier level frozen for good). "no_freeze": True for ALL of levels[0..phase-1] -- nothing
    ever freezes, each phase just grows the trainable set by one more level."""
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    new_levels = list(filter_spec.levels)
    idxs = range(phase) if model.cfg.curriculum_mode == "no_freeze" else (phase - 1,)
    for i in idxs:
        if 0 <= i < len(model.levels):
            new_levels[i] = jax.tree_util.tree_map(lambda x: eqx.is_array(x), model.levels[i])
    return eqx.tree_at(lambda m: m.levels, filter_spec, new_levels)


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array)))


def cast_pytree(tree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def replicate(pytree, n_devices: int):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape)
                                   if eqx.is_array(x) else x, pytree)


def unreplicate(pytree):
    """Extracts device 0's copy (post-pmean, every device holds the same value)."""
    return jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, pytree)


def to_host(pytree):
    """Round-trips every array leaf through host numpy and back to a FRESH, uncommitted jax
    array -- use this (not to_single_device) before a leaf re-enters eqx.partition+replicate()
    for the NEXT phase. Two bugs found 2026-09-11, both from reusing a pmap-output leaf directly
    across phases: (1) to_single_device's explicit device_put COMMITS the array to device0; a
    level trainable across two consecutive phases (e.g. level1 in phases 1&2) then carries that
    commitment into the next phase's replicate() (jnp.broadcast_to inherits it), conflicting with
    pmap's expected NamedSharding("d") -- "Sharding passed to jit does not match the sharding on
    the respective arg". (2) leaving leaves as raw numpy (plain device_get, no jnp.asarray) broke
    eqx.partition's is_array-based filtering when that host-numpy model re-entered the next
    phase's pmap trace -- "TracerArrayConversionError: __array__() called on a traced array" --
    jnp.asarray gives back a genuine (uncommitted) jax array, not numpy. to_single_device stays
    correct for its original purpose (forcing a single concrete device right before a Pallas/
    Mosaic kernel call in run_gen_eval), just not for round-tripping through another phase's pmap."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(jax.device_get(x)) if eqx.is_array(x) else x, pytree)


def to_single_device(tree, device=None):
    """Forces every array leaf onto one concrete device -- needed before calling any Pallas/
    Mosaic kernel (splash_attention) on a pytree pulled out of pmap via unreplicate(): indexing a
    PmapSharding-backed array that way doesn't fully strip its multi-device sharding metadata,
    and Mosaic can't auto-partition (ported from run_lagcodec.py, confirmed 2026-09-09)."""
    device = device or jax.local_devices()[0]
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, tree)


def save_compare_grid(gen: np.ndarray, gt: np.ndarray, path: Path, pad: int = 2) -> None:
    from PIL import Image
    n, h, w, c = gen.shape
    grid = np.full((n * (h + pad) + pad, 2 * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        y = pad + i * (h + pad)
        grid[y:y + h, pad:pad + w] = gen[i]
        grid[y:y + h, 2 * pad + w:2 * pad + 2 * w] = gt[i]
    Image.fromarray(grid).save(path)


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
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight", "lag",
                  "weight_sharing", "precision", "curriculum_mode", "quantize_mode",
                  "gumbel_temperature", "gumbel_at_inference")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--epochs_per_phase", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="sinkgd", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={"sinkhorn_iters": 1, "weight_decay": 0})
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--train_subset_n", type=int, default=100)
    p.add_argument("--qual_gen_n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--d_model", type=_tuple_arg, default=Config.d_model)
    p.add_argument("--n_layers", type=_tuple_arg, default=Config.n_layers)
    p.add_argument("--n_heads", type=_tuple_arg, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=_tuple_arg, default=Config.n_kv_heads)
    p.add_argument("--strides", type=_tuple_arg, default=Config.strides)
    p.add_argument("--code_vocab", type=_tuple_arg, default=Config.code_vocab)
    p.add_argument("--pq_chunks", type=_tuple_arg, default=Config.pq_chunks)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
    p.add_argument("--ntp_weight", type=float, default=Config.ntp_weight)
    p.add_argument("--lag", type=int, default=Config.lag)
    p.add_argument("--weight_sharing", type=lambda x: x.lower() != "false", default=Config.weight_sharing)
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--curriculum_mode", type=str, default=Config.curriculum_mode, choices=["freeze", "no_freeze"])
    p.add_argument("--quantize_mode", type=str, default=Config.quantize_mode, choices=["argmax", "gumbel"])
    p.add_argument("--gumbel_temperature", type=float, default=Config.gumbel_temperature)
    p.add_argument("--gumbel_at_inference", type=lambda x: x.lower() != "false", default=Config.gumbel_at_inference)
    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    if unknown:
        p.error(f"--config {pre_args.config} sets unknown field(s): {sorted(unknown)}")
    p.set_defaults(**config_vars)
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = pre_args.config.stem

    n_devices = args.n_devices or jax.local_device_count()
    print(f"jax devices ({n_devices} used of {jax.local_device_count()} local): {jax.devices()}")
    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})
    n_levels = len(cfg.strides)
    n_phases = n_levels - 1

    (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size, n_devices,
                                shuffle=True, seed=args.seed)

    rng = jax.random.PRNGKey(args.seed)
    model = HierEncDec(rng, cfg)
    n_params = count_params(model)

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"n_levels={n_levels} n_phases={n_phases} params={n_params / 1e6:.2f}M "
           f"weight_sharing={cfg.weight_sharing}")

    compute_dtype = jnp.bfloat16 if cfg.precision == "bf16" else jnp.float32
    recon_prompt = train_np[:args.qual_gen_n]
    flat_prompt = jnp.array(recon_prompt.reshape(args.qual_gen_n, SEQ_LEN))
    gt_img = recon_prompt.astype(np.uint8)

    def run_gen_eval(eval_model, top: int, tag: str, include_reconstruct: bool = False) -> tuple:
        """CASCADE (codes[top] given, levels top..0 all generated) gen eval by default -- the
        real cross-level exposure-bias test. Reconstruct (real codes[0] given, decode_generate
        bytes directly) is OFF by default (chat 2026-09-11): it's confusing side-by-side with
        CASCADE since at top=0 they're numerically identical, and reconstruct never changes once
        level0 freezes (masking how bad the cascade actually is at later phases) -- pass
        include_reconstruct=True to get it back. top=phase-1 mid-training (the phase's just-
        completed decoder), top=n_levels-2 at the end. Saves samples_{tag}_cascade.png (+
        samples_{tag}_reconstruct.png if enabled), logs gen_cascade_acc (+ gen_recon_acc)."""
        m = cast_pytree(eval_model, compute_dtype)
        x = m.levels[0].own_input_embed[flat_prompt]
        target = flat_prompt
        codes, codes_soft = [], []
        eval_rngs = ([None] * (top + 1) if not cfg.gumbel_at_inference
                     else list(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), hash(tag) % (2**31)), top + 1)))
        for i in range(top + 1):
            out = m.levels[i].encode(x, target, rng=eval_rngs[i])
            codes.append(out["code_idx"])
            codes_soft.append(out["code_soft"])
            if i < top:
                x = code_embed(out["code_soft"], m.levels[i + 1].own_input_embed)
                target = out["code_idx"]

        recon_acc = None
        if include_reconstruct:
            recon = m.levels[0].decode_generate(codes[0], cfg.lag, greedy=True, seed=0)
            recon_acc = float(jnp.mean(recon == flat_prompt))
            recon_img = np.asarray(recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
            save_compare_grid(recon_img, gt_img, run_dir / f"samples_{tag}_reconstruct.png")

        cur_code = codes[top]
        for i in range(top, 0, -1):
            cur_code = m.levels[i].decode_generate(cur_code, cfg.lag, greedy=True, seed=0)
        cascade_recon = m.levels[0].decode_generate(cur_code, cfg.lag, greedy=True, seed=0)
        cascade_acc = float(jnp.mean(cascade_recon == flat_prompt))
        cascade_img = np.asarray(cascade_recon).reshape(args.qual_gen_n, 32, 32, 3).astype(np.uint8)
        save_compare_grid(cascade_img, gt_img, run_dir / f"samples_{tag}_cascade.png")

        msg = f"[{tag}] top={top} CASCADE gen_byte_acc={cascade_acc:.4f}"
        rec = dict(tag=tag, gen_cascade_acc=cascade_acc)
        if include_reconstruct:
            msg += f" reconstruct gen_byte_acc={recon_acc:.4f}"
            rec["gen_recon_acc"] = recon_acc
        logger(msg, **rec)
        return recon_acc, cascade_acc

    step = 0
    for phase in range(1, n_phases + 1):
        filter_spec = phase_trainable_filter(model, phase)
        diff_model, static_model = eqx.partition(model, filter_spec)

        def loss_fn(diff_model, static_model, flat_bytes, rng, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            return phase_forward(m, flat_bytes, phase, rng=rng)

        lr_schedule = warmup_const_schedule(args.lr, args.warmup_steps)
        if args.optimizer == "sinkgd":
            optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
        else:
            optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay, **args.optimizer_kwargs)
        opt_state = optimizer.init(diff_model)

        def train_step(diff_model, opt_state, rng, flat_bytes, static_model=static_model):
            rng, step_rng = jax.random.split(rng)
            use_rng = step_rng if cfg.quantize_mode == "gumbel" else None
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(diff_model, static_model, flat_bytes, use_rng)
            grads = jax.lax.pmean(grads, axis_name="d")
            loss = jax.lax.pmean(loss, axis_name="d")
            aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            return diff_model, opt_state, rng, loss, aux

        train_step = jax.pmap(train_step, axis_name="d")
        p_diff_model = replicate(diff_model, n_devices)
        p_opt_state = replicate(opt_state, n_devices)
        p_rng = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), phase), n_devices)

        trained_desc = f"levels0-{phase - 1}" if cfg.curriculum_mode == "no_freeze" else f"level{phase - 1}"
        active_desc = f"phase{phase}[{trained_desc}]"
        logger(f"=== starting {active_desc} for {args.epochs_per_phase} epochs ===")
        pbar = tqdm(range(1, args.epochs_per_phase + 1), desc=active_desc, mininterval=10.0)
        for epoch in pbar:
            epoch_losses = []
            for flat in train_iter:
                flat = jnp.array(flat)
                p_diff_model, p_opt_state, p_rng, loss, aux = train_step(p_diff_model, p_opt_state, p_rng, flat)
                step += 1
                loss0 = float(loss[0])
                epoch_losses.append(loss0)
                bpb, acc, ntp_bpb, ntp_acc, util = [float(a[0]) for a in aux]
                pbar.set_postfix(step=step, loss=f"{loss0:.3f}", bpb=f"{bpb:.3f}", acc=f"{acc:.3f}",
                                  ntp_bpb=f"{ntp_bpb:.3f}", ntp_acc=f"{ntp_acc:.3f}")
                if step % args.log_every == 0:
                    logger(f"phase={phase} epoch={epoch} step={step} loss={loss0:.4f} "
                           f"dec_bpb={bpb:.4f} dec_acc={acc:.4f} ntp_bpb={ntp_bpb:.4f} "
                           f"ntp_acc={ntp_acc:.4f} util={util:.3f}",
                           phase=phase, epoch=epoch, step=step, loss=loss0, dec_bpb=bpb,
                           dec_acc=acc, ntp_bpb=ntp_bpb, ntp_acc=ntp_acc, util=util)

            if epoch % 500 == 0:
                snapshot = eqx.combine(to_single_device(unreplicate(p_diff_model)), static_model)
                run_gen_eval(snapshot, top=phase - 1, tag=f"phase{phase}_epoch{epoch}")

        diff_model = to_host(unreplicate(p_diff_model))
        model = eqx.combine(diff_model, static_model)
        freeze_msg = "no freeze (no_freeze mode)" if cfg.curriculum_mode == "no_freeze" else f"freezing level {phase - 1}"
        logger(f"=== {active_desc} done, {freeze_msg} ===")
        ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(ckpt_dir / "model.eqx", model)
        run_gen_eval(model, top=phase - 1, tag=f"phase{phase}_final")

    logger("=== all phases done, running final top-down cascade eval ===")
    run_gen_eval(model, top=n_levels - 2, tag="final")
    logger("training done")


if __name__ == "__main__":
    main()
