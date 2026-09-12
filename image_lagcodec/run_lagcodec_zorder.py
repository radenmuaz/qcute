"""Fork of run_lagcodec_sampler.py (chat 2026-09-12) -- same cascade rollout sampler mechanism,
curriculum, everything else unchanged. Extensions layered on top since:

1. BYTE-GROUP GENERALIZATION: level0's own input/target alphabet is Config.byte_group members
   of a 256-vocab (byte_group=1: one byte per position, old is_byte_level behavior; byte_group=3:
   one PIXEL (R,G,B) per position, predicted/embedded as ONE group, same convention every other
   level already uses for its own PQ code). flat_bytes is always (B, n_positions, byte_group).

2. TOKEN HEAD (chat 2026-09-12, renamed from "group head"/"mtp" -- this predicts a position's
   OWN in_pq_chunks/byte_group MEMBERS, e.g. one pixel's R,G,B -- NOT future timesteps, so it
   isn't really "multi-token-prediction"; that name is reserved for extension 4 below).
   Config.token_head_type is a per-level tuple, one of:
     - "linears" (default): every member predicted independently in parallel off one shared
       linear projection (reshape_pq) -- cheapest, matches every level's existing code_head/
       ntp_head convention.
     - "ar": tiny causal chain (ported from run_ar_clockwork.py's SequentialRGBHead) -- member m
       conditions on a shared ctx projection AND the real (teacher-forced) values of members
       0..m-1 via a shared embedding table, one tiny self-attention + residual + UNTIED linear
       head (no mlp, no weight tying between input embedding and output head).
     - "diffusion" (KIV, kept not deleted): masked bidirectional single-shot (ported from
       run_ar_clockwork.py's DiffusionRGBHead) -- flaky, near-zero generation even after fixing
       a real splash-attention-padding bug (dense_self_attention) and widening its masking
       schedule; left in place at the user's request rather than removed, but not a working
       option today (mask is hardcoded to always-True as an unresolved sanity check).
   Config.token_dim/token_n_heads are PER-LEVEL (renamed from mtp_dim/mtp_n_heads).

3. Z-ORDER (MORTON) TRAVERSAL: Config.traversal="raster" (default) or "zorder" -- pixels visited
   in Morton-curve order instead of row-major; each pixel's R,G,B stays contiguous regardless of
   byte_group. Fixed permutation table, applied in BatchIterator / inverted in sample-saving.

4. TRUE MTP -- TIMESTEP-WISE multi-token prediction (chat 2026-09-12, Medusa/DeepSeek-MTP
   style): from the SAME decoder hidden state h_t, predict K FUTURE positions t+1..t+K directly
   (no recurrence needed for the prediction itself). Config.mtp_horizon (K, per level, must be
   <=that level's own stride or __post_init__ raises) and Config.mtp_mode (per level,
   "parallel"|"ar" -- how the K future steps relate to each other) are new, ORTHOGONAL to
   token_head_type (which governs how ONE position's own in_pq_chunks members relate). Two
   combos implemented: (linears,parallel) -- K independent linear heads off h_t, pure Medusa, no
   chaining anywhere; (ar,ar), chat 2026-09-12 -- "two nested ar transformers": an OUTER causal
   mini-transformer chains the K future TIMESTEPS (teacher-forced real future groups during
   training, summed via the INNER token-ar chain's own token_member_embed table), each outer
   step's output projected back to D and fed into the SAME INNER _token_teacher_forced_ar to
   predict that timestep's own in_pq_chunks members. Both are trained via an AUXILIARY loss
   (Config.mtp_weight, see _mtp_loss) against the real future groups -- confirmed 2026-09-12 that
   without this the mtp params never receive gradient at all (decode_logits_and_target never
   touched them before this fix; only the inference-time no-verify decode did, forever-random).
   Both combos have a "no-verify" decode consumer (mtp_predict_no_verify_standalone -- takes the
   K drafted positions directly, no check against the real sequential decode; "ar" mode chains
   its own sampled groups at generation time since there's no real future to teacher-force with).
   NOT YET implemented: (linears,ar) and (ar,parallel) combos, and the verified self-speculative
   decode mode (draft via MTP heads, verify against the real one-step causal decode, accept
   longest matching prefix). mtp_horizon=1 (default) disables all of this -- decode_generate
   behaves exactly as before.

pmap-parallel across all local devices.

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/<name>.py
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

from image_lagcodec.eqx_common import (Attention, Block, RMSNorm, apply_rope, rmsnorm, rope_cos_sin,
                                        sinkgd, warmup_const_schedule)

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
TOTAL_BYTES = 32 * 32 * 3   # raw byte count per image, fixed regardless of byte_group/traversal


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
    code_vocab: tuple = (4, 4, 4, 4)
    pq_chunks: tuple = (5, 5, 5, 5)
    mlp_mult: tuple = 2          # chat 2026-09-12 -- now PER-LEVEL (was a single global int);
    # a bare int (the default, or any config's override) broadcasts to a uniform tuple sized to
    # however many levels THIS config actually has -- see bcast() in __post_init__. A fixed-
    # length tuple default would silently break any config with a different level count.
    rope_base: tuple = 10000.0    # chat 2026-09-12 -- now PER-LEVEL, bare-scalar default
    ntp_weight: float = 1.0     # stays global -- a loss-mixing coefficient, not a capacity knob
    lag: tuple = 0   # chat 2026-09-12 -- now PER-LEVEL (was a single global int), bare-scalar default
    weight_sharing: tuple = True   # chat 2026-09-12 -- now PER-LEVEL, bare-scalar default
    precision: str = "bf16"           # stays global -- whole-model compute dtype
    curriculum_mode: str = "freeze"   # stays global -- training-loop control
    quantize_mode: str = "argmax"     # stays global -- training-time quantization strategy
    gumbel_temperature: tuple = 1.0   # chat 2026-09-12 -- now PER-LEVEL, bare-scalar default
    gumbel_at_inference: bool = False
    cascade_rollout_prob: float = 0.5

    byte_group: int = 1          # level0's own group size, in {1, 3}. 1 = one byte per position
    # (old is_byte_level behavior, default). 3 = one pixel (R,G,B) per position.
    token_head_type: tuple = "linears"   # per level, bare-scalar default, one of "linears"/"ar"/
    # "diffusion" -- see module docstring point 2 (renamed from group_head_type). "diffusion" is
    # KIV (kept, not deleted, chat 2026-09-12) -- it was flaky (near-zero generation even after
    # fixing a real splash-attention-padding bug and widening its masking schedule) but the user
    # asked to leave it in place rather than remove it.
    token_dim: tuple = 64       # per level (renamed from mtp_dim), bare-scalar default, only used
    # by levels with token_head_type in ("ar","diffusion") -- size to roughly match "linears"'
    # own dec_head param count per level (audited 2026-09-12: a single global value badly under/
    # over-parameterizes different levels' alphabets).
    token_n_heads: tuple = 4        # per level (renamed from mtp_n_heads), bare-scalar default
    token_mask_prob: float = 0.15   # "diffusion" token head masking probability (KIV, unused by
    # the current sanity-check version which hardcodes always-mask -- see _token_teacher_forced_
    # diffusion's docstring; renamed from mask_prob)

    mtp_horizon: tuple = 1         # chat 2026-09-12 -- NEW, true timestep-wise MTP (see module
    # docstring point 4). Per level, bare-scalar default, K future positions predicted from one
    # hidden state. 1 = disabled (today's plain one-step decode). Must be <=that level's own
    # stride. Trained via an AUXILIARY loss (Config.mtp_weight) against the real future groups --
    # see _mtp_loss; without this the mtp head/chain would never receive gradient at all (only
    # decode_generate_mtp_no_verify would ever read it, at inference, forever-random).
    mtp_mode: tuple = "parallel"   # chat 2026-09-12 -- NEW, per level, bare-scalar default,
    # per level, "parallel" (K independent linear heads off h_t, no chaining -- requires
    # token_head_type="linears") or "ar" (chat 2026-09-12: nested causal chain -- requires
    # token_head_type="ar"; OUTER causal mini-transformer chains the K future TIMESTEPS
    # (teacher-forced real future groups, summed via the token head's own member-embed table),
    # each outer step's output projected back to D and fed into the SAME INNER token-ar chain
    # to predict that timestep's own in_pq_chunks members -- "two nested ar transformers"). Only
    # (linears,parallel) and (ar,ar) combos are implemented; anything else with mtp_horizon>1
    # raises NotImplementedError in __post_init__.
    mtp_weight: float = 0.1   # chat 2026-09-12 -- stays global (loss-mixing coefficient, like
    # ntp_weight), scales the auxiliary MTP loss added on top of the main per-position loss.

    traversal: str = "raster"    # "raster" (default, row-major) or "zorder" (Morton curve over
    # pixels, RGB stays contiguous per pixel regardless of byte_group -- see module docstring).

    def __post_init__(self):
        n = len(self.strides)

        def bcast(name, types):
            val = getattr(self, name)
            if isinstance(val, types):
                setattr(self, name, (val,) * n)

        # chat 2026-09-12: a bare scalar broadcasts to a same-length tuple -- convenience so a
        # config doesn't have to spell out e.g. "(4,)*5" for a value uniform across every level.
        bcast("mlp_mult", int)
        bcast("rope_base", (int, float))
        bcast("lag", int)
        bcast("weight_sharing", bool)
        bcast("gumbel_temperature", (int, float))
        bcast("token_head_type", str)
        bcast("token_dim", int)
        bcast("token_n_heads", int)
        bcast("mtp_horizon", int)
        bcast("mtp_mode", str)

        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.lag) == n
        assert len(self.weight_sharing) == n and len(self.gumbel_temperature) == n
        assert len(self.token_head_type) == n
        assert all(t in ("linears", "ar", "diffusion") for t in self.token_head_type)
        assert len(self.mtp_horizon) == n and len(self.mtp_mode) == n
        assert all(m in ("parallel", "ar") for m in self.mtp_mode)
        for i in range(n):
            if self.token_head_type[i] in ("ar", "diffusion"):
                assert i < len(self.token_dim) and i < len(self.token_n_heads), \
                    f"level {i} uses token_head_type={self.token_head_type[i]!r} but token_dim/" \
                    f"token_n_heads only has {len(self.token_dim)} entries -- set one per level"
                assert self.token_dim[i] % self.token_n_heads[i] == 0
            if self.mtp_horizon[i] > 1:
                combo_ok = (self.token_head_type[i] in ("linears", "ar") and self.mtp_mode[i] == "parallel") \
                    or (self.token_head_type[i] == "ar" and self.mtp_mode[i] == "ar")
                if not combo_ok:
                    raise NotImplementedError(
                        f"level {i}: token_head_type={self.token_head_type[i]!r} x "
                        f"mtp_mode={self.mtp_mode[i]!r} with mtp_horizon>1 is not implemented -- "
                        "only (linears,parallel), (ar,parallel), and (ar,ar) are supported")
            stride_i = self.strides[i]
            if stride_i != -1:
                assert self.mtp_horizon[i] >= 1
                assert self.mtp_horizon[i] <= stride_i, \
                    f"level {i}: mtp_horizon={self.mtp_horizon[i]} exceeds its own stride=" \
                    f"{stride_i} -- can't predict further ahead than one stride group"
        assert self.byte_group in (1, 3), "byte_group must be 1 (per-byte) or 3 (per-pixel RGB)"
        assert TOTAL_BYTES % self.byte_group == 0
        assert self.traversal in ("raster", "zorder")
        assert self.strides[-1] == -1, "top level's stride is unused -- use -1 as the don't-care convention"
        assert all(s >= 1 for s in self.strides[:-1])
        n_positions = TOTAL_BYTES // self.byte_group
        assert n_positions % math.prod(self.strides[:-1]) == 0
        assert self.precision in ("bf16", "fp32")
        assert self.curriculum_mode in ("freeze", "no_freeze")
        assert self.curriculum_mode == "no_freeze", \
            "run_lagcodec_zorder requires curriculum_mode='no_freeze' -- a level conditioned on " \
            "cascade-simulated ctx must stay trainable to adapt to it (see module docstring)"
        assert 0.0 <= self.cascade_rollout_prob <= 1.0
        assert self.quantize_mode in ("argmax", "gumbel")
        resolved_kv = []
        for i in range(n):
            kv = self.n_kv_heads[i] if self.n_kv_heads[i] is not None else max(1, self.n_heads[i] // 4)
            assert self.n_heads[i] % kv == 0
            assert self.d_model[i] % self.n_heads[i] == 0
            resolved_kv.append(kv)
        self.n_kv_heads = tuple(resolved_kv)


def n_positions_of(cfg: Config) -> int:
    return TOTAL_BYTES // cfg.byte_group


# ---------------------------------------------------------------------------
# Z-order (Morton) traversal -- pure function of img_size, no data dependence
# ---------------------------------------------------------------------------

def zorder_pixel_order(img_size: int) -> np.ndarray:
    """Returns a (img_size**2,) permutation: order[t] = raster pixel-index (row*img_size+col)
    visited at traversal-step t. Built by interleaving the bits of (x, y) into a Morton code and
    stable-sorting raster indices by it -- works for any img_size, not just powers of 2."""
    def part1by1(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint32) & 0x0000ffff
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    ys, xs = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing="ij")
    raster_idx = (ys * img_size + xs).reshape(-1)
    morton = part1by1(xs.reshape(-1)) | (part1by1(ys.reshape(-1)) << 1)
    order = raster_idx[np.argsort(morton, kind="stable")]
    return order


def pixel_order_for(cfg: Config) -> np.ndarray:
    if cfg.traversal == "raster":
        return np.arange(cfg.img_size * cfg.img_size)
    return zorder_pixel_order(cfg.img_size)


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


def images_to_positions(images: np.ndarray, cfg: Config, pixel_order: np.ndarray) -> np.ndarray:
    """images: (N,img_size,img_size,3) uint8/int -> (N, n_positions, byte_group) int32, pixels
    visited in `pixel_order` (traversal), each pixel's R,G,B kept contiguous regardless of
    byte_group (byte_group=3: one pixel per position; byte_group=1: 3 consecutive positions per
    pixel, in R,G,B order -- "every pixel goes through its RGB first")."""
    n = images.shape[0]
    pix = images.reshape(n, cfg.img_size * cfg.img_size, 3)[:, pixel_order, :]
    if cfg.byte_group == 3:
        return pix.astype(np.int32)
    return pix.reshape(n, cfg.img_size * cfg.img_size * 3, 1).astype(np.int32)


def positions_to_image(positions: np.ndarray, cfg: Config, pixel_order: np.ndarray) -> np.ndarray:
    """Inverse of images_to_positions: (B, n_positions, byte_group) in TRAVERSAL order ->
    (B, img_size, img_size, 3) raster image (uint8)."""
    B = positions.shape[0]
    pix_traversal = positions.reshape(B, cfg.img_size * cfg.img_size, 3)
    raster = np.zeros_like(pix_traversal)
    raster[:, pixel_order, :] = pix_traversal
    return raster.reshape(B, cfg.img_size, cfg.img_size, 3).astype(np.uint8)


class BatchIterator:
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, n_devices: int,
                 shuffle: bool, seed: int, cfg: Config):
        self.images, self.labels = images, labels
        self.batch_size, self.n_devices = batch_size, n_devices
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.total = batch_size * n_devices
        self.cfg = cfg
        self.pixel_order = pixel_order_for(cfg)
        self.n_positions = n_positions_of(cfg)

    def __len__(self):
        return len(self.images) // self.total

    def __iter__(self):
        n = len(self.images)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n - self.total + 1, self.total):
            sel = idx[start:start + self.total]
            img = self.images[sel]
            positions = images_to_positions(img, self.cfg, self.pixel_order)
            yield positions.reshape(self.n_devices, self.batch_size, self.n_positions, self.cfg.byte_group)


# ---------------------------------------------------------------------------
# Quantization (identical convention to run_lagcodec_sampler.py)
# ---------------------------------------------------------------------------

def quantize_hard(logits: jnp.ndarray) -> tuple:
    soft = jax.nn.softmax(logits, axis=-1)
    idx = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=soft.dtype)
    code_soft = soft + jax.lax.stop_gradient(hard - soft)
    return code_soft, idx


def quantize_gumbel(logits: jnp.ndarray, rng, temperature: float = 1.0) -> tuple:
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


def sample_idx(logits: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
    if greedy:
        return jnp.argmax(logits, axis=-1), rng
    rng, k_ = jax.random.split(rng)
    return jax.random.categorical(k_, logits / temperature, axis=-1), rng


def dense_self_attention(attn: Attention, x: jnp.ndarray, causal: bool = False) -> jnp.ndarray:
    """Plain (non-Pallas) dense self-attention -- the ATTENTION BRANCH ONLY (no residual, no
    norm, no mlp). Originally built for the "diffusion" token head's non-causal case (KIV, kept
    not deleted -- diffusion itself is flaky/near-zero generation): Attention.__call__(causal=
    False) routes through splash_attention, which pads T up to 128 (_SPLASH_BLOCK);
    splash_attention_mask.FullMask's own docstring says it "allows all tokens to attend to all
    other tokens" -- no real-length truncation, unlike CausalMask, so every real diffusion-head
    query would attend over ~124 garbage zero-padded key positions, corrupting the computation.

    chat 2026-09-12: causal=True now ALSO uses this dense path, for a DIFFERENT reason --
    splash_attention's 128-padding is correctness-safe under causal masking (padded kv sits past
    any real causal query, never attended to), but NOT memory-safe: the Pallas kernel still
    allocates/computes at the padded size (128) regardless of the real length. Every "ar" token-
    head / mtp-chain call here has a tiny real sequence (chunks or mtp_horizon, typically <=8),
    so routing it through splash_attention wastes ~16-42x the compute/memory it needs -- and the
    (ar,ar) nested MTP design makes 5 such padded calls per position (1 outer + mtp_horizon
    inner), compounding into the OOM confirmed 2026-09-12 (33-50G required vs 30.75G available).
    Dense attention computes at the REAL length, no padding, for either case.

    This function replicates Attention.__call__'s exact math (qkv proj, per-head RMSNorm, RoPE,
    GQA repeat, out proj) but with a manual dense softmax instead of splash_attention."""
    B, T, D = x.shape
    hd = D // attn.n_heads
    qkv = x @ attn.qkv
    q, k, v = jnp.split(qkv, [D, D + attn.n_kv_heads * hd], axis=-1)
    q = q.reshape(B, T, attn.n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    q, k = rmsnorm(q, attn.q_norm), rmsnorm(k, attn.k_norm)
    cos, sin = rope_cos_sin(T, hd, attn.rope_base)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = attn.n_heads // attn.n_kv_heads
    k, v = jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)
    scale = 1.0 / math.sqrt(hd)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    if causal:
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ attn.out


def token_ar_teacher_forced(in_proj, member_embed, norm1, attn, ln_f, out_head, dim, in_code_vocab,
                             h: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Free-function form of the token-ar chain (chat 2026-09-12, refactored out of EncDecLevel
    so both a level's OWN token head and a "duplicate ar heads" MTP bank -- K independent copies
    of this exact mechanism, mtp_mode="parallel" x token_head_type="ar" -- can share it). h:
    (...,D) one context vector per group. target: (...,chunks) real member values. Returns
    logits (...,chunks,vocab): member m predicted from ctx + real members[0..m-1]."""
    lead = h.shape[:-1]
    D = h.shape[-1]
    chunks = target.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    ctx = (h.reshape(N, D) @ in_proj)[:, None, :]
    tgt_flat = target.reshape(N, chunks)
    member_embeds = member_embed[tgt_flat[:, :chunks - 1]] if chunks > 1 else \
        jnp.zeros((N, 0, dim), dtype=ctx.dtype)
    seq_in = jnp.concatenate([ctx, member_embeds], axis=1)
    h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
    logits = ln_f(h1) @ out_head
    return logits.reshape(*lead, chunks, in_code_vocab)


def token_ar_generate(in_proj, member_embed, norm1, attn, ln_f, out_head, chunks: int,
                       h: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
    lead = h.shape[:-1]
    D = h.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    ctx = (h.reshape(N, D) @ in_proj)[:, None, :]
    collected = [ctx]
    vals = []
    for m in range(chunks):
        seq_in = jnp.concatenate(collected, axis=1)
        h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
        logit_m = ln_f(h1)[:, -1, :] @ out_head
        val_m, rng = sample_idx(logit_m, rng, greedy, temperature)
        vals.append(val_m)
        if m < chunks - 1:
            collected.append(member_embed[val_m][:, None, :])
    idx = jnp.stack(vals, axis=1).reshape(*lead, chunks)
    return idx, rng


# ---------------------------------------------------------------------------
# EncDecLevel -- one module per level, optionally shared between encoder/decoder roles
# ---------------------------------------------------------------------------

class EncDecLevel(eqx.Module):
    blocks: list
    ln_f: RMSNorm
    own_input_embed: jnp.ndarray
    ntp_head: jnp.ndarray
    code_head: jnp.ndarray
    bos_embed: jnp.ndarray
    ctx_embed: jnp.ndarray
    dec_blocks: list
    dec_ln_f: RMSNorm
    dec_target_embed: jnp.ndarray
    dec_head: jnp.ndarray
    # token_head_type in ("ar","diffusion") params (None unless has_decoder and token_head_type
    # needs them) -- self-attention + residual + UNTIED linear head only (no mlp, no weight
    # tying -- chat 2026-09-12). token_mask_embed/token_channel_embed are "diffusion"-only (KIV).
    token_in_proj: jnp.ndarray
    token_member_embed: jnp.ndarray
    token_mask_embed: jnp.ndarray
    token_channel_embed: jnp.ndarray
    token_norm1: RMSNorm
    token_attn: Attention
    token_ln_f: RMSNorm
    token_out_head: jnp.ndarray
    # true MTP params (chat 2026-09-12, None unless has_decoder and mtp_horizon>1):
    # (linears,parallel) -- mtp_out_head only, K independent linear heads packed into one
    # (D, K*ntp_out) matrix. (ar,ar) -- the OUTER causal chain's own params (reuses the INNER
    # token-ar chain's token_attn/token_member_embed/token_out_head unchanged, "nested").
    # (ar,parallel) -- mtp_heads_* lists, K FULLY INDEPENDENT copies of the token-ar mechanism
    # ("duplicate ar heads"), each applied to the SAME h_t, no chaining across timesteps.
    mtp_out_head: jnp.ndarray
    mtp_in_proj: jnp.ndarray
    mtp_attn: Attention
    mtp_norm1: RMSNorm
    mtp_ln_f: RMSNorm
    mtp_out_proj: jnp.ndarray
    mtp_heads_in_proj: list
    mtp_heads_member_embed: list
    mtp_heads_norm1: list
    mtp_heads_attn: list
    mtp_heads_ln_f: list
    mtp_heads_out_head: list
    has_decoder: bool = eqx.field(static=True)
    weight_sharing: bool = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    in_pq_chunks: int = eqx.field(static=True)
    in_code_vocab: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    quantize_mode: str = eqx.field(static=True)
    gumbel_temperature: float = eqx.field(static=True)
    token_head_type: str = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    token_mask_prob: float = eqx.field(static=True)
    mtp_horizon: int = eqx.field(static=True)
    mtp_mode: str = eqx.field(static=True)
    mtp_weight: float = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, has_decoder: bool, weight_sharing: bool):
        D = cfg.d_model[level]
        self.K = cfg.strides[level] if cfg.strides[level] != -1 else 1
        self.n_heads, self.n_kv_heads = cfg.n_heads[level], cfg.n_kv_heads[level]
        self.quantize_mode = cfg.quantize_mode
        self.gumbel_temperature = cfg.gumbel_temperature[level]
        is_byte_level = (level == 0)
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[level], cfg.code_vocab[level]
        # level0's own in-stream is (byte_group, 256) instead of a special is_byte_level case --
        # numerically identical to the old behavior when byte_group=1.
        self.in_pq_chunks = cfg.byte_group if is_byte_level else cfg.pq_chunks[level - 1]
        self.in_code_vocab = 256 if is_byte_level else cfg.code_vocab[level - 1]
        self.has_decoder = has_decoder
        self.weight_sharing = weight_sharing
        self.token_head_type = cfg.token_head_type[level]
        self.token_dim = cfg.token_dim[level] if level < len(cfg.token_dim) else None
        self.token_mask_prob = cfg.token_mask_prob
        self.mtp_horizon = cfg.mtp_horizon[level]
        self.mtp_mode = cfg.mtp_mode[level]
        self.mtp_weight = cfg.mtp_weight
        own_vocab = 256 if is_byte_level else self.in_code_vocab
        ntp_out = self.in_pq_chunks * self.in_code_vocab
        keys = jax.random.split(key, 20)

        self.own_input_embed = jax.random.normal(keys[0], (own_vocab, D)) * 0.02
        n_layers = cfg.n_layers[level]
        block_keys = jax.random.split(keys[1], n_layers)
        self.blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level])
                       for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.code_head = jax.random.normal(keys[2], (D, self.pq_chunks * self.code_vocab)) * 0.02
        self.ntp_head = jax.random.normal(keys[3], (D, ntp_out)) * 0.02
        self.bos_embed = jax.random.normal(keys[4], (D,)) * 0.02
        self.ctx_embed = jax.random.normal(keys[5], (self.code_vocab, D)) * 0.02

        if has_decoder and not weight_sharing:
            dec_block_keys = jax.random.split(keys[6], n_layers)
            self.dec_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level])
                               for k in dec_block_keys]
            self.dec_ln_f = RMSNorm(D)
            self.dec_target_embed = jax.random.normal(keys[7], (own_vocab, D)) * 0.02
            self.dec_head = jax.random.normal(keys[8], (D, ntp_out)) * 0.02
        else:
            self.dec_blocks, self.dec_ln_f, self.dec_target_embed, self.dec_head = None, None, None, None

        if has_decoder and self.token_head_type in ("ar", "diffusion"):
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            self.token_in_proj = jax.random.normal(keys[9], (D, tdim)) * 0.02
            self.token_member_embed = jax.random.normal(keys[10], (self.in_code_vocab, tdim)) * 0.02
            self.token_norm1 = RMSNorm(tdim)
            self.token_attn = Attention(keys[11], tdim, theads, theads, cfg.rope_base[level])
            self.token_ln_f = RMSNorm(tdim)
            self.token_out_head = jax.random.normal(keys[12], (tdim, self.in_code_vocab)) * 0.02
            if self.token_head_type == "diffusion":
                self.token_mask_embed = jax.random.normal(keys[14], (tdim,)) * 0.02
                self.token_channel_embed = jax.random.normal(keys[15], (self.in_pq_chunks, tdim)) * 0.02
            else:
                self.token_mask_embed, self.token_channel_embed = None, None
        else:
            (self.token_in_proj, self.token_member_embed, self.token_mask_embed,
             self.token_channel_embed, self.token_norm1, self.token_attn, self.token_ln_f,
             self.token_out_head) = (None,) * 8

        (self.mtp_heads_in_proj, self.mtp_heads_member_embed, self.mtp_heads_norm1,
         self.mtp_heads_attn, self.mtp_heads_ln_f, self.mtp_heads_out_head) = (None,) * 6
        if has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "parallel" and self.token_head_type == "linears":
            self.mtp_out_head = jax.random.normal(keys[13], (D, self.mtp_horizon * ntp_out)) * 0.02
            (self.mtp_in_proj, self.mtp_attn, self.mtp_norm1, self.mtp_ln_f, self.mtp_out_proj) = (None,) * 5
        elif has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "parallel" and self.token_head_type == "ar":
            # "duplicate ar heads" (chat 2026-09-12): K FULLY INDEPENDENT copies of the token-ar
            # mechanism, one per future timestep, all applied to the SAME h_t -- no chaining.
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            head_keys = jax.random.split(keys[19], self.mtp_horizon * 3)
            self.mtp_heads_in_proj = [jax.random.normal(head_keys[3 * k], (D, tdim)) * 0.02
                                       for k in range(self.mtp_horizon)]
            self.mtp_heads_member_embed = [jax.random.normal(head_keys[3 * k + 1], (self.in_code_vocab, tdim)) * 0.02
                                            for k in range(self.mtp_horizon)]
            self.mtp_heads_attn = [Attention(head_keys[3 * k + 2], tdim, theads, theads, cfg.rope_base[level])
                                    for k in range(self.mtp_horizon)]
            self.mtp_heads_norm1 = [RMSNorm(tdim) for _ in range(self.mtp_horizon)]
            self.mtp_heads_ln_f = [RMSNorm(tdim) for _ in range(self.mtp_horizon)]
            self.mtp_heads_out_head = [jax.random.normal(k, (tdim, self.in_code_vocab)) * 0.02
                                        for k in jax.random.split(keys[19], self.mtp_horizon)]
            (self.mtp_out_head, self.mtp_in_proj, self.mtp_attn, self.mtp_norm1,
             self.mtp_ln_f, self.mtp_out_proj) = (None,) * 6
        elif has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "ar":
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            self.mtp_in_proj = jax.random.normal(keys[16], (D, tdim)) * 0.02
            self.mtp_attn = Attention(keys[17], tdim, theads, theads, cfg.rope_base[level])
            self.mtp_norm1 = RMSNorm(tdim)
            self.mtp_ln_f = RMSNorm(tdim)
            self.mtp_out_proj = jax.random.normal(keys[18], (tdim, D)) * 0.02
            self.mtp_out_head = None
        else:
            (self.mtp_out_head, self.mtp_in_proj, self.mtp_attn, self.mtp_norm1, self.mtp_ln_f,
             self.mtp_out_proj) = (None,) * 6

    # --- encoder role (mirrors HierEncoder.EncoderLevel.forward) ---

    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, rng=None) -> dict:
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
        """Embeds THIS level's own previous target group as ONE summed token for the OUTER
        (group-to-group) causal decoder -- unaffected by token_head_type, which only governs HOW
        a group's own members get predicted, not how a whole group is fed back in."""
        table = self.own_input_embed if self.weight_sharing else self.dec_target_embed
        return code_embed(idx, table)

    def _dec_head_w(self) -> jnp.ndarray:
        return self.ntp_head if self.weight_sharing else self.dec_head

    # --- token head dispatch: "linears" (parallel, existing), "ar" (causal chain, ported from
    # run_ar_clockwork.py's SequentialRGBHead) -- predicts a position's OWN in_pq_chunks members ---

    def _token_logits_linears(self, h: jnp.ndarray) -> jnp.ndarray:
        logits = h @ self._dec_head_w()
        return reshape_pq(logits, self.in_pq_chunks, self.in_code_vocab)

    def _token_teacher_forced_ar(self, h: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """h: (...,D) one context vector per group. target: (...,chunks) real member values.
        Returns logits (...,chunks,vocab): member m predicted from ctx + real members[0..m-1].
        causal=True uses real splash_attention directly (padding-safe -- unlike non-causal/
        bidirectional attention, padded kv sits past any real causal query and is never
        attended to, confirmed 2026-09-12 during the (now-removed) diffusion head's audit)."""
        return token_ar_teacher_forced(self.token_in_proj, self.token_member_embed, self.token_norm1,
                                        self.token_attn, self.token_ln_f, self.token_out_head,
                                        self.token_dim, self.in_code_vocab, h, target)

    def _token_generate_ar(self, h: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
        return token_ar_generate(self.token_in_proj, self.token_member_embed, self.token_norm1,
                                  self.token_attn, self.token_ln_f, self.token_out_head,
                                  self.in_pq_chunks, h, rng, greedy, temperature)

    def _token_teacher_forced_diffusion(self, h: jnp.ndarray, target: jnp.ndarray, rng) -> tuple:
        """KIV (kept, not deleted, chat 2026-09-12) -- flaky, near-zero generation even after
        fixing a real splash-attention-padding bug (see dense_self_attention's docstring) and
        widening its masking schedule. Returns (logits (...,chunks,vocab), mask (...,chunks)
        bool -- True where masked). mask is currently hardcoded to always-True (sanity check --
        exactly matches inference's always-fully-masked input, no train/inference mismatch at
        all); Uniform(0.05,1.0) token_mask_prob sampling is commented out below, not deleted."""
        lead = h.shape[:-1]
        D = h.shape[-1]
        chunks = self.in_pq_chunks
        N = int(np.prod(lead)) if lead else 1
        ctx = (h.reshape(N, D) @ self.token_in_proj)
        tgt_flat = target.reshape(N, chunks)
        # prob_rng, mask_rng = jax.random.split(rng)
        # mask_prob = jax.random.uniform(prob_rng, (), minval=0.05, maxval=1.0)
        # mask = jax.random.bernoulli(mask_rng, mask_prob, (N, chunks))
        mask = jnp.ones((N, chunks), dtype=bool)   # sanity check: always mask (matches inference exactly)
        real = self.token_member_embed[tgt_flat]
        tok = jnp.where(mask[..., None], self.token_mask_embed, real) \
            + self.token_channel_embed[None, :, :] + ctx[:, None, :]
        h1 = tok + dense_self_attention(self.token_attn, self.token_norm1(tok))
        logits = self.token_ln_f(h1) @ self.token_out_head
        return logits.reshape(*lead, chunks, self.in_code_vocab), mask.reshape(*lead, chunks)

    def _token_generate_diffusion(self, h: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
        lead = h.shape[:-1]
        D = h.shape[-1]
        chunks = self.in_pq_chunks
        N = int(np.prod(lead)) if lead else 1
        ctx = (h.reshape(N, D) @ self.token_in_proj)
        tok = self.token_mask_embed[None, None, :] + self.token_channel_embed[None, :, :] + ctx[:, None, :]
        tok = jnp.broadcast_to(tok, (N, chunks, self.token_dim))
        h1 = tok + dense_self_attention(self.token_attn, self.token_norm1(tok))
        logits = self.token_ln_f(h1) @ self.token_out_head
        idx, rng = sample_idx(logits, rng, greedy, temperature)
        return idx.reshape(*lead, chunks), rng

    # --- true MTP auxiliary training loss (chat 2026-09-12) -- predicts REAL future groups
    # t+1..t+K from h_t, so the mtp params actually receive gradient (without this the mtp head/
    # chain is never touched by the main per-position loss and stays at random init forever --
    # only decode_generate_mtp_no_verify would ever read it, at inference). Only positions with
    # a full K-window of real future groups still inside the decoded sequence contribute. ---

    def _mtp_loss_parallel(self, h_t: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        B, T, D = h_t.shape
        K, chunks, vocab = self.mtp_horizon, self.in_pq_chunks, self.in_code_vocab
        valid_T = T - K
        if valid_T <= 0:
            return jnp.array(0.0, dtype=h_t.dtype)
        logits = (h_t[:, :valid_T, :] @ self.mtp_out_head).reshape(B, valid_T, K, chunks, vocab)
        future = jnp.stack([target[:, k + 1:k + 1 + valid_T, :] for k in range(K)], axis=2)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, future[..., None], axis=-1)[..., 0]
        return jnp.mean(nll)

    def _mtp_loss_ar_parallel(self, h_t: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """"Duplicate ar heads" (chat 2026-09-12): K fully independent copies of the token-ar
        mechanism, each applied to the SAME h_t (no chaining across timesteps, unlike (ar,ar))."""
        B, T, D = h_t.shape
        K = self.mtp_horizon
        valid_T = T - K
        if valid_T <= 0:
            return jnp.array(0.0, dtype=h_t.dtype)
        h_valid = h_t[:, :valid_T, :]
        future = jnp.stack([target[:, k + 1:k + 1 + valid_T, :] for k in range(K)], axis=2)
        losses = []
        for k in range(K):
            logits_k = token_ar_teacher_forced(
                self.mtp_heads_in_proj[k], self.mtp_heads_member_embed[k], self.mtp_heads_norm1[k],
                self.mtp_heads_attn[k], self.mtp_heads_ln_f[k], self.mtp_heads_out_head[k],
                self.token_dim, self.in_code_vocab, h_valid, future[:, :, k, :])
            loss_k, _ = self._dec_loss_acc(logits_k, future[:, :, k, :])
            losses.append(loss_k)
        return jnp.mean(jnp.stack(losses))

    def _mtp_loss_ar_ar(self, h_t: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Nested: OUTER causal chain over the K future timesteps (real future groups
        teacher-forced, summed into one token each via the INNER token-ar chain's own
        token_member_embed table), each outer output projected back to D and fed into the SAME
        INNER _token_teacher_forced_ar to predict that timestep's own in_pq_chunks members."""
        B, T, D = h_t.shape
        K, chunks = self.mtp_horizon, self.in_pq_chunks
        valid_T = T - K
        if valid_T <= 0:
            return jnp.array(0.0, dtype=h_t.dtype)
        N = B * valid_T
        h_valid = h_t[:, :valid_T, :].reshape(N, D)
        ctx0 = (h_valid @ self.mtp_in_proj)[:, None, :]
        future = jnp.stack([target[:, k + 1:k + 1 + valid_T, :] for k in range(K)], axis=2)
        future_flat = future.reshape(N, K, chunks)
        group_embeds = code_embed(future_flat[:, :K - 1, :], self.token_member_embed)
        seq_in = jnp.concatenate([ctx0, group_embeds], axis=1)
        h1 = seq_in + dense_self_attention(self.mtp_attn, self.mtp_norm1(seq_in), causal=True)
        outer_out = self.mtp_ln_f(h1)
        ctx_per_k = outer_out @ self.mtp_out_proj
        losses = []
        for k in range(K):
            inner_logits = self._token_teacher_forced_ar(ctx_per_k[:, k, :], future_flat[:, k, :])
            loss_k, _ = self._dec_loss_acc(inner_logits, future_flat[:, k, :])
            losses.append(loss_k)
        return jnp.mean(jnp.stack(losses))

    def _mtp_loss(self, h_t: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        if self.mtp_horizon <= 1:
            return jnp.array(0.0, dtype=h_t.dtype)
        if self.mtp_mode == "parallel" and self.token_head_type == "linears":
            return self._mtp_loss_parallel(h_t, target)
        if self.mtp_mode == "parallel" and self.token_head_type == "ar":
            return self._mtp_loss_ar_parallel(h_t, target)
        return self._mtp_loss_ar_ar(h_t, target)

    def _dec_loss_acc(self, logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray = None) -> tuple:
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
        correct = (jnp.argmax(logits, -1) == target).astype(jnp.float32)
        if mask is None:
            return jnp.mean(nll), jnp.mean(correct)
        m = mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(m), 1.0)
        return jnp.sum(nll * m) / denom, jnp.sum(correct * m) / denom

    def decode_logits_and_target(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, lag: int,
                                  rng=None) -> tuple:
        """Returns (logits, target, mask). mask is None except for "diffusion" (KIV -- only
        masked positions count toward loss/acc, standard MLM convention). target_seq/
        ctx_code_soft same shapes/semantics as run_lagcodec_sampler.py's decode_logits -- see
        that file's docstring."""
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
        h_t = h[:, pred_pos, :][:, :n_blocks * self.K, :]
        target = target_seq[:, :n_blocks * self.K]

        if self.token_head_type == "linears":
            logits, mask = self._token_logits_linears(h_t), None
        elif self.token_head_type == "ar":
            logits, mask = self._token_teacher_forced_ar(h_t, target), None
        else:
            assert rng is not None, "diffusion token head needs an rng even at eval (masking is inherent)"
            logits, mask = self._token_teacher_forced_diffusion(h_t, target, rng)
        mtp_loss = self._mtp_loss(h_t, target)
        return logits, target, mask, mtp_loss

    def decode(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, lag: int, rng=None) -> tuple:
        logits, target, mask, mtp_loss = self.decode_logits_and_target(target_seq, ctx_code_soft, lag, rng=rng)
        loss, acc = self._dec_loss_acc(logits, target, mask)
        return loss + self.mtp_weight * mtp_loss, acc

    def decode_generate(self, ctx_idx: jnp.ndarray, lag: int, greedy: bool = True,
                         temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = lag + 1
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        ctx_tok_g = jnp.swapaxes(ctx_tok.reshape(B, n_groups, G, D), 0, 1)

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

        def token_predict(h_pos, rng):
            if self.token_head_type == "linears":
                logits = self._token_logits_linears(h_pos)
                return sample_idx(logits, rng, greedy, temperature)
            elif self.token_head_type == "ar":
                return self._token_generate_ar(h_pos, rng, greedy, temperature)
            else:
                return self._token_generate_diffusion(h_pos, rng, greedy, temperature)

        def group_step(carry, group_codes):
            cache_k, cache_v, pos, rng = carry
            bos_in = jnp.broadcast_to(self.bos_embed, (B, 1, D))
            chunk = jnp.concatenate([group_codes, bos_in], axis=1)
            h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, pos)
            pos = pos + (G + 1)
            h = h_chunk[:, -1, :]
            val, rng = token_predict(h, rng)
            vals = [val]
            x_input = self._dec_embed_target(val)

            for _ in range(G * self.K - 1):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos = pos + 1
                val, rng = token_predict(h, rng)
                vals.append(val)
                x_input = self._dec_embed_target(val)

            _, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
            pos = pos + 1
            return (cache_k, cache_v, pos, rng), jnp.stack(vals, axis=1)

        cache_k0 = jnp.zeros((len(blocks), B, self.n_kv_heads, L_total, hd))
        cache_v0 = jnp.zeros_like(cache_k0)
        init_carry = (cache_k0, cache_v0, jnp.array(0), jax.random.PRNGKey(seed))

        @jax.jit
        def run_scan(carry, xs):
            return jax.lax.scan(group_step, carry, xs)

        _, vals_all = run_scan(init_carry, ctx_tok_g)
        vals_all = jnp.moveaxis(vals_all, 0, 1)
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        return out[:, :n_blocks * self.K]

    def decode_generate_mtp_no_verify(self, ctx_idx: jnp.ndarray, lag: int, greedy: bool = True,
                                       temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """chat 2026-09-12: true-MTP "no-verify" decode -- draws mtp_horizon positions per KV-
        cache step instead of one, directly accepting the draft with no check against the real
        sequential decode (see mtp_predict_no_verify's docstring). Only valid when
        self.mtp_horizon>1; falls back to plain decode_generate() otherwise. Structurally
        identical to decode_generate() except group_step advances mtp_horizon positions per
        outer step using the SAME cached hidden state (the K draws share one h -- genuinely
        parallel, not autoregressive)."""
        if self.mtp_horizon <= 1:
            return self.decode_generate(ctx_idx, lag, greedy, temperature, seed)
        K = self.mtp_horizon
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = lag + 1
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len
        ctx_tok = code_embed(ctx_idx, self.ctx_embed)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        ctx_tok_g = jnp.swapaxes(ctx_tok.reshape(B, n_groups, G, D), 0, 1)

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
            draft, rng = mtp_predict_no_verify_standalone(self, h, rng, greedy, temperature)  # (B,K,chunks)
            vals = [draft[:, k, :] for k in range(K)]
            # advance the real KV cache by feeding the DRAFTED values (no-verify -- accepted as-is)
            for k in range(K):
                x_input = self._dec_embed_target(vals[k])
                _, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos = pos + 1
            remaining = G * self.K - K
            for _ in range(remaining):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
                pos = pos + 1
                val, rng = token_predict_standalone(self, h, rng, greedy, temperature)
                vals.append(val)
                x_input = self._dec_embed_target(val)
            return (cache_k, cache_v, pos, rng), jnp.stack(vals, axis=1)

        cache_k0 = jnp.zeros((len(blocks), B, self.n_kv_heads, L_total, hd))
        cache_v0 = jnp.zeros_like(cache_k0)
        init_carry = (cache_k0, cache_v0, jnp.array(0), jax.random.PRNGKey(seed))

        @jax.jit
        def run_scan(carry, xs):
            return jax.lax.scan(group_step, carry, xs)

        _, vals_all = run_scan(init_carry, ctx_tok_g)
        vals_all = jnp.moveaxis(vals_all, 0, 1)
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        return out[:, :n_blocks * self.K]


def mtp_predict_no_verify_standalone(level: EncDecLevel, h_pos, rng, greedy, temperature):
    """Drafts K future groups directly off ONE hidden state, NO check against what the real
    sequential decode would have produced ("no-verify" -- a verified self-speculative mode
    isn't implemented yet). mtp_mode="parallel": K independent packed linear heads. "ar":
    OUTER causal chain -- but at GENERATION time there's no real future to teacher-force with,
    so each outer step feeds back its OWN sampled group (embedded via token_member_embed),
    chained causally, same nesting as training just without ground truth."""
    K, chunks, vocab = level.mtp_horizon, level.in_pq_chunks, level.in_code_vocab
    if level.mtp_mode == "parallel":
        logits = (h_pos @ level.mtp_out_head).reshape(*h_pos.shape[:-1], K, chunks, vocab)
        return sample_idx(logits, rng, greedy, temperature)
    lead = h_pos.shape[:-1]
    D = h_pos.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    ctx0 = (h_pos.reshape(N, D) @ level.mtp_in_proj)[:, None, :]
    collected = [ctx0]
    groups = []
    for k in range(K):
        seq_in = jnp.concatenate(collected, axis=1)
        h1 = seq_in + dense_self_attention(level.mtp_attn, level.mtp_norm1(seq_in), causal=True)
        outer_out = level.mtp_ln_f(h1)[:, -1, :]
        ctx_k = outer_out @ level.mtp_out_proj
        group_k, rng = level._token_generate_ar(ctx_k, rng, greedy, temperature)
        groups.append(group_k)
        if k < K - 1:
            collected.append(code_embed(group_k, level.token_member_embed)[:, None, :])
    idx = jnp.stack(groups, axis=1).reshape(*lead, K, chunks)
    return idx, rng


def token_predict_standalone(level: EncDecLevel, h_pos, rng, greedy, temperature):
    if level.token_head_type == "linears":
        logits = level._token_logits_linears(h_pos)
        return sample_idx(logits, rng, greedy, temperature)
    elif level.token_head_type == "ar":
        return level._token_generate_ar(h_pos, rng, greedy, temperature)
    else:
        return level._token_generate_diffusion(h_pos, rng, greedy, temperature)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HierEncDec(eqx.Module):
    levels: list
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        keys = jax.random.split(key, n)
        self.levels = [EncDecLevel(keys[i], cfg, level=i, has_decoder=(i < n - 1),
                                    weight_sharing=cfg.weight_sharing[i]) for i in range(n)]


def phase_forward(model: HierEncDec, flat_bytes: jnp.ndarray, phase: int, rng=None,
                   use_cascade=None) -> tuple:
    """Same cascade rollout sampler design as run_lagcodec_sampler.py -- see that file's module
    docstring. flat_bytes is always (B, n_positions, byte_group)."""
    levels = model.levels
    x = code_embed(flat_bytes, levels[0].own_input_embed)
    target = flat_bytes
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils = [], [], []
    # per-level: one rng for encode()'s optional gumbel noise, one for decode()'s optional
    # diffusion token-head masking (KIV -- see _token_teacher_forced_diffusion).
    level_rngs = [None] * (2 * phase) if rng is None else list(jax.random.split(rng, 2 * phase))
    for i in range(phase):
        out = levels[i].encode(x, target, rng=level_rngs[2 * i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        enc_losses.append(out["ntp_loss"])
        enc_accs.append(out["ntp_acc"])
        utils.append(out["util"])
        if i < phase - 1:
            x = code_embed(out["code_soft"], levels[i + 1].own_input_embed)
            target = out["code_idx"]

    dec_losses, dec_accs = [], []
    ctx = codes_soft[phase - 1]
    for i in range(phase - 1, -1, -1):
        dec_target = flat_bytes if i == 0 else codes[i - 1]
        dec_rng = level_rngs[2 * i + 1]
        logits, target_i, mask_i, mtp_loss_i = levels[i].decode_logits_and_target(
            dec_target, ctx, model.cfg.lag[i], rng=dec_rng)
        loss_i, acc_i = levels[i]._dec_loss_acc(logits, target_i, mask_i)
        loss_i = loss_i + levels[i].mtp_weight * mtp_loss_i
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if i > 0:
            real_ctx = codes_soft[i - 1]
            if use_cascade is None:
                ctx = real_ctx
            else:
                pseudo_ctx, _ = quantize_hard(logits)
                ctx = jnp.where(use_cascade, pseudo_ctx, real_ctx)

    dec_loss_total = jnp.mean(jnp.stack(dec_losses))
    byte_acc = dec_accs[-1]
    ntp_loss_total = jnp.mean(jnp.stack(enc_losses))
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total
    bpb = dec_loss_total / jnp.log(2.0)
    return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)))


def phase_trainable_filter(model: HierEncDec, phase: int):
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
    return jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, pytree)


def to_host(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(jax.device_get(x)) if eqx.is_array(x) else x, pytree)


def to_single_device(tree, device=None):
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


def _float_tuple_arg(s: str) -> tuple:
    return tuple(float(x) for x in s.split(","))


def _bool_tuple_arg(s: str) -> tuple:
    return tuple(x.strip().lower() != "false" for x in s.split(","))


def _str_tuple_arg(s: str) -> tuple:
    return tuple(x.strip() for x in s.split(","))


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
                  "gumbel_temperature", "gumbel_at_inference", "cascade_rollout_prob",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob",
                  "mtp_horizon", "mtp_mode", "mtp_weight", "traversal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--epochs_per_phase", type=_tuple_arg, default=(1000,),
                    help="epochs per phase -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase (length must equal n_phases)")
    p.add_argument("--no_curriculum", type=lambda x: x.lower() != "false", default=False,
                    help="chat 2026-09-12 -- skip the phase-by-phase curriculum entirely: train "
                         "ALL levels jointly from step 1 (curriculum_mode='no_freeze' still "
                         "required). Reuses the same phase loop with phase fixed at n_phases for "
                         "its only iteration; epochs_per_phase's single/last entry is used.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="sinkgd", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={"sinkhorn_iters": 1, "weight_decay": 0})
    p.add_argument("--grad_clip", type=lambda x: None if x.lower() == "none" else float(x), default=1.0,
                    help="global-norm gradient clip threshold, applied before the optimizer "
                         "update; 'none' disables it")
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
    p.add_argument("--mlp_mult", type=_tuple_arg, default=Config.mlp_mult)
    p.add_argument("--rope_base", type=_float_tuple_arg, default=Config.rope_base)
    p.add_argument("--ntp_weight", type=float, default=Config.ntp_weight)
    p.add_argument("--lag", type=_tuple_arg, default=Config.lag)
    p.add_argument("--weight_sharing", type=_bool_tuple_arg, default=Config.weight_sharing)
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--curriculum_mode", type=str, default=Config.curriculum_mode, choices=["freeze", "no_freeze"])
    p.add_argument("--quantize_mode", type=str, default=Config.quantize_mode, choices=["argmax", "gumbel"])
    p.add_argument("--gumbel_temperature", type=_float_tuple_arg, default=Config.gumbel_temperature)
    p.add_argument("--gumbel_at_inference", type=lambda x: x.lower() != "false", default=Config.gumbel_at_inference)
    p.add_argument("--cascade_rollout_prob", type=float, default=Config.cascade_rollout_prob)
    p.add_argument("--byte_group", type=int, default=Config.byte_group)
    # type=str (NOT _str_tuple_arg): argparse auto-applies `type=` to ANY string-valued default
    # (even one set via set_defaults from a config file) -- with _str_tuple_arg that silently
    # comma-splits a bare broadcast string like "linears" into a 1-tuple BEFORE Config ever sees
    # it, breaking Config's own bcast() (confirmed 2026-09-12). type=str is a no-op on a plain
    # string, so a config file's bare-string broadcast value survives untouched; a genuine CLI
    # multi-value string ("--token_head_type linears,ar") is NOT supported this way -- set a real
    # per-level tuple in a config file instead (tuples are never touched by this argparse quirk).
    p.add_argument("--token_head_type", type=str, default=Config.token_head_type)
    p.add_argument("--token_dim", type=_tuple_arg, default=Config.token_dim)
    p.add_argument("--token_n_heads", type=_tuple_arg, default=Config.token_n_heads)
    p.add_argument("--token_mask_prob", type=float, default=Config.token_mask_prob)
    p.add_argument("--mtp_horizon", type=_tuple_arg, default=Config.mtp_horizon)
    p.add_argument("--mtp_mode", type=str, default=Config.mtp_mode)   # see --token_head_type's note
    p.add_argument("--mtp_weight", type=float, default=Config.mtp_weight)
    p.add_argument("--traversal", type=str, default=Config.traversal, choices=["raster", "zorder"])
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
    n_positions = n_positions_of(cfg)
    pixel_order = pixel_order_for(cfg)

    # chat 2026-09-12: epochs_per_phase is now per-PHASE -- a bare int, or a length-1 tuple
    # (Config-file-literal path where it wasn't parsed through _tuple_arg's CLI string form)
    # broadcasts uniformly; otherwise its length must match n_phases exactly.
    if isinstance(args.epochs_per_phase, int):
        args.epochs_per_phase = (args.epochs_per_phase,) * n_phases
    elif len(args.epochs_per_phase) == 1:
        args.epochs_per_phase = args.epochs_per_phase * n_phases
    assert len(args.epochs_per_phase) == n_phases, \
        f"epochs_per_phase has {len(args.epochs_per_phase)} entries, need {n_phases} (one per phase)"

    (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size, n_devices,
                                shuffle=True, seed=args.seed, cfg=cfg)

    rng = jax.random.PRNGKey(args.seed)
    model = HierEncDec(rng, cfg)
    n_params = count_params(model)

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"n_levels={n_levels} n_phases={n_phases} n_positions={n_positions} "
           f"params={n_params / 1e6:.2f}M")

    compute_dtype = jnp.bfloat16 if cfg.precision == "bf16" else jnp.float32
    recon_prompt = train_np[:args.qual_gen_n]
    flat_prompt = jnp.array(images_to_positions(recon_prompt, cfg, pixel_order))
    gt_img = recon_prompt.astype(np.uint8)

    def run_gen_eval(eval_model, top: int, tag: str, include_reconstruct: bool = False) -> tuple:
        m = cast_pytree(eval_model, compute_dtype)
        x = code_embed(flat_prompt, m.levels[0].own_input_embed)
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
            recon = m.levels[0].decode_generate(codes[0], cfg.lag[0], greedy=True, seed=0)
            recon_acc = float(jnp.mean(recon == flat_prompt))
            recon_img = positions_to_image(np.asarray(recon), cfg, pixel_order)
            save_compare_grid(recon_img, gt_img, run_dir / f"samples_{tag}_reconstruct.png")

        cur_code = codes[top]
        for i in range(top, 0, -1):
            cur_code = m.levels[i].decode_generate(cur_code, cfg.lag[i], greedy=True, seed=0)
        cascade_recon = m.levels[0].decode_generate(cur_code, cfg.lag[0], greedy=True, seed=0)
        cascade_acc = float(jnp.mean(cascade_recon == flat_prompt))
        cascade_img = positions_to_image(np.asarray(cascade_recon), cfg, pixel_order)
        save_compare_grid(cascade_img, gt_img, run_dir / f"samples_{tag}_cascade.png")

        msg = f"[{tag}] top={top} CASCADE gen_byte_acc={cascade_acc:.4f}"
        rec = dict(tag=tag, gen_cascade_acc=cascade_acc)
        if include_reconstruct:
            msg += f" reconstruct gen_byte_acc={recon_acc:.4f}"
            rec["gen_recon_acc"] = recon_acc
        logger(msg, **rec)
        return recon_acc, cascade_acc

    step = 0
    phase_iter = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    for phase in phase_iter:
        filter_spec = phase_trainable_filter(model, phase)
        diff_model, static_model = eqx.partition(model, filter_spec)

        def loss_fn(diff_model, static_model, flat_bytes, rng, use_cascade, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            return phase_forward(m, flat_bytes, phase, rng=rng, use_cascade=use_cascade)

        lr_schedule = warmup_const_schedule(args.lr, args.warmup_steps)
        if args.optimizer == "sinkgd":
            optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
        else:
            optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay, **args.optimizer_kwargs)
        if args.grad_clip is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip), optimizer)
        opt_state = optimizer.init(diff_model)

        def train_step(diff_model, opt_state, rng, flat_bytes, static_model=static_model):
            rng, level_rng, cascade_rng = jax.random.split(rng, 3)
            use_cascade = jax.random.bernoulli(cascade_rng, p=cfg.cascade_rollout_prob)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                diff_model, static_model, flat_bytes, level_rng, use_cascade)
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
        phase_epochs = args.epochs_per_phase[phase - 1]
        logger(f"=== starting {active_desc} for {phase_epochs} epochs ===")
        pbar = tqdm(range(1, phase_epochs + 1), desc=active_desc, mininterval=10.0)
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

            if epoch % 10 == 0:
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
