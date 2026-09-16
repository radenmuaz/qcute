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

5. WEIGHT AVERAGING (chat 2026-09-14) -- fork of run_lagcodec_zorder.py. Two independent schemes,
   Config-orthogonal (--wa_mode, default "none"): both operate on the trainable diff_model params
   only (static_model, e.g. norm scales that aren't eqx arrays, is untouched) and are saved as a
   SEPARATE checkpoint file under run_dir/checkpoints/wa/, alongside (not replacing) the regular
   per-step model checkpoint.
     - "ema": one shadow copy, updated every --wa_every steps as
       ema = wa_ema_decay*ema + (1-wa_ema_decay)*current_params (Polyak/exponential averaging).
       Shadow is (re)initialized from the phase's starting params at the first update of each
       phase -- EMA does NOT carry smoothing across a phase boundary (the trainable-param SET
       itself can grow between phases under curriculum_mode, so restarting is the simplest
       correct behavior; document, don't silently carry stale/mismatched structure).
     - "wma": a FIFO stack of up to --wa_stack_size raw parameter snapshots, one pushed every
       --wa_every steps (oldest evicted once full). Once the stack is full (reaches
       wa_stack_size), a plain elementwise mean over every snapshot currently in it is computed
       and saved -- this fires again on every subsequent push, so it's a ROLLING average over the
       last wa_stack_size snapshots, not a one-shot.

6. STEP-BASED TRAINING LOOP (chat 2026-09-14) -- the nested epoch/step double loop is replaced by
   a single loop driven by GLOBAL step count. --epochs_per_phase is still given in epochs (user-
   facing unit) but is immediately converted to steps (epochs * steps_per_epoch) once
   steps_per_epoch is known; --gen_eval_every/--ckpt_every/--wa_every are ALL given directly in
   steps (not epochs) in this fork. The dataloader (BatchIterator) still cycles in full epochs
   internally -- each pass through it is one shuffled epoch -- but the outer loop, tqdm bar,
   logger, and every periodic trigger (gen-eval/checkpoint/WA) count against the flat global step,
   so a phase can end mid-epoch with no special-casing. Checkpoint resume tracks phase_step
   (steps completed within the CURRENT phase) instead of epoch for the same reason.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/<name>.py
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import tarfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from image_lagcodec.eqx_common import (Attention, Block, RMSNorm, apply_rope, apply_xsa, init_matrix,
                                        init_vector, make_lr_schedule, rmsnorm, rope_cos_sin, sinkgd,
                                        warmup_const_schedule)

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
    decoder_ncodes: tuple = 1   # chat 2026-09-13 -- renamed from lag=0 (an offset) to a direct
    # count of codes the decoder conditions on (1 = current position only, no lookback), avoiding
    # the G=lag+1 off-by-one. Per-level, bare-scalar default.
    weight_sharing: tuple = True   # chat 2026-09-12 -- now PER-LEVEL, bare-scalar default
    precision: str = "bf16"           # stays global -- whole-model compute dtype
    curriculum_mode: str = "freeze"   # stays global -- training-loop control
    quantize_mode: str = "argmax"     # stays global -- training-time quantization strategy
    quantize_drop: float = 0.0   # chat 2026-09-14 -- probability of skipping the straight-through
    # hard commit for a given position, using the raw soft distribution instead (softmax alone
    # under argmax mode, or the gumbel-noised softmax at gumbel_temperature under gumbel mode).
    # Stays global. Default 0.0 -- always hard-commit, existing behavior unchanged.
    gumbel_temperature: tuple = 1.0   # chat 2026-09-12 -- now PER-LEVEL, bare-scalar default
    gumbel_at_inference: bool = False
    cascade_rollout_prob: float = 0.5
    init_scheme: str = "llama"   # chat 2026-09-13 -- weight init: "llama" (N(0,0.02^2), residual-
    # output projections scaled by 1/sqrt(2*n_layers), see eqx_common.init_matrix) or "zero"
    # (Zhao et al. 2021 arXiv:2110.12661 ZerO init -- deterministic identity/Hadamard, residual-
    # output projections forced to literal zero). Stays global, not per-level.
    use_xsa: bool = False   # chat 2026-09-13 -- Exclusive Self-Attention (arXiv:2603.09078):
    # removes the attention output's projection onto the query's own value vector, applied right
    # after every attention call (see eqx_common.apply_xsa). Stays global; default off, enable
    # with True.
    use_qknorm: bool = True   # chat 2026-09-13 -- per-head RMSNorm on q/k before RoPE+scores.
    # Stays global; disable with False.

    remat: bool = False   # chat 2026-09-14 -- gradient rematerialization (jax.checkpoint) applied
    # per-block, every level's encoder AND decoder block stack -- trades recompute for activation
    # memory. Stays global (uniform on/off), default off.

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
    pq_dim: tuple = None   # chat 2026-09-14 -- per-level embedding width for code_embed_proj
    # (own_input_embed/ctx_embed/dec_target_embed -- the level's own PQ/byte codebook embedding,
    # concat-then-linear-map scheme). None (default) -> that level's own d_model. Set a per-level
    # tuple or a bare int (broadcasts uniformly) to use a narrower per-chunk width, e.g. 64 for a
    # byte level's 3 RGB chunks, 32 for a PQ level's 4 chunks -- cheaper than full d_model per
    # chunk since the down-projection still restores width D regardless.
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

    entropy_weight: float = 0.0   # chat 2026-09-13 -- IBQ-style (arXiv:2412.02692) entropy bonus
    # on the encoder's per-position code distribution, pushing codebook usage toward uniform
    # (combats index collapse -- see codebook_utilization, computed but otherwise unused). Default
    # 0.0 (off) -- opt-in, existing configs unaffected. Stays global, like ntp_weight/mtp_weight.

    traversal: str = "raster"    # "raster" (default, row-major) or "zorder" (Morton curve over
    # pixels, RGB stays contiguous per pixel regardless of byte_group -- see module docstring).

    mse_weight: float = 0.0   # chat 2026-09-14 -- level0's byte-decode logits, softmax'd (soft,
    # no straight-through -- chat 2026-09-15) then dotted with arange(256) to get a differentiable
    # pixel-value estimate, then MSE'd against the real byte value and normalized by /255 (squared)
    # so it's on a comparable [0,1] scale to the CE loss -- otherwise (confirmed 2026-09-14) raw
    # 0-255 pixel MSE totally dominates at mse_weight=1.0 (loss~9658 vs ~11.75 without it). Default 0.0.
    mse_softmax_tau: float = 0.1   # chat 2026-09-15 -- temperature on the soft byte distribution
    # used for mse_weight's pixel-value expectation (lower = sharper/closer to one-hot).

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
        bcast("decoder_ncodes", int)
        bcast("weight_sharing", bool)
        bcast("gumbel_temperature", (int, float))
        bcast("token_head_type", str)
        bcast("token_dim", int)
        bcast("token_n_heads", int)
        bcast("mtp_horizon", int)
        bcast("mtp_mode", str)
        if self.pq_dim is None:
            self.pq_dim = self.d_model
        else:
            bcast("pq_dim", int)

        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.decoder_ncodes) == n
        assert len(self.weight_sharing) == n and len(self.gumbel_temperature) == n
        assert len(self.token_head_type) == n
        assert len(self.pq_dim) == n
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
                max_horizon = self.decoder_ncodes[i] * stride_i
                assert self.mtp_horizon[i] <= max_horizon, \
                    f"level {i}: mtp_horizon={self.mtp_horizon[i]} exceeds decoder_ncodes*stride=" \
                    f"{self.decoder_ncodes[i]}*{stride_i}={max_horizon} -- can't predict further " \
                    "ahead than one decode block's own group of target codes"
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
        assert self.init_scheme in ("llama", "zero")
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
        tmp_path = tar_path.with_name(tar_path.name + ".tmp")
        print(f"downloading {CIFAR10_URL} -> {tar_path}")
        urllib.request.urlretrieve(CIFAR10_URL, tmp_path)   # download to temp first -- an
        tmp_path.rename(tar_path)   # interrupted/truncated download must never land at tar_path,
        # or it silently poisons every future run on this node (confirmed 2026-09-14: a truncated
        # tar.gz from an earlier interrupted download made tarfile/gzip EOFError on every relaunch
        # since `if not tar_path.exists()` skipped re-downloading the corrupt file).
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

def quantize_hard(logits: jnp.ndarray, rng=None, quantize_drop: float = 0.0, tau: float = 1.0) -> tuple:
    soft = jax.nn.softmax(logits / tau, axis=-1)
    idx = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=soft.dtype)
    st = soft + jax.lax.stop_gradient(hard - soft)
    if quantize_drop > 0 and rng is not None:
        drop = jax.random.bernoulli(rng, p=quantize_drop, shape=soft.shape[:-1])[..., None]
        code_soft = jnp.where(drop, soft, st)
    else:
        code_soft = st
    return code_soft, idx


def quantize_gumbel(logits: jnp.ndarray, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    rng, drop_rng = jax.random.split(rng)
    u = jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
    gumbel_noise = -jnp.log(-jnp.log(u))
    noisy_logits = (logits + gumbel_noise) / tau
    soft = jax.nn.softmax(noisy_logits, axis=-1)
    idx = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=soft.dtype)
    st = soft + jax.lax.stop_gradient(hard - soft)
    if quantize_drop > 0:
        drop = jax.random.bernoulli(drop_rng, p=quantize_drop, shape=soft.shape[:-1])[..., None]
        code_soft = jnp.where(drop, soft, st)
    else:
        code_soft = st
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
    """table: (vocab, D). code: (...,C) int indices or (...,C,vocab) soft, C=chunks (byte_group
    or pq_chunks -- this function is agnostic, used identically for both).

    CONCATENATIVE (chat 2026-09-14): each of the C chunk positions gets its OWN disjoint slice of
    the D-dim embedding (D split as evenly as possible across C slices) instead of all C chunks
    sharing the full D-dim row and being summed -- preserves per-chunk identity (e.g. byte_group=3
    R/G/B no longer alias through a shared sum, since summing let e.g. R=200,G=50,B=10 collide
    with other combinations that happen to sum close to the same vector). Reuses the SAME (vocab,
    D) table as before (no new params) -- just slices it differently per chunk index.
    """
    # --- additive (original) path, commented for easy revert ---
    # if jnp.issubdtype(code.dtype, jnp.integer):
    #     return table[code].sum(-2)
    # return (code @ table).sum(-2)
    D = table.shape[-1]
    is_int = jnp.issubdtype(code.dtype, jnp.integer)
    C = code.shape[-1] if is_int else code.shape[-2]
    bounds = [round(i * D / C) for i in range(C + 1)]
    parts = []
    for i in range(C):
        lo, hi = bounds[i], bounds[i + 1]
        if is_int:
            parts.append(table[code[..., i], lo:hi])
        else:
            parts.append(code[..., i, :] @ table[:, lo:hi])
    return jnp.concatenate(parts, axis=-1)


def code_embed_proj(code: jnp.ndarray, table: jnp.ndarray, proj: jnp.ndarray) -> jnp.ndarray:
    """chat 2026-09-14: concat-then-linear-map upgrade over code_embed's disjoint-slice concat --
    used for own_input_embed/ctx_embed/dec_target_embed (the level's own PQ/byte codebook
    embedding), NOT token_member_embed (still plain code_embed, unaffected).

    table: (vocab, pq_dim) -- each of the C chunks gets the FULL pq_dim width (no slicing).
    proj: (C*pq_dim, D). Mathematically decomposes as sum_c(embed_c(code_c) @ proj_c), proj_c
    being the c-th (pq_dim,D) block of proj -- a strict superset of both plain additive (proj_c=I
    padded) and disjoint-slice concat (proj_c a 0/1 selection matrix): the network can LEARN
    whichever combination is best, at the cost of C*pq_dim*D extra params vs a free gather+slice.
    Still no genuine cross-chunk interaction (this is still linear/additive across chunks)."""
    is_int = jnp.issubdtype(code.dtype, jnp.integer)
    C = code.shape[-1] if is_int else code.shape[-2]
    if is_int:
        parts = [table[code[..., i]] for i in range(C)]
    else:
        parts = [code[..., i, :] @ table for i in range(C)]
    return jnp.concatenate(parts, axis=-1) @ proj


def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


def sample_idx(logits: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
    if greedy:
        return jnp.argmax(logits, axis=-1), rng
    rng, k_ = jax.random.split(rng)
    return jax.random.categorical(k_, logits / temperature, axis=-1), rng


def run_block(blk: Block, x: jnp.ndarray, remat: bool) -> jnp.ndarray:
    """chat 2026-09-14: optional per-block gradient rematerialization (jax.checkpoint) -- only
    the block's OUTPUT is kept for backward, its internal activations (qkv, attention scores,
    mlp hidden) are recomputed instead of stored. Applied uniformly to every level's encoder and
    decoder block stacks when Config.remat=True (default off)."""
    return jax.checkpoint(blk)(x) if remat else blk(x)


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
    if attn.use_qknorm:
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
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)  # (B,H,T,hd)
    if attn.use_xsa:
        y = apply_xsa(y, v)
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
    own_input_proj: jnp.ndarray
    ntp_head: jnp.ndarray
    code_head: jnp.ndarray
    bos_embed: jnp.ndarray
    ctx_embed: jnp.ndarray
    ctx_proj: jnp.ndarray
    dec_blocks: list
    dec_ln_f: RMSNorm
    dec_target_embed: jnp.ndarray
    dec_target_proj: jnp.ndarray
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
    quantize_drop: float = eqx.field(static=True)
    gumbel_temperature: float = eqx.field(static=True)
    token_head_type: str = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    token_mask_prob: float = eqx.field(static=True)
    mtp_horizon: int = eqx.field(static=True)
    mtp_mode: str = eqx.field(static=True)
    mtp_weight: float = eqx.field(static=True)
    remat: bool = eqx.field(static=True)
    pq_dim: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, has_decoder: bool, weight_sharing: bool):
        D = cfg.d_model[level]
        self.K = cfg.strides[level] if cfg.strides[level] != -1 else 1
        self.n_heads, self.n_kv_heads = cfg.n_heads[level], cfg.n_kv_heads[level]
        self.quantize_mode = cfg.quantize_mode
        self.quantize_drop = cfg.quantize_drop
        self.remat = cfg.remat
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
        self.pq_dim = cfg.pq_dim[level]
        own_vocab = 256 if is_byte_level else self.in_code_vocab
        ntp_out = self.in_pq_chunks * self.in_code_vocab
        keys = jax.random.split(key, 23)

        scheme, use_xsa, use_qknorm = cfg.init_scheme, cfg.use_xsa, cfg.use_qknorm
        self.own_input_embed = init_matrix(keys[0], (own_vocab, self.pq_dim), scheme)
        self.own_input_proj = init_matrix(keys[20], (self.in_pq_chunks * self.pq_dim, D), scheme)
        n_layers = cfg.n_layers[level]
        block_keys = jax.random.split(keys[1], n_layers)
        self.blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level],
                             n_layers=n_layers, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm) for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.code_head = init_matrix(keys[2], (D, self.pq_chunks * self.code_vocab), scheme)
        self.ntp_head = init_matrix(keys[3], (D, ntp_out), scheme)
        self.bos_embed = init_vector(keys[4], D, scheme)
        self.ctx_embed = init_matrix(keys[5], (self.code_vocab, self.pq_dim), scheme)
        self.ctx_proj = init_matrix(keys[21], (self.pq_chunks * self.pq_dim, D), scheme)

        if has_decoder and not weight_sharing:
            dec_block_keys = jax.random.split(keys[6], n_layers)
            self.dec_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level],
                                     n_layers=n_layers, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
                               for k in dec_block_keys]
            self.dec_ln_f = RMSNorm(D)
            self.dec_target_embed = init_matrix(keys[7], (own_vocab, self.pq_dim), scheme)
            self.dec_target_proj = init_matrix(keys[22], (self.in_pq_chunks * self.pq_dim, D), scheme)
            self.dec_head = init_matrix(keys[8], (D, ntp_out), scheme)
        else:
            self.dec_blocks, self.dec_ln_f, self.dec_target_embed, self.dec_target_proj, self.dec_head = None, None, None, None, None

        if has_decoder and self.token_head_type in ("ar", "diffusion"):
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            self.token_in_proj = init_matrix(keys[9], (D, tdim), scheme)
            self.token_member_embed = init_matrix(keys[10], (self.in_code_vocab, tdim), scheme)
            self.token_norm1 = RMSNorm(tdim)
            self.token_attn = Attention(keys[11], tdim, theads, theads, cfg.rope_base[level], n_layers=1,
                                         init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
            self.token_ln_f = RMSNorm(tdim)
            self.token_out_head = init_matrix(keys[12], (tdim, self.in_code_vocab), scheme)
            if self.token_head_type == "diffusion":
                self.token_mask_embed = init_vector(keys[14], tdim, scheme)
                self.token_channel_embed = init_matrix(keys[15], (self.in_pq_chunks, tdim), scheme)
            else:
                self.token_mask_embed, self.token_channel_embed = None, None
        else:
            (self.token_in_proj, self.token_member_embed, self.token_mask_embed,
             self.token_channel_embed, self.token_norm1, self.token_attn, self.token_ln_f,
             self.token_out_head) = (None,) * 8

        (self.mtp_heads_in_proj, self.mtp_heads_member_embed, self.mtp_heads_norm1,
         self.mtp_heads_attn, self.mtp_heads_ln_f, self.mtp_heads_out_head) = (None,) * 6
        if has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "parallel" and self.token_head_type == "linears":
            self.mtp_out_head = init_matrix(keys[13], (D, self.mtp_horizon * ntp_out), scheme)
            (self.mtp_in_proj, self.mtp_attn, self.mtp_norm1, self.mtp_ln_f, self.mtp_out_proj) = (None,) * 5
        elif has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "parallel" and self.token_head_type == "ar":
            # "duplicate ar heads" (chat 2026-09-12): K FULLY INDEPENDENT copies of the token-ar
            # mechanism, one per future timestep, all applied to the SAME h_t -- no chaining.
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            head_keys = jax.random.split(keys[19], self.mtp_horizon * 3)
            self.mtp_heads_in_proj = [init_matrix(head_keys[3 * k], (D, tdim), scheme)
                                       for k in range(self.mtp_horizon)]
            self.mtp_heads_member_embed = [init_matrix(head_keys[3 * k + 1], (self.in_code_vocab, tdim), scheme)
                                            for k in range(self.mtp_horizon)]
            self.mtp_heads_attn = [Attention(head_keys[3 * k + 2], tdim, theads, theads, cfg.rope_base[level],
                                              n_layers=1, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
                                    for k in range(self.mtp_horizon)]
            self.mtp_heads_norm1 = [RMSNorm(tdim) for _ in range(self.mtp_horizon)]
            self.mtp_heads_ln_f = [RMSNorm(tdim) for _ in range(self.mtp_horizon)]
            self.mtp_heads_out_head = [init_matrix(k, (tdim, self.in_code_vocab), scheme)
                                        for k in jax.random.split(keys[19], self.mtp_horizon)]
            (self.mtp_out_head, self.mtp_in_proj, self.mtp_attn, self.mtp_norm1,
             self.mtp_ln_f, self.mtp_out_proj) = (None,) * 6
        elif has_decoder and self.mtp_horizon > 1 and self.mtp_mode == "ar":
            tdim, theads = cfg.token_dim[level], cfg.token_n_heads[level]
            self.mtp_in_proj = init_matrix(keys[16], (D, tdim), scheme)
            self.mtp_attn = Attention(keys[17], tdim, theads, theads, cfg.rope_base[level], n_layers=1,
                                       init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
            self.mtp_norm1 = RMSNorm(tdim)
            self.mtp_ln_f = RMSNorm(tdim)
            self.mtp_out_proj = init_matrix(keys[18], (tdim, D), scheme)
            self.mtp_out_head = None
        else:
            (self.mtp_out_head, self.mtp_in_proj, self.mtp_attn, self.mtp_norm1, self.mtp_ln_f,
             self.mtp_out_proj) = (None,) * 6

    # --- encoder role (mirrors HierEncoder.EncoderLevel.forward) ---

    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, rng=None) -> dict:
        h = x
        for blk in self.blocks:
            h = run_block(blk, h, self.remat)
        h = self.ln_f(h)
        M, L, D = h.shape
        n_blocks = L // self.K
        h_blocks = h[:, :n_blocks * self.K, :].reshape(M, n_blocks, self.K, D)
        pooled = h_blocks[:, :, self.K - 1, :]
        logits = reshape_pq(pooled @ self.code_head, self.pq_chunks, self.code_vocab)
        if rng is not None and self.quantize_mode == "gumbel":
            code_soft, code_idx = quantize_gumbel(logits, rng, self.gumbel_temperature, self.quantize_drop)
        else:
            code_soft, code_idx = quantize_hard(logits, rng, self.quantize_drop, self.gumbel_temperature)

        # IBQ-style (arXiv:2412.02692) entropy bonus: negative entropy of the batch-averaged soft
        # code distribution (differentiable, unlike codebook_utilization's hard-idx entropy used
        # only for logging) -- minimizing this maximizes codebook usage entropy, combating index
        # collapse. Config.entropy_weight (default 0.0) scales this in phase_forward.
        probs = jax.nn.softmax(logits, axis=-1)
        p_avg = jnp.mean(probs, axis=(0, 1))
        entropy_loss = jnp.mean(jnp.sum(p_avg * jnp.log(jnp.maximum(p_avg, 1e-9)), axis=-1))

        ntp_logits = reshape_pq(h[:, :-1, :] @ self.ntp_head, self.in_pq_chunks, self.in_code_vocab)
        tgt = target_idx[:, 1:]
        logp = jax.nn.log_softmax(ntp_logits, axis=-1)
        ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
        ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
        util = codebook_utilization(code_idx, self.code_vocab)
        return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                    entropy_loss=entropy_loss)

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
        proj = self.own_input_proj if self.weight_sharing else self.dec_target_proj
        return code_embed_proj(idx, table, proj)

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

    def decode_logits_and_target(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, decoder_ncodes: int,
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
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed_proj(ctx_code_soft, self.ctx_embed, self.ctx_proj)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
            te = jnp.pad(te, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        ctx_g = ctx_tok.reshape(B, n_groups, G, D)
        te_g = te.reshape(B, n_groups, G * self.K, D)
        bos_g = jnp.broadcast_to(self.bos_embed, (B, n_groups, 1, D))
        per_group_len = G + 1 + G * self.K
        xe = jnp.concatenate([ctx_g, bos_g, te_g], axis=2).reshape(B, n_groups * per_group_len, D)
        for blk in blocks:
            xe = run_block(blk, xe, self.remat)
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

    def decode(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, decoder_ncodes: int, rng=None) -> tuple:
        logits, target, mask, mtp_loss = self.decode_logits_and_target(target_seq, ctx_code_soft, decoder_ncodes, rng=rng)
        loss, acc = self._dec_loss_acc(logits, target, mask)
        return loss + self.mtp_weight * mtp_loss, acc

    def decode_generate(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                         temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len
        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)
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

    def decode_generate_mtp_no_verify(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                                       temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
        """chat 2026-09-12: true-MTP "no-verify" decode -- draws mtp_horizon positions per KV-
        cache step instead of one, directly accepting the draft with no check against the real
        sequential decode (see mtp_predict_no_verify's docstring). Only valid when
        self.mtp_horizon>1; falls back to plain decode_generate() otherwise. Structurally
        identical to decode_generate() except group_step advances mtp_horizon positions per
        outer step using the SAME cached hidden state (the K draws share one h -- genuinely
        parallel, not autoregressive)."""
        if self.mtp_horizon <= 1:
            return self.decode_generate(ctx_idx, decoder_ncodes, greedy, temperature, seed)
        K = self.mtp_horizon
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len
        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)
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
    x = code_embed_proj(flat_bytes, levels[0].own_input_embed, levels[0].own_input_proj)
    target = flat_bytes
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses = [], [], [], []
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
        entropy_losses.append(out["entropy_loss"])
        if i < phase - 1:
            x = code_embed_proj(out["code_soft"], levels[i + 1].own_input_embed, levels[i + 1].own_input_proj)
            target = out["code_idx"]

    dec_losses, dec_accs = [], []
    byte_mse = None
    ctx = codes_soft[phase - 1]
    for i in range(phase - 1, -1, -1):
        dec_target = flat_bytes if i == 0 else codes[i - 1]
        dec_rng = level_rngs[2 * i + 1]
        logits, target_i, mask_i, mtp_loss_i = levels[i].decode_logits_and_target(
            dec_target, ctx, model.cfg.decoder_ncodes[i], rng=dec_rng)
        loss_i, acc_i = levels[i]._dec_loss_acc(logits, target_i, mask_i)
        loss_i = loss_i + levels[i].mtp_weight * mtp_loss_i
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if i == 0:
            # level0's own logits predict real byte VALUES (0-255) -- pixel-space MSE only makes
            # sense here, not at levels>0 (those predict PQ codebook indices, not pixel values).
            pred_bytes = jnp.argmax(logits, axis=-1).astype(jnp.float32)
            byte_mse = jnp.mean((pred_bytes - target_i.astype(jnp.float32)) ** 2)   # logging only, non-diff
            if model.cfg.mse_weight > 0:
                # soft (non-ST) pixel-value expectation (chat 2026-09-15): softmax(logits/tau)
                # dotted with arange(256), no straight-through hard commit -- the FORWARD value is
                # the distribution's mean pixel value, not the argmax byte. mse_softmax_tau sharpens
                # (<1) or flattens (>1) the distribution before taking the expectation.
                byte_probs = jax.nn.softmax(logits / model.cfg.mse_softmax_tau, axis=-1)
                byte_values = jnp.arange(byte_probs.shape[-1], dtype=byte_probs.dtype)
                pred_pixel = jnp.sum(byte_probs * byte_values, axis=-1)
                max_val = byte_probs.shape[-1] - 1   # 255 for a byte -- normalizes MSE to [0,1]
                mse_loss = jnp.mean(((pred_pixel - target_i.astype(jnp.float32)) / max_val) ** 2)
            else:
                mse_loss = 0.0
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
    entropy_loss_total = jnp.mean(jnp.stack(entropy_losses))
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + model.cfg.mse_weight * mse_loss
    bpb = dec_loss_total / jnp.log(2.0)
    return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)), byte_mse)


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


# ---------------------------------------------------------------------------
# Full resumability (chat 2026-09-14): model + optimizer state + live RNG + dataloader shuffle
# state + phase/epoch/step position, so a killed run resumes bit-for-bit (same n_devices) rather
# than just reloading final weights. p_rng is the REPLICATED (n_devices,2) array saved as-is (not
# unreplicated) -- each device's row diverges after splits inside train_step, so only the full
# array reproduces every device's exact stream; opt_state/model ARE unreplicated first by the
# caller since pmean keeps every device's copy identical, so device 0 is fully representative.
# ---------------------------------------------------------------------------

def save_checkpoint(ckpt_dir: Path, model, opt_state, p_rng, train_iter: "BatchIterator",
                     phase: int, phase_step: int, step: int, seed: int) -> None:
    """chat 2026-09-14: epoch -> phase_step (steps completed within the CURRENT phase) -- this
    fork's training loop is step-driven, not epoch-driven (see module docstring point 6)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    eqx.tree_serialise_leaves(ckpt_dir / "p_rng.eqx", p_rng)
    (ckpt_dir / "dataloader_state.json").write_text(json.dumps(train_iter.rng.bit_generator.state))
    (ckpt_dir / "meta.json").write_text(json.dumps(dict(phase=phase, phase_step=phase_step, step=step, seed=seed)))


def find_latest_checkpoint(run_dir: Path):
    """Highest (phase, phase_step) checkpoint under run_dir/checkpoints, or None if there isn't
    one. Skips the wa/ subdirectory (weight-averaged snapshots, not resumable training state)."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    candidates = []
    for d in ckpt_root.iterdir():
        if d.name == "wa":
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            candidates.append((meta["phase"], meta["phase_step"], d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    """Deletes all but the keep most recent (phase, phase_step) checkpoints under
    run_dir/checkpoints (wa/ untouched). keep=None disables pruning entirely."""
    if keep is None:
        return
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return
    candidates = []
    for d in ckpt_root.iterdir():
        if d.name == "wa":
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            candidates.append((meta["phase"], meta["phase_step"], d))
    candidates.sort(key=lambda t: (t[0], t[1]))
    for _, _, d in candidates[:-keep] if keep > 0 else candidates:
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Weight averaging (chat 2026-09-14) -- see module docstring point 5. Both operate on diff_model
# (trainable params only) as plain pytrees; caller is responsible for combining with static_model
# and unreplicating/host-transferring before calling these.
# ---------------------------------------------------------------------------

def ema_update(ema_tree, new_tree, decay: float):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1 - decay) * p if eqx.is_array(e) else e, ema_tree, new_tree)


def stack_average(stack: list, weights=None):
    """Weighted elementwise mean over every snapshot currently in the FIFO stack (oldest first).
    weights=None (default): uniform (1/n each). A tuple of raw scores, one per stack slot, is
    softmax-normalized to sum to 1 -- so only relative magnitude matters, not absolute scale
    (e.g. a rising score sequence weights more-recent/more-converged snapshots higher)."""
    n = len(stack)
    if weights is None:
        w = [1.0 / n] * n
    else:
        assert len(weights) == n, f"wa_wma_weights has {len(weights)} entries, need {n} (wa_stack_size)"
        exp = [math.exp(x) for x in weights]
        total = sum(exp)
        w = [e / total for e in exp]
    return jax.tree_util.tree_map(
        lambda *xs: sum(wi * x for wi, x in zip(w, xs)) if eqx.is_array(xs[0]) else xs[0], *stack)


def pixel_mse(gen: np.ndarray, gt: np.ndarray) -> float:
    """Mean squared pixel error, in float64 to avoid uint8 overflow -- averaged over EVERY
    element (n*h*w*c), not summed. Verified (chat 2026-09-14): mean(per_image_mse) == this,
    confirming no aggregation blow-up."""
    return float(np.mean((gen.astype(np.float64) - gt.astype(np.float64)) ** 2))


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
        # tqdm.write(line)   # chat 2026-09-13: defaults to stdout, block-buffered when piped
        # through tee -- lags behind the bar (stderr, unbuffered), then dumps in a burst. Revert
        # to this line to undo.
        tqdm.write(line, file=sys.stderr)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        rec = {"elapsed_s": elapsed_s, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(_round_floats(rec)) + "\n")
        self.json_f.flush()


def _fmt_lr(lr: float) -> str:
    mantissa, exp = f"{lr:.3e}".split("e")
    return f"{mantissa}e{int(exp)}"


def _round_floats(obj, ndigits: int = 4):
    """Recursively truncates every float to ndigits decimal places (dict/list/tuple-aware) --
    chat 2026-09-14, the raw config/resolved_config dumps had long float repr noise (bf16/float32
    roundtrip artifacts like 0.10000000149011612) that made the log hard to read."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_round_floats(v, ndigits) for v in obj)
    return obj


def _pretty_dict(d: dict, per_line: int = 4) -> str:
    items = [f"{k}={v}" for k, v in d.items()]
    lines = ["\t".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\n" + "\n".join(lines)


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
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight", "decoder_ncodes",
                  "weight_sharing", "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_temperature", "gumbel_at_inference", "cascade_rollout_prob", "init_scheme", "use_xsa",
                  "use_qknorm", "remat",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "mtp_horizon", "mtp_mode", "mtp_weight", "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--batch_size", type=_tuple_arg, default=(16,),
                    help="training batch size -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase (length must equal n_phases)")
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
    p.add_argument("--lr_schedule", type=str, default="const", choices=["const", "cosine"],
                    help="const: warmup then flat forever (default). cosine: warmup then cosine "
                         "decay to 0 over this phase's own epoch_count*steps_per_epoch")
    p.add_argument("--lr_min", type=float, default=0.0,
                    help="cosine only: lr floor the decay reaches (default 0)")
    p.add_argument("--lr_min_epoch", type=float, default=None,
                    help="cosine only: epoch (within this phase) at which lr_min is reached; lr "
                         "holds flat at lr_min for the rest of the phase. Default: reach lr_min "
                         "exactly at phase end (old behavior)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="sinkgd", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={"sinkhorn_iters": 1, "weight_decay": 0})
    p.add_argument("--grad_clip", type=lambda x: None if x.lower() == "none" else float(x), default=1.0,
                    help="global-norm gradient clip threshold, applied before the optimizer "
                         "update; 'none' disables it")
    p.add_argument("--log_every", type=int, default=10, help="in steps")
    p.add_argument("--gen_eval_every", type=int, default=10, help="mid-phase gen-eval cadence in EPOCHS -- auto-converted to steps via batch_size")
    p.add_argument("--ckpt_every", type=int, default=10,
                    help="save a full resumable checkpoint (model+optim+rng+dataloader state) "
                         "every N EPOCHS, in addition to always at phase end -- auto-converted to steps")
    p.add_argument("--ckpt_keep", type=lambda x: None if x.lower() == "none" else int(x), default=None,
                    help="keep only the N most recent checkpoints under checkpoints/ (wa/ "
                         "untouched), deleting older ones after each save. 'none' (default) "
                         "disables pruning -- keep every checkpoint")
    p.add_argument("--resume", type=lambda x: x.lower() != "false", default=False,
                    help="resume from the latest checkpoint under this run's log dir, if any")
    p.add_argument("--wa_mode", type=str, default="none", choices=["none", "ema", "wma"],
                    help="weight averaging: 'ema' (Polyak shadow copy) or 'wma' (rolling "
                         "mean over a FIFO stack of raw snapshots). 'none' (default) disables "
                         "both -- see module docstring point 5")
    p.add_argument("--wa_every", type=int, default=10, help="WA update cadence, in EPOCHS -- auto-converted to steps")
    p.add_argument("--wa_ema_decay", type=float, default=0.999, help="ema mode only")
    p.add_argument("--wa_stack_size", type=int, default=3, help="wma mode only")
    p.add_argument("--wa_wma_weights", type=_float_tuple_arg, default=None,
                    help="wma mode only: one raw score per stack slot (oldest first), "
                         "softmax-normalized to sum to 1 -- length must equal wa_stack_size. "
                         "Default None: uniform (1/wa_stack_size each)")
    p.add_argument("--train_subset_n", type=int, default=100)
    p.add_argument("--qual_gen_n", type=int, default=8)
    p.add_argument("--val_batch_size", type=_tuple_arg, default=(2,),
                    help="how many train-set images run_gen_eval reconstructs/generates from -- "
                         "kept small (default 2) since decode_generate is far more memory-heavy "
                         "per-example than a teacher-forced training step. Bare int applies "
                         "uniformly to every phase; a tuple gives one value per phase")
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
    p.add_argument("--decoder_ncodes", type=_tuple_arg, default=Config.decoder_ncodes)
    p.add_argument("--weight_sharing", type=_bool_tuple_arg, default=Config.weight_sharing)
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--curriculum_mode", type=str, default=Config.curriculum_mode, choices=["freeze", "no_freeze"])
    p.add_argument("--quantize_mode", type=str, default=Config.quantize_mode, choices=["argmax", "gumbel"])
    p.add_argument("--quantize_drop", type=float, default=Config.quantize_drop)
    p.add_argument("--gumbel_temperature", type=_float_tuple_arg, default=Config.gumbel_temperature)
    p.add_argument("--gumbel_at_inference", type=lambda x: x.lower() != "false", default=Config.gumbel_at_inference)
    p.add_argument("--cascade_rollout_prob", type=float, default=Config.cascade_rollout_prob)
    p.add_argument("--init_scheme", type=str, default=Config.init_scheme, choices=["llama", "zero"])
    p.add_argument("--use_xsa", type=lambda x: x.lower() != "false", default=Config.use_xsa)
    p.add_argument("--use_qknorm", type=lambda x: x.lower() != "false", default=Config.use_qknorm)
    p.add_argument("--remat", type=lambda x: x.lower() != "false", default=Config.remat)
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
    p.add_argument("--pq_dim", type=_tuple_arg, default=Config.pq_dim)
    p.add_argument("--token_mask_prob", type=float, default=Config.token_mask_prob)
    p.add_argument("--mtp_horizon", type=_tuple_arg, default=Config.mtp_horizon)
    p.add_argument("--mtp_mode", type=str, default=Config.mtp_mode)   # see --token_head_type's note
    p.add_argument("--mtp_weight", type=float, default=Config.mtp_weight)
    p.add_argument("--entropy_weight", type=float, default=Config.entropy_weight)
    p.add_argument("--mse_weight", type=float, default=Config.mse_weight)
    p.add_argument("--mse_softmax_tau", type=float, default=Config.mse_softmax_tau)
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

    # chat 2026-09-12/14: epochs_per_phase/batch_size/val_batch_size are per-PHASE -- a bare int,
    # or a length-1 tuple (Config-file-literal path where it wasn't parsed through _tuple_arg's
    # CLI string form) broadcasts uniformly; otherwise its length must match n_phases exactly.
    def _bcast_per_phase(name):
        val = getattr(args, name)
        if isinstance(val, int):
            val = (val,) * n_phases
        elif len(val) == 1:
            val = val * n_phases
        assert len(val) == n_phases, f"{name} has {len(val)} entries, need {n_phases} (one per phase)"
        setattr(args, name, val)

    _bcast_per_phase("epochs_per_phase")
    _bcast_per_phase("batch_size")
    _bcast_per_phase("val_batch_size")

    (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    # train_iter is rebuilt fresh each phase (see phase loop below) since batch_size is now
    # per-phase -- no single shared BatchIterator here anymore.

    rng = jax.random.PRNGKey(args.seed)
    model = HierEncDec(rng, cfg)
    n_params = count_params(model)

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"n_levels={n_levels} n_phases={n_phases} n_positions={n_positions} "
           f"params={n_params / 1e6:.2f}M")
    resolved = {k: v for k, v in sorted(vars(args).items()) if k != "config"}
    logger(f"resolved_config:{_pretty_dict(_round_floats(resolved))}")

    resume_meta, resume_ckpt_dir = None, None
    if args.resume:
        resume_ckpt_dir = find_latest_checkpoint(run_dir)
        if resume_ckpt_dir is not None:
            resume_meta = json.loads((resume_ckpt_dir / "meta.json").read_text())
            model = eqx.tree_deserialise_leaves(resume_ckpt_dir / "model.eqx", model)
            logger(f"resuming from {resume_ckpt_dir}: phase={resume_meta['phase']} "
                   f"phase_step={resume_meta['phase_step']} step={resume_meta['step']}")
        else:
            logger("--resume set but no checkpoint found under this run_dir -- starting fresh")

    compute_dtype = jnp.bfloat16 if cfg.precision == "bf16" else jnp.float32
    # recon_prompt/flat_prompt/gt_img are rebuilt fresh each phase (see phase loop below) since
    # val_batch_size is now per-phase -- run_gen_eval() (defined once, below) closes over these
    # as free variables and picks up whatever they're reassigned to at call time.
    recon_prompt = flat_prompt = gt_img = None

    def run_gen_eval(eval_model, top: int, tag: str, include_reconstruct: bool = False) -> tuple:
        gen_t0 = time.monotonic()
        m = cast_pytree(eval_model, compute_dtype)
        x = code_embed_proj(flat_prompt, m.levels[0].own_input_embed, m.levels[0].own_input_proj)
        target = flat_prompt
        codes, codes_soft = [], []
        eval_rngs = ([None] * (top + 1) if not cfg.gumbel_at_inference
                     else list(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), hash(tag) % (2**31)), top + 1)))
        for i in range(top + 1):
            out = m.levels[i].encode(x, target, rng=eval_rngs[i])
            codes.append(out["code_idx"])
            codes_soft.append(out["code_soft"])
            if i < top:
                x = code_embed_proj(out["code_soft"], m.levels[i + 1].own_input_embed, m.levels[i + 1].own_input_proj)
                target = out["code_idx"]

        recon_acc = recon_mse = None
        # if include_reconstruct:
        #     recon = m.levels[0].decode_generate(codes[0], cfg.decoder_ncodes[0], greedy=True, seed=0)
        #     recon_acc = float(jnp.mean(recon == flat_prompt))
        #     recon_img = positions_to_image(np.asarray(recon), cfg, pixel_order)
        #     recon_mse = pixel_mse(recon_img, gt_img)
        #     save_compare_grid(recon_img, gt_img, run_dir / f"samples_{tag}_reconstruct.png")

        cur_code = codes[top]
        for i in range(top, 0, -1):
            cur_code = m.levels[i].decode_generate(cur_code, cfg.decoder_ncodes[i], greedy=True, seed=0)
        cascade_recon = m.levels[0].decode_generate(cur_code, cfg.decoder_ncodes[0], greedy=True, seed=0)
        cascade_acc = float(jnp.mean(cascade_recon == flat_prompt))
        cascade_img = positions_to_image(np.asarray(cascade_recon), cfg, pixel_order)
        cascade_mse = pixel_mse(cascade_img, gt_img)
        save_compare_grid(cascade_img, gt_img, run_dir / f"samples_{tag}_cascade.png")

        gen_time_s = time.monotonic() - gen_t0
        msg = f"[{tag}] top={top} CASCADE gen_byte_acc={cascade_acc:.4f} gen_cascade_mse={cascade_mse:.2f}"
        rec = dict(tag=tag, gen_cascade_acc=cascade_acc, gen_cascade_mse=cascade_mse, gen_time_s=gen_time_s)
        # if include_reconstruct:
        #     msg += f" reconstruct gen_byte_acc={recon_acc:.4f} gen_recon_mse={recon_mse:.2f}"
        #     rec["gen_recon_acc"] = recon_acc
        #     rec["gen_recon_mse"] = recon_mse
        msg += f" gen_time={gen_time_s:.1f}s"
        logger(msg, **rec)
        return recon_acc, cascade_acc

    def run_val_eval(eval_model, phase: int, tag: str) -> tuple:
        """chat 2026-09-15 -- teacher-forced VALIDATION loss/acc: a plain FORWARD PASS only (no
        grad), same phase_forward used by train_step's loss_fn, but on flat_prompt (the held-out
        val batch already prepared for run_gen_eval) instead of a train batch. use_cascade left
        at its default (None -> always real ctx, never the cascade-rollout substitution) for a
        clean, deterministic signal -- NOT directly comparable to train's own logged loss/acc,
        which uses a stochastic use_cascade draw per cfg.cascade_rollout_prob."""
        val_t0 = time.monotonic()
        m = cast_pytree(eval_model, compute_dtype)
        loss, aux = phase_forward(m, flat_prompt, phase, rng=None)
        bpb, acc, ntp_bpb, ntp_acc, util, val_mse = [float(a) for a in aux]
        loss = float(loss)   # forces device sync -- val_time_s below includes the full forward pass
        val_time_s = time.monotonic() - val_t0
        logger(f"[{tag}] VAL loss={loss:.2f} val_dec_acc={acc:.2f} val_ntp_acc={ntp_acc:.2f} "
               f"val_time={val_time_s:.1f}s",
               tag=tag, val_loss=loss, val_dec_acc=acc, val_dec_bpb=bpb,
               val_ntp_acc=ntp_acc, val_ntp_bpb=ntp_bpb, val_util=util, val_mse=val_mse,
               val_time_s=val_time_s)
        return loss, acc

    if args.wa_mode == "wma" and args.wa_wma_weights is not None:
        assert len(args.wa_wma_weights) == args.wa_stack_size, \
            f"wa_wma_weights has {len(args.wa_wma_weights)} entries, need " \
            f"wa_stack_size={args.wa_stack_size}"

    step = resume_meta["step"] if resume_meta else 0
    all_phases = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    total_all_steps = sum(
        args.epochs_per_phase[p - 1] * (len(train_np) // (args.batch_size[p - 1] * n_devices))
        for p in all_phases)
    global_pbar = tqdm(total=total_all_steps, initial=step, desc="total", dynamic_ncols=True, position=1, leave=True)
    last_global_step = step
    phase_iter = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    if resume_meta is not None:
        resume_phase = resume_meta["phase"]
        # cheap throwaway BatchIterator just to learn that phase's steps_per_epoch (no data copy,
        # just wraps the numpy arrays) -- needed to tell whether the checkpointed phase_step means
        # "phase complete" (next phase starts fresh) or "mid-phase" (resume within resume_phase).
        steps_per_epoch_resume = len(BatchIterator(
            train_np, train_labels[:len(train_np)], args.batch_size[resume_phase - 1], n_devices,
            shuffle=True, seed=args.seed, cfg=cfg))
        phase_steps_resume = args.epochs_per_phase[resume_phase - 1] * steps_per_epoch_resume
        phase_complete = resume_meta["phase_step"] >= phase_steps_resume
        phase_iter = [p for p in phase_iter if p > resume_phase] if phase_complete \
            else [p for p in phase_iter if p >= resume_phase]
    for phase in phase_iter:
        train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size[phase - 1],
                                    n_devices, shuffle=True, seed=args.seed, cfg=cfg)
        recon_prompt = val_np[:args.val_batch_size[phase - 1]]   # held-out, not train_np -- see
        # chat 2026-09-14: run_gen_eval's gen_recon_mse/gen_cascade_mse are genuine validation
        # metrics now, not train-set metrics.
        flat_prompt = jnp.array(images_to_positions(recon_prompt, cfg, pixel_order))
        gt_img = recon_prompt.astype(np.uint8)

        filter_spec = phase_trainable_filter(model, phase)
        diff_model, static_model = eqx.partition(model, filter_spec)

        def loss_fn(diff_model, static_model, flat_bytes, rng, use_cascade, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            return phase_forward(m, flat_bytes, phase, rng=rng, use_cascade=use_cascade)

        phase_epochs = args.epochs_per_phase[phase - 1]
        steps_per_epoch_lr = len(train_iter)
        total_steps = phase_epochs * steps_per_epoch_lr
        min_epoch = args.lr_min_epoch if args.lr_min_epoch is not None else phase_epochs
        lr_decay_steps = max(1, round(min_epoch * steps_per_epoch_lr) - args.warmup_steps)
        lr_schedule = make_lr_schedule(args.lr_schedule, args.lr, args.warmup_steps, total_steps,
                                        end_value=args.lr_min, decay_steps=lr_decay_steps)
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
            grad_norm = optax.global_norm(grads)   # before clip -- shows when/how hard grad_clip fires
            aux = aux + (grad_norm,)
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            return diff_model, opt_state, rng, loss, aux

        train_step = jax.pmap(train_step, axis_name="d")
        p_diff_model = replicate(diff_model, n_devices)
        p_opt_state = replicate(opt_state, n_devices)
        p_rng = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), phase), n_devices)

        phase_steps = phase_epochs * len(train_iter)   # user gave epochs -- converted to steps once
        start_phase_step = 0
        if resume_meta is not None and phase == resume_meta["phase"]:
            # diff_model/p_diff_model already reflect the resumed weights (model was
            # deserialised from checkpoint before the phase loop) -- only optimizer state, RNG,
            # and the dataloader's shuffle stream need restoring here.
            p_opt_state = replicate(
                eqx.tree_deserialise_leaves(resume_ckpt_dir / "opt_state.eqx", opt_state), n_devices)
            p_rng = eqx.tree_deserialise_leaves(resume_ckpt_dir / "p_rng.eqx", p_rng)
            train_iter.rng.bit_generator.state = json.loads(
                (resume_ckpt_dir / "dataloader_state.json").read_text())
            start_phase_step = resume_meta["phase_step"]
            logger(f"resumed phase {phase}: optimizer/rng/dataloader state restored, "
                   f"continuing from phase_step {start_phase_step}")

        trained_desc = f"levels0-{phase - 1}" if cfg.curriculum_mode == "no_freeze" else f"level{phase - 1}"
        active_desc = f"phase{phase}[{trained_desc}]"
        logger(f"=== starting {active_desc} for {phase_epochs} epochs ({phase_steps} steps) ===")

        steps_per_epoch = len(train_iter)
        gen_eval_every_steps = args.gen_eval_every * steps_per_epoch
        ckpt_every_steps = args.ckpt_every * steps_per_epoch
        wa_every_steps = args.wa_every * steps_per_epoch

        wa_ema = None
        wa_stack = deque(maxlen=args.wa_stack_size)
        wa_dir = run_dir / "checkpoints" / "wa"

        pbar = tqdm(total=phase_steps, initial=start_phase_step, desc=active_desc, dynamic_ncols=True, position=0)
        jit_timed = False
        phase_step = start_phase_step
        epoch_num = start_phase_step // steps_per_epoch
        while phase_step < phase_steps:
            # one full pass through train_iter = one shuffled epoch (BatchIterator.__iter__
            # reshuffles via its own persistent rng each call) -- the dataloader still cycles in
            # full epochs; only the OUTER bookkeeping (pbar/logger/triggers) is step-based.
            epoch_num += 1
            logger(f"{active_desc}: epoch {epoch_num} (step {step})")
            global_pbar.update(step - last_global_step)
            last_global_step = step
            for flat in train_iter:
                if phase_step >= phase_steps:
                    break
                flat = jnp.array(flat)
                if not jit_timed:
                    jit_t0 = time.monotonic()
                p_diff_model, p_opt_state, p_rng, loss, aux = train_step(p_diff_model, p_opt_state, p_rng, flat)
                step += 1
                phase_step += 1
                pbar.update(1)
                loss0 = float(loss[0])   # forces device sync -- first call includes jit compile
                if not jit_timed:
                    logger(f"{active_desc}: first train_step (incl. jit compile) took "
                           f"{time.monotonic() - jit_t0:.1f}s")
                    jit_timed = True
                bpb, acc, ntp_bpb, ntp_acc, util, train_mse, grad_norm = [float(a[0]) for a in aux]
                lr = float(lr_schedule(step - 1))
                lr_str = _fmt_lr(lr)
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}",
                                  acc=f"{acc:.2f}",
                                #   ntp_acc=f"{ntp_acc:.2f}",
                                  lr=lr_str, gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    logger(f"\n"
                           f"[p={phase} s={step}] loss={loss0:.2f} dec_acc={acc:.2f} "
                           f"ntp_acc={ntp_acc:.2f} util={util:.2f} train_mse={train_mse:.1f} "
                           f"lr={lr_str} grad_norm={grad_norm:.2f}",
                           phase=phase, step=step, loss=loss0, dec_bpb=bpb,
                           dec_acc=acc, ntp_bpb=ntp_bpb, ntp_acc=ntp_acc, util=util,
                           train_mse=train_mse, lr=lr, grad_norm=grad_norm)

                if step % gen_eval_every_steps == 0:
                    snapshot = eqx.combine(to_single_device(unreplicate(p_diff_model)), static_model)
                    run_val_eval(snapshot, phase, tag=f"phase{phase}_step{step}")
                    run_gen_eval(snapshot, top=phase - 1, tag=f"phase{phase}_step{step}", include_reconstruct=True)

                if step % ckpt_every_steps == 0:
                    ckpt_model = eqx.combine(to_host(unreplicate(p_diff_model)), static_model)
                    ckpt_opt_state = to_host(unreplicate(p_opt_state))
                    ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
                    save_checkpoint(ckpt_dir, ckpt_model, ckpt_opt_state, to_host(p_rng), train_iter,
                                     phase=phase, phase_step=phase_step, step=step, seed=args.seed)
                    prune_checkpoints(run_dir, args.ckpt_keep)
                    logger(f"checkpoint saved: {ckpt_dir}")

                if args.wa_mode != "none" and step % wa_every_steps == 0:
                    cur_diff_model = to_host(unreplicate(p_diff_model))
                    wa_dir.mkdir(parents=True, exist_ok=True)
                    if args.wa_mode == "ema":
                        wa_ema = cur_diff_model if wa_ema is None else \
                            ema_update(wa_ema, cur_diff_model, args.wa_ema_decay)
                        eqx.tree_serialise_leaves(wa_dir / f"ema_phase{phase}_step{step}.eqx", wa_ema)
                        logger(f"wa (ema) snapshot saved at step {step}")
                    else:
                        wa_stack.append(cur_diff_model)
                        if len(wa_stack) == args.wa_stack_size:
                            avg = stack_average(list(wa_stack), weights=args.wa_wma_weights)
                            eqx.tree_serialise_leaves(wa_dir / f"wma_phase{phase}_step{step}.eqx", avg)
                            logger(f"wa (wma, n={len(wa_stack)}) average saved at step {step}")

        diff_model = to_host(unreplicate(p_diff_model))
        model = eqx.combine(diff_model, static_model)
        freeze_msg = "no freeze (no_freeze mode)" if cfg.curriculum_mode == "no_freeze" else f"freezing level {phase - 1}"
        logger(f"=== {active_desc} done, {freeze_msg} ===")
        ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
        save_checkpoint(ckpt_dir, model, to_host(unreplicate(p_opt_state)), to_host(p_rng), train_iter,
                         phase=phase, phase_step=phase_steps, step=step, seed=args.seed)
        prune_checkpoints(run_dir, args.ckpt_keep)
        run_val_eval(model, phase, tag=f"phase{phase}_final")
        run_gen_eval(model, top=phase - 1, tag=f"phase{phase}_final", include_reconstruct=True)

    global_pbar.update(step - last_global_step)
    global_pbar.close()
    logger("=== all phases done, running final top-down cascade eval ===")
    run_val_eval(model, n_levels - 1, tag="final")
    run_gen_eval(model, top=n_levels - 2, tag="final", include_reconstruct=True)
    logger("training done")


if __name__ == "__main__":
    main()
