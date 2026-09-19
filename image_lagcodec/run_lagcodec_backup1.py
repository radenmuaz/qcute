from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import tarfile
import time
import warnings
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
                                        init_vector, make_lr_schedule, rmsnorm, rope_cos_sin,
                                        rope_cos_sin_pos, rotate_half, sinkgd, warmup_const_schedule)

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
def total_bytes_of(cfg) -> int:
    return cfg.img_size * cfg.img_size * 3


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
    mlp_mult: tuple = 2
    rope_base: tuple = 10000.0
    ntp_weight: float = 1.0
    decoder_ncodes: tuple = 1
    ncodes_window: tuple = 0
    streaming: tuple = True
    decode_past: tuple = 0
    decode_future: tuple = 0
    sync: tuple = False
    n_refine_passes: tuple = 1
    refine_window: tuple = 0
    weight_sharing: tuple = True
    precision: str = "bf16"
    curriculum_mode: str = "freeze"
    quantize_mode: str = "argmax"
    quantize_drop: float = 0.0
    gumbel_at_inference: bool = False
    init_scheme: str = "llama"
    use_xsa: bool = False
    use_qknorm: bool = True

    attn_window: tuple = -1
    attn_lookahead: tuple = 0
    use_sink: bool = False

    remat: bool = False

    byte_group: int = 1
    token_head_type: tuple = "linears"
    token_dim: tuple = 64
    token_n_heads: tuple = 4
    pq_dim: tuple = None
    token_mask_prob: float = 0.15

    mtp_horizon: tuple = 1
    mtp_mode: tuple = "parallel"
    mtp_weight: float = 0.1

    entropy_weight: float = 0.0

    traversal: str = "raster"

    mse_weight: float = 0.0
    mse_softmax_tau: float = 1.0

    label_reg_weight: float = 0.0

    multipass_detach: bool = True
    refine_gumbel: bool = False
    refine_temperature: float = 1.0
    refine_quantize_drop: float = 0.0
    refine_remat: bool = None

    def __post_init__(self):
        n = len(self.strides)

        def bcast(name, types):
            val = getattr(self, name)
            if isinstance(val, types):
                setattr(self, name, (val,) * n)

        bcast("mlp_mult", int)
        bcast("rope_base", (int, float))
        bcast("decoder_ncodes", int)
        bcast("ncodes_window", int)
        bcast("streaming", bool)
        bcast("decode_past", int)
        bcast("decode_future", int)
        bcast("sync", bool)
        bcast("n_refine_passes", int)
        bcast("refine_window", int)
        bcast("weight_sharing", bool)
        bcast("token_head_type", str)
        bcast("token_dim", int)
        bcast("token_n_heads", int)
        bcast("mtp_horizon", int)
        bcast("mtp_mode", str)
        bcast("attn_window", int)
        bcast("attn_lookahead", int)
        if self.pq_dim is None:
            self.pq_dim = self.d_model
        else:
            bcast("pq_dim", int)

        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.decoder_ncodes) == n
        assert len(self.ncodes_window) == n and len(self.streaming) == n
        assert len(self.decode_past) == n and len(self.decode_future) == n and len(self.sync) == n
        assert len(self.n_refine_passes) == n and len(self.refine_window) == n
        for i in range(n):
            assert self.ncodes_window[i] >= -1, \
                f"level {i}: ncodes_window={self.ncodes_window[i]} must be -1 (all) or >=0 " \
                f"(disjoint at 0, bounded lookback above)"
            if not self.streaming[i] and self.ncodes_window[i] > 0:
                raise NotImplementedError(
                    f"level {i}: streaming=False with bounded ncodes_window={self.ncodes_window[i]} "
                    f"(symmetric/bidirectional local window) is not implemented -- only ncodes_window="
                    f"-1 (true fullctx) or 0 (disjoint, streaming moot) are supported under streaming=False")
            assert self.decode_past[i] >= 0 and self.decode_future[i] >= 0, \
                f"level {i}: decode_past={self.decode_past[i]}/decode_future={self.decode_future[i]} must be >=0"
            assert self.n_refine_passes[i] >= 1, \
                f"level {i}: n_refine_passes={self.n_refine_passes[i]} must be >=1 (1=off, current behavior)"
            assert self.refine_window[i] >= 0, \
                f"level {i}: refine_window={self.refine_window[i]} must be >=0"
            if self.sync[i]:
                raise NotImplementedError(
                    f"level {i}: sync=True is a stub (TODO) -- real cross-group pipelining "
                    f"(sequential group scan with a shared/growing cache, so a later group can "
                    f"read an earlier group's ACTUAL decode_future output instead of drafting its "
                    f"own private guess) is not implemented yet. Async mode (sync=False, default) "
                    f"already supports decode_past/decode_future at TRAINING time (both are real, "
                    f"teacher-forced); at GENERATION time only decode_past is used (the group's own "
                    f"private redecode of past content, discarded after conditioning) -- "
                    f"decode_future is a training-only regularizer for now and is skipped entirely "
                    f"during decode_generate_pardec, regardless of its value, until sync=True lands")
        assert len(self.attn_window) == n and len(self.attn_lookahead) == n
        for i in range(n):
            assert self.attn_window[i] == -1 or self.attn_window[i] >= 1, \
                f"level {i}: attn_window={self.attn_window[i]} must be -1 (unbounded/flash) or >=1 (splash LocalMask)"
            assert self.attn_lookahead[i] >= 0, \
                f"level {i}: attn_lookahead={self.attn_lookahead[i]} must be >=0 (0=plain causal)"

        top_level_trainable = self.strides[-1] != -1
        code_count = total_bytes_of(self) // self.byte_group
        for i in range(n):
            K_i = self.strides[i] if self.strides[i] != -1 else 1
            code_count = code_count // K_i
            if i == n - 1 and not top_level_trainable:
                break
            n_blocks_i, G_i, N_i, S_i = code_count, self.decoder_ncodes[i], self.ncodes_window[i], self.streaming[i]
            assert G_i >= 1, f"level {i}: decoder_ncodes={G_i} must be >=1"
            if G_i > n_blocks_i:
                warnings.warn(
                    f"level {i}: decoder_ncodes={G_i} exceeds n_blocks={n_blocks_i} (this level's "
                    f"own code count) -- clamps to one single group, same as decoder_ncodes="
                    f"{n_blocks_i} (the fully-sequential 'original' degenerate case); recommend "
                    f"setting decoder_ncodes={n_blocks_i} explicitly for clarity")
            n_groups_i = -(-n_blocks_i // G_i)
            if G_i >= n_blocks_i and N_i not in (0, -1):
                warnings.warn(
                    f"level {i}: decoder_ncodes={G_i}>=n_blocks={n_blocks_i} (single group, falls "
                    f"back to the fast original decode) -- ncodes_window={N_i} has NO EFFECT here; "
                    f"recommend setting it to 0 for clarity (it's ignored either way)")
            elif N_i != -1 and N_i >= n_groups_i:
                warnings.warn(
                    f"level {i}: ncodes_window={N_i} >= n_groups={n_groups_i} -- every group "
                    f"already sees ALL earlier groups at this setting; recommend -1 (unbounded) "
                    f"instead for the same effect with clearer intent")
            elif N_i == -1 and S_i and G_i < max(1, n_blocks_i // 8):
                warnings.warn(
                    f"level {i}: ncodes_window=-1 streaming=True (causal unbounded) with a small "
                    f"decoder_ncodes={G_i} relative to n_blocks={n_blocks_i} (n_groups={n_groups_i}) "
                    f"-- the naive causal window pads EVERY group to the FULL n_blocks width, so "
                    f"compute/memory scales as O(n_groups*n_blocks); recommend a larger "
                    f"decoder_ncodes or a bounded ncodes_window instead")
            if self.refine_window[i] > 0 and self.n_refine_passes[i] <= 1:
                warnings.warn(
                    f"level {i}: refine_window={self.refine_window[i]} has NO EFFECT with "
                    f"n_refine_passes={self.n_refine_passes[i]} (need >1 for a second pass to use "
                    f"it) -- either raise n_refine_passes or set refine_window=0 for clarity")
            if self.n_refine_passes[i] > 1 and self.refine_window[i] <= 0:
                warnings.warn(
                    f"level {i}: n_refine_passes={self.n_refine_passes[i]} runs extra passes with "
                    f"ZERO peer context (refine_window=0) -- each extra pass degenerates to "
                    f"recomputing pass 1 (wasted compute, not a no-op); set refine_window>0 or "
                    f"n_refine_passes=1")
        if self.refine_gumbel and not any(self.n_refine_passes[i] > 1 and self.refine_window[i] > 0 for i in range(n)):
            warnings.warn(
                "refine_gumbel=True has NO EFFECT -- no level has both n_refine_passes>1 and "
                "refine_window>0, so no refine pass ever runs")
        if self.refine_quantize_drop > 0 and self.multipass_detach:
            warnings.warn(
                f"refine_quantize_drop={self.refine_quantize_drop} has NO EFFECT with "
                f"multipass_detach=True (fully-detached refine passes only ever use the hard "
                f"argmax index, never the soft/drop-mixed code) -- set multipass_detach=False or "
                f"refine_quantize_drop=0 for clarity")

        assert len(self.weight_sharing) == n
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
        assert total_bytes_of(self) % self.byte_group == 0
        assert self.traversal in ("raster", "zorder")
        assert self.strides[-1] == -1 or self.strides[-1] >= 1, \
            "top level's stride is either -1 (don't-care, legacy: top level stays untrained/wasted " \
            "-- see top_level_trainable) or a real stride >=1 (top level becomes fully trainable: " \
            "its own encoder gets a phase, and it gets a real decoder too)"
        assert all(s >= 1 for s in self.strides[:-1])
        n_positions = total_bytes_of(self) // self.byte_group
        assert n_positions % math.prod(self.strides[:-1]) == 0
        assert self.precision in ("bf16", "fp32")
        assert self.curriculum_mode in ("freeze", "no_freeze")
        assert self.curriculum_mode == "no_freeze", \
            "run_lagcodec_zorder requires curriculum_mode='no_freeze' -- a level conditioned on " \
            "cascade-simulated ctx must stay trainable to adapt to it (see module docstring)"
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
    return total_bytes_of(cfg) // cfg.byte_group


def zorder_pixel_order(img_size: int) -> np.ndarray:
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


CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def load_cifar10(data_root: Path) -> tuple:
    data_root.mkdir(parents=True, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        import urllib.request
        tmp_path = tar_path.with_name(tar_path.name + ".tmp")
        print(f"downloading {CIFAR10_URL} -> {tar_path}")
        urllib.request.urlretrieve(CIFAR10_URL, tmp_path)
        tmp_path.rename(tar_path)
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


def load_imagenet64(data_root: Path, resolution: int = 64) -> tuple:
    def load_split(split: str) -> np.ndarray:
        shards = sorted(data_root.glob(f"imagenet64_{split}_*.npy"))
        assert shards, f"no imagenet64_{split}_*.npy shards found under {data_root} -- run " \
            f"image_gen_jax_1/scripts/download_imagenet64.py --split {split} --out_dir {data_root} first"
        parts = [np.load(s, mmap_mode="r") for s in shards]
        flat = np.concatenate(parts, axis=0)
        return flat.reshape(-1, resolution, resolution, 3)

    train = load_split("train")
    val = load_split("validation")
    return (train, np.zeros(len(train), dtype=np.int32)), (val, np.zeros(len(val), dtype=np.int32))


def images_to_positions(images: np.ndarray, cfg: Config, pixel_order: np.ndarray) -> np.ndarray:
    n = images.shape[0]
    pix = images.reshape(n, cfg.img_size * cfg.img_size, 3)[:, pixel_order, :]
    if cfg.byte_group == 3:
        return pix.astype(np.int32)
    return pix.reshape(n, cfg.img_size * cfg.img_size * 3, 1).astype(np.int32)


def positions_to_image(positions: np.ndarray, cfg: Config, pixel_order: np.ndarray) -> np.ndarray:
    B = positions.shape[0]
    pix_traversal = positions.reshape(B, cfg.img_size * cfg.img_size, 3)
    raster = np.zeros_like(pix_traversal)
    raster[:, pixel_order, :] = pix_traversal
    return raster.reshape(B, cfg.img_size, cfg.img_size, 3).astype(np.uint8)


def byte_to_pq_idx_jax(byte_vals: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    bits_per_chunk = max(1, round(math.log2(code_vocab)))
    total_bits = pq_chunks * bits_per_chunk
    shifted = byte_vals >> (8 - total_bits) if total_bits <= 8 else byte_vals << (total_bits - 8)
    chunks = [(shifted >> ((pq_chunks - 1 - c) * bits_per_chunk)) & (code_vocab - 1) for c in range(pq_chunks)]
    return jnp.stack(chunks, axis=-1)


def default_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order: np.ndarray, n_blocks: int,
                          pq_chunks: int, code_vocab: int, method: str = "bilinear") -> jnp.ndarray:
    M = flat_bytes.shape[0]
    pix_traversal = flat_bytes.reshape(M, cfg.img_size * cfg.img_size, 3).astype(jnp.float32)
    raster = jnp.zeros_like(pix_traversal).at[:, pixel_order, :].set(pix_traversal)
    img = raster.reshape(M, cfg.img_size, cfg.img_size, 3)
    side = max(1, round(math.sqrt(n_blocks)))
    small = jax.image.resize(img, (M, side, side, 3), method=method)
    gray = jnp.mean(small, axis=-1)
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)
    flat_gray = gray.reshape(M, side * side)[:, low_order]
    if side * side > n_blocks:
        flat_gray = flat_gray[:, :n_blocks]
    elif side * side < n_blocks:
        flat_gray = jnp.pad(flat_gray, ((0, 0), (0, n_blocks - side * side)))
    byte_vals = jnp.round(jnp.clip(flat_gray, 0, 255)).astype(jnp.int32)
    return byte_to_pq_idx_jax(byte_vals, pq_chunks, code_vocab)


def default_label_fn_pil(images: np.ndarray, cfg: Config, pixel_order: np.ndarray, n_blocks: int,
                          pq_chunks: int, code_vocab: int) -> np.ndarray:
    from PIL import Image
    side = max(1, round(math.sqrt(n_blocks)))
    out = np.zeros((images.shape[0], side, side), dtype=np.float32)
    for b in range(images.shape[0]):
        pil = Image.fromarray(images[b])
        while min(pil.size) >= 2 * side:
            pil = pil.resize(tuple(x // 2 for x in pil.size), resample=Image.BOX)
        pil = pil.resize((side, side), resample=Image.BICUBIC)
        out[b] = np.asarray(pil.convert("RGB"), dtype=np.float32).mean(axis=-1)
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)
    flat_gray = out.reshape(images.shape[0], side * side)[:, low_order]
    if side * side > n_blocks:
        flat_gray = flat_gray[:, :n_blocks]
    elif side * side < n_blocks:
        flat_gray = np.pad(flat_gray, ((0, 0), (0, n_blocks - side * side)))
    byte_vals = np.round(np.clip(flat_gray, 0, 255)).astype(np.int64)
    bits_per_chunk = max(1, round(math.log2(code_vocab)))
    total_bits = pq_chunks * bits_per_chunk
    shifted = byte_vals >> (8 - total_bits) if total_bits <= 8 else byte_vals << (total_bits - 8)
    chunks = [(shifted >> ((pq_chunks - 1 - c) * bits_per_chunk)) & (code_vocab - 1) for c in range(pq_chunks)]
    return np.stack(chunks, axis=-1)


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


def run_block(blk: Block, x: jnp.ndarray, remat: bool, rng=None, drop_prob: float = 0.0) -> jnp.ndarray:
    out = jax.checkpoint(blk)(x) if remat else blk(x)
    if rng is not None and drop_prob > 0.0:
        keep = jax.random.bernoulli(rng, p=1.0 - drop_prob)
        out = jnp.where(keep, out, x)
    return out


def dense_self_attention(attn: Attention, x: jnp.ndarray, causal: bool = False) -> jnp.ndarray:
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
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    if attn.use_xsa:
        y = apply_xsa(y, v)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ attn.out


def pardec_step(attn: Attention, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray,
                 cache_pos, rope_pos: jnp.ndarray, min_valid_pos: jnp.ndarray, T_max: int) -> tuple:
    Bc, D = x_new.shape
    hd = D // attn.n_heads
    qkv = x_new @ attn.qkv
    q, k, v = jnp.split(qkv, [D, D + attn.n_kv_heads * hd], axis=-1)
    q, k, v = q.reshape(Bc, attn.n_heads, hd), k.reshape(Bc, attn.n_kv_heads, hd), v.reshape(Bc, attn.n_kv_heads, hd)
    if attn.use_qknorm:
        q, k = rmsnorm(q, attn.q_norm), rmsnorm(k, attn.k_norm)
    cos, sin = rope_cos_sin_pos(rope_pos, hd, attn.rope_base)
    q = (q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]).astype(q.dtype)
    k = (k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]).astype(k.dtype)
    cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :].astype(cache_k.dtype), (0, 0, cache_pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :].astype(cache_v.dtype), (0, 0, cache_pos, 0))
    n_rep = attn.n_heads // attn.n_kv_heads
    k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
    v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
    idx = jnp.arange(T_max)
    valid = (idx[None, None, :] <= cache_pos) & (idx[None, None, :] >= min_valid_pos[:, None, None])
    logits = jnp.where(valid, logits, -1e9)
    attn_w = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bht,bhtd->bhd", attn_w, v_full)
    if attn.use_xsa:
        v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
        y = apply_xsa(y, v_self)
    y = y.reshape(Bc, D)
    return y @ attn.out, cache_k, cache_v


def pardec_chunk_step(attn: Attention, x_chunk: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray,
                       cache_pos_start, rope_pos_ids: jnp.ndarray, min_valid_pos: jnp.ndarray,
                       T_max: int) -> tuple:
    Bc, T, D = x_chunk.shape
    hd = D // attn.n_heads
    qkv = x_chunk @ attn.qkv
    q, k, v = jnp.split(qkv, [D, D + attn.n_kv_heads * hd], axis=-1)
    q = q.reshape(Bc, T, attn.n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(Bc, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(Bc, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    if attn.use_qknorm:
        q, k = rmsnorm(q, attn.q_norm), rmsnorm(k, attn.k_norm)
    cos, sin = rope_cos_sin_pos(rope_pos_ids, hd, attn.rope_base)
    cos_b, sin_b = cos[:, None, :, :], sin[:, None, :, :]
    q = (q * cos_b + rotate_half(q) * sin_b).astype(q.dtype)
    k = (k * cos_b + rotate_half(k) * sin_b).astype(k.dtype)
    cache_k = jax.lax.dynamic_update_slice(cache_k, k.astype(cache_k.dtype), (0, 0, cache_pos_start, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v.astype(cache_v.dtype), (0, 0, cache_pos_start, 0))
    n_rep = attn.n_heads // attn.n_kv_heads
    k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
    v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
    scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
    logits = jnp.einsum("bhtd,bhsd->bhts", q, k_full) * scale
    pos_ids_abs = cache_pos_start + jnp.arange(T)
    idx = jnp.arange(T_max)
    causal = idx[None, :] <= pos_ids_abs[:, None]
    validmin = idx[None, None, :] >= min_valid_pos[:, None, None]
    valid = causal[None] & validmin
    logits = jnp.where(valid[:, None], logits, -1e9)
    attn_w = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", attn_w, v_full)
    if attn.use_xsa:
        v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
        y = apply_xsa(y, v_self)
    y = y.transpose(0, 2, 1, 3).reshape(Bc, T, D)
    return y @ attn.out, cache_k, cache_v


def pardec_block_step(blk: Block, x_new, cache_k, cache_v, cache_pos, rope_pos, min_valid_pos, T_max):
    attn_out, ck, cv = pardec_step(blk.attn, blk.norm1(x_new), cache_k, cache_v, cache_pos,
                                    rope_pos, min_valid_pos, T_max)
    x = x_new + attn_out
    x = x + blk.mlp(blk.norm2(x))
    return x, ck, cv


def pardec_block_chunk_step(blk: Block, x_chunk, cache_k, cache_v, cache_pos_start, rope_pos_ids,
                             min_valid_pos, T_max):
    attn_out, ck, cv = pardec_chunk_step(blk.attn, blk.norm1(x_chunk), cache_k, cache_v,
                                          cache_pos_start, rope_pos_ids, min_valid_pos, T_max)
    x = x_chunk + attn_out
    x = x + blk.mlp(blk.norm2(x))
    return x, ck, cv


def dense_self_attention_pardec(attn: Attention, x: jnp.ndarray, rope_pos_ids: jnp.ndarray,
                                 min_valid_pos: jnp.ndarray) -> jnp.ndarray:
    Bc, T, D = x.shape
    hd = D // attn.n_heads
    qkv = x @ attn.qkv
    q, k, v = jnp.split(qkv, [D, D + attn.n_kv_heads * hd], axis=-1)
    q = q.reshape(Bc, T, attn.n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(Bc, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(Bc, T, attn.n_kv_heads, hd).transpose(0, 2, 1, 3)
    if attn.use_qknorm:
        q, k = rmsnorm(q, attn.q_norm), rmsnorm(k, attn.k_norm)
    cos, sin = rope_cos_sin_pos(rope_pos_ids, hd, attn.rope_base)
    cos_b, sin_b = cos[:, None], sin[:, None]
    q = (q * cos_b + rotate_half(q) * sin_b).astype(q.dtype)
    k = (k * cos_b + rotate_half(k) * sin_b).astype(k.dtype)
    rep = attn.n_heads // attn.n_kv_heads
    if rep > 1:
        k, v = jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)
    scale = 1.0 / math.sqrt(hd)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    idx = jnp.arange(T)
    causal = idx[None, :] <= idx[:, None]
    validmin = idx[None, :] >= min_valid_pos[:, None]
    mask = causal[None] & validmin[:, None, :]
    scores = jnp.where(mask[:, None], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    if attn.use_xsa:
        y = apply_xsa(y, v)
    y = y.transpose(0, 2, 1, 3).reshape(Bc, T, D)
    return y @ attn.out


def run_block_pardec(blk: Block, x: jnp.ndarray, rope_pos_ids: jnp.ndarray, min_valid_pos: jnp.ndarray,
                      remat: bool) -> jnp.ndarray:
    def f(x):
        x = x + dense_self_attention_pardec(blk.attn, blk.norm1(x), rope_pos_ids, min_valid_pos)
        x = x + blk.mlp(blk.norm2(x))
        return x
    return jax.checkpoint(f)(x) if remat else f(x)


def token_ar_teacher_forced(in_proj, member_embed, norm1, attn, ln_f, out_head, dim, in_code_vocab,
                             h: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
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
    token_in_proj: jnp.ndarray
    token_member_embed: jnp.ndarray
    token_mask_embed: jnp.ndarray
    token_channel_embed: jnp.ndarray
    token_norm1: RMSNorm
    token_attn: Attention
    token_ln_f: RMSNorm
    token_out_head: jnp.ndarray
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
    token_head_type: str = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    token_mask_prob: float = eqx.field(static=True)
    mtp_horizon: int = eqx.field(static=True)
    mtp_mode: str = eqx.field(static=True)
    mtp_weight: float = eqx.field(static=True)
    remat: bool = eqx.field(static=True)
    pq_dim: int = eqx.field(static=True)
    ncodes_window: int = eqx.field(static=True)
    streaming: bool = eqx.field(static=True)
    decode_past: int = eqx.field(static=True)
    decode_future: int = eqx.field(static=True)
    attn_lookahead: int = eqx.field(static=True)
    n_refine_passes: int = eqx.field(static=True)
    refine_window: int = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, has_decoder: bool, weight_sharing: bool):
        D = cfg.d_model[level]
        self.K = cfg.strides[level] if cfg.strides[level] != -1 else 1
        self.n_heads, self.n_kv_heads = cfg.n_heads[level], cfg.n_kv_heads[level]
        self.quantize_mode = cfg.quantize_mode
        self.quantize_drop = cfg.quantize_drop
        self.remat = cfg.remat
        self.ncodes_window = cfg.ncodes_window[level]
        self.streaming = cfg.streaming[level]
        self.decode_past = cfg.decode_past[level]
        self.decode_future = cfg.decode_future[level]
        self.attn_lookahead = cfg.attn_lookahead[level]
        self.n_refine_passes = cfg.n_refine_passes[level]
        self.refine_window = cfg.refine_window[level]
        is_byte_level = (level == 0)
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[level], cfg.code_vocab[level]
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
        enc_window = None if cfg.attn_window[level] == -1 else cfg.attn_window[level]
        enc_lookahead = cfg.attn_lookahead[level]
        self.blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level],
                             n_layers=n_layers, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm,
                             window=enc_window, lookahead=enc_lookahead, use_sink=cfg.use_sink) for k in block_keys]
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


    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, rng=None, encode_temperature: float = 1.0,
               layer_drop_prob=None) -> dict:
        h = x
        n_blk = len(self.blocks)
        if rng is not None:
            layer_rngs = list(jax.random.split(rng, n_blk + 1))
            quant_rng = layer_rngs[-1]
        else:
            layer_rngs = [None] * n_blk
            quant_rng = None
        if layer_drop_prob is None:
            layer_drop_prob = (0.0,) * n_blk
        elif isinstance(layer_drop_prob, (int, float)):
            layer_drop_prob = (layer_drop_prob,) * n_blk
        for i, blk in enumerate(self.blocks):
            h = run_block(blk, h, self.remat, rng=layer_rngs[i], drop_prob=layer_drop_prob[i])
        h = self.ln_f(h)
        M, L, D = h.shape
        n_blocks = L // self.K
        h_blocks = h[:, :n_blocks * self.K, :].reshape(M, n_blocks, self.K, D)
        pooled = h_blocks[:, :, self.K - 1, :]
        logits = reshape_pq(pooled @ self.code_head, self.pq_chunks, self.code_vocab)
        if quant_rng is not None and self.quantize_mode == "gumbel":
            code_soft, code_idx = quantize_gumbel(logits, quant_rng, encode_temperature, self.quantize_drop)
        else:
            code_soft, code_idx = quantize_hard(logits, quant_rng, self.quantize_drop, encode_temperature)

        probs = jax.nn.softmax(logits, axis=-1)
        p_avg = jnp.mean(probs, axis=(0, 1))
        entropy_loss = jnp.mean(jnp.sum(p_avg * jnp.log(jnp.maximum(p_avg, 1e-9)), axis=-1))

        ntp_shift = 1 + self.attn_lookahead
        if L > ntp_shift:
            ntp_logits = reshape_pq(h[:, :-ntp_shift, :] @ self.ntp_head, self.in_pq_chunks, self.in_code_vocab)
            tgt = target_idx[:, ntp_shift:]
            logp = jax.nn.log_softmax(ntp_logits, axis=-1)
            ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
            ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
        else:
            ntp_loss = jnp.array(0.0, dtype=h.dtype)
            ntp_acc = jnp.array(0.0, dtype=h.dtype)
        util = codebook_utilization(code_idx, self.code_vocab)
        return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                    entropy_loss=entropy_loss, logits=logits)


    def _dec_blocks(self):
        return self.blocks if self.weight_sharing else self.dec_blocks

    def _dec_ln_f(self):
        return self.ln_f if self.weight_sharing else self.dec_ln_f

    def _dec_embed_target(self, idx: jnp.ndarray) -> jnp.ndarray:
        table = self.own_input_embed if self.weight_sharing else self.dec_target_embed
        proj = self.own_input_proj if self.weight_sharing else self.dec_target_proj
        return code_embed_proj(idx, table, proj)

    def _dec_head_w(self) -> jnp.ndarray:
        return self.ntp_head if self.weight_sharing else self.dec_head


    def _token_logits_linears(self, h: jnp.ndarray) -> jnp.ndarray:
        logits = h @ self._dec_head_w()
        return reshape_pq(logits, self.in_pq_chunks, self.in_code_vocab)

    def _token_teacher_forced_ar(self, h: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return token_ar_teacher_forced(self.token_in_proj, self.token_member_embed, self.token_norm1,
                                        self.token_attn, self.token_ln_f, self.token_out_head,
                                        self.token_dim, self.in_code_vocab, h, target)

    def _token_generate_ar(self, h: jnp.ndarray, rng, greedy: bool, temperature: float) -> tuple:
        return token_ar_generate(self.token_in_proj, self.token_member_embed, self.token_norm1,
                                  self.token_attn, self.token_ln_f, self.token_out_head,
                                  self.in_pq_chunks, h, rng, greedy, temperature)

    def _token_teacher_forced_diffusion(self, h: jnp.ndarray, target: jnp.ndarray, rng) -> tuple:
        lead = h.shape[:-1]
        D = h.shape[-1]
        chunks = self.in_pq_chunks
        N = int(np.prod(lead)) if lead else 1
        ctx = (h.reshape(N, D) @ self.token_in_proj)
        tgt_flat = target.reshape(N, chunks)
        mask = jnp.ones((N, chunks), dtype=bool)
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

    def decode_logits_and_target_pardec(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray,
                                         decoder_ncodes: int, rng=None, decode_past_override: int = None,
                                         draft_override: jnp.ndarray = None,
                                         draft_embed_override: jnp.ndarray = None,
                                         remat_override: bool = None) -> tuple:
        n_blocks_check = ctx_code_soft.shape[1]
        if decoder_ncodes >= n_blocks_check and decode_past_override is None:
            return self.decode_logits_and_target(target_seq, ctx_code_soft, decoder_ncodes, rng=rng)
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B = target_seq.shape[0]
        D = self.bos_embed.shape[-1]
        G = decoder_ncodes
        n_blocks = ctx_code_soft.shape[1]
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        fullctx = (not self.streaming) and (self.ncodes_window == -1)
        N = self.ncodes_window if self.ncodes_window >= 0 else (n_groups - 1)
        Wg = n_blocks_p if fullctx else (N + 1) * G
        per_group_len = Wg + 1 + G * self.K

        ctx_tok = code_embed_proj(ctx_code_soft, self.ctx_embed, self.ctx_proj)
        target_p = target_seq
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
            target_p = jnp.pad(target_p, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        if not fullctx and N > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (N * G, 0), (0, 0)))

        B2 = B * n_groups
        if fullctx:
            ctx_flat = jnp.broadcast_to(ctx_tok[:, None, :, :], (B, n_groups, Wg, D)).reshape(B2, Wg, D)
            min_valid_pos = jnp.zeros((B2,), dtype=jnp.int32)
            rope_ctx_g = jnp.broadcast_to(jnp.arange(Wg)[None, :], (n_groups, Wg))
        else:
            ctx_windows = jnp.stack([ctx_tok[:, g * G:g * G + Wg, :] for g in range(n_groups)], axis=1)
            ctx_flat = ctx_windows.reshape(B2, Wg, D)
            fake_counts = jnp.array([max(0, N - g) * G for g in range(n_groups)])
            min_valid_pos = jnp.broadcast_to(fake_counts[None, :], (B, n_groups)).reshape(B2)
            rope_ctx_g = jnp.stack([jnp.clip(jnp.arange(Wg) - N * G + g * G, 0, None) for g in range(n_groups)], axis=0)

        target_windows = jnp.stack(
            [target_p[:, g * G * self.K:(g + 1) * G * self.K] for g in range(n_groups)], axis=1)

        Pp = self.decode_past if decode_past_override is None else decode_past_override
        Pf = self.decode_future
        Kspan = G * self.K
        widened_len = Pp + Kspan + Pf
        if Pf > 0:
            tail_p = jnp.pad(target_p, ((0, 0), (0, Pf), (0, 0)))
            real_tail_windows = jnp.stack(
                [tail_p[:, g * Kspan:g * Kspan + Kspan + Pf] for g in range(n_groups)], axis=1)
        else:
            real_tail_windows = target_windows
        real_tail_flat = real_tail_windows.reshape(B2, Kspan + Pf, *target_seq.shape[2:])
        real_tail_te = self._dec_embed_target(real_tail_flat)

        if Pp > 0:
            if draft_embed_override is not None:
                draft_embed_p = jnp.pad(draft_embed_override, ((0, 0), (Pp, 0), (0, 0)))
                draft_te = jnp.stack(
                    [draft_embed_p[:, g * Kspan:g * Kspan + Pp, :] for g in range(n_groups)], axis=1
                ).reshape(B2, Pp, D)
            else:
                if draft_override is None:
                    draft_source = target_p
                elif pad_blocks > 0:
                    draft_source = jnp.pad(draft_override, ((0, 0), (0, pad_blocks * self.K)) +
                                            ((0, 0),) * (draft_override.ndim - 2))
                else:
                    draft_source = draft_override
                draft_p = jnp.pad(draft_source, ((0, 0), (Pp, 0)) + ((0, 0),) * (draft_source.ndim - 2))
                draft_windows = jnp.stack([draft_p[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1)
                draft_flat = draft_windows.reshape(B2, Pp, *target_seq.shape[2:])
                draft_te = self._dec_embed_target(draft_flat)
            te_flat = jnp.concatenate([draft_te, real_tail_te], axis=1)
        else:
            te_flat = real_tail_te
        bos = jnp.broadcast_to(self.bos_embed, (B2, 1, D))
        xe = jnp.concatenate([ctx_flat, bos, te_flat], axis=1)
        per_group_len = per_group_len + Pp + Pf
        remat = self.remat if remat_override is None else remat_override

        rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])[:, None]
        rope_draft = jnp.stack([(g + 1) * G - Pp + jnp.arange(Pp) for g in range(n_groups)], axis=0)
        rope_real_tail = jnp.stack(
            [(g + 1) * G + 1 + jnp.arange(widened_len - Pp) for g in range(n_groups)], axis=0)
        rope_target = jnp.clip(jnp.concatenate([rope_draft, rope_real_tail], axis=1), 0, None)
        rope_pos_ids_g = jnp.concatenate([rope_ctx_g, rope_bos, rope_target], axis=1)
        rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (B, n_groups, per_group_len)).reshape(B2, per_group_len)

        x = xe
        for blk in blocks:
            x = run_block_pardec(blk, x, rope_pos_ids, min_valid_pos, remat)
        h = ln_f(x)
        pred_pos = Wg + Pp + jnp.arange(G * self.K)
        h_t = h[:, pred_pos, :]
        h_t = h_t.reshape(B, n_groups * G * self.K, D)
        target_out = target_windows.reshape(B, n_groups * G * self.K, *target_seq.shape[2:])
        valid_len = n_blocks * self.K
        h_t = h_t[:, :valid_len, :]
        target_out = target_out[:, :valid_len]

        if self.token_head_type == "linears":
            logits, mask = self._token_logits_linears(h_t), None
        elif self.token_head_type == "ar":
            logits, mask = self._token_teacher_forced_ar(h_t, target_out), None
        else:
            assert rng is not None, "diffusion token head needs an rng even at eval (masking is inherent)"
            logits, mask = self._token_teacher_forced_diffusion(h_t, target_out, rng)
        mtp_loss = self._mtp_loss(h_t, target_out)
        return logits, target_out, mask, mtp_loss

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

    def decode_generate_pardec(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                                temperature: float = 1.0, seed: int = 0, decode_past_override: int = None,
                                draft_override_flat: jnp.ndarray = None) -> jnp.ndarray:
        n_blocks_check = ctx_idx.shape[1]
        if decoder_ncodes >= n_blocks_check and decode_past_override is None:
            return self.decode_generate(ctx_idx, decoder_ncodes, greedy, temperature, seed)
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        fullctx = (not self.streaming) and (self.ncodes_window == -1)
        N = self.ncodes_window if self.ncodes_window >= 0 else (n_groups - 1)
        Wg = n_blocks_p if fullctx else (N + 1) * G
        Pp = self.decode_past if decode_past_override is None else decode_past_override
        per_group_len = Wg + 1 + Pp + G * self.K

        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        if not fullctx and N > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (N * G, 0), (0, 0)))
        B2 = B * n_groups
        if fullctx:
            ctx_tok_flat = jnp.broadcast_to(ctx_tok[:, None, :, :], (B, n_groups, Wg, D)).reshape(B2, Wg, D)
            min_valid_pos = jnp.zeros((B2,), dtype=jnp.int32)
            rope_ctx_flat = jnp.broadcast_to(jnp.arange(Wg)[None, :], (B2, Wg))
        else:
            ctx_windows = jnp.stack([ctx_tok[:, g * G:g * G + Wg, :] for g in range(n_groups)], axis=1)
            ctx_tok_flat = ctx_windows.reshape(B2, Wg, D)
            fake_counts = jnp.array([max(0, N - g) * G for g in range(n_groups)])
            min_valid_pos = jnp.broadcast_to(fake_counts[None, :], (B, n_groups)).reshape(B2)
            rope_ctx = jnp.stack([jnp.clip(jnp.arange(Wg) - N * G + g * G, 0, None) for g in range(n_groups)], axis=0)
            rope_ctx_flat = jnp.broadcast_to(rope_ctx[None], (B, n_groups, Wg)).reshape(B2, Wg)
        rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])
        rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B, n_groups)).reshape(B2)

        def self_step(x_new, ck, cv, pos, rope_pos_row):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = pardec_block_step(blk, x, ck[i], cv[i], pos, rope_pos_row, min_valid_pos, per_group_len)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        def self_chunk_step(x_chunk, ck, cv, pos_start, rope_pos_ids_chunk):
            new_ck, new_cv = [], []
            x = x_chunk
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = pardec_block_chunk_step(blk, x, ck[i], cv[i], pos_start,
                                                          rope_pos_ids_chunk, min_valid_pos, per_group_len)
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

        cache_k = jnp.zeros((len(blocks), B2, self.n_kv_heads, per_group_len, hd))
        cache_v = jnp.zeros_like(cache_k)
        rng = jax.random.PRNGKey(seed)

        total_steps = Pp + G * self.K

        def widened_pos(t):
            if t < Pp:
                return jnp.clip(rope_bos_flat - Pp + t, 0, None)
            return rope_bos_flat + 1 + (t - Pp)

        @jax.jit
        def run_pardec(ctx_tok_flat, cache_k, cache_v, rng):
            bos_in = jnp.broadcast_to(self.bos_embed, (B2, 1, D))
            chunk = jnp.concatenate([ctx_tok_flat, bos_in], axis=1)
            chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
            h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
            pos = Wg + 1
            rope_pos_row = widened_pos(0)
            h = h_chunk[:, -1, :]
            val, rng = token_predict(h, rng)
            vals = [val] if Pp == 0 else []
            x_input = self._dec_embed_target(val if draft_override_flat is None else draft_override_flat[:, 0])
            for t in range(1, total_steps):
                h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rope_pos_row)
                pos = pos + 1
                rope_pos_row = widened_pos(t)
                val, rng = token_predict(h, rng)
                if t >= Pp:
                    vals.append(val)
                if draft_override_flat is not None and t < Pp:
                    val = draft_override_flat[:, t]
                x_input = self._dec_embed_target(val)
            _, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rope_pos_row)
            return jnp.stack(vals, axis=1)

        vals_all = run_pardec(ctx_tok_flat, cache_k, cache_v, rng)
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        return out[:, :n_blocks * self.K]

    def decode_generate_mtp_no_verify(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                                       temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
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
            draft, rng = mtp_predict_no_verify_standalone(self, h, rng, greedy, temperature)
            vals = [draft[:, k, :] for k in range(K)]
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


def decode_logits_and_target_multipass(level: EncDecLevel, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray,
                                        decoder_ncodes: int, rng=None, multipass_detach: bool = True,
                                        refine_gumbel: bool = False, refine_temperature: float = 1.0,
                                        refine_quantize_drop: float = 0.0, refine_rng=None,
                                        refine_remat: bool = None) -> tuple:
    logits, target_out, mask, mtp_loss = level.decode_logits_and_target_pardec(
        target_seq, ctx_code_soft, decoder_ncodes, rng=rng)
    if level.n_refine_passes <= 1 or level.refine_window <= 0:
        return logits, target_out, mask, mtp_loss
    Kspan = decoder_ncodes * level.K
    Pp = level.refine_window * Kspan
    n_extra = level.n_refine_passes - 1
    refine_rngs = [None] * n_extra if refine_rng is None else list(jax.random.split(refine_rng, n_extra))
    for p_idx in range(n_extra):
        r_rng = refine_rngs[p_idx]
        if refine_gumbel:
            assert r_rng is not None, "refine_gumbel=True needs refine_rng"
            code_soft, idx = quantize_gumbel(logits, r_rng, refine_temperature, refine_quantize_drop)
        else:
            code_soft, idx = quantize_hard(logits, r_rng if refine_quantize_drop > 0 else None,
                                            refine_quantize_drop, refine_temperature)
        if multipass_detach:
            kwargs = dict(draft_override=jax.lax.stop_gradient(idx))
        else:
            kwargs = dict(draft_embed_override=level._dec_embed_target(code_soft))
        logits, target_out, mask, mtp_loss = level.decode_logits_and_target_pardec(
            target_seq, ctx_code_soft, decoder_ncodes, rng=rng,
            decode_past_override=Pp, remat_override=refine_remat, **kwargs)
    return logits, target_out, mask, mtp_loss


def _decode_generate_pardec_call(level, ctx_idx, decoder_ncodes, greedy, temperature, seed,
                                  decode_past_override, draft_override_flat):
    return level.decode_generate_pardec(ctx_idx, decoder_ncodes, greedy=greedy, temperature=temperature,
                                         seed=seed, decode_past_override=decode_past_override,
                                         draft_override_flat=draft_override_flat)


_decode_generate_pardec_jit = eqx.filter_jit(_decode_generate_pardec_call)


def decode_generate_multipass(level: EncDecLevel, ctx_idx: jnp.ndarray, decoder_ncodes: int,
                               greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
    pred = _decode_generate_pardec_jit(level, ctx_idx, decoder_ncodes, greedy, temperature, seed, None, None)
    if level.n_refine_passes <= 1 or level.refine_window <= 0:
        return pred
    B, n_blocks = ctx_idx.shape[0], ctx_idx.shape[1]
    G = decoder_ncodes
    pad_blocks = (-n_blocks) % G
    n_groups = (n_blocks + pad_blocks) // G
    Kspan = G * level.K
    Pp = level.refine_window * Kspan
    for _ in range(level.n_refine_passes - 1):
        pred_p = pred if pad_blocks == 0 else jnp.pad(
            pred, ((0, 0), (0, pad_blocks * level.K)) + ((0, 0),) * (pred.ndim - 2))
        draft_p = jnp.pad(pred_p, ((0, 0), (Pp, 0)) + ((0, 0),) * (pred_p.ndim - 2))
        draft_windows = jnp.stack([draft_p[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1)
        draft_override_flat = draft_windows.reshape(B * n_groups, Pp, *pred.shape[2:])
        pred = _decode_generate_pardec_jit(level, ctx_idx, decoder_ncodes, greedy, temperature, seed,
                                            Pp, draft_override_flat)
    return pred


def mtp_predict_no_verify_standalone(level: EncDecLevel, h_pos, rng, greedy, temperature):
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


class HierEncDec(eqx.Module):
    levels: list
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        top_level_trainable = cfg.strides[-1] != -1
        keys = jax.random.split(key, n)
        self.levels = [EncDecLevel(keys[i], cfg, level=i, has_decoder=(i < n - 1) or top_level_trainable,
                                    weight_sharing=cfg.weight_sharing[i]) for i in range(n)]


def phase_forward(model: HierEncDec, flat_bytes: jnp.ndarray, phase: int, rng=None,
                   level_gt_drop=None, cascade_rng=None, encode_temperature: float = 1.0,
                   layer_drop_prob=None, label_reg_weight: float = 0.0, label_fn=None,
                   pixel_order=None, feedback_p=None, feedback_rng=None, feedback_detach: bool = True,
                   _feedback_recursed: bool = False) -> tuple:
    levels = model.levels
    x = code_embed_proj(flat_bytes, levels[0].own_input_embed, levels[0].own_input_proj)
    target = flat_bytes
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses = [], [], [], [], []
    level_rngs = [None] * (2 * phase) if rng is None else list(jax.random.split(rng, 2 * phase))
    for i in range(phase):
        out = levels[i].encode(x, target, rng=level_rngs[2 * i], encode_temperature=encode_temperature,
                                layer_drop_prob=layer_drop_prob)
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        enc_losses.append(out["ntp_loss"])
        enc_accs.append(out["ntp_acc"])
        utils.append(out["util"])
        entropy_losses.append(out["entropy_loss"])
        if label_reg_weight > 0:
            enc_logits = out["logits"]
            n_blocks_i = enc_logits.shape[1]
            label_tgt = label_fn(flat_bytes, model.cfg, pixel_order, n_blocks_i,
                                  model.cfg.pq_chunks[i], model.cfg.code_vocab[i])
            logp_i = jax.nn.log_softmax(enc_logits, axis=-1)
            label_losses.append(-jnp.mean(jnp.take_along_axis(logp_i, label_tgt[..., None], axis=-1)))
        if i < phase - 1:
            x = code_embed_proj(out["code_soft"], levels[i + 1].own_input_embed, levels[i + 1].own_input_proj)
            target = out["code_idx"]

    dec_losses, dec_accs = [], []
    feedback_losses = []
    byte_mse = None
    ctx = codes_soft[phase - 1]
    cascade_rngs = [None] * phase if cascade_rng is None else list(jax.random.split(cascade_rng, phase))
    feedback_rngs = [None] * phase if feedback_rng is None else list(jax.random.split(feedback_rng, phase))
    for i in range(phase - 1, -1, -1):
        dec_target = flat_bytes if i == 0 else codes[i - 1]
        dec_rng = level_rngs[2 * i + 1]
        logits, target_i, mask_i, mtp_loss_i = decode_logits_and_target_multipass(
            levels[i], dec_target, ctx, model.cfg.decoder_ncodes[i], rng=dec_rng,
            multipass_detach=model.cfg.multipass_detach, refine_gumbel=model.cfg.refine_gumbel,
            refine_temperature=model.cfg.refine_temperature,
            refine_quantize_drop=model.cfg.refine_quantize_drop, refine_rng=dec_rng,
            refine_remat=model.cfg.refine_remat)
        loss_i, acc_i = levels[i]._dec_loss_acc(logits, target_i, mask_i)
        loss_i = loss_i + levels[i].mtp_weight * mtp_loss_i
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if i == 0:
            pred_bytes = jnp.argmax(logits, axis=-1).astype(jnp.float32)
            byte_mse = jnp.mean((pred_bytes - target_i.astype(jnp.float32)) ** 2)
            if model.cfg.mse_weight > 0:
                byte_probs = jax.nn.softmax(logits / model.cfg.mse_softmax_tau, axis=-1)
                byte_values = jnp.arange(byte_probs.shape[-1], dtype=byte_probs.dtype)
                pred_pixel = jnp.sum(byte_probs * byte_values, axis=-1)
                max_val = byte_probs.shape[-1] - 1
                mse_loss = jnp.mean(((pred_pixel - target_i.astype(jnp.float32)) / max_val) ** 2)
            else:
                mse_loss = 0.0
        if feedback_p is not None:
            feedback_p_i = feedback_p if isinstance(feedback_p, (int, float)) else feedback_p[i]
            if feedback_p_i > 0 and not (i == 0 and _feedback_recursed):
                fire_i = jax.random.bernoulli(feedback_rngs[i], p=feedback_p_i)
                zero = jnp.array(0.0, dtype=loss_i.dtype)
                if i > 0:
                    def _feedback_fire(i=i, logits=logits):
                        pseudo_ctx = quantize_hard(logits)[0]
                        if feedback_detach:
                            pseudo_ctx = jax.lax.stop_gradient(pseudo_ctx)
                        c, total = pseudo_ctx, 0.0
                        for j in range(i - 1, -1, -1):
                            dj = flat_bytes if j == 0 else codes[j - 1]
                            lj, tj, mj, mtpj = levels[j].decode_logits_and_target_pardec(
                                dj, c, model.cfg.decoder_ncodes[j])
                            lossj, _ = levels[j]._dec_loss_acc(lj, tj, mj)
                            total = total + lossj + levels[j].mtp_weight * mtpj
                            if j > 0:
                                c = codes_soft[j - 1]
                        return total / i
                else:
                    def _feedback_fire(logits=logits):
                        pseudo_bytes = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))
                        l2, _ = phase_forward(
                            model, pseudo_bytes, phase, rng=rng, level_gt_drop=level_gt_drop,
                            cascade_rng=cascade_rng, encode_temperature=encode_temperature,
                            layer_drop_prob=layer_drop_prob, label_reg_weight=0.0, label_fn=None,
                            pixel_order=None, feedback_p=None, feedback_rng=None,
                            _feedback_recursed=True)
                        return l2
                feedback_losses.append(jax.lax.cond(fire_i, _feedback_fire, lambda: zero))
        if i > 0:
            real_ctx = codes_soft[i - 1]
            if level_gt_drop is None:
                ctx = real_ctx
            else:
                level_gt_drop_i = level_gt_drop if isinstance(level_gt_drop, (int, float)) \
                    else level_gt_drop[i]
                use_cascade_i = jax.random.bernoulli(cascade_rngs[i], p=level_gt_drop_i)
                pseudo_ctx, _ = quantize_hard(logits)
                ctx = jnp.where(use_cascade_i, pseudo_ctx, real_ctx)

    dec_loss_total = jnp.mean(jnp.stack(dec_losses))
    byte_acc = dec_accs[-1]
    ntp_loss_total = jnp.mean(jnp.stack(enc_losses))
    entropy_loss_total = jnp.mean(jnp.stack(entropy_losses))
    label_loss_total = jnp.mean(jnp.stack(label_losses)) if label_losses else 0.0
    feedback_loss_total = jnp.mean(jnp.stack(feedback_losses)) if feedback_losses else 0.0
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + model.cfg.mse_weight * mse_loss + label_reg_weight * label_loss_total + feedback_loss_total
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


def save_checkpoint(ckpt_dir: Path, model, opt_state, p_rng, train_iter: "BatchIterator",
                     phase: int, phase_step: int, step: int, seed: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    eqx.tree_serialise_leaves(ckpt_dir / "p_rng.eqx", p_rng)
    (ckpt_dir / "dataloader_state.json").write_text(json.dumps(train_iter.rng.bit_generator.state))
    (ckpt_dir / "meta.json").write_text(json.dumps(dict(phase=phase, phase_step=phase_step, step=step, seed=seed)))


def find_latest_checkpoint(run_dir: Path):
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


def ema_update(ema_tree, new_tree, decay: float):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1 - decay) * p if eqx.is_array(e) else e, ema_tree, new_tree)


def stack_average(stack: list, weights=None):
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
                  "ncodes_window", "streaming", "decode_past", "decode_future", "sync",
                  "n_refine_passes", "refine_window", "multipass_detach", "refine_gumbel",
                  "refine_temperature", "refine_quantize_drop", "refine_remat",
                  "weight_sharing", "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_at_inference", "init_scheme", "use_xsa",
                  "use_qknorm", "remat", "attn_window", "attn_lookahead", "use_sink",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "mtp_horizon", "mtp_mode", "mtp_weight", "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal", "label_reg_weight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet64"],
                    help="cifar (default): downloads/caches under --data_root. imagenet64: reads "
                         "pre-built shards from --data_root (produced by "
                         "image_gen_jax_1/scripts/download_imagenet64.py -- does NOT download "
                         "itself, run that script first). Config.img_size must match (32 for "
                         "cifar, 64 for imagenet64).")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--batch_size", type=_tuple_arg, default=(16,),
                    help="training batch size -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase (length must equal n_phases)")
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--level_steps", type=_tuple_arg, default=None,
                    help="steps per phase -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase. At most one of --level_steps/"
                         "--level_epochs may be set")
    p.add_argument("--level_epochs", type=_tuple_arg, default=None,
                    help="epochs per phase -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase. At most one of --level_steps/"
                         "--level_epochs may be set. Default (both unset): 1000 epochs")
    p.add_argument("--no_curriculum", type=lambda x: x.lower() != "false", default=False,
                    help="skip the phase-by-phase curriculum entirely: train ALL levels jointly "
                         "from step 1 (curriculum_mode='no_freeze' still required). Reuses the "
                         "same phase loop with phase fixed at n_phases for its only iteration; "
                         "level_steps/level_epochs's single/last entry is used.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=None,
                    help="warmup length, in steps. At most one of --warmup_steps/--warmup_epochs "
                         "may be set. Default (both unset): 100 steps")
    p.add_argument("--warmup_epochs", type=float, default=None,
                    help="warmup length, in epochs (converted using this phase's own "
                         "steps_per_epoch). Default (both unset): 100 steps")
    p.add_argument("--lr_schedule", type=str, default="const", choices=["const", "cosine"],
                    help="const: warmup then flat forever (default). cosine: warmup then cosine "
                         "decay to 0 over this phase's own epoch_count*steps_per_epoch")
    p.add_argument("--lr_min", type=float, default=0.0,
                    help="cosine only: lr floor the decay reaches (default 0)")
    p.add_argument("--lr_min_step", type=int, default=None,
                    help="cosine only: step (within this phase) at which lr_min is reached; lr "
                         "holds flat at lr_min for the rest of the phase. At most one of "
                         "--lr_min_step/--lr_min_epoch may be set")
    p.add_argument("--lr_min_epoch", type=float, default=None,
                    help="cosine only: epoch (within this phase) at which lr_min is reached; lr "
                         "holds flat at lr_min for the rest of the phase. At most one of "
                         "--lr_min_step/--lr_min_epoch may be set. Default (both unset): reach "
                         "lr_min exactly at phase end (old behavior)")
    p.add_argument("--optimizer", type=str, default="sinkgd", choices=["adamw", "sinkgd"])
    p.add_argument("--optimizer_kwargs", type=json.loads, default={"sinkhorn_iters": 1, "weight_decay": 0})
    p.add_argument("--grad_clip", type=lambda x: None if x.lower() == "none" else float(x), default=1.0,
                    help="global-norm gradient clip threshold, applied before the optimizer "
                         "update; 'none' disables it")
    p.add_argument("--log_every", type=int, default=10, help="in steps")
    p.add_argument("--gen_eval_every_step", type=int, default=None, help="mid-phase gen-eval cadence, in steps")
    p.add_argument("--gen_eval_every_epoch", type=float, default=None,
                    help="mid-phase gen-eval cadence, in epochs (auto-converted to steps via "
                         "this phase's own steps_per_epoch). At most one of --gen_eval_every_step/"
                         "--gen_eval_every_epoch may be set. Default (both unset): 10 epochs")
    p.add_argument("--ckpt_every_step", type=int, default=None,
                    help="save a full resumable checkpoint (model+optim+rng+dataloader state) "
                         "every N steps, in addition to always at phase end")
    p.add_argument("--ckpt_every_epoch", type=float, default=None,
                    help="checkpoint cadence, in epochs (auto-converted to steps). At most one "
                         "of --ckpt_every_step/--ckpt_every_epoch may be set. Default (both "
                         "unset): 10 epochs")
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
    p.add_argument("--wa_verbose", type=lambda x: x.lower() != "false", default=True,
                    help="log a line every time a wa (ema/wma) snapshot is saved. Default True; "
                         "set False to suppress (the snapshot is still saved either way)")
    p.add_argument("--epoch_verbose", type=lambda x: x.lower() != "false", default=True,
                    help="log a line at the start of every epoch. Default True; "
                         "set False to suppress it")
    p.add_argument("--final_eval", type=lambda x: x.lower() != "false", default=False,
                    help="run the val loss + gen-eval at the END of each phase (tag "
                         "'level{N}_final'). Default False (skip); the periodic in-phase eval "
                         "(--gen_eval_every_step/--gen_eval_every_epoch) and the all-phases-done "
                         "final eval still run regardless of this flag")
    p.add_argument("--verbose", type=lambda x: x.lower() != "false", default=True,
                    help="gen-eval: when the eval batch has fewer than 10 samples, also log a "
                         "per-sample mse1=.. mse2=.. line. Default True")
    p.add_argument("--wa_every_step", type=int, default=None, help="WA update cadence, in steps")
    p.add_argument("--wa_every_epoch", type=float, default=None,
                    help="WA update cadence, in epochs (auto-converted to steps). At most one "
                         "of --wa_every_step/--wa_every_epoch may be set. Default (both unset): "
                         "10 epochs")
    p.add_argument("--wa_ema_decay", type=float, default=0.999, help="ema mode only")
    p.add_argument("--wa_stack_size", type=int, default=3, help="wma mode only")
    p.add_argument("--wa_wma_weights", type=_float_tuple_arg, default=None,
                    help="wma mode only: one raw score per stack slot (oldest first), "
                         "softmax-normalized to sum to 1 -- length must equal wa_stack_size. "
                         "Default None: uniform (1/wa_stack_size each)")
    p.add_argument("--train_subset_n", type=int, default=None)
    p.add_argument("--val_subset_n", type=int, default=None,
                    help="cap the val pool to the first N images (None: use the full val set). "
                         "Independent of val_batch_size, which only controls how many of this "
                         "pool are used per single eval/gen-eval call")
    p.add_argument("--eval_gen_train", type=lambda x: x.lower() != "false", default=True,
                    help="also run gen-eval (cascade generation + sample grid) on a TRAIN-set "
                         "prompt, in addition to the usual val-set one -- same count as "
                         "val_batch_size, tagged '<tag>_train'. Default True")
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
    p.add_argument("--ncodes_window", type=_tuple_arg, default=Config.ncodes_window)
    p.add_argument("--streaming", type=_bool_tuple_arg, default=Config.streaming)
    p.add_argument("--decode_past", type=_tuple_arg, default=Config.decode_past,
                    help="redecode this many extra target positions before a group's own real "
                         "span (teacher-forced at training; at generation, the group's own private "
                         "redecode/draft, never shared) -- pruned from the final output either way")
    p.add_argument("--decode_future", type=_tuple_arg, default=Config.decode_future,
                    help="redecode this many extra target positions after a group's own real span "
                         "-- TRAINING ONLY (teacher-forced); ignored at generation until sync=True "
                         "lands (stub, not implemented) -- pruned from the final output")
    p.add_argument("--sync", type=_bool_tuple_arg, default=Config.sync,
                    help="stub (TODO), not implemented -- raises NotImplementedError if set True")
    p.add_argument("--n_refine_passes", type=_tuple_arg, default=Config.n_refine_passes,
                    help="1 (default) = current single-pass decode, unchanged. >1: after pass 1 "
                         "(unchanged, ctx-only), each further pass re-decodes every group with "
                         "extra causal peer context -- refine_window preceding groups' PREVIOUS "
                         "PASS decoded codes at this same level (own predictions during training, "
                         "stop-gradient'd; own generated codes during generation) -- not real "
                         "ground truth, so train and generate see the same (imperfect) signal. "
                         "More passes = closer to full AR across groups, but each pass is a full "
                         "extra decode call (cost scales ~linearly with n_refine_passes)")
    p.add_argument("--refine_window", type=_tuple_arg, default=Config.refine_window,
                    help="n_refine_passes>1 only: how many preceding groups' previous-pass decoded "
                         "codes are visible as extra causal context each refinement pass (like "
                         "ncodes_window, but sourced from this level's own decode output, not the "
                         "level above's ctx)")
    p.add_argument("--multipass_detach", type=lambda x: x.lower() != "false", default=Config.multipass_detach,
                    help="refine passes (n_refine_passes>1): True (default) fully stop_gradient's "
                         "the draft (own prediction, no gradient reaches the earlier pass that "
                         "produced it). False: STE instead, so the last pass's loss gradient flows "
                         "back through every earlier pass's decoder output -- backward cost grows "
                         "with n_refine_passes (BPTT-like), unlike the detached default")
    p.add_argument("--refine_gumbel", type=lambda x: x.lower() != "false", default=Config.refine_gumbel,
                    help="refine passes only: gumbel-perturb which code gets drafted each pass "
                         "(own dedicated knob, NOT the encoder's quantize_mode). Default False "
                         "(plain deterministic argmax draft)")
    p.add_argument("--refine_temperature", type=float, default=Config.refine_temperature,
                    help="own dedicated temperature, not shared with the encoder's "
                         "encode_temperature. Used by refine_gumbel=True's gumbel-softmax, and "
                         "also as the plain argmax path's tau when refine_quantize_drop>0 (shapes "
                         "the soft component that gets mixed into the draft)")
    p.add_argument("--refine_quantize_drop", type=float, default=Config.refine_quantize_drop,
                    help="refine passes only, and only when multipass_detach=False -- probability "
                         "of mixing the soft (not STE-hard) code into the draft embedding. Own "
                         "dedicated knob, not shared with the encoder's quantize_drop. No effect "
                         "under multipass_detach=True (detached passes only ever use the hard "
                         "argmax index)")
    p.add_argument("--refine_remat", type=lambda x: None if x.lower() == "none" else x.lower() != "false",
                    default=Config.refine_remat,
                    help="overrides --remat for just the refine passes (pass 1 always uses --remat "
                         "unchanged). 'none' (default): refine passes also use --remat, unchanged")
    p.add_argument("--weight_sharing", type=_bool_tuple_arg, default=Config.weight_sharing)
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--curriculum_mode", type=str, default=Config.curriculum_mode, choices=["freeze", "no_freeze"])
    p.add_argument("--quantize_mode", type=str, default=Config.quantize_mode, choices=["argmax", "gumbel"])
    p.add_argument("--quantize_drop", type=float, default=Config.quantize_drop)
    p.add_argument("--encode_temperature", type=_float_tuple_arg, default=(1.0,),
                    help="gumbel-softmax temperature -- global (not per-level), per-phase tuple. "
                         "A bare scalar broadcasts to every phase")
    p.add_argument("--gumbel_at_inference", type=lambda x: x.lower() != "false", default=Config.gumbel_at_inference)
    p.add_argument("--level_gt_drop", type=_float_tuple_arg, default=(0.5,),
                    help="probability of using cascade-simulated rollout (dropping ground-truth "
                         "ctx) at each level transition during training -- independent draw per "
                         "level, not one shared draw for the whole step. Per-phase tuple (bare "
                         "scalar broadcasts to every phase); each phase entry may itself be a "
                         "scalar (same prob for every level transition) or a tuple (one prob per "
                         "level, config.py only -- not expressible on the CLI)")
    p.add_argument("--layer_drop_prob", type=_float_tuple_arg, default=(0.0,),
                    help="stochastic-depth drop probability per transformer layer -- bare scalar "
                         "broadcasts to every phase uniformly; a flat tuple (length n_phases) "
                         "gives one value per phase; a nested tuple-of-tuples (config.py only, "
                         "not expressible on the CLI) gives one value per phase per layer")
    p.add_argument("--feedback_p", type=_float_tuple_arg, default=(0.0,),
                    help="probability per level of an additive self-feedback pass: stop-gradient "
                         "argmax that level's own decode reconstruction and re-decode the levels "
                         "below it (or, for level0, the whole model) on that pseudo input, adding "
                         "the loss unweighted on top -- tests idempotence/robustness to its own "
                         "predictions, gradient never reaches the level whose output was argmax'd. "
                         "Per-phase tuple (bare scalar broadcasts to every phase); each phase entry "
                         "may itself be a scalar or a per-level tuple (config.py only)")
    p.add_argument("--feedback_detach", type=lambda x: x.lower() != "false", default=True,
                    help="feedback_p's i>0 path: True (default) fully stop-gradients the pseudo "
                         "ctx; False leaves quantize_hard's straight-through path intact so "
                         "gradient reaches level i's decoder (torch encoder_ste_p additive+STE "
                         "shape). i==0's recursive whole-model pass is always fully detached "
                         "regardless (argmax has no gradient either way)")
    p.add_argument("--init_scheme", type=str, default=Config.init_scheme, choices=["llama", "zero"])
    p.add_argument("--use_xsa", type=lambda x: x.lower() != "false", default=Config.use_xsa)
    p.add_argument("--use_qknorm", type=lambda x: x.lower() != "false", default=Config.use_qknorm)
    p.add_argument("--remat", type=lambda x: x.lower() != "false", default=Config.remat)
    p.add_argument("--attn_window", type=_tuple_arg, default=Config.attn_window)
    p.add_argument("--attn_lookahead", type=_tuple_arg, default=Config.attn_lookahead,
                    help="encoder self-attention shifted-triangular lookahead -- 0 (default) "
                         "plain causal, int>0 query may additionally see keys up to that many "
                         "positions ahead (splash LocalMask's native right-side window)")
    p.add_argument("--use_sink", type=lambda x: x.lower() != "false", default=Config.use_sink)
    p.add_argument("--byte_group", type=int, default=Config.byte_group)
    p.add_argument("--token_head_type", type=str, default=Config.token_head_type)
    p.add_argument("--token_dim", type=_tuple_arg, default=Config.token_dim)
    p.add_argument("--token_n_heads", type=_tuple_arg, default=Config.token_n_heads)
    p.add_argument("--pq_dim", type=_tuple_arg, default=Config.pq_dim)
    p.add_argument("--token_mask_prob", type=float, default=Config.token_mask_prob)
    p.add_argument("--mtp_horizon", type=_tuple_arg, default=Config.mtp_horizon)
    p.add_argument("--mtp_mode", type=str, default=Config.mtp_mode)
    p.add_argument("--mtp_weight", type=float, default=Config.mtp_weight)
    p.add_argument("--entropy_weight", type=float, default=Config.entropy_weight)
    p.add_argument("--mse_weight", type=float, default=Config.mse_weight)
    p.add_argument("--mse_softmax_tau", type=float, default=Config.mse_softmax_tau)
    p.add_argument("--label_reg_weight", type=float, default=Config.label_reg_weight,
                    help="auxiliary regularization: cross-entropy each level's own code_head "
                         "logits against a pseudo-label built by downsampling the real image to "
                         "that level's own block-grid resolution and bit-packing the resulting "
                         "byte value into that level's (pq_chunks, code_vocab) shape (default "
                         "label generator: default_label_fn_jax, pure-JAX/on-device; a slower "
                         "PIL-based alternative, default_label_fn_pil, is also provided -- set "
                         "'label_fn' in a config.py to swap it, not CLI-representable). Default "
                         "0.0 (off)")
    p.add_argument("--traversal", type=str, default=Config.traversal, choices=["raster", "zorder"])
    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    label_fn = config_vars.pop("label_fn", default_label_fn_jax)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    if unknown:
        p.error(f"--config {pre_args.config} sets unknown field(s): {sorted(unknown)}")
    p.set_defaults(**config_vars)
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = pre_args.config.stem

    def _resolve_pair(step_name, epoch_name, default_step=None):
        s, e = getattr(args, step_name), getattr(args, epoch_name)
        assert s is None or e is None, \
            f"at most one of --{step_name}/--{epoch_name} may be set (got {step_name}={s}, {epoch_name}={e})"
        if s is None and e is None and default_step is not None:
            setattr(args, step_name, default_step)

    _resolve_pair("level_steps", "level_epochs", default_step=None)
    if args.level_steps is None and args.level_epochs is None:
        args.level_epochs = (1000,)
    _resolve_pair("warmup_steps", "warmup_epochs", default_step=100)
    _resolve_pair("lr_min_step", "lr_min_epoch")
    _resolve_pair("gen_eval_every_step", "gen_eval_every_epoch")
    if args.gen_eval_every_step is None and args.gen_eval_every_epoch is None:
        args.gen_eval_every_epoch = 10
    _resolve_pair("ckpt_every_step", "ckpt_every_epoch")
    if args.ckpt_every_step is None and args.ckpt_every_epoch is None:
        args.ckpt_every_epoch = 10
    _resolve_pair("wa_every_step", "wa_every_epoch")
    if args.wa_every_step is None and args.wa_every_epoch is None:
        args.wa_every_epoch = 10

    n_devices = args.n_devices or jax.local_device_count()
    print(f"jax devices ({n_devices} used of {jax.local_device_count()} local): {jax.devices()}")
    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})
    n_levels = len(cfg.strides)
    top_level_trainable = cfg.strides[-1] != -1
    n_phases = n_levels if top_level_trainable else n_levels - 1
    n_positions = n_positions_of(cfg)
    pixel_order = pixel_order_for(cfg)

    def _bcast_per_phase(name):
        val = getattr(args, name)
        if val is None:
            return
        if isinstance(val, (int, float)):
            val = (val,) * n_phases
        elif len(val) == 1:
            val = val * n_phases
        assert len(val) == n_phases, f"{name} has {len(val)} entries, need {n_phases} (one per phase)"
        setattr(args, name, val)

    _bcast_per_phase("level_steps")
    _bcast_per_phase("level_epochs")
    _bcast_per_phase("batch_size")
    _bcast_per_phase("val_batch_size")
    _bcast_per_phase("encode_temperature")
    _bcast_per_phase("level_gt_drop")
    _bcast_per_phase("layer_drop_prob")
    _bcast_per_phase("feedback_p")

    if args.dataset == "imagenet64":
        (train_np, train_labels), (val_np, val_labels) = load_imagenet64(Path(args.data_root))
    else:
        (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    if args.val_subset_n:
        val_np = val_np[:args.val_subset_n]

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
    recon_prompt = flat_prompt = gt_img = None
    train_recon_prompt = train_flat_prompt = train_gt_img = None
    gen_jit_timed = [False]

    def run_gen_eval(eval_model, top: int, tag: str, flat_prompt, gt_img) -> tuple:
        gen_t0 = time.monotonic()
        m = cast_pytree(eval_model, compute_dtype)
        x = code_embed_proj(flat_prompt, m.levels[0].own_input_embed, m.levels[0].own_input_proj)
        target = flat_prompt
        codes, codes_soft = [], []
        eval_rngs = ([None] * (top + 1) if not cfg.gumbel_at_inference
                     else list(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), hash(tag) % (2**31)), top + 1)))
        for i in range(top + 1):
            out = m.levels[i].encode(x, target, rng=eval_rngs[i], encode_temperature=args.encode_temperature[phase - 1])
            codes.append(out["code_idx"])
            codes_soft.append(out["code_soft"])
            if i < top:
                x = code_embed_proj(out["code_soft"], m.levels[i + 1].own_input_embed, m.levels[i + 1].own_input_proj)
                target = out["code_idx"]

        recon_acc = recon_mse = None

        cascade_t0 = time.monotonic()
        cur_code = codes[top]
        for i in range(top, 0, -1):
            cur_code = decode_generate_multipass(m.levels[i], cur_code, cfg.decoder_ncodes[i], greedy=True, seed=0)
        cascade_recon = decode_generate_multipass(m.levels[0], cur_code, cfg.decoder_ncodes[0], greedy=True, seed=0)
        gen_compile_s = None
        if not gen_jit_timed[0]:
            gen_compile_s = time.monotonic() - cascade_t0
            gen_jit_timed[0] = True
        cascade_acc = float(jnp.mean(cascade_recon == flat_prompt))
        cascade_img = positions_to_image(np.asarray(cascade_recon), cfg, pixel_order)
        cascade_mse = pixel_mse(cascade_img, gt_img)
        save_compare_grid(cascade_img, gt_img, run_dir / f"samples_{tag}.png")

        gen_time_s = time.monotonic() - gen_t0
        msg = f"[{tag}] top={top} CASCADE gen_byte_acc={cascade_acc:.4f} gen_cascade_mse={cascade_mse:.2f}"
        rec = dict(tag=tag, gen_cascade_acc=cascade_acc, gen_cascade_mse=cascade_mse, gen_time_s=gen_time_s)
        msg += f" gen_time={gen_time_s:.1f}s"
        if gen_compile_s is not None:
            msg += f" (first call, incl. jit compile: {gen_compile_s:.1f}s)"
            rec["gen_compile_s"] = gen_compile_s
        logger(msg, **rec)
        if args.verbose and cascade_img.shape[0] < 10:
            per_sample_mse = [pixel_mse(cascade_img[i:i + 1], gt_img[i:i + 1]) for i in range(cascade_img.shape[0])]
            logger(" ".join(f"mse{i + 1}={m:.2f}" for i, m in enumerate(per_sample_mse)))
        return recon_acc, cascade_acc

    def run_gen_eval_both(eval_model, top: int, tag: str) -> tuple:
        result = run_gen_eval(eval_model, top, f"{tag}_val", flat_prompt, gt_img)
        if args.eval_gen_train:
            run_gen_eval(eval_model, top, f"{tag}_train", train_flat_prompt, train_gt_img)
        return result

    val_eval_jit = eqx.filter_jit(phase_forward)
    val_jit_timed = [False]

    def run_val_eval(eval_model, phase: int, tag: str) -> tuple:
        val_t0 = time.monotonic()
        m = cast_pytree(eval_model, compute_dtype)
        bs = args.val_batch_size[phase - 1]
        n = len(val_np)
        sums = np.zeros(6, dtype=np.float64)
        total_loss = 0.0
        total_n = 0
        val_compile_s = None
        for start in range(0, n, bs):
            batch_imgs = val_np[start:start + bs]
            bn = len(batch_imgs)
            batch_flat = jnp.array(images_to_positions(batch_imgs, cfg, pixel_order))
            batch_t0 = time.monotonic()
            loss_b, aux_b = val_eval_jit(m, batch_flat, phase, rng=None,
                                          encode_temperature=args.encode_temperature[phase - 1],
                                          label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                          pixel_order=pixel_order)
            if not val_jit_timed[0]:
                val_compile_s = time.monotonic() - batch_t0
                val_jit_timed[0] = True
            sums += bn * np.array([float(a) for a in aux_b])
            total_loss += bn * float(loss_b)
            total_n += bn
        bpb, acc, ntp_bpb, ntp_acc, util, val_mse = (sums / total_n).tolist()
        loss = total_loss / total_n
        val_time_s = time.monotonic() - val_t0
        msg = (f"[{tag}] VAL loss={loss:.2f} val_dec_acc={acc:.2f} val_mse={val_mse:.4f} "
               f"val_ntp_acc={ntp_acc:.2f} val_time={val_time_s:.1f}s")
        rec = dict(tag=tag, val_loss=loss, val_dec_acc=acc, val_dec_bpb=bpb,
                    val_ntp_acc=ntp_acc, val_ntp_bpb=ntp_bpb, val_util=util, val_mse=val_mse,
                    val_time_s=val_time_s)
        if val_compile_s is not None:
            msg += f" (first batch, incl. jit compile: {val_compile_s:.1f}s)"
            rec["val_compile_s"] = val_compile_s
        logger(msg, **rec)
        return loss, acc

    if args.wa_mode == "wma" and args.wa_wma_weights is not None:
        assert len(args.wa_wma_weights) == args.wa_stack_size, \
            f"wa_wma_weights has {len(args.wa_wma_weights)} entries, need " \
            f"wa_stack_size={args.wa_stack_size}"

    def _phase_total_steps(idx, steps_per_epoch):
        if args.level_steps is not None:
            return args.level_steps[idx]
        return round(args.level_epochs[idx] * steps_per_epoch)

    def _every_steps(step_val, epoch_val, steps_per_epoch):
        return step_val if step_val is not None else round(epoch_val * steps_per_epoch)

    step = resume_meta["step"] if resume_meta else 0
    all_phases = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    total_all_steps = sum(
        _phase_total_steps(p - 1, len(train_np) // (args.batch_size[p - 1] * n_devices))
        for p in all_phases)
    global_pbar = tqdm(total=total_all_steps, initial=step, desc="total", dynamic_ncols=True, position=1, leave=True)
    last_global_step = step
    phase_iter = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    if resume_meta is not None:
        resume_phase = resume_meta["phase"]
        steps_per_epoch_resume = len(BatchIterator(
            train_np, train_labels[:len(train_np)], args.batch_size[resume_phase - 1], n_devices,
            shuffle=True, seed=args.seed, cfg=cfg))
        phase_steps_resume = _phase_total_steps(resume_phase - 1, steps_per_epoch_resume)
        phase_complete = resume_meta["phase_step"] >= phase_steps_resume
        phase_iter = [p for p in phase_iter if p > resume_phase] if phase_complete \
            else [p for p in phase_iter if p >= resume_phase]
    for phase in phase_iter:
        train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size[phase - 1],
                                    n_devices, shuffle=True, seed=args.seed, cfg=cfg)
        recon_prompt = val_np[:args.val_batch_size[phase - 1]]
        flat_prompt = jnp.array(images_to_positions(recon_prompt, cfg, pixel_order))
        gt_img = recon_prompt.astype(np.uint8)
        if args.eval_gen_train:
            train_recon_prompt = train_np[:args.val_batch_size[phase - 1]]
            train_flat_prompt = jnp.array(images_to_positions(train_recon_prompt, cfg, pixel_order))
            train_gt_img = train_recon_prompt.astype(np.uint8)

        filter_spec = phase_trainable_filter(model, phase)
        diff_model, static_model = eqx.partition(model, filter_spec)

        encode_temperature_phase = args.encode_temperature[phase - 1]
        level_gt_drop_phase = args.level_gt_drop[phase - 1]
        layer_drop_prob_phase = args.layer_drop_prob[phase - 1]
        feedback_p_phase = args.feedback_p[phase - 1]

        def loss_fn(diff_model, static_model, flat_bytes, rng, cascade_rng, feedback_rng, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            return phase_forward(m, flat_bytes, phase, rng=rng,
                                  level_gt_drop=level_gt_drop_phase, cascade_rng=cascade_rng,
                                  encode_temperature=encode_temperature_phase,
                                  layer_drop_prob=layer_drop_prob_phase,
                                  label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                  pixel_order=pixel_order,
                                  feedback_p=feedback_p_phase, feedback_rng=feedback_rng,
                                  feedback_detach=args.feedback_detach)

        steps_per_epoch_lr = len(train_iter)
        phase_total_steps = _phase_total_steps(phase - 1, steps_per_epoch_lr)
        total_steps = phase_total_steps
        warmup_steps_resolved = _every_steps(args.warmup_steps, args.warmup_epochs, steps_per_epoch_lr)
        min_step = (args.lr_min_step if args.lr_min_step is not None
                    else round(args.lr_min_epoch * steps_per_epoch_lr) if args.lr_min_epoch is not None
                    else phase_total_steps)
        lr_decay_steps = max(1, min_step - warmup_steps_resolved)
        lr_schedule = make_lr_schedule(args.lr_schedule, args.lr, warmup_steps_resolved, total_steps,
                                        end_value=args.lr_min, decay_steps=lr_decay_steps)
        if args.optimizer == "sinkgd":
            optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
        else:
            optimizer = optax.adamw(lr_schedule, **args.optimizer_kwargs)
        if args.grad_clip is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip), optimizer)
        opt_state = optimizer.init(diff_model)

        def train_step(diff_model, opt_state, rng, flat_bytes, static_model=static_model):
            rng, level_rng, cascade_rng, feedback_rng = jax.random.split(rng, 4)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                diff_model, static_model, flat_bytes, level_rng, cascade_rng, feedback_rng)
            grads = jax.lax.pmean(grads, axis_name="d")
            loss = jax.lax.pmean(loss, axis_name="d")
            aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
            grad_norm = optax.global_norm(grads)
            aux = aux + (grad_norm,)
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            return diff_model, opt_state, rng, loss, aux

        train_step = jax.pmap(train_step, axis_name="d")
        p_diff_model = replicate(diff_model, n_devices)
        p_opt_state = replicate(opt_state, n_devices)
        p_rng = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), phase), n_devices)

        start_phase_step = 0
        if resume_meta is not None and phase == resume_meta["phase"]:
            p_opt_state = replicate(
                eqx.tree_deserialise_leaves(resume_ckpt_dir / "opt_state.eqx", opt_state), n_devices)
            p_rng = eqx.tree_deserialise_leaves(resume_ckpt_dir / "p_rng.eqx", p_rng)
            train_iter.rng.bit_generator.state = json.loads(
                (resume_ckpt_dir / "dataloader_state.json").read_text())
            start_phase_step = resume_meta["phase_step"]
            logger(f"resumed phase {phase}: optimizer/rng/dataloader state restored, "
                   f"continuing from phase_step {start_phase_step}")

        active_desc = f"level{phase - 1}"
        logger(f"=== starting {active_desc} for {phase_total_steps / steps_per_epoch_lr:.3g} "
               f"epochs ({phase_total_steps} steps) ===")

        steps_per_epoch = len(train_iter)
        gen_eval_every_steps = _every_steps(args.gen_eval_every_step, args.gen_eval_every_epoch, steps_per_epoch)
        ckpt_every_steps = _every_steps(args.ckpt_every_step, args.ckpt_every_epoch, steps_per_epoch)
        wa_every_steps = _every_steps(args.wa_every_step, args.wa_every_epoch, steps_per_epoch)

        wa_ema = None
        wa_stack = deque(maxlen=args.wa_stack_size)
        wa_dir = run_dir / "checkpoints" / "wa"

        pbar = tqdm(total=phase_total_steps, initial=start_phase_step, desc=active_desc, dynamic_ncols=True, position=0)
        jit_timed = False
        phase_step = start_phase_step
        epoch_num = start_phase_step // steps_per_epoch
        while phase_step < phase_total_steps:
            epoch_num += 1
            if args.epoch_verbose:
                logger(f"{active_desc}: epoch {epoch_num} (step {step})")
            global_pbar.update(step - last_global_step)
            last_global_step = step
            for flat in train_iter:
                if phase_step >= phase_total_steps:
                    break
                flat = jnp.array(flat)
                if not jit_timed:
                    jit_t0 = time.monotonic()
                p_diff_model, p_opt_state, p_rng, loss, aux = train_step(p_diff_model, p_opt_state, p_rng, flat)
                step += 1
                phase_step += 1
                pbar.update(1)
                loss0 = float(loss[0])
                if not jit_timed:
                    logger(f"{active_desc}: first train_step (incl. jit compile) took "
                           f"{time.monotonic() - jit_t0:.1f}s")
                    jit_timed = True
                bpb, acc, ntp_bpb, ntp_acc, util, train_mse, grad_norm = [float(a[0]) for a in aux]
                lr = float(lr_schedule(step - 1))
                lr_str = _fmt_lr(lr)
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}",
                                  acc=f"{acc:.2f}",
                                  lr=lr_str, gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    logger(f"l={phase - 1} e={epoch_num} s={step} loss={loss0:.2f} dec_acc={acc:.2f} "
                           f"ntp_acc={ntp_acc:.2f} util={util:.2f} train_mse={train_mse:.1f} "
                           f"lr={lr_str} grad_norm={grad_norm:.2f}",
                           level=phase - 1, epoch=epoch_num, step=step, loss=loss0, dec_bpb=bpb,
                           dec_acc=acc, ntp_bpb=ntp_bpb, ntp_acc=ntp_acc, util=util,
                           train_mse=train_mse, lr=lr, grad_norm=grad_norm)

                if step % gen_eval_every_steps == 0:
                    snapshot = eqx.combine(to_single_device(unreplicate(p_diff_model)), static_model)
                    run_val_eval(snapshot, phase, tag=f"level{phase - 1}_step{step}")
                    run_gen_eval_both(snapshot, top=phase - 1, tag=f"level{phase - 1}_step{step}")

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
                        if args.wa_verbose:
                            logger(f"wa (ema) snapshot saved at step {step}")
                    else:
                        wa_stack.append(cur_diff_model)
                        if len(wa_stack) == args.wa_stack_size:
                            avg = stack_average(list(wa_stack), weights=args.wa_wma_weights)
                            eqx.tree_serialise_leaves(wa_dir / f"wma_phase{phase}_step{step}.eqx", avg)
                            if args.wa_verbose:
                                logger(f"wa (wma, n={len(wa_stack)}) average saved at step {step}")

        diff_model = to_host(unreplicate(p_diff_model))
        model = eqx.combine(diff_model, static_model)
        freeze_msg = "no freeze (no_freeze mode)" if cfg.curriculum_mode == "no_freeze" else f"freezing level {phase - 1}"
        logger(f"=== {active_desc} done, {freeze_msg} ===")
        ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
        save_checkpoint(ckpt_dir, model, to_host(unreplicate(p_opt_state)), to_host(p_rng), train_iter,
                         phase=phase, phase_step=phase_total_steps, step=step, seed=args.seed)
        prune_checkpoints(run_dir, args.ckpt_keep)
        if args.final_eval:
            run_val_eval(model, phase, tag=f"level{phase - 1}_final")
            run_gen_eval_both(model, top=phase - 1, tag=f"level{phase - 1}_final")

    global_pbar.update(step - last_global_step)
    global_pbar.close()
    logger("=== all phases done, running final top-down cascade eval ===")
    run_val_eval(model, n_levels - 1, tag="final")
    run_gen_eval_both(model, top=n_levels - 2, tag="final")
    logger("training done")


if __name__ == "__main__":
    main()
