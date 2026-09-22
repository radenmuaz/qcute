from __future__ import annotations

import argparse
import copy
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


def n_blocks_for_level(cfg, j: int) -> int:
    code_count = total_bytes_of(cfg) // cfg.byte_group
    for i in range(j + 1):
        K_i = cfg.strides[i] if cfg.strides[i] != -1 else 1
        code_count = code_count // K_i
    return code_count


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
    stream_chunks: tuple = 0
    decode_past: tuple = 0
    decode_future: tuple = 0
    sync: tuple = False
    level_refine_passes: tuple = 1
    level_refine_window: tuple = 0
    cond_depth: tuple = 1
    cond_drop: tuple = 0.0
    cond_window: tuple = -1
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
    dec_attn_window: tuple = -1
    attn_lookahead: tuple = 0
    use_sink: bool = False

    remat: bool = False
    remat_level: bool = False

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
    level_refine_gumbel: bool = False
    level_refine_gt_drop: float = 1.0
    level_refine_drop: float = 0.0
    gen_temperature: float = 1.0
    gen_top_k: int = 8
    gen_sync: tuple = False
    dense_decode: tuple = False
    interleave_decode: tuple = False
    level_refine_temperature: float = 1.0
    refine_quantize_drop: float = 0.0
    refine_remat: bool = None

    cycle_refine_passes: int = 1
    cyclic_revise_detach: bool = True
    cyclic_revise_remat: bool = True

    def _chunk_codes(self, i: int):
        # parent codes per chunk at level i (None when it cannot be resolved statically); 0 chunks = one group
        sc, G = self.stream_chunks[i], self.decoder_ncodes[i]
        if sc <= 0:
            return G
        code_count = total_bytes_of(self) // self.byte_group
        for j in range(i + 1):
            code_count //= self.strides[j] if self.strides[j] != -1 else 1
        n_groups = -(-code_count // G)
        return -(-n_groups // sc) * G

    def __post_init__(self):
        if self.remat and self.remat_level:
            warnings.warn("remat and remat_level both set: remat_level wins (whole encoder/decoder stacks are "
                          "checkpointed, not individual blocks)")
        n = len(self.strides)

        def bcast(name, types):
            val = getattr(self, name)
            if isinstance(val, types):
                setattr(self, name, (val,) * n)

        bcast("mlp_mult", int)
        bcast("rope_base", (int, float))
        bcast("decoder_ncodes", int)
        bcast("ncodes_window", int)
        bcast("stream_chunks", int)
        bcast("gen_sync", bool)
        bcast("dense_decode", bool)
        bcast("interleave_decode", bool)
        bcast("decode_past", int)
        bcast("decode_future", int)
        bcast("sync", bool)
        bcast("level_refine_passes", int)
        bcast("level_refine_window", int)
        bcast("cond_depth", int)
        bcast("cond_drop", (int, float))
        bcast("cond_window", int)
        bcast("weight_sharing", bool)
        bcast("token_head_type", str)
        bcast("token_dim", int)
        bcast("token_n_heads", int)
        bcast("mtp_horizon", int)
        bcast("mtp_mode", str)
        bcast("attn_window", int)
        bcast("dec_attn_window", int)
        bcast("attn_lookahead", int)
        if self.pq_dim is None:
            self.pq_dim = self.d_model
        else:
            bcast("pq_dim", int)

        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.decoder_ncodes) == n
        assert len(self.ncodes_window) == n and len(self.stream_chunks) == n and len(self.gen_sync) == n and len(self.dense_decode) == n and len(self.interleave_decode) == n
        assert len(self.decode_past) == n and len(self.decode_future) == n and len(self.sync) == n
        assert len(self.level_refine_passes) == n and len(self.level_refine_window) == n
        assert len(self.cond_depth) == n
        for i in range(n):
            assert self.ncodes_window[i] >= -1, \
                f"level {i}: ncodes_window={self.ncodes_window[i]} must be -1 (all) or >=0 " \
                f"(disjoint at 0, bounded lookback above)"
            assert self.stream_chunks[i] >= 0, \
                f"level {i}: stream_chunks={self.stream_chunks[i]} must be 0 (per-group streaming) or >=1 (chunks)"
            assert self.decode_past[i] >= 0 and self.decode_future[i] >= 0, \
                f"level {i}: decode_past={self.decode_past[i]}/decode_future={self.decode_future[i]} must be >=0"
            assert self.level_refine_passes[i] >= 1, \
                f"level {i}: level_refine_passes={self.level_refine_passes[i]} must be >=1 (1=off, current behavior)"
            assert self.level_refine_window[i] >= 0, \
                f"level {i}: level_refine_window={self.level_refine_window[i]} must be >=0"
            assert 1 <= self.cond_depth[i] <= n - i, \
                f"level {i}: cond_depth={self.cond_depth[i]} must be >=1 (1=own level only, current " \
                f"behavior) and <= {n - i} (can't condition past the top level)"
            assert self.cond_window[i] == -1 or self.cond_window[i] >= 1, \
                f"level {i}: cond_window={self.cond_window[i]} must be -1 (unbounded) or >=1"
            if self.cond_window[i] != -1 and self.cond_depth[i] <= 1:
                warnings.warn(f"level {i}: cond_window={self.cond_window[i]} has NO EFFECT (needs cond_depth>1)")
            if self.cond_depth[i] > 1:
                up_stride = 1
                for k in range(1, self.cond_depth[i]):
                    up_stride *= self.strides[i + k] if self.strides[i + k] != -1 else 1
                    chunk_codes = self._chunk_codes(i)
                    if chunk_codes is not None and chunk_codes % up_stride != 0:
                        warnings.warn(
                            f"level {i}: cond_depth={self.cond_depth[i]} extra level {i + k} is coarser by a "
                            f"cumulative stride of {up_stride} but decoder_ncodes[{i}]={self.decoder_ncodes[i]} "
                            f"is not a multiple of it -- using the strictly causal 'complete' alignment: group g "
                            f"sees only level-{i + k} codes whose whole span ends at or before its visible end "
                            f"(floor(end/{up_stride})), so the level-{i + k} code covering the group's own span "
                            f"stays hidden until that span finishes. Use a decoder_ncodes / chunk size that is a "
                            f"multiple of {up_stride}, or stream_chunks=1, to see it")
            assert 0.0 <= self.cond_drop[i] <= 1.0, \
                f"level {i}: cond_drop={self.cond_drop[i]} must be in [0,1]"
            if self.cond_drop[i] > 0 and self.cond_depth[i] <= 1:
                warnings.warn(
                    f"level {i}: cond_drop={self.cond_drop[i]} has NO EFFECT with cond_depth="
                    f"{self.cond_depth[i]} (no extra ctx blocks to drop)")
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
        assert len(self.attn_window) == n and len(self.attn_lookahead) == n and len(self.dec_attn_window) == n
        for i in range(n):
            assert self.attn_window[i] == -1 or self.attn_window[i] >= 1, \
                f"level {i}: attn_window={self.attn_window[i]} must be -1 (unbounded/flash) or >=1 (splash LocalMask)"
            assert self.dec_attn_window[i] == -1 or self.dec_attn_window[i] >= 1, \
                f"level {i}: dec_attn_window={self.dec_attn_window[i]} must be -1 (unbounded) or >=1"
            if self.dense_decode[i] and (self.cond_depth[i] > 1 or self.decode_future[i] != 0
                                          or self.level_refine_passes[i] > 1 or self.gen_sync[i]):
                warnings.warn(
                    f"level {i}: dense_decode=True IGNORES cond_depth/decode_future/level_refine_passes/gen_sync "
                    f"entirely (decode_logits_and_target/decode_generate don't take extra ctx, a future tail, "
                    f"refine passes, or sync -- there's nothing to refine/sync, every step already sees the real "
                    f"whole prefix)")
            if self.dense_decode[i] and self.interleave_decode[i]:
                raise ValueError(f"level {i}: dense_decode and interleave_decode are mutually exclusive "
                                  f"(interleave_decode is dense_decode + cond_depth support)")
            if self.interleave_decode[i] and self.cond_depth[i] > 2:
                raise NotImplementedError(f"level {i}: interleave_decode only supports cond_depth<=2 (hardcoded, "
                                          f"one flat sequence with at most one coarser level's codes interleaved)")
            if self.interleave_decode[i] and (self.decode_future[i] != 0 or self.level_refine_passes[i] > 1
                                              or self.gen_sync[i]):
                warnings.warn(
                    f"level {i}: interleave_decode=True IGNORES decode_future/level_refine_passes/gen_sync "
                    f"entirely, same reasons as dense_decode")
            if self.dec_attn_window[i] != -1 and self.weight_sharing:
                warnings.warn(
                    f"level {i}: dec_attn_window={self.dec_attn_window[i]} has NO EFFECT with weight_sharing=True "
                    f"(decoder reuses the encoder's own blocks/window instead of its own dec_blocks)")
            assert self.attn_lookahead[i] >= 0, \
                f"level {i}: attn_lookahead={self.attn_lookahead[i]} must be >=0 (0=plain causal)"

        top_level_trainable = self.strides[-1] != -1
        code_count = total_bytes_of(self) // self.byte_group
        for i in range(n):
            K_i = self.strides[i] if self.strides[i] != -1 else 1
            code_count = code_count // K_i
            if i == n - 1 and not top_level_trainable:
                break
            n_blocks_i, G_i, N_i, S_i = code_count, self.decoder_ncodes[i], self.ncodes_window[i], self.stream_chunks[i]
            assert G_i >= 1, f"level {i}: decoder_ncodes={G_i} must be >=1"
            if G_i > n_blocks_i:
                warnings.warn(
                    f"level {i}: decoder_ncodes={G_i} exceeds n_blocks={n_blocks_i} (this level's "
                    f"own code count) -- clamps to one single group, same as decoder_ncodes="
                    f"{n_blocks_i} (the fully-sequential 'original' degenerate case); recommend "
                    f"setting decoder_ncodes={n_blocks_i} explicitly for clarity")
            n_groups_i = -(-n_blocks_i // G_i)
            assert S_i <= n_groups_i, f"level {i}: stream_chunks={S_i} exceeds n_groups={n_groups_i}"
            if 0 < S_i == n_groups_i:
                warnings.warn(f"level {i}: stream_chunks={S_i} == n_groups (same as 0, per-group streaming)")
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
            elif N_i == -1 and S_i == 0 and G_i < max(1, n_blocks_i // 8):
                warnings.warn(
                    f"level {i}: ncodes_window=-1 stream_chunks=0 (causal unbounded) with a small "
                    f"decoder_ncodes={G_i} relative to n_blocks={n_blocks_i} (n_groups={n_groups_i}) "
                    f"-- the naive causal window pads EVERY group to the FULL n_blocks width, so "
                    f"compute/memory scales as O(n_groups*n_blocks); recommend a larger "
                    f"decoder_ncodes or a bounded ncodes_window instead")
            if self.level_refine_window[i] > 0 and self.level_refine_passes[i] <= 1:
                warnings.warn(
                    f"level {i}: level_refine_window={self.level_refine_window[i]} has NO EFFECT with "
                    f"level_refine_passes={self.level_refine_passes[i]} (need >1 for a second pass to use "
                    f"it) -- either raise level_refine_passes or set level_refine_window=0 for clarity")
            if self.gen_sync[i] and self.level_refine_passes[i] > 1:
                warnings.warn(
                    f"level {i}: gen_sync=True with level_refine_passes={self.level_refine_passes[i]} -- refine "
                    f"passes are not yet composed with sync, they still redraft privately on top of the sync "
                    f"pass-1 output")
            if self.level_refine_passes[i] > 1 and self.level_refine_window[i] <= 0:
                warnings.warn(
                    f"level {i}: level_refine_passes={self.level_refine_passes[i]} runs extra passes with "
                    f"ZERO peer context (level_refine_window=0) -- each extra pass degenerates to "
                    f"recomputing pass 1 (wasted compute, not a no-op); set level_refine_window>0 or "
                    f"level_refine_passes=1")
        if self.level_refine_gumbel and not any(self.level_refine_passes[i] > 1 and self.level_refine_window[i] > 0 for i in range(n)):
            warnings.warn(
                "level_refine_gumbel=True has NO EFFECT -- no level has both level_refine_passes>1 and "
                "level_refine_window>0, so no refine pass ever runs")
        assert 0.0 <= self.level_refine_gt_drop <= 1.0, f"level_refine_gt_drop={self.level_refine_gt_drop} must be in [0,1]"
        assert 0.0 <= self.level_refine_drop < 1.0, f"level_refine_drop={self.level_refine_drop} must be in [0,1)"
        if self.level_refine_drop > 0.0 and not any(
                self.level_refine_passes[i] > 2 for i in range(n)) and not any(
                self.level_refine_passes[i] > 1 and self.level_refine_window[i] > 0 for i in range(n)):
            warnings.warn("level_refine_drop>0 has NO EFFECT -- no level runs a refine pass")
        if self.level_refine_gt_drop < 1.0 and not any(
                self.level_refine_passes[i] > 1 and self.level_refine_window[i] > 0 for i in range(n)):
            warnings.warn("level_refine_gt_drop<1 has NO EFFECT -- no level runs a refine pass")
        if self.refine_quantize_drop > 0 and self.multipass_detach:
            warnings.warn(
                f"refine_quantize_drop={self.refine_quantize_drop} has NO EFFECT with "
                f"multipass_detach=True (fully-detached refine passes only ever use the hard "
                f"argmax index, never the soft/drop-mixed code) -- set multipass_detach=False or "
                f"refine_quantize_drop=0 for clarity")
        assert self.cycle_refine_passes >= 1, \
            f"cycle_refine_passes={self.cycle_refine_passes} must be >=1 (1=off, current behavior)"
        if self.cycle_refine_passes > 1 and n < 2:
            warnings.warn(
                f"cycle_refine_passes={self.cycle_refine_passes} has NO EFFECT with only {n} level(s) "
                f"-- needs at least 2 (operates on the top two levels of whatever phase is active)")

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
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="cifar-10") as pbar:
            def _hook(n_blocks, block_size, total_size):
                if pbar.total is None and total_size > 0:
                    pbar.total = total_size
                pbar.update(n_blocks * block_size - pbar.n)
            urllib.request.urlretrieve(CIFAR10_URL, tmp_path, reporthook=_hook)
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


def load_imagenet(data_root: Path, resolution: int = 64, train_shards: int = None) -> tuple:
    def load_split(split: str, limit=None) -> np.ndarray:
        shards = sorted(data_root.glob(f"imagenet{resolution}_{split}_*.npy"))
        assert shards, f"no imagenet{resolution}_{split}_*.npy shards under {data_root} -- run " \
            f"image_lagcodec/scripts/imagenet/download_imagenet{resolution}.py --split {split} --out_dir {data_root}"
        parts = [np.load(s, mmap_mode="r") for s in shards[:limit]]
        return np.concatenate(parts, axis=0).reshape(-1, resolution, resolution, 3)

    train = load_split("train", train_shards)
    val = load_split("validation")
    return (train, np.zeros(len(train), dtype=np.int32)), (val, np.zeros(len(val), dtype=np.int32))


def load_imagenet64(data_root: Path, resolution: int = 64) -> tuple:
    return load_imagenet(data_root, resolution)


def load_dataset(name: str, data_root: Path, img_size: int = None, train_shards: int = None) -> tuple:
    # train_shards: only load the first N imagenet train shards (off-training scripts need a few images, not 15GB)
    if name == "cifar":
        res = 32
    else:
        assert name.startswith("imagenet"), f"unknown dataset {name!r}"
        res = int(name[len("imagenet"):])
    assert img_size is None or img_size == res, f"dataset {name} is {res}px but img_size={img_size}"
    return load_cifar10(data_root) if name == "cifar" else load_imagenet(data_root, res, train_shards)


def dataset_from_config(cv: dict, repo_root: Path, train_shards: int = 1) -> tuple:
    root = Path(cv.get("data_root") or repo_root / "datasets")
    return load_dataset(cv.get("dataset", "cifar"), root, cv.get("img_size"), train_shards)


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
        self.pc, self.pi = jax.process_count(), jax.process_index()
        self.total = batch_size * n_devices
        self.cfg = cfg
        self.pixel_order = pixel_order_for(cfg)
        self.n_positions = n_positions_of(cfg)

    def __len__(self):
        return len(self.images) // (self.total * self.pc)

    def __iter__(self):
        n = len(self.images)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        g = self.total * self.pc
        for start in range(0, n - g + 1, g):
            sel = idx[start + self.pi * self.total:start + (self.pi + 1) * self.total]
            img = self.images[sel]
            positions = images_to_positions(img, self.cfg, self.pixel_order)
            yield positions.reshape(self.n_devices, self.batch_size, self.n_positions, self.cfg.byte_group)


def safe_argmax(x: jnp.ndarray) -> jnp.ndarray:
    # first index of the max over the last axis. jnp.argmax fused into a following gather returns the max
    # value's float bits instead of the index under jit on TPU (XLA bug; seen at >=256 rows in generation),
    # so use plain max/min reductions.
    V = x.shape[-1]
    m = jnp.max(x, axis=-1, keepdims=True)
    return jnp.minimum(jnp.min(jnp.where(x == m, jnp.arange(V), V), axis=-1), V - 1)


def quantize_hard(logits: jnp.ndarray, rng=None, quantize_drop: float = 0.0, tau: float = 1.0) -> tuple:
    soft = jax.nn.softmax(logits / tau, axis=-1)
    idx = safe_argmax(soft)
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
    idx = safe_argmax(soft)
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


def extra_ctx_visible_counts(n_groups: int, G: int, up_stride: int, ends=None) -> list:
    # "complete" alignment: group g sees coarser code j iff its whole span [j*S,(j+1)*S) ends at or before the
    # group's visible end (default (g+1)*G, i.e. the group's own end), i.e. count = end(g) // S.
    ends = [(g + 1) * G for g in range(n_groups)] if ends is None else ends
    return [e // up_stride for e in ends]


def causal_extra_ctx_windows(extra_val, embed_table, proj_table, up_stride: int, G: int,
                              n_groups: int, B: int, D: int, window: int = -1, ends=None, groups=None) -> tuple:
    # per-group window of the coarser code; positions before the sequence start are zero-filled and marked invalid.
    # groups: optional subset of global group indices to materialize (default all) -- used by decode_generate_pardec_sync
    # to build only one wave's rows while keeping the SAME global chunk-boundary math (ends is for ALL n_groups).
    counts_all = extra_ctx_visible_counts(n_groups, G, up_stride, ends)
    full = max(counts_all)
    Wg_j = full if window < 0 else min(window, full)
    gidx = list(range(n_groups)) if groups is None else list(groups)
    counts = [counts_all[g] for g in gidx]
    extra_tok = code_embed_proj(extra_val, embed_table, proj_table)
    M = extra_tok.shape[1]
    eff = [min(c, M) for c in counts]
    padded = jnp.concatenate([jnp.zeros((B, Wg_j, D), extra_tok.dtype), extra_tok], axis=1)
    windows = jnp.stack([padded[:, c:c + Wg_j, :] for c in eff], axis=1)
    rope = jnp.stack([jnp.clip(jnp.arange(Wg_j) - Wg_j + c, 0, None) for c in eff], axis=0)
    valid = np.stack([(np.arange(Wg_j) - Wg_j + c) >= 0 for c in eff], axis=0)
    return windows.reshape(B * len(gidx), Wg_j, D), rope, valid, Wg_j


def _draft_past_valid_mask(n_groups: int, Pp: int, Kspan: int, valid_len: int) -> np.ndarray:
    abs_idx = np.array([[g * Kspan - Pp + t for t in range(Pp)] for g in range(n_groups)])
    return (abs_idx >= 0) & (abs_idx < valid_len)


def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


def sample_idx(logits: jnp.ndarray, rng, greedy: bool, temperature: float, top_k: int = 0) -> tuple:
    if greedy:
        return safe_argmax(logits), rng
    rng, k_ = jax.random.split(rng)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        lg = jnp.where(lg < jax.lax.top_k(lg, top_k)[0][..., -1:], -jnp.inf, lg)
    return safe_argmax(lg + jax.random.gumbel(k_, lg.shape)), rng


def run_block(blk: Block, x: jnp.ndarray, remat: bool, rng=None, drop_prob: float = 0.0) -> jnp.ndarray:
    out = eqx.filter_checkpoint(blk)(x) if remat else blk(x)
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
                 cache_pos, rope_pos: jnp.ndarray, key_valid: jnp.ndarray, T_max: int) -> tuple:
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
    valid = (idx[None, None, :] <= cache_pos) & key_valid[:, None, :]
    logits = jnp.where(valid, logits, -1e9)
    attn_w = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bht,bhtd->bhd", attn_w, v_full)
    if attn.use_xsa:
        v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
        y = apply_xsa(y, v_self)
    y = y.reshape(Bc, D)
    return y @ attn.out, cache_k, cache_v


def pardec_chunk_step(attn: Attention, x_chunk: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray,
                       cache_pos_start, rope_pos_ids: jnp.ndarray, key_valid: jnp.ndarray,
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
    valid = causal[None] & key_valid[:, None, :]
    logits = jnp.where(valid[:, None], logits, -1e9)
    attn_w = jax.nn.softmax(logits, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", attn_w, v_full)
    if attn.use_xsa:
        v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
        y = apply_xsa(y, v_self)
    y = y.transpose(0, 2, 1, 3).reshape(Bc, T, D)
    return y @ attn.out, cache_k, cache_v


def pardec_block_step(blk: Block, x_new, cache_k, cache_v, cache_pos, rope_pos, key_valid, T_max):
    attn_out, ck, cv = pardec_step(blk.attn, blk.norm1(x_new), cache_k, cache_v, cache_pos,
                                    rope_pos, key_valid, T_max)
    x = x_new + attn_out
    x = x + blk.mlp(blk.norm2(x))
    return x, ck, cv


def pardec_block_chunk_step(blk: Block, x_chunk, cache_k, cache_v, cache_pos_start, rope_pos_ids,
                             key_valid, T_max):
    attn_out, ck, cv = pardec_chunk_step(blk.attn, blk.norm1(x_chunk), cache_k, cache_v,
                                          cache_pos_start, rope_pos_ids, key_valid, T_max)
    x = x_chunk + attn_out
    x = x + blk.mlp(blk.norm2(x))
    return x, ck, cv


def dense_self_attention_pardec(attn: Attention, x: jnp.ndarray, rope_pos_ids: jnp.ndarray,
                                 key_valid: jnp.ndarray) -> jnp.ndarray:
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
    mask = causal[None] & key_valid[:, None, :]
    scores = jnp.where(mask[:, None], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    y = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    if attn.use_xsa:
        y = apply_xsa(y, v)
    y = y.transpose(0, 2, 1, 3).reshape(Bc, T, D)
    return y @ attn.out


def run_block_pardec(blk: Block, x: jnp.ndarray, rope_pos_ids: jnp.ndarray, key_valid: jnp.ndarray,
                      remat: bool) -> jnp.ndarray:
    def f(x):
        x = x + dense_self_attention_pardec(blk.attn, blk.norm1(x), rope_pos_ids, key_valid)
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
                       h: jnp.ndarray, rng, greedy: bool, temperature: float, top_k: int = 0) -> tuple:
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
        val_m, rng = sample_idx(logit_m, rng, greedy, temperature, top_k)
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
    remat_level: bool = eqx.field(static=True)
    gen_top_k: int = eqx.field(static=True)
    gen_sync: bool = eqx.field(static=True)
    dense_decode: bool = eqx.field(static=True)
    interleave_decode: bool = eqx.field(static=True)
    pq_dim: int = eqx.field(static=True)
    ncodes_window: int = eqx.field(static=True)
    stream_chunks: int = eqx.field(static=True)
    decode_past: int = eqx.field(static=True)
    decode_future: int = eqx.field(static=True)
    attn_lookahead: int = eqx.field(static=True)
    level_refine_passes: int = eqx.field(static=True)
    level_refine_window: int = eqx.field(static=True)
    cond_depth: int = eqx.field(static=True)
    cond_drop: float = eqx.field(static=True)
    cond_window: int = eqx.field(static=True)
    extra_ctx_embed: list
    extra_ctx_proj: list
    extra_ctx_n_blocks: tuple = eqx.field(static=True)
    extra_ctx_up_stride: tuple = eqx.field(static=True)
    revision_embed: jnp.ndarray
    revision_proj: jnp.ndarray
    revision_up_stride: int = eqx.field(static=True)
    revision_n_blocks: int = eqx.field(static=True)
    cyclic_revise_detach: bool = eqx.field(static=True)
    cyclic_revise_remat: bool = eqx.field(static=True)

    def __init__(self, key, cfg: Config, level: int, has_decoder: bool, weight_sharing: bool):
        D = cfg.d_model[level]
        self.K = cfg.strides[level] if cfg.strides[level] != -1 else 1
        self.n_heads, self.n_kv_heads = cfg.n_heads[level], cfg.n_kv_heads[level]
        self.quantize_mode = cfg.quantize_mode
        self.quantize_drop = cfg.quantize_drop
        self.remat = cfg.remat
        self.remat_level = cfg.remat_level
        self.gen_top_k = cfg.gen_top_k
        self.gen_sync = cfg.gen_sync[level]
        self.dense_decode = cfg.dense_decode[level]
        self.interleave_decode = cfg.interleave_decode[level]
        self.ncodes_window = cfg.ncodes_window[level]
        self.stream_chunks = cfg.stream_chunks[level]
        self.decode_past = cfg.decode_past[level]
        self.decode_future = cfg.decode_future[level]
        self.attn_lookahead = cfg.attn_lookahead[level]
        self.level_refine_passes = cfg.level_refine_passes[level]
        self.level_refine_window = cfg.level_refine_window[level]
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

        self.cond_depth = cfg.cond_depth[level]
        self.cond_drop = cfg.cond_drop[level]
        self.cond_window = cfg.cond_window[level]
        self.extra_ctx_embed, self.extra_ctx_proj = [], []
        extra_n_blocks, extra_up_stride = [], []
        if self.cond_depth > 1:
            extra_keys = jax.random.split(jax.random.fold_in(key, 9001), 3 * (self.cond_depth - 1))
            up_stride = 1
            for k in range(1, self.cond_depth):
                j = level + k
                self.extra_ctx_embed.append(
                    init_matrix(extra_keys[3 * (k - 1)], (cfg.code_vocab[j], cfg.pq_dim[j]), scheme))
                self.extra_ctx_proj.append(
                    init_matrix(extra_keys[3 * (k - 1) + 1], (cfg.pq_chunks[j] * cfg.pq_dim[j], D), scheme))
                extra_n_blocks.append(n_blocks_for_level(cfg, j))
                up_stride *= cfg.strides[j] if cfg.strides[j] != -1 else 1
                extra_up_stride.append(up_stride)
        self.extra_ctx_n_blocks = tuple(extra_n_blocks)
        self.extra_ctx_up_stride = tuple(extra_up_stride)

        self.cyclic_revise_detach = cfg.cyclic_revise_detach
        self.cyclic_revise_remat = cfg.cyclic_revise_remat if cfg.cyclic_revise_remat is not None else cfg.remat
        n_levels = len(cfg.d_model)
        if cfg.cycle_refine_passes > 1 and level + 1 < n_levels:
            j = level + 1
            rev_keys = jax.random.split(jax.random.fold_in(key, 9003), 3)
            self.revision_embed = init_matrix(rev_keys[0], (cfg.code_vocab[j], cfg.pq_dim[j]), scheme)
            self.revision_proj = init_matrix(rev_keys[1], (cfg.pq_chunks[j] * cfg.pq_dim[j], D), scheme)
            self.revision_up_stride = cfg.strides[j] if cfg.strides[j] != -1 else 1
            self.revision_n_blocks = n_blocks_for_level(cfg, j)
        else:
            self.revision_embed = self.revision_proj = None
            self.revision_up_stride = 1
            self.revision_n_blocks = 0

        if has_decoder and not weight_sharing:
            dec_block_keys = jax.random.split(keys[6], n_layers)
            dec_window = None if cfg.dec_attn_window[level] == -1 else cfg.dec_attn_window[level]
            self.dec_blocks = [Block(k, D, self.n_heads, self.n_kv_heads, cfg.mlp_mult[level], cfg.rope_base[level],
                                     n_layers=n_layers, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm,
                                     window=dec_window) for k in dec_block_keys]
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
        def _enc_stack(h):
            for i, blk in enumerate(self.blocks):
                h = run_block(blk, h, self.remat and not self.remat_level, rng=layer_rngs[i], drop_prob=layer_drop_prob[i])
            return h
        h = jax.checkpoint(_enc_stack)(h) if self.remat_level else _enc_stack(h)
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

    def _extra_ctx_table(self, k: int) -> tuple:
        # cond_depth's extra_ctx_* lists hold one dedicated table per (distinct, statically coarser)
        # level; cyclic-refine revision slots all reference the SAME coarser level (level+1) at
        # different refinement passes, so they share one table -- any k beyond the cond_depth list
        # falls back to it.
        n_cond = len(self.extra_ctx_embed)
        if k < n_cond:
            return (self.extra_ctx_embed[k], self.extra_ctx_proj[k],
                    self.extra_ctx_up_stride[k], self.extra_ctx_n_blocks[k])
        return self.revision_embed, self.revision_proj, self.revision_up_stride, self.revision_n_blocks

    def _pardec_ctx_rows(self, ctx_tok, extra_codes, G: int, n_groups: int, n_blocks: int, rng=None,
                          drop_extras: bool = False, groups=None) -> tuple:
        # Shared by training and generation. Row order: [extras (coarsest first) | own-level window]. Slots that do
        # not exist (before the sequence start, past the last real code, cond_drop) are zero-filled and marked
        # invalid: they stay in the row for static shapes but are masked as keys (never attended).
        # groups: optional subset of GLOBAL group indices to materialize (default all) -- lets a caller build just
        # one wave's rows (decode_generate_pardec_sync) while chunk boundaries stay computed over the full sequence.
        B, _, D = ctx_tok.shape
        n_blocks_p = n_groups * G
        gidx = list(range(n_groups)) if groups is None else list(groups)
        Bn = len(gidx)
        B2 = B * Bn
        # chunked visibility: group g sees parent codes up to end(g), the end of its chunk (0 = its own group)
        chunk_groups = 1 if self.stream_chunks <= 0 else -(-n_groups // self.stream_chunks)
        ends_all = [min((g // chunk_groups + 1) * chunk_groups, n_groups) * G for g in range(n_groups)]
        ends = [ends_all[g] for g in gidx]
        Wg = n_blocks_p if self.ncodes_window < 0 else min(self.ncodes_window * G + chunk_groups * G, n_blocks_p)
        padded = jnp.pad(ctx_tok, ((0, 0), (Wg, n_blocks_p - n_blocks), (0, 0)))
        own = jnp.stack([padded[:, e:e + Wg, :] for e in ends], axis=1).reshape(B2, Wg, D)
        rope_own = jnp.stack([jnp.clip(jnp.arange(Wg) - Wg + e, 0, None) for e in ends], axis=0)
        abs_idx = np.stack([np.arange(Wg) - Wg + e for e in ends])
        valid_own = jnp.broadcast_to(jnp.asarray((abs_idx >= 0) & (abs_idx < n_blocks))[None],
                                      (B, Bn, Wg)).reshape(B2, Wg)
        parts, ropes, valids, extra_len = [], [], [], 0
        for k, extra in enumerate(extra_codes or []):
            if extra is None:
                continue
            embed_t, proj_t, up_stride, _ = self._extra_ctx_table(k)
            w, r, v, Wj = causal_extra_ctx_windows(extra, embed_t, proj_t, up_stride, G, n_groups, B, D,
                                                    self.cond_window, ends_all, gidx)
            v = jnp.broadcast_to(jnp.asarray(v)[None], (B, Bn, Wj)).reshape(B2, Wj)
            if drop_extras:
                v = jnp.zeros_like(v)
            elif self.cond_drop > 0 and rng is not None:
                keep = jax.random.bernoulli(jax.random.fold_in(rng, k), p=1.0 - self.cond_drop, shape=(B, 1))
                v = v & jnp.repeat(keep, Bn, axis=0)
            parts.append(w)
            ropes.append(r)
            valids.append(v)
            extra_len += Wj
        ctx_flat = jnp.concatenate(list(reversed(parts)) + [own], axis=1)
        rope = jnp.concatenate(list(reversed(ropes)) + [rope_own], axis=1)
        valid = jnp.concatenate(list(reversed(valids)) + [valid_own], axis=1)
        return ctx_flat, rope, valid, Wg, extra_len

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
                                  self.in_pq_chunks, h, rng, greedy, temperature, self.gen_top_k)

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
        idx, rng = sample_idx(logits, rng, greedy, temperature, self.gen_top_k)
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
        def _dec_stack(xe):
            for blk in blocks:
                xe = run_block(blk, xe, self.remat and not self.remat_level)
            return xe
        xe = jax.checkpoint(_dec_stack)(xe) if self.remat_level else _dec_stack(xe)
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
                                         draft_windowed_override: jnp.ndarray = None,
                                         draft_valid_windowed: jnp.ndarray = None,
                                         remat_override: bool = None,
                                         extra_ctx_code_soft: list = None) -> tuple:
        n_blocks_check = ctx_code_soft.shape[1]
        if self.interleave_decode and decode_past_override is None:
            logits, target_out, mask, mtp_loss = self.decode_logits_and_target_interleave(
                target_seq, ctx_code_soft, decoder_ncodes, extra_ctx_code_soft, rng)
            zero = jnp.array(0.0, dtype=logits.dtype)
            return logits, target_out, mask, mtp_loss, zero, zero
        if self.dense_decode or (decoder_ncodes >= n_blocks_check and decode_past_override is None
                                  and not extra_ctx_code_soft and self.decode_future == 0):
            logits, target_out, mask, mtp_loss = self.decode_logits_and_target(
                target_seq, ctx_code_soft, decoder_ncodes, rng=rng)
            zero = jnp.array(0.0, dtype=logits.dtype)
            return logits, target_out, mask, mtp_loss, zero, zero
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B = target_seq.shape[0]
        D = self.bos_embed.shape[-1]
        G = decoder_ncodes
        n_blocks = ctx_code_soft.shape[1]
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed_proj(ctx_code_soft, self.ctx_embed, self.ctx_proj)
        target_p = target_seq
        if pad_blocks > 0:
            target_p = jnp.pad(target_p, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        B2 = B * n_groups
        ctx_flat, rope_ctx_g, valid_ctx, Wg, extra_len_total = self._pardec_ctx_rows(
            ctx_tok, extra_ctx_code_soft, G, n_groups, n_blocks, rng)
        per_group_len = Wg + 1 + G * self.K

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
            # draft positions before the sequence's real start are zero-filled and masked as keys (key_valid)
            if draft_windowed_override is not None:
                draft_te = self._dec_embed_target(draft_windowed_override)
                valid = draft_valid_windowed
            elif draft_embed_override is not None:
                draft_embed_p = jnp.pad(draft_embed_override, ((0, 0), (Pp, 0), (0, 0)))
                draft_te = jnp.stack(
                    [draft_embed_p[:, g * Kspan:g * Kspan + Pp, :] for g in range(n_groups)], axis=1
                ).reshape(B2, Pp, D)
                valid = _draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * self.K)
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
                valid = _draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * self.K)
            draft_valid_b2 = jnp.ones((B2, Pp), dtype=bool) if valid is None else jnp.broadcast_to(
                jnp.asarray(valid)[None], (B, n_groups, Pp)).reshape(B2, Pp)
            # an invalid slot is masked as a key, but as a query its hidden state still predicts the next token,
            # so its input embedding is zeroed (a constant), never a real vocab embedding
            draft_te = jnp.where(draft_valid_b2[:, :, None], draft_te, 0.0)
            te_flat = jnp.concatenate([draft_te, real_tail_te], axis=1)
        else:
            draft_valid_b2 = jnp.ones((B2, 0), dtype=bool)
            te_flat = real_tail_te
        key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1), dtype=bool), draft_valid_b2,
                                      jnp.ones((B2, Kspan + Pf), dtype=bool)], axis=1)
        bos = jnp.broadcast_to(self.bos_embed, (B2, 1, D))
        xe = jnp.concatenate([ctx_flat, bos, te_flat], axis=1)
        per_group_len = per_group_len + Pp + Pf + extra_len_total
        remat = self.remat if remat_override is None else remat_override

        rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])[:, None]
        rope_draft = jnp.stack([(g + 1) * G - Pp + jnp.arange(Pp) for g in range(n_groups)], axis=0)
        rope_real_tail = jnp.stack(
            [(g + 1) * G + 1 + jnp.arange(widened_len - Pp) for g in range(n_groups)], axis=0)
        rope_target = jnp.clip(jnp.concatenate([rope_draft, rope_real_tail], axis=1), 0, None)
        rope_pos_ids_g = jnp.concatenate([rope_ctx_g, rope_bos, rope_target], axis=1)
        rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (B, n_groups, per_group_len)).reshape(B2, per_group_len)

        def _pardec_stack(x):
            for blk in blocks:
                x = run_block_pardec(blk, x, rope_pos_ids, key_valid, remat and not self.remat_level)
            return x
        x = jax.checkpoint(_pardec_stack)(xe) if (self.remat_level and remat_override is not False) else _pardec_stack(xe)
        h = ln_f(x)
        pred_pos = Wg + extra_len_total + Pp + jnp.arange(G * self.K)
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
        aux_loss, aux_acc = self._widened_aux_ntp(h, target_p, Wg, extra_len_total, Pp, Pf, Kspan,
                                                    n_groups, B, n_blocks, rng)
        return logits, target_out, mask, mtp_loss, aux_loss, aux_acc

    def _widened_aux_ntp(self, h, target_p, Wg, extra_len_total, Pp, Pf, Kspan, n_groups, B, n_blocks,
                          rng) -> tuple:
        # decode_past/decode_future are real ground-truth tokens EMBEDDED AS INPUT (causal, shifted --
        # no leakage) purely to give the core Kspan prediction extra context. Historically that context
        # was silently unused/unscored (Pp/Pf never appeared in pred_pos) -- pure wasted lookahead
        # instead of a genuine forecast. This scores them too, as a separate NTP loss/acc: a real
        # next-token prediction at those positions using only causally-available (already-attended) info.
        if Pp == 0 and Pf == 0:
            zero = jnp.array(0.0, dtype=h.dtype)
            return zero, zero
        ctx_len = Wg + extra_len_total
        ext_target_p = jnp.pad(target_p, ((0, 0), (Pp, Pf)) + ((0, 0),) * (target_p.ndim - 2))
        D = h.shape[-1]
        h_parts, tgt_parts, valid_parts = [], [], []
        if Pp > 0:
            pred_pos_pp = ctx_len + jnp.arange(Pp)
            h_parts.append(h[:, pred_pos_pp, :].reshape(B, n_groups, Pp, D))
            tgt_parts.append(jnp.stack(
                [ext_target_p[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1))
            abs_idx = np.array([[g * Kspan - Pp + t for t in range(Pp)] for g in range(n_groups)])
            valid_parts.append((abs_idx >= 0) & (abs_idx < n_blocks * self.K))
        if Pf > 0:
            pred_pos_pf = ctx_len + Pp + Kspan + jnp.arange(Pf)
            h_parts.append(h[:, pred_pos_pf, :].reshape(B, n_groups, Pf, D))
            tgt_parts.append(jnp.stack(
                [ext_target_p[:, (g + 1) * Kspan + Pp:(g + 1) * Kspan + Pp + Pf] for g in range(n_groups)],
                axis=1))
            abs_idx = np.array([[(g + 1) * Kspan + t for t in range(Pf)] for g in range(n_groups)])
            valid_parts.append((abs_idx >= 0) & (abs_idx < n_blocks * self.K))
        h_extra = jnp.concatenate(h_parts, axis=2)
        target_extra = jnp.concatenate(tgt_parts, axis=2)
        valid_np = np.concatenate(valid_parts, axis=1)
        valid = jnp.broadcast_to(jnp.asarray(valid_np)[None, :, :, None], (B, n_groups, Pp + Pf, 1))
        if self.token_head_type == "linears":
            logits_extra, mask_extra = self._token_logits_linears(h_extra), None
        elif self.token_head_type == "ar":
            logits_extra, mask_extra = self._token_teacher_forced_ar(h_extra, target_extra), None
        else:
            aux_rng = jax.random.fold_in(rng, 0x5eed) if rng is not None else None
            assert aux_rng is not None, "diffusion token head needs an rng even at eval (masking is inherent)"
            logits_extra, mask_extra = self._token_teacher_forced_diffusion(h_extra, target_extra, aux_rng)
        full_mask = valid if mask_extra is None else (valid & mask_extra)
        full_mask = jnp.broadcast_to(full_mask, target_extra.shape)
        return self._dec_loss_acc(logits_extra, target_extra, full_mask)

    def decode(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray, decoder_ncodes: int, rng=None) -> tuple:
        logits, target, mask, mtp_loss = self.decode_logits_and_target(target_seq, ctx_code_soft, decoder_ncodes, rng=rng)
        loss, acc = self._dec_loss_acc(logits, target, mask)
        return loss + self.mtp_weight * mtp_loss, acc

    def _interleave_reveal_schedule(self, n_groups: int, G: int, up_stride: int, extra_n_blocks: int) -> list:
        # static (Python-level, no tracing): for each own-group g, how many NEW extra-level codes become
        # causally revealed by the end of that group (same "complete" rule as extra_ctx_visible_counts, but
        # per-group delta, exact -- no chunk rounding, ever). Returns a list of length n_groups.
        revealed = [min(((g + 1) * G) // up_stride, extra_n_blocks) for g in range(n_groups)]
        return [revealed[0]] + [revealed[g] - revealed[g - 1] for g in range(1, n_groups)]

    def decode_logits_and_target_interleave(self, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray,
                                             decoder_ncodes: int, extra_ctx_code_soft=None, rng=None) -> tuple:
        # Hardcoded interleave, cond_depth<=2 only: one flat causal sequence per image, own codes and (at most
        # one) coarser level's codes appearing in strict causal order -- [.. extra codes revealed so far ..,
        # own_code_g, BOS, K target bytes, own_code_{g+1}, BOS, K target bytes, .. more extra codes when THEY
        # become revealed ..]. No windowing, no padding, no chunk-rounding: positions are natural sequence
        # order (0,1,2,...), so rope just works via the existing dense splash call, same as decode_logits_and_target.
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B = target_seq.shape[0]
        D = self.bos_embed.shape[-1]
        G = decoder_ncodes
        n_blocks = ctx_code_soft.shape[1]
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        te = self._dec_embed_target(target_seq)
        ctx_tok = code_embed_proj(ctx_code_soft, self.ctx_embed, self.ctx_proj)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
            te = jnp.pad(te, ((0, 0), (0, pad_blocks * self.K), (0, 0)))
        bos = jnp.broadcast_to(self.bos_embed, (B, 1, D))

        has_extra = bool(extra_ctx_code_soft) and extra_ctx_code_soft[0] is not None
        extra_tok = extra_reveal = None
        if has_extra:
            assert len(extra_ctx_code_soft) == 1, "decode_logits_and_target_interleave only supports cond_depth<=2"
            embed_t, proj_t, up_stride, extra_n_blocks = self._extra_ctx_table(0)
            assert up_stride >= G, "decode_logits_and_target_interleave needs up_stride>=G (at most 1 new extra/own-group)"
            extra_tok = code_embed_proj(extra_ctx_code_soft[0], embed_t, proj_t)
            extra_reveal = self._interleave_reveal_schedule(n_groups, G, up_stride, extra_n_blocks)

        # ALWAYS reserve exactly one extra-code slot per own-group (real embedding when revealed this group,
        # a zero placeholder otherwise -- matching decode_generate_interleave's fixed per-group layout exactly,
        # so training and generation see own_code_g/BOS_g/target_g at the SAME absolute (rope) position; a
        # variable-width layout here would silently desync the two, since decode_logits_and_target_interleave's
        # plain dense splash call has no masking to hide an unused slot the way generation does).
        # Vectorized (reshape-based, like decode_logits_and_target) -- a Python per-group loop building
        # n_groups*3+ tiny concatenated slices compiles/runs far too slowly at real scale (measured: 32s/step,
        # dominated by XLA fusing hundreds of small ops -- see 2026-09-22 smoke12 audit).
        extra_slot = 1 if has_extra else 0
        per_group_len = extra_slot + G + 1 + G * self.K
        if has_extra:
            reveal = np.asarray(extra_reveal)
            idx_in_extra = np.clip(np.cumsum(reveal) - 1, 0, extra_n_blocks - 1)
            extra_g = extra_tok[:, idx_in_extra, :]  # (B, n_groups, D), static gather
            valid = jnp.asarray(reveal > 0)[None, :, None]
            extra_g = jnp.where(valid, extra_g, jnp.zeros_like(extra_g))[:, :, None, :]
        else:
            extra_g = jnp.zeros((B, n_groups, 0, D), dtype=ctx_tok.dtype)
        own_g = ctx_tok.reshape(B, n_groups, G, D)
        bos_g = jnp.broadcast_to(bos[:, None, :, :], (B, n_groups, 1, D))
        te_g = te.reshape(B, n_groups, G * self.K, D)
        row = jnp.concatenate([extra_g, own_g, bos_g, te_g], axis=2)
        xe = row.reshape(B, n_groups * per_group_len, D)
        base = np.arange(n_groups) * per_group_len + extra_slot + G  # BOS position of each group (predicts target[0])
        pred_pos = jnp.asarray((base[:, None] + np.arange(G * self.K)[None, :]).reshape(-1))

        def _dec_stack(xe):
            for blk in blocks:
                xe = run_block(blk, xe, self.remat and not self.remat_level)
            return xe
        xe = jax.checkpoint(_dec_stack)(xe) if self.remat_level else _dec_stack(xe)
        h = ln_f(xe)
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
                return sample_idx(logits, rng, greedy, temperature, self.gen_top_k)
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
                                draft_override_flat: jnp.ndarray = None, draft_valid_flat: jnp.ndarray = None,
                                extra_ctx_idx: list = None, gen_decode_future: bool = False) -> jnp.ndarray:
        # gen_decode_future=True (default False, not used anywhere yet): also autoregressively generate
        # each group's decode_future tail and return it as a second array, shape (B, n_groups, Pf,
        # *out_extra) -- raw per-group future speculation. Caller decides how to use it (e.g. feed to
        # this level's own encode() as a lookahead prefill); this function does not consume its own output.
        if gen_decode_future:
            assert self.decode_future > 0, "gen_decode_future=True needs decode_future>0 for this level"
        n_blocks_check = ctx_idx.shape[1]
        if self.interleave_decode and decode_past_override is None and not gen_decode_future:
            return self.decode_generate_interleave(ctx_idx, decoder_ncodes, greedy, temperature, seed, extra_ctx_idx)
        if self.dense_decode or (decoder_ncodes >= n_blocks_check and decode_past_override is None
                                  and not extra_ctx_idx and not gen_decode_future):
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
        Pp = self.decode_past if decode_past_override is None else decode_past_override
        Pf = self.decode_future if gen_decode_future else 0
        Kspan = G * self.K
        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)
        B2 = B * n_groups
        ctx_tok_flat, rope_ctx_g, valid_ctx, Wg, extra_len_total = self._pardec_ctx_rows(
            ctx_tok, extra_ctx_idx, G, n_groups, n_blocks, None, drop_extras=self.cond_drop >= 1.0)
        rope_ctx_flat = jnp.broadcast_to(rope_ctx_g[None], (B, n_groups, rope_ctx_g.shape[1])).reshape(B2, -1)
        if Pp > 0:
            draft_valid = draft_valid_flat if draft_valid_flat is not None else jnp.broadcast_to(
                jnp.asarray(_draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * self.K))[None],
                (B, n_groups, Pp)).reshape(B2, Pp)
        else:
            draft_valid = jnp.ones((B2, 0), dtype=bool)
        key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1), dtype=bool), draft_valid,
                                      jnp.ones((B2, Kspan + Pf), dtype=bool)], axis=1)
        per_group_len = Wg + extra_len_total + 1 + Pp + Kspan + Pf
        rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])
        rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B, n_groups)).reshape(B2)

        def self_step(x_new, ck, cv, pos, rope_pos_row):
            new_ck, new_cv = [], []
            x = x_new
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = pardec_block_step(blk, x, ck[i], cv[i], pos, rope_pos_row, key_valid, per_group_len)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        def self_chunk_step(x_chunk, ck, cv, pos_start, rope_pos_ids_chunk):
            new_ck, new_cv = [], []
            x = x_chunk
            for i, blk in enumerate(blocks):
                x, ck_i, cv_i = pardec_block_chunk_step(blk, x, ck[i], cv[i], pos_start,
                                                          rope_pos_ids_chunk, key_valid, per_group_len)
                new_ck.append(ck_i)
                new_cv.append(cv_i)
            return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

        def token_predict(h_pos, rng):
            if self.token_head_type == "linears":
                logits = self._token_logits_linears(h_pos)
                return sample_idx(logits, rng, greedy, temperature, self.gen_top_k)
            elif self.token_head_type == "ar":
                return self._token_generate_ar(h_pos, rng, greedy, temperature)
            else:
                return self._token_generate_diffusion(h_pos, rng, greedy, temperature)

        cache_k = jnp.zeros((len(blocks), B2, self.n_kv_heads, per_group_len, hd))
        cache_v = jnp.zeros_like(cache_k)
        rng = jax.random.PRNGKey(seed)

        total_steps = Pp + G * self.K + Pf
        Ktot = G * self.K

        def widened_pos(t):
            return jnp.where(t < Pp, jnp.clip(rope_bos_flat - Pp + t, 0, None), rope_bos_flat + 1 + (t - Pp))

        def embed_tok(val, t):
            te = self._dec_embed_target(val)
            if Pp == 0:
                return te
            tc = jnp.minimum(t, Pp - 1)
            return jnp.where((t < Pp) & draft_valid[:, tc][:, None], te, jnp.where(t < Pp, 0.0, te))

        @jax.jit
        def run_pardec(ctx_tok_flat, cache_k, cache_v, rng):
            bos_in = jnp.broadcast_to(self.bos_embed, (B2, 1, D))
            chunk = jnp.concatenate([ctx_tok_flat, bos_in], axis=1)
            chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
            h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
            val0, rng = token_predict(h_chunk[:, -1, :], rng)
            x0 = embed_tok(val0 if draft_override_flat is None else draft_override_flat[:, 0], jnp.array(0))

            def step(carry, t):
                x_input, ck, cv, rng_c = carry
                h, ck, cv = self_step(x_input, ck, cv, Wg + extra_len_total + t, widened_pos(t - 1))
                val, rng_c = token_predict(h, rng_c)
                if draft_override_flat is not None and Pp > 0:
                    src = jnp.where(t < Pp, draft_override_flat[:, jnp.minimum(t, Pp - 1)], val)
                else:
                    src = val
                return (embed_tok(src, t), ck, cv, rng_c), val

            carry, vals_rest = jax.lax.scan(step, (x0, cache_k, cache_v, rng), jnp.arange(1, total_steps))
            all_vals = jnp.concatenate([val0[None], vals_rest], axis=0)      # (total_steps, B2, ...)
            vals_out = jnp.moveaxis(all_vals[Pp:Pp + Ktot], 0, 1)
            future_out = jnp.moveaxis(all_vals[Pp + Ktot:], 0, 1) if Pf > 0 else None
            return vals_out, future_out

        vals_all, future_all = run_pardec(ctx_tok_flat, cache_k, cache_v, rng)
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        out = out[:, :n_blocks * self.K]
        if not gen_decode_future:
            return out
        future_out = future_all.reshape(B, n_groups, Pf, *out_extra).astype(jnp.int32)
        return out, future_out

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


    def decode_generate_pardec_sync(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                                     temperature: float = 1.0, seed: int = 0, extra_ctx_idx: list = None,
                                     wave_groups: int = None) -> jnp.ndarray:
        # Real cross-wave sync (the "sync" flag's actual mechanism, generalized): splits the n_groups into
        # sequential waves of `wave_groups` groups each (default: self.stream_chunks's own chunk size -- reuses
        # that single knob, wave_groups=n_groups is degenerate/identical to plain decode_generate_pardec,
        # wave_groups=1 is the fully-causal group-by-group extreme). Within a wave groups still decode in one
        # parallel batch with zero mutual info (same as today -- level_refine is still the fix for that, not
        # this function). Across waves, decode_past's draft is the REAL previous wave's output (read off a
        # running `emitted` buffer), not a private per-group redecode -- this is what makes train (teacher-forced,
        # always real) and generate agree for decode_past, which plain decode_generate_pardec does not.
        # Scope (2026-09-22): only cond_depth<=1 is exercised/tested; cond_depth>1 extra_ctx_idx is threaded
        # through generically (composes via the same `ends`/chunk-boundary math _pardec_ctx_rows already uses)
        # but not covered by a test yet. cycle_refine_passes is an orchestration layer above this function
        # (cyclic_refine_generate calls decode_generate_pardec/decode_generate_multipass directly) and is
        # unaffected either way.
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        Pp = self.decode_past
        Kspan = G * self.K
        chunk_groups = 1 if self.stream_chunks <= 0 else -(-n_groups // self.stream_chunks)
        Wgg = wave_groups if wave_groups is not None else chunk_groups
        n_waves = -(-n_groups // Wgg)
        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)

        def token_predict(h_pos, rng):
            if self.token_head_type == "linears":
                logits = self._token_logits_linears(h_pos)
                return sample_idx(logits, rng, greedy, temperature, self.gen_top_k)
            elif self.token_head_type == "ar":
                return self._token_generate_ar(h_pos, rng, greedy, temperature)
            else:
                return self._token_generate_diffusion(h_pos, rng, greedy, temperature)

        rng = jax.random.PRNGKey(seed)
        emitted = jnp.zeros((B, n_groups * Kspan) + out_extra, dtype=jnp.int32)
        wave_outs = []
        for w in range(n_waves):
            groups_real = list(range(w * Wgg, min((w + 1) * Wgg, n_groups)))
            n_real = len(groups_real)
            groups_w = groups_real + [groups_real[-1]] * (Wgg - n_real)  # pad to a uniform Wgg with a repeated group
            wave_start_tok = groups_real[0] * Kspan
            B2w = B * Wgg

            ctx_tok_flat, rope_ctx_g, valid_ctx, Wg, extra_len_total = self._pardec_ctx_rows(
                ctx_tok, extra_ctx_idx, G, n_groups, n_blocks, None, drop_extras=self.cond_drop >= 1.0,
                groups=groups_w)
            rope_ctx_flat = jnp.broadcast_to(rope_ctx_g[None], (B, Wgg, rope_ctx_g.shape[1])).reshape(B2w, -1)
            per_group_len = Wg + extra_len_total + 1 + Pp + Kspan
            rope_bos = jnp.array([(g + 1) * G for g in groups_w])
            rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B, Wgg)).reshape(B2w)

            if Pp > 0:
                abs_pos = np.array([[g * Kspan - Pp + t for t in range(Pp)] for g in groups_w])  # (Wgg, Pp)
                before_start = abs_pos < 0
                is_real_np = (~before_start) & (abs_pos < wave_start_tok)
                clipped = np.clip(abs_pos, 0, n_groups * Kspan - 1)
                real_vals = emitted[:, clipped].reshape((B2w, Pp) + out_extra)
                is_real = jnp.broadcast_to(jnp.asarray(is_real_np)[None], (B, Wgg, Pp)).reshape(B2w, Pp)
                # draft_valid == is_real (NOT ~before_start): a not-yet-real position (same, still-unfinished wave)
                # must be masked exactly like "before sequence start", never filled with this row's own private
                # guess -- that guess is computed in total isolation from whichever row actually owns that
                # position, so treating it as valid content would be hallucination, not history.
                draft_valid = is_real
            else:
                real_vals = jnp.zeros((B2w, 0) + out_extra, dtype=jnp.int32)
                is_real = jnp.zeros((B2w, 0), dtype=bool)
                draft_valid = jnp.ones((B2w, 0), dtype=bool)
            key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2w, 1), dtype=bool), draft_valid,
                                          jnp.ones((B2w, Kspan), dtype=bool)], axis=1)

            def self_step(x_new, ck, cv, pos, rope_pos_row):
                new_ck, new_cv = [], []
                x = x_new
                for i, blk in enumerate(blocks):
                    x, ck_i, cv_i = pardec_block_step(blk, x, ck[i], cv[i], pos, rope_pos_row, key_valid,
                                                      per_group_len)
                    new_ck.append(ck_i)
                    new_cv.append(cv_i)
                return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

            def self_chunk_step(x_chunk, ck, cv, pos_start, rope_pos_ids_chunk):
                new_ck, new_cv = [], []
                x = x_chunk
                for i, blk in enumerate(blocks):
                    x, ck_i, cv_i = pardec_block_chunk_step(blk, x, ck[i], cv[i], pos_start,
                                                              rope_pos_ids_chunk, key_valid, per_group_len)
                    new_ck.append(ck_i)
                    new_cv.append(cv_i)
                return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

            def widened_pos(t):
                return jnp.where(t < Pp, jnp.clip(rope_bos_flat - Pp + t, 0, None), rope_bos_flat + 1 + (t - Pp))

            def embed_tok(val, t):
                te = self._dec_embed_target(val)
                if Pp == 0:
                    return te
                tc = jnp.minimum(t, Pp - 1)
                return jnp.where((t < Pp) & draft_valid[:, tc][:, None], te, jnp.where(t < Pp, 0.0, te))

            def src_at(val, t):
                if Pp == 0:
                    return val
                tc = jnp.minimum(t, Pp - 1)
                return jnp.where((t < Pp) & is_real[:, tc][:, None], real_vals[:, tc], val)

            total_steps = Pp + Kspan

            @jax.jit
            def run_wave(ctx_tok_flat, rng):
                cache_k = jnp.zeros((len(blocks), B2w, self.n_kv_heads, per_group_len, hd))
                cache_v = jnp.zeros_like(cache_k)
                bos_in = jnp.broadcast_to(self.bos_embed, (B2w, 1, D))
                chunk = jnp.concatenate([ctx_tok_flat, bos_in], axis=1)
                chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
                h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
                val0, rng = token_predict(h_chunk[:, -1, :], rng)
                x0 = embed_tok(src_at(val0, jnp.array(0)), jnp.array(0))

                def step(carry, t):
                    x_input, ck, cv, rng_c = carry
                    h, ck, cv = self_step(x_input, ck, cv, Wg + extra_len_total + t, widened_pos(t - 1))
                    val, rng_c = token_predict(h, rng_c)
                    src = src_at(val, t)
                    return (embed_tok(src, t), ck, cv, rng_c), val

                carry, vals_rest = jax.lax.scan(step, (x0, cache_k, cache_v, rng), jnp.arange(1, total_steps))
                all_vals = jnp.concatenate([val0[None], vals_rest], axis=0)
                return jnp.moveaxis(all_vals[Pp:], 0, 1)  # (B2w, Kspan, *out_extra)

            wave_out = run_wave(ctx_tok_flat, rng)
            rng = jax.random.fold_in(rng, w)
            wave_out = wave_out.reshape(B, Wgg, Kspan, *out_extra)[:, :n_real].reshape(B, n_real * Kspan, *out_extra)
            wave_outs.append(wave_out)
            emitted = jax.lax.dynamic_update_slice_in_dim(emitted, wave_out.astype(jnp.int32), wave_start_tok, axis=1)

        out = jnp.concatenate(wave_outs, axis=1).astype(jnp.int32)
        return out[:, :n_blocks * self.K]



    def decode_generate_interleave(self, ctx_idx: jnp.ndarray, decoder_ncodes: int, greedy: bool = True,
                                    temperature: float = 1.0, seed: int = 0,
                                    extra_ctx_idx: list = None) -> jnp.ndarray:
        # Hardcoded interleave, cond_depth<=2 only: generation counterpart of decode_logits_and_target_interleave.
        # ONE real, single, always-growing KV cache, plain blk.step/chunk_step (same simple incremental primitives
        # as decode_generate -- no masking, no pardec machinery). Own codes and (at most one) coarser level's
        # codes prefilled into it in strict causal order; every own-group reserves exactly one extra-code slot
        # (real embedding when revealed this group, a zero placeholder otherwise) -- a FIXED per-group width,
        # matching decode_logits_and_target_interleave's layout exactly so train/generate agree on every
        # absolute (rope) position. lax.scan over groups (compiled once; a Python-unrolled loop here doesn't
        # compile in reasonable time at real n_groups, confirmed 2026-09-22).
        blocks, ln_f = self._dec_blocks(), self._dec_ln_f()
        B, n_blocks, _ = ctx_idx.shape
        D = self.bos_embed.shape[-1]
        hd = D // self.n_heads
        out_extra = (self.in_pq_chunks,)
        G = decoder_ncodes
        pad_blocks = (-n_blocks) % G
        n_blocks_p = n_blocks + pad_blocks
        n_groups = n_blocks_p // G
        ctx_tok = code_embed_proj(ctx_idx, self.ctx_embed, self.ctx_proj)
        if pad_blocks > 0:
            ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        ctx_g = jnp.swapaxes(ctx_tok.reshape(B, n_groups, G, D), 0, 1)  # (n_groups, B, G, D)

        has_extra = extra_ctx_idx is not None and len(extra_ctx_idx) > 0 and extra_ctx_idx[0] is not None
        if has_extra:
            assert len(extra_ctx_idx) == 1, "decode_generate_interleave only supports cond_depth<=2"
            embed_t, proj_t, up_stride, extra_n_blocks = self._extra_ctx_table(0)
            assert up_stride >= G, "decode_generate_interleave needs up_stride>=G (at most 1 new extra/own-group)"
            extra_tok_real = code_embed_proj(extra_ctx_idx[0], embed_t, proj_t)  # (B, extra_n_blocks, D)
            reveal = np.asarray(self._interleave_reveal_schedule(n_groups, G, up_stride, extra_n_blocks))  # (n_groups,)
            idx_in_extra = np.clip(np.cumsum(reveal) - 1, 0, extra_n_blocks - 1)  # which extra code slot g holds
            real_slot_g = jnp.swapaxes(extra_tok_real[:, idx_in_extra, :], 0, 1)[:, :, None, :]  # (n_groups,B,1,D)
            valid_g = jnp.asarray(reveal > 0)[:, None, None, None]  # (n_groups,1,1,1)
            extra_slot_g = jnp.where(valid_g, real_slot_g, jnp.zeros_like(real_slot_g))  # zero when not revealed
            per_group_len = 1 + G + 1 + G * self.K  # extra slot + own codes + BOS + target
        else:
            extra_slot_g = jnp.zeros((n_groups, B, 0, D))
            per_group_len = G + 1 + G * self.K
        L_total = n_groups * per_group_len

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
                return sample_idx(logits, rng, greedy, temperature, self.gen_top_k)
            elif self.token_head_type == "ar":
                return self._token_generate_ar(h_pos, rng, greedy, temperature)
            else:
                return self._token_generate_diffusion(h_pos, rng, greedy, temperature)

        def group_step(carry, xs):
            cache_k, cache_v, pos, rng = carry
            own_g, extra_g = xs
            bos_in = jnp.broadcast_to(self.bos_embed, (B, 1, D))
            if has_extra:
                prefill = jnp.concatenate([extra_g, own_g, bos_in], axis=1)
            else:
                prefill = jnp.concatenate([own_g, bos_in], axis=1)
            h_chunk, cache_k, cache_v = self_chunk_step(prefill, cache_k, cache_v, pos)
            pos = pos + prefill.shape[1]
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
        xs = (ctx_g, extra_slot_g)

        @jax.jit
        def run_all(cache_k, cache_v, rng):
            carry, vals_all = jax.lax.scan(group_step, (cache_k, cache_v, jnp.array(0), rng), xs)
            return jnp.moveaxis(vals_all, 0, 1)

        vals_all = run_all(cache_k0, cache_v0, jax.random.PRNGKey(seed))
        out = vals_all.reshape(B, n_groups * G * self.K, *out_extra).astype(jnp.int32)
        return out[:, :n_blocks * self.K]


def decode_logits_and_target_cyclic_revision(levelN, dec_target_iN, ctx_code_soft, decoder_ncodes,
                                              levelT_code_soft, cycle_refine_passes, rng,
                                              encode_temperature, extra_ctx_code_soft=None) -> tuple:
    # levelN conditions on a growing stack of revisions of the (fixed) coarser level's own code:
    # pass 1 sees just v1 = levelT_code_soft (levelT's real, already-encoded code -- cond_depth-style
    # extra ctx, unfilled slots trainable-pad); after each pass, levelN's own decode gets re-encoded
    # (via levelN's own encoder) into the next revision, filling one more slot each pass. levelN never
    # sees its own target as input -- only ever a revised version of a DIFFERENT (coarser) level's code.
    detach = levelN.cyclic_revise_detach
    revision_stack = [levelT_code_soft] + [None] * (cycle_refine_passes - 1)
    static_extra = list(extra_ctx_code_soft) if extra_ctx_code_soft else []
    logits = target_out = mask = mtp_loss = aux_loss = aux_acc = None
    rngs = [None] * cycle_refine_passes if rng is None else list(jax.random.split(rng, cycle_refine_passes))
    for p in range(cycle_refine_passes):
        logits, target_out, mask, mtp_loss, aux_loss, aux_acc = levelN.decode_logits_and_target_pardec(
            dec_target_iN, ctx_code_soft, decoder_ncodes, rng=rngs[p],
            extra_ctx_code_soft=static_extra + revision_stack,
            remat_override=(levelN.cyclic_revise_remat if not detach else None))
        if p < cycle_refine_passes - 1:
            code_soft, code_idx = quantize_hard(logits)
            re_input = code_idx if detach else code_soft
            x_re = code_embed_proj(re_input, levelN.own_input_embed, levelN.own_input_proj)
            enc_re = levelN.encode(x_re, code_idx, rng=None, encode_temperature=encode_temperature)
            revision_stack[p + 1] = jax.lax.stop_gradient(enc_re["code_idx"]) if detach else enc_re["code_soft"]
    return logits, target_out, mask, mtp_loss, aux_loss, aux_acc


def decode_logits_and_target_multipass(level: EncDecLevel, target_seq: jnp.ndarray, ctx_code_soft: jnp.ndarray,
                                        decoder_ncodes: int, rng=None, multipass_detach: bool = True,
                                        level_refine_gumbel: bool = False, level_refine_temperature: float = 1.0,
                                        level_refine_gt_drop: float = 1.0, refine_quantize_drop: float = 0.0, refine_rng=None,
                                        refine_remat: bool = None, extra_ctx_code_soft: list = None,
                                        refine_active=None) -> tuple:
    logits, target_out, mask, mtp_loss, aux_loss, aux_acc = level.decode_logits_and_target_pardec(
        target_seq, ctx_code_soft, decoder_ncodes, rng=rng, extra_ctx_code_soft=extra_ctx_code_soft)
    if level.level_refine_passes <= 1 or level.level_refine_window <= 0:
        return logits, target_out, mask, mtp_loss, aux_loss, aux_acc
    Kspan = decoder_ncodes * level.K
    Pp = level.level_refine_window * Kspan
    n_extra = level.level_refine_passes - 1
    refine_rngs = [None] * n_extra if refine_rng is None else list(jax.random.split(refine_rng, n_extra))
    state = (logits, target_out, mask, mtp_loss, aux_loss, aux_acc)

    def _refine_pass(state, r_rng):
        logits, target_out = state[0], state[1]
        if level_refine_gumbel and r_rng is not None:
            code_soft, idx = quantize_gumbel(logits, r_rng, level_refine_temperature, refine_quantize_drop)
        else:
            code_soft, idx = quantize_hard(logits, r_rng if refine_quantize_drop > 0 else None,
                                            refine_quantize_drop, level_refine_temperature)
        if level_refine_gt_drop < 1.0 and r_rng is not None:
            # per-token: prob level_refine_gt_drop keeps the sampled own output, else the real target
            own = jax.random.bernoulli(jax.random.fold_in(r_rng, 1), p=level_refine_gt_drop, shape=idx.shape)
            gt_idx = target_out.astype(idx.dtype)
            code_soft = jnp.where(own[..., None], code_soft, jax.nn.one_hot(gt_idx, code_soft.shape[-1], dtype=code_soft.dtype))
            idx = jnp.where(own, idx, gt_idx)
        if multipass_detach:
            kwargs = dict(draft_override=jax.lax.stop_gradient(idx))
        else:
            kwargs = dict(draft_embed_override=level._dec_embed_target(code_soft))
        out = level.decode_logits_and_target_pardec(
            target_seq, ctx_code_soft, decoder_ncodes, rng=rng,
            decode_past_override=Pp, remat_override=refine_remat,
            extra_ctx_code_soft=extra_ctx_code_soft, **kwargs)
        return jax.tree_util.tree_map(lambda n, o: n.astype(o.dtype), out, state)

    for p_idx in range(n_extra):
        if refine_active is None:
            state = _refine_pass(state, refine_rngs[p_idx])
        else:
            state = jax.lax.cond(refine_active[p_idx], lambda st, r=refine_rngs[p_idx]: _refine_pass(st, r),
                                 lambda st: st, state)
    logits, target_out, mask, mtp_loss, aux_loss, aux_acc = state
    return logits, target_out, mask, mtp_loss, aux_loss, aux_acc


def _decode_generate_pardec_call(level, ctx_idx, decoder_ncodes, greedy, temperature, seed,
                                  decode_past_override, draft_override_flat, extra_ctx_idx=None,
                                  draft_valid_flat=None):
    return level.decode_generate_pardec(ctx_idx, decoder_ncodes, greedy=greedy, temperature=temperature,
                                         seed=seed, decode_past_override=decode_past_override,
                                         draft_override_flat=draft_override_flat, extra_ctx_idx=extra_ctx_idx,
                                         draft_valid_flat=draft_valid_flat)


_decode_generate_pardec_jit = eqx.filter_jit(_decode_generate_pardec_call)


def decode_generate_multipass(level: EncDecLevel, ctx_idx: jnp.ndarray, decoder_ncodes: int,
                               greedy: bool = True, temperature: float = 1.0, seed: int = 0,
                               extra_ctx_idx: list = None) -> jnp.ndarray:
    if level.gen_sync:
        pred = level.decode_generate_pardec_sync(ctx_idx, decoder_ncodes, greedy, temperature, seed, extra_ctx_idx)
    else:
        pred = _decode_generate_pardec_jit(level, ctx_idx, decoder_ncodes, greedy, temperature, seed, None, None,
                                            extra_ctx_idx)
    if level.level_refine_passes <= 1 or level.level_refine_window <= 0:
        return pred
    B, n_blocks = ctx_idx.shape[0], ctx_idx.shape[1]
    G = decoder_ncodes
    pad_blocks = (-n_blocks) % G
    n_groups = (n_blocks + pad_blocks) // G
    Kspan = G * level.K
    Pp = level.level_refine_window * Kspan
    draft_valid_np = _draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * level.K)
    for _ in range(level.level_refine_passes - 1):
        pred_p = pred if pad_blocks == 0 else jnp.pad(
            pred, ((0, 0), (0, pad_blocks * level.K)) + ((0, 0),) * (pred.ndim - 2))
        draft_p = jnp.pad(pred_p, ((0, 0), (Pp, 0)) + ((0, 0),) * (pred_p.ndim - 2))
        draft_windows = jnp.stack([draft_p[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1)
        draft_override_flat = draft_windows.reshape(B * n_groups, Pp, *pred.shape[2:])
        draft_valid_flat = jnp.broadcast_to(jnp.asarray(draft_valid_np)[None], (B, n_groups, Pp)).reshape(
            B * n_groups, Pp)
        pred = _decode_generate_pardec_jit(level, ctx_idx, decoder_ncodes, greedy, temperature, seed,
                                            Pp, draft_override_flat, extra_ctx_idx, draft_valid_flat)
    return pred


def cyclic_refine_generate(levels, phase, ctx_idx_top, decoder_ncodes_list, greedy, temperature, seed,
                            encode_temperature, n_cycles):
    # levelT generates once, via the standard single-pass path (matches training's "levelT decodes
    # exactly once" design). levelN then re-decodes n_cycles times, conditioning on a growing stack
    # of revisions of levelT's code (v1 = pred_T; each subsequent slot = a re-encode of levelN's own
    # previous-cycle output) via the same extra_ctx mechanism cond_depth uses.
    iT, iN = phase - 1, phase - 2
    levelT, levelN = levels[iT], levels[iN]
    GT = decoder_ncodes_list[iT]

    pred_T = _decode_generate_pardec_jit(levelT, ctx_idx_top, GT, greedy, temperature, seed, None, None)

    revision_stack = [pred_T] + [None] * (n_cycles - 1)
    pred_N = None
    for cyc in range(n_cycles):
        pred_N = decode_generate_multipass(levelN, pred_T, decoder_ncodes_list[iN],
                                            greedy=greedy, temperature=temperature, seed=seed,
                                            extra_ctx_idx=list(revision_stack))
        if cyc < n_cycles - 1:
            x_re = code_embed_proj(pred_N, levelN.own_input_embed, levelN.own_input_proj)
            enc_re = levelN.encode(x_re, pred_N, rng=None, encode_temperature=encode_temperature)
            revision_stack[cyc + 1] = enc_re["code_idx"]
    return pred_N


def encoder_hidden(level: EncDecLevel, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for blk in level.blocks:
        h = run_block(blk, h, False)
    return level.ln_f(h)


def encoder_ntp_logits(level: EncDecLevel, h: jnp.ndarray) -> jnp.ndarray:
    return reshape_pq(h @ level.ntp_head, level.in_pq_chunks, level.in_code_vocab)


def _sample_tokens(logits: jnp.ndarray, rng, greedy: bool, temperature, top_k: int = 0) -> jnp.ndarray:
    # gumbel-max with safe_argmax (jnp.argmax feeding a gather is miscompiled on TPU, see safe_argmax)
    if greedy:
        return safe_argmax(logits)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        kth = jax.lax.top_k(lg, top_k)[0][..., -1:]
        lg = jnp.where(lg < kth, -jnp.inf, lg)
    return safe_argmax(lg + jax.random.gumbel(rng, lg.shape))


def _encoder_free_run(level: EncDecLevel, tokens: jnp.ndarray, P, rng, temperature, greedy: bool,
                      top_k: int) -> jnp.ndarray:
    # tokens (B,total_len,C) holds the prompt in [:P] (P may be traced); the rest is overwritten
    x = code_embed_proj(tokens, level.own_input_embed, level.own_input_proj)

    def body(t, carry):
        tokens, x = carry
        # exact training-time encoder forward over the whole fixed-length buffer; the encoder is causal, so
        # positions <= t-1 never see the not-yet-generated (junk) positions after them
        h = encoder_hidden(level, x)
        lg = encoder_ntp_logits(level, jax.lax.dynamic_index_in_dim(h, t - 1, axis=1, keepdims=False))
        tok = _sample_tokens(lg, jax.random.fold_in(rng, t), greedy, temperature, top_k)
        tokens = tokens.at[:, t].set(tok.astype(tokens.dtype))
        x = x.at[:, t].set(code_embed_proj(tok, level.own_input_embed, level.own_input_proj))
        return tokens, x

    tokens, _ = jax.lax.fori_loop(P, tokens.shape[1], body, (tokens, x))
    return tokens


_encoder_free_run_jit = eqx.filter_jit(_encoder_free_run)


def encoder_free_run(level: EncDecLevel, prompt_tokens: jnp.ndarray, total_len: int, rng, greedy: bool = False,
                     temperature: float = 1.0, top_k: int = 0) -> jnp.ndarray:
    """Free-run one level's encoder as a language model over its own input tokens (its NTP head): keep the
    prompt tokens, then sample the rest. greedy=True is argmax; otherwise temperature / top_k sampling."""
    assert level.attn_lookahead == 0, \
        f"encoder free-run needs attn_lookahead=0 (got {level.attn_lookahead}): a lookahead shifts the NTP target"
    B, P, C = prompt_tokens.shape
    assert 1 <= P <= total_len, f"prompt length {P} must be in [1, {total_len}]"
    tokens = jnp.zeros((B, total_len, C), prompt_tokens.dtype).at[:, :P].set(prompt_tokens)
    return _encoder_free_run_jit(level, tokens, jnp.asarray(P, jnp.int32), rng,
                                 jnp.asarray(temperature, jnp.float32), greedy, top_k)


def generate_from_prompt(model: HierEncDec, cfg: Config, prompt_bytes: jnp.ndarray, total_positions: int,
                          sample_level: int, rng, greedy: bool = False, temperature: float = 1.0,
                          top_k: int = 0, encode_temperature: float = 1.0, decode_greedy: bool = True,
                          decode_temperature: float = 1.0, decode_seed: int = 0) -> dict:
    """Prompted generation through an ENCODER's own next-token head (only the top two levels allowed).
    prompt_bytes (B,P,byte_group) = leading positions of an image. The prompt is encoded up to `sample_level`,
    that level's encoder is free-run (sampling its own input tokens: bytes at level 0, level-(L-1) codes above),
    the completed sequence is encoded back up to the top (the codes the encoder emits), and those codes are
    decoded down the usual cascade. Returns image bytes for decode_from "emitted" (cascade from the top emitted
    code) and, when available, "sampled" (the free-run tokens themselves: the image for level 0, decoded from the
    sampled level-(L-1) codes above it). greedy/temperature/top_k control the encoder free-run,
    decode_greedy/decode_temperature the decoder cascade."""
    levels = model.levels
    n = len(levels)
    assert n - 2 <= sample_level <= n - 1, f"sample_level {sample_level} must be one of the top two levels of {n}"
    K0 = levels[0].K
    B, P, _ = prompt_bytes.shape
    assert P % K0 == 0, f"prompt length {P} must be a multiple of the level-0 stride {K0}"
    tok = prompt_bytes
    for i in range(sample_level):
        x = code_embed_proj(tok, levels[i].own_input_embed, levels[i].own_input_proj)
        tok = levels[i].encode(x, tok, rng=None, encode_temperature=encode_temperature)["code_idx"]
        assert tok.shape[1] >= 1, "prompt too short to produce a single code at the sampling level"
    ds = 1
    for i in range(sample_level):
        ds *= levels[i].K
    tokens_L = encoder_free_run(levels[sample_level], tok, total_positions // ds, rng, greedy, temperature, top_k)

    codes, x, tgt = {}, code_embed_proj(tokens_L, levels[sample_level].own_input_embed,
                                        levels[sample_level].own_input_proj), tokens_L
    for i in range(sample_level, n):
        out = levels[i].encode(x, tgt, rng=None, encode_temperature=encode_temperature)
        codes[i] = out["code_idx"]
        if i < n - 1:
            x = code_embed_proj(out["code_soft"], levels[i + 1].own_input_embed, levels[i + 1].own_input_proj)
            tgt = out["code_idx"]

    def cascade(cur, from_level):
        for i in range(from_level, -1, -1):
            lv = levels[i]
            extra = [codes.get(j) for j in range(i + 1, i + lv.cond_depth)] if lv.cond_depth > 1 else None
            cur = decode_generate_multipass(lv, cur, cfg.decoder_ncodes[i], greedy=decode_greedy,
                                            temperature=decode_temperature, seed=decode_seed, extra_ctx_idx=extra)
        return cur

    res = dict(sampled_tokens=tokens_L, emitted_codes=codes, emitted=cascade(codes[n - 1], n - 1))
    if sample_level == 0:
        res["sampled"] = tokens_L
    else:
        res["sampled"] = cascade(tokens_L, sample_level - 1)
    return res


def mtp_predict_no_verify_standalone(level: EncDecLevel, h_pos, rng, greedy, temperature):
    K, chunks, vocab = level.mtp_horizon, level.in_pq_chunks, level.in_code_vocab
    if level.mtp_mode == "parallel":
        logits = (h_pos @ level.mtp_out_head).reshape(*h_pos.shape[:-1], K, chunks, vocab)
        return sample_idx(logits, rng, greedy, temperature, level.gen_top_k)
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
        return sample_idx(logits, rng, greedy, temperature, level.gen_top_k)
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


def level_forward(model: HierEncDec, flat_bytes: jnp.ndarray, phase: int, rng=None,
                   level_gt_drop=None, cascade_rng=None, encode_temperature: float = 1.0,
                   layer_drop_prob=None, label_reg_weight: float = 0.0, label_fn=None,
                   pixel_order=None, feedback_p=None, feedback_rng=None, feedback_detach: bool = True,
                   _feedback_recursed: bool = False, refine_active=None) -> tuple:
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
    aux_ntp_losses, aux_ntp_accs = [], []

    def _aux_applies(level) -> bool:
        return level.decode_future > 0 or level.decode_past > 0 or \
            (level.level_refine_passes > 1 and level.level_refine_window > 0)

    feedback_losses = []
    byte_mse = None
    mse_loss = 0.0
    ctx = codes_soft[phase - 1]
    cascade_rngs = [None] * phase if cascade_rng is None else list(jax.random.split(cascade_rng, phase))
    feedback_rngs = [None] * phase if feedback_rng is None else list(jax.random.split(feedback_rng, phase))

    start_i = phase - 1
    cyclic_iN = phase - 2 if (model.cfg.cycle_refine_passes > 1 and phase >= 2) else None

    for i in range(start_i, -1, -1):
        dec_target = flat_bytes if i == 0 else codes[i - 1]
        dec_rng = level_rngs[2 * i + 1]
        extra_ctx_i = [codes_soft[j] if j < phase else None for j in range(i + 1, i + levels[i].cond_depth)] \
            if levels[i].cond_depth > 1 else None
        if i == cyclic_iN:
            logits, target_i, mask_i, mtp_loss_i, aux_loss_i, aux_acc_i = decode_logits_and_target_cyclic_revision(
                levels[i], dec_target, ctx, model.cfg.decoder_ncodes[i], codes_soft[i + 1],
                model.cfg.cycle_refine_passes, dec_rng, encode_temperature, extra_ctx_code_soft=extra_ctx_i)
        else:
            logits, target_i, mask_i, mtp_loss_i, aux_loss_i, aux_acc_i = decode_logits_and_target_multipass(
                levels[i], dec_target, ctx, model.cfg.decoder_ncodes[i], rng=dec_rng,
                multipass_detach=model.cfg.multipass_detach, level_refine_gumbel=model.cfg.level_refine_gumbel,
                level_refine_gt_drop=model.cfg.level_refine_gt_drop,
                level_refine_temperature=model.cfg.level_refine_temperature,
                refine_quantize_drop=model.cfg.refine_quantize_drop, refine_rng=dec_rng,
                refine_remat=model.cfg.refine_remat, extra_ctx_code_soft=extra_ctx_i, refine_active=refine_active)
        loss_i, acc_i = levels[i]._dec_loss_acc(logits, target_i, mask_i)
        loss_i = loss_i + levels[i].mtp_weight * mtp_loss_i
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if _aux_applies(levels[i]):
            aux_ntp_losses.append(aux_loss_i)
            aux_ntp_accs.append(aux_acc_i)
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
                            lj, tj, mj, mtpj, _, _ = levels[j].decode_logits_and_target_pardec(
                                dj, c, model.cfg.decoder_ncodes[j])
                            lossj, _ = levels[j]._dec_loss_acc(lj, tj, mj)
                            total = total + lossj + levels[j].mtp_weight * mtpj
                            if j > 0:
                                c = codes_soft[j - 1]
                        return total / i
                else:
                    def _feedback_fire(logits=logits):
                        pseudo_bytes = jax.lax.stop_gradient(safe_argmax(logits))
                        l2, _ = level_forward(
                            model, pseudo_bytes, phase, rng=rng, level_gt_drop=level_gt_drop,
                            cascade_rng=cascade_rng, encode_temperature=encode_temperature,
                            layer_drop_prob=layer_drop_prob, label_reg_weight=0.0, label_fn=None,
                            pixel_order=None, feedback_p=None, feedback_rng=None,
                            _feedback_recursed=True, refine_active=refine_active)
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
    if aux_ntp_losses:
        aux_ntp_loss_total = jnp.mean(jnp.stack(aux_ntp_losses))
        aux_ntp_acc_total = jnp.mean(jnp.stack(aux_ntp_accs))
    else:
        aux_ntp_loss_total = jnp.array(0.0)
        aux_ntp_acc_total = jnp.array(0.0)
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + model.cfg.mse_weight * mse_loss + label_reg_weight * label_loss_total + feedback_loss_total \
        + model.cfg.ntp_weight * aux_ntp_loss_total
    bpb = dec_loss_total / jnp.log(2.0)
    return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)), byte_mse, aux_ntp_loss_total / jnp.log(2.0), aux_ntp_acc_total)


def _zero_drop_model(model: HierEncDec) -> HierEncDec:
    new_levels = []
    for lvl in model.levels:
        new_lvl = copy.copy(lvl)
        object.__setattr__(new_lvl, "quantize_drop", 0.0)
        object.__setattr__(new_lvl, "cond_drop", 0.0)
        new_levels.append(new_lvl)
    return eqx.tree_at(lambda m: m.levels, model, replace=new_levels)


def phase_forward_additive_drop(model: HierEncDec, flat_bytes: jnp.ndarray, phase: int, rng=None,
                                 level_gt_drop=None, cascade_rng=None, encode_temperature: float = 1.0,
                                 layer_drop_prob=None, label_reg_weight: float = 0.0, label_fn=None,
                                 pixel_order=None, feedback_p=None, feedback_rng=None,
                                 feedback_detach: bool = True, refine_active=None) -> tuple:
    loss1, aux1 = level_forward(model, flat_bytes, phase, rng=rng, level_gt_drop=level_gt_drop,
                                 cascade_rng=cascade_rng, encode_temperature=encode_temperature,
                                 layer_drop_prob=layer_drop_prob, label_reg_weight=label_reg_weight,
                                 label_fn=label_fn, pixel_order=pixel_order, feedback_p=feedback_p,
                                 feedback_rng=feedback_rng, feedback_detach=feedback_detach,
                                 refine_active=refine_active)
    clean_model = _zero_drop_model(model)
    loss2, aux2 = level_forward(clean_model, flat_bytes, phase, rng=rng, level_gt_drop=None, cascade_rng=None,
                                 encode_temperature=encode_temperature, layer_drop_prob=layer_drop_prob,
                                 label_reg_weight=label_reg_weight, label_fn=label_fn, pixel_order=pixel_order,
                                 feedback_p=None, feedback_rng=None, refine_active=refine_active)
    return loss1 + loss2, aux1


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


def local_array(x):
    # this process's addressable slice of a pmap output (multi-host arrays are not fully addressable)
    if getattr(x, "is_fully_addressable", True):
        return x
    return np.concatenate([np.asarray(s.data).reshape((-1,) + x.shape[1:]) for s in x.addressable_shards])


def unreplicate(pytree):
    return jax.tree_util.tree_map(lambda x: local_array(x)[0] if eqx.is_array(x) else x, pytree)


def to_host(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(jax.device_get(local_array(x))) if eqx.is_array(x) else x, pytree)


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
                  "ncodes_window", "stream_chunks", "decode_past", "decode_future", "sync",
                  "level_refine_passes", "level_refine_window", "multipass_detach", "level_refine_gumbel", "level_refine_gt_drop", "level_refine_drop", "gen_temperature", "gen_top_k", "gen_sync", "dense_decode", "interleave_decode",
                  "level_refine_temperature", "refine_quantize_drop", "refine_remat", "cycle_refine_passes",
                  "cyclic_revise_detach", "cyclic_revise_remat",
                  "cond_depth", "cond_drop", "cond_window",
                  "weight_sharing", "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_at_inference", "init_scheme", "use_xsa",
                  "use_qknorm", "remat", "remat_level", "attn_window", "attn_lookahead", "dec_attn_window", "use_sink",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "mtp_horizon", "mtp_mode", "mtp_weight", "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal", "label_reg_weight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet64", "imagenet256"],
                    help="cifar (default): downloads/caches under --data_root. imagenetN: reads "
                         "pre-built shards from --data_root (scripts/imagenet/download_imagenetN.py; "
                         "does not download itself). Config.img_size must match (32 cifar, N imagenetN).")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--batch_size", type=_tuple_arg, default=(16,),
                    help="training batch size -- a bare int applies uniformly to every phase; a "
                         "tuple gives one value per phase (length must equal n_phases)")
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--multihost", type=lambda x: x.lower() != "false", default=False,
                    help="jax.distributed.initialize() for a multi-host TPU slice: run the same command on every host; "
                         "batch_size stays per device, each host feeds its own slice of the global batch")
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
    p.add_argument("--stream_chunks", type=_tuple_arg, default=Config.stream_chunks,
                    help="per level: parent codes arrive in this many chunks and a group decodes once its whole chunk "
                         "is available (sees all parent codes up to the chunk end). 0 = per-group streaming (sees "
                         "up to its own group end), 1 = wait once (sees all), n = wait n times (e.g. 4 = quarter "
                         "image). ncodes_window bounds the history before the chunk (-1 = all)")
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
    p.add_argument("--level_refine_passes", type=_tuple_arg, default=Config.level_refine_passes,
                    help="1 (default) = current single-pass decode, unchanged. >1: after pass 1 "
                         "(unchanged, ctx-only), each further pass re-decodes every group with "
                         "extra causal peer context -- level_refine_window preceding groups' PREVIOUS "
                         "PASS decoded codes at this same level (own predictions during training, "
                         "stop-gradient'd; own generated codes during generation) -- not real "
                         "ground truth, so train and generate see the same (imperfect) signal. "
                         "More passes = closer to full AR across groups, but each pass is a full "
                         "extra decode call (cost scales ~linearly with level_refine_passes)")
    p.add_argument("--level_refine_window", type=_tuple_arg, default=Config.level_refine_window,
                    help="level_refine_passes>1 only: how many preceding groups' previous-pass decoded "
                         "codes are visible as extra causal context each refinement pass (like "
                         "ncodes_window, but sourced from this level's own decode output, not the "
                         "level above's ctx)")
    p.add_argument("--multipass_detach", type=lambda x: x.lower() != "false", default=Config.multipass_detach,
                    help="refine passes (level_refine_passes>1): True (default) fully stop_gradient's "
                         "the draft (own prediction, no gradient reaches the earlier pass that "
                         "produced it). False: STE instead, so the last pass's loss gradient flows "
                         "back through every earlier pass's decoder output -- backward cost grows "
                         "with level_refine_passes (BPTT-like), unlike the detached default")
    p.add_argument("--level_refine_gumbel", type=lambda x: x.lower() != "false", default=Config.level_refine_gumbel,
                    help="refine passes only: gumbel-perturb which code gets drafted each pass "
                         "(own dedicated knob, NOT the encoder's quantize_mode). Default False "
                         "(plain deterministic argmax draft)")
    p.add_argument("--level_refine_gt_drop", type=float, default=Config.level_refine_gt_drop,
                    help="refine passes, training only: per-token prob of drafting the sampled own output; "
                         "with prob 1-this the real target is drafted instead. Default 1.0 (always own output)")
    p.add_argument("--level_refine_drop", type=float, default=Config.level_refine_drop,
                    help="refine early exit, training only: before each extra refine pass, stop with this prob "
                         "(once stopped, later passes are skipped too), so a step runs 1..level_refine_passes passes "
                         "(geometric); the same draw applies to every level and device. Default 0.0 (all passes)")
    p.add_argument("--gen_temperature", type=float, default=Config.gen_temperature,
                    help="temperature of the sampled (non-argmax) generation eval / decoder sampling")
    p.add_argument("--gen_top_k", type=int, default=Config.gen_top_k,
                    help="top-k of decoder sampling (0 = off); only used when sampling, never for argmax")
    p.add_argument("--dense_decode", type=_bool_tuple_arg, default=Config.dense_decode,
                    help="regress this level to the original, fully-interleaved [code,BOS,K bytes,code,BOS,...] "
                         "flat causal decoder (decode_logits_and_target / decode_generate): no windowing, no "
                         "groups/batching approximation, no decode_past/level_refine/cond_depth/stream_chunks/"
                         "gen_sync (all ignored). Training uses splash (full causal, O(T) memory); generation "
                         "uses a single real growing KV cache (lax.scan over own-codes), no padding/magic numbers "
                         "-- every step genuinely sees the whole real prefix. O(T^2) total compute either way, "
                         "same as any correct full-attention causal LM; dec_attn_window bounds it if desired.")
    p.add_argument("--interleave_decode", type=_bool_tuple_arg, default=Config.interleave_decode,
                    help="dense_decode + cond_depth<=2 support (hardcoded): one flat causal sequence, at most one "
                         "coarser level's codes prefilled into it exactly when each becomes causally revealed. "
                         "Same real-single-growing-cache property as dense_decode -- no padding, no chunking, no "
                         "gen_sync/refine (ignored).")
    p.add_argument("--gen_sync", type=_bool_tuple_arg, default=Config.gen_sync,
                    help="generation only (no effect on training, which is already teacher-forced/real): use "
                         "decode_generate_pardec_sync instead of the plain private-redraft decode_past path. "
                         "wave_groups = this level's own stream_chunks chunk size (default stream_chunks=0 -> "
                         "wave_groups=1, fully causal group-by-group). Not yet composed with level_refine_passes>1 "
                         "(a warning is printed if both are set; the refine passes still run on top of the sync "
                         "pass-1 output, same private-redraft mechanism as before, for now)")
    p.add_argument("--level_refine_temperature", type=float, default=Config.level_refine_temperature,
                    help="own dedicated temperature, not shared with the encoder's "
                         "encode_temperature. Used by level_refine_gumbel=True's gumbel-softmax, and "
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
    p.add_argument("--cycle_refine_passes", type=int, default=Config.cycle_refine_passes,
                    help="1 (default): off. >1: the lower of the top two levels of the active phase "
                         "(levelN) re-decodes cycle_refine_passes times, each pass conditioning on a "
                         "growing stack of revisions of the coarser level's (levelT) own code -- "
                         "pass 1 sees just v1 (levelT's real code, cond_depth style extra ctx, "
                         "remaining slots trainable-pad); after each pass, levelN's own decoded "
                         "output is re-encoded (via levelT's own encoder) into the next revision "
                         "(v2, v3, ...), filling one more slot each pass. levelT itself decodes "
                         "exactly once, via the normal per-level loop (level_gt_drop etc. apply as "
                         "usual) -- this flag never gives levelN's own literal target as input to "
                         "itself. See --cyclic_revise_detach/--cyclic_revise_remat for how gradients "
                         "flow across passes. Loss on the final pass only")
    p.add_argument("--cyclic_revise_detach", type=lambda x: x.lower() != "false",
                    default=Config.cyclic_revise_detach,
                    help="cycle_refine_passes>1 only: True (default) stop_gradient's each pass's "
                         "re-encoded revision before feeding it to the next pass (no BPTT-like "
                         "gradient across passes). False lets gradients flow through the whole "
                         "revision chain -- pair with --cyclic_revise_remat to control memory")
    p.add_argument("--cyclic_revise_remat", type=lambda x: x.lower() != "false",
                    default=Config.cyclic_revise_remat,
                    help="cycle_refine_passes>1 and cyclic_revise_detach=False only: whether the "
                         "revision passes remat (checkpoint) their activations to bound memory while "
                         "gradients flow across the whole revision chain. True (default). No effect "
                         "under cyclic_revise_detach=True")
    p.add_argument("--cond_depth", type=_tuple_arg, default=Config.cond_depth,
                    help="1 (default): level i's decode ctx = its own code only. >1: also condition on the "
                         "next (cond_depth-1) coarser levels' own codes as extra ctx blocks (pervasive "
                         "conditioning without cross-attention). Group g sees coarser code j iff its whole span "
                         "ends at or before the group's visible end (its chunk end, see --stream_chunks); if the "
                         "chunk is not a multiple of the cumulative stride a warning is printed and the code "
                         "covering the group's own span stays hidden.")
    p.add_argument("--cond_window", type=_tuple_arg, default=Config.cond_window,
                    help="-1 (default): unbounded, each group sees all visible coarser-level codes (padded "
                         "with masked slots up to the full width). >=1: only the last cond_window "
                         "visible coarser codes. Needs cond_depth>1.")
    p.add_argument("--cond_drop", type=_float_tuple_arg, default=Config.cond_drop,
                    help="cond_depth>1 only. Per-extra-ctx-block independent bernoulli during "
                         "training, p=cond_drop probability of zeroing that block's embedding for a "
                         "given example; kept (non-dropped) blocks are scaled by 1/(1-cond_drop) "
                         "(standard inverted dropout, matches generation's always-full-ctx "
                         "magnitude). Generation never drops stochastically -- EXCEPT cond_drop=1.0 "
                         "exactly, where the block is permanently zeroed at generation too (its "
                         "extra_ctx_embed/proj weights never saw real content during training at "
                         "that setting, so feeding them real content at generation would be "
                         "undefined/out-of-distribution). 0 (default) = off")
    p.add_argument("--additive_drop_loss", type=lambda x: x.lower() != "false", default=False,
                    help="hacky 2nd-forward-pass variant of every rng-driven drop mechanism "
                         "(quantize_drop, level_gt_drop, feedback_p, cond_drop): runs level_forward "
                         "TWICE per step -- once normally (stochastic, as configured) and once on a "
                         "copy of the model with quantize_drop/cond_drop forced to 0 and "
                         "level_gt_drop/feedback_p forced off (always real/full ctx) -- and adds the "
                         "two losses. Not true marginalization (that would need per-site weighted "
                         "branches and blows up combinatorially with the number of drop sites); this "
                         "just guarantees a 'clean' gradient signal every step in addition to the "
                         "stochastic one, at 2x forward-pass cost. Default False")
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
    p.add_argument("--remat_level", type=lambda x: x.lower() != "false", default=Config.remat_level,
                    help="checkpoint each level's whole encoder / decoder block stack as one unit (recompute at the "
                         "level border) instead of per transformer block; less recompute, more live memory. "
                         "Takes precedence over --remat inside the stacks")
    p.add_argument("--attn_window", type=_tuple_arg, default=Config.attn_window)
    p.add_argument("--dec_attn_window", type=_tuple_arg, default=Config.dec_attn_window,
                    help="decoder-side causal sliding window (splash LocalMask in decode_logits_and_target's "
                         "dense/flat form; a plain narrowed mask in decode_generate's incremental KV-cache form -- "
                         "see Attention.step/chunk_step). -1 (default): unbounded, current behavior unchanged. "
                         ">=1: bound the decoder's own causal attention span. No effect under weight_sharing=True "
                         "(decoder reuses the encoder's blocks, and attn_window, instead)")
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
    if "streaming" in config_vars:
        p.error(f"--config {pre_args.config}: 'streaming' was replaced by 'stream_chunks' "
                f"(0 = per-group streaming, 1 = wait once / old streaming=False, n = n chunks)")
    label_fn = config_vars.pop("label_fn", default_label_fn_jax)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    # helper constants (e.g. DEPTH = 4) are allowed: warn and ignore; imports/functions are ignored silently
    consts = sorted(k for k in unknown if not callable(config_vars[k]) and not isinstance(config_vars[k], type(argparse)))
    if consts:
        warnings.warn(f"--config {pre_args.config}: ignoring non-field constant(s) {consts}")
    config_vars = {k: v for k, v in config_vars.items() if k in known}
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

    if args.multihost:
        jax.distributed.initialize()
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

    (train_np, train_labels), (val_np, val_labels) = load_dataset(args.dataset, Path(args.data_root), cfg.img_size)
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
    if cfg.precision != "bf16":
        jax.config.update("jax_default_matmul_precision", "highest")
    recon_prompt = flat_prompt = gt_img = None
    train_recon_prompt = train_flat_prompt = train_gt_img = None
    gen_jit_timed = [False]

    def run_gen_eval(eval_model, top: int, tag: str, flat_prompt, gt_img, sample: bool = False) -> tuple:
        gen_t0 = time.monotonic()
        tag = tag + ("_sample" if sample else "")
        g_kw = dict(greedy=not sample, temperature=cfg.gen_temperature, seed=1 if sample else 0)
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
        cyclic_fired_gen = cfg.cycle_refine_passes > 1 and top >= 1
        if cyclic_fired_gen:
            iN_gen = top - 1
            cur_code = cyclic_refine_generate(m.levels, top + 1, codes[top], cfg.decoder_ncodes,
                                               not sample, cfg.gen_temperature, g_kw["seed"], args.encode_temperature[phase - 1],
                                               cfg.cycle_refine_passes)
            loop_start = iN_gen - 1
        else:
            cur_code = codes[top]
            loop_start = top
        for i in range(loop_start, 0, -1):
            extra_ctx_i = [codes[j] if j <= top else None for j in range(i + 1, i + m.levels[i].cond_depth)] \
                if m.levels[i].cond_depth > 1 else None
            cur_code = decode_generate_multipass(m.levels[i], cur_code, cfg.decoder_ncodes[i], **g_kw,
                                                  extra_ctx_idx=extra_ctx_i)
        if cyclic_fired_gen and iN_gen == 0:
            cascade_recon = cur_code
        else:
            extra_ctx_0 = [codes[j] if j <= top else None for j in range(1, m.levels[0].cond_depth)] \
                if m.levels[0].cond_depth > 1 else None
            cascade_recon = decode_generate_multipass(m.levels[0], cur_code, cfg.decoder_ncodes[0], **g_kw,
                                                        extra_ctx_idx=extra_ctx_0)
        gen_compile_s = None
        if not gen_jit_timed[0]:
            gen_compile_s = time.monotonic() - cascade_t0
            gen_jit_timed[0] = True
        cascade_acc = float(jnp.mean(cascade_recon == flat_prompt))
        cascade_img = positions_to_image(np.asarray(cascade_recon), cfg, pixel_order)
        cascade_mse = pixel_mse(cascade_img, gt_img)
        save_compare_grid(cascade_img, gt_img, run_dir / f"samples_{tag}.png")

        gen_time_s = time.monotonic() - gen_t0
        msg = f"[{tag}] top={top} CASCADE{' (sampled T=%g k=%d)' % (cfg.gen_temperature, cfg.gen_top_k) if sample else ''} gen_byte_acc={cascade_acc:.4f} gen_cascade_mse={cascade_mse:.2f}"
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
        run_gen_eval(eval_model, top, f"{tag}_val", flat_prompt, gt_img, sample=True)
        if args.eval_gen_train:
            run_gen_eval(eval_model, top, f"{tag}_train", train_flat_prompt, train_gt_img)
            run_gen_eval(eval_model, top, f"{tag}_train", train_flat_prompt, train_gt_img, sample=True)
        return result

    val_eval_jit = eqx.filter_jit(level_forward)
    val_jit_timed = [False]

    def run_val_eval(eval_model, phase: int, tag: str) -> tuple:
        val_t0 = time.monotonic()
        m = cast_pytree(eval_model, compute_dtype)
        bs = args.val_batch_size[phase - 1]
        n = len(val_np)
        sums = np.zeros(8, dtype=np.float64)
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
        _bpb, acc, _ntp_bpb, ntp_acc, util, val_mse, _aux_ntp_bpb, aux_ntp_acc = (sums / total_n).tolist()
        loss = total_loss / total_n
        val_time_s = time.monotonic() - val_t0
        msg = (f"[{tag}] VAL loss={loss:.2f} val_dec_acc={acc:.2f} val_mse={val_mse:.4f} "
               f"val_e_ntp_acc={ntp_acc:.2f} val_d_ntp_acc={aux_ntp_acc:.2f} val_time={val_time_s:.1f}s")
        rec = dict(tag=tag, val_loss=loss, val_dec_acc=acc,
                    val_e_ntp_acc=ntp_acc, val_util=util, val_mse=val_mse,
                    val_d_ntp_acc=aux_ntp_acc,
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
        _phase_total_steps(p - 1, len(train_np) // (args.batch_size[p - 1] * n_devices * jax.process_count()))
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

        n_extra_max = max(cfg.level_refine_passes[i] - 1 for i in range(n_levels))
        use_refine_drop = cfg.level_refine_drop > 0 and n_extra_max > 0

        def loss_fn(diff_model, static_model, flat_bytes, rng, cascade_rng, feedback_rng, refine_active, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            phase_forward_fn = phase_forward_additive_drop if args.additive_drop_loss else level_forward
            return phase_forward_fn(m, flat_bytes, phase, rng=rng,
                                     level_gt_drop=level_gt_drop_phase, cascade_rng=cascade_rng,
                                     encode_temperature=encode_temperature_phase,
                                     layer_drop_prob=layer_drop_prob_phase,
                                     label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                     pixel_order=pixel_order,
                                     feedback_p=feedback_p_phase, feedback_rng=feedback_rng,
                                     feedback_detach=args.feedback_detach,
                                     refine_active=refine_active if use_refine_drop else None)

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

        def train_step(diff_model, opt_state, rng, flat_bytes, refine_active, static_model=static_model):
            rng, level_rng, cascade_rng, feedback_rng = jax.random.split(rng, 4)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                diff_model, static_model, flat_bytes, level_rng, cascade_rng, feedback_rng, refine_active)
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
        p_rng_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), phase)
        if jax.process_count() > 1:
            p_rng_key = jax.random.fold_in(p_rng_key, jax.process_index())
        p_rng = jax.random.split(p_rng_key, n_devices)

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

        refine_drop_rng = np.random.default_rng(args.seed + 7919)
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
                # refine early exit: geometric stop before each extra pass; same draw on every device/host
                u = refine_drop_rng.random(max(n_extra_max, 1))
                refine_active = np.cumprod(u >= cfg.level_refine_drop) > 0
                p_refine_active = jnp.broadcast_to(jnp.asarray(refine_active), (n_devices, refine_active.shape[0]))
                p_diff_model, p_opt_state, p_rng, loss, aux = train_step(p_diff_model, p_opt_state, p_rng, flat,
                                                                          p_refine_active)
                step += 1
                phase_step += 1
                pbar.update(1)
                loss0 = float(local_array(loss)[0])
                if not jit_timed:
                    logger(f"{active_desc}: first train_step (incl. jit compile) took "
                           f"{time.monotonic() - jit_t0:.1f}s")
                    jit_timed = True
                _bpb, acc, _ntp_bpb, ntp_acc, util, train_mse, _aux_ntp_bpb, aux_ntp_acc, grad_norm = \
                    [float(local_array(a)[0]) for a in aux]
                lr = float(lr_schedule(step - 1))
                lr_str = _fmt_lr(lr)
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}",
                                  acc=f"{acc:.2f}",
                                  lr=lr_str, gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    logger(f"l={phase - 1} e={epoch_num} s={step} loss={loss0:.2f} dec_acc={acc:.2f} "
                           f"e_ntp_acc={ntp_acc:.2f} util={util:.2f} mse={train_mse:.1f} "
                           f"d_ntp_acc={aux_ntp_acc:.2f} "
                           f"lr={lr_str} grad_norm={grad_norm:.2f}",
                           level=phase - 1, epoch=epoch_num, step=step, loss=loss0,
                           dec_acc=acc, e_ntp_acc=ntp_acc, util=util,
                           mse=train_mse, d_ntp_acc=aux_ntp_acc,
                           lr=lr, grad_norm=grad_norm)

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
                        eqx.tree_serialise_leaves(wa_dir / "ema_latest.eqx", wa_ema)
                        if args.wa_verbose:
                            logger(f"wa (ema) snapshot saved at step {step}")
                    else:
                        wa_stack.append(cur_diff_model)
                        if len(wa_stack) == args.wa_stack_size:
                            avg = stack_average(list(wa_stack), weights=args.wa_wma_weights)
                            eqx.tree_serialise_leaves(wa_dir / "wma_latest.eqx", avg)
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
    for top in range(n_phases - 1, -1, -1):
        run_gen_eval_both(model, top=top, tag=f"final_top{top}")
    logger("training done")


if __name__ == "__main__":
    main()
