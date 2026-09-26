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
                                        rope_cos_sin_pos, rotate_half, sinkgd, splash_future_attention,
                                        warmup_const_schedule)

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
    # Each of CodeLM/Downsampler/Upsampler is fully independent: its own dedicated fields, own
    # defaults, no fallback chain to a shared "base" field and no override-of-a-generic-field
    # pattern. codelm_* feeds CodeLM only.
    codelm_d_model: tuple = (256, 256, 256, 256)
    codelm_n_layers: tuple = (2, 2, 2, 2)
    codelm_n_heads: tuple = (4, 4, 4, 4)
    codelm_n_kv_heads: tuple = (None, None, None, None)  # None per-level entry -> defaults to
    # max(1, codelm_n_heads//4) for that level (resolved below) -- within-module default, not a
    # cross-module fallback.
    downsampler_d_model: tuple = 512  # PardecLM instance used by encode_pardec_downsampler -- own
    downsampler_n_layers: tuple = 4   # dedicated capacity, not reused from codelm_d_model
    downsampler_n_heads: tuple = 8    # (per the "codelm=enc, downsampler/upsampler=two separate
    downsampler_n_kv_heads: tuple = 8  # decoders" framing)
    downsampler_window: tuple = 4  # context_window_groups, in GROUPS not raw positions (same unit
    # as ncodes_window elsewhere) -- must stay bounded (not -1) given many small groups at
    # output_group_size=1, confirmed via standalone test (unbounded OOM'd even at tiny sizes)
    downsampler_decode_past: tuple = 0   # downsampler's OWN decode_past/decode_future/remat --
    downsampler_decode_future: tuple = 0  # independent of the upsampler's (and old dec_blocks
    downsampler_remat: tuple = None  # path's) decode_past/decode_future/remat -- previously the
    # downsampler silently ignored decode_past/decode_future entirely (always 0/0) and shared one
    # global remat with the upsampler; None (default) for downsampler_remat falls back to cfg.remat
    use_pardec_downsampler: bool = False  # False (default) = old naive-pick/code_head path in
    # encode(); True = encode_pardec_downsampler (teacher-forced against label_fn, requires
    # label_reg_weight>0 -- that's now the ONLY loss training the downsampler, not an aux term)
    upsampler_d_model: tuple = 1024  # PardecLM instance used by decode_logits_and_target_pardec/
    upsampler_n_layers: tuple = 4    # decode_generate_pardec when use_pardec_upsampler=True -- own
    upsampler_n_heads: tuple = 8     # dedicated capacity (mirrors downsampler_*), NOT reused from
    upsampler_n_kv_heads: tuple = 8  # own dedicated capacity; context_hidden_dim = codelm_d_model
    # (CodeLM's own dim) -- same as downsampler's, no separate ctx dimension for the upsampler
    upsampler_window: tuple = 4  # context_window_groups, in GROUPS of decoder_ncodes codes (same
    # unit as ncodes_window) -- must stay bounded (not -1), same OOM lesson as downsampler_window
    # NOTE: batching granularity (context_group_size=output_group_size) stays tied to the shared
    # decoder_ncodes field, not a separate upsampler_ncodes -- each level's downsampler/upsampler
    # pair is fixed to that level's own stride K (output_expansion=K for upsampler, context_group_
    # size=K for downsampler), by design, not a freely-configurable expansion factor. Genuine
    # multi-rate training (e.g. both a stride=4 and a direct stride=16 path) would need SEVERAL
    # dedicated downsampler/upsampler pairs per level, one per supported stride, selected by
    # whatever samples the entry_level/depth (see level_forward_multires/sample_multires_entry,
    # not yet wired into main()) -- not a single flexible module. Deferred, noted for later.
    upsampler_decode_past: tuple = None  # None falls back to the legacy decode_past/decode_future
    upsampler_decode_future: tuple = None  # (still used by the old dec_blocks path too) -- set
    # explicitly to decouple the upsampler's own values, e.g. once downsampler_decode_future also
    # needs an independent, different value
    upsampler_remat: tuple = None  # None falls back to cfg.remat (previous shared behavior)
    use_pardec_upsampler: bool = False  # False (default) = old hand-rolled dec_blocks/ctx_embed
    # pardec decode path in decode_logits_and_target_pardec/decode_generate_pardec (dense_decode/
    # decode_past-at-generation/stream_chunks all supported there). True = PardecLM-based path
    # (pardec_score/pardec_generate, context_group_size=output_group_size=decoder_ncodes,
    # output_expansion=K) -- shares the exact same machinery as the downsampler, just expanding
    # instead of contracting; decode_past is scoring-only here (pardec_generate's own documented
    # limitation), no dense_decode/stream_chunks equivalent yet
    share_downsampler_upsampler_lm: bool = False  # False (default) = fully separate downsampler/
    # upsampler PardecLM instances (current behavior, unchanged). True = the two SHARE the same
    # blocks/ln_f (the core transformer LM) -- everything else (context_proj, bos_embed,
    # target_embed/proj, token_* AR head, output_head_linear) stays independent per module. Needs
    # downsampler_d_model==upsampler_d_model (and matching n_layers/n_heads/n_kv_heads), asserted.
    strides: tuple = (3, 16, 16, -1)
    # code_vocab/pq_chunks/pq_dim are SINGLE shared fields, not one copy per module -- CodeLM,
    # Downsampler and Upsampler all construct their own_input_embed/target_embed/code_head/etc
    # directly from these same three values (see LagCodecModel.__init__), so they are structurally
    # guaranteed to speak the same categorical vocab (e.g. RGB pixel level: code_vocab=256,
    # pq_chunks=3) -- there is no way for them to diverge, by construction, not by convention.
    code_vocab: tuple = (4, 4, 4, 4)
    pq_chunks: tuple = (5, 5, 5, 5)
    mlp_mult: tuple = 2
    rope_base: tuple = 10000.0
    ntp_weight: float = 1.0
    decoder_ncodes: tuple = 1
    ncodes_window: tuple = 0
    stream_chunks: tuple = 0
    stream_lag: tuple = 0  # alternate way to set stream_chunks: desired own-group wait cadence L (>=1), resolved
    # per-level to stream_chunks=ceil(n_groups/L) once n_groups is known (level-independent lag, unlike stream_chunks
    # itself which is n_groups-dependent). XOR with stream_chunks -- set only one (default 0 = unset for both).
    decode_past: tuple = 0
    decode_future: tuple = 0
    sync: tuple = False
    precision: str = "bf16"
    curriculum_mode: str = "freeze"
    quantize_mode: str = "argmax"
    quantize_drop: float = 0.0
    gumbel_at_inference: bool = False
    init_scheme: str = "llama"
    use_xsa: bool = False
    use_qknorm: bool = True

    attn_window: tuple = -1  # symmetric base (-1=unbounded), used by CodeLM's own self-attention
    # unless overridden below (None-means-unset-override pattern). When overridden, the value
    # itself uses attn_window's own convention (-1=unbounded, >=1=window). decoder_attn_window is
    # dead (only fed the removed old dec_blocks path), kept declared/inert.
    encoder_attn_window: tuple = None
    decoder_attn_window: tuple = None
    attn_lookahead: tuple = 0
    use_sink: bool = False

    use_codelm_bos: bool = False  # False (default) = no BOS/anchor token for CodeLM's own free-run --
    # position 0 is whatever a real image's leading content looks like (dataset-biased), same as
    # before this option existed. True = CodeLM gets its own learned bos_embed table (one row per
    # codelm_bos_rates, e.g. one anchor per target scale/zoom for genuinely unconditional free-run
    # generation, see encoder_free_run's use_bos/rate_id) -- substituted at position 0 with
    # probability codelm_bos_prob during training so it's learned, not untrained noise at inference.
    # Purely additive (a new field/table) -- opt-in per training run, including fine-tuning an
    # already-trained (use_codelm_bos=False) checkpoint by turning it on and continuing training.
    codelm_bos_prob: float = 0.1
    codelm_bos_rates: tuple = 1  # per-level n_rates for the bos_embed table; 1 (default) = single
    # generic anchor. No effect when use_codelm_bos=False.

    remat: bool = False
    remat_level: bool = False

    byte_group: int = 1
    token_head_type: tuple = "linears"
    token_dim: tuple = 64
    token_n_heads: tuple = 4
    pq_dim: tuple = 64  # own independent default -- NOT derived from codelm_d_model/anything else
    token_mask_prob: float = 0.15

    entropy_weight: float = 0.0

    traversal: str = "raster"

    mse_weight: float = 0.0
    mse_softmax_tau: float = 1.0

    label_reg_weight: float = 0.0
    label_mse_weight: float = 0.0  # 0 = metric only (always computed when label_reg_weight>0);
    # >0 additionally backprops a soft/differentiable version, mirrors mse_weight/mse_loss below

    gen_temperature: float = 1.0
    gen_top_k: int = 8
    dense_decode: tuple = False

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

        def bcast_opt(name, types):
            # like bcast, but a bare None (the "fully unset" default) broadcasts to (None,)*n instead of being
            # left alone -- a per-level tuple (possibly mixing None and real values) still passes through as-is.
            val = getattr(self, name)
            if val is None:
                setattr(self, name, (None,) * n)
            elif isinstance(val, types):
                setattr(self, name, (val,) * n)

        bcast("mlp_mult", int)
        bcast("rope_base", (int, float))
        bcast("decoder_ncodes", int)
        bcast("ncodes_window", int)
        bcast("stream_chunks", int)
        bcast("stream_lag", int)
        bcast("dense_decode", bool)
        bcast("decode_past", int)
        bcast("decode_future", int)
        bcast("sync", bool)
        bcast("codelm_d_model", int)
        bcast("codelm_n_layers", int)
        bcast("codelm_n_heads", int)
        bcast_opt("codelm_n_kv_heads", int)
        bcast("downsampler_d_model", int)
        bcast("downsampler_n_layers", int)
        bcast("downsampler_n_heads", int)
        bcast("downsampler_n_kv_heads", int)
        bcast("downsampler_window", int)
        bcast("downsampler_decode_past", int)
        bcast("downsampler_decode_future", int)
        bcast_opt("downsampler_remat", bool)
        bcast("upsampler_d_model", int)
        bcast("upsampler_n_layers", int)
        bcast("upsampler_n_heads", int)
        bcast("upsampler_n_kv_heads", int)
        bcast("upsampler_window", int)
        bcast_opt("upsampler_decode_past", int)
        bcast_opt("upsampler_decode_future", int)
        bcast_opt("upsampler_remat", bool)
        bcast("token_head_type", str)
        bcast("token_dim", int)
        bcast("token_n_heads", int)
        bcast("attn_window", int)
        bcast_opt("encoder_attn_window", int)
        bcast_opt("decoder_attn_window", int)
        bcast("attn_lookahead", int)
        bcast("codelm_bos_rates", int)
        bcast("pq_dim", int)

        assert len(self.codelm_d_model) == n and len(self.codelm_n_layers) == n and len(self.codelm_n_heads) == n \
            and len(self.codelm_n_kv_heads) == n and len(self.code_vocab) == n and len(self.pq_chunks) == n
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.decoder_ncodes) == n
        assert len(self.ncodes_window) == n and len(self.stream_chunks) == n and len(self.stream_lag) == n and len(self.dense_decode) == n
        assert len(self.decode_past) == n and len(self.decode_future) == n and len(self.sync) == n
        for i in range(n):
            assert self.ncodes_window[i] >= -1, \
                f"level {i}: ncodes_window={self.ncodes_window[i]} must be -1 (all) or >=0 " \
                f"(disjoint at 0, bounded lookback above)"
            assert self.stream_chunks[i] >= 0, \
                f"level {i}: stream_chunks={self.stream_chunks[i]} must be 0 (per-group streaming) or >=1 (chunks)"
            assert self.decode_past[i] >= 0 and self.decode_future[i] >= 0, \
                f"level {i}: decode_past={self.decode_past[i]}/decode_future={self.decode_future[i]} must be >=0"
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
        assert (len(self.attn_window) == n and len(self.attn_lookahead) == n
                and len(self.encoder_attn_window) == n and len(self.decoder_attn_window) == n
                and len(self.codelm_bos_rates) == n)
        for i in range(n):
            assert self.attn_window[i] == -1 or self.attn_window[i] >= 1, \
                f"level {i}: attn_window={self.attn_window[i]} must be -1 (unbounded/flash) or >=1 (splash LocalMask)"
            for name, val in (("encoder_attn_window", self.encoder_attn_window[i]),
                               ("decoder_attn_window", self.decoder_attn_window[i])):
                if val is not None:
                    assert val == -1 or val >= 1, f"level {i}: {name}={val} must be -1 (unbounded) or >=1"
            if self.dense_decode[i] and self.decode_future[i] != 0:
                warnings.warn(
                    f"level {i}: dense_decode=True IGNORES decode_future entirely "
                    f"(decode_logits_and_target/decode_generate don't take a future tail -- every "
                    f"step already sees the real whole prefix)")
            assert self.attn_lookahead[i] >= 0, \
                f"level {i}: attn_lookahead={self.attn_lookahead[i]} must be >=0 (0=plain causal)"

        top_level_trainable = self.strides[-1] != -1
        code_count = total_bytes_of(self) // self.byte_group
        stream_chunks_resolved = list(self.stream_chunks)
        for i in range(n):
            K_i = self.strides[i] if self.strides[i] != -1 else 1
            if self.traversal == "zorder" and K_i > 1:
                is_pow4 = (K_i & (K_i - 1)) == 0 and (K_i.bit_length() - 1) % 2 == 0
                assert is_pow4, \
                    f"level {i}: strides={K_i} must be a power of 4 (1, 4, 16, 64, ...) under " \
                    f"traversal='zorder' -- Z-order groups consecutive positions into a genuinely " \
                    f"square 2^k x 2^k spatial block only when the group size is 4^k; any other " \
                    f"stride (e.g. 8) silently groups a rectangular, non-square region instead. A " \
                    f"stride of 16 already gives a direct 4x linear-rate downsample (e.g. 32x32 -> " \
                    f"8x8) in a single level -- no need to chain two stride-4 levels for that."
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
            if self.stream_lag[i] != 0:
                assert S_i == 0, (f"level {i}: stream_chunks={S_i} and stream_lag={self.stream_lag[i]} are "
                                   f"mutually exclusive (XOR) -- set only one")
                assert self.stream_lag[i] >= 1, \
                    f"level {i}: stream_lag={self.stream_lag[i]} must be >=1 (own-groups to wait before the window advances)"
                S_i = stream_chunks_resolved[i] = -(-n_groups_i // self.stream_lag[i])  # ceil(n_groups_i / lag)
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
        self.stream_chunks = tuple(stream_chunks_resolved)  # any stream_lag[i]!=0 entries now hold the resolved value

        assert (len(self.downsampler_d_model) == n
                and len(self.downsampler_n_layers) == n and len(self.downsampler_n_heads) == n
                and len(self.downsampler_n_kv_heads) == n and len(self.downsampler_window) == n
                and len(self.downsampler_decode_past) == n and len(self.downsampler_decode_future) == n
                and len(self.downsampler_remat) == n
                and len(self.upsampler_d_model) == n and len(self.upsampler_n_layers) == n
                and len(self.upsampler_n_heads) == n and len(self.upsampler_n_kv_heads) == n
                and len(self.upsampler_window) == n
                and len(self.upsampler_decode_past) == n and len(self.upsampler_decode_future) == n
                and len(self.upsampler_remat) == n)
        # SINGLETON model: exactly one CodeLM, one Downsampler, one Upsampler for the whole model
        # (see LagCodecModel) -- every field that determines a weight SHAPE must therefore be uniform
        # across ALL n levels (index 0 is what LagCodecModel actually builds from). Runtime-only
        # grouping fields (decoder_ncodes, ncodes_window, decode_past/future, dense_decode,
        # stream_chunks/lag, ...) are exempt -- those vary per level as plain call-time arguments.
        singleton_uniform_fields = (
            "codelm_d_model", "codelm_n_layers", "codelm_n_heads", "codelm_n_kv_heads",
            "code_vocab", "pq_chunks", "pq_dim", "mlp_mult", "rope_base", "attn_lookahead",
            "attn_window", "encoder_attn_window", "decoder_attn_window", "use_sink", "use_xsa",
            "use_qknorm", "init_scheme", "quantize_mode", "quantize_drop", "remat", "remat_level",
            "downsampler_d_model", "downsampler_n_layers", "downsampler_n_heads", "downsampler_n_kv_heads",
            "downsampler_window", "downsampler_decode_past", "downsampler_decode_future", "downsampler_remat",
            "upsampler_d_model", "upsampler_n_layers", "upsampler_n_heads", "upsampler_n_kv_heads",
            "upsampler_window", "upsampler_decode_past", "upsampler_decode_future", "upsampler_remat",
            "token_dim", "token_n_heads", "token_head_type", "byte_group",
        )
        for f in singleton_uniform_fields:
            vals = getattr(self, f, None)
            if isinstance(vals, tuple) and len(vals) == n:
                assert len(set(vals)) <= 1, \
                    f"singleton CodeLM/Downsampler/Upsampler needs uniform '{f}' across ALL levels " \
                    f"(one shared module for the whole model, not one per level), got {vals}"
        assert len(self.token_head_type) == n
        assert len(self.pq_dim) == n
        assert all(t in ("linears", "ar") for t in self.token_head_type)
        for i in range(n):
            if self.token_head_type[i] == "ar":
                assert i < len(self.token_dim) and i < len(self.token_n_heads), \
                    f"level {i} uses token_head_type={self.token_head_type[i]!r} but token_dim/" \
                    f"token_n_heads only has {len(self.token_dim)} entries -- set one per level"
                assert self.token_dim[i] % self.token_n_heads[i] == 0
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
            kv = self.codelm_n_kv_heads[i] if self.codelm_n_kv_heads[i] is not None else max(1, self.codelm_n_heads[i] // 4)
            assert self.codelm_n_heads[i] % kv == 0
            assert self.codelm_d_model[i] % self.codelm_n_heads[i] == 0
            resolved_kv.append(kv)
        self.codelm_n_kv_heads = tuple(resolved_kv)
        if self.share_downsampler_upsampler_lm:
            assert (self.downsampler_d_model[0] == self.upsampler_d_model[0]
                    and self.downsampler_n_layers[0] == self.upsampler_n_layers[0]
                    and self.downsampler_n_heads[0] == self.upsampler_n_heads[0]
                    and self.downsampler_n_kv_heads[0] == self.upsampler_n_kv_heads[0]), \
                "share_downsampler_upsampler_lm=True needs downsampler_d_model/n_layers/n_heads/" \
                "n_kv_heads == the matching upsampler_* values (they'd share the same blocks/ln_f)"


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


def rgb_byte_pq_fn(flat_bytes: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    # Default CodeLM byte->pq conversion for the standard RGB case: images_to_positions already
    # produces (..., byte_group) with byte_group separate 0-255 channel values (not one packed
    # scalar) -- when byte_group==pq_chunks and code_vocab==256, each channel already IS one pq
    # digit directly, no bit-packing needed (same reasoning as rgb_label_fn_jax, which this
    # mirrors). For other factorizations (binary, hex, a genuinely packed byte_group=1 stream,
    # ...) pass a different byte_pq_fn explicitly -- e.g. byte_to_pq_idx_jax for bit-packing a
    # scalar byte into pq_chunks digits.
    assert code_vocab == 256, f"rgb_byte_pq_fn needs code_vocab=256 (one chunk per byte value), got {code_vocab}"
    assert flat_bytes.shape[-1] == pq_chunks, \
        f"rgb_byte_pq_fn needs byte_group(={flat_bytes.shape[-1]}) == pq_chunks(={pq_chunks})"
    return flat_bytes


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


def rgb_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order: np.ndarray, n_blocks: int,
                      pq_chunks: int, code_vocab: int, method: str = "bilinear") -> jnp.ndarray:
    # default_label_fn_jax grayscales (jnp.mean over channels) THEN bit-packs into pq_chunks -- for
    # pq_chunks=3,code_vocab=256 the packing degenerates to [byte, 0, 0] (byte_to_pq_idx_jax's >8-bit
    # branch only fills the first chunk), so on top of the grayscale collapse the result is
    # structurally red-only when read as RGB. This keeps all 3 real channels instead: chunk c IS
    # channel c's own downsampled byte value directly, no grayscale averaging, no bit-slicing.
    # Only valid when pq_chunks==3 and code_vocab==256 (asserted).
    assert pq_chunks == 3 and code_vocab == 256, \
        f"rgb_label_fn_jax needs pq_chunks=3,code_vocab=256 (one chunk per RGB channel), got {pq_chunks},{code_vocab}"
    M = flat_bytes.shape[0]
    pix_traversal = flat_bytes.reshape(M, cfg.img_size * cfg.img_size, 3).astype(jnp.float32)
    raster = jnp.zeros_like(pix_traversal).at[:, pixel_order, :].set(pix_traversal)
    img = raster.reshape(M, cfg.img_size, cfg.img_size, 3)
    side = max(1, round(math.sqrt(n_blocks)))
    small = jax.image.resize(img, (M, side, side, 3), method=method)
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)
    flat_rgb = small.reshape(M, side * side, 3)[:, low_order, :]
    if side * side > n_blocks:
        flat_rgb = flat_rgb[:, :n_blocks, :]
    elif side * side < n_blocks:
        flat_rgb = jnp.pad(flat_rgb, ((0, 0), (0, n_blocks - side * side), (0, 0)))
    return jnp.round(jnp.clip(flat_rgb, 0, 255)).astype(jnp.int32)


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
        # epoch_rng only ever draws one int per epoch (the epoch's own permutation seed) -- kept
        # separate from that per-epoch seed so a checkpoint mid-epoch can restore both the epoch's
        # exact shuffle (re-derivable from epoch_seed) and the future-epoch RNG stream, instead of
        # a raw post-permutation bit-generator state that can't regenerate the same permutation.
        self.epoch_rng = np.random.default_rng(seed)
        self.epoch_seed = None
        self.pos = 0  # batches already yielded this epoch -- resume continues here, no reshuffle
        self.pc, self.pi = jax.process_count(), jax.process_index()
        self.total = batch_size * n_devices
        self.cfg = cfg
        self.pixel_order = pixel_order_for(cfg)
        self.n_positions = n_positions_of(cfg)

    def __len__(self):
        return len(self.images) // (self.total * self.pc)

    def __iter__(self):
        n = len(self.images)
        if self.epoch_seed is None:
            self.epoch_seed = int(self.epoch_rng.integers(0, 2 ** 31 - 1))
            self.pos = 0
        idx = np.random.default_rng(self.epoch_seed).permutation(n) if self.shuffle else np.arange(n)
        g = self.total * self.pc
        starts = list(range(0, n - g + 1, g))
        for bi in range(self.pos, len(starts)):
            start = starts[bi]
            sel = idx[start + self.pi * self.total:start + (self.pi + 1) * self.total]
            img = self.images[sel]
            positions = images_to_positions(img, self.cfg, self.pixel_order)
            self.pos = bi + 1
            yield positions.reshape(self.n_devices, self.batch_size, self.n_positions, self.cfg.byte_group)
        self.epoch_seed = None
        self.pos = 0


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


class PardecLM(eqx.Module):
    # Shared pardec LM: downsampler and upsampler are both instances of this, independent weights,
    # differing only in output_expansion/vocab. Context = CodeLM's cached hidden states (via
    # context_proj), not a dedicated embed table.
    blocks: list
    ln_f: RMSNorm
    context_proj: jnp.ndarray
    bos_embed: jnp.ndarray  # (n_rates, hidden_dim) -- one BOS row per supported sample rate/stride,
    # so ONE set of weights can (eventually) be conditioned on which rate a group is decoding at,
    # simply by which row of bos_embed gets fed in (see pardec_score/pardec_generate's rate_id).
    # n_rates=1 (current default, every existing caller) keeps this identical to a single bos vector.
    target_embed: jnp.ndarray
    target_proj: jnp.ndarray
    # AR token head: predicts the output_chunks digits of each output token ONE AT A TIME (a small
    # inner causal attention over the chunks-so-far), not a single parallel linear -- matches
    # token_head_type="ar" elsewhere in this file (token_ar_teacher_forced/token_ar_generate).
    token_in_proj: jnp.ndarray
    token_member_embed: jnp.ndarray
    token_norm1: RMSNorm
    token_attn: Attention
    token_ln_f: RMSNorm
    token_out_head: jnp.ndarray
    # Parallel linear head, used ONLY by pardec_generate_gumbel (the downsampler's differentiable
    # training rollout -- no external target exists for that direction, see pardec_generate_gumbel's
    # docstring, so the finer per-chunk AR head above isn't used there; scoring/real generation
    # still use the AR head).
    output_head_linear: jnp.ndarray
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    n_rates: int = eqx.field(static=True)
    output_expansion: int = eqx.field(static=True)
    context_window_groups: int = eqx.field(static=True)  # -1 = unbounded, else bounded in GROUPS
    output_vocab: int = eqx.field(static=True)
    output_chunks: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    token_n_heads: int = eqx.field(static=True)
    decode_past: int = eqx.field(static=True)  # extra real-ground-truth tokens embedded as input
    # BEFORE bos, purely to give predictions extra causal context (see pardec_score) -- 0 = off
    decode_future: int = eqx.field(static=True)  # extra real-ground-truth tokens embedded as input
    # AFTER each group's own target span, scored separately as a widened-lookahead NTP aux signal
    # (see pardec_score) -- 0 = off
    remat: bool = eqx.field(static=True)

    def __init__(self, key, context_hidden_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 n_layers: int, mlp_mult: int, rope_base: float, output_expansion: int,
                 context_window_groups: int, output_vocab: int, output_chunks: int, pq_dim: int,
                 token_dim: int, token_n_heads: int, decode_past: int = 0, decode_future: int = 0,
                 n_rates: int = 1, init_scheme: str = "llama", use_xsa: bool = True,
                 use_qknorm: bool = True, remat: bool = False, window: int = None,
                 shared_blocks: list = None, shared_ln_f: RMSNorm = None):
        # shared_blocks/shared_ln_f (both None by default): when given, REUSE these instead of
        # building a fresh transformer stack -- the LM weight-sharing option (see
        # cfg.share_downsampler_upsampler_lm): downsampler and upsampler can share the SAME
        # blocks/ln_f (the core "LM"), while everything else (context_proj, bos_embed, target_embed/
        # proj, token_* AR head, output_head_linear) stays independent per instance. Both instances
        # must then agree on hidden_dim/n_heads/n_kv_heads/n_layers (asserted in Config).
        keys = jax.random.split(key, 8)
        if shared_blocks is not None:
            self.blocks = shared_blocks
            self.ln_f = shared_ln_f
        else:
            block_keys = jax.random.split(keys[0], n_layers)
            self.blocks = [Block(k, hidden_dim, n_heads, n_kv_heads, mlp_mult, rope_base, n_layers=n_layers,
                                  init_scheme=init_scheme, use_xsa=use_xsa, use_qknorm=use_qknorm,
                                  window=window) for k in block_keys]
            self.ln_f = RMSNorm(hidden_dim)
        self.context_proj = init_matrix(keys[1], (context_hidden_dim, hidden_dim), init_scheme)
        self.n_rates = n_rates
        bos_keys = jax.random.split(keys[2], n_rates)
        self.bos_embed = jnp.stack([init_vector(k, hidden_dim, init_scheme) for k in bos_keys], axis=0)
        self.target_embed = init_matrix(keys[3], (output_vocab, pq_dim), init_scheme)
        self.target_proj = init_matrix(keys[4], (output_chunks * pq_dim, hidden_dim), init_scheme)
        self.token_in_proj = init_matrix(keys[5], (hidden_dim, token_dim), init_scheme)
        self.token_member_embed = init_matrix(keys[6], (output_vocab, token_dim), init_scheme)
        self.token_norm1 = RMSNorm(token_dim)
        self.token_attn = Attention(keys[7], token_dim, token_n_heads, token_n_heads, rope_base, n_layers=1,
                                     init_scheme=init_scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
        self.token_ln_f = RMSNorm(token_dim)
        self.token_out_head = init_matrix(jax.random.fold_in(key, 8001), (token_dim, output_vocab), init_scheme)
        self.output_head_linear = init_matrix(jax.random.fold_in(key, 8002),
                                               (hidden_dim, output_chunks * output_vocab), init_scheme)
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.output_expansion, self.context_window_groups = output_expansion, context_window_groups
        self.output_vocab, self.output_chunks = output_vocab, output_chunks
        self.token_dim, self.token_n_heads = token_dim, token_n_heads
        self.decode_past, self.decode_future = decode_past, decode_future
        self.remat = remat


def pardec_context_windows(pardec: PardecLM, context_h_padded: jnp.ndarray, context_group_size: int,
                            n_groups: int, n_context_positions: int) -> tuple:
    # Windowed, per-group-batched context rows from CodeLM's cached hidden states (proven
    # _pardec_ctx_rows' "own window" branch, no extra-context-source loop). context_group_size is
    # the raw context's own granularity -- equals output_group_size for the upsampler, but is
    # LARGER for the downsampler (many raw positions feed one batched group of output codes).
    # context_h_padded must already be padded to n_groups*context_group_size positions.
    batch = context_h_padded.shape[0]
    context_tok = context_h_padded @ pardec.context_proj
    hidden_dim = context_tok.shape[-1]
    n_context_positions_padded = n_groups * context_group_size
    window_size = (n_context_positions_padded if pardec.context_window_groups < 0 else
                   min(pardec.context_window_groups * context_group_size + context_group_size, n_context_positions_padded))
    padded = jnp.pad(context_tok, ((0, 0), (window_size, 0), (0, 0)))
    group_ends = [(g + 1) * context_group_size for g in range(n_groups)]
    own_window = jnp.stack([padded[:, end:end + window_size, :] for end in group_ends], axis=1)
    own_window = own_window.reshape(batch * n_groups, window_size, hidden_dim)
    rope_ids = jnp.stack([jnp.clip(jnp.arange(window_size) - window_size + end, 0, None)
                           for end in group_ends], axis=0)
    position_offsets = np.stack([np.arange(window_size) - window_size + end for end in group_ends])
    valid_mask = jnp.broadcast_to(
        jnp.asarray((position_offsets >= 0) & (position_offsets < n_context_positions))[None],
        (batch, n_groups, window_size)).reshape(batch * n_groups, window_size)
    return own_window, rope_ids, valid_mask, window_size, group_ends


def pardec_score(pardec: PardecLM, target_seq: jnp.ndarray, context_h: jnp.ndarray,
                  context_group_size: int, output_group_size: int, rate_id: int = 0,
                  output_expansion: int = None) -> tuple:
    # Teacher-forced scoring (basic pardec, see PardecLM). context_h = CodeLM's cached hidden
    # states, already contextualized (replaces ctx_code_soft/ctx_embed). target_seq is at the
    # OUTPUT's own granularity, length = n_context_positions*output_group_size//context_group_size*output_expansion.
    # output_expansion: plain int, None (default) falls back to pardec.output_expansion (the
    # construction-time value) -- pass explicitly to call the SAME shared PardecLM at a different
    # stride/rate (paired with rate_id selecting the matching bos_embed row). Kept a plain Python
    # int (not a traced array), same as context_group_size/output_group_size, so eqx.filter_jit
    # treats it as static and retraces per distinct stride, matching the existing convention.
    oe = pardec.output_expansion if output_expansion is None else output_expansion
    batch, n_context_positions, _ = context_h.shape
    n_groups = -(-n_context_positions // context_group_size)
    pad_amount = n_groups * context_group_size - n_context_positions
    context_h_padded = jnp.pad(context_h, ((0, 0), (0, pad_amount), (0, 0))) if pad_amount > 0 else context_h
    own_window, rope_ids, valid_mask, window_size, group_ends = pardec_context_windows(
        pardec, context_h_padded, context_group_size, n_groups, n_context_positions)
    hidden_dim = own_window.shape[-1]

    decode_past, decode_future = pardec.decode_past, pardec.decode_future
    target_len_per_group = output_group_size * oe
    widened_len = decode_past + target_len_per_group + decode_future
    n_output_positions = n_context_positions * output_group_size // context_group_size
    target_len_per_group_padded = n_groups * output_group_size * oe - n_output_positions * oe
    target_padded = target_seq
    if target_len_per_group_padded > 0:
        target_padded = jnp.pad(target_seq, ((0, 0), (0, target_len_per_group_padded), (0, 0)))

    if decode_future > 0:
        tail_padded = jnp.pad(target_padded, ((0, 0), (0, decode_future), (0, 0)))
        real_tail_windows = jnp.stack(
            [tail_padded[:, g * target_len_per_group:g * target_len_per_group + target_len_per_group + decode_future]
             for g in range(n_groups)], axis=1)
    else:
        real_tail_windows = target_padded.reshape(batch, n_groups, target_len_per_group, *target_seq.shape[2:])
    real_tail_flat = real_tail_windows.reshape(batch * n_groups, target_len_per_group + decode_future, *target_seq.shape[2:])
    real_tail_embedded = code_embed_proj(real_tail_flat, pardec.target_embed, pardec.target_proj)

    if decode_past > 0:
        draft_padded = jnp.pad(target_padded, ((0, 0), (decode_past, 0), (0, 0)))
        draft_windows = jnp.stack(
            [draft_padded[:, g * target_len_per_group:g * target_len_per_group + decode_past]
             for g in range(n_groups)], axis=1)
        draft_flat = draft_windows.reshape(batch * n_groups, decode_past, *target_seq.shape[2:])
        draft_embedded = code_embed_proj(draft_flat, pardec.target_embed, pardec.target_proj)
        draft_valid = _draft_past_valid_mask(n_groups, decode_past, target_len_per_group, n_output_positions * oe)
        draft_valid_flat = jnp.broadcast_to(jnp.asarray(draft_valid)[None], (batch, n_groups, decode_past)).reshape(batch * n_groups, decode_past)
        draft_embedded = jnp.where(draft_valid_flat[:, :, None], draft_embedded, 0.0)
        target_embedded_flat = jnp.concatenate([draft_embedded, real_tail_embedded], axis=1)
    else:
        draft_valid_flat = jnp.ones((batch * n_groups, 0), dtype=bool)
        target_embedded_flat = real_tail_embedded

    bos = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch * n_groups, 1, hidden_dim))
    row_flat = jnp.concatenate([own_window, bos, target_embedded_flat], axis=1)
    per_group_len = window_size + 1 + decode_past + target_len_per_group + decode_future

    key_valid = jnp.concatenate([valid_mask, jnp.ones((batch * n_groups, 1), dtype=bool), draft_valid_flat,
                                  jnp.ones((batch * n_groups, target_len_per_group + decode_future), dtype=bool)], axis=1)
    rope_bos = jnp.array(group_ends)[:, None]
    rope_draft = jnp.stack([end - decode_past + jnp.arange(decode_past) for end in group_ends], axis=0)
    rope_real_tail = jnp.stack([end + 1 + jnp.arange(target_len_per_group + decode_future) for end in group_ends], axis=0)
    rope_target = jnp.clip(jnp.concatenate([rope_draft, rope_real_tail], axis=1), 0, None)
    rope_pos_ids_g = jnp.concatenate([rope_ids, rope_bos, rope_target], axis=1)
    rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (batch, n_groups, per_group_len)).reshape(batch * n_groups, per_group_len)

    def run_stack(x):
        for blk in pardec.blocks:
            x = run_block_pardec(blk, x, rope_pos_ids, key_valid, pardec.remat)
        return x
    hidden = jax.checkpoint(run_stack)(row_flat) if pardec.remat else run_stack(row_flat)
    hidden = pardec.ln_f(hidden)
    prediction_positions = window_size + decode_past + jnp.arange(target_len_per_group)
    predicted_hidden = hidden[:, prediction_positions, :]
    predicted_hidden = predicted_hidden.reshape(batch, n_groups * target_len_per_group, hidden_dim)
    valid_len = n_output_positions * oe
    predicted_hidden = predicted_hidden[:, :valid_len, :]
    target_out = target_seq[:, :valid_len]
    logits = token_ar_teacher_forced(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                      pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                      pardec.token_dim, pardec.output_vocab, predicted_hidden, target_out)
    return logits, target_out


def pardec_generate(pardec: PardecLM, context_h: jnp.ndarray, context_group_size: int,
                     output_group_size: int, rng, greedy: bool = True, temperature: float = 1.0,
                     top_k: int = 0, rate_id: int = 0, output_expansion: int = None) -> jnp.ndarray:
    # Generation counterpart of pardec_score: one growing KV cache per group, all groups batched.
    # output_expansion: see pardec_score's docstring -- plain int, None falls back to
    # pardec.output_expansion, pass explicitly to call the SAME shared PardecLM at a different
    # stride/rate (paired with rate_id).
    # TODO: decode_past not implemented here -- needs the previous group's generated tail as real
    # input, a cross-group dependency this all-groups-parallel scan doesn't handle yet.
    oe = pardec.output_expansion if output_expansion is None else output_expansion
    batch, n_context_positions, _ = context_h.shape
    n_groups = -(-n_context_positions // context_group_size)
    pad_amount = n_groups * context_group_size - n_context_positions
    context_h_padded = jnp.pad(context_h, ((0, 0), (0, pad_amount), (0, 0))) if pad_amount > 0 else context_h
    own_window, rope_ids, valid_mask, window_size, group_ends = pardec_context_windows(
        pardec, context_h_padded, context_group_size, n_groups, n_context_positions)
    hidden_dim = own_window.shape[-1]
    batch2 = batch * n_groups
    target_len_per_group = output_group_size * oe
    total_steps = window_size + 1 + target_len_per_group

    hidden_dim_per_head = hidden_dim // pardec.n_heads
    cache_k0 = jnp.zeros((len(pardec.blocks), batch2, pardec.n_kv_heads, total_steps, hidden_dim_per_head))
    cache_v0 = jnp.zeros_like(cache_k0)

    def self_step(x_new, cache_k, cache_v, pos):
        new_cache_k, new_cache_v = [], []
        x = x_new
        for i, blk in enumerate(pardec.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], pos, total_steps)
            new_cache_k.append(ck_i)
            new_cache_v.append(cv_i)
        return pardec.ln_f(x), jnp.stack(new_cache_k), jnp.stack(new_cache_v)

    # context prefill: feed the window one position at a time -- blk.step handles a single new
    # position per call (2D x_new, no seq-len axis), so this is a scan over the window.
    def context_step(carry, x_t):
        cache_k, cache_v, pos = carry
        _, cache_k, cache_v = self_step(x_t, cache_k, cache_v, pos)
        return (cache_k, cache_v, pos + 1), None

    (cache_k, cache_v, pos), _ = jax.lax.scan(
        context_step, (cache_k0, cache_v0, jnp.array(0)), jnp.swapaxes(own_window, 0, 1))

    bos_in = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch2, hidden_dim))
    hidden, cache_k, cache_v = self_step(bos_in, cache_k, cache_v, pos)
    pos = pos + 1

    def gen_step(carry, _):
        cache_k, cache_v, pos, rng, x_input, hidden = carry
        val, rng = token_ar_generate(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                      pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                      pardec.output_chunks, hidden, rng, greedy, temperature, top_k)
        x_next = code_embed_proj(val, pardec.target_embed, pardec.target_proj)
        hidden_next, cache_k, cache_v = self_step(x_next, cache_k, cache_v, pos)
        return (cache_k, cache_v, pos + 1, rng, x_next, hidden_next), val

    init_carry = (cache_k, cache_v, pos, rng, bos_in, hidden)
    _, vals = jax.lax.scan(gen_step, init_carry, None, length=target_len_per_group)
    vals = jnp.moveaxis(vals, 0, 1)  # (batch2, target_len_per_group, output_chunks)
    vals = vals.reshape(batch, n_groups * target_len_per_group, *vals.shape[2:])
    n_output_positions = n_context_positions * output_group_size // context_group_size
    valid_len = n_output_positions * oe
    return vals[:, :valid_len]


def pardec_generate_gumbel(pardec: PardecLM, context_h: jnp.ndarray, context_group_size: int,
                            output_group_size: int, rng, temperature: float = 1.0,
                            quantize_drop: float = 0.0, rate_id: int = 0) -> tuple:
    # Differentiable autoregressive rollout for the DOWNSAMPLER direction: unlike pardec_score
    # (teacher-forced against a real target), there IS no real target here -- the output codes are
    # learned latents, exactly like the existing code_head/quantize_gumbel bottleneck elsewhere in
    # this file, just produced one at a time autoregressively (via a real growing KV cache) instead
    # of in one parallel pooled shot. Reuses quantize_gumbel (proven) for the per-step quantization,
    # via a plain linear head (output_head_linear) rather than the AR token head -- the outer
    # sequence is already autoregressive (each step sees all previous steps' real KV), so the finer
    # per-chunk AR head's extra within-code dependency isn't needed for this training-only path;
    # actual generation/inference still uses pardec_generate's AR head. Requires
    # pardec.output_expansion == 1 (one code per outer AR step, no further sub-expansion).
    assert pardec.output_expansion == 1, \
        f"pardec_generate_gumbel needs output_expansion=1 (downsampler only), got {pardec.output_expansion}"
    batch, n_context_positions, _ = context_h.shape
    n_groups = -(-n_context_positions // context_group_size)
    pad_amount = n_groups * context_group_size - n_context_positions
    context_h_padded = jnp.pad(context_h, ((0, 0), (0, pad_amount), (0, 0))) if pad_amount > 0 else context_h
    own_window, rope_ids, valid_mask, window_size, group_ends = pardec_context_windows(
        pardec, context_h_padded, context_group_size, n_groups, n_context_positions)
    hidden_dim = own_window.shape[-1]
    batch2 = batch * n_groups
    target_len_per_group = output_group_size
    total_steps = window_size + 1 + target_len_per_group

    hidden_dim_per_head = hidden_dim // pardec.n_heads
    cache_k0 = jnp.zeros((len(pardec.blocks), batch2, pardec.n_kv_heads, total_steps, hidden_dim_per_head))
    cache_v0 = jnp.zeros_like(cache_k0)

    def self_step(x_new, cache_k, cache_v, pos):
        new_cache_k, new_cache_v = [], []
        x = x_new
        for i, blk in enumerate(pardec.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], pos, total_steps)
            new_cache_k.append(ck_i)
            new_cache_v.append(cv_i)
        return pardec.ln_f(x), jnp.stack(new_cache_k), jnp.stack(new_cache_v)

    def context_step(carry, x_t):
        cache_k, cache_v, pos = carry
        _, cache_k, cache_v = self_step(x_t, cache_k, cache_v, pos)
        return (cache_k, cache_v, pos + 1), None

    (cache_k, cache_v, pos), _ = jax.lax.scan(
        context_step, (cache_k0, cache_v0, jnp.array(0)), jnp.swapaxes(own_window, 0, 1))

    bos_in = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch2, hidden_dim))
    hidden, cache_k, cache_v = self_step(bos_in, cache_k, cache_v, pos)
    pos = pos + 1

    def gen_step(carry, step_rng):
        cache_k, cache_v, pos, hidden = carry
        logits = reshape_pq(hidden @ pardec.output_head_linear, pardec.output_chunks, pardec.output_vocab)
        code_soft, code_idx = quantize_gumbel(logits, step_rng, temperature, quantize_drop)
        x_next = code_embed_proj(code_soft, pardec.target_embed, pardec.target_proj)
        hidden_next, cache_k, cache_v = self_step(x_next, cache_k, cache_v, pos)
        return (cache_k, cache_v, pos + 1, hidden_next), (code_soft, code_idx)

    step_rngs = jax.random.split(rng, target_len_per_group)
    init_carry = (cache_k, cache_v, pos, hidden)
    _, (code_softs, code_idxs) = jax.lax.scan(gen_step, init_carry, step_rngs)
    code_softs = jnp.moveaxis(code_softs, 0, 1).reshape(batch, n_groups * target_len_per_group, *code_softs.shape[2:])
    code_idxs = jnp.moveaxis(code_idxs, 0, 1).reshape(batch, n_groups * target_len_per_group, *code_idxs.shape[2:])
    n_output_positions = n_context_positions * output_group_size // context_group_size
    return code_softs[:, :n_output_positions], code_idxs[:, :n_output_positions]


class CodeLM(eqx.Module):
    # The "encoder" of the seq2seq framing (CodeLM=encoder, downsampler/upsampler=two separate
    # decoders). SINGLETON: exactly one CodeLM for the whole model, shared across all levels --
    # NOT one per level. A level's raw byte input (level 0) and a level's code input (level>0)
    # already share the same categorical structure by construction (byte_group==pq_chunks,
    # 256==code_vocab, enforced uniform in Config.__post_init__), which is what makes one shared
    # embedding/code_head/ntp_head correct for every level. K (stride) is a runtime grouping
    # argument passed to encode(), not baked into any weight shape -- levels may still use
    # different K. "Which level" is communicated via rate_id into bos_embed (see use_codelm_bos),
    # not via separate weights -- that's the whole point of this being a singleton.
    blocks: list
    ln_f: RMSNorm
    own_input_embed: jnp.ndarray
    own_input_proj: jnp.ndarray
    ntp_head: jnp.ndarray
    code_head: jnp.ndarray
    bos_embed: jnp.ndarray  # (n_levels, D_enc) if use_codelm_bos else (1, D_enc) -- one anchor row
    # per level index (rate_id IS the level index), always allocated (inert when
    # use_codelm_bos=False) so toggling that flag alone never changes checkpoint structure.
    remat: bool = eqx.field(static=True)
    remat_level: bool = eqx.field(static=True)
    attn_lookahead: int = eqx.field(static=True)
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    quantize_mode: str = eqx.field(static=True)
    quantize_drop: float = eqx.field(static=True)
    use_codelm_bos: bool = eqx.field(static=True)
    codelm_bos_prob: float = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        # Single shared architecture: Config.__post_init__ asserts codelm_d_model/codelm_n_layers/
        # codelm_n_heads/codelm_n_kv_heads/pq_chunks/code_vocab/pq_dim/mlp_mult/rope_base/
        # attn_lookahead/attn_window are uniform across ALL levels -- index 0 is the representative
        # value. codelm_* are CodeLM's own fully independent fields (no fallback to anything else).
        D_enc = cfg.codelm_d_model[0]
        n_layers_enc = cfg.codelm_n_layers[0]
        n_heads_enc = cfg.codelm_n_heads[0]
        n_kv_heads_enc = cfg.codelm_n_kv_heads[0]
        self.remat = cfg.remat
        self.remat_level = cfg.remat_level
        self.attn_lookahead = cfg.attn_lookahead[0]
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[0], cfg.code_vocab[0]
        self.quantize_mode = cfg.quantize_mode
        self.quantize_drop = cfg.quantize_drop
        self.use_codelm_bos = cfg.use_codelm_bos
        self.codelm_bos_prob = cfg.codelm_bos_prob
        pq_dim = cfg.pq_dim[0]
        own_vocab = self.code_vocab  # raw bytes and codes share one categorical structure (see above)
        ntp_out = self.pq_chunks * self.code_vocab
        keys = jax.random.split(key, 4)

        scheme, use_xsa, use_qknorm = cfg.init_scheme, cfg.use_xsa, cfg.use_qknorm
        self.own_input_embed = init_matrix(keys[0], (own_vocab, pq_dim), scheme)
        self.own_input_proj = init_matrix(keys[1], (self.pq_chunks * pq_dim, D_enc), scheme)
        block_keys = jax.random.split(jax.random.fold_in(key, 8801), n_layers_enc)
        enc_window_val = cfg.encoder_attn_window[0] if cfg.encoder_attn_window[0] is not None else cfg.attn_window[0]
        enc_window = None if enc_window_val == -1 else enc_window_val
        self.blocks = [Block(k, D_enc, n_heads_enc, n_kv_heads_enc, cfg.mlp_mult[0], cfg.rope_base[0],
                             n_layers=n_layers_enc, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm,
                             window=enc_window, lookahead=self.attn_lookahead, use_sink=cfg.use_sink) for k in block_keys]
        self.ln_f = RMSNorm(D_enc)
        self.code_head = init_matrix(keys[2], (D_enc, self.pq_chunks * self.code_vocab), scheme)
        self.ntp_head = init_matrix(keys[3], (D_enc, ntp_out), scheme)
        n_levels = len(cfg.strides)
        n_bos_rates = n_levels if cfg.use_codelm_bos else 1  # rate_id IS the level index -- derived,
        # not a separate manual knob, since there's exactly one CodeLM and n_levels is already known
        bos_keys = jax.random.split(jax.random.fold_in(key, 8901), n_bos_rates)
        self.bos_embed = jnp.stack([init_vector(k, D_enc, scheme) for k in bos_keys], axis=0)

    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, K: int, rng=None, encode_temperature: float = 1.0,
               layer_drop_prob=None, rate_id: int = 0, force_bos: bool = False) -> dict:
        if force_bos:
            x = x.at[:, 0, :].set(self.bos_embed[rate_id])
        elif self.use_codelm_bos and rng is not None:
            B = x.shape[0]
            do_sub = jax.random.bernoulli(jax.random.fold_in(rng, 424242), p=self.codelm_bos_prob, shape=(B,))
            x0 = jnp.where(do_sub[:, None], self.bos_embed[rate_id], x[:, 0, :])
            x = x.at[:, 0, :].set(x0)
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
        n_blocks = L // K
        # TODO: CodePoolAttention removed (superseded by the PardecLM-based downsampler, not yet
        # wired in) -- naive position-(K-1) pick only, for now.
        h_blocks = h[:, :n_blocks * K, :].reshape(M, n_blocks, K, D)
        pooled = h_blocks[:, :, K - 1, :]
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
            ntp_logits = reshape_pq(h[:, :-ntp_shift, :] @ self.ntp_head, self.pq_chunks, self.code_vocab)
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




def encode_pardec_downsampler(codelm: CodeLM, downsampler: PardecLM, x: jnp.ndarray, target_idx: jnp.ndarray,
                               flat_bytes: jnp.ndarray, cfg: "Config", pixel_order, label_fn,
                               K: int, rate_id: int = 0, rng=None) -> dict:
    # CodeLM forward (same as CodeLM.encode()'s first half), then the SHARED downsampler PardecLM
    # teacher-forced against label_fn's real downsampled-image target (context_group_size=K,
    # output_group_size=1 -- one code per K-block, genuinely autoregressive over context). No
    # gumbel/quantize_hard here -- this is supervised classification against a real target, not an
    # unsupervised VQ bottleneck, so code_idx/code_soft come directly from the logits. Returns the
    # SAME dict shape as CodeLM.encode() so level_forward's surrounding loss code doesn't change.
    h = x
    def _enc_stack(h):
        for blk in codelm.blocks:
            h = run_block(blk, h, codelm.remat and not codelm.remat_level)
        return h
    h = jax.checkpoint(_enc_stack)(h) if codelm.remat_level else _enc_stack(h)
    h = codelm.ln_f(h)
    M, L, D = h.shape
    n_blocks = L // K
    label_tgt = label_fn(flat_bytes, cfg, pixel_order, n_blocks, codelm.pq_chunks, codelm.code_vocab)
    # context_group_size=K, rate_id=this level's index -- the SAME shared downsampler weights are
    # called with a different (K, rate_id) pair per level, modulated via the downsampler's own
    # bos_embed row (see PardecLM), not via separate weights.
    logits, _ = pardec_score(downsampler, label_tgt, h, context_group_size=K, output_group_size=1, rate_id=rate_id)
    code_idx = jnp.argmax(logits, axis=-1)
    code_soft = jax.nn.softmax(logits, axis=-1)

    probs = jax.nn.softmax(logits, axis=-1)
    p_avg = jnp.mean(probs, axis=(0, 1))
    entropy_loss = jnp.mean(jnp.sum(p_avg * jnp.log(jnp.maximum(p_avg, 1e-9)), axis=-1))

    ntp_shift = 1 + codelm.attn_lookahead
    if L > ntp_shift:
        ntp_logits = reshape_pq(h[:, :-ntp_shift, :] @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab)
        tgt = target_idx[:, ntp_shift:]
        logp = jax.nn.log_softmax(ntp_logits, axis=-1)
        ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
        ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
    else:
        ntp_loss = jnp.array(0.0, dtype=h.dtype)
        ntp_acc = jnp.array(0.0, dtype=h.dtype)
    util = codebook_utilization(code_idx, codelm.code_vocab)
    return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                entropy_loss=entropy_loss, logits=logits)


def decode_logits_and_target_multipass(model: "LagCodecModel", level_idx: int, target_seq: jnp.ndarray,
                                        ctx_code_soft: jnp.ndarray, decoder_ncodes: int, rng=None,
                                        **_unused) -> tuple:
    # SHARED upsampler, modulated by (output_expansion=K(level_idx), rate_id=level_idx) -- not a
    # separate weight set per level. **_unused absorbs any stale kwargs from callers.
    # No special-casing: the Upsampler's context is built EXACTLY like the Downsampler's -- CodeLM
    # is the shared feature extractor for both. Run this level's own code through CodeLM's own
    # embed table + block stack (same as encoder_hidden/CodeLM.encode()'s first half) to get real
    # contextualized hidden states, not a separate dedicated ctx_embed/ctx_proj lookup shortcut.
    codelm = model.codelm
    x_ctx = code_embed_proj(ctx_code_soft, codelm.own_input_embed, codelm.own_input_proj)
    h_ctx = encoder_hidden(codelm, x_ctx)
    logits, target_out = pardec_score(model.upsampler, target_seq, h_ctx,
                                       context_group_size=decoder_ncodes, output_group_size=decoder_ncodes,
                                       rate_id=level_idx, output_expansion=model.K(level_idx))
    zero = jnp.array(0.0, dtype=logits.dtype)
    return logits, target_out, None, zero, zero


def _decode_generate_pardec_call(model, level_idx, ctx_idx, decoder_ncodes, greedy, temperature, seed):
    codelm = model.codelm
    x_ctx = code_embed_proj(ctx_idx, codelm.own_input_embed, codelm.own_input_proj)
    h_ctx = encoder_hidden(codelm, x_ctx)
    rng = jax.random.PRNGKey(seed)
    return pardec_generate(model.upsampler, h_ctx, context_group_size=decoder_ncodes,
                            output_group_size=decoder_ncodes, rng=rng, greedy=greedy, temperature=temperature,
                            top_k=model.cfg.gen_top_k, rate_id=level_idx, output_expansion=model.K(level_idx))


_decode_generate_pardec_jit = eqx.filter_jit(_decode_generate_pardec_call)


def decode_generate_multipass(model: "LagCodecModel", level_idx: int, ctx_idx: jnp.ndarray, decoder_ncodes: int,
                               greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
    return _decode_generate_pardec_jit(model, level_idx, ctx_idx, decoder_ncodes, greedy, temperature, seed)


def encoder_hidden(codelm: CodeLM, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for blk in codelm.blocks:
        h = run_block(blk, h, False)
    return codelm.ln_f(h)


def encoder_ntp_logits(codelm: CodeLM, h: jnp.ndarray) -> jnp.ndarray:
    return reshape_pq(h @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab)


def _sample_tokens(logits: jnp.ndarray, rng, greedy: bool, temperature, top_k: int = 0) -> jnp.ndarray:
    # gumbel-max with safe_argmax (jnp.argmax feeding a gather is miscompiled on TPU, see safe_argmax)
    if greedy:
        return safe_argmax(logits)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        kth = jax.lax.top_k(lg, top_k)[0][..., -1:]
        lg = jnp.where(lg < kth, -jnp.inf, lg)
    return safe_argmax(lg + jax.random.gumbel(rng, lg.shape))


def _encoder_free_run(codelm: CodeLM, tokens: jnp.ndarray, P, K: int, rng, temperature, greedy: bool,
                      top_k: int, use_bos: bool = False, rate_id: int = 0) -> jnp.ndarray:
    # tokens (B,total_len,C) holds the prompt in [:P] (P may be traced); the rest is overwritten
    x = code_embed_proj(tokens, codelm.own_input_embed, codelm.own_input_proj)
    if use_bos:
        # position 0's discrete `tokens[:,0]` value is meaningless once its embedding is replaced --
        # only the embedding matters for this level's own free-run (P=1, everything after is genuinely
        # free-sampled with zero real content: a true unconditional/free rollout, not just a small
        # real prompt). Downstream re-encoding of the returned tokens is the caller's concern.
        x = x.at[:, 0, :].set(codelm.bos_embed[rate_id])

    def body(t, carry):
        tokens, x = carry
        # exact training-time encoder forward over the whole fixed-length buffer; the encoder is causal, so
        # positions <= t-1 never see the not-yet-generated (junk) positions after them
        h = encoder_hidden(codelm, x)
        lg = encoder_ntp_logits(codelm, jax.lax.dynamic_index_in_dim(h, t - 1, axis=1, keepdims=False))
        tok = _sample_tokens(lg, jax.random.fold_in(rng, t), greedy, temperature, top_k)
        tokens = tokens.at[:, t].set(tok.astype(tokens.dtype))
        x = x.at[:, t].set(code_embed_proj(tok, codelm.own_input_embed, codelm.own_input_proj))
        return tokens, x

    tokens, _ = jax.lax.fori_loop(P, tokens.shape[1], body, (tokens, x))
    return tokens


_encoder_free_run_jit = eqx.filter_jit(_encoder_free_run)


def encoder_free_run(codelm: CodeLM, prompt_tokens: jnp.ndarray, total_len: int, K: int, rng, greedy: bool = False,
                     temperature: float = 1.0, top_k: int = 0, use_bos: bool = False, rate_id: int = 0) -> jnp.ndarray:
    """Free-run the SHARED CodeLM as a language model over its own input tokens (its NTP head), at
    whichever level's granularity K (a runtime grouping argument, not baked into any weight):
    keep the prompt tokens, then sample the rest. greedy=True is argmax; otherwise temperature/top_k
    sampling. use_bos=True (needs cfg.use_codelm_bos): replace position 0's embedding with
    codelm.bos_embed[rate_id] regardless of prompt_tokens -- a true free rollout (P should be 1),
    not a real-byte-prompted completion."""
    assert codelm.attn_lookahead == 0, \
        f"encoder free-run needs attn_lookahead=0 (got {codelm.attn_lookahead}): a lookahead shifts the NTP target"
    assert not use_bos or codelm.use_codelm_bos, \
        "use_bos=True needs cfg.use_codelm_bos=True (bos_embed was never trained)"
    B, P, C = prompt_tokens.shape
    assert 1 <= P <= total_len, f"prompt length {P} must be in [1, {total_len}]"
    tokens = jnp.zeros((B, total_len, C), prompt_tokens.dtype).at[:, :P].set(prompt_tokens)
    return _encoder_free_run_jit(codelm, tokens, jnp.asarray(P, jnp.int32), K, rng,
                                 jnp.asarray(temperature, jnp.float32), greedy, top_k, use_bos, rate_id)


def generate_from_prompt(model: "LagCodecModel", cfg: Config, prompt_bytes: jnp.ndarray, total_positions: int,
                          sample_level: int, rng, greedy: bool = False, temperature: float = 1.0,
                          top_k: int = 0, encode_temperature: float = 1.0, decode_greedy: bool = True,
                          decode_temperature: float = 1.0, decode_seed: int = 0, byte_pq_fn=None) -> dict:
    """Prompted generation through the SHARED CodeLM's own next-token head (only the top two levels
    allowed). prompt_bytes (B,P,byte_group) = leading positions of an image. byte_pq_fn (default
    rgb_byte_pq_fn) converts raw bytes into CodeLM's own (pq_chunks, code_vocab) representation --
    user-settable, e.g. byte_to_pq_idx_jax for a bit-packed (binary/hex) factorization instead of
    the default direct-RGB-channel mapping. The prompt is encoded up to `sample_level`, that level
    is free-run (sampling CodeLM's own input tokens: bytes at level 0, level-(L-1) codes above),
    the completed sequence is encoded back up to the top, and those codes are decoded down the
    usual cascade. Returns image bytes for decode_from "emitted" (cascade from the top emitted
    code) and, when available, "sampled" (the free-run tokens themselves)."""
    n = len(cfg.strides)
    assert n - 2 <= sample_level <= n - 1, f"sample_level {sample_level} must be one of the top two levels of {n}"
    codelm = model.codelm
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    K0 = model.K(0)
    B, P, _ = prompt_bytes.shape
    assert P % K0 == 0, f"prompt length {P} must be a multiple of the level-0 stride {K0}"
    tok = byte_pq_fn(prompt_bytes, codelm.pq_chunks, codelm.code_vocab)
    for i in range(sample_level):
        x = code_embed_proj(tok, codelm.own_input_embed, codelm.own_input_proj)
        tok = codelm.encode(x, tok, model.K(i), rng=None, encode_temperature=encode_temperature, rate_id=i)["code_idx"]
        assert tok.shape[1] >= 1, "prompt too short to produce a single code at the sampling level"
    ds = 1
    for i in range(sample_level):
        ds *= model.K(i)
    tokens_L = encoder_free_run(codelm, tok, total_positions // ds, model.K(sample_level), rng, greedy,
                                 temperature, top_k, rate_id=sample_level)

    codes, x, tgt = {}, code_embed_proj(tokens_L, codelm.own_input_embed, codelm.own_input_proj), tokens_L
    for i in range(sample_level, n):
        out = codelm.encode(x, tgt, model.K(i), rng=None, encode_temperature=encode_temperature, rate_id=i)
        codes[i] = out["code_idx"]
        if i < n - 1:
            x = code_embed_proj(out["code_soft"], codelm.own_input_embed, codelm.own_input_proj)
            tgt = out["code_idx"]

    def cascade(cur, from_level):
        for i in range(from_level, -1, -1):
            cur = decode_generate_multipass(model, i, cur, cfg.decoder_ncodes[i], greedy=decode_greedy,
                                            temperature=decode_temperature, seed=decode_seed)
        return cur

    res = dict(sampled_tokens=tokens_L, emitted_codes=codes, emitted=cascade(codes[n - 1], n - 1))
    if sample_level == 0:
        res["sampled"] = tokens_L
    else:
        res["sampled"] = cascade(tokens_L, sample_level - 1)
    return res


class LagCodecModel(eqx.Module):
    # SINGLETON model: exactly one CodeLM, one Downsampler, one Upsampler total -- NOT one per
    # level. "Which level" is communicated to each shared module via rate_id (a bos_embed row),
    # not via separate weights -- see CodeLM/PardecLM. Architecture (d_model, n_layers, pq_chunks,
    # code_vocab, pq_dim, ...) must therefore be uniform across ALL levels (enforced in
    # Config.__post_init__); only runtime grouping (K, decoder_ncodes, ncodes_window, decode_past/
    # future, ...) may still vary per level, passed as plain call-time arguments.
    codelm: CodeLM
    downsampler: PardecLM
    upsampler: PardecLM
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        keys = jax.random.split(key, 4)
        self.codelm = CodeLM(keys[0], cfg)
        # CodeLM is the shared feature extractor for BOTH decode heads: Downsampler's context is
        # CodeLM's hidden states from this level's own INPUT; Upsampler's context is CodeLM's
        # hidden states from re-running this level's own CODE through the exact same embed table +
        # block stack (see decode_logits_and_target_multipass/_decode_generate_pardec_call) -- no
        # special-cased separate ctx_embed/ctx_proj shortcut for the Upsampler. Both therefore share
        # the same context_hidden_dim = CodeLM's own D_enc.
        D_enc = cfg.codelm_d_model[0]
        pq_dim, code_vocab, pq_chunks = cfg.pq_dim[0], cfg.code_vocab[0], cfg.pq_chunks[0]
        scheme, use_xsa, use_qknorm = cfg.init_scheme, cfg.use_xsa, cfg.use_qknorm

        downsampler_remat = cfg.remat if cfg.downsampler_remat[0] is None else cfg.downsampler_remat[0]
        self.downsampler = PardecLM(
            jax.random.fold_in(keys[1], 9101), context_hidden_dim=D_enc, hidden_dim=cfg.downsampler_d_model[0],
            n_heads=cfg.downsampler_n_heads[0], n_kv_heads=cfg.downsampler_n_kv_heads[0],
            n_layers=cfg.downsampler_n_layers[0], mlp_mult=cfg.mlp_mult[0], rope_base=cfg.rope_base[0],
            output_expansion=1, context_window_groups=cfg.downsampler_window[0],
            output_vocab=code_vocab, output_chunks=pq_chunks, pq_dim=pq_dim,
            token_dim=cfg.token_dim[0], token_n_heads=cfg.token_n_heads[0],
            decode_past=cfg.downsampler_decode_past[0], decode_future=cfg.downsampler_decode_future[0],
            n_rates=n, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, remat=downsampler_remat)

        upsampler_decode_past = cfg.decode_past[0] if cfg.upsampler_decode_past[0] is None else cfg.upsampler_decode_past[0]
        upsampler_decode_future = cfg.decode_future[0] if cfg.upsampler_decode_future[0] is None else cfg.upsampler_decode_future[0]
        upsampler_remat = cfg.remat if cfg.upsampler_remat[0] is None else cfg.upsampler_remat[0]
        K0 = cfg.strides[0] if cfg.strides[0] != -1 else 1  # output_expansion default -- overridden
        # per-call via decode_logits_and_target_multipass/decode_generate_multipass's model.K(level_idx)
        self.upsampler = PardecLM(
            jax.random.fold_in(keys[3], 9201), context_hidden_dim=D_enc, hidden_dim=cfg.upsampler_d_model[0],
            n_heads=cfg.upsampler_n_heads[0], n_kv_heads=cfg.upsampler_n_kv_heads[0],
            n_layers=cfg.upsampler_n_layers[0], mlp_mult=cfg.mlp_mult[0], rope_base=cfg.rope_base[0],
            output_expansion=K0, context_window_groups=cfg.upsampler_window[0],
            output_vocab=code_vocab, output_chunks=pq_chunks, pq_dim=pq_dim,
            token_dim=cfg.token_dim[0], token_n_heads=cfg.token_n_heads[0],
            decode_past=upsampler_decode_past, decode_future=upsampler_decode_future,
            n_rates=n, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, remat=upsampler_remat,
            shared_blocks=self.downsampler.blocks if cfg.share_downsampler_upsampler_lm else None,
            shared_ln_f=self.downsampler.ln_f if cfg.share_downsampler_upsampler_lm else None)

    def K(self, level: int) -> int:
        return self.cfg.strides[level] if self.cfg.strides[level] != -1 else 1


def dec_loss_acc(logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray = None) -> tuple:
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    correct = (jnp.argmax(logits, -1) == target).astype(jnp.float32)
    if mask is None:
        return jnp.mean(nll), jnp.mean(correct)
    m = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(m), 1.0)
    return jnp.sum(nll * m) / denom, jnp.sum(correct * m) / denom


def level_forward(model: LagCodecModel, flat_bytes: jnp.ndarray, phase: int, rng=None,
                   level_gt_drop=None, cascade_rng=None, encode_temperature: float = 1.0,
                   layer_drop_prob=None, label_reg_weight: float = 0.0, label_fn=None,
                   pixel_order=None, byte_pq_fn=None) -> tuple:
    codelm = model.codelm
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    # flat_bytes stays raw (fed to label_fn as-is, which interprets literal byte/pixel values);
    # tok0 is the SAME raw bytes converted into CodeLM's own (pq_chunks, code_vocab) categorical
    # representation via byte_pq_fn -- user-settable, defaults to rgb_byte_pq_fn. Level 0's own
    # input/NTP-target and levels>0's own input/NTP-target now share this exact representation,
    # which is what lets a single shared CodeLM process every level.
    tok0 = byte_pq_fn(flat_bytes, codelm.pq_chunks, codelm.code_vocab)
    x = code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
    target = tok0
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses = [], [], [], [], []
    label_mses, label_mse_losses = [], []
    level_rngs = [None] * (2 * phase) if rng is None else list(jax.random.split(rng, 2 * phase))
    for i in range(phase):
        if model.cfg.use_pardec_downsampler:
            out = encode_pardec_downsampler(codelm, model.downsampler, x, target, flat_bytes, model.cfg,
                                             pixel_order, label_fn, model.K(i), rate_id=i, rng=level_rngs[2 * i])
        else:
            out = codelm.encode(x, target, model.K(i), rng=level_rngs[2 * i],
                                 encode_temperature=encode_temperature, layer_drop_prob=layer_drop_prob, rate_id=i)
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
            # hard metric (argmax, like decoder's byte_mse) -- always computed, logging only
            pred_label_hard = jnp.argmax(enc_logits, axis=-1).astype(jnp.float32)
            label_mses.append(jnp.mean((pred_label_hard - label_tgt.astype(jnp.float32)) ** 2))
            # soft/differentiable version (softmax-expected value, like decoder's mse_loss) --
            # backprops only if label_mse_weight>0, kept computed unconditionally so flipping the
            # weight on later needs no code change
            label_probs_i = jax.nn.softmax(enc_logits, axis=-1)
            label_values_i = jnp.arange(label_probs_i.shape[-1], dtype=label_probs_i.dtype)
            pred_label_soft = jnp.sum(label_probs_i * label_values_i, axis=-1)
            label_mse_losses.append(jnp.mean((pred_label_soft - label_tgt.astype(jnp.float32)) ** 2))
        if i < phase - 1:
            x = code_embed_proj(out["code_soft"], codelm.own_input_embed, codelm.own_input_proj)
            target = out["code_idx"]

    dec_losses, dec_accs = [], []
    aux_ntp_losses, aux_ntp_accs = [], []
    aux_applies = model.upsampler.decode_future > 0 or model.upsampler.decode_past > 0

    byte_mse = None
    mse_loss = 0.0
    ctx = codes_soft[phase - 1]
    cascade_rngs = [None] * phase if cascade_rng is None else list(jax.random.split(cascade_rng, phase))

    start_i = phase - 1

    for i in range(start_i, -1, -1):
        dec_target = tok0 if i == 0 else codes[i - 1]
        dec_rng = level_rngs[2 * i + 1]
        logits, target_i, mask_i, aux_loss_i, aux_acc_i = decode_logits_and_target_multipass(
            model, i, dec_target, ctx, model.cfg.decoder_ncodes[i], rng=dec_rng)
        loss_i, acc_i = dec_loss_acc(logits, target_i, mask_i)
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if aux_applies:
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
    label_mse_total = jnp.mean(jnp.stack(label_mses)) if label_mses else jnp.array(0.0)
    label_mse_loss_total = jnp.mean(jnp.stack(label_mse_losses)) if label_mse_losses else 0.0
    if aux_ntp_losses:
        aux_ntp_loss_total = jnp.mean(jnp.stack(aux_ntp_losses))
        aux_ntp_acc_total = jnp.mean(jnp.stack(aux_ntp_accs))
    else:
        aux_ntp_loss_total = jnp.array(0.0)
        aux_ntp_acc_total = jnp.array(0.0)
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + model.cfg.mse_weight * mse_loss + label_reg_weight * label_loss_total \
        + model.cfg.ntp_weight * aux_ntp_loss_total + model.cfg.label_mse_weight * label_mse_loss_total
    bpb = dec_loss_total / jnp.log(2.0)
    return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)), byte_mse, aux_ntp_loss_total / jnp.log(2.0), aux_ntp_acc_total,
                  label_mse_total)


def sample_multires_entry(py_rng, n_levels: int) -> tuple:
    # Python-level (not jax.random) sampling: entry_level/depth must be static Python ints, chosen
    # BEFORE jax.jit traces the training step -- same constraint `phase` already has (it's a plain
    # int, not a traced array, so distinct values simply retrace/cache separately). One unified
    # sampler covers all three cases from the original design discussion: entry_level=0,
    # depth=n_levels-1 (full 32->16->8), entry_level=0, depth<n_levels-1 (32->16 early-stop),
    # entry_level>0 (16->8 skip-first, treating a resized-down real image as if it were level
    # entry_level's own native input).
    entry_level = py_rng.randrange(0, n_levels - 1)  # leaves room for >=1 recursive step
    depth = py_rng.randrange(1, n_levels - entry_level)
    return entry_level, depth


def level_forward_multires(model: LagCodecModel, flat_bytes: jnp.ndarray, entry_level: int, depth: int,
                            rng=None, encode_temperature: float = 1.0, label_reg_weight: float = 0.0,
                            label_fn=None, pixel_order=None, byte_pq_fn=None) -> tuple:
    # Same idea as level_forward, but the encode cascade starts at entry_level (not always 0) and
    # runs only `depth` further steps. entry_level=0 is equivalent to level_forward(..., phase=depth)
    # in spirit (though the aux tuple shape differs slightly, see below). entry_level>0's input is
    # NOT produced by running the model's own (real) encoder up to that depth -- it's synthesized
    # directly from the real image via the same resize+quantize label_fn already uses for aux
    # targets, standing in for "a native input at this level's resolution". Every level reuses the
    # SAME shared CodeLM/downsampler/upsampler (see LagCodecModel) -- this is what actually trains
    # those shared parameters across every resolution they need to work at, not just the one fixed
    # depth level_forward always uses.
    codelm = model.codelm
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    if entry_level == 0:
        entry_code = byte_pq_fn(flat_bytes, codelm.pq_chunks, codelm.code_vocab)
    else:
        n_blocks_entry = n_blocks_for_level(model.cfg, entry_level - 1)
        entry_code = label_fn(flat_bytes, model.cfg, pixel_order, n_blocks_entry,
                               model.cfg.pq_chunks[entry_level - 1], model.cfg.code_vocab[entry_level - 1])
    x = code_embed_proj(entry_code, codelm.own_input_embed, codelm.own_input_proj)
    target = entry_code
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses, label_mses = [], [], [], [], [], []
    level_rngs = [None] * (2 * depth) if rng is None else list(jax.random.split(rng, 2 * depth))
    for d in range(depth):
        i = entry_level + d
        out = codelm.encode(x, target, model.K(i), rng=level_rngs[2 * d],
                             encode_temperature=encode_temperature, rate_id=i)
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
            pred_label_hard = jnp.argmax(enc_logits, axis=-1).astype(jnp.float32)
            label_mses.append(jnp.mean((pred_label_hard - label_tgt.astype(jnp.float32)) ** 2))
        if d < depth - 1:
            x = code_embed_proj(out["code_soft"], codelm.own_input_embed, codelm.own_input_proj)
            target = out["code_idx"]

    dec_losses, dec_accs = [], []
    ctx = codes_soft[depth - 1]
    dec_rngs = [None] * depth if rng is None else list(jax.random.split(jax.random.fold_in(rng, 777), depth))
    for d in range(depth - 1, -1, -1):
        i = entry_level + d
        dec_target = entry_code if d == 0 else codes[d - 1]
        logits, target_i, mask_i, aux_loss_i, aux_acc_i = decode_logits_and_target_multipass(
            model, i, dec_target, ctx, model.cfg.decoder_ncodes[i], rng=dec_rngs[d])
        loss_i, acc_i = dec_loss_acc(logits, target_i, mask_i)
        dec_losses.append(loss_i)
        dec_accs.append(acc_i)
        if d > 0:
            ctx = codes_soft[d - 1]

    dec_loss_total = jnp.mean(jnp.stack(dec_losses))
    byte_acc = dec_accs[-1]  # accuracy of the deepest->entry_level+1 decode step, matches
    # level_forward's own-code-accuracy convention (last-computed = shallowest/own-level step)
    ntp_loss_total = jnp.mean(jnp.stack(enc_losses))
    entropy_loss_total = jnp.mean(jnp.stack(entropy_losses))
    label_loss_total = jnp.mean(jnp.stack(label_losses)) if label_losses else 0.0
    label_mse_total = jnp.mean(jnp.stack(label_mses)) if label_mses else jnp.array(0.0)
    loss = dec_loss_total + model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + label_reg_weight * label_loss_total
    bpb = dec_loss_total / jnp.log(2.0)
    zero = jnp.array(0.0)
    # aux tuple shape matches level_forward's (bpb, byte_acc, ntp_bpb, e_acc, util, byte_mse,
    # aux_ntp_bpb, aux_ntp_acc, label_mse) so run_val_eval/the train logger work unchanged --
    # byte_mse/aux_ntp_* don't apply here (no raw-pixel reconstruction when entry_level>0), zeroed.
    return loss, (bpb, byte_acc, ntp_loss_total / jnp.log(2.0), jnp.mean(jnp.stack(enc_accs)),
                  jnp.mean(jnp.stack(utils)), zero, zero, zero, label_mse_total)


def phase_trainable_filter(model: LagCodecModel, phase: int):
    # SINGLETON model: codelm/downsampler/upsampler are the SAME shared weights used by every
    # level, so there is no per-level subset to selectively freeze -- curriculum_mode is asserted
    # 'no_freeze' unconditionally (see __post_init__), and with one shared parameter set, "no_freeze"
    # is the only sensible behavior anyway: the whole model is trainable regardless of phase.
    return jax.tree_util.tree_map(lambda x: eqx.is_array(x), model)


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
                     phase: int, phase_step: int, step: int, seed: int, schedule_meta: dict = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    eqx.tree_serialise_leaves(ckpt_dir / "p_rng.eqx", p_rng)
    (ckpt_dir / "dataloader_state.json").write_text(json.dumps(dict(
        epoch_rng_state=train_iter.epoch_rng.bit_generator.state,
        epoch_seed=train_iter.epoch_seed, pos=train_iter.pos)))
    meta = dict(phase=phase, phase_step=phase_step, step=step, seed=seed)
    if schedule_meta is not None:
        meta["schedule"] = schedule_meta
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))


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


def plot_encoder_outs(model: "LagCodecModel", cfg: Config, imgs: np.ndarray, pixel_order: np.ndarray,
                       path: Path, level: int = 0, label_fn=None):
    # Encodes real images with `level`'s own encoder and plots GT | target downsample (from
    # label_fn, the same target label_reg_weight/label_mse supervise against) | pred downsample
    # (code_idx read directly as RGB bytes, no upsampling). Only meaningful when pq_chunks[level]==3
    # and code_vocab[level]==256 (maps 1:1 onto RGB) -- silently skipped otherwise. Spatial-coherence
    # sanity check on the code space (does it look like a structured downsample of the real image,
    # or noise), not a generation audit.
    if cfg.pq_chunks[level] != 3 or cfg.code_vocab[level] != 256:
        return None
    codelm = model.codelm
    flat_raw = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    flat = rgb_byte_pq_fn(flat_raw, codelm.pq_chunks, codelm.code_vocab) if level == 0 else flat_raw
    x = code_embed_proj(flat, codelm.own_input_embed, codelm.own_input_proj)
    out = codelm.encode(x, flat, model.K(level), rng=None, rate_id=level)
    code_idx = np.asarray(out["code_idx"])
    util = float(out["util"])
    M, n_blocks, C = code_idx.shape
    side = round(n_blocks ** 0.5)
    if side * side != n_blocks:
        return util
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)

    def to_grid(idx: np.ndarray) -> np.ndarray:
        raster = np.zeros((M, side * side, C), dtype=np.uint8)
        raster[:, low_order, :] = idx.astype(np.uint8)
        return raster.reshape(M, side, side, C)

    pred_grid = to_grid(code_idx)
    gt = imgs.astype(np.uint8)
    panels, titles = [gt], ["ground truth"]
    label_mse = None
    if label_fn is not None:
        label_tgt = np.asarray(label_fn(flat, cfg, pixel_order, n_blocks, cfg.pq_chunks[level], cfg.code_vocab[level]))
        panels.append(to_grid(label_tgt))
        titles.append("target downsample")
        label_mse = float(np.mean((code_idx.astype(np.float64) - label_tgt.astype(np.float64)) ** 2))
    panels.append(pred_grid)
    titles.append("pred downsample")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = len(panels)
    fig, axes = plt.subplots(M, ncols, figsize=(2 * ncols, 2 * M))
    axes = axes.reshape(M, ncols)
    for i in range(M):
        for j, (panel, title) in enumerate(zip(panels, titles)):
            axes[i, j].imshow(panel[i])
            axes[i, j].set_title(title if i == 0 else "", fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    suptitle = f"level{level} encoder outs -- util={util:.3f}"
    if label_mse is not None:
        suptitle += f"  label_mse={label_mse:.2f}"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return util


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


def _opt_bool_tuple_arg(s: str) -> tuple:
    return tuple(None if x.strip().lower() == "none" else x.strip().lower() != "false" for x in s.split(","))


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


CONFIG_FIELDS = ("img_size", "codelm_d_model", "codelm_n_layers", "codelm_n_heads", "codelm_n_kv_heads",
                  "downsampler_d_model", "downsampler_n_layers", "downsampler_n_heads",
                  "downsampler_n_kv_heads", "downsampler_window", "use_pardec_downsampler",
                  "downsampler_decode_past", "downsampler_decode_future", "downsampler_remat",
                  "upsampler_d_model", "upsampler_n_layers", "upsampler_n_heads",
                  "upsampler_n_kv_heads", "upsampler_window", "use_pardec_upsampler",
                  "upsampler_decode_past", "upsampler_decode_future", "upsampler_remat",
                  "share_downsampler_upsampler_lm", "strides",
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight", "decoder_ncodes",
                  "ncodes_window", "stream_chunks", "stream_lag", "decode_past", "decode_future", "sync",
                  "gen_temperature", "gen_top_k", "dense_decode",
                  "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_at_inference", "init_scheme", "use_xsa",
                  "use_qknorm", "remat", "remat_level", "attn_window", "attn_lookahead",
                  "encoder_attn_window", "decoder_attn_window", "use_sink",
                  "use_codelm_bos", "codelm_bos_prob", "codelm_bos_rates",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal", "label_reg_weight", "label_mse_weight")


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
    p.add_argument("--codelm_d_model", type=_tuple_arg, default=Config.codelm_d_model)
    p.add_argument("--codelm_n_layers", type=_tuple_arg, default=Config.codelm_n_layers)
    p.add_argument("--codelm_n_heads", type=_tuple_arg, default=Config.codelm_n_heads)
    p.add_argument("--codelm_n_kv_heads", type=_tuple_arg, default=Config.codelm_n_kv_heads)
    p.add_argument("--downsampler_d_model", type=_tuple_arg, default=Config.downsampler_d_model)
    p.add_argument("--downsampler_n_layers", type=_tuple_arg, default=Config.downsampler_n_layers)
    p.add_argument("--downsampler_n_heads", type=_tuple_arg, default=Config.downsampler_n_heads)
    p.add_argument("--downsampler_n_kv_heads", type=_tuple_arg, default=Config.downsampler_n_kv_heads)
    p.add_argument("--downsampler_window", type=_tuple_arg, default=Config.downsampler_window,
                    help="downsampler PardecLM's context_window_groups, in GROUPS -- must stay "
                         "bounded (not -1) at output_group_size=1, unbounded OOMs")
    p.add_argument("--use_pardec_downsampler", type=lambda x: x.lower() != "false",
                    default=Config.use_pardec_downsampler,
                    help="True = encode_pardec_downsampler (teacher-forced against label_fn) "
                         "instead of the naive-pick/code_head path in encode(). Requires "
                         "label_reg_weight>0 -- that becomes the downsampler's only loss")
    p.add_argument("--downsampler_decode_past", type=_tuple_arg, default=Config.downsampler_decode_past,
                    help="downsampler PardecLM's OWN decode_past (independent of decode_past/"
                         "upsampler_decode_past). Default 0 (previously always 0, unconfigurable)")
    p.add_argument("--downsampler_decode_future", type=_tuple_arg, default=Config.downsampler_decode_future,
                    help="downsampler PardecLM's OWN decode_future, independent of the upsampler's. "
                         "Default 0 (previously always 0, unconfigurable)")
    p.add_argument("--downsampler_remat", type=_opt_bool_tuple_arg, default=Config.downsampler_remat,
                    help="downsampler PardecLM's OWN remat, independent of the upsampler's. "
                         "'none' (default) falls back to --remat (previous shared behavior)")
    p.add_argument("--upsampler_d_model", type=_tuple_arg, default=Config.upsampler_d_model)
    p.add_argument("--upsampler_n_layers", type=_tuple_arg, default=Config.upsampler_n_layers)
    p.add_argument("--upsampler_n_heads", type=_tuple_arg, default=Config.upsampler_n_heads)
    p.add_argument("--upsampler_n_kv_heads", type=_tuple_arg, default=Config.upsampler_n_kv_heads)
    p.add_argument("--upsampler_window", type=_tuple_arg, default=Config.upsampler_window,
                    help="upsampler PardecLM's context_window_groups, in GROUPS of decoder_ncodes "
                         "codes -- must stay bounded (not -1), same OOM lesson as downsampler_window")
    p.add_argument("--use_pardec_upsampler", type=lambda x: x.lower() != "false",
                    default=Config.use_pardec_upsampler,
                    help="True = PardecLM-based decode (pardec_score/pardec_generate, shared with "
                         "the downsampler's machinery, output_expansion=K) instead of the old "
                         "hand-rolled dec_blocks/ctx_embed pardec decode path. decode_past is "
                         "scoring-only in this path (no dense_decode/stream_chunks equivalent yet)")
    p.add_argument("--upsampler_decode_past", type=_tuple_arg, default=Config.upsampler_decode_past,
                    help="upsampler PardecLM's OWN decode_past. 'none' (default) falls back to "
                         "--decode_past (the legacy/shared field, still used by the old dec_blocks path)")
    p.add_argument("--upsampler_decode_future", type=_tuple_arg, default=Config.upsampler_decode_future,
                    help="upsampler PardecLM's OWN decode_future. 'none' (default) falls back to "
                         "--decode_future (the legacy/shared field, still used by the old dec_blocks path)")
    p.add_argument("--upsampler_remat", type=_opt_bool_tuple_arg, default=Config.upsampler_remat,
                    help="upsampler PardecLM's OWN remat, independent of the downsampler's. "
                         "'none' (default) falls back to --remat (previous shared behavior)")
    p.add_argument("--share_downsampler_upsampler_lm", type=lambda x: x.lower() != "false",
                    default=Config.share_downsampler_upsampler_lm,
                    help="False (default): fully separate downsampler/upsampler PardecLM weights. "
                         "True: they share the same blocks/ln_f (the core transformer LM) -- "
                         "context_proj/bos_embed/target_embed/token_* AR head/output_head_linear "
                         "stay independent. Needs downsampler_d_model/n_layers/n_heads/n_kv_heads "
                         "== the matching upsampler_* values")
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
                         "image). ncodes_window bounds the history before the chunk (-1 = all). Mutually exclusive "
                         "(XOR) with --stream_lag -- set only one.")
    p.add_argument("--stream_lag", type=_tuple_arg, default=Config.stream_lag,
                    help="alternate way to set stream_chunks: per level, the desired own-group wait cadence L>=1 "
                         "directly (1 = eager/per-group streaming, n_groups = fully offline), resolved once "
                         "n_groups is known to stream_chunks=ceil(n_groups/L). Unlike stream_chunks itself (whose "
                         "effective lag depends on n_groups, which varies per level/decoder_ncodes), this gives a "
                         "level-independent lag. Mutually exclusive (XOR) with --stream_chunks -- set only one "
                         "(default 0 = unset for both, meaning stream_chunks applies as given).")
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
    p.add_argument("--gen_temperature", type=float, default=Config.gen_temperature,
                    help="temperature of the sampled (non-argmax) generation eval / decoder sampling")
    p.add_argument("--gen_top_k", type=int, default=Config.gen_top_k,
                    help="top-k of decoder sampling (0 = off); only used when sampling, never for argmax")
    p.add_argument("--dense_decode", type=_bool_tuple_arg, default=Config.dense_decode,
                    help="regress this level to the original, fully-interleaved [code,BOS,K bytes,code,BOS,...] "
                         "flat causal decoder (decode_logits_and_target / decode_generate): no windowing, no "
                         "groups/batching approximation, no decode_past/stream_chunks "
                         "(all ignored). Training uses splash (full causal, O(T) memory); generation "
                         "uses a single real growing KV cache (lax.scan over own-codes), no padding/magic numbers "
                         "-- every step genuinely sees the whole real prefix. O(T^2) total compute either way, "
                         "same as any correct full-attention causal LM; decoder_attn_window bounds it if desired.")
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
    p.add_argument("--init_scheme", type=str, default=Config.init_scheme, choices=["llama", "zero"])
    p.add_argument("--use_xsa", type=lambda x: x.lower() != "false", default=Config.use_xsa)
    p.add_argument("--use_qknorm", type=lambda x: x.lower() != "false", default=Config.use_qknorm)
    p.add_argument("--remat", type=lambda x: x.lower() != "false", default=Config.remat)
    p.add_argument("--remat_level", type=lambda x: x.lower() != "false", default=Config.remat_level,
                    help="checkpoint each level's whole encoder / decoder block stack as one unit (recompute at the "
                         "level border) instead of per transformer block; less recompute, more live memory. "
                         "Takes precedence over --remat inside the stacks")
    p.add_argument("--attn_window", type=_tuple_arg, default=Config.attn_window,
                    help="symmetric base causal window (-1=unbounded), used by BOTH encoder and decoder "
                         "self-attention unless overridden per-side below.")
    p.add_argument("--encoder_attn_window", type=_tuple_arg, default=Config.encoder_attn_window,
                    help="per-level override of the encoder's own attn_window (None entries fall back to "
                         "--attn_window).")
    p.add_argument("--decoder_attn_window", type=_tuple_arg, default=Config.decoder_attn_window,
                    help="dead: only fed the OLD dec_blocks decode path, removed when the singleton "
                         "PardecLM upsampler became the only decode path. Kept declared, inert.")
    p.add_argument("--attn_lookahead", type=_tuple_arg, default=Config.attn_lookahead,
                    help="encoder self-attention shifted-triangular lookahead -- 0 (default) "
                         "plain causal, int>0 query may additionally see keys up to that many "
                         "positions ahead (splash LocalMask's native right-side window)")
    p.add_argument("--use_sink", type=lambda x: x.lower() != "false", default=Config.use_sink)
    p.add_argument("--use_codelm_bos", type=lambda x: x.lower() != "false", default=Config.use_codelm_bos,
                    help="False (default): no BOS/anchor for CodeLM's own free-run, position 0 stays "
                         "dataset-biased. True: CodeLM gets its own learned bos_embed table (one row "
                         "per --codelm_bos_rates, e.g. one anchor per target scale), substituted at "
                         "position 0 with probability --codelm_bos_prob during training. The bos_embed "
                         "param always exists (harmless/inert when this is False) so toggling this "
                         "flag alone never breaks checkpoint structure")
    p.add_argument("--codelm_bos_prob", type=float, default=Config.codelm_bos_prob,
                    help="probability an example's position 0 is substituted with bos_embed[rate_id] "
                         "during training (only when --use_codelm_bos=True)")
    p.add_argument("--codelm_bos_rates", type=_tuple_arg, default=Config.codelm_bos_rates,
                    help="per-level n_rates for the bos_embed table (1 default = single generic "
                         "anchor). No effect when --use_codelm_bos=False")
    p.add_argument("--byte_group", type=int, default=Config.byte_group)
    p.add_argument("--token_head_type", type=str, default=Config.token_head_type)
    p.add_argument("--token_dim", type=_tuple_arg, default=Config.token_dim)
    p.add_argument("--token_n_heads", type=_tuple_arg, default=Config.token_n_heads)
    p.add_argument("--pq_dim", type=_tuple_arg, default=Config.pq_dim)
    p.add_argument("--token_mask_prob", type=float, default=Config.token_mask_prob)
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
    p.add_argument("--label_mse_weight", type=float, default=Config.label_mse_weight,
                    help="soft/differentiable MSE between the label-regularized code_head's "
                         "softmax-expected value and the label target, added to the loss (mirrors "
                         "mse_weight/mse_loss for the decoder side). 0.0 (off) means label_mse is "
                         "still computed and logged (hard, argmax-based) whenever label_reg_weight>0, "
                         "just not backpropped")
    p.add_argument("--traversal", type=str, default=Config.traversal, choices=["raster", "zorder"])
    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    if "streaming" in config_vars:
        p.error(f"--config {pre_args.config}: 'streaming' was replaced by 'stream_chunks' "
                f"(0 = per-group streaming, 1 = wait once / old streaming=False, n = n chunks)")
    label_fn_registry = {"default_label_fn_jax": default_label_fn_jax, "rgb_label_fn_jax": rgb_label_fn_jax,
                          "default_label_fn_pil": default_label_fn_pil}
    label_fn_raw = config_vars.pop("label_fn", "default_label_fn_jax")
    label_fn = label_fn_registry[label_fn_raw] if isinstance(label_fn_raw, str) else label_fn_raw
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

    (train_np, train_labels), (val_np, val_labels) = load_dataset(args.dataset, Path(args.data_root), cfg.img_size)
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    if args.val_subset_n:
        val_np = val_np[:args.val_subset_n]

    rng = jax.random.PRNGKey(args.seed)
    model = LagCodecModel(rng, cfg)
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
        codelm = m.codelm
        tok0 = rgb_byte_pq_fn(flat_prompt, codelm.pq_chunks, codelm.code_vocab)
        x = code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
        target = tok0
        codes, codes_soft = [], []
        eval_rngs = ([None] * (top + 1) if not cfg.gumbel_at_inference
                     else list(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), hash(tag) % (2**31)), top + 1)))
        for i in range(top + 1):
            out = codelm.encode(x, target, m.K(i), rng=eval_rngs[i],
                                 encode_temperature=args.encode_temperature[phase - 1], rate_id=i)
            codes.append(out["code_idx"])
            codes_soft.append(out["code_soft"])
            if i < top:
                x = code_embed_proj(out["code_soft"], codelm.own_input_embed, codelm.own_input_proj)
                target = out["code_idx"]

        recon_acc = recon_mse = None

        cascade_t0 = time.monotonic()
        cur_code = codes[top]
        loop_start = top
        for i in range(loop_start, 0, -1):
            cur_code = decode_generate_multipass(m, i, cur_code, cfg.decoder_ncodes[i], **g_kw)
        cascade_recon = decode_generate_multipass(m, 0, cur_code, cfg.decoder_ncodes[0], **g_kw)
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
        sums = np.zeros(9, dtype=np.float64)
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
        _bpb, acc, _ntp_bpb, ntp_acc, util, val_mse, _aux_ntp_bpb, aux_ntp_acc, val_label_mse = \
            (sums / total_n).tolist()
        loss = total_loss / total_n
        val_time_s = time.monotonic() - val_t0
        msg = (f"[{tag}] VAL loss={loss:.2f} val_dec_acc={acc:.2f} val_mse={val_mse:.4f} "
               f"val_e_ntp_acc={ntp_acc:.2f} val_d_ntp_acc={aux_ntp_acc:.2f} "
               f"val_label_mse={val_label_mse:.2f} val_time={val_time_s:.1f}s")
        rec = dict(tag=tag, val_loss=loss, val_dec_acc=acc,
                    val_e_ntp_acc=ntp_acc, val_util=util, val_mse=val_mse,
                    val_d_ntp_acc=aux_ntp_acc, val_label_mse=val_label_mse,
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
        def loss_fn(diff_model, static_model, flat_bytes, rng, cascade_rng, phase=phase):
            m = eqx.combine(diff_model, static_model)
            m = cast_pytree(m, compute_dtype)
            return level_forward(m, flat_bytes, phase, rng=rng,
                                     level_gt_drop=level_gt_drop_phase, cascade_rng=cascade_rng,
                                     encode_temperature=encode_temperature_phase,
                                     layer_drop_prob=layer_drop_prob_phase,
                                     label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                     pixel_order=pixel_order)

        steps_per_epoch_lr = len(train_iter)
        phase_total_steps = _phase_total_steps(phase - 1, steps_per_epoch_lr)
        total_steps = phase_total_steps
        warmup_steps_resolved = _every_steps(args.warmup_steps, args.warmup_epochs, steps_per_epoch_lr)
        min_step = (args.lr_min_step if args.lr_min_step is not None
                    else round(args.lr_min_epoch * steps_per_epoch_lr) if args.lr_min_epoch is not None
                    else phase_total_steps)
        lr_decay_steps = max(1, min_step - warmup_steps_resolved)
        lr_kind, lr_peak, lr_min_val = args.lr_schedule, args.lr, args.lr_min
        resuming_this_phase = resume_meta is not None and phase == resume_meta["phase"]
        if resuming_this_phase and resume_meta.get("schedule") is not None:
            sm = resume_meta["schedule"]
            if (sm["total_steps"], sm["warmup_steps"], sm["lr_decay_steps"], sm["kind"], sm["lr"], sm["lr_min"]) \
                    != (total_steps, warmup_steps_resolved, lr_decay_steps, lr_kind, lr_peak, lr_min_val):
                logger(f"resume: current args would give a DIFFERENT lr schedule for phase {phase} than the "
                       f"checkpoint's -- freezing to the checkpoint's own schedule (total_steps={sm['total_steps']}, "
                       f"warmup={sm['warmup_steps']}, decay_steps={sm['lr_decay_steps']}) for a kink-free "
                       f"continuation. Start a fresh run (no --resume) to deliberately change the schedule.")
            total_steps = phase_total_steps = sm["total_steps"]
            warmup_steps_resolved = sm["warmup_steps"]
            lr_decay_steps = sm["lr_decay_steps"]
            lr_kind, lr_peak, lr_min_val = sm["kind"], sm["lr"], sm["lr_min"]
        schedule_meta = dict(total_steps=total_steps, warmup_steps=warmup_steps_resolved,
                              lr_decay_steps=lr_decay_steps, kind=lr_kind, lr=lr_peak, lr_min=lr_min_val)
        lr_schedule = make_lr_schedule(lr_kind, lr_peak, warmup_steps_resolved, total_steps,
                                        end_value=lr_min_val, decay_steps=lr_decay_steps)
        if args.optimizer == "sinkgd":
            optimizer = sinkgd(lr_schedule, **args.optimizer_kwargs)
        else:
            optimizer = optax.adamw(lr_schedule, **args.optimizer_kwargs)
        if args.grad_clip is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip), optimizer)
        opt_state = optimizer.init(diff_model)

        def train_step(diff_model, opt_state, rng, flat_bytes, static_model=static_model):
            rng, level_rng, cascade_rng = jax.random.split(rng, 3)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                diff_model, static_model, flat_bytes, level_rng, cascade_rng)
            grads = jax.lax.pmean(grads, axis_name="d")
            loss = jax.lax.pmean(loss, axis_name="d")
            aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
            # no gradient tying needed: model.codelm/downsampler/upsampler are singleton modules --
            # level_forward calling them at several different level indices within this one trace
            # already gets correctly-summed gradients from JAX's autodiff, same as any other value
            # used multiple times in one function.
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
            dl_state = json.loads((resume_ckpt_dir / "dataloader_state.json").read_text())
            train_iter.epoch_rng.bit_generator.state = dl_state["epoch_rng_state"]
            train_iter.epoch_seed = dl_state["epoch_seed"]
            train_iter.pos = dl_state["pos"]
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
                loss0 = float(local_array(loss)[0])
                if not jit_timed:
                    logger(f"{active_desc}: first train_step (incl. jit compile) took "
                           f"{time.monotonic() - jit_t0:.1f}s")
                    jit_timed = True
                _bpb, acc, _ntp_bpb, ntp_acc, util, train_mse, _aux_ntp_bpb, aux_ntp_acc, label_mse, grad_norm = \
                    [float(local_array(a)[0]) for a in aux]
                lr = float(lr_schedule(step - 1))
                lr_str = _fmt_lr(lr)
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}",
                                  acc=f"{acc:.2f}",
                                  lr=lr_str, gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    logger(f"l={phase - 1} e={epoch_num} s={step} loss={loss0:.2f} dec_acc={acc:.2f} "
                           f"e_ntp_acc={ntp_acc:.2f} util={util:.2f} mse={train_mse:.1f} "
                           f"d_ntp_acc={aux_ntp_acc:.2f} label_mse={label_mse:.2f} "
                           f"lr={lr_str} grad_norm={grad_norm:.2f}",
                           level=phase - 1, epoch=epoch_num, step=step, loss=loss0,
                           dec_acc=acc, e_ntp_acc=ntp_acc, util=util,
                           mse=train_mse, d_ntp_acc=aux_ntp_acc, label_mse=label_mse,
                           lr=lr, grad_norm=grad_norm)

                if step % gen_eval_every_steps == 0:
                    snapshot = eqx.combine(to_single_device(unreplicate(p_diff_model)), static_model)
                    run_val_eval(snapshot, phase, tag=f"level{phase - 1}_step{step}")
                    run_gen_eval_both(snapshot, top=phase - 1, tag=f"level{phase - 1}_step{step}")
                    plot_encoder_outs(snapshot, cfg, val_np[:args.val_batch_size[phase - 1]], pixel_order,
                                       run_dir / f"samples_level{phase - 1}_step{step}_codegrid.png",
                                       level=phase - 1, label_fn=label_fn)

                if step % ckpt_every_steps == 0:
                    ckpt_model = eqx.combine(to_host(unreplicate(p_diff_model)), static_model)
                    ckpt_opt_state = to_host(unreplicate(p_opt_state))
                    ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
                    save_checkpoint(ckpt_dir, ckpt_model, ckpt_opt_state, to_host(p_rng), train_iter,
                                     phase=phase, phase_step=phase_step, step=step, seed=args.seed,
                                     schedule_meta=schedule_meta)
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
                         phase=phase, phase_step=phase_total_steps, step=step, seed=args.seed,
                         schedule_meta=schedule_meta)
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
