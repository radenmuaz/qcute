"""
run_lagcodec_res_denoise_torch: single-file PyTorch port of run_lagcodec_res_denoise.py (2026-10-03).
Same Config fields, flags and configs; same math (JAX weights load 1:1, checked by
scripts/res_denoise_torch_parity_check.py). One process, one device: --device auto|cpu|cuda|mps.
RNG uses JAX-style explicit keys (Key.fold/split) so code paths mirror the JAX file; random streams differ.
uv run python3 -m image_lagcodec.run_lagcodec_res_denoise_torch --config image_lagcodec/configs/<name>.py --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import shutil
import sys
import tarfile
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent


class Key:
    # JAX-style explicit PRNG key: sub-keys by fold/split, each draw from a freshly seeded generator
    __slots__ = ("seed",)

    def __init__(self, seed: int):
        self.seed = int(seed) % (2 ** 63)

    def fold(self, data: int) -> "Key":
        h = hashlib.blake2b(f"{self.seed}/{int(data)}".encode(), digest_size=8).digest()
        return Key(int.from_bytes(h, "little"))

    def split(self, n: int) -> list:
        return [self.fold(-1 - i) for i in range(n)]


def _uniform(key: Key, shape, device, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    dev = torch.device(device)
    gdev = dev if dev.type == "cuda" else torch.device("cpu")
    g = torch.Generator(device=gdev)
    g.manual_seed(key.seed)
    return (torch.rand(tuple(shape), generator=g, device=gdev) * (hi - lo) + lo).to(dev)


def rand_gumbel(key: Key, shape, device) -> torch.Tensor:
    return -torch.log(-torch.log(_uniform(key, shape, device, 1e-8, 1.0 - 1e-8)))


def rand_bernoulli(key: Key, p: float, shape, device) -> torch.Tensor:
    return _uniform(key, shape, device) < p


RECURRENT_BACKBONES = ("gru", "linear_gru", "ssm")
BACKBONES = ("transformer",) + RECURRENT_BACKBONES


def total_bytes_of(cfg) -> int:
    return cfg.img_size * cfg.img_size * 3


def cycle_stack_slots(cfg, i: int) -> int:
    return cfg.level_cycles[i] - 1 if cfg.level_cycle_mode == "stack" else 0


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
    downsampler_window: tuple = 4  # context_window_groups, in GROUPS not raw positions -- must stay
    # bounded (not -1) given many small groups at output_group_size=1, confirmed via standalone
    # test (unbounded OOM'd even at tiny sizes)
    downsampler_decode_past: tuple = 0   # downsampler's OWN decode_past/decode_future/remat --
    downsampler_decode_future: tuple = 0  # independent of the upsampler's, own direct defaults
    downsampler_remat: tuple = None  # (0/0), no fallback chain -- None (default) for
    # downsampler_remat falls back to cfg.remat (remat granularity has no natural own-module default)
    downsampler_ncodes: tuple = 1  # batching granularity for the downsampler's OWN pardec_score/
    # pardec_generate calls: output_group_size=downsampler_ncodes, context_group_size=K*
    # downsampler_ncodes -- e.g. stride K=4, downsampler_ncodes=1 (default): 4 raw positions -> 1
    # code, computed one code at a time. downsampler_ncodes=4: 16 raw positions -> 4 codes, batched
    # into one group/call. Inverse of upsampler_ncodes in spirit (that one EXPANDS one context group
    # into upsampler_ncodes output codes; this one CONTRACTS downsampler_ncodes context groups into
    # downsampler_ncodes output codes scored/generated together) but same underlying mechanism.
    # Downsampler/upsampler always route through the PardecLM-based path (encode_pardec_downsampler/
    # decode_logits_and_target_multipass) -- the old naive code_head/dec_blocks alternatives were
    # removed 2026-09-27 (use_pardec_downsampler/use_pardec_upsampler flags deleted).
    upsampler_d_model: tuple = 1024  # PardecLM instance used by decode_logits_and_target_multipass/
    upsampler_n_layers: tuple = 4    # decode_generate_multipass -- own
    upsampler_n_heads: tuple = 8     # dedicated capacity (mirrors downsampler_*), NOT reused from
    upsampler_n_kv_heads: tuple = 8  # own dedicated capacity; context_hidden_dim = codelm_d_model
    # (CodeLM's own dim) -- same as downsampler's, no separate ctx dimension for the upsampler
    upsampler_window: tuple = 4  # context_window_groups, in GROUPS of upsampler_ncodes codes --
    # must stay bounded (not -1), same OOM lesson as downsampler_window. Each level's downsampler/
    # upsampler pair is fixed to that level's own stride K (output_expansion=K for upsampler,
    # context_group_size=K*downsampler_ncodes for downsampler), by design, not a freely-configurable
    # expansion factor. Genuine multi-rate training (e.g. both a stride=4 and a direct stride=16
    # path) would need SEVERAL dedicated downsampler/upsampler pairs per level, one per supported
    # stride, selected by whatever samples the entry_level/depth (see
    # level_forward_multires/sample_multires_entry, not yet wired into main()) -- not a single
    # flexible module. Deferred, noted for later.
    upsampler_decode_past: tuple = 0     # upsampler's OWN decode_past/decode_future -- own direct
    upsampler_decode_future: tuple = 0   # defaults, fully independent of downsampler's, no fallback
    # chain to any shared/generic field (that pattern was removed 2026-09-27: it only ever fed
    # this field's own default anyway, since the downsampler never consumed it).
    upsampler_remat: tuple = None  # None falls back to cfg.remat (remat granularity has no natural
    # own-module default, same reasoning as downsampler_remat)
    share_across_levels: bool = True  # True (default, current/original behavior) = exactly ONE
    # CodeLM/Downsampler/Upsampler instance for the whole model, called at every level -- "which
    # level" communicated only via rate_id into each module's own bos_embed table (see
    # LagCodecModel.codelm_for/bos_rate_id). This is the ONLY JAX-correct way to get true weight
    # tying: one leaf position in the pytree, reused n times inside a single traced loss -- storing
    # the SAME instance at multiple tuple positions instead would NOT tie weights under jax.grad
    # (each tuple position is an independent pytree leaf for autodiff, so gradients would diverge
    # after one optimizer step even though the initial values were equal). False = one fully
    # independent CodeLM/Downsampler/Upsampler PER LEVEL (own weights, own architecture -- may use
    # different codelm_d_model/n_layers/etc per level, singleton_uniform_fields assert skipped),
    # each with its own private bos_embed (n_rates=codelm_bos_rates[level] if use_codelm_bos else 1,
    # rate_id always 0) since level identity is already structural, not via a shared index.
    bos_rate_mode: str = "relative"  # only matters when share_across_levels=True (rate_id is always
    # 0 in the False/per-level-instance case). "relative" (default): rate_id keys off the level's
    # own EFFECTIVE stride value (see bos_rate_map), so levels with the SAME stride share the SAME
    # bos row -- e.g. strides=(4,4,4) collapses to n_rates=1 (one rate detected), strides=
    # (3,16,16,-1) collapses (3,16,16,1) to n_rates=3 (levels 1&2 share a row). "absolute": rate_id
    # IS the raw level index (old behavior) -- n_rates=n always, one row per level even if two
    # levels happen to share the same stride.
    context_source: str = "codelm"  # how the downsampler/upsampler get their CONTEXT (the hidden
    # state they attend to, before context_proj). "codelm" (default, current/original behavior):
    # re-run this level's own input through CodeLM's own causal self-attention block stack
    # (encoder_hidden) -- context is CONTEXTUALIZED (every position has seen its own causal past via
    # CodeLM's attention). "own_embed": skip CodeLM's block stack entirely -- context is a PLAIN
    # per-position embedding (own_ctx_embed/own_ctx_proj, same shape as CodeLM's own_input_embed/
    # own_input_proj) with NO cross-position mixing at all, downsampler and upsampler each get their
    # OWN dedicated table. "shared_embed": same plain-embedding bypass, but downsampler and upsampler
    # SHARE one table. Added 2026-09-28 to test the hypothesis that CodeLM's causal self-attention
    # context (vs. the old pre-refactor run_lagcodec.py's dedicated, non-contextualized ctx_embed
    # table) is responsible for the periodic every-8th-row/col decode artifact (see
    # audit_gen_dots.py) -- weight-sharing, label-antialiasing and group-size (upsampler_ncodes) were
    # all ruled out as the cause first. Purely additive; "codelm" is unchanged/default so no existing
    # config's behavior changes.
    pardec_token_head: str = "ar"  # how a PardecLM (downsampler AND upsampler) predicts the
    # pq_chunks digits of one code. "ar" (default, current behavior): small autoregressive head,
    # digit m conditioned on digits <m (teacher-forced digits in pardec_score, own digits in
    # pardec_generate). "linear": ONE parallel linear head (output_head_linear) predicts all digits
    # independently from the hidden state, like the original lagcodec code_head -- no digit-level
    # dependence at all, so no digit-level teacher forcing/leak either. Incompatible with
    # downsampler_rollout (the rollout self-feeds digits through the AR head).
    codelm_token_head: str = "linear"  # CodeLM's own NTP head (its aux NTP loss AND encoder_free_run
    # sampling). "linear" (default, current behavior): ntp_head predicts all pq_chunks digits of the
    # next token in parallel/independently. "ar": a small autoregressive digit head (same design as
    # the PardecLM AR head, token_dim/token_n_heads) -- digit m conditioned on digits <m of the SAME
    # token (teacher-forced real digits in the NTP loss, own sampled digits in free-run generation).
    # The AR head's params are only allocated in "ar" mode. Prefill (KV-cache build over the prompt)
    # is head-independent; only the per-step next-token sampling changes.
    downsampler_rollout: bool = False  # False (default, current behavior): the downsampler is
    # TEACHER-FORCED on label_fn(image) digits inside pardec_score, and the code it emits is
    # quantized from those teacher-forced logits (train/gen mismatch; a ground-truth leak into a
    # latent when label_reg_weight=0). True: the downsampler instead SELF-FEEDS -- digits are
    # sampled (straight-through gumbel / hard, per quantize_mode) and fed back through the AR head,
    # exactly as at inference; label_fn is then only used for the optional label_reg aux loss.
    # Needs downsampler_ncodes==1, downsampler_decode_past/future==0, pardec_token_head="ar".
    # Warns when False while label_reg_weight==0.
    downsampler_rollout_prob: float = 1.0  # probability (per train step, one draw per level) of
    # using the rollout path instead of the teacher-forced one when downsampler_rollout=True.
    # Eval (rng=None) always uses the rollout. Warns when !=1 while label_reg_weight==0.
    upsampler_rollout: bool = False  # same idea as downsampler_rollout, but for the upsampler's own
    # digit-AR head (decode_logits_and_target_multipass): False (default) teacher-forces every
    # digit-level AR step on the real target (pardec_score/token_ar_teacher_forced) -- the decode
    # LOSS always matches this, but pardec_generate's actual inference-time digit sampling (self-fed,
    # sequential) is then never exercised during training at all. True: digits are self-fed
    # (token_ar_rollout, straight-through per quantize_mode) while the loss still scores against the
    # real target -- training the digit head the way it's actually used at generation time. Needs
    # upsampler_ncodes==1, upsampler_decode_past/future==0, pardec_token_head="ar".
    upsampler_rollout_prob: float = 1.0  # probability (per decode step, one draw per level) of using
    # the rollout path instead of teacher-forced when upsampler_rollout=True. Eval (rng=None) always
    # uses the rollout.
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
    upsampler_ncodes: tuple = 1
    sync: tuple = False
    precision: str = "bf16"
    # "freeze": in phase p, levels < p-1 (codelm/downsampler/upsampler) are frozen; only the newly
    # added top level trains. Needs share_across_levels=False (per-level weights to freeze).
    curriculum_mode: str = "no_freeze"
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

    # gen-eval forward (encode) direction: "generate" (default) self-generates each level's code via
    # the downsampler's own AR head (encode_pardec_downsampler_generate, no ground-truth digit target
    # -- still contextualized on the real prompt image via CodeLM, just not teacher-forced on
    # label_fn's target); "teacher_force" instead uses the real image's label_fn target to produce
    # each level's code (encode_pardec_downsampler, the original/only behavior before this flag).
    # Either way the SAME backward cascade (decode_generate_multipass) reconstructs down to level 0.
    # NOTE: neither mode's digit-level choice is affected by quantize_mode -- "generate" samples via
    # greedy/temperature (see gen_eval_greedy_only below), "teacher_force" via quantize_dispatch
    # (itself argmax unless gumbel_at_inference=True). Default (gen_eval_greedy_only=True) means BOTH
    # modes are plain argmax out of the box -- flip gen_eval_greedy_only=False for "generate" to
    # actually sample.
    gen_eval_encode_mode: str = "generate"
    # skip the sample=True half of run_gen_eval_both (temperature/top_k sampled decode, and -- for
    # gen_eval_encode_mode="generate" -- the forward-encode direction's own sampling too, since both
    # share the same greedy=not sample wiring) and only run the greedy/argmax decode -- halves
    # (quarters, with eval_gen_train) gen-eval cost when the sampled variant isn't needed. Default
    # True: assures gen-eval is deterministic argmax by default, regardless of gen_eval_encode_mode.
    gen_eval_greedy_only: bool = True
    # mid-phase (gen_eval_every_step/epoch) and end-of-phase (--final_eval) gen-eval normally only
    # evaluate top=phase-1 (the level just being trained). True: also loop top=0..phase-2, same as the
    # all-phases-done final loop at the end of main() already does -- gives per-level reconstruction
    # visibility at every eval checkpoint, not just at the very end. Multiplies gen-eval cost by phase
    # (each lower top re-does its own full encode+cascade), so left off by default.
    gen_eval_all_levels: bool = False
    # sanity reconstruction, run alongside the normal gen-eval: SAME cross-level ctx cascade as
    # training (honors cfg.level_gt_drop as-is -- 1.0 means ctx between levels is still the model's
    # own encoded prediction, not real codes, matching what training actually conditions on), but
    # every digit-level AR step is forced teacher-forced (pardec_score, never pardec_generate/
    # token_ar_rollout) regardless of upsampler_rollout -- isolates whether decode works at all once
    # digit-level self-feeding is removed, on both train and val images. Saves
    # samples_{tag}_tfsanity.png. Added 2026-10-03.
    gen_eval_teacher_force_sanity: bool = False

    # level_forward's decode-cascade `ctx` (the context fed from level i to level i-1's decode --
    # real_ctx/pseudo_ctx) is a straight-through estimator output by
    # default: forward is hard, but gradient flows BACKWARD through it into whatever produced it --
    # the encoder for the initial ctx/real_ctx, or the level-ABOVE's upsampler for pseudo_ctx. True:
    # stop_gradient's `ctx` every time it's (re)built, so each level's decode loss only trains that
    # level's own weights (plus its own encode, via label_reg/ntp), never leaks gradient across levels
    # through this path -- matches level 0's target (raw bytes), which was never differentiable to
    # begin with. Unrelated to level_gt_drop (which controls the VALUE distribution -- real vs
    # self-fed -- not whether gradient flows through whichever value is chosen).
    # "pseudo": detach ONLY the predicted ctx (upper upsampler's quantized logits); the real encoder
    # ctx (top code, real_ctx) keeps its gradient -- the encoders' only reconstruction signal.
    ctx_stop_gradient: bool | str = False
    # level refine (upsampler, per level): passes>1 re-decodes each group with a draft of its
    # level_refine_window preceding groups = argmax of the previous pass (detached). Same at
    # generation (draft = previous generated pass). Loss is averaged over all passes.
    level_refine_passes: tuple = 1
    level_refine_window: tuple = 0
    # training only, per draft position: prob of drafting the model's own prediction; else the real
    # target (1.0 = always own, like level_gt_drop). Eval/no-rng always drafts own prediction.
    level_refine_gt_drop: float = 1.0
    # "fixed": every pass has the same row [cond|bos|draft slot|targets]; pass 1's slot (and any slot
    # before the image start) holds a learned mask token. "variable": pass 1 has no slot, so targets
    # sit Pp positions closer to bos than in refine passes; out-of-image slots zeroed + masked.
    level_refine_layout: str = "fixed"
    # training draft from the previous pass's logits (always detached): "argmax" (matches greedy gen) or
    # "sample" (gumbel-max at level_refine_draft_temperature, matches sampled gen). Eval/no-rng: argmax.
    level_refine_draft_mode: str = "argmax"
    level_refine_draft_temperature: float = 1.0
    # same-level decode cycles (see module docstring). per level, 1 = off
    level_cycles: tuple = 1
    level_cycle_mode: str = "memoryless"  # memoryless | stack
    # training source of the tokens re-encoded into the next cycle's code (generation always free-runs)
    level_cycle_input: str = "rollout"  # rollout | pss | gt
    level_cycle_pss_prob: float = 1.0  # pss: per-position prob of own argmax, else GT. Eval: always own
    level_cycle_detach: bool = True  # False: grad flows into the re-encoder via quantize_mode's estimator
    level_cycle_loss: str = "all"  # all = average over cycles | last
    gen_level_cycles: tuple = None  # generation override per level (None = level_cycles); stack: <= level_cycles
    # per level: transformer | gru | linear_gru | ssm (recurrent: causal, attention-only fields ignored)
    codelm_backbone: tuple = "transformer"
    downsampler_backbone: tuple = "transformer"
    upsampler_backbone: tuple = "transformer"
    ssm_state_dim: int = 16

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

        bcast("code_vocab", int)
        bcast("pq_chunks", int)
        bcast("mlp_mult", int)
        bcast("rope_base", (int, float))
        bcast("upsampler_ncodes", int)
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
        bcast("downsampler_ncodes", int)
        bcast_opt("downsampler_remat", bool)
        bcast("upsampler_d_model", int)
        bcast("upsampler_n_layers", int)
        bcast("upsampler_n_heads", int)
        bcast("upsampler_n_kv_heads", int)
        bcast("upsampler_decode_past", int)
        bcast("upsampler_decode_future", int)
        bcast("upsampler_window", int)
        bcast("level_refine_passes", int)
        bcast("level_refine_window", int)
        bcast("level_cycles", int)
        if self.gen_level_cycles is None:
            self.gen_level_cycles = self.level_cycles
        bcast("gen_level_cycles", int)
        bcast("codelm_backbone", str)
        bcast("downsampler_backbone", str)
        bcast("upsampler_backbone", str)
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
        assert len(self.mlp_mult) == n and len(self.rope_base) == n and len(self.upsampler_ncodes) == n
        assert len(self.sync) == n
        assert len(self.upsampler_decode_past) == n and len(self.upsampler_decode_future) == n
        for i in range(n):
            assert self.upsampler_decode_past[i] >= 0 and self.upsampler_decode_future[i] >= 0, \
                f"level {i}: upsampler_decode_past={self.upsampler_decode_past[i]}/" \
                f"upsampler_decode_future={self.upsampler_decode_future[i]} must be >=0"
            assert self.downsampler_decode_past[i] >= 0 and self.downsampler_decode_future[i] >= 0, \
                f"level {i}: downsampler_decode_past={self.downsampler_decode_past[i]}/" \
                f"downsampler_decode_future={self.downsampler_decode_future[i]} must be >=0"
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
            assert self.attn_lookahead[i] >= 0, \
                f"level {i}: attn_lookahead={self.attn_lookahead[i]} must be >=0 (0=plain causal)"

        top_level_trainable = self.strides[-1] != -1
        code_count = total_bytes_of(self) // self.byte_group
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
            n_blocks_i, G_i = code_count, self.upsampler_ncodes[i]
            assert G_i >= 1, f"level {i}: upsampler_ncodes={G_i} must be >=1"
            if G_i > n_blocks_i:
                warnings.warn(
                    f"level {i}: upsampler_ncodes={G_i} exceeds n_blocks={n_blocks_i} (this level's "
                    f"own code count) -- clamps to one single group, same as upsampler_ncodes="
                    f"{n_blocks_i} (the fully-sequential 'original' degenerate case); recommend "
                    f"setting upsampler_ncodes={n_blocks_i} explicitly for clarity")
            D_i = self.downsampler_ncodes[i]
            assert D_i >= 1, f"level {i}: downsampler_ncodes={D_i} must be >=1"
            n_kblocks_i = n_blocks_i  # downsampler's own group count is in units of K-blocks (n_blocks_i), same as n_blocks_i above
            if D_i > n_kblocks_i:
                warnings.warn(
                    f"level {i}: downsampler_ncodes={D_i} exceeds n_blocks={n_kblocks_i} -- clamps to "
                    f"one single group (all codes at this level scored/generated together)")

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
        # share_across_levels=True (default): exactly one CodeLM, one Downsampler, one Upsampler for
        # the whole model (see LagCodecModel) -- every field that determines a weight SHAPE must
        # therefore be uniform across ALL n levels (index 0 is what LagCodecModel actually builds
        # from). share_across_levels=False: one independent CodeLM/Downsampler/Upsampler per level,
        # so these fields may legitimately differ per level -- assert skipped entirely in that mode.
        # Runtime-only grouping fields (upsampler_ncodes, downsampler_ncodes, decode_past/future,
        # ...) are exempt in EITHER mode -- those vary per level as plain call-time arguments.
        if self.share_across_levels:
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
                "codelm_backbone", "downsampler_backbone", "upsampler_backbone",
            )
            for f in singleton_uniform_fields:
                vals = getattr(self, f, None)
                if isinstance(vals, tuple) and len(vals) == n:
                    assert len(set(vals)) <= 1, \
                        f"share_across_levels=True needs uniform '{f}' across ALL levels " \
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
        assert self.bos_rate_mode in ("relative", "absolute")
        assert self.context_source in ("codelm", "own_embed", "shared_embed")
        assert self.pardec_token_head in ("ar", "linear"), self.pardec_token_head
        assert self.codelm_token_head in ("ar", "linear"), self.codelm_token_head
        assert 0.0 <= self.downsampler_rollout_prob <= 1.0, self.downsampler_rollout_prob
        if self.downsampler_rollout:
            if self.pardec_token_head != "ar":
                raise ValueError("downsampler_rollout is incompatible with pardec_token_head='linear' "
                                 "(the rollout self-feeds digits through the AR token head)")
            assert all(g == 1 for g in self.downsampler_ncodes), \
                f"downsampler_rollout needs downsampler_ncodes==1 at every level, got {self.downsampler_ncodes}"
            assert all(p == 0 for p in self.downsampler_decode_past) \
                and all(f == 0 for f in self.downsampler_decode_future), \
                "downsampler_rollout needs downsampler_decode_past/future==0 (they embed real label tokens)"
        assert 0.0 <= self.upsampler_rollout_prob <= 1.0, self.upsampler_rollout_prob
        if self.upsampler_rollout:
            if self.pardec_token_head != "ar":
                raise ValueError("upsampler_rollout is incompatible with pardec_token_head='linear' "
                                 "(the rollout self-feeds digits through the AR token head)")
            assert all(g == 1 for g in self.upsampler_ncodes), \
                f"upsampler_rollout needs upsampler_ncodes==1 at every level, got {self.upsampler_ncodes}"
            assert all(p == 0 for p in self.upsampler_decode_past) \
                and all(f == 0 for f in self.upsampler_decode_future), \
                "upsampler_rollout needs upsampler_decode_past/future==0 (they embed real target tokens)"
        _ds_leak = (self.pardec_token_head == "ar" or any(g > 1 for g in self.downsampler_ncodes)
                    or any(pst > 0 for pst in self.downsampler_decode_past))
        if self.label_reg_weight == 0 and _ds_leak:
            if not self.downsampler_rollout:
                warnings.warn("label_reg_weight=0 but downsampler_rollout=False: the downsampler is "
                              "teacher-forced on label_fn(image) digits/codes even though nothing supervises "
                              "it toward them (ground-truth leak into a latent + train/gen mismatch). "
                              "Consider downsampler_rollout=True.")
            elif self.downsampler_rollout_prob != 1.0:
                warnings.warn(f"label_reg_weight=0 with downsampler_rollout_prob="
                              f"{self.downsampler_rollout_prob}!=1: that fraction of steps still "
                              "teacher-forces the downsampler on label_fn digits/codes (ground-truth leak).")
        assert self.strides[-1] == -1 or self.strides[-1] >= 1, \
            "top level's stride is either -1 (don't-care, legacy: top level stays untrained/wasted " \
            "-- see top_level_trainable) or a real stride >=1 (top level becomes fully trainable: " \
            "its own encoder gets a phase, and it gets a real decoder too)"
        assert all(s >= 1 for s in self.strides[:-1])
        n_positions = total_bytes_of(self) // self.byte_group
        assert n_positions % math.prod(self.strides[:-1]) == 0
        assert self.precision in ("bf16", "fp32")
        assert self.curriculum_mode in ("freeze", "no_freeze")
        if self.curriculum_mode == "freeze":
            assert not self.share_across_levels, \
                "curriculum_mode='freeze' needs share_across_levels=False (shared weights can't be frozen per level)"
        assert self.quantize_mode in ("argmax", "gumbel", "zgr", "reinmax_limit")
        assert self.gen_eval_encode_mode in ("generate", "teacher_force")
        if not isinstance(self.ctx_stop_gradient, str):
            self.ctx_stop_gradient = bool(self.ctx_stop_gradient)
        assert self.ctx_stop_gradient in (False, True, "pseudo"), self.ctx_stop_gradient
        for i, (rp, rw) in enumerate(zip(self.level_refine_passes, self.level_refine_window)):
            assert rp >= 1 and rw >= 0, f"level {i}: level_refine_passes={rp} (>=1), level_refine_window={rw} (>=0)"
            assert rp == 1 or rw > 0, f"level {i}: level_refine_passes={rp} needs level_refine_window>0"
            assert rp == 1 or self.upsampler_decode_past[i] == 0, \
                f"level {i}: level refine and upsampler_decode_past share one slot"
        assert 0.0 <= self.level_refine_gt_drop <= 1.0, self.level_refine_gt_drop
        assert self.level_refine_layout in ("fixed", "variable"), self.level_refine_layout
        assert self.level_refine_draft_mode in ("argmax", "sample"), self.level_refine_draft_mode
        assert self.level_refine_draft_temperature > 0.0, self.level_refine_draft_temperature
        assert len(self.level_cycles) == n and len(self.gen_level_cycles) == n
        assert self.level_cycle_mode in ("memoryless", "stack"), self.level_cycle_mode
        assert self.level_cycle_input in ("rollout", "pss", "gt"), self.level_cycle_input
        assert 0.0 <= self.level_cycle_pss_prob <= 1.0, self.level_cycle_pss_prob
        assert self.level_cycle_loss in ("all", "last"), self.level_cycle_loss
        for i in range(n):
            c, gc = self.level_cycles[i], self.gen_level_cycles[i]
            assert c >= 1 and gc >= 1, f"level {i}: level_cycles={c}, gen_level_cycles={gc} must be >=1"
            if self.level_cycle_mode == "stack":
                assert gc <= c, f"level {i}: stack mode has {c - 1} slots, gen_level_cycles={gc} > level_cycles"
            if c > 1 or gc > 1:
                assert self.upsampler_decode_past[i] == 0, f"level {i}: cycles need upsampler_decode_past=0"
                assert self.downsampler_ncodes[i] == 1 and self.downsampler_decode_past[i] == 0, \
                    f"level {i}: cycles need downsampler_ncodes=1, downsampler_decode_past=0 (self-fed re-encode)"
                assert c > 1 or self.level_cycle_mode == "memoryless", \
                    f"level {i}: gen_level_cycles>1 with level_cycles=1 only for memoryless (no trained slot params)"
        assert len(self.codelm_backbone) == n and len(self.downsampler_backbone) == n \
            and len(self.upsampler_backbone) == n
        for i in range(n):
            for name in ("codelm_backbone", "downsampler_backbone", "upsampler_backbone"):
                assert getattr(self, name)[i] in BACKBONES, f"level {i}: {name}={getattr(self, name)[i]!r}"
            if self.codelm_backbone[i] != "transformer":
                assert self.attn_lookahead[i] == 0, f"level {i}: recurrent codelm_backbone needs attn_lookahead=0"
        if self.share_downsampler_upsampler_lm:
            assert self.downsampler_backbone == self.upsampler_backbone, \
                "share_downsampler_upsampler_lm needs downsampler_backbone == upsampler_backbone"
        assert self.ssm_state_dim >= 1, self.ssm_state_dim
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

def byte_to_pq_idx(byte_vals: torch.Tensor, pq_chunks: int, code_vocab: int) -> torch.Tensor:
    bits_per_chunk = max(1, round(math.log2(code_vocab)))
    total_bits = pq_chunks * bits_per_chunk
    shifted = byte_vals >> (8 - total_bits) if total_bits <= 8 else byte_vals << (total_bits - 8)
    chunks = [(shifted >> ((pq_chunks - 1 - c) * bits_per_chunk)) & (code_vocab - 1) for c in range(pq_chunks)]
    return torch.stack(chunks, dim=-1)


def rgb_byte_pq_fn(flat_bytes: torch.Tensor, pq_chunks: int, code_vocab: int) -> torch.Tensor:
    # byte_group==pq_chunks and code_vocab==256: each RGB channel already IS one pq digit
    assert code_vocab == 256, f"rgb_byte_pq_fn needs code_vocab=256 (one chunk per byte value), got {code_vocab}"
    assert flat_bytes.shape[-1] == pq_chunks, \
        f"rgb_byte_pq_fn needs byte_group(={flat_bytes.shape[-1]}) == pq_chunks(={pq_chunks})"
    return flat_bytes


def _resize_weight_mat(inp: int, out: int, device) -> torch.Tensor:
    # jax.image.resize's antialiased triangle-kernel weights (scale_and_translate), (inp, out)
    inv = inp / out
    sample = (torch.arange(out, dtype=torch.float32, device=device) + 0.5) * inv - 0.5
    x = (sample[None, :] - torch.arange(inp, dtype=torch.float32, device=device)[:, None]).abs() / max(inv, 1.0)
    w = torch.clamp(1 - x, min=0)
    tot = w.sum(0, keepdim=True)
    w = torch.where(tot.abs() > 1000.0 * float(np.finfo(np.float32).eps), w / torch.where(tot != 0, tot, torch.ones_like(tot)),
                    torch.zeros_like(w))
    return torch.where(((sample >= -0.5) & (sample <= inp - 0.5))[None, :], w, torch.zeros_like(w))


def _resize_bilinear(img: torch.Tensor, side: int) -> torch.Tensor:
    # (M,H,W,C) -> (M,side,side,C) like jax.image.resize(method="bilinear"); float32 summation order differs
    # from XLA's, so ~0.1% of exact .5 values round the other way (labels off by 1)
    wh = _resize_weight_mat(img.shape[1], side, img.device)
    ww = _resize_weight_mat(img.shape[2], side, img.device)
    return torch.einsum("mHwc,wW->mHWc", torch.einsum("mhwc,hH->mHwc", img, wh), ww)


def _raster_image(flat_bytes: torch.Tensor, cfg: "Config", pixel_order: np.ndarray) -> torch.Tensor:
    M = flat_bytes.shape[0]
    pix = flat_bytes.reshape(M, cfg.img_size * cfg.img_size, 3).float()
    raster = torch.zeros_like(pix)
    raster[:, torch.as_tensor(pixel_order, device=pix.device)] = pix
    return raster.reshape(M, cfg.img_size, cfg.img_size, 3)


def default_label_fn(flat_bytes: torch.Tensor, cfg: "Config", pixel_order: np.ndarray, n_blocks: int,
                     pq_chunks: int, code_vocab: int) -> torch.Tensor:
    M = flat_bytes.shape[0]
    side = max(1, round(math.sqrt(n_blocks)))
    gray = _resize_bilinear(_raster_image(flat_bytes, cfg, pixel_order), side).mean(-1)
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)
    flat_gray = gray.reshape(M, side * side)[:, torch.as_tensor(low_order, device=gray.device)]
    if side * side > n_blocks:
        flat_gray = flat_gray[:, :n_blocks]
    elif side * side < n_blocks:
        flat_gray = F.pad(flat_gray, (0, n_blocks - side * side))
    byte_vals = torch.round(flat_gray.clamp(0, 255)).long()
    return byte_to_pq_idx(byte_vals, pq_chunks, code_vocab)


def rgb_label_fn(flat_bytes: torch.Tensor, cfg: "Config", pixel_order: np.ndarray, n_blocks: int,
                 pq_chunks: int, code_vocab: int) -> torch.Tensor:
    # chunk c = channel c's own downsampled byte (no grayscale, no bit-slicing); pq_chunks=3, code_vocab=256
    assert pq_chunks == 3 and code_vocab == 256, \
        f"rgb_label_fn needs pq_chunks=3,code_vocab=256 (one chunk per RGB channel), got {pq_chunks},{code_vocab}"
    M = flat_bytes.shape[0]
    side = max(1, round(math.sqrt(n_blocks)))
    small = _resize_bilinear(_raster_image(flat_bytes, cfg, pixel_order), side)
    low_order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(side * side)
    flat_rgb = small.reshape(M, side * side, 3)[:, torch.as_tensor(low_order, device=small.device)]
    if side * side > n_blocks:
        flat_rgb = flat_rgb[:, :n_blocks]
    elif side * side < n_blocks:
        flat_rgb = F.pad(flat_rgb, (0, 0, 0, n_blocks - side * side))
    return torch.round(flat_rgb.clamp(0, 255)).long()


def default_label_fn_pil(images: np.ndarray, cfg: "Config", pixel_order: np.ndarray, n_blocks: int,
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


LABEL_FNS = {"default_label_fn_jax": default_label_fn, "rgb_label_fn_jax": rgb_label_fn,
             "default_label_fn": default_label_fn, "rgb_label_fn": rgb_label_fn,
             "default_label_fn_pil": default_label_fn_pil}


class BatchIterator:
    # same epoch-seeded shuffle/resume state as the JAX version (single process)
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, n_devices: int,
                 shuffle: bool, seed: int, cfg: "Config"):
        self.images, self.labels = images, labels
        self.batch_size, self.n_devices = batch_size, n_devices
        self.shuffle = shuffle
        self.epoch_rng = np.random.default_rng(seed)
        self.epoch_seed = None
        self.pos = 0
        self.total = batch_size * n_devices
        self.cfg = cfg
        self.pixel_order = pixel_order_for(cfg)
        self.n_positions = n_positions_of(cfg)

    def __len__(self):
        return len(self.images) // self.total

    def __iter__(self):
        n = len(self.images)
        if self.epoch_seed is None:
            self.epoch_seed = int(self.epoch_rng.integers(0, 2 ** 31 - 1))
            self.pos = 0
        idx = np.random.default_rng(self.epoch_seed).permutation(n) if self.shuffle else np.arange(n)
        g = self.total
        starts = list(range(0, n - g + 1, g))
        for bi in range(self.pos, len(starts)):
            sel = idx[starts[bi]:starts[bi] + g]
            positions = images_to_positions(self.images[sel], self.cfg, self.pixel_order)
            self.pos = bi + 1
            yield positions.reshape(self.total, self.n_positions, self.cfg.byte_group)
        self.epoch_seed = None
        self.pos = 0


def safe_argmax(x: torch.Tensor) -> torch.Tensor:
    return torch.argmax(x, dim=-1)  # first index of the max, like the JAX safe_argmax


def _one_hot(idx: torch.Tensor, n: int, dtype) -> torch.Tensor:
    return F.one_hot(idx, n).to(dtype)


def quantize_hard(logits: torch.Tensor, rng=None, quantize_drop: float = 0.0, tau: float = 1.0) -> tuple:
    soft = torch.softmax(logits / tau, dim=-1)
    idx = safe_argmax(soft)
    hard = _one_hot(idx, logits.shape[-1], soft.dtype)
    st = soft + (hard - soft).detach()
    if quantize_drop > 0 and rng is not None:
        drop = rand_bernoulli(rng, quantize_drop, soft.shape[:-1], soft.device)[..., None]
        return torch.where(drop, soft, st), idx
    return st, idx


def quantize_gumbel(logits: torch.Tensor, rng: Key, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    rng, drop_rng = rng.split(2)
    noisy = (logits + rand_gumbel(rng, logits.shape, logits.device)) / tau
    soft = torch.softmax(noisy, dim=-1)
    idx = safe_argmax(soft)
    hard = _one_hot(idx, logits.shape[-1], soft.dtype)
    st = soft + (hard - soft).detach()
    if quantize_drop > 0:
        drop = rand_bernoulli(drop_rng, quantize_drop, soft.shape[:-1], soft.device)[..., None]
        return torch.where(drop, soft, st), idx
    return st, idx


def quantize_zgr(logits: torch.Tensor, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    if rng is not None:
        rng, drop_rng = rng.split(2)
        idx = safe_argmax(logits + rand_gumbel(rng, logits.shape, logits.device))
    else:
        drop_rng = None
        idx = safe_argmax(logp)
    y = _one_hot(idx, logits.shape[-1], logp.dtype)
    logpx = (logp * y).sum(-1, keepdim=True)
    dx = (p + (y - p.detach()) * logpx) / 2
    st = y + (dx - dx.detach())
    if quantize_drop > 0 and drop_rng is not None:
        drop = rand_bernoulli(drop_rng, quantize_drop, p.shape[:-1], p.device)[..., None]
        return torch.where(drop, p, st), idx
    return st, idx


def quantize_reinmax_limit(logits: torch.Tensor, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    K = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)
    if rng is not None:
        rng, drop_rng = rng.split(2)
        idx = safe_argmax(logits + rand_gumbel(rng, logits.shape, logits.device))
    else:
        drop_rng = None
        idx = safe_argmax(p)
    y = _one_hot(idx, K, p.dtype)
    p_x = (p * y).sum(-1, keepdim=True).clamp_min(1e-8)
    col = (y - p) / p_x
    rank1 = col[..., :, None] * y[..., None, :]
    eye = torch.eye(K, dtype=p.dtype, device=p.device)
    S = (eye + rank1) / (2 * K) - torch.ones(K, K, dtype=p.dtype, device=p.device) / (2 * K * K)
    dx = torch.einsum("...ij,...j->...i", S.detach(), logits)
    st = y + (dx - dx.detach())
    if quantize_drop > 0 and drop_rng is not None:
        drop = rand_bernoulli(drop_rng, quantize_drop, p.shape[:-1], p.device)[..., None]
        return torch.where(drop, p, st), idx
    return st, idx


def quantize_dispatch(mode: str, logits: torch.Tensor, rng, tau: float, quantize_drop: float) -> tuple:
    if rng is not None and mode == "gumbel":
        return quantize_gumbel(logits, rng, tau, quantize_drop)
    if rng is not None and mode == "zgr":
        return quantize_zgr(logits, rng, tau, quantize_drop)
    if rng is not None and mode == "reinmax_limit":
        return quantize_reinmax_limit(logits, rng, tau, quantize_drop)
    return quantize_hard(logits, rng, quantize_drop, tau)


def codebook_utilization(idx: torch.Tensor, vocab: int) -> torch.Tensor:
    flat = idx.reshape(-1, idx.shape[-1])
    utils = []
    for c in range(flat.shape[-1]):
        counts = torch.bincount(flat[:, c], minlength=vocab).float()
        probs = counts / counts.sum().clamp_min(1)
        ent = -(probs * probs.clamp_min(1e-9).log()).sum()
        utils.append(ent.exp() / vocab)
    return torch.stack(utils).mean()


def code_embed_proj(code: torch.Tensor, table: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(code):
        parts = [table[code[..., i]] for i in range(code.shape[-1])]
    else:
        parts = [code[..., i, :].to(table.dtype) @ table for i in range(code.shape[-2])]
    return torch.cat(parts, dim=-1) @ proj


def _draft_past_valid_mask(n_groups: int, Pp: int, Kspan: int, valid_len: int) -> np.ndarray:
    abs_idx = np.array([[g * Kspan - Pp + t for t in range(Pp)] for g in range(n_groups)])
    return (abs_idx >= 0) & (abs_idx < valid_len)


def _windows(src: torch.Tensor, starts, length: int) -> torch.Tensor:
    # src (B, L, ...) -> (B, len(starts), length, ...) with window g = src[:, starts[g]:starts[g]+length]
    idx = torch.as_tensor(starts, device=src.device)[:, None] + torch.arange(length, device=src.device)[None]
    return src[:, idx]


def group_draft_windows(src: torch.Tensor, n_groups: int, Kspan: int, Pp: int, valid_len: int) -> tuple:
    # per group g: the Pp tokens of src just BEFORE the group, never the group's own span
    B = src.shape[0]
    tail = n_groups * Kspan - src.shape[1]
    if tail > 0:
        src = torch.cat([src, src.new_zeros((B, tail) + tuple(src.shape[2:]))], dim=1)
    padded = torch.cat([src.new_zeros((B, Pp) + tuple(src.shape[2:])), src], dim=1)
    win = _windows(padded, [g * Kspan for g in range(n_groups)], Pp).reshape(B * n_groups, Pp, *src.shape[2:])
    valid = torch.as_tensor(_draft_past_valid_mask(n_groups, Pp, Kspan, valid_len), device=src.device)
    return win, valid[None].expand(B, n_groups, Pp).reshape(B * n_groups, Pp)


def reshape_pq(logits: torch.Tensor, pq_chunks: int, code_vocab: int) -> torch.Tensor:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


def sample_idx(logits: torch.Tensor, rng, greedy: bool, temperature: float, top_k: int = 0) -> tuple:
    if greedy:
        return safe_argmax(logits), rng
    rng, k_ = rng.split(2)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        lg = torch.where(lg < torch.topk(lg, top_k, dim=-1).values[..., -1:], torch.full_like(lg, -float("inf")), lg)
    return safe_argmax(lg + rand_gumbel(k_, lg.shape, lg.device)), rng


# ---------------------------------------------------------------- layers
def init_matrix(shape: tuple, scheme: str, residual_out: bool = False, n_layers: int = None) -> torch.Tensor:
    if scheme == "zero":
        if residual_out:
            return torch.zeros(shape)
        p, q = shape
        if p >= q:
            return torch.eye(p, q)
        m = 1
        while m < q:
            m *= 2
        H = torch.ones(1, 1)
        while H.shape[0] < m:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        return (H / math.sqrt(m))[:p, :q]
    assert scheme == "llama", f"unknown init_scheme {scheme!r}"
    std = 0.02 / math.sqrt(2 * n_layers) if (residual_out and n_layers) else 0.02
    return torch.randn(shape) * std


def init_vector(dim: int, scheme: str) -> torch.Tensor:
    return torch.zeros(dim) if scheme == "zero" else torch.randn(dim) * 0.02


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def apply_xsa(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_hat = v * torch.rsqrt(v.pow(2).sum(-1, keepdim=True) + 1e-8)
    return y - (y * v_hat).sum(-1, keepdim=True) * v_hat


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int, n_layers: int = None, init_scheme: str = "llama"):
        super().__init__()
        hidden = d_model * mlp_mult
        self.gate = nn.Parameter(init_matrix((d_model, hidden), init_scheme))
        self.up = nn.Parameter(init_matrix((d_model, hidden), init_scheme))
        self.down = nn.Parameter(init_matrix((hidden, d_model), init_scheme, residual_out=True, n_layers=n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.gate) * (x @ self.up)) @ self.down


def rope_cos_sin_pos(pos: torch.Tensor, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=pos.device) / head_dim))
    freqs = pos.to(torch.float32)[..., None] * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


def _attend(q, k, v, mask, sink=None, neg=-1e9) -> torch.Tensor:
    # q (B,H,T,hd) already scaled; k/v (B,Hkv,S,hd); mask broadcastable to (B,H,T,S) bool
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    logits = q @ k.transpose(-1, -2)
    if mask is not None:
        logits = logits.masked_fill(~mask, neg)
    if sink is not None:
        s = sink.to(logits.dtype)[None, :, None, None].expand(*logits.shape[:3], 1)
        w = torch.softmax(torch.cat([logits, s], dim=-1), dim=-1)[..., :-1]
    else:
        w = torch.softmax(logits, dim=-1)
    return w @ v


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_base: float, n_layers: int = None,
                 init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True, window: int = None,
                 lookahead: int = 0, use_sink: bool = False):
        super().__init__()
        hd = d_model // n_heads
        if init_scheme == "zero":
            qkv = torch.cat([init_matrix((d_model, d_model), init_scheme),
                             init_matrix((d_model, n_kv_heads * hd), init_scheme),
                             init_matrix((d_model, n_kv_heads * hd), init_scheme)], dim=-1)
        else:
            qkv = init_matrix((d_model, d_model + 2 * n_kv_heads * hd), init_scheme)
        self.qkv = nn.Parameter(qkv)
        self.out = nn.Parameter(init_matrix((d_model, d_model), init_scheme, residual_out=True, n_layers=n_layers))
        self.q_norm = nn.Parameter(torch.ones(hd))
        self.k_norm = nn.Parameter(torch.ones(hd))
        self.n_heads, self.n_kv_heads, self.rope_base = n_heads, n_kv_heads, rope_base
        self.use_xsa, self.use_qknorm, self.window, self.lookahead = use_xsa, use_qknorm, window, lookahead
        self.sink = nn.Parameter(torch.zeros(n_heads)) if use_sink else None

    def _split(self, x: torch.Tensor) -> tuple:
        D = x.shape[-1]
        hd = D // self.n_heads
        q, k, v = torch.split(x @ self.qkv, [D, self.n_kv_heads * hd, self.n_kv_heads * hd], dim=-1)
        lead = x.shape[:-1]
        q = q.reshape(*lead, self.n_heads, hd)
        k = k.reshape(*lead, self.n_kv_heads, hd)
        v = v.reshape(*lead, self.n_kv_heads, hd)
        if self.use_qknorm:
            q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        return q, k, v, hd

    def _xsa_out(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.use_xsa:
            rep = self.n_heads // self.n_kv_heads
            y = apply_xsa(y, v.repeat_interleave(rep, dim=1) if rep > 1 else v)
        return y

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v, hd = self._split(x)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = rope_cos_sin_pos(torch.arange(T, device=x.device), hd, self.rope_base)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        t = torch.arange(T, device=x.device)
        if self.window is not None or self.lookahead > 0:  # splash LocalMask: t-window <= s <= t+lookahead
            mask = t[None, :] <= t[:, None] + self.lookahead
            if self.window is not None:
                mask = mask & (t[None, :] >= t[:, None] - self.window)
        else:
            mask = (t[None, :] <= t[:, None]) if causal else None
        y = _attend(q * (1.0 / math.sqrt(hd)), k, v, mask, self.sink, neg=-float("inf"))
        y = self._xsa_out(y, v)
        return y.transpose(1, 2).reshape(B, T, D) @ self.out

    def step(self, x_new: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor, pos: int, T_max: int,
             extra_valid: torch.Tensor = None) -> tuple:
        # single position, caches (Bc,Hkv,T_max,hd) written in place (generation runs without grad)
        Bc, D = x_new.shape
        q, k, v, hd = self._split(x_new)
        cos, sin = rope_cos_sin_pos(torch.tensor(float(pos), device=x_new.device), hd, self.rope_base)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        cache_k[:, :, pos] = k.to(cache_k.dtype)
        cache_v[:, :, pos] = v.to(cache_v.dtype)
        idx = torch.arange(T_max, device=x_new.device)
        valid = idx <= pos
        if self.window is not None:
            valid = valid & (idx >= pos - self.window)
        valid = valid[None, None, None, :].expand(Bc, 1, 1, T_max)
        if extra_valid is not None:
            valid = valid & extra_valid[:, None, None, :]
        y = _attend(q[:, :, None, :] * (1.0 / math.sqrt(hd)), cache_k.to(q.dtype), cache_v.to(q.dtype), valid)
        y = self._xsa_out(y, v[:, :, None, :])[:, :, 0]
        return y.reshape(Bc, D) @ self.out, cache_k, cache_v


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float,
                 n_layers: int = None, init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True,
                 window: int = None, lookahead: int = 0, use_sink: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, rope_base, n_layers=n_layers, init_scheme=init_scheme,
                              use_xsa=use_xsa, use_qknorm=use_qknorm, window=window, lookahead=lookahead,
                              use_sink=use_sink)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult, n_layers=n_layers, init_scheme=init_scheme)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.mlp(self.norm2(x))

    def step(self, x_new, cache_k, cache_v, pos, T_max, extra_valid=None) -> tuple:
        attn_out, ck, cv = self.attn.step(self.norm1(x_new), cache_k, cache_v, pos, T_max, extra_valid)
        x = x_new + attn_out
        return x + self.mlp(self.norm2(x)), ck, cv


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # h_t = a_t * h_{t-1} + b_t along dim 1, h_{-1} = 0 (Hillis-Steele parallel scan, autograd-safe)
    off, T = 1, a.shape[1]
    while off < T:
        b = torch.cat([b[:, :off], b[:, off:] + a[:, off:] * b[:, :-off]], dim=1)
        a = torch.cat([a[:, :off], a[:, off:] * a[:, :-off]], dim=1)
        off *= 2
    return b


class RecurrentMixer(nn.Module):
    # causal fixed-state token mixer replacing attention: gru | linear_gru (minGRU) | ssm (diagonal selective)
    def __init__(self, d_model: int, kind: str, state_dim: int = 16, n_layers: int = None, init_scheme: str = "llama"):
        super().__init__()
        assert kind in RECURRENT_BACKBONES, kind
        D = d_model
        self.kind = kind
        self.w_in = nn.Parameter(init_matrix((D, (3 if kind == "gru" else 2) * D), init_scheme))
        self.w_h = nn.Parameter(init_matrix((D, 3 * D), init_scheme)) if kind == "gru" else None
        self.b = nn.Parameter(torch.zeros((3 if kind == "gru" else 2) * D)) if kind != "ssm" else None
        if kind == "ssm":
            self.w_dt = nn.Parameter(init_matrix((D, D), init_scheme))
            dt = torch.exp(torch.rand(D) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3))
            self.b_dt = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            self.w_bc = nn.Parameter(init_matrix((D, 2 * state_dim), init_scheme))
            self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).expand(D, -1).clone())
            self.d_skip = nn.Parameter(torch.ones(D))
        else:
            self.w_dt = self.b_dt = self.w_bc = self.a_log = self.d_skip = None
        self.out = nn.Parameter(init_matrix((D, D), init_scheme, residual_out=True, n_layers=n_layers))

    def init_state(self, batch: int) -> torch.Tensor:
        D = self.w_in.shape[0]
        shape = (batch, D, self.a_log.shape[1]) if self.kind == "ssm" else (batch, D)
        return torch.zeros(shape, dtype=self.w_in.dtype, device=self.w_in.device)

    def _gates(self, x: torch.Tensor) -> tuple:
        D = x.shape[-1]
        g = x @ self.w_in
        if self.kind == "gru":
            return (g + self.b,)
        if self.kind == "linear_gru":
            g = g + self.b
            z = torch.sigmoid(g[..., :D])
            return 1 - z, z * g[..., D:]
        u, gate = g[..., :D], g[..., D:]
        delta = F.softplus(u @ self.w_dt + self.b_dt)
        bc = u @ self.w_bc
        n = self.a_log.shape[1]
        a = torch.exp(delta[..., None] * -torch.exp(self.a_log))
        bx = (delta * u)[..., None] * bc[..., None, :n]
        return a, bx, bc[..., n:], u, gate

    def _gru_cell(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        D = h.shape[-1]
        gh = h @ self.w_h
        z = torch.sigmoid(g[..., :D] + gh[..., :D])
        r = torch.sigmoid(g[..., D:2 * D] + gh[..., D:2 * D])
        n = torch.tanh(g[..., 2 * D:] + r * gh[..., 2 * D:])
        return (1 - z) * n + z * h

    def _readout(self, h: torch.Tensor, gates: tuple) -> torch.Tensor:
        if self.kind != "ssm":
            return h
        _, _, c, u, gate = gates
        y = torch.einsum("...dn,...n->...d", h, c) + self.d_skip * u
        return y * F.silu(gate)

    def forward(self, x: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
        v = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if valid is None else valid
        gates = self._gates(x)
        if self.kind == "gru":
            h = x.new_zeros((x.shape[0], x.shape[-1]))
            hs = []
            for t in range(x.shape[1]):
                h = torch.where(v[:, t, None], self._gru_cell(h, gates[0][:, t]), h)
                hs.append(h)
            h = torch.stack(hs, dim=1)
        else:
            a, bx = gates[0], gates[1]
            vb = v.reshape(v.shape + (1,) * (a.ndim - 2))
            h = linear_scan(torch.where(vb, a, torch.ones_like(a)), torch.where(vb, bx, torch.zeros_like(bx)))
        return self._readout(h, gates) @ self.out

    def step(self, x: torch.Tensor, state: torch.Tensor, valid: torch.Tensor = None) -> tuple:
        gates = self._gates(x)
        st = state.to(x.dtype)
        h = self._gru_cell(st, gates[0]) if self.kind == "gru" else gates[0] * st + gates[1]
        if valid is not None:
            h = torch.where(valid.reshape(valid.shape + (1,) * (h.ndim - 1)), h, st)
        return self._readout(h, gates) @ self.out, h.to(state.dtype)


class RecurrentBlock(nn.Module):
    def __init__(self, d_model: int, kind: str, mlp_mult: int, state_dim: int = 16, n_layers: int = None,
                 init_scheme: str = "llama"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = RecurrentMixer(d_model, kind, state_dim, n_layers=n_layers, init_scheme=init_scheme)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult, n_layers=n_layers, init_scheme=init_scheme)

    def forward(self, x: torch.Tensor, causal: bool = True, valid: torch.Tensor = None) -> torch.Tensor:
        assert causal, "recurrent backbones are causal only"
        x = x + self.mixer(self.norm1(x), valid)
        return x + self.mlp(self.norm2(x))

    def step(self, x_new: torch.Tensor, state: torch.Tensor, valid: torch.Tensor = None) -> tuple:
        y, state = self.mixer.step(self.norm1(x_new), state, valid)
        x = x_new + y
        return x + self.mlp(self.norm2(x)), state


def make_block(backbone: str, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float,
               n_layers: int = None, init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True,
               window: int = None, lookahead: int = 0, use_sink: bool = False, state_dim: int = 16):
    if backbone == "transformer":
        return Block(d_model, n_heads, n_kv_heads, mlp_mult, rope_base, n_layers=n_layers, init_scheme=init_scheme,
                     use_xsa=use_xsa, use_qknorm=use_qknorm, window=window, lookahead=lookahead, use_sink=use_sink)
    return RecurrentBlock(d_model, backbone, mlp_mult, state_dim, n_layers=n_layers, init_scheme=init_scheme)


def block_cache_init(blk, batch: int, T_max: int, device):
    # generation cache: (k, v) for attention, fixed-size state for recurrent (no KV cache)
    if isinstance(blk, RecurrentBlock):
        return blk.mixer.init_state(batch)
    hd = blk.attn.qkv.shape[0] // blk.attn.n_heads
    return (torch.zeros(batch, blk.attn.n_kv_heads, T_max, hd, device=device),
            torch.zeros(batch, blk.attn.n_kv_heads, T_max, hd, device=device))


def block_step(blk, x_new: torch.Tensor, cache, pos: int, T_max: int, extra_valid: torch.Tensor = None) -> tuple:
    if isinstance(blk, RecurrentBlock):
        return blk.step(x_new, cache, None if extra_valid is None else extra_valid[:, pos])
    x, ck, cv = blk.step(x_new, cache[0], cache[1], pos, T_max, extra_valid)
    return x, (ck, cv)


def _maybe_ckpt(f, x, remat: bool):
    return torch_checkpoint(f, x, use_reentrant=False) if (remat and torch.is_grad_enabled()) else f(x)


def run_block(blk, x: torch.Tensor, remat: bool, rng=None, drop_prob: float = 0.0) -> torch.Tensor:
    out = _maybe_ckpt(blk, x, remat)
    if rng is not None and drop_prob > 0.0:
        keep = rand_bernoulli(rng, 1.0 - drop_prob, (), x.device)
        out = torch.where(keep, out, x)
    return out


def dense_self_attention(attn: Attention, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
    B, T, D = x.shape
    q, k, v, hd = attn._split(x)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = rope_cos_sin_pos(torch.arange(T, device=x.device), hd, attn.rope_base)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    rep = attn.n_heads // attn.n_kv_heads
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)) if causal else None
    y = _attend(q * (1.0 / math.sqrt(hd)), k, v, mask, neg=-float("inf"))
    if attn.use_xsa:
        y = apply_xsa(y, v)
    return y.transpose(1, 2).reshape(B, T, D) @ attn.out


def dense_self_attention_pardec(attn: Attention, x: torch.Tensor, rope_pos_ids: torch.Tensor,
                                key_valid: torch.Tensor) -> torch.Tensor:
    Bc, T, D = x.shape
    q, k, v, hd = attn._split(x)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = rope_cos_sin_pos(rope_pos_ids, hd, attn.rope_base)
    q, k = _rope(q, cos[:, None], sin[:, None]), _rope(k, cos[:, None], sin[:, None])
    idx = torch.arange(T, device=x.device)
    mask = (idx[None, :] <= idx[:, None])[None] & key_valid[:, None, :]
    rep = attn.n_heads // attn.n_kv_heads
    y = _attend(q * (1.0 / math.sqrt(hd)), k, v, mask[:, None])
    if attn.use_xsa:
        y = apply_xsa(y, v.repeat_interleave(rep, dim=1) if rep > 1 else v)
    return y.transpose(1, 2).reshape(Bc, T, D) @ attn.out


def run_block_pardec(blk, x: torch.Tensor, rope_pos_ids: torch.Tensor, key_valid: torch.Tensor,
                     remat: bool) -> torch.Tensor:
    if isinstance(blk, RecurrentBlock):  # order-based, rope ids unused; invalid keys skip the state update
        return _maybe_ckpt(lambda x: blk(x, valid=key_valid), x, remat)

    def f(x):
        x = x + dense_self_attention_pardec(blk.attn, blk.norm1(x), rope_pos_ids, key_valid)
        return x + blk.mlp(blk.norm2(x))
    return _maybe_ckpt(f, x, remat)


def token_ar_teacher_forced(in_proj, member_embed, norm1, attn, ln_f, out_head, dim, in_code_vocab,
                            h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lead = h.shape[:-1]
    D = h.shape[-1]
    chunks = target.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    ctx = (h.reshape(N, D) @ in_proj)[:, None, :]
    tgt_flat = target.reshape(N, chunks)
    member = member_embed[tgt_flat[:, :chunks - 1]] if chunks > 1 else ctx.new_zeros((N, 0, dim))
    seq_in = torch.cat([ctx, member.to(ctx.dtype)], dim=1)
    h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
    return (ln_f(h1) @ out_head).reshape(*lead, chunks, in_code_vocab)


def token_ar_generate(in_proj, member_embed, norm1, attn, ln_f, out_head, chunks: int, h: torch.Tensor,
                      rng, greedy: bool, temperature: float, top_k: int = 0) -> tuple:
    lead = h.shape[:-1]
    D = h.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    collected = [(h.reshape(N, D) @ in_proj)[:, None, :]]
    vals = []
    for m in range(chunks):
        seq_in = torch.cat(collected, dim=1)
        h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
        val_m, rng = sample_idx(ln_f(h1)[:, -1, :] @ out_head, rng, greedy, temperature, top_k)
        vals.append(val_m)
        if m < chunks - 1:
            collected.append(member_embed[val_m][:, None, :].to(seq_in.dtype))
    return torch.stack(vals, dim=1).reshape(*lead, chunks), rng


def token_ar_rollout(in_proj, member_embed, norm1, attn, ln_f, out_head, chunks: int, h: torch.Tensor, rng,
                     quant_fn) -> tuple:
    # self-fed digit rollout: each digit drawn by quant_fn(logits, rng) and fed back via its soft one-hot
    lead = h.shape[:-1]
    D = h.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    collected = [(h.reshape(N, D) @ in_proj)[:, None, :]]
    softs, idxs, lgs = [], [], []
    for m in range(chunks):
        rng, k_ = rng.split(2)
        seq_in = torch.cat(collected, dim=1)
        h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
        logit_m = ln_f(h1)[:, -1, :] @ out_head
        soft_m, idx_m = quant_fn(logit_m, k_)
        softs.append(soft_m)
        idxs.append(idx_m)
        lgs.append(logit_m)
        if m < chunks - 1:
            collected.append((soft_m.to(member_embed.dtype) @ member_embed)[:, None, :])
    V = lgs[0].shape[-1]
    return (torch.stack(softs, 1).reshape(*lead, chunks, V), torch.stack(idxs, 1).reshape(*lead, chunks),
            torch.stack(lgs, 1).reshape(*lead, chunks, V))


# ---------------------------------------------------------------- PardecLM
class PardecLM(nn.Module):
    # downsampler/upsampler: rows [context window | bos | (cycle slots) | (draft) | targets] per group
    def __init__(self, context_hidden_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, n_layers: int,
                 mlp_mult: int, rope_base: float, output_expansion: int, context_window_groups: int,
                 output_vocab: int, output_chunks: int, pq_dim: int, token_dim: int, token_n_heads: int,
                 decode_past: int = 0, decode_future: int = 0, n_rates: int = 1, init_scheme: str = "llama",
                 use_xsa: bool = True, use_qknorm: bool = True, remat: bool = False, window: int = None,
                 shared_blocks=None, shared_ln_f=None, ctx_vocab: int = None, ctx_pq_chunks: int = None,
                 ctx_pq_dim: int = None, shared_ctx_embed=None, shared_ctx_proj=None, token_head: str = "ar",
                 backbone: str = "transformer", state_dim: int = 16, cycle_slots: int = 0):
        super().__init__()
        if shared_blocks is not None:
            self.blocks, self.ln_f = shared_blocks, shared_ln_f
        else:
            self.blocks = nn.ModuleList([make_block(backbone, hidden_dim, n_heads, n_kv_heads, mlp_mult, rope_base,
                                                    n_layers=n_layers, init_scheme=init_scheme, use_xsa=use_xsa,
                                                    use_qknorm=use_qknorm, window=window, state_dim=state_dim)
                                         for _ in range(n_layers)])
            self.ln_f = RMSNorm(hidden_dim)
        P = lambda t: nn.Parameter(t)
        self.context_proj = P(init_matrix((context_hidden_dim, hidden_dim), init_scheme))
        self.n_rates = n_rates
        self.bos_embed = P(torch.stack([init_vector(hidden_dim, init_scheme) for _ in range(n_rates)]))
        self.target_embed = P(init_matrix((output_vocab, pq_dim), init_scheme))
        self.target_proj = P(init_matrix((output_chunks * pq_dim, hidden_dim), init_scheme))
        self.token_in_proj = P(init_matrix((hidden_dim, token_dim), init_scheme))
        self.token_member_embed = P(init_matrix((output_vocab, token_dim), init_scheme))
        self.token_norm1 = RMSNorm(token_dim)
        self.token_attn = Attention(token_dim, token_n_heads, token_n_heads, rope_base, n_layers=1,
                                    init_scheme=init_scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
        self.token_ln_f = RMSNorm(token_dim)
        self.token_out_head = P(init_matrix((token_dim, output_vocab), init_scheme))
        self.output_head_linear = P(init_matrix((hidden_dim, output_chunks * output_vocab), init_scheme))
        self.draft_mask_embed = P(init_vector(hidden_dim, init_scheme))
        if cycle_slots > 0:
            self.cycle_mask_embed = P(init_vector(hidden_dim, init_scheme))
            self.cycle_slot_embed = P(torch.stack([init_vector(hidden_dim, init_scheme) for _ in range(cycle_slots)]))
        else:
            self.cycle_slot_embed = self.cycle_mask_embed = None
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.output_expansion, self.context_window_groups = output_expansion, context_window_groups
        self.output_vocab, self.output_chunks = output_vocab, output_chunks
        self.token_dim, self.token_n_heads = token_dim, token_n_heads
        self.decode_past, self.decode_future = decode_past, decode_future
        self.remat = remat
        self.token_head = token_head
        if shared_ctx_embed is not None:
            self.own_ctx_embed, self.own_ctx_proj = shared_ctx_embed, shared_ctx_proj
        else:
            self.own_ctx_embed = P(init_matrix((ctx_vocab, ctx_pq_dim), init_scheme))
            self.own_ctx_proj = P(init_matrix((ctx_pq_chunks * ctx_pq_dim, context_hidden_dim), init_scheme))


def pardec_context_windows(pardec: PardecLM, context_h_padded: torch.Tensor, context_group_size: int,
                           n_groups: int, n_context_positions: int) -> tuple:
    batch = context_h_padded.shape[0]
    dev = context_h_padded.device
    context_tok = context_h_padded @ pardec.context_proj
    hidden_dim = context_tok.shape[-1]
    n_padded = n_groups * context_group_size
    window_size = (n_padded if pardec.context_window_groups < 0 else
                   min(pardec.context_window_groups * context_group_size + context_group_size, n_padded))
    padded = F.pad(context_tok, (0, 0, window_size, 0))
    group_ends = [(g + 1) * context_group_size for g in range(n_groups)]
    own_window = _windows(padded, group_ends, window_size).reshape(batch * n_groups, window_size, hidden_dim)
    offsets = torch.arange(window_size, device=dev)[None] - window_size + torch.as_tensor(group_ends, device=dev)[:, None]
    rope_ids = offsets.clamp(min=0)
    valid = (offsets >= 0) & (offsets < n_context_positions)
    valid_mask = valid[None].expand(batch, n_groups, window_size).reshape(batch * n_groups, window_size)
    return own_window, rope_ids, valid_mask, window_size, group_ends


def cycle_slot_rows(pardec: PardecLM, cycle_ctx: list, context_group_size: int, n_groups: int,
                    n_context_positions: int, valid_mask: torch.Tensor, dtype) -> tuple:
    # stack slots, each the same window as the main context: a filled slot = windowed revision hidden,
    # an empty one (None) = mask token; + its slot tag. Out-of-image window positions zeroed + masked.
    batch2, window_size = valid_mask.shape
    hidden_dim = pardec.context_proj.shape[1]
    pad = n_groups * context_group_size - n_context_positions
    rows = []
    for s, h in enumerate(cycle_ctx):
        if h is None:
            w = pardec.cycle_mask_embed.to(dtype).expand(batch2, window_size, hidden_dim)
        else:
            hp = F.pad(h, (0, 0, 0, pad)) if pad > 0 else h
            w = pardec_context_windows(pardec, hp, context_group_size, n_groups, n_context_positions)[0].to(dtype)
        w = w + pardec.cycle_slot_embed[s].to(dtype)
        rows.append(torch.where(valid_mask[:, :, None], w, torch.zeros_like(w)))
    return torch.cat(rows, dim=1), torch.cat([valid_mask] * len(cycle_ctx), dim=1)


def pardec_score(pardec: PardecLM, target_seq: torch.Tensor, context_h: torch.Tensor, context_group_size: int,
                 output_group_size: int, rate_id: int = 0, output_expansion: int = None, return_hidden: bool = False,
                 draft_seq: torch.Tensor = None, draft_len: int = 0, draft_fill: str = "zero",
                 cycle_ctx: list = None) -> tuple:
    # teacher-forced scoring of every group in parallel (dense); see the JAX pardec_score for the row layout
    oe = pardec.output_expansion if output_expansion is None else output_expansion
    batch, n_context_positions, _ = context_h.shape
    dev = context_h.device
    n_groups = -(-n_context_positions // context_group_size)
    pad_amount = n_groups * context_group_size - n_context_positions
    context_h_padded = F.pad(context_h, (0, 0, 0, pad_amount)) if pad_amount > 0 else context_h
    own_window, rope_ids, valid_mask, window_size, group_ends = pardec_context_windows(
        pardec, context_h_padded, context_group_size, n_groups, n_context_positions)
    hidden_dim = own_window.shape[-1]
    B2 = batch * n_groups

    refine = draft_seq is not None or draft_len > 0
    assert draft_fill in ("zero", "mask"), draft_fill
    assert draft_seq is not None or draft_len == 0 or draft_fill == "mask", "an empty draft slot needs draft_fill='mask'"
    assert not (refine and pardec.decode_past > 0), "draft_seq (level refine) and decode_past>0 share one slot"
    decode_past = draft_len if refine else pardec.decode_past
    decode_future = pardec.decode_future
    T = output_group_size * oe
    n_output_positions = n_context_positions * output_group_size // context_group_size
    tail_pad = n_groups * T - n_output_positions * oe
    target_padded = target_seq
    if tail_pad > 0:
        target_padded = torch.cat([target_seq, target_seq.new_zeros((batch, tail_pad) + tuple(target_seq.shape[2:]))], 1)
    if decode_future > 0:
        tail_padded = torch.cat([target_padded, target_padded.new_zeros((batch, decode_future) + tuple(target_seq.shape[2:]))], 1)
        real_tail_windows = _windows(tail_padded, [g * T for g in range(n_groups)], T + decode_future)
    else:
        real_tail_windows = target_padded.reshape(batch, n_groups, T, *target_seq.shape[2:])
    real_tail_flat = real_tail_windows.reshape(B2, T + decode_future, *target_seq.shape[2:])
    real_tail_embedded = code_embed_proj(real_tail_flat, pardec.target_embed, pardec.target_proj)

    if decode_past > 0 and refine and draft_seq is None:
        draft_embedded = pardec.draft_mask_embed.to(real_tail_embedded.dtype).expand(B2, decode_past, hidden_dim)
        draft_valid_flat = torch.ones(B2, decode_past, dtype=torch.bool, device=dev)
        target_embedded_flat = torch.cat([draft_embedded, real_tail_embedded], 1)
    elif decode_past > 0:
        draft_flat, draft_valid_flat = group_draft_windows(draft_seq if refine else target_padded, n_groups, T,
                                                           decode_past, n_output_positions * oe)
        draft_embedded = code_embed_proj(draft_flat, pardec.target_embed, pardec.target_proj)
        if refine and draft_fill == "mask":
            draft_embedded = torch.where(draft_valid_flat[:, :, None], draft_embedded,
                                         pardec.draft_mask_embed.to(draft_embedded.dtype))
            draft_valid_flat = torch.ones_like(draft_valid_flat)
        else:
            draft_embedded = torch.where(draft_valid_flat[:, :, None], draft_embedded, torch.zeros_like(draft_embedded))
        target_embedded_flat = torch.cat([draft_embedded, real_tail_embedded], 1)
    else:
        draft_valid_flat = torch.ones(B2, 0, dtype=torch.bool, device=dev)
        target_embedded_flat = real_tail_embedded

    bos = pardec.bos_embed[rate_id].to(own_window.dtype).expand(B2, 1, hidden_dim)
    slot_len = 0
    parts, valids = [own_window, bos], [valid_mask, torch.ones(B2, 1, dtype=torch.bool, device=dev)]
    if cycle_ctx:
        slot_rows, slot_valid = cycle_slot_rows(pardec, cycle_ctx, context_group_size, n_groups,
                                                n_context_positions, valid_mask, own_window.dtype)
        slot_len = slot_rows.shape[1]
        parts.append(slot_rows)
        valids.append(slot_valid)
    row_flat = torch.cat(parts + [target_embedded_flat.to(own_window.dtype)], 1)
    per_group_len = window_size + 1 + slot_len + decode_past + T + decode_future
    key_valid = torch.cat(valids + [draft_valid_flat, torch.ones(B2, T + decode_future, dtype=torch.bool, device=dev)], 1)
    ends = torch.as_tensor(group_ends, device=dev)[:, None]
    rope_bos = torch.cat([ends, ends + 1 + torch.arange(slot_len, device=dev)[None]], 1)
    if refine:
        rope_draft = ends + 1 + slot_len + torch.arange(decode_past, device=dev)[None]
        tail_start = 1 + slot_len + decode_past
    else:
        rope_draft = ends - decode_past + torch.arange(decode_past, device=dev)[None]
        tail_start = 1 + slot_len
    rope_real_tail = ends + tail_start + torch.arange(T + decode_future, device=dev)[None]
    rope_target = torch.cat([rope_draft, rope_real_tail], 1).clamp(min=0)
    rope_g = torch.cat([rope_ids, rope_bos, rope_target], 1)
    rope_pos_ids = rope_g[None].expand(batch, n_groups, per_group_len).reshape(B2, per_group_len)

    def run_stack(x):
        for blk in pardec.blocks:
            x = run_block_pardec(blk, x, rope_pos_ids, key_valid, pardec.remat)
        return x
    hidden = pardec.ln_f(_maybe_ckpt(run_stack, row_flat, pardec.remat))
    pred_pos = window_size + slot_len + decode_past + torch.arange(T, device=dev)
    predicted_hidden = hidden[:, pred_pos].reshape(batch, n_groups * T, hidden_dim)
    valid_len = n_output_positions * oe
    predicted_hidden = predicted_hidden[:, :valid_len]
    target_out = target_seq[:, :valid_len]
    if return_hidden:
        return predicted_hidden
    if pardec.token_head == "linear":
        logits = reshape_pq(predicted_hidden @ pardec.output_head_linear, pardec.output_chunks, pardec.output_vocab)
    else:
        logits = token_ar_teacher_forced(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                         pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                         pardec.token_dim, pardec.output_vocab, predicted_hidden, target_out)
    if decode_future > 0:
        aux_pos = window_size + slot_len + decode_past + T + torch.arange(decode_future, device=dev)
        aux_hidden = hidden[:, aux_pos].reshape(batch, n_groups * decode_future, hidden_dim)
        aux_target = real_tail_windows[:, :, T:].reshape(batch, n_groups * decode_future, *target_seq.shape[2:])
        abs_idx = np.array([[(g + 1) * T + k for k in range(decode_future)] for g in range(n_groups)])
        aux_valid = torch.as_tensor(abs_idx < valid_len, device=dev).reshape(1, n_groups * decode_future, 1)
        aux_valid = aux_valid.expand_as(aux_target).float()
        if pardec.token_head == "linear":
            aux_logits = reshape_pq(aux_hidden @ pardec.output_head_linear, pardec.output_chunks, pardec.output_vocab)
        else:
            aux_logits = token_ar_teacher_forced(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                                 pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                                 pardec.token_dim, pardec.output_vocab, aux_hidden, aux_target)
        nll = -torch.log_softmax(aux_logits, -1).gather(-1, aux_target[..., None])[..., 0]
        denom = aux_valid.sum().clamp_min(1.0)
        aux_loss = (nll * aux_valid).sum() / denom
        aux_acc = ((aux_logits.argmax(-1) == aux_target).float() * aux_valid).sum() / denom
    else:
        aux_loss = aux_acc = logits.new_zeros(())
    return logits, target_out, aux_loss, aux_acc


@torch.no_grad()
def pardec_generate(pardec: PardecLM, context_h: torch.Tensor, context_group_size: int, output_group_size: int,
                    rng, greedy: bool = True, temperature: float = 1.0, top_k: int = 0, rate_id: int = 0,
                    output_expansion: int = None, draft_seq: torch.Tensor = None, draft_len: int = 0,
                    draft_fill: str = "zero", cycle_ctx: list = None) -> torch.Tensor:
    # incremental counterpart of pardec_score: one KV cache (or recurrent state) per group, groups batched
    oe = pardec.output_expansion if output_expansion is None else output_expansion
    batch, n_context_positions, _ = context_h.shape
    dev = context_h.device
    n_groups = -(-n_context_positions // context_group_size)
    pad_amount = n_groups * context_group_size - n_context_positions
    context_h_padded = F.pad(context_h, (0, 0, 0, pad_amount)) if pad_amount > 0 else context_h
    own_window, _, valid_mask, window_size, _ = pardec_context_windows(
        pardec, context_h_padded, context_group_size, n_groups, n_context_positions)
    hidden_dim = own_window.shape[-1]
    B2 = batch * n_groups
    T = output_group_size * oe
    assert draft_fill in ("zero", "mask"), draft_fill
    assert draft_seq is not None or draft_len == 0 or draft_fill == "mask", "an empty draft slot needs draft_fill='mask'"
    Pp = draft_len
    slot_len = 0
    prefix, valids = [own_window, pardec.bos_embed[rate_id].to(own_window.dtype).expand(B2, 1, hidden_dim)], \
        [valid_mask, torch.ones(B2, 1, dtype=torch.bool, device=dev)]
    if cycle_ctx:
        slot_rows, slot_valid = cycle_slot_rows(pardec, cycle_ctx, context_group_size, n_groups,
                                                n_context_positions, valid_mask, own_window.dtype)
        slot_len = slot_rows.shape[1]
        prefix.append(slot_rows)
        valids.append(slot_valid)
    n_output_positions = n_context_positions * output_group_size // context_group_size
    if Pp > 0 and draft_seq is None:
        prefix.append(pardec.draft_mask_embed.to(own_window.dtype).expand(B2, Pp, hidden_dim))
        valids.append(torch.ones(B2, Pp, dtype=torch.bool, device=dev))
    elif Pp > 0:
        draft_flat, draft_valid_flat = group_draft_windows(draft_seq, n_groups, T, Pp, n_output_positions * oe)
        draft_emb = code_embed_proj(draft_flat, pardec.target_embed, pardec.target_proj)
        if draft_fill == "mask":
            draft_emb = torch.where(draft_valid_flat[:, :, None], draft_emb, pardec.draft_mask_embed.to(draft_emb.dtype))
            draft_valid_flat = torch.ones_like(draft_valid_flat)
        else:
            draft_emb = torch.where(draft_valid_flat[:, :, None], draft_emb, torch.zeros_like(draft_emb))
        prefix.append(draft_emb.to(own_window.dtype))
        valids.append(draft_valid_flat)
    prefix = torch.cat(prefix, 1)
    total_steps = window_size + 1 + slot_len + Pp + T
    extra_valid = torch.cat(valids + [torch.ones(B2, T, dtype=torch.bool, device=dev)], 1)
    caches = [block_cache_init(blk, B2, total_steps, dev) for blk in pardec.blocks]

    def self_step(x, pos):
        for i, blk in enumerate(pardec.blocks):
            x, caches[i] = block_step(blk, x, caches[i], pos, total_steps, extra_valid)
        return pardec.ln_f(x)

    hidden = None
    for pos in range(prefix.shape[1]):
        hidden = self_step(prefix[:, pos], pos)
    pos = prefix.shape[1]
    vals = []
    for _ in range(T):
        if pardec.token_head == "linear":
            val, rng = sample_idx(reshape_pq(hidden @ pardec.output_head_linear, pardec.output_chunks,
                                             pardec.output_vocab), rng, greedy, temperature, top_k)
        else:
            val, rng = token_ar_generate(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                         pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                         pardec.output_chunks, hidden, rng, greedy, temperature, top_k)
        vals.append(val)
        hidden = self_step(code_embed_proj(val, pardec.target_embed, pardec.target_proj).to(prefix.dtype), pos)
        pos += 1
    vals = torch.stack(vals, dim=1).reshape(batch, n_groups * T, -1)
    return vals[:, :n_output_positions * oe]


def bos_rate_map(cfg: "Config") -> tuple:
    n = len(cfg.strides)
    if cfg.bos_rate_mode == "absolute":
        return tuple(range(n))
    seen, out = {}, []
    for i in range(n):
        k = cfg.strides[i] if cfg.strides[i] != -1 else 1
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return tuple(out)


def bos_n_rates(cfg: "Config") -> int:
    return len(set(bos_rate_map(cfg)))


# ---------------------------------------------------------------- CodeLM
class CodeLM(nn.Module):
    # the "encoder": contextualizes a level's input (and, for the upsampler, its own code)
    def __init__(self, cfg: "Config", level_idx: int = 0):
        super().__init__()
        D_enc = cfg.codelm_d_model[level_idx]
        n_layers_enc = cfg.codelm_n_layers[level_idx]
        n_heads_enc = cfg.codelm_n_heads[level_idx]
        n_kv_heads_enc = cfg.codelm_n_kv_heads[level_idx]
        self.remat, self.remat_level = cfg.remat, cfg.remat_level
        self.attn_lookahead = cfg.attn_lookahead[level_idx]
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[level_idx], cfg.code_vocab[level_idx]
        self.quantize_mode, self.quantize_drop = cfg.quantize_mode, cfg.quantize_drop
        self.use_codelm_bos, self.codelm_bos_prob = cfg.use_codelm_bos, cfg.codelm_bos_prob
        self.n_heads, self.n_kv_heads = n_heads_enc, n_kv_heads_enc
        pq_dim = cfg.pq_dim[level_idx]
        scheme = cfg.init_scheme
        P = lambda t: nn.Parameter(t)
        self.own_input_embed = P(init_matrix((self.code_vocab, pq_dim), scheme))
        self.own_input_proj = P(init_matrix((self.pq_chunks * pq_dim, D_enc), scheme))
        enc_window_val = (cfg.encoder_attn_window[level_idx] if cfg.encoder_attn_window[level_idx] is not None
                          else cfg.attn_window[level_idx])
        enc_window = None if enc_window_val == -1 else enc_window_val
        self.blocks = nn.ModuleList([make_block(cfg.codelm_backbone[level_idx], D_enc, n_heads_enc, n_kv_heads_enc,
                                                cfg.mlp_mult[level_idx], cfg.rope_base[level_idx], n_layers=n_layers_enc,
                                                init_scheme=scheme, use_xsa=cfg.use_xsa, use_qknorm=cfg.use_qknorm,
                                                window=enc_window, lookahead=self.attn_lookahead,
                                                use_sink=cfg.use_sink, state_dim=cfg.ssm_state_dim)
                                     for _ in range(n_layers_enc)])
        self.ln_f = RMSNorm(D_enc)
        self.code_head = P(init_matrix((D_enc, self.pq_chunks * self.code_vocab), scheme))
        self.ntp_head = P(init_matrix((D_enc, self.pq_chunks * self.code_vocab), scheme))
        n_bos_rates = (len(cfg.strides) if cfg.share_across_levels else cfg.codelm_bos_rates[level_idx]) \
            if cfg.use_codelm_bos else 1
        self.bos_embed = P(torch.stack([init_vector(D_enc, scheme) for _ in range(n_bos_rates)]))
        self.token_head = cfg.codelm_token_head
        self.token_dim = cfg.token_dim[level_idx]
        if self.token_head == "ar":
            td, tnh = cfg.token_dim[level_idx], cfg.token_n_heads[level_idx]
            self.tok_in_proj = P(init_matrix((D_enc, td), scheme))
            self.tok_member_embed = P(init_matrix((self.code_vocab, td), scheme))
            self.tok_norm1 = RMSNorm(td)
            self.tok_attn = Attention(td, tnh, tnh, cfg.rope_base[level_idx], n_layers=1, init_scheme=scheme,
                                      use_xsa=cfg.use_xsa, use_qknorm=cfg.use_qknorm)
            self.tok_ln_f = RMSNorm(td)
            self.tok_out_head = P(init_matrix((td, self.code_vocab), scheme))
        else:
            self.tok_in_proj = self.tok_member_embed = self.tok_norm1 = self.tok_attn = None
            self.tok_ln_f = self.tok_out_head = None


def _run_stack(blocks, h: torch.Tensor, remat: bool, remat_level: bool) -> torch.Tensor:
    def f(h):
        for blk in blocks:
            h = run_block(blk, h, remat and not remat_level)
        return h
    return _maybe_ckpt(f, h, remat_level)


def pardec_context_hidden(codelm: CodeLM, pardec: PardecLM, raw: torch.Tensor, cfg: "Config", rate_id: int, rng,
                          group_size: int) -> torch.Tensor:
    # downsampler/upsampler context, per cfg.context_source (no bos substitution, see the JAX file)
    if cfg.context_source == "codelm":
        x = code_embed_proj(raw, codelm.own_input_embed, codelm.own_input_proj)
        return codelm.ln_f(_run_stack(codelm.blocks, x, codelm.remat, codelm.remat_level))
    return code_embed_proj(raw, pardec.own_ctx_embed, pardec.own_ctx_proj)


def codelm_ntp_logits_tf(codelm: CodeLM, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if codelm.token_head == "linear":
        return reshape_pq(h @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab)
    return token_ar_teacher_forced(codelm.tok_in_proj, codelm.tok_member_embed, codelm.tok_norm1, codelm.tok_attn,
                                   codelm.tok_ln_f, codelm.tok_out_head, codelm.token_dim, codelm.code_vocab, h, target)


def encode_pardec_downsampler(codelm: CodeLM, downsampler: PardecLM, raw: torch.Tensor, target_idx: torch.Tensor,
                              flat_bytes: torch.Tensor, cfg: "Config", pixel_order, label_fn, K: int,
                              rate_id: int = 0, rng=None, downsampler_ncodes: int = 1,
                              encode_temperature: float = 1.0, codelm_rate_id: int = None) -> dict:
    codelm_rate_id = rate_id if codelm_rate_id is None else codelm_rate_id
    h = pardec_context_hidden(codelm, downsampler, raw, cfg, codelm_rate_id, rng, group_size=K * downsampler_ncodes)
    M, L, D = h.shape
    n_blocks = L // K

    def _teacher_forced():
        label_tgt = label_fn(flat_bytes, cfg, pixel_order, n_blocks, codelm.pq_chunks, codelm.code_vocab)
        lg, _, _, _ = pardec_score(downsampler, label_tgt, h, context_group_size=K * downsampler_ncodes,
                                   output_group_size=downsampler_ncodes, rate_id=rate_id)
        cs, ci = quantize_dispatch(codelm.quantize_mode, lg, rng, encode_temperature, codelm.quantize_drop)
        return lg, cs, ci

    def _rollout():
        hid = pardec_score(downsampler, h.new_zeros((M, n_blocks, codelm.pq_chunks), dtype=torch.long), h,
                           context_group_size=K * downsampler_ncodes, output_group_size=downsampler_ncodes,
                           rate_id=rate_id, return_hidden=True)
        qfn = lambda lg_m, k_: quantize_dispatch(codelm.quantize_mode, lg_m, k_ if rng is not None else None,
                                                 encode_temperature, codelm.quantize_drop)
        cs, ci, lg = token_ar_rollout(downsampler.token_in_proj, downsampler.token_member_embed,
                                      downsampler.token_norm1, downsampler.token_attn, downsampler.token_ln_f,
                                      downsampler.token_out_head, downsampler.output_chunks, hid,
                                      rng if rng is not None else Key(0), qfn)
        return lg, cs, ci

    if not cfg.downsampler_rollout:
        logits, code_soft, code_idx = _teacher_forced()
    elif rng is None or cfg.downsampler_rollout_prob >= 1.0:
        logits, code_soft, code_idx = _rollout()
    elif cfg.downsampler_rollout_prob <= 0.0:
        logits, code_soft, code_idx = _teacher_forced()
    else:
        use_roll = bool(rand_bernoulli(rng.fold(8), cfg.downsampler_rollout_prob, (), h.device))
        logits, code_soft, code_idx = _rollout() if use_roll else _teacher_forced()

    p_avg = torch.softmax(logits, -1).mean(dim=(0, 1))
    entropy_loss = (p_avg * p_avg.clamp_min(1e-9).log()).sum(-1).mean()
    ntp_shift = 1 + codelm.attn_lookahead
    if L > ntp_shift:
        tgt = target_idx[:, ntp_shift:]
        ntp_logits = codelm_ntp_logits_tf(codelm, h[:, :-ntp_shift], tgt)
        ntp_loss = -torch.log_softmax(ntp_logits, -1).gather(-1, tgt[..., None]).mean()
        ntp_acc = (ntp_logits.argmax(-1) == tgt).float().mean()
    else:
        ntp_loss = ntp_acc = h.new_zeros(())
    util = codebook_utilization(code_idx, codelm.code_vocab)
    return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                entropy_loss=entropy_loss, logits=logits)


@torch.no_grad()
def encode_pardec_downsampler_generate(codelm: CodeLM, downsampler: PardecLM, raw: torch.Tensor, K: int,
                                       cfg: "Config", rate_id: int = 0, rng=None, greedy: bool = True,
                                       temperature: float = 1.0, top_k: int = 0, downsampler_ncodes: int = 1,
                                       codelm_rate_id: int = None) -> dict:
    codelm_rate_id = rate_id if codelm_rate_id is None else codelm_rate_id
    h = pardec_context_hidden(codelm, downsampler, raw, cfg, codelm_rate_id, rng, group_size=K * downsampler_ncodes)
    code_idx = pardec_generate(downsampler, h, context_group_size=K * downsampler_ncodes,
                               output_group_size=downsampler_ncodes, rng=rng if rng is not None else Key(0),
                               greedy=greedy, temperature=temperature, top_k=top_k, rate_id=rate_id,
                               output_expansion=1)
    return dict(code_soft=_one_hot(code_idx, codelm.code_vocab, h.dtype), code_idx=code_idx)


# ---------------------------------------------------------------- decode (refine passes, cycles)
def decode_logits_and_target_multipass(model: "LagCodecModel", level_idx: int, target_seq: torch.Tensor,
                                       ctx_code_soft: torch.Tensor, upsampler_ncodes: int, rng=None,
                                       encode_temperature: float = 1.0, force_teacher_forced: bool = False,
                                       return_passes: bool = False, cycle_ctx: list = None, **_unused):
    codelm = model.codelm_for(level_idx)
    codelm_rate_id = model.codelm_bos_rate_id(level_idx)
    rate_id = model.bos_rate_id(level_idx)
    upsampler = model.upsampler_for(level_idx)
    cfg = model.cfg
    h_ctx = pardec_context_hidden(codelm, upsampler, ctx_code_soft, cfg, codelm_rate_id, rng,
                                  group_size=upsampler_ncodes)
    oe = model.K(level_idx)

    def _teacher_forced(**dkw):
        return pardec_score(upsampler, target_seq, h_ctx, context_group_size=upsampler_ncodes,
                            output_group_size=upsampler_ncodes, rate_id=rate_id, output_expansion=oe, **dkw)

    def _rollout(**dkw):
        hid = pardec_score(upsampler, target_seq, h_ctx, context_group_size=upsampler_ncodes,
                           output_group_size=upsampler_ncodes, rate_id=rate_id, output_expansion=oe,
                           return_hidden=True, **dkw)
        t_out = target_seq[:, :hid.shape[1]]
        qfn = lambda lg_m, k_: quantize_dispatch(cfg.quantize_mode, lg_m, k_ if rng is not None else None,
                                                 encode_temperature, cfg.quantize_drop)
        _, _, lg = token_ar_rollout(upsampler.token_in_proj, upsampler.token_member_embed, upsampler.token_norm1,
                                    upsampler.token_attn, upsampler.token_ln_f, upsampler.token_out_head,
                                    upsampler.output_chunks, hid, rng if rng is not None else Key(0), qfn)
        zero = lg.new_zeros(())
        return lg, t_out, zero, zero

    def _one_pass(**dkw):
        if force_teacher_forced or not cfg.upsampler_rollout:
            return _teacher_forced(**dkw)
        if rng is None or cfg.upsampler_rollout_prob >= 1.0:
            return _rollout(**dkw)
        if cfg.upsampler_rollout_prob <= 0.0:
            return _teacher_forced(**dkw)
        use_roll = bool(rand_bernoulli(rng.fold(9), cfg.upsampler_rollout_prob, (), h_ctx.device))
        return _rollout(**dkw) if use_roll else _teacher_forced(**dkw)

    n_pass = cfg.level_refine_passes[level_idx]
    Pp = cfg.level_refine_window[level_idx] * upsampler_ncodes * oe
    fixed = n_pass > 1 and cfg.level_refine_layout == "fixed"
    fill = "mask" if fixed else "zero"
    ckw = dict(cycle_ctx=cycle_ctx) if cycle_ctx else {}
    passes = [_one_pass(draft_len=Pp, draft_fill="mask", **ckw) if fixed else _one_pass(**ckw)]
    for p in range(1, n_pass):
        prev_logits, prev_target = passes[-1][0], passes[-1][1]
        if cfg.level_refine_draft_mode == "sample" and rng is not None:
            lg = prev_logits.float() / cfg.level_refine_draft_temperature
            lg = lg + rand_gumbel(rng.fold(20 + p), lg.shape, lg.device)
        else:
            lg = prev_logits
        draft = safe_argmax(lg)
        if cfg.level_refine_gt_drop < 1.0 and rng is not None:
            own = rand_bernoulli(rng.fold(10 + p), cfg.level_refine_gt_drop, draft.shape[:2], draft.device)
            draft = torch.where(own[..., None], draft, prev_target.long())
        passes.append(_one_pass(draft_seq=draft.detach(), draft_len=Pp, draft_fill=fill, **ckw))
    outs = [(lg, t, None, al, ac) for lg, t, al, ac in passes]
    return outs if return_passes else outs[-1]


@torch.no_grad()
def _decode_generate_pardec_call(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                 draft_seq=None, draft_len=0, draft_fill="zero", cycle_ctx=None, rng=None):
    codelm = model.codelm_for(level_idx)
    rng = Key(seed) if rng is None else rng
    h_ctx = pardec_context_hidden(codelm, model.upsampler_for(level_idx), ctx_idx, model.cfg,
                                  model.codelm_bos_rate_id(level_idx), rng, group_size=upsampler_ncodes)
    ckw = dict(cycle_ctx=cycle_ctx) if cycle_ctx else {}
    return pardec_generate(model.upsampler_for(level_idx), h_ctx, context_group_size=upsampler_ncodes,
                           output_group_size=upsampler_ncodes, rng=rng, greedy=greedy, temperature=temperature,
                           top_k=model.cfg.gen_top_k, rate_id=model.bos_rate_id(level_idx),
                           output_expansion=model.K(level_idx), draft_seq=draft_seq, draft_len=draft_len,
                           draft_fill=draft_fill, **ckw)


def decode_generate_multipass(model: "LagCodecModel", level_idx: int, ctx_idx: torch.Tensor, upsampler_ncodes: int,
                              greedy: bool = True, temperature: float = 1.0, seed: int = 0, cycle_ctx: list = None,
                              rng=None) -> torch.Tensor:
    n_pass = model.cfg.level_refine_passes[level_idx]
    Pp = model.cfg.level_refine_window[level_idx] * upsampler_ncodes * model.K(level_idx)
    fixed = n_pass > 1 and model.cfg.level_refine_layout == "fixed"
    fill = "mask" if fixed else "zero"
    ckw = dict(cycle_ctx=cycle_ctx, rng=rng)
    if fixed:
        pred = _decode_generate_pardec_call(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                            None, Pp, "mask", **ckw)
    else:
        pred = _decode_generate_pardec_call(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                            **ckw)
    for _ in range(n_pass - 1):
        pred = _decode_generate_pardec_call(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                            pred, Pp, fill, **ckw)
    return pred


def cycle_reencode(model: "LagCodecModel", level_idx: int, tokens: torch.Tensor, rng=None,
                   encode_temperature: float = 1.0) -> tuple:
    # downsampler i's code for level-i tokens, self-fed (never label_fn/GT), digits per quantize_mode
    cfg = model.cfg
    codelm, ds = model.codelm_for(level_idx), model.downsampler_for(level_idx)
    K = model.K(level_idx)
    h = pardec_context_hidden(codelm, ds, tokens, cfg, model.codelm_bos_rate_id(level_idx), rng, group_size=K)
    n_blocks = h.shape[1] // K
    hid = pardec_score(ds, h.new_zeros((h.shape[0], n_blocks, ds.output_chunks), dtype=torch.long), h,
                       context_group_size=K, output_group_size=1, rate_id=model.bos_rate_id(level_idx),
                       return_hidden=True)
    qfn = lambda lg, k_: quantize_dispatch(cfg.quantize_mode, lg, k_, encode_temperature, cfg.quantize_drop)
    if ds.token_head == "linear":
        return qfn(reshape_pq(hid @ ds.output_head_linear, ds.output_chunks, ds.output_vocab), rng)
    if rng is None:
        qfn = lambda lg, k_: quantize_dispatch(cfg.quantize_mode, lg, None, encode_temperature, cfg.quantize_drop)
    cs, ci, _ = token_ar_rollout(ds.token_in_proj, ds.token_member_embed, ds.token_norm1, ds.token_attn,
                                 ds.token_ln_f, ds.token_out_head, ds.output_chunks, hid,
                                 Key(0) if rng is None else rng, qfn)
    return cs, ci


def decode_logits_and_target_cycles(model: "LagCodecModel", level_idx: int, target_seq: torch.Tensor,
                                    ctx_code_soft: torch.Tensor, upsampler_ncodes: int, rng=None,
                                    encode_temperature: float = 1.0, force_teacher_forced: bool = False) -> list:
    # level_cycles: one multipass passes-list per cycle; between cycles the decoded tokens
    # (level_cycle_input) are re-encoded into this level's code c(t+1)
    cfg = model.cfg
    n_cyc = cfg.level_cycles[level_idx]
    kw = dict(encode_temperature=encode_temperature, force_teacher_forced=force_teacher_forced, return_passes=True)
    if n_cyc <= 1:
        return [decode_logits_and_target_multipass(model, level_idx, target_seq, ctx_code_soft, upsampler_ncodes,
                                                   rng=rng, **kw)]
    stack = cfg.level_cycle_mode == "stack"
    slots = [None] * (n_cyc - 1) if stack else None
    ctx_t, cycles = ctx_code_soft, []
    for t in range(n_cyc):
        rng_t = rng if (t == 0 or rng is None) else rng.fold(100 + t)
        passes = decode_logits_and_target_multipass(model, level_idx, target_seq, ctx_t, upsampler_ncodes,
                                                    rng=rng_t, cycle_ctx=slots, **kw)
        cycles.append(passes)
        if t == n_cyc - 1:
            break
        sub = (lambda s: None) if rng_t is None else rng_t.fold
        if cfg.level_cycle_input == "gt":
            tokens = target_seq
        elif cfg.level_cycle_input == "pss":
            tokens = safe_argmax(passes[-1][0])
            if cfg.level_cycle_pss_prob < 1.0 and rng_t is not None:
                own = rand_bernoulli(sub(31), cfg.level_cycle_pss_prob, tokens.shape[:2], tokens.device)
                tokens = torch.where(own[..., None], tokens, passes[-1][1].long())
        else:  # rollout: free-run decode under this cycle's exact conditioning, as at generation
            greedy = rng_t is None or cfg.quantize_mode == "argmax"
            tokens = decode_generate_multipass(model, level_idx, ctx_t.detach(), upsampler_ncodes, greedy=greedy,
                                               temperature=1.0,
                                               cycle_ctx=None if slots is None else [s if s is None else s.detach()
                                                                                    for s in slots],
                                               rng=sub(30))
        rev_soft, _ = cycle_reencode(model, level_idx, tokens.detach(), sub(32), encode_temperature)
        if cfg.level_cycle_detach:
            rev_soft = rev_soft.detach()
        if stack:
            slots = list(slots)
            slots[t] = pardec_context_hidden(model.codelm_for(level_idx), model.upsampler_for(level_idx), rev_soft,
                                             cfg, model.codelm_bos_rate_id(level_idx), rng_t,
                                             group_size=upsampler_ncodes)
        else:
            ctx_t = rev_soft
    return cycles


@torch.no_grad()
def _cycle_reencode_generate(model, level_idx, tokens, upsampler_ncodes, greedy, temperature, seed, stack):
    codelm = model.codelm_for(level_idx)
    out = encode_pardec_downsampler_generate(codelm, model.downsampler_for(level_idx), tokens, model.K(level_idx),
                                             model.cfg, rate_id=model.bos_rate_id(level_idx), rng=Key(seed),
                                             greedy=greedy, temperature=temperature, top_k=model.cfg.gen_top_k,
                                             downsampler_ncodes=1, codelm_rate_id=model.codelm_bos_rate_id(level_idx))
    if not stack:
        return out["code_idx"], None
    h = pardec_context_hidden(codelm, model.upsampler_for(level_idx), out["code_idx"], model.cfg,
                              model.codelm_bos_rate_id(level_idx), None, group_size=upsampler_ncodes)
    return out["code_idx"], h


@torch.no_grad()
def decode_generate_cycles(model: "LagCodecModel", level_idx: int, ctx_idx: torch.Tensor, upsampler_ncodes: int,
                           greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> torch.Tensor:
    cfg = model.cfg
    n_cyc = cfg.gen_level_cycles[level_idx]
    if n_cyc <= 1:
        return decode_generate_multipass(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed)
    stack = cfg.level_cycle_mode == "stack"
    slots = [None] * cycle_stack_slots(cfg, level_idx) if stack else None
    ctx_t = ctx_idx
    for t in range(n_cyc):
        pred = decode_generate_multipass(model, level_idx, ctx_t, upsampler_ncodes, greedy, temperature,
                                         seed if t == 0 else seed + 100 + t, cycle_ctx=slots)
        if t == n_cyc - 1:
            return pred
        code_idx, h = _cycle_reencode_generate(model, level_idx, pred, upsampler_ncodes, greedy, temperature,
                                               seed + 1000 * (t + 1), stack)
        if stack:
            slots = list(slots)
            slots[t] = h
        else:
            ctx_t = code_idx


# ---------------------------------------------------------------- CodeLM free-run / prompted generation
def encoder_hidden(codelm: CodeLM, x: torch.Tensor) -> torch.Tensor:
    h = x
    for blk in codelm.blocks:
        h = run_block(blk, h, False)
    return codelm.ln_f(h)


def codelm_sample_next(codelm: CodeLM, h_prev: torch.Tensor, rng, greedy: bool, temperature, top_k: int = 0):
    if codelm.token_head == "linear":
        return _sample_tokens(reshape_pq(h_prev @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab),
                              rng, greedy, temperature, top_k)
    idx, _ = token_ar_generate(codelm.tok_in_proj, codelm.tok_member_embed, codelm.tok_norm1, codelm.tok_attn,
                               codelm.tok_ln_f, codelm.tok_out_head, codelm.pq_chunks, h_prev, rng, greedy,
                               temperature, top_k)
    return idx


def _sample_tokens(logits: torch.Tensor, rng, greedy: bool, temperature, top_k: int = 0) -> torch.Tensor:
    if greedy:
        return safe_argmax(logits)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        kth = torch.topk(lg, top_k, dim=-1).values[..., -1:]
        lg = torch.where(lg < kth, torch.full_like(lg, -float("inf")), lg)
    return safe_argmax(lg + rand_gumbel(rng, lg.shape, lg.device))


@torch.no_grad()
def encoder_free_run(codelm: CodeLM, prompt_tokens: torch.Tensor, total_len: int, K: int, rng, greedy: bool = False,
                     temperature: float = 1.0, top_k: int = 0, use_bos: bool = False, rate_id: int = 0) -> torch.Tensor:
    # keep the prompt, sample the rest with CodeLM's NTP head (KV cache / recurrent state)
    assert codelm.attn_lookahead == 0, "encoder free-run needs attn_lookahead=0"
    assert not use_bos or codelm.use_codelm_bos, "use_bos=True needs cfg.use_codelm_bos=True"
    B, P, C = prompt_tokens.shape
    assert 1 <= P <= total_len
    dev = prompt_tokens.device
    tokens = torch.zeros(B, total_len, C, dtype=prompt_tokens.dtype, device=dev)
    tokens[:, :P] = prompt_tokens
    caches = [block_cache_init(blk, B, total_len, dev) for blk in codelm.blocks]

    def self_step(x, pos):
        for i, blk in enumerate(codelm.blocks):
            x, caches[i] = block_step(blk, x, caches[i], pos, total_len)
        return codelm.ln_f(x)

    x = code_embed_proj(tokens[:, :P], codelm.own_input_embed, codelm.own_input_proj)
    if use_bos:
        x[:, 0] = codelm.bos_embed[rate_id]
    h = None
    for pos in range(P):
        h = self_step(x[:, pos], pos)
    for t in range(P, total_len):
        tok = codelm_sample_next(codelm, h, rng.fold(t), greedy, temperature, top_k)
        tokens[:, t] = tok.to(tokens.dtype)
        h = self_step(code_embed_proj(tok, codelm.own_input_embed, codelm.own_input_proj), t)
    return tokens


def generate_from_prompt(model: "LagCodecModel", cfg: "Config", prompt_bytes: torch.Tensor, total_positions: int,
                         sample_level: int, rng, greedy: bool = False, temperature: float = 1.0, top_k: int = 0,
                         encode_temperature: float = 1.0, decode_greedy: bool = True, decode_temperature: float = 1.0,
                         decode_seed: int = 0, byte_pq_fn=None) -> dict:
    n = len(cfg.strides)
    assert n - 2 <= sample_level <= n - 1
    codelm0 = model.codelm_for(0)
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    B, P, _ = prompt_bytes.shape
    assert P % model.K(0) == 0
    tok = byte_pq_fn(prompt_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    for i in range(sample_level):
        tok = encode_pardec_downsampler_generate(model.codelm_for(i), model.downsampler_for(i), tok, model.K(i), cfg,
                                                 rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
                                                 rng=rng.fold(100000 + i), greedy=greedy, temperature=encode_temperature,
                                                 downsampler_ncodes=cfg.downsampler_ncodes[i])["code_idx"]
    ds = 1
    for i in range(sample_level):
        ds *= model.K(i)
    use_bos = cfg.use_codelm_bos and cfg.codelm_bos_prob >= 1.0
    tokens_L = encoder_free_run(model.codelm_for(sample_level), tok, total_positions // ds, model.K(sample_level), rng,
                                greedy, temperature, top_k, use_bos=use_bos,
                                rate_id=model.codelm_bos_rate_id(sample_level))
    codes, raw = {}, tokens_L
    for i in range(sample_level, n):
        out = encode_pardec_downsampler_generate(model.codelm_for(i), model.downsampler_for(i), raw, model.K(i), cfg,
                                                 rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
                                                 rng=rng.fold(200000 + i), greedy=greedy, temperature=encode_temperature,
                                                 downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes[i] = out["code_idx"]
        if i < n - 1:
            raw = out["code_soft"]

    def cascade(cur, from_level):
        for i in range(from_level, -1, -1):
            cur = decode_generate_cycles(model, i, cur, cfg.upsampler_ncodes[i], greedy=decode_greedy,
                                         temperature=decode_temperature, seed=decode_seed)
        return cur

    res = dict(sampled_tokens=tokens_L, emitted_codes=codes, emitted=cascade(codes[n - 1], n - 1))
    res["sampled"] = tokens_L if sample_level == 0 else cascade(tokens_L, sample_level - 1)
    return res


# ---------------------------------------------------------------- model + losses
class LagCodecModel(nn.Module):
    # share_across_levels=True: one CodeLM/Downsampler/Upsampler for every level (rate_id rows);
    # False: one of each per level
    def __init__(self, cfg: "Config"):
        super().__init__()
        self.cfg = cfg
        n = len(cfg.strides)
        n_instances = 1 if cfg.share_across_levels else n
        codelms, downsamplers, upsamplers = [], [], []
        for j in range(n_instances):
            li = j
            codelms.append(CodeLM(cfg, level_idx=li))
            D_enc = cfg.codelm_d_model[li]
            pq_dim, code_vocab, pq_chunks = cfg.pq_dim[li], cfg.code_vocab[li], cfg.pq_chunks[li]
            n_rates = bos_n_rates(cfg) if cfg.share_across_levels else 1
            common = dict(context_hidden_dim=D_enc, mlp_mult=cfg.mlp_mult[li], rope_base=cfg.rope_base[li],
                          output_vocab=code_vocab, output_chunks=pq_chunks, pq_dim=pq_dim,
                          token_dim=cfg.token_dim[li], token_n_heads=cfg.token_n_heads[li], n_rates=n_rates,
                          init_scheme=cfg.init_scheme, use_xsa=cfg.use_xsa, use_qknorm=cfg.use_qknorm,
                          ctx_vocab=code_vocab, ctx_pq_chunks=pq_chunks, ctx_pq_dim=pq_dim,
                          token_head=cfg.pardec_token_head, state_dim=cfg.ssm_state_dim)
            ds_remat = cfg.remat if cfg.downsampler_remat[li] is None else cfg.downsampler_remat[li]
            downsampler = PardecLM(hidden_dim=cfg.downsampler_d_model[li], n_heads=cfg.downsampler_n_heads[li],
                                   n_kv_heads=cfg.downsampler_n_kv_heads[li], n_layers=cfg.downsampler_n_layers[li],
                                   output_expansion=1, context_window_groups=cfg.downsampler_window[li],
                                   decode_past=cfg.downsampler_decode_past[li],
                                   decode_future=cfg.downsampler_decode_future[li], remat=ds_remat,
                                   backbone=cfg.downsampler_backbone[li], **common)
            downsamplers.append(downsampler)
            up_remat = cfg.remat if cfg.upsampler_remat[li] is None else cfg.upsampler_remat[li]
            K_default = cfg.strides[li] if cfg.strides[li] != -1 else 1
            share_lm = cfg.share_downsampler_upsampler_lm
            shared_emb = cfg.context_source == "shared_embed"
            upsampler = PardecLM(hidden_dim=cfg.upsampler_d_model[li], n_heads=cfg.upsampler_n_heads[li],
                                 n_kv_heads=cfg.upsampler_n_kv_heads[li], n_layers=cfg.upsampler_n_layers[li],
                                 output_expansion=K_default, context_window_groups=cfg.upsampler_window[li],
                                 decode_past=cfg.upsampler_decode_past[li],
                                 decode_future=cfg.upsampler_decode_future[li], remat=up_remat,
                                 shared_blocks=downsampler.blocks if share_lm else None,
                                 shared_ln_f=downsampler.ln_f if share_lm else None,
                                 shared_ctx_embed=downsampler.own_ctx_embed if shared_emb else None,
                                 shared_ctx_proj=downsampler.own_ctx_proj if shared_emb else None,
                                 backbone=cfg.upsampler_backbone[li],
                                 cycle_slots=max(cycle_stack_slots(cfg, i) for i in (range(n) if cfg.share_across_levels else [j])),
                                 **common)
            upsamplers.append(upsampler)
        self.codelms = nn.ModuleList(codelms)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.upsamplers = nn.ModuleList(upsamplers)

    def codelm_for(self, level_idx: int) -> CodeLM:
        return self.codelms[0] if self.cfg.share_across_levels else self.codelms[level_idx]

    def downsampler_for(self, level_idx: int) -> PardecLM:
        return self.downsamplers[0] if self.cfg.share_across_levels else self.downsamplers[level_idx]

    def upsampler_for(self, level_idx: int) -> PardecLM:
        return self.upsamplers[0] if self.cfg.share_across_levels else self.upsamplers[level_idx]

    def bos_rate_id(self, level_idx: int) -> int:
        return bos_rate_map(self.cfg)[level_idx] if self.cfg.share_across_levels else 0

    def codelm_bos_rate_id(self, level_idx: int) -> int:
        return level_idx if self.cfg.share_across_levels else 0

    def K(self, level: int) -> int:
        return self.cfg.strides[level] if self.cfg.strides[level] != -1 else 1


def dec_loss_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> tuple:
    nll = -torch.log_softmax(logits.float(), -1).gather(-1, target[..., None].long())[..., 0]
    correct = (logits.argmax(-1) == target).float()
    if mask is None:
        return nll.mean(), correct.mean()
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (nll * m).sum() / denom, (correct * m).sum() / denom


def cycle_loss(cycles: list, mode: str) -> torch.Tensor:
    # mean over refine passes, then over cycles ("all") or the last cycle only
    per = [torch.stack([dec_loss_acc(lg, t, m)[0] for lg, t, m, _, _ in passes]).mean() for passes in cycles]
    return per[-1] if (mode == "last" or len(per) == 1) else torch.stack(per).mean()


def level_forward(model: LagCodecModel, flat_bytes: torch.Tensor, phase: int, rng=None, level_gt_drop=None,
                  cascade_rng=None, encode_temperature: float = 1.0, layer_drop_prob=None,
                  label_reg_weight: float = 0.0, label_fn=None, pixel_order=None, byte_pq_fn=None,
                  digit_teacher_force: bool = False, return_recon: bool = False, ctx_ablation: str = None) -> tuple:
    cfg = model.cfg
    codelm0 = model.codelm_for(0)
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    tok0 = byte_pq_fn(flat_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses = [], [], [], [], []
    label_mses, label_mse_losses = [], []
    level_rngs = [None] * (2 * phase) if rng is None else rng.split(2 * phase)
    for i in range(phase):
        out = encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, target, flat_bytes, cfg,
                                        pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                        codelm_rate_id=model.codelm_bos_rate_id(i), rng=level_rngs[2 * i],
                                        downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        enc_losses.append(out["ntp_loss"])
        enc_accs.append(out["ntp_acc"])
        utils.append(out["util"])
        entropy_losses.append(out["entropy_loss"])
        if label_reg_weight > 0:
            enc_logits = out["logits"]
            label_tgt = label_fn(flat_bytes, cfg, pixel_order, enc_logits.shape[1], cfg.pq_chunks[i], cfg.code_vocab[i])
            logp_i = torch.log_softmax(enc_logits.float(), -1)
            label_losses.append(-logp_i.gather(-1, label_tgt[..., None]).mean())
            label_mses.append(((enc_logits.argmax(-1).float() - label_tgt.float()) ** 2).mean())
            probs_i = torch.softmax(enc_logits.float(), -1)
            soft = (probs_i * torch.arange(probs_i.shape[-1], device=probs_i.device, dtype=probs_i.dtype)).sum(-1)
            label_mse_losses.append(((soft - label_tgt.float()) ** 2).mean())
        if i < phase - 1:
            raw, target = out["code_soft"], out["code_idx"]

    dec_losses, dec_accs, aux_ntp_losses, aux_ntp_accs = [], [], [], []
    aux_applies = any(u.decode_future > 0 or u.decode_past > 0 for u in model.upsamplers)
    byte_mse, mse_loss = None, 0.0
    ctx = codes_soft[phase - 1]
    if ctx_ablation == "zero":
        ctx = torch.zeros_like(ctx)
    elif ctx_ablation == "shuffle":
        ctx = torch.roll(ctx, shifts=1, dims=0)
    elif ctx_ablation is not None:
        raise ValueError(f"ctx_ablation must be None/'zero'/'shuffle', got {ctx_ablation!r}")
    if cfg.ctx_stop_gradient is True:
        ctx = ctx.detach()
    sg_pseudo = (lambda c: c.detach()) if cfg.ctx_stop_gradient == "pseudo" else (lambda c: c)
    cascade_rngs = [None] * phase if cascade_rng is None else cascade_rng.split(phase)
    pred_bytes = None
    for i in range(phase - 1, -1, -1):
        dec_target = tok0 if i == 0 else codes[i - 1]
        cycles = decode_logits_and_target_cycles(model, i, dec_target, ctx, cfg.upsampler_ncodes[i],
                                                 rng=level_rngs[2 * i + 1], encode_temperature=encode_temperature,
                                                 force_teacher_forced=digit_teacher_force)
        logits, target_i, mask_i, aux_loss_i, aux_acc_i = cycles[-1][-1]
        dec_losses.append(cycle_loss(cycles, cfg.level_cycle_loss))
        dec_accs.append(dec_loss_acc(logits, target_i, mask_i)[1])
        if aux_applies:
            aux_ntp_losses.append(aux_loss_i)
            aux_ntp_accs.append(aux_acc_i)
        if i == 0:
            pred_bytes = logits.argmax(-1).float()
            byte_mse = ((pred_bytes - target_i.float()) ** 2).mean()
            if cfg.mse_weight > 0:
                byte_probs = torch.softmax(logits.float() / cfg.mse_softmax_tau, -1)
                values = torch.arange(byte_probs.shape[-1], device=byte_probs.device, dtype=byte_probs.dtype)
                pred_pixel = (byte_probs * values).sum(-1)
                mse_loss = (((pred_pixel - target_i.float()) / (byte_probs.shape[-1] - 1)) ** 2).mean()
        if i > 0:
            real_ctx = codes_soft[i - 1]
            rng_i = cascade_rngs[i]
            gt_drop_i = None if level_gt_drop is None else (
                level_gt_drop if isinstance(level_gt_drop, (int, float)) else level_gt_drop[i])
            if gt_drop_i is None or gt_drop_i == 0.0 or (gt_drop_i < 1.0 and rng_i is None):
                ctx = real_ctx
            else:
                pseudo_ctx, _ = quantize_dispatch(cfg.quantize_mode, logits, None if rng_i is None else rng_i.fold(0),
                                                  encode_temperature, cfg.quantize_drop)
                pseudo_ctx = sg_pseudo(pseudo_ctx)
                if gt_drop_i == 1.0:
                    ctx = pseudo_ctx
                else:
                    use = rand_bernoulli(rng_i, gt_drop_i, (), logits.device)
                    ctx = torch.where(use, pseudo_ctx, real_ctx)
            if cfg.ctx_stop_gradient is True:
                ctx = ctx.detach()

    zero = dec_losses[0].new_zeros(())
    dec_loss_total = torch.stack(dec_losses).mean()
    ntp_loss_total = torch.stack(enc_losses).mean()
    entropy_loss_total = torch.stack(entropy_losses).mean()
    label_loss_total = torch.stack(label_losses).mean() if label_losses else zero
    label_mse_total = torch.stack(label_mses).mean() if label_mses else zero
    label_mse_loss_total = torch.stack(label_mse_losses).mean() if label_mse_losses else zero
    aux_ntp_loss_total = torch.stack(aux_ntp_losses).mean() if aux_ntp_losses else zero
    aux_ntp_acc_total = torch.stack(aux_ntp_accs).mean() if aux_ntp_accs else zero
    loss = dec_loss_total + cfg.ntp_weight * ntp_loss_total + cfg.entropy_weight * entropy_loss_total \
        + cfg.mse_weight * mse_loss + label_reg_weight * label_loss_total \
        + cfg.ntp_weight * aux_ntp_loss_total + cfg.label_mse_weight * label_mse_loss_total
    aux = (dec_loss_total, dec_accs[-1], ntp_loss_total, torch.stack(enc_accs).mean(), torch.stack(utils).mean(),
           byte_mse, aux_ntp_loss_total / math.log(2.0), aux_ntp_acc_total, label_mse_total)
    if return_recon:
        aux = aux + (pred_bytes,)
    return loss, aux


def sample_multires_entry(py_rng, n_levels: int) -> tuple:
    entry_level = py_rng.randrange(0, n_levels - 1)
    return entry_level, py_rng.randrange(1, n_levels - entry_level)


def sample_level_range(py_rng, probs: tuple) -> tuple:
    # (entry_level, depth): independent start-walk and end-walk over the same probs (see the JAX docstring)
    n = len(probs) + 1
    s = n - 1
    for i in range(n - 1):
        if py_rng.random() < probs[i]:
            s = i
            break
    e = n - 1
    for j in range(s, n - 1):
        if py_rng.random() < probs[j]:
            e = j
            break
    return s, e - s + 1


def _encode_chain_upto(model, flat_bytes, upto_level, cfg, label_fn, pixel_order, byte_pq_fn, rng):
    codelm0 = model.codelm_for(0)
    tok0 = byte_pq_fn(flat_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    level_rngs = [None] * upto_level if rng is None else rng.split(upto_level)
    code_idx = code_soft = None
    for i in range(upto_level):
        out = encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, target, flat_bytes, cfg,
                                        pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                        codelm_rate_id=model.codelm_bos_rate_id(i), rng=level_rngs[i],
                                        downsampler_ncodes=cfg.downsampler_ncodes[i])
        code_idx, code_soft = out["code_idx"], out["code_soft"]
        raw, target = code_soft, code_idx
    return code_idx, code_soft


def level_forward_multires(model: LagCodecModel, flat_bytes: torch.Tensor, entry_level: int, depth: int, rng=None,
                           encode_temperature: float = 1.0, label_reg_weight: float = 0.0, label_fn=None,
                           pixel_order=None, byte_pq_fn=None, entry_gt_drop: float = None) -> tuple:
    cfg = model.cfg
    codelm_entry = model.codelm_for(entry_level)
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    if entry_level == 0:
        entry_code = byte_pq_fn(flat_bytes, codelm_entry.pq_chunks, codelm_entry.code_vocab)
        raw, target = entry_code, entry_code
    else:
        n_blocks_entry = n_blocks_for_level(cfg, entry_level - 1)
        label_shortcut = label_fn(flat_bytes, cfg, pixel_order, n_blocks_entry, cfg.pq_chunks[entry_level - 1],
                                  cfg.code_vocab[entry_level - 1])
        if entry_gt_drop is not None and rng is not None:
            rng, chain_rng, blend_rng = rng.split(3)
            chain_idx, chain_soft = _encode_chain_upto(model, flat_bytes, entry_level, cfg, label_fn, pixel_order,
                                                       byte_pq_fn, chain_rng)
            chain_idx, chain_soft = chain_idx.detach(), chain_soft.detach()
            use_real = rand_bernoulli(blend_rng, entry_gt_drop, (flat_bytes.shape[0],), flat_bytes.device)
            label_soft = _one_hot(label_shortcut, cfg.code_vocab[entry_level - 1], chain_soft.dtype)
            target = torch.where(use_real.reshape((-1,) + (1,) * (label_shortcut.ndim - 1)), chain_idx, label_shortcut)
            raw = torch.where(use_real.reshape((-1,) + (1,) * (label_soft.ndim - 1)), chain_soft, label_soft)
        else:
            raw, target = label_shortcut, label_shortcut
    entry_code = target
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses, label_mses = [], [], [], [], [], []
    level_rngs = [None] * (2 * depth) if rng is None else rng.split(2 * depth)
    for d in range(depth):
        i = entry_level + d
        out = encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, target, flat_bytes, cfg,
                                        pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                        codelm_rate_id=model.codelm_bos_rate_id(i), rng=level_rngs[2 * d],
                                        downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        enc_losses.append(out["ntp_loss"])
        enc_accs.append(out["ntp_acc"])
        utils.append(out["util"])
        entropy_losses.append(out["entropy_loss"])
        if label_reg_weight > 0:
            enc_logits = out["logits"]
            label_tgt = label_fn(flat_bytes, cfg, pixel_order, enc_logits.shape[1], cfg.pq_chunks[i], cfg.code_vocab[i])
            label_losses.append(-torch.log_softmax(enc_logits.float(), -1).gather(-1, label_tgt[..., None]).mean())
            label_mses.append(((enc_logits.argmax(-1).float() - label_tgt.float()) ** 2).mean())
        if d < depth - 1:
            raw, target = out["code_soft"], out["code_idx"]
    dec_losses, dec_accs = [], []
    ctx = codes_soft[depth - 1]
    dec_rngs = [None] * depth if rng is None else rng.fold(2).split(depth)
    for d in range(depth - 1, -1, -1):
        i = entry_level + d
        cycles = decode_logits_and_target_cycles(model, i, entry_code if d == 0 else codes[d - 1], ctx,
                                                 cfg.upsampler_ncodes[i], rng=dec_rngs[d],
                                                 encode_temperature=encode_temperature)
        logits, target_i, mask_i, _, _ = cycles[-1][-1]
        dec_losses.append(cycle_loss(cycles, cfg.level_cycle_loss))
        dec_accs.append(dec_loss_acc(logits, target_i, mask_i)[1])
        if d > 0:
            ctx = codes_soft[d - 1]
    zero = dec_losses[0].new_zeros(())
    dec_loss_total = torch.stack(dec_losses).mean()
    ntp_loss_total = torch.stack(enc_losses).mean()
    label_loss_total = torch.stack(label_losses).mean() if label_losses else zero
    label_mse_total = torch.stack(label_mses).mean() if label_mses else zero
    loss = dec_loss_total + cfg.ntp_weight * ntp_loss_total + cfg.entropy_weight * torch.stack(entropy_losses).mean() \
        + label_reg_weight * label_loss_total
    return loss, (dec_loss_total / math.log(2.0), dec_accs[-1], ntp_loss_total / math.log(2.0),
                  torch.stack(enc_accs).mean(), torch.stack(utils).mean(), zero, zero, zero, label_mse_total)


def phase_trainable_modules(model: LagCodecModel, phase: int) -> list:
    # curriculum_mode="freeze": phase p trains only level p-1's codelm/downsampler/upsampler
    if model.cfg.curriculum_mode != "freeze":
        return [model]
    j = phase - 1
    return [model.codelms[j], model.downsamplers[j], model.upsamplers[j]]


def trainable_named_params(model: LagCodecModel, phase: int) -> list:
    keep = {id(p) for m in phase_trainable_modules(model, phase) for p in m.parameters()}
    return [(n, p) for n, p in model.named_parameters() if id(p) in keep]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pixel_mse(gen: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((gen.astype(np.float64) - gt.astype(np.float64)) ** 2))


@torch.no_grad()
def plot_encoder_outs(model: LagCodecModel, cfg: "Config", imgs: np.ndarray, pixel_order: np.ndarray, path: Path,
                      level: int = 0, label_fn=None, device="cpu"):
    # GT | label_fn target downsample | pred code read as RGB (pq_chunks=3, code_vocab=256 only)
    if cfg.pq_chunks[level] != 3 or cfg.code_vocab[level] != 256:
        return None
    flat_raw = torch.as_tensor(images_to_positions(imgs, cfg, pixel_order), device=device).long()
    tok0 = rgb_byte_pq_fn(flat_raw, model.codelm_for(0).pq_chunks, model.codelm_for(0).code_vocab)
    flat, raw = tok0, tok0
    for i in range(level):
        out_i = encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, flat, flat_raw, cfg,
                                          pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                          codelm_rate_id=model.codelm_bos_rate_id(i),
                                          downsampler_ncodes=cfg.downsampler_ncodes[i])
        flat, raw = out_i["code_idx"], out_i["code_soft"]
    out = encode_pardec_downsampler(model.codelm_for(level), model.downsampler_for(level), raw, flat, flat_raw, cfg,
                                    pixel_order, label_fn, model.K(level), rate_id=model.bos_rate_id(level),
                                    codelm_rate_id=model.codelm_bos_rate_id(level),
                                    downsampler_ncodes=cfg.downsampler_ncodes[level])
    code_idx = out["code_idx"].cpu().numpy()
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

    panels, titles = [imgs.astype(np.uint8)], ["ground truth"]
    label_mse = None
    if label_fn is not None:
        label_tgt = label_fn(flat_raw, cfg, pixel_order, n_blocks, cfg.pq_chunks[level], cfg.code_vocab[level])
        label_tgt = label_tgt.cpu().numpy() if torch.is_tensor(label_tgt) else np.asarray(label_tgt)
        panels.append(to_grid(label_tgt))
        titles.append("target downsample")
        label_mse = float(np.mean((code_idx.astype(np.float64) - label_tgt.astype(np.float64)) ** 2))
    panels.append(to_grid(code_idx))
    titles.append("pred downsample")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(M, len(panels), figsize=(2 * len(panels), 2 * M))
    axes = np.asarray(axes).reshape(M, len(panels))
    for i in range(M):
        for j, (panel, title) in enumerate(zip(panels, titles)):
            axes[i, j].imshow(panel[i])
            axes[i, j].set_title(title if i == 0 else "", fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(f"level{level} encoder outs -- util={util:.3f}" +
                 (f"  label_mse={label_mse:.2f}" if label_mse is not None else ""), fontsize=10)
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
                  "downsampler_n_kv_heads", "downsampler_window",
                  "downsampler_decode_past", "downsampler_decode_future", "downsampler_remat", "downsampler_ncodes",
                  "upsampler_d_model", "upsampler_n_layers", "upsampler_n_heads",
                  "upsampler_n_kv_heads", "upsampler_window",
                  "upsampler_decode_past", "upsampler_decode_future", "upsampler_remat",
                  "share_across_levels", "bos_rate_mode", "context_source", "pardec_token_head", "codelm_token_head", "downsampler_rollout",
                  "downsampler_rollout_prob", "upsampler_rollout", "upsampler_rollout_prob",
                  "share_downsampler_upsampler_lm", "strides",
                  "code_vocab", "pq_chunks", "mlp_mult", "rope_base", "ntp_weight", "upsampler_ncodes",
                  "sync", "gen_temperature", "gen_top_k", "gen_eval_encode_mode", "gen_eval_greedy_only",
                  "gen_eval_all_levels", "gen_eval_teacher_force_sanity", "ctx_stop_gradient",
                  "level_refine_passes", "level_refine_window", "level_refine_gt_drop", "level_refine_layout",
                  "level_refine_draft_mode", "level_refine_draft_temperature",
                  "level_cycles", "level_cycle_mode", "level_cycle_input", "level_cycle_pss_prob",
                  "level_cycle_detach", "level_cycle_loss", "gen_level_cycles",
                  "codelm_backbone", "downsampler_backbone", "upsampler_backbone", "ssm_state_dim",
                  "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_at_inference", "init_scheme", "use_xsa",
                  "use_qknorm", "remat", "remat_level", "attn_window", "attn_lookahead",
                  "encoder_attn_window", "decoder_attn_window", "use_sink",
                  "use_codelm_bos", "codelm_bos_prob", "codelm_bos_rates",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal", "label_reg_weight", "label_mse_weight")

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        assert torch.cuda.is_available(), "--device cuda but torch.cuda.is_available() is False"
    return torch.device(name)


def make_lr_schedule(kind: str, peak_lr: float, warmup_steps: int, total_steps: int = None,
                     end_value: float = 0.0, decay_steps: int = None):
    # same values as eqx_common.make_lr_schedule (optax warmup_const / warmup_cosine_decay)
    if kind == "const":
        return lambda step: min(1.0, (step + 1) / max(warmup_steps, 1)) * peak_lr
    assert kind == "cosine", f"unknown lr_schedule {kind!r}"
    ds = decay_steps if decay_steps is not None else (total_steps - warmup_steps)
    ds = max(ds, warmup_steps + 1)

    def sched(step):
        if step < warmup_steps:
            return peak_lr * step / warmup_steps
        t = min(step - warmup_steps, ds - warmup_steps)
        return end_value + (peak_lr - end_value) * 0.5 * (1 + math.cos(math.pi * t / (ds - warmup_steps)))
    return sched


def sr_sinkhorn(g: torch.Tensor, iterations: int = 2) -> torch.Tensor:
    n, m = g.shape
    x = g
    for _ in range(iterations):
        x = math.sqrt(n) * x / (x.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
        x = math.sqrt(m) * x / (x.pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    return x


class Optimizer:
    # adamw (optax defaults: wd 1e-4) or sinkgd (stateless Sinkhorn SGD for 2D non-embedding weights,
    # AdamW for the rest), global-norm clip like optax.clip_by_global_norm
    def __init__(self, name: str, named_params: list, kwargs: dict, grad_clip: float = None):
        kw = dict(kwargs)
        self.grad_clip = grad_clip
        self.params = [p for _, p in named_params]
        b1, b2, eps = kw.pop("b1", 0.9), kw.pop("b2", 0.999), kw.pop("eps", 1e-8)
        wd = kw.pop("weight_decay", 1e-4 if name == "adamw" else 0.0)
        if name == "sinkgd":
            self.linear_lr_scale = kw.pop("linear_lr_scale", 0.05)
            self.sinkhorn_iters = kw.pop("sinkhorn_iters", 2)
            is_sink = lambda n, p: p.ndim == 2 and not any(k in n.lower() for k in ("embed", "bootstrap"))
            self.sink = [p for n, p in named_params if is_sink(n, p)]
            adam = [p for n, p in named_params if not is_sink(n, p)]
        else:
            self.sink, adam = [], self.params
        assert not kw, f"unused optimizer_kwargs {kw}"
        self.adam = torch.optim.AdamW(adam, lr=0.0, betas=(b1, b2), eps=eps, weight_decay=wd) if adam else None

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr: float) -> float:
        grads = [p.grad for p in self.params if p.grad is not None]
        norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads)) if grads else torch.zeros(())
        if self.grad_clip is not None and grads:
            scale = torch.clamp(self.grad_clip / norm.clamp_min(1e-30), max=1.0)
            for g in grads:
                g.mul_(scale.to(g.dtype))
        with torch.no_grad():
            for p in self.sink:
                if p.grad is not None:
                    p.add_(sr_sinkhorn(p.grad, self.sinkhorn_iters), alpha=-lr * self.linear_lr_scale)
        if self.adam is not None:
            for gr in self.adam.param_groups:
                gr["lr"] = lr
            self.adam.step()
        return float(norm)

    def state_dict(self) -> dict:
        return {"adam": self.adam.state_dict() if self.adam is not None else None}

    def load_state_dict(self, sd: dict):
        if self.adam is not None and sd.get("adam") is not None:
            self.adam.load_state_dict(sd["adam"])


def save_checkpoint(ckpt_dir: Path, model, opt: Optimizer, train_iter: BatchIterator, phase: int, phase_step: int,
                    step: int, seed: int, schedule_meta: dict = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(opt.state_dict(), ckpt_dir / "opt_state.pt")
    (ckpt_dir / "dataloader_state.json").write_text(json.dumps(dict(
        epoch_rng_state=train_iter.epoch_rng.bit_generator.state, epoch_seed=train_iter.epoch_seed,
        pos=train_iter.pos)))
    meta = dict(phase=phase, phase_step=phase_step, step=step, seed=seed)
    if schedule_meta is not None:
        meta["schedule"] = schedule_meta
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))


def _ckpt_candidates(run_dir: Path) -> list:
    root = run_dir / "checkpoints"
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        if d.name != "wa" and (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            out.append((meta["phase"], meta["phase_step"], d))
    return sorted(out, key=lambda t: (t[0], t[1]))


def find_latest_checkpoint(run_dir: Path):
    c = _ckpt_candidates(run_dir)
    return c[-1][2] if c else None


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    if keep is None:
        return
    c = _ckpt_candidates(run_dir)
    for _, _, d in (c[:-keep] if keep > 0 else c):
        shutil.rmtree(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet64", "imagenet256"],
                    help="cifar (default): downloads/caches under --data_root. imagenetN: reads "
                         "pre-built shards from --data_root (scripts/imagenet/download_imagenetN.py; "
                         "does not download itself). Config.img_size must match (32 cifar, N imagenetN). "
                         "TODO(modality-generalization): 'text'/'audio' choices + a modality: str "
                         "Config field bundling {load_fn, pixel_order_fn, label_fn/byte_pq_fn "
                         "defaults} -- see load_text/load_audio/bpe_label_fn/resample_label_fn stubs "
                         "below rgb_label_fn_jax (not wired in yet).")
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
    p.add_argument("--require_staged_curriculum", type=lambda x: x.lower() != "false", default=False,
                    help="raise instead of warn when level_steps/level_epochs has a later phase with "
                         "nonzero steps while an earlier phase was skipped (0) -- i.e. refuse to jump "
                         "straight to joint multi-level training without staging lower levels first")
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
    p.add_argument("--downsampler_decode_past", type=_tuple_arg, default=Config.downsampler_decode_past,
                    help="downsampler PardecLM's OWN decode_past (independent of decode_past/"
                         "upsampler_decode_past). Default 0 (previously always 0, unconfigurable)")
    p.add_argument("--downsampler_decode_future", type=_tuple_arg, default=Config.downsampler_decode_future,
                    help="downsampler PardecLM's OWN decode_future, independent of the upsampler's. "
                         "Default 0 (previously always 0, unconfigurable)")
    p.add_argument("--downsampler_remat", type=_opt_bool_tuple_arg, default=Config.downsampler_remat,
                    help="downsampler PardecLM's OWN remat, independent of the upsampler's. "
                         "'none' (default) falls back to --remat (previous shared behavior)")
    p.add_argument("--downsampler_ncodes", type=_tuple_arg, default=Config.downsampler_ncodes,
                    help="batching granularity for the downsampler's own pardec_score/pardec_generate "
                         "calls: output_group_size=downsampler_ncodes, context_group_size=K*"
                         "downsampler_ncodes. Default 1 (one code at a time, K raw positions -> 1 "
                         "code); e.g. 4 groups 4 codes' worth of raw input (4*K positions) into one "
                         "batched call producing 4 codes together.")
    p.add_argument("--upsampler_d_model", type=_tuple_arg, default=Config.upsampler_d_model)
    p.add_argument("--upsampler_n_layers", type=_tuple_arg, default=Config.upsampler_n_layers)
    p.add_argument("--upsampler_n_heads", type=_tuple_arg, default=Config.upsampler_n_heads)
    p.add_argument("--upsampler_n_kv_heads", type=_tuple_arg, default=Config.upsampler_n_kv_heads)
    p.add_argument("--upsampler_window", type=_tuple_arg, default=Config.upsampler_window,
                    help="upsampler PardecLM's context_window_groups, in GROUPS of upsampler_ncodes "
                         "codes -- must stay bounded (not -1), same OOM lesson as downsampler_window")
    p.add_argument("--upsampler_decode_past", type=_tuple_arg, default=Config.upsampler_decode_past,
                    help="upsampler PardecLM's OWN decode_past, independent of the downsampler's own "
                         "downsampler_decode_past. Default 0")
    p.add_argument("--upsampler_decode_future", type=_tuple_arg, default=Config.upsampler_decode_future,
                    help="upsampler PardecLM's OWN decode_future, independent of the downsampler's own "
                         "downsampler_decode_future. Default 0")
    p.add_argument("--upsampler_remat", type=_opt_bool_tuple_arg, default=Config.upsampler_remat,
                    help="upsampler PardecLM's OWN remat, independent of the downsampler's. "
                         "'none' (default) falls back to --remat (previous shared behavior)")
    p.add_argument("--share_across_levels", type=lambda x: x.lower() != "false",
                    default=Config.share_across_levels,
                    help="True (default): one shared CodeLM/Downsampler/Upsampler for the whole "
                         "model, differentiated per level via rate_id/bos_embed (true JAX weight "
                         "tying). False: one fully independent CodeLM/Downsampler/Upsampler PER "
                         "LEVEL (own weights, may use different architecture per level).")
    p.add_argument("--bos_rate_mode", type=str, default=Config.bos_rate_mode, choices=("relative", "absolute"),
                    help="Only matters when share_across_levels=True. 'relative' (default): rate_id "
                         "keys off each level's EFFECTIVE stride value, so levels sharing the same "
                         "stride share the same bos row (e.g. strides=(4,4,4) -> n_rates=1). "
                         "'absolute': rate_id is the raw level index, n_rates=n always (old behavior).")
    p.add_argument("--context_source", type=str, default=Config.context_source,
                    choices=("codelm", "own_embed", "shared_embed"),
                    help="How downsampler/upsampler get their context. 'codelm' (default): CodeLM's "
                         "own contextualized hidden states (encoder_hidden). 'own_embed': a plain "
                         "per-position embedding table, no self-attention, own table per module. "
                         "'shared_embed': same, but downsampler and upsampler share one table.")
    p.add_argument("--pardec_token_head", type=str, default=Config.pardec_token_head, choices=("ar", "linear"),
                    help="'ar' (default): AR digit head. 'linear': one parallel linear head for all digits "
                         "(incompatible with --downsampler_rollout).")
    p.add_argument("--codelm_token_head", type=str, default=Config.codelm_token_head, choices=("linear", "ar"),
                    help="CodeLM's own NTP/free-run head: 'linear' (default, parallel digits) or 'ar' "
                         "(small autoregressive digit head).")
    p.add_argument("--downsampler_rollout", type=lambda x: x.lower() != "false",
                    default=Config.downsampler_rollout,
                    help="True: downsampler self-feeds its own (straight-through) digits instead of being "
                         "teacher-forced on label_fn digits. Warns when False and label_reg_weight==0.")
    p.add_argument("--downsampler_rollout_prob", type=float, default=Config.downsampler_rollout_prob,
                    help="probability per step of using the rollout when --downsampler_rollout is on "
                         "(warns if !=1 and label_reg_weight==0).")
    p.add_argument("--upsampler_rollout", type=lambda x: x.lower() != "false",
                    default=Config.upsampler_rollout,
                    help="True: upsampler's digit-AR head self-feeds (token_ar_rollout) during training "
                         "instead of being teacher-forced, while the decode loss still scores against the "
                         "real target -- trains the digit head the way pardec_generate actually uses it.")
    p.add_argument("--upsampler_rollout_prob", type=float, default=Config.upsampler_rollout_prob,
                    help="probability per decode step of using the rollout when --upsampler_rollout is on.")
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
    p.add_argument("--upsampler_ncodes", type=_tuple_arg, default=Config.upsampler_ncodes)
    p.add_argument("--sync", type=_bool_tuple_arg, default=Config.sync,
                    help="stub (TODO), not implemented -- raises NotImplementedError if set True")
    p.add_argument("--gen_temperature", type=float, default=Config.gen_temperature,
                    help="temperature of the sampled (non-argmax) generation eval / decoder sampling")
    p.add_argument("--gen_top_k", type=int, default=Config.gen_top_k,
                    help="top-k of decoder sampling (0 = off); only used when sampling, never for argmax")
    p.add_argument("--gen_eval_encode_mode", type=str, default=Config.gen_eval_encode_mode,
                    choices=["generate", "teacher_force"],
                    help="gen-eval forward/encode direction: 'generate' (default) self-generates each "
                         "level's code via the downsampler's own AR head; 'teacher_force' uses the real "
                         "image's label_fn target instead")
    p.add_argument("--gen_eval_greedy_only", type=lambda x: x.lower() != "false", default=Config.gen_eval_greedy_only,
                    help="skip the sampled (temperature/top_k) half of gen-eval, greedy decode only")
    p.add_argument("--gen_eval_all_levels", type=lambda x: x.lower() != "false", default=Config.gen_eval_all_levels,
                    help="mid-phase/end-of-phase gen-eval also loops top=0..phase-2 (not just phase-1)")
    p.add_argument("--gen_eval_teacher_force_sanity", type=lambda x: x.lower() != "false",
                    default=Config.gen_eval_teacher_force_sanity,
                    help="alongside normal gen-eval, run a sanity reconstruction with every digit-level "
                         "AR step forced teacher-forced (cross-level ctx still honors level_gt_drop), on "
                         "train and val -- saves samples_{tag}_tfsanity.png")
    p.add_argument("--ctx_stop_gradient",
                    type=lambda x: "pseudo" if x.lower() == "pseudo" else x.lower() != "false",
                    default=Config.ctx_stop_gradient,
                    help="True: detach every decode-cascade ctx. 'pseudo': detach only the predicted ctx "
                         "(upper upsampler's output), keep gradient on the real encoder ctx. False: no detach")
    p.add_argument("--level_refine_passes", type=_tuple_arg, default=Config.level_refine_passes,
                    help="per level: 1 = off. >1: extra upsampler passes, each seeing a draft (previous pass's "
                         "argmax) of level_refine_window preceding groups")
    p.add_argument("--level_refine_window", type=_tuple_arg, default=Config.level_refine_window,
                    help="per level: number of preceding groups whose draft each refine pass sees")
    p.add_argument("--level_refine_gt_drop", type=float, default=Config.level_refine_gt_drop,
                    help="training only: per draft position, prob of drafting own prediction (else real target)")
    p.add_argument("--level_refine_layout", type=str, default=Config.level_refine_layout, choices=["fixed", "variable"],
                    help="fixed: same row layout every pass, pass 1's draft slot = learned mask token. "
                         "variable: pass 1 has no draft slot (targets shift by the draft length in later passes)")
    p.add_argument("--level_refine_draft_mode", type=str, default=Config.level_refine_draft_mode,
                    choices=["argmax", "sample"],
                    help="training draft: argmax (matches greedy gen) or sample (gumbel-max, matches sampled gen)")
    p.add_argument("--level_refine_draft_temperature", type=float, default=Config.level_refine_draft_temperature,
                    help="temperature for level_refine_draft_mode=sample")
    p.add_argument("--level_cycles", type=_tuple_arg, default=Config.level_cycles,
                    help="per level: decode cycles (1 = off). Each cycle re-encodes the decoded tokens into this "
                         "level's code with downsampler i and decodes again")
    p.add_argument("--level_cycle_mode", type=str, default=Config.level_cycle_mode, choices=["memoryless", "stack"],
                    help="memoryless: cycle t decodes from c(t) alone (DAE). stack: c(0) + (cycles-1) slots, "
                         "mask token until filled")
    p.add_argument("--level_cycle_input", type=str, default=Config.level_cycle_input,
                    choices=["rollout", "pss", "gt"],
                    help="training tokens re-encoded between cycles: rollout (free-run, exact) | pss (teacher-"
                         "forced argmax, mixed with GT at --level_cycle_pss_prob) | gt (overfit sanity)")
    p.add_argument("--level_cycle_pss_prob", type=float, default=Config.level_cycle_pss_prob,
                    help="pss: per-position prob of own argmax (else GT)")
    p.add_argument("--level_cycle_detach", type=lambda x: x.lower() != "false", default=Config.level_cycle_detach,
                    help="False: cycle loss also trains the re-encoder through quantize_mode's estimator")
    p.add_argument("--level_cycle_loss", type=str, default=Config.level_cycle_loss, choices=["all", "last"])
    p.add_argument("--gen_level_cycles", type=_tuple_arg, default=Config.gen_level_cycles,
                    help="per level: generation cycles (default = level_cycles)")
    for name in ("codelm_backbone", "downsampler_backbone", "upsampler_backbone"):
        # argparse also runs `type` on string defaults: keep a bare value a str so Config broadcasts it
        p.add_argument(f"--{name}", type=lambda s: _str_tuple_arg(s) if "," in s else s.strip(),
                        default=getattr(Config, name), help="per level: transformer | gru | linear_gru | ssm")
    p.add_argument("--ssm_state_dim", type=int, default=Config.ssm_state_dim)
    p.add_argument("--precision", type=str, default=Config.precision, choices=["bf16", "fp32"])
    p.add_argument("--curriculum_mode", type=str, default=Config.curriculum_mode, choices=["freeze", "no_freeze"])
    p.add_argument("--quantize_mode", type=str, default=Config.quantize_mode,
                   choices=["argmax", "gumbel", "zgr", "reinmax_limit"])
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
    p.add_argument("--level_select_prob", type=_float_tuple_arg, default=None,
                    help="If set (length n_levels-1, NOT per-phase -- one flat tuple), enables "
                         "any-level training for every active phase: each step independently samples "
                         "(entry_level, depth) via sample_level_range, reusing this one tuple for both "
                         "the start-walk and the end-walk (see sample_level_range's own docstring for "
                         "why this is sound -- every (entry_level, depth) pair including a single "
                         "level alone at any index, including the top, is reachable by construction), "
                         "instead of level_forward's fixed-phase cascade (level 0 up through "
                         "phase-1). None (default): unchanged fixed-phase behavior.")
    p.add_argument("--multires_entry_gt_drop", type=_float_tuple_arg, default=None,
                    help="Forked from --level_gt_drop, for --level_select_prob's any-level training "
                         "specifically (length n_levels, indexed by entry_level -- index 0 is unused, "
                         "entry_level=0 never needs this). With this probability, entry_level>0's own "
                         "input is the REAL (stop_gradient'd) encoder chain's output instead of "
                         "label_fn's resize-based shortcut -- see level_forward_multires's own "
                         "entry_gt_drop param. None (default): always the label_fn shortcut.")
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
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="torch device; auto = cuda > mps > cpu")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    label_fn_raw = config_vars.pop("label_fn", "default_label_fn_jax")
    label_fn = LABEL_FNS[label_fn_raw] if isinstance(label_fn_raw, str) else label_fn_raw
    known = {a.dest for a in p._actions}
    consts = sorted(k for k in set(config_vars) - known
                    if not callable(config_vars[k]) and not isinstance(config_vars[k], type(argparse)))
    if consts:
        warnings.warn(f"--config {pre_args.config}: ignoring non-field constant(s) {consts}")
    p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = pre_args.config.stem

    def _resolve_pair(step_name, epoch_name, default_step=None):
        s, e = getattr(args, step_name), getattr(args, epoch_name)
        assert s is None or e is None, f"at most one of --{step_name}/--{epoch_name} may be set"
        if s is None and e is None and default_step is not None:
            setattr(args, step_name, default_step)

    _resolve_pair("level_steps", "level_epochs")
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
    if args.multihost or (args.n_devices or 1) > 1:
        warnings.warn("torch port is single-device: ignoring --multihost/--n_devices")
    n_devices = 1
    device = resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} device={device} threads={torch.get_num_threads()}")

    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})
    n_levels = len(cfg.strides)
    n_phases = n_levels if cfg.strides[-1] != -1 else n_levels - 1
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

    for name in ("level_steps", "level_epochs", "batch_size", "val_batch_size", "encode_temperature",
                 "level_gt_drop", "layer_drop_prob"):
        _bcast_per_phase(name)
    _phase_steps = args.level_steps if args.level_steps is not None else args.level_epochs
    skip_ahead = any(any(_phase_steps[j] == 0 for j in range(i)) and _phase_steps[i] != 0
                     for i in range(len(_phase_steps)))
    if skip_ahead:
        msg = f"level_steps/level_epochs={_phase_steps}: a later phase runs while an earlier one was skipped"
        if args.require_staged_curriculum:
            raise ValueError(msg)
        warnings.warn(msg)
    if cfg.curriculum_mode == "freeze":
        assert not args.no_curriculum and args.level_select_prob is None, \
            "curriculum_mode='freeze' needs the phase-by-phase curriculum (no --no_curriculum / level_select_prob)"
        if skip_ahead:
            raise ValueError("curriculum_mode='freeze' with a skipped phase would freeze an untrained level")

    (train_np, train_labels), (val_np, val_labels) = load_dataset(args.dataset, Path(args.data_root), cfg.img_size)
    if args.train_subset_n:
        train_np = train_np[:args.train_subset_n]
    if args.val_subset_n:
        val_np = val_np[:args.val_subset_n]

    torch.manual_seed(args.seed)
    model = LagCodecModel(cfg).to(device)
    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"[torch {device}] n_levels={n_levels} n_phases={n_phases} n_positions={n_positions_of(cfg)} "
           f"params={count_params(model) / 1e6:.2f}M")
    logger(f"resolved_config:{_pretty_dict(_round_floats({k: v for k, v in sorted(vars(args).items()) if k != 'config'}))}")

    resume_meta, resume_ckpt_dir = None, None
    if args.resume:
        resume_ckpt_dir = find_latest_checkpoint(run_dir)
        if resume_ckpt_dir is not None:
            resume_meta = json.loads((resume_ckpt_dir / "meta.json").read_text())
            model.load_state_dict(torch.load(resume_ckpt_dir / "model.pt", map_location=device))
            logger(f"resuming from {resume_ckpt_dir}: phase={resume_meta['phase']} step={resume_meta['step']}")
        else:
            logger("--resume set but no checkpoint found under this run_dir -- starting fresh")

    use_amp = cfg.precision == "bf16" and device.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else (lambda: torch.autocast("cpu", enabled=False))
    if cfg.precision == "bf16" and not use_amp:
        logger(f"precision=bf16 runs fp32 on {device.type} (bf16 autocast only on cuda)")
    to_dev = lambda a: torch.as_tensor(np.asarray(a), device=device).long()
    tag_key = lambda tag: Key(0).fold(int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(), "little"))
    prompts = {}

    @torch.no_grad()
    def run_gen_eval(top: int, tag: str, flat_prompt, gt_img, sample: bool = False):
        model.eval()
        t0 = time.monotonic()
        tag = tag + ("_sample" if sample else "")
        g_kw = dict(greedy=not sample, temperature=cfg.gen_temperature, seed=1 if sample else 0)
        with amp():
            tok0 = rgb_byte_pq_fn(flat_prompt, model.codelm_for(0).pq_chunks, model.codelm_for(0).code_vocab)
            raw, target = tok0, tok0
            codes = []
            if cfg.gen_eval_encode_mode == "generate":
                eval_rngs = tag_key(tag).split(top + 1)
            else:
                eval_rngs = [None] * (top + 1) if not cfg.gumbel_at_inference else tag_key(tag).split(top + 1)
            for i in range(top + 1):
                if cfg.gen_eval_encode_mode == "generate":
                    out = encode_pardec_downsampler_generate(model.codelm_for(i), model.downsampler_for(i), raw,
                                                             model.K(i), cfg, rate_id=model.bos_rate_id(i),
                                                             codelm_rate_id=model.codelm_bos_rate_id(i),
                                                             rng=eval_rngs[i], greedy=not sample,
                                                             temperature=cfg.gen_temperature, top_k=cfg.gen_top_k,
                                                             downsampler_ncodes=cfg.downsampler_ncodes[i])
                else:
                    out = encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, target,
                                                    flat_prompt, cfg, pixel_order, label_fn, model.K(i),
                                                    rate_id=model.bos_rate_id(i),
                                                    codelm_rate_id=model.codelm_bos_rate_id(i), rng=eval_rngs[i],
                                                    downsampler_ncodes=cfg.downsampler_ncodes[i])
                codes.append(out["code_idx"])
                if i < top:
                    raw, target = out["code_soft"], out["code_idx"]
            cur = codes[top]
            for i in range(top, 0, -1):
                cur = decode_generate_cycles(model, i, cur, cfg.upsampler_ncodes[i], **g_kw)
            recon = decode_generate_cycles(model, 0, cur, cfg.upsampler_ncodes[0], **g_kw)
        acc = float((recon == flat_prompt).float().mean())
        img = positions_to_image(recon.cpu().numpy(), cfg, pixel_order)
        mse = pixel_mse(img, gt_img)
        save_compare_grid(img, gt_img, run_dir / f"samples_{tag}.png")
        dt = time.monotonic() - t0
        logger(f"[{tag}] top={top} CASCADE{' (sampled T=%g k=%d)' % (cfg.gen_temperature, cfg.gen_top_k) if sample else ''}"
               f" gen_byte_acc={acc:.4f} gen_cascade_mse={mse:.2f} gen_time={dt:.1f}s",
               tag=tag, gen_cascade_acc=acc, gen_cascade_mse=mse, gen_time_s=dt)
        if args.verbose and img.shape[0] < 10:
            logger(" ".join(f"mse{i + 1}={pixel_mse(img[i:i + 1], gt_img[i:i + 1]):.2f}" for i in range(img.shape[0])))
        model.train()

    def run_gen_eval_both(top: int, tag: str):
        run_gen_eval(top, f"{tag}_val", prompts["val"], prompts["val_gt"])
        if not cfg.gen_eval_greedy_only:
            run_gen_eval(top, f"{tag}_val", prompts["val"], prompts["val_gt"], sample=True)
        if args.eval_gen_train:
            run_gen_eval(top, f"{tag}_train", prompts["train"], prompts["train_gt"])
            if not cfg.gen_eval_greedy_only:
                run_gen_eval(top, f"{tag}_train", prompts["train"], prompts["train_gt"], sample=True)

    @torch.no_grad()
    def run_tf_sanity(top: int, tag: str, flat_prompt, gt_img):
        model.eval()
        phase = top + 1
        with amp():
            _, aux = level_forward(model, flat_prompt, phase, rng=None, level_gt_drop=args.level_gt_drop[phase - 1],
                                   encode_temperature=args.encode_temperature[phase - 1], label_reg_weight=0.0,
                                   label_fn=label_fn, pixel_order=pixel_order, digit_teacher_force=True,
                                   return_recon=True)
        img = positions_to_image(aux[-1].long().cpu().numpy(), cfg, pixel_order)
        mse = pixel_mse(img, gt_img)
        save_compare_grid(img, gt_img, run_dir / f"samples_{tag}_tfsanity.png")
        logger(f"[{tag}] top={top} TF_SANITY tf_sanity_mse={mse:.2f}", tag=tag, tf_sanity_mse=float(mse))
        model.train()

    def run_tf_sanity_both(top: int, tag: str):
        run_tf_sanity(top, f"{tag}_val", prompts["val"], prompts["val_gt"])
        if args.eval_gen_train:
            run_tf_sanity(top, f"{tag}_train", prompts["train"], prompts["train_gt"])

    @torch.no_grad()
    def run_val_eval(phase: int, tag: str):
        model.eval()
        t0 = time.monotonic()
        bs = args.val_batch_size[phase - 1]
        sums, total_loss, total_n = np.zeros(9), 0.0, 0
        for start in range(0, len(val_np), bs):
            imgs = val_np[start:start + bs]
            with amp():
                loss_b, aux_b = level_forward(model, to_dev(images_to_positions(imgs, cfg, pixel_order)), phase,
                                              rng=None, encode_temperature=args.encode_temperature[phase - 1],
                                              label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                              pixel_order=pixel_order)
            sums += len(imgs) * np.array([float(a) for a in aux_b])
            total_loss += len(imgs) * float(loss_b)
            total_n += len(imgs)
        dec_loss, dec_acc, enc_loss, enc_acc, util, val_mse, _, aux_acc, label_mse = (sums / total_n).tolist()
        dt = time.monotonic() - t0
        logger(f"[{tag}] VAL loss={total_loss / total_n:.2f} val_dec_loss={dec_loss:.2f} val_dec_acc={dec_acc:.2f} "
               f"val_mse={val_mse:.4f} val_enc_acc={enc_acc:.2f} val_d_ntp_acc={aux_acc:.2f} "
               f"val_label_mse={label_mse:.2f} val_time={dt:.1f}s",
               tag=tag, val_loss=total_loss / total_n, val_dec_loss=dec_loss, val_dec_acc=dec_acc,
               val_enc_loss=enc_loss, val_enc_acc=enc_acc, val_util=util, val_mse=val_mse, val_d_ntp_acc=aux_acc,
               val_label_mse=label_mse, val_time_s=dt)
        model.train()

    def eval_all(phase: int, tag_fn):
        levels = range(phase - 1, -1, -1) if cfg.gen_eval_all_levels else [phase - 1]
        for lvl in levels:
            run_gen_eval_both(lvl, tag_fn(lvl))
            if cfg.gen_eval_teacher_force_sanity:
                run_tf_sanity_both(lvl, tag_fn(lvl))

    def _phase_total_steps(idx, steps_per_epoch):
        return args.level_steps[idx] if args.level_steps is not None else round(args.level_epochs[idx] * steps_per_epoch)

    def _every(step_val, epoch_val, steps_per_epoch):
        return step_val if step_val is not None else round(epoch_val * steps_per_epoch)

    step = resume_meta["step"] if resume_meta else 0
    phase_iter = [n_phases] if args.no_curriculum else list(range(1, n_phases + 1))
    total_all_steps = sum(_phase_total_steps(ph - 1, len(train_np) // args.batch_size[ph - 1]) for ph in phase_iter)
    step_w = len(str(total_all_steps))
    if resume_meta is not None:
        rp = resume_meta["phase"]
        spe = len(BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size[rp - 1], 1, True, args.seed, cfg))
        done = resume_meta["phase_step"] >= _phase_total_steps(rp - 1, spe)
        phase_iter = [ph for ph in phase_iter if (ph > rp if done else ph >= rp)]
    multires_py_rng = random.Random(args.seed)
    model.train()
    for phase in phase_iter:
        if (args.level_steps is not None and args.level_steps[phase - 1] == 0) or \
                (args.level_epochs is not None and args.level_epochs[phase - 1] == 0):
            logger(f"level{phase - 1}: 0 steps, skipping phase entirely")
            continue
        train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size[phase - 1], 1,
                                   shuffle=True, seed=args.seed, cfg=cfg)
        vb = args.val_batch_size[phase - 1]
        prompts.update(val=to_dev(images_to_positions(val_np[:vb], cfg, pixel_order)), val_gt=val_np[:vb].astype(np.uint8),
                       train=to_dev(images_to_positions(train_np[:vb], cfg, pixel_order)),
                       train_gt=train_np[:vb].astype(np.uint8))
        named = trainable_named_params(model, phase)
        keep = {id(p) for _, p in named}
        for prm in model.parameters():
            prm.requires_grad_(id(prm) in keep)
        steps_per_epoch = len(train_iter)
        phase_total_steps = _phase_total_steps(phase - 1, steps_per_epoch)
        warmup = _every(args.warmup_steps, args.warmup_epochs, steps_per_epoch)
        min_step = (args.lr_min_step if args.lr_min_step is not None else
                    round(args.lr_min_epoch * steps_per_epoch) if args.lr_min_epoch is not None else phase_total_steps)
        lr_decay_steps = max(1, min_step - warmup)
        lr_kind, lr_peak, lr_min_val = args.lr_schedule, args.lr, args.lr_min
        if resume_meta is not None and phase == resume_meta["phase"] and resume_meta.get("schedule") is not None:
            sm = resume_meta["schedule"]
            phase_total_steps, warmup, lr_decay_steps = sm["total_steps"], sm["warmup_steps"], sm["lr_decay_steps"]
            lr_kind, lr_peak, lr_min_val = sm["kind"], sm["lr"], sm["lr_min"]
        schedule_meta = dict(total_steps=phase_total_steps, warmup_steps=warmup, lr_decay_steps=lr_decay_steps,
                             kind=lr_kind, lr=lr_peak, lr_min=lr_min_val)
        lr_schedule = make_lr_schedule(lr_kind, lr_peak, warmup, phase_total_steps, end_value=lr_min_val,
                                       decay_steps=lr_decay_steps)
        opt = Optimizer(args.optimizer, named, args.optimizer_kwargs, args.grad_clip)
        start_phase_step = 0
        if resume_meta is not None and phase == resume_meta["phase"]:
            opt.load_state_dict(torch.load(resume_ckpt_dir / "opt_state.pt", map_location=device))
            dl = json.loads((resume_ckpt_dir / "dataloader_state.json").read_text())
            train_iter.epoch_rng.bit_generator.state = dl["epoch_rng_state"]
            train_iter.epoch_seed, train_iter.pos = dl["epoch_seed"], dl["pos"]
            start_phase_step = resume_meta["phase_step"]
        rng_phase = Key(args.seed).fold(phase)
        multires_active = args.level_select_prob is not None
        active = f"level{phase - 1}" if not multires_active else f"anylevel(phase{phase})"
        logger(f"=== starting {active} for {phase_total_steps / steps_per_epoch:.3g} epochs ({phase_total_steps} steps) ===")
        gen_every = _every(args.gen_eval_every_step, args.gen_eval_every_epoch, steps_per_epoch)
        ckpt_every = _every(args.ckpt_every_step, args.ckpt_every_epoch, steps_per_epoch)
        wa_every = _every(args.wa_every_step, args.wa_every_epoch, steps_per_epoch)
        wa_ema, wa_stack = None, deque(maxlen=args.wa_stack_size)
        wa_dir = run_dir / "checkpoints" / "wa"
        pbar = tqdm(total=phase_total_steps, initial=start_phase_step, desc=active, dynamic_ncols=True)
        phase_step = start_phase_step
        epoch_num = start_phase_step // max(steps_per_epoch, 1)
        t_first = None
        while phase_step < phase_total_steps:
            epoch_num += 1
            if args.epoch_verbose:
                logger(f"{active}: epoch {epoch_num} (step {step})")
            for flat_np in train_iter:
                if phase_step >= phase_total_steps:
                    break
                flat = to_dev(flat_np)
                step_rng = rng_phase.fold(step)
                level_rng, cascade_rng = step_rng.split(2)
                if t_first is None:
                    t_first = time.monotonic()
                with amp():
                    if multires_active:
                        s_lvl, d_lvl = sample_level_range(multires_py_rng, args.level_select_prob)
                        egd = (args.multires_entry_gt_drop[s_lvl]
                               if args.multires_entry_gt_drop is not None and s_lvl > 0 else None)
                        loss, aux = level_forward_multires(model, flat, s_lvl, d_lvl, rng=level_rng,
                                                           encode_temperature=args.encode_temperature[phase - 1],
                                                           label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                                           pixel_order=pixel_order, entry_gt_drop=egd)
                    else:
                        loss, aux = level_forward(model, flat, phase, rng=level_rng,
                                                  level_gt_drop=args.level_gt_drop[phase - 1], cascade_rng=cascade_rng,
                                                  encode_temperature=args.encode_temperature[phase - 1],
                                                  layer_drop_prob=args.layer_drop_prob[phase - 1],
                                                  label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                                  pixel_order=pixel_order)
                opt.zero_grad()
                loss.backward()
                lr = lr_schedule(phase_step)
                grad_norm = opt.step(lr)
                step += 1
                phase_step += 1
                pbar.update(1)
                loss0 = float(loss)
                if step - (resume_meta["step"] if resume_meta else 0) == 1:
                    logger(f"{active}: first train_step took {time.monotonic() - t_first:.1f}s")
                dec_loss, dec_acc, enc_loss, enc_acc, util, train_mse, _, aux_acc, label_mse = \
                    [float(a) if a is not None else 0.0 for a in aux]
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}", acc=f"{dec_acc:.2f}", lr=_fmt_lr(lr),
                                 gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    logger(f"l={phase - 1} e={epoch_num} s={step} loss={loss0:.2f} dec_loss={dec_loss:.2f} "
                           f"dec_acc={dec_acc:.2f} enc_loss={enc_loss:.2f} enc_acc={enc_acc:.2f} util={util:.2f} "
                           f"mse={train_mse:.1f} label_mse={label_mse:.2f} lr={_fmt_lr(lr)} grad_norm={grad_norm:.2f}",
                           level=phase - 1, epoch=epoch_num, step=step, loss=loss0, dec_loss=dec_loss,
                           dec_acc=dec_acc, enc_acc=enc_acc, util=util, mse=train_mse, label_mse=label_mse, lr=lr,
                           grad_norm=grad_norm)
                if step % gen_every == 0:
                    st = f"{step:0{step_w}d}"
                    run_val_eval(phase, f"level{phase - 1}_step{st}")
                    eval_all(phase, lambda lvl: f"level{lvl}_step{st}")
                    try:
                        plot_encoder_outs(model, cfg, val_np[:vb], pixel_order,
                                          run_dir / f"samples_level{phase - 1}_step{st}_codegrid.png",
                                          level=phase - 1, label_fn=label_fn, device=device)
                    except Exception as e:
                        print(e)
                if step % ckpt_every == 0:
                    ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
                    save_checkpoint(ckpt_dir, model, opt, train_iter, phase, phase_step, step, args.seed, schedule_meta)
                    prune_checkpoints(run_dir, args.ckpt_keep)
                    logger(f"checkpoint saved: {ckpt_dir}")
                if args.wa_mode != "none" and step % wa_every == 0:
                    wa_dir.mkdir(parents=True, exist_ok=True)
                    cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    if args.wa_mode == "ema":
                        wa_ema = cur if wa_ema is None else {k: args.wa_ema_decay * wa_ema[k] + (1 - args.wa_ema_decay) * cur[k]
                                                              if torch.is_floating_point(cur[k]) else cur[k] for k in cur}
                        torch.save(wa_ema, wa_dir / "ema_latest.pt")
                    else:
                        wa_stack.append(cur)
                        if len(wa_stack) == args.wa_stack_size:
                            w = ([1.0 / len(wa_stack)] * len(wa_stack) if args.wa_wma_weights is None else
                                 [math.exp(x) / sum(math.exp(y) for y in args.wa_wma_weights) for x in args.wa_wma_weights])
                            avg = {k: sum(wi * s[k] for wi, s in zip(w, wa_stack)) if torch.is_floating_point(cur[k]) else cur[k]
                                   for k in cur}
                            torch.save(avg, wa_dir / "wma_latest.pt")
                    if args.wa_verbose:
                        logger(f"wa ({args.wa_mode}) snapshot saved at step {step}")
        pbar.close()
        logger(f"=== {active} done, {'no freeze' if cfg.curriculum_mode == 'no_freeze' else f'freezing level {phase - 1}'} ===")
        ckpt_dir = run_dir / "checkpoints" / f"phase_{phase}_step{step}"
        save_checkpoint(ckpt_dir, model, opt, train_iter, phase, phase_total_steps, step, args.seed, schedule_meta)
        prune_checkpoints(run_dir, args.ckpt_keep)
        if args.final_eval:
            run_val_eval(phase, f"level{phase - 1}_final")
            eval_all(phase, lambda lvl: f"level{lvl}_final")
    logger("=== all phases done, running final top-down cascade eval ===")
    run_val_eval(n_phases, "final")
    for top in range(n_phases - 1, -1, -1):
        run_gen_eval_both(top, f"final_top{top}")
        if cfg.gen_eval_teacher_force_sanity:
            run_tf_sanity_both(top, f"final_top{top}")
    logger("training done")


if __name__ == "__main__":
    main()
