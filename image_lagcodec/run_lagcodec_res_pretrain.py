"""
run_lagcodec_res_denoise: fork of run_lagcodec_res.py (2026-10-03). Defaults are bit-identical to it.
1. Same-level decode cycles (run_lagcodec's cycle_refine_passes, per level instead of across levels):
   decode level-i tokens from code c(t), re-encode them with downsampler i (quantize_mode) into c(t+1),
   decode again. level_cycle_mode: memoryless (default; cycle t decodes from c(t) alone -- DAE on own
   quantized output) | stack (c(0) always + cycles-1 slots after bos; t0 all slots = mask token, filled
   one per cycle). Re-encoded tokens (level_cycle_input): rollout (free-run, exact, slowest) | pss
   (teacher-forced argmax, detached, mixed with GT at level_cycle_pss_prob) | gt (GT, overfit sanity).
   Fixed vs run_lagcodec: TF-argmax revisions leaked in-group GT; revisions were finer-level codes
   windowed with the coarser level's stride/table; slot 0 was the coarser code in training but the
   decoded code at generation; slots shared one table + rope (order invisible); empty slots dropped.
2. Recurrent backbones (codelm/downsampler/upsampler_backbone): transformer | gru | linear_gru (minGRU)
   | ssm (diagonal selective, Mamba-style). Fixed-size state, no KV cache; invalid keys skip the update.

TODO (multi-res generalization, not implemented -- proposal only, 2026-09-27):
Currently `strides` is a flat tuple, one fixed K per level (e.g. (4,4)). To let each level support
SEVERAL stride options (e.g. level0 in {4,16,32}, level1 in {4,16}) instead of one linear cascade:
  1. strides: tuple[int] -> tuple[tuple[int]], one set of allowed K per level.
  2. No new weight sets needed -- K is already a runtime arg to pardec_score/pardec_generate/
     encode_pardec_downsampler, not baked into weight shapes.
  3. Generalize bos_rate_map's dedup key from level_idx to (level_idx, stride_choice).
  4. Extend sample_multires_entry/level_forward_multires to also sample a stride PER visited level,
     not just which levels are visited.
  5. n_blocks_for_level must depend on the specific sampled stride path, not a fixed cumulative
     product.
  6. generate_from_prompt's cascade must record which K encoded each level and replay the SAME K
     on the matching decode step.
"""
from __future__ import annotations

import argparse
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
# jax.config.update("jax_memory_fitting_level", "O3")
from image_lagcodec.eqx_common import (Attention, Block, RMSNorm, SwiGLU, apply_rope, apply_xsa, init_matrix,
                                        init_vector, make_lr_schedule, rmsnorm, rope_cos_sin,
                                        rope_cos_sin_pos, rotate_half, sinkgd)
from image_lagcodec import run_lagcodec_res as full_runner

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
RECURRENT_BACKBONES = ("gru", "linear_gru", "ssm")
BACKBONES = ("transformer",) + RECURRENT_BACKBONES


MODALITIES = ("image", "text", "audio", "binary")


def total_bytes_of(cfg) -> int:
    if getattr(cfg, "modality", "image") == "image":
        return cfg.img_size * cfg.img_size * 3
    return cfg.seq_len * cfg.byte_group


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
    # data: image (img_size^2 RGB pixels) | text | audio | binary (1D: seq_len positions of byte_group bytes,
    # traversal "raster"). dataset="folder" reads every file under data_root (see load_folder)
    modality: str = "image"
    seq_len: int = 4096
    audio_sample_rate: int = 16000
    audio_encoding: str = "mulaw8"  # mulaw8 (byte_group 1) | pcm16 (byte_group 2: high, low byte)
    # Each of CodeLM/Downsampler/Upsampler is fully independent: its own dedicated fields, own
    # defaults, no fallback chain to a shared "base" field and no override-of-a-generic-field
    # pattern. codelm_* feeds CodeLM only.
    codelm_d_model: tuple = (256, 256, 256, 256)
    codelm_n_layers: tuple = (2, 2, 2, 2)
    codelm_n_heads: tuple = (4, 4, 4, 4)
    codelm_remat: tuple = None  # per level, None falls back to cfg.remat (like downsampler/upsampler_remat)
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
    # rows (groups) never attend to each other: >1 runs the pardec stack over that many row chunks one after
    # another, each rematerialized in backward (peak activations ~1/chunks). 1 = off (default)
    downsampler_remat_chunks: tuple = 1
    upsampler_remat_chunks: tuple = 1
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
    # "codelm_upper": upsampler i's context (code i) is processed by CodeLM i+1, whose own input is code i
    # (downsampler i keeps CodeLM i); share_across_levels=False adds one decoderless CodeLM-only level on top
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
    # Needs downsampler_decode_past/future==0, pardec_token_head="ar". downsampler_ncodes>1: the row is rolled
    # out too, one pass per row code (exact); downsampler_pss_passes>1 caps the passes (approximate).
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
    # upsampler_decode_past/future==0, pardec_token_head="ar". Row TOKENS stay teacher-forced (any
    # upsampler_ncodes); upsampler_pss_passes self-feeds those.
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
    encoder_level_loss_weights: tuple | None = None
    decoder_level_loss_weights: tuple | None = None
    log_levelwise_metrics: bool = False
    log_levelwise_eval: bool = False
    log_levelwise_gen: bool = False
    encoder_only_pretrain: bool = True
    load_encoder_checkpoint: str | None = None

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
    # parallel scheduled sampling on a pardec row's own shifted token inputs (not the context, see level_gt_drop):
    # pass 1 teacher-forces GT, each later pass re-feeds the previous pass's detached prediction; loss = last pass
    upsampler_pss_passes: tuple = 1  # per level; 1 = off, -1 = one pass per row token (exact rollout inputs)
    downsampler_pss_passes: tuple = 1  # same for the downsampler; only matters when downsampler_ncodes > 1.
    # On the downsampler_rollout path: 1/-1 = one pass per row code, >1 = cap (see downsampler_rollout).
    upsampler_pss_prob: float = 1.0  # per-position prob of own prediction, else GT. Eval: always own
    downsampler_pss_prob: float = 1.0
    pss_input_mode: str = "argmax"  # argmax | sample (gumbel-max at pss_temperature; eval: argmax)
    pss_temperature: float = 1.0
    # TODO: CodeLM pss -- its inputs are true codes from the prefill today, only meaningful for free generation
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

        if self.encoder_level_loss_weights is not None:
            if isinstance(self.encoder_level_loss_weights, (int, float)):
                n_encoder_weights = n + (self.context_source == "codelm_upper")
                self.encoder_level_loss_weights = (float(self.encoder_level_loss_weights),) * n_encoder_weights
            else:
                self.encoder_level_loss_weights = tuple(float(x) for x in self.encoder_level_loss_weights)
                n_encoder_weights = n + (self.context_source == "codelm_upper")
                if len(self.encoder_level_loss_weights) != n_encoder_weights:
                    raise ValueError(f"encoder_level_loss_weights needs {n_encoder_weights} values")
        if self.decoder_level_loss_weights is not None:
            if isinstance(self.decoder_level_loss_weights, (int, float)):
                self.decoder_level_loss_weights = (float(self.decoder_level_loss_weights),) * n
            else:
                self.decoder_level_loss_weights = tuple(float(x) for x in self.decoder_level_loss_weights)

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
        bcast_opt("codelm_remat", bool)
        bcast("downsampler_d_model", int)
        bcast("downsampler_n_layers", int)
        bcast("downsampler_n_heads", int)
        bcast("downsampler_n_kv_heads", int)
        bcast("downsampler_window", int)
        bcast("downsampler_decode_past", int)
        bcast("downsampler_decode_future", int)
        bcast("downsampler_ncodes", int)
        bcast_opt("downsampler_remat", bool)
        bcast("downsampler_remat_chunks", int)
        bcast("upsampler_remat_chunks", int)
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
        bcast("upsampler_pss_passes", int)
        bcast("downsampler_pss_passes", int)
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
                "downsampler_remat_chunks", "upsampler_remat_chunks", "codelm_remat",
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
        assert self.modality in MODALITIES, self.modality
        if self.modality == "image":
            assert self.byte_group in (1, 3), "byte_group must be 1 (per-byte) or 3 (per-pixel RGB)"
        else:
            assert self.traversal == "raster", f"modality={self.modality} is 1D: traversal must be 'raster'"
            assert self.byte_group >= 1 and self.seq_len >= 1
            assert self.audio_encoding in ("mulaw8", "pcm16"), self.audio_encoding
            if self.modality == "audio":
                need = 2 if self.audio_encoding == "pcm16" else 1
                assert self.byte_group == need, f"audio_encoding={self.audio_encoding} needs byte_group={need}"
        assert total_bytes_of(self) % self.byte_group == 0
        assert self.traversal in ("raster", "zorder")
        assert self.bos_rate_mode in ("relative", "absolute")
        assert self.context_source in ("codelm", "codelm_upper", "own_embed", "shared_embed")
        assert self.pardec_token_head in ("ar", "linear"), self.pardec_token_head
        assert self.codelm_token_head in ("ar", "linear"), self.codelm_token_head
        assert 0.0 <= self.downsampler_rollout_prob <= 1.0, self.downsampler_rollout_prob
        if self.downsampler_rollout:
            if self.pardec_token_head != "ar":
                raise ValueError("downsampler_rollout is incompatible with pardec_token_head='linear' "
                                 "(the rollout self-feeds digits through the AR token head)")
            assert all(p == 0 for p in self.downsampler_decode_past) \
                and all(f == 0 for f in self.downsampler_decode_future), \
                "downsampler_rollout needs downsampler_decode_past/future==0 (they embed real label tokens)"
        assert 0.0 <= self.upsampler_rollout_prob <= 1.0, self.upsampler_rollout_prob
        if self.upsampler_rollout:
            if self.pardec_token_head != "ar":
                raise ValueError("upsampler_rollout is incompatible with pardec_token_head='linear' "
                                 "(the rollout self-feeds digits through the AR token head)")
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
        assert all(x == -1 or x >= 1 for x in self.upsampler_pss_passes + self.downsampler_pss_passes), "pss_passes: -1 or >=1"
        assert 0.0 <= self.upsampler_pss_prob <= 1.0 and 0.0 <= self.downsampler_pss_prob <= 1.0
        assert self.pss_input_mode in ("argmax", "sample") and self.pss_temperature > 0.0
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
        assert len(self.downsampler_remat_chunks) == n and len(self.upsampler_remat_chunks) == n \
            and len(self.codelm_remat) == n
        assert all(c >= 1 for c in self.downsampler_remat_chunks + self.upsampler_remat_chunks), \
            (self.downsampler_remat_chunks, self.upsampler_remat_chunks)
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
    if cfg.modality != "image":
        return np.arange(n_positions_of(cfg))
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


def load_dataset(name: str, data_root: Path, img_size: int = None, train_shards: int = None, cfg=None) -> tuple:
    # train_shards: only load the first N imagenet train shards (off-training scripts need a few images, not 15GB)
    if name == "folder":
        assert cfg is not None, "dataset='folder' needs the Config (modality/seq_len/...)"
        return load_folder(data_root, cfg)
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
    if cfg.modality != "image":  # samples are already (n, seq_len, byte_group) sequences
        return images.reshape(n, n_positions_of(cfg), cfg.byte_group).astype(np.int32)
    pix = images.reshape(n, cfg.img_size * cfg.img_size, 3)[:, pixel_order, :]
    if cfg.byte_group == 3:
        return pix.astype(np.int32)
    return pix.reshape(n, cfg.img_size * cfg.img_size * 3, 1).astype(np.int32)


def positions_to_image(positions: np.ndarray, cfg: Config, pixel_order: np.ndarray) -> np.ndarray:
    B = positions.shape[0]
    if cfg.modality != "image":
        return positions.reshape(B, n_positions_of(cfg), cfg.byte_group).astype(np.uint8)
    pix_traversal = positions.reshape(B, cfg.img_size * cfg.img_size, 3)
    raster = np.zeros_like(pix_traversal)
    raster[:, pixel_order, :] = pix_traversal
    return raster.reshape(B, cfg.img_size, cfg.img_size, 3).astype(np.uint8)


def mulaw_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    # float waveform in [-1, 1] -> uint8 (8-bit mu-law)
    y = np.sign(x) * np.log1p(mu * np.abs(np.clip(x, -1, 1))) / np.log1p(mu)
    return np.clip(np.round((y + 1) / 2 * mu), 0, mu).astype(np.uint8)


def mulaw_decode(b: np.ndarray, mu: int = 255) -> np.ndarray:
    y = 2 * (b.astype(np.float64) / mu) - 1
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)


def audio_to_bytes(x: np.ndarray, encoding: str) -> np.ndarray:
    # float waveform (T,) -> (T, byte_group) uint8: mulaw8 -> 1 byte, pcm16 -> (high, low) of offset binary
    if encoding == "mulaw8":
        return mulaw_encode(x)[:, None]
    u = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int32) + 32768
    return np.stack([u >> 8, u & 255], axis=-1).astype(np.uint8)


def bytes_to_audio(b: np.ndarray, encoding: str) -> np.ndarray:
    # (..., byte_group) bytes -> float waveform (...)
    if encoding == "mulaw8":
        return mulaw_decode(b[..., 0])
    u = b[..., 0].astype(np.int32) * 256 + b[..., 1].astype(np.int32)
    return ((u - 32768) / 32768.0).astype(np.float32)


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    # band-limited (anti-aliased) resampling: scipy polyphase if available, else windowed-sinc low-pass + interp
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    try:
        from scipy.signal import resample_poly
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        pass
    if sr_out < sr_in:
        fc = 0.5 * sr_out / sr_in
        n = np.arange(-32, 33)
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(len(n))
        x = np.convolve(x, h / h.sum(), mode="same")
    t_out = np.arange(int(len(x) * sr_out / sr_in)) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def load_audio_file(path: Path, sr: int) -> np.ndarray:
    # any format soundfile reads (wav/flac/ogg/...), else stdlib wave (PCM wav) -> mono float32 at `sr`
    try:
        import soundfile
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        import wave
        with wave.open(str(path), "rb") as w:
            rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width])
        data = raw.reshape(-1, ch).astype(np.float32)
        data = (data - 128) / 128 if width == 1 else data / float(2 ** (8 * width - 1))
    return resample_audio(data.mean(axis=1), rate, sr)


def _folder_files(root: Path) -> list:
    return sorted(f for f in root.rglob("*") if f.is_file() and not f.name.startswith("."))


def _folder_split(root: Path) -> tuple:
    # train/ + val/ (or validation/, test/) subfolders if present, else every 20th file is val; fewer than 20
    # files -> (files, None): the concatenated stream is split instead (images: last image reused as val)
    for val_name in ("val", "validation", "test"):
        if (root / "train").is_dir() and (root / val_name).is_dir():
            return _folder_files(root / "train"), _folder_files(root / val_name)
    files = _folder_files(root)
    if len(files) < 20:
        return files, None
    val = files[::20]
    return [f for f in files if f not in set(val)], val


def _seq_windows(stream: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(stream) // seq_len
    return stream[:n * seq_len].reshape(n, seq_len, stream.shape[-1])


def _seq_stream(files: list, cfg) -> np.ndarray:
    # (T, byte_group) uint8: text/binary raw bytes (text files joined by newline), audio decoded + re-encoded
    if cfg.modality == "audio":
        parts = [audio_to_bytes(load_audio_file(f, cfg.audio_sample_rate), cfg.audio_encoding) for f in files]
        return np.concatenate(parts) if parts else np.zeros((0, cfg.byte_group), np.uint8)
    data = (b"\n" if cfg.modality == "text" else b"").join(f.read_bytes() for f in files)
    n = len(data) // cfg.byte_group * cfg.byte_group
    return np.frombuffer(data[:n], dtype=np.uint8).reshape(-1, cfg.byte_group)


def _load_image_file(path: Path, size: int):
    # any PIL-readable image -> RGB, shorter side resized to `size`, center crop; None if unreadable
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def load_folder(data_root: Path, cfg) -> tuple:
    # dataset="folder": every file under data_root, per cfg.modality -> ((train, labels), (val, labels));
    # images (n, img_size, img_size, 3), sequences (n, seq_len, byte_group) non-overlapping windows
    train_files, val_files = _folder_split(data_root)
    if cfg.modality == "image":
        load = lambda fs: np.stack([a for a in (_load_image_file(f, cfg.img_size) for f in fs) if a is not None])
        train = load(train_files)
        val = load(val_files) if val_files else train[-1:]
    elif val_files is None:
        stream = _seq_stream(train_files, cfg)
        n_val = max(cfg.seq_len, len(stream) // 20)
        train, val = _seq_windows(stream[:-n_val], cfg.seq_len), _seq_windows(stream[-n_val:], cfg.seq_len)
    else:
        train = _seq_windows(_seq_stream(train_files, cfg), cfg.seq_len)
        val = _seq_windows(_seq_stream(val_files, cfg), cfg.seq_len)
    assert len(train) and len(val), \
        f"folder dataset {data_root}: no {cfg.modality} samples (train={len(train)}, val={len(val)}, seq_len={cfg.seq_len})"
    return (train, np.zeros(len(train), np.int32)), (val, np.zeros(len(val), np.int32))


def save_samples(gen: np.ndarray, gt: np.ndarray, path: Path, cfg) -> None:
    # image: side-by-side PNG grid; text: one .txt; audio: <stem>_<i>_{gen,gt}.wav; binary: .bin + hex preview
    if cfg.modality == "image":
        return save_compare_grid(gen, gt, path)
    stem = path.with_suffix("")
    if cfg.modality == "audio":
        import wave
        for i in range(gen.shape[0]):
            for name, arr in (("gen", gen[i]), ("gt", gt[i])):
                pcm = (np.clip(bytes_to_audio(arr.astype(np.uint8), cfg.audio_encoding), -1, 1) * 32767).astype(np.int16)
                with wave.open(f"{stem}_{i}_{name}.wav", "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(cfg.audio_sample_rate)
                    w.writeframes(pcm.tobytes())
        return
    lines = []
    for i in range(gen.shape[0]):
        g, t = gen[i].astype(np.uint8).tobytes(), gt[i].astype(np.uint8).tobytes()
        if cfg.modality == "text":
            lines += [f"=== sample {i} generated ===", g.decode("utf-8", "replace"),
                      f"=== sample {i} ground truth ===", t.decode("utf-8", "replace")]
        else:
            Path(f"{stem}_{i}_gen.bin").write_bytes(g)
            lines += [f"=== sample {i} generated (hex, first 256 B) ===", g[:256].hex(" "),
                      f"=== sample {i} ground truth (hex, first 256 B) ===", t[:256].hex(" ")]
    Path(f"{stem}.txt").write_text("\n".join(lines) + "\n")


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


def _block_reduce_jax(x: jnp.ndarray, n_blocks: int, method: str) -> jnp.ndarray:
    # (M, L, C) -> (M, n_blocks, C): mean over each code's stride window, or linear interp at its center
    M, L, C = x.shape
    K = L // n_blocks
    blocks = x[:, :n_blocks * K].reshape(M, n_blocks, K, C)
    if method == "mean":
        return blocks.mean(axis=2)
    return (blocks[:, :, (K - 1) // 2] + blocks[:, :, K // 2]) / 2


def _bytes_to_digits_jax(vals: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    # (M, n, C) byte values -> (M, n, pq_chunks): one digit per channel when they line up (C == pq_chunks,
    # code_vocab 256), else the channel mean bit-packed like default_label_fn_jax
    if vals.shape[-1] == pq_chunks and code_vocab == 256:
        return jnp.round(jnp.clip(vals, 0, 255)).astype(jnp.int32)
    return byte_to_pq_idx_jax(jnp.round(jnp.clip(vals.mean(-1), 0, 255)).astype(jnp.int32), pq_chunks, code_vocab)


def byte_mean_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order, n_blocks: int, pq_chunks: int,
                           code_vocab: int) -> jnp.ndarray:
    # text/binary label: each code's target = mean of the bytes in its stride window
    return _bytes_to_digits_jax(_block_reduce_jax(flat_bytes.astype(jnp.float32), n_blocks, "mean"), pq_chunks,
                                code_vocab)


def byte_interp_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order, n_blocks: int, pq_chunks: int,
                             code_vocab: int) -> jnp.ndarray:
    # text/binary label: bytes linearly interpolated at each stride window's center
    return _bytes_to_digits_jax(_block_reduce_jax(flat_bytes.astype(jnp.float32), n_blocks, "interp"), pq_chunks,
                                code_vocab)


def bytes_to_audio_jax(b: jnp.ndarray, encoding: str) -> jnp.ndarray:
    if encoding == "mulaw8":
        y = 2 * (b[..., 0].astype(jnp.float32) / 255) - 1
        return jnp.sign(y) * jnp.expm1(jnp.abs(y) * jnp.log1p(255.0)) / 255
    return (b[..., 0].astype(jnp.int32) * 256 + b[..., 1].astype(jnp.int32) - 32768).astype(jnp.float32) / 32768.0


def audio_to_bytes_jax(x: jnp.ndarray, encoding: str) -> jnp.ndarray:
    if encoding == "mulaw8":
        y = jnp.sign(x) * jnp.log1p(255 * jnp.abs(jnp.clip(x, -1, 1))) / jnp.log1p(255.0)
        return jnp.clip(jnp.round((y + 1) / 2 * 255), 0, 255)[..., None]
    u = jnp.clip(jnp.round(x * 32767), -32768, 32767).astype(jnp.int32) + 32768
    return jnp.stack([u >> 8, u & 255], axis=-1).astype(jnp.float32)


def sinc_lowpass(factor: int) -> np.ndarray:
    # windowed-sinc FIR, cutoff at the decimated Nyquist (0.5/factor), unit DC gain
    n = np.arange(-4 * factor, 4 * factor + 1)
    h = np.sinc(n / factor) * np.hamming(len(n))
    return (h / h.sum()).astype(np.float32)


def audio_resample_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order, n_blocks: int, pq_chunks: int,
                                code_vocab: int) -> jnp.ndarray:
    # audio label: decode the waveform, anti-alias low-pass + decimate to each code's stride-window center
    # (integer factor = positions per code), re-encode with cfg.audio_encoding
    M, L, _ = flat_bytes.shape
    K = L // n_blocks
    wave_ = bytes_to_audio_jax(flat_bytes, cfg.audio_encoding)
    if K > 1:
        h = jnp.asarray(sinc_lowpass(K))
        wave_ = jax.vmap(lambda w: jnp.convolve(w, h, mode="same"))(wave_)
    centers = wave_[:, :n_blocks * K].reshape(M, n_blocks, K)[:, :, K // 2]
    return _bytes_to_digits_jax(audio_to_bytes_jax(centers, cfg.audio_encoding), pq_chunks, code_vocab)


class HierarchicalBPE:
    # STUB: sentencepiece-style BPE per level (level 0: bytes/letters -> V0, level l: level-(l-1) ids -> V_l);
    # a code's level-l label = the token covering its block center. fit() learns merges, encode() -> ids.
    def __init__(self, vocab_sizes: tuple):
        self.vocab_sizes = tuple(vocab_sizes)
        self.merges = [[] for _ in self.vocab_sizes]  # per level: [(left_id, right_id, new_id), ...]

    def fit(self, stream: np.ndarray) -> "HierarchicalBPE":
        raise NotImplementedError("HierarchicalBPE.fit: stub")

    def encode(self, stream: np.ndarray, level: int) -> np.ndarray:
        raise NotImplementedError("HierarchicalBPE.encode: stub")


def bpe_label_fn_jax(flat_bytes: jnp.ndarray, cfg: Config, pixel_order, n_blocks: int, pq_chunks: int,
                     code_vocab: int, bpe: HierarchicalBPE = None) -> jnp.ndarray:
    # STUB: label = HierarchicalBPE token at each code block's center, bit-packed into (pq_chunks, code_vocab)
    raise NotImplementedError("bpe_label_fn: stub, see HierarchicalBPE")


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


LABEL_FNS = {"default_label_fn_jax": default_label_fn_jax, "rgb_label_fn_jax": rgb_label_fn_jax,
             "default_label_fn_pil": default_label_fn_pil, "byte_mean_label_fn": byte_mean_label_fn_jax,
             "byte_interp_label_fn": byte_interp_label_fn_jax, "audio_resample_label_fn": audio_resample_label_fn_jax,
             "bpe_label_fn": bpe_label_fn_jax}
DEFAULT_LABEL_FN = {"image": "default_label_fn_jax", "text": "byte_mean_label_fn", "binary": "byte_mean_label_fn",
                    "audio": "audio_resample_label_fn"}


def resolve_label_fn(spec, modality: str):
    # config `label_fn`: None (modality default) | a LABEL_FNS name | "package.module:function" | a callable;
    # signature fn(flat_bytes, cfg, pixel_order, n_blocks, pq_chunks, code_vocab) -> (M, n_blocks, pq_chunks)
    if spec is None:
        spec = DEFAULT_LABEL_FN[modality]
    if callable(spec):
        return spec
    if spec in LABEL_FNS:
        return LABEL_FNS[spec]
    assert ":" in spec, f"unknown label_fn {spec!r}: use one of {sorted(LABEL_FNS)} or 'package.module:function'"
    import importlib
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


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


def quantize_zgr(logits: jnp.ndarray, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    # ZGR (Zero-Gumbel-Revised, https://github.com/James-Hooper123/Generalized-and-Optimal-Straight-Through-Estimators):
    # averages the plain straight-through surrogate (dx_ST=p) with a REINFORCE-style term
    # (dx_RE=(y-p)*logpx) -- same (logits, rng, tau, quantize_drop)->(code_soft, idx) contract as
    # quantize_hard/quantize_gumbel, so every call site dispatches identically. idx is drawn via the
    # Gumbel-max trick (a real categorical sample, needed for the RE term to be meaningful) when
    # rng is given, else falls back to the mode (argmax).
    logp = jax.nn.log_softmax(logits, axis=-1)
    p = jnp.exp(logp)
    if rng is not None:
        rng, drop_rng = jax.random.split(rng)
        u = jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
        idx = safe_argmax(logits - jnp.log(-jnp.log(u)))
    else:
        drop_rng = None
        idx = safe_argmax(logp)
    y = jax.nn.one_hot(idx, logits.shape[-1], dtype=logp.dtype)
    dx_ST = p
    logpx = jnp.sum(logp * y, axis=-1, keepdims=True)
    dx_RE = (y - jax.lax.stop_gradient(p)) * logpx
    dx = (dx_ST + dx_RE) / 2
    st = y + (dx - jax.lax.stop_gradient(dx))
    if quantize_drop > 0 and drop_rng is not None:
        drop = jax.random.bernoulli(drop_rng, p=quantize_drop, shape=p.shape[:-1])[..., None]
        code_soft = jnp.where(drop, p, st)
    else:
        code_soft = st
    return code_soft, idx


def quantize_reinmax_limit(logits: jnp.ndarray, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    # reinmax_limit without the (K,K) matrix: S @ logits = logits/(2K) + col*(y.logits)/(2K) - sum(logits)/(2K^2)
    # with col, y stop-gradient -- same forward (y) and Jacobian (S) as quantize_reinmax_limit_dense, O(K) memory
    del tau
    K = logits.shape[-1]
    p = jax.nn.softmax(logits, axis=-1)
    if rng is not None:
        rng, drop_rng = jax.random.split(rng)
        u = jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
        idx = safe_argmax(logits - jnp.log(-jnp.log(u)))
    else:
        drop_rng = None
        idx = safe_argmax(p)
    y = jax.nn.one_hot(idx, K, dtype=p.dtype)
    p_x = jnp.maximum(jnp.sum(p * y, axis=-1, keepdims=True), 1e-8)
    col = jax.lax.stop_gradient((y - p) / p_x)
    dx = (logits + col * jnp.sum(y * logits, axis=-1, keepdims=True)) / (2 * K) \
        - jnp.sum(logits, axis=-1, keepdims=True) / (2 * K * K)
    st = y + (dx - jax.lax.stop_gradient(dx))
    if quantize_drop > 0 and drop_rng is not None:
        drop = jax.random.bernoulli(drop_rng, p=quantize_drop, shape=p.shape[:-1])[..., None]
        code_soft = jnp.where(drop, p, st)
    else:
        code_soft = st
    return code_soft, idx


def quantize_reinmax_limit_dense(logits: jnp.ndarray, rng, tau: float = 1.0, quantize_drop: float = 0.0) -> tuple:
    # slow reference for quantize_reinmax_limit (correctness checks only): materializes S per digit position.
    # reinmax_limit: closed-form asymptotic limit of the MVE estimator (no Cholesky/lstsq solve --
    # see github.com/James-Hooper123/Generalized-and-Optimal-Straight-Through-Estimators). Builds the
    # (K,K) surrogate matrix S directly via a rank-1 correction instead of solving a linear system --
    # O(K^2) flops/memory per position vs exact MVE's O(K^3), still far more than ZGR/ST's O(K). With
    # K=code_vocab=256 this materializes one 256x256 matrix PER DIGIT POSITION (M*n_blocks*pq_chunks
    # of them) -- can be memory-heavy at large n_blocks; prefer ZGR unless this estimator's specific
    # variance properties are needed. `tau` is accepted for signature parity but unused (reinmax_limit
    # has no temperature dependence, per the source).
    del tau
    K = logits.shape[-1]
    p = jax.nn.softmax(logits, axis=-1)
    if rng is not None:
        rng, drop_rng = jax.random.split(rng)
        u = jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0 - 1e-8)
        idx = safe_argmax(logits - jnp.log(-jnp.log(u)))
    else:
        drop_rng = None
        idx = safe_argmax(p)
    y = jax.nn.one_hot(idx, K, dtype=p.dtype)
    p_x = jnp.maximum(jnp.sum(p * y, axis=-1, keepdims=True), 1e-8)
    col = (y - p) / p_x
    rank1 = col[..., :, None] * y[..., None, :]
    eye = jnp.eye(K, dtype=p.dtype)
    S = (eye + rank1) / (2 * K) - jnp.ones((K, K), dtype=p.dtype) / (2 * K * K)
    dx = jnp.einsum('...ij,...j->...i', jax.lax.stop_gradient(S), logits)
    st = y + (dx - jax.lax.stop_gradient(dx))
    if quantize_drop > 0 and drop_rng is not None:
        drop = jax.random.bernoulli(drop_rng, p=quantize_drop, shape=p.shape[:-1])[..., None]
        code_soft = jnp.where(drop, p, st)
    else:
        code_soft = st
    return code_soft, idx


def quantize_dispatch(mode: str, logits: jnp.ndarray, rng, tau: float, quantize_drop: float) -> tuple:
    # single dispatch point for quantize_mode -- mirrors the (logits, rng, tau, quantize_drop) ->
    # (code_soft, idx) contract shared by quantize_hard/quantize_gumbel/quantize_zgr.
    if rng is not None and mode == "gumbel":
        return quantize_gumbel(logits, rng, tau, quantize_drop)
    if rng is not None and mode == "zgr":
        return quantize_zgr(logits, rng, tau, quantize_drop)
    if rng is not None and mode == "reinmax_limit":
        return quantize_reinmax_limit(logits, rng, tau, quantize_drop)
    return quantize_hard(logits, rng, quantize_drop, tau)


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


def group_draft_windows(src: jnp.ndarray, n_groups: int, Kspan: int, Pp: int, valid_len: int) -> tuple:
    # per group g: the Pp tokens of `src` just BEFORE the group (positions g*Kspan-Pp..g*Kspan-1),
    # never the group's own span. Shared by pardec_score and pardec_generate so they can't diverge.
    B = src.shape[0]
    tail = n_groups * Kspan - src.shape[1]
    if tail > 0:
        src = jnp.pad(src, ((0, 0), (0, tail)) + ((0, 0),) * (src.ndim - 2))
    padded = jnp.pad(src, ((0, 0), (Pp, 0)) + ((0, 0),) * (src.ndim - 2))
    win = jnp.stack([padded[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1)
    win = win.reshape(B * n_groups, Pp, *src.shape[2:])
    valid = _draft_past_valid_mask(n_groups, Pp, Kspan, valid_len)
    valid_flat = jnp.broadcast_to(jnp.asarray(valid)[None], (B, n_groups, Pp)).reshape(B * n_groups, Pp)
    return win, valid_flat


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


def remat_row_chunks(fn, rows: jnp.ndarray, rope_pos_ids: jnp.ndarray, key_valid: jnp.ndarray,
                     n_chunks: int) -> jnp.ndarray:
    # pardec rows are independent groups: fn over n_chunks row chunks in sequence (lax.map), each under
    # jax.checkpoint -- backward keeps only chunk inputs. Rows padded to a multiple of n_chunks, then dropped.
    B2 = rows.shape[0]
    per = -(-B2 // n_chunks)
    pad = per * n_chunks - B2
    if pad:
        rows = jnp.pad(rows, ((0, pad), (0, 0), (0, 0)))
        rope_pos_ids = jnp.pad(rope_pos_ids, ((0, pad), (0, 0)))
        key_valid = jnp.pad(key_valid, ((0, pad), (0, 0)), constant_values=True)
    split = lambda a: a.reshape((n_chunks, per) + a.shape[1:])
    out = jax.lax.map(jax.checkpoint(lambda a: fn(*a)), (split(rows), split(rope_pos_ids), split(key_valid)))
    return out.reshape((n_chunks * per,) + out.shape[2:])[:B2]


def run_block_pardec(blk: Block, x: jnp.ndarray, rope_pos_ids: jnp.ndarray, key_valid: jnp.ndarray,
                      remat: bool) -> jnp.ndarray:
    if isinstance(blk, RecurrentBlock):  # order-based, rope ids unused; invalid keys skip the state update
        g = lambda x: blk(x, valid=key_valid)
        return jax.checkpoint(g)(x) if remat else g(x)

    def f(x):
        x = x + dense_self_attention_pardec(blk.attn, blk.norm1(x), rope_pos_ids, key_valid)
        x = x + blk.mlp(blk.norm2(x))
        return x
    return jax.checkpoint(f)(x) if remat else f(x)


def _linear_scan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # h_t = a_t * h_{t-1} + b_t along axis 1, h_{-1} = 0 (parallel associative scan)
    def op(x, y):
        return x[0] * y[0], y[0] * x[1] + y[1]
    return jax.lax.associative_scan(op, (a, b), axis=1)[1]


class RecurrentMixer(eqx.Module):
    # causal fixed-state token mixer replacing attention: gru | linear_gru (minGRU, Feng et al. 2024) |
    # ssm (diagonal selective SSM, Mamba-style, no conv). valid=False positions leave the state untouched.
    w_in: jnp.ndarray
    w_h: jnp.ndarray
    b: jnp.ndarray
    w_dt: jnp.ndarray
    b_dt: jnp.ndarray
    w_bc: jnp.ndarray
    a_log: jnp.ndarray
    d_skip: jnp.ndarray
    out: jnp.ndarray
    kind: str = eqx.field(static=True)

    def __init__(self, key, d_model: int, kind: str, state_dim: int = 16, n_layers: int = None,
                 init_scheme: str = "llama"):
        assert kind in RECURRENT_BACKBONES, kind
        keys = jax.random.split(key, 6)
        D = d_model
        self.kind = kind
        self.w_in = init_matrix(keys[0], (D, (3 if kind == "gru" else 2) * D), init_scheme)
        self.w_h = init_matrix(keys[1], (D, 3 * D), init_scheme) if kind == "gru" else None
        self.b = jnp.zeros(((3 if kind == "gru" else 2) * D,)) if kind != "ssm" else None
        if kind == "ssm":
            self.w_dt = init_matrix(keys[2], (D, D), init_scheme)
            dt = jnp.exp(jax.random.uniform(keys[3], (D,), minval=math.log(1e-3), maxval=math.log(1e-1)))
            self.b_dt = dt + jnp.log(-jnp.expm1(-dt))  # softplus^-1(dt), dt log-uniform in [1e-3, 1e-1]
            self.w_bc = init_matrix(keys[4], (D, 2 * state_dim), init_scheme)
            self.a_log = jnp.log(jnp.broadcast_to(jnp.arange(1, state_dim + 1, dtype=jnp.float32), (D, state_dim)))
            self.d_skip = jnp.ones((D,))
        else:
            self.w_dt = self.b_dt = self.w_bc = self.a_log = self.d_skip = None
        self.out = init_matrix(keys[5], (D, D), init_scheme, residual_out=True, n_layers=n_layers)

    def init_state(self, batch: int) -> jnp.ndarray:
        D = self.w_in.shape[0]
        shape = (batch, D, self.a_log.shape[1]) if self.kind == "ssm" else (batch, D)
        return jnp.zeros(shape, self.w_in.dtype)

    def _gates(self, x: jnp.ndarray) -> tuple:
        # per-position input projections (no state dependence); x (..., D)
        D = x.shape[-1]
        g = x @ self.w_in.astype(x.dtype)
        if self.kind == "gru":
            return (g + self.b.astype(x.dtype),)
        if self.kind == "linear_gru":
            g = g + self.b.astype(x.dtype)
            z = jax.nn.sigmoid(g[..., :D])
            return 1 - z, z * g[..., D:]
        u, gate = g[..., :D], g[..., D:]
        delta = jax.nn.softplus(u @ self.w_dt.astype(x.dtype) + self.b_dt.astype(x.dtype))
        bc = u @ self.w_bc.astype(x.dtype)
        n = self.a_log.shape[1]
        a = jnp.exp(delta[..., None] * -jnp.exp(self.a_log.astype(x.dtype)))
        bx = (delta * u)[..., None] * bc[..., None, :n]
        return a, bx, bc[..., n:], u, gate

    def _gru_cell(self, h: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        D = h.shape[-1]
        gh = h @ self.w_h.astype(h.dtype)
        z = jax.nn.sigmoid(g[..., :D] + gh[..., :D])
        r = jax.nn.sigmoid(g[..., D:2 * D] + gh[..., D:2 * D])
        n = jnp.tanh(g[..., 2 * D:] + r * gh[..., 2 * D:])
        return (1 - z) * n + z * h

    def _readout(self, h: jnp.ndarray, gates: tuple) -> jnp.ndarray:
        if self.kind != "ssm":
            return h
        _, _, c, u, gate = gates
        y = jnp.einsum("...dn,...n->...d", h, c) + self.d_skip.astype(u.dtype) * u
        return y * jax.nn.silu(gate)

    def __call__(self, x: jnp.ndarray, valid: jnp.ndarray = None) -> jnp.ndarray:
        # x (B,T,D), valid (B,T) bool or None
        v = jnp.ones(x.shape[:2], dtype=bool) if valid is None else valid
        gates = self._gates(x)
        if self.kind == "gru":
            def step(h, inp):
                g_t, v_t = inp
                h = jnp.where(v_t[:, None], self._gru_cell(h, g_t), h)
                return h, h
            h0 = jnp.zeros((x.shape[0], x.shape[-1]), x.dtype)
            _, hs = jax.lax.scan(step, h0, (jnp.swapaxes(gates[0], 0, 1), jnp.swapaxes(v, 0, 1)))
            h = jnp.swapaxes(hs, 0, 1)
        else:
            a, bx = gates[0], gates[1]
            vb = v.reshape(v.shape + (1,) * (a.ndim - 2))
            h = _linear_scan(jnp.where(vb, a, 1.0), jnp.where(vb, bx, 0.0))
        return self._readout(h, gates) @ self.out.astype(x.dtype)

    def step(self, x: jnp.ndarray, state: jnp.ndarray, valid: jnp.ndarray = None) -> tuple:
        # one position: x (B,D) -> (y (B,D), state)
        gates = self._gates(x)
        st = state.astype(x.dtype)
        h = self._gru_cell(st, gates[0]) if self.kind == "gru" else gates[0] * st + gates[1]
        if valid is not None:
            h = jnp.where(valid.reshape(valid.shape + (1,) * (h.ndim - 1)), h, st)
        return self._readout(h, gates) @ self.out.astype(x.dtype), h.astype(state.dtype)


class RecurrentBlock(eqx.Module):
    # Block-compatible residual block: recurrent mixer in place of attention, same norms + SwiGLU
    norm1: RMSNorm
    mixer: RecurrentMixer
    norm2: RMSNorm
    mlp: SwiGLU

    def __init__(self, key, d_model: int, kind: str, mlp_mult: int, state_dim: int = 16, n_layers: int = None,
                 init_scheme: str = "llama"):
        k1, k2 = jax.random.split(key, 2)
        self.norm1 = RMSNorm(d_model)
        self.mixer = RecurrentMixer(k1, d_model, kind, state_dim, n_layers=n_layers, init_scheme=init_scheme)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k2, d_model, mlp_mult, n_layers=n_layers, init_scheme=init_scheme)

    def __call__(self, x: jnp.ndarray, causal: bool = True, valid: jnp.ndarray = None) -> jnp.ndarray:
        assert causal, "recurrent backbones are causal only"
        x = x + self.mixer(self.norm1(x), valid)
        return x + self.mlp(self.norm2(x))

    def step(self, x_new: jnp.ndarray, state: jnp.ndarray, valid: jnp.ndarray = None) -> tuple:
        y, state = self.mixer.step(self.norm1(x_new), state, valid)
        x = x_new + y
        return x + self.mlp(self.norm2(x)), state


def make_block(key, backbone: str, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float,
               n_layers: int = None, init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True,
               window: int = None, lookahead: int = 0, use_sink: bool = False, state_dim: int = 16):
    if backbone == "transformer":
        return Block(key, d_model, n_heads, n_kv_heads, mlp_mult, rope_base, n_layers=n_layers,
                     init_scheme=init_scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, window=window,
                     lookahead=lookahead, use_sink=use_sink)
    return RecurrentBlock(key, d_model, backbone, mlp_mult, state_dim, n_layers=n_layers, init_scheme=init_scheme)


def block_cache_init(blk, batch: int, T_max: int):
    # generation cache: (k, v) for attention, fixed-size state for recurrent (no KV cache)
    if isinstance(blk, RecurrentBlock):
        return blk.mixer.init_state(batch)
    hd = blk.attn.qkv.shape[0] // blk.attn.n_heads
    z = jnp.zeros((batch, blk.attn.n_kv_heads, T_max, hd))
    return z, jnp.zeros_like(z)


def block_step(blk, x_new: jnp.ndarray, cache, pos, T_max: int, extra_valid: jnp.ndarray = None) -> tuple:
    if isinstance(blk, RecurrentBlock):
        valid = None if extra_valid is None else jax.lax.dynamic_index_in_dim(extra_valid, pos, axis=1, keepdims=False)
        return blk.step(x_new, cache, valid)
    x, ck, cv = blk.step(x_new, cache[0], cache[1], pos, T_max, extra_valid)
    return x, (ck, cv)


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


def token_ar_rollout(in_proj, member_embed, norm1, attn, ln_f, out_head, chunks: int,
                     h: jnp.ndarray, rng, quant_fn) -> tuple:
    # Differentiable SELF-FED digit rollout (downsampler_rollout): like token_ar_generate, but each
    # digit is drawn with quant_fn(logits, rng) -> (code_soft, idx) (straight-through gumbel/hard)
    # and fed back through member_embed via its soft one-hot, so gradients flow through the digits.
    # Returns (code_soft (...,chunks,V), idx (...,chunks), logits (...,chunks,V)).
    lead = h.shape[:-1]
    D = h.shape[-1]
    N = int(np.prod(lead)) if lead else 1
    collected = [(h.reshape(N, D) @ in_proj)[:, None, :]]
    softs, idxs, lgs = [], [], []
    for m in range(chunks):
        rng, k_ = jax.random.split(rng)
        seq_in = jnp.concatenate(collected, axis=1)
        h1 = seq_in + dense_self_attention(attn, norm1(seq_in), causal=True)
        logit_m = ln_f(h1)[:, -1, :] @ out_head
        soft_m, idx_m = quant_fn(logit_m, k_)
        softs.append(soft_m)
        idxs.append(idx_m)
        lgs.append(logit_m)
        if m < chunks - 1:
            collected.append((soft_m.astype(member_embed.dtype) @ member_embed)[:, None, :])
    V = lgs[0].shape[-1]
    return (jnp.stack(softs, axis=1).reshape(*lead, chunks, V), jnp.stack(idxs, axis=1).reshape(*lead, chunks),
            jnp.stack(lgs, axis=1).reshape(*lead, chunks, V))


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
    own_ctx_embed: jnp.ndarray  # (ctx_vocab, ctx_pq_dim) -- ALWAYS allocated (inert when
    # cfg.context_source=="codelm") so toggling that flag alone never changes checkpoint structure.
    # Same shape convention as CodeLM's own_input_embed/own_input_proj -- own_ctx_proj outputs
    # context_hidden_dim (CodeLM's own D_enc), NOT hidden_dim directly, so the existing context_proj
    # step (context_hidden_dim -> hidden_dim, see pardec_context_windows) still applies uniformly
    # regardless of context_source, no downstream code needs to change. Used directly (no self-
    # attention/block stack at all) as the context when cfg.context_source in ("own_embed",
    # "shared_embed") -- see pardec_context_hidden. "shared_embed": downsampler and upsampler
    # literally reuse the SAME array (shared_ctx_embed/shared_ctx_proj), mirroring
    # share_downsampler_upsampler_lm's pattern.
    own_ctx_proj: jnp.ndarray  # (ctx_pq_chunks * ctx_pq_dim, context_hidden_dim)
    draft_mask_embed: jnp.ndarray  # (hidden_dim,) level-refine 'fixed' layout: fills a draft slot with no draft
    cycle_slot_embed: jnp.ndarray  # (n_slots, hidden_dim) or None: per-slot tag, level_cycle_mode="stack"
    cycle_mask_embed: jnp.ndarray  # (hidden_dim,) or None: a stack slot not yet filled by a cycle
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
    remat_chunks: int = eqx.field(static=True)  # see Config.upsampler_remat_chunks
    token_head: str = eqx.field(static=True)  # "ar" | "linear", see Config.pardec_token_head

    def __init__(self, key, context_hidden_dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 n_layers: int, mlp_mult: int, rope_base: float, output_expansion: int,
                 context_window_groups: int, output_vocab: int, output_chunks: int, pq_dim: int,
                 token_dim: int, token_n_heads: int, decode_past: int = 0, decode_future: int = 0,
                 n_rates: int = 1, init_scheme: str = "llama", use_xsa: bool = True,
                 use_qknorm: bool = True, remat: bool = False, window: int = None,
                 shared_blocks: list = None, shared_ln_f: RMSNorm = None,
                 ctx_vocab: int = None, ctx_pq_chunks: int = None, ctx_pq_dim: int = None,
                 shared_ctx_embed: jnp.ndarray = None, shared_ctx_proj: jnp.ndarray = None,
                 token_head: str = "ar", backbone: str = "transformer", state_dim: int = 16,
                 cycle_slots: int = 0, remat_chunks: int = 1):
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
            self.blocks = [make_block(k, backbone, hidden_dim, n_heads, n_kv_heads, mlp_mult, rope_base,
                                      n_layers=n_layers, init_scheme=init_scheme, use_xsa=use_xsa,
                                      use_qknorm=use_qknorm, window=window, state_dim=state_dim)
                           for k in block_keys]
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
        self.token_out_head = init_matrix(jax.random.fold_in(key, 0), (token_dim, output_vocab), init_scheme)
        self.output_head_linear = init_matrix(jax.random.fold_in(key, 1),
                                               (hidden_dim, output_chunks * output_vocab), init_scheme)
        self.draft_mask_embed = init_vector(jax.random.fold_in(key, 3), hidden_dim, init_scheme)
        if cycle_slots > 0:
            slot_keys = jax.random.split(jax.random.fold_in(key, 4), cycle_slots + 1)
            self.cycle_slot_embed = jnp.stack([init_vector(k, hidden_dim, init_scheme) for k in slot_keys[1:]])
            self.cycle_mask_embed = init_vector(slot_keys[0], hidden_dim, init_scheme)
        else:
            self.cycle_slot_embed = self.cycle_mask_embed = None
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.output_expansion, self.context_window_groups = output_expansion, context_window_groups
        self.output_vocab, self.output_chunks = output_vocab, output_chunks
        self.token_dim, self.token_n_heads = token_dim, token_n_heads
        self.decode_past, self.decode_future = decode_past, decode_future
        self.remat = remat
        self.remat_chunks = remat_chunks
        self.token_head = token_head
        if shared_ctx_embed is not None:
            self.own_ctx_embed, self.own_ctx_proj = shared_ctx_embed, shared_ctx_proj
        else:
            ctx_keys = jax.random.split(jax.random.fold_in(key, 2), 2)
            self.own_ctx_embed = init_matrix(ctx_keys[0], (ctx_vocab, ctx_pq_dim), init_scheme)
            self.own_ctx_proj = init_matrix(ctx_keys[1], (ctx_pq_chunks * ctx_pq_dim, context_hidden_dim), init_scheme)


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


def cycle_slot_rows(pardec: PardecLM, cycle_ctx: list, context_group_size: int, n_groups: int,
                    n_context_positions: int, valid_mask: jnp.ndarray, dtype) -> tuple:
    # stack slots, each the same window as the main context: a filled slot = windowed revision hidden,
    # an empty one (None) = mask token; + its slot tag. Out-of-image window positions zeroed + masked.
    batch2, window_size = valid_mask.shape
    hidden_dim = pardec.context_proj.shape[1]
    pad = n_groups * context_group_size - n_context_positions
    rows = []
    for s, h in enumerate(cycle_ctx):
        if h is None:
            w = jnp.broadcast_to(pardec.cycle_mask_embed.astype(dtype), (batch2, window_size, hidden_dim))
        else:
            hp = jnp.pad(h, ((0, 0), (0, pad), (0, 0))) if pad > 0 else h
            w = pardec_context_windows(pardec, hp, context_group_size, n_groups, n_context_positions)[0].astype(dtype)
        w = w + pardec.cycle_slot_embed[s].astype(dtype)
        rows.append(jnp.where(valid_mask[:, :, None], w, 0.0))
    return jnp.concatenate(rows, axis=1), jnp.concatenate([valid_mask] * len(cycle_ctx), axis=1)


def pardec_score(pardec: PardecLM, target_seq: jnp.ndarray, context_h: jnp.ndarray,
                  context_group_size: int, output_group_size: int, rate_id: int = 0,
                  output_expansion: int = None, return_hidden: bool = False,
                  draft_seq: jnp.ndarray = None, draft_len: int = 0, draft_fill: str = "zero",
                  cycle_ctx: list = None, input_seq: jnp.ndarray = None) -> tuple:
    # input_seq (pss): tokens embedded as the row's own inputs instead of target_seq, which stays the
    # loss / digit-head target. Same shape as target_seq.
    # cycle_ctx (level_cycle_mode="stack"): per slot a (batch, n_context_positions, context_hidden)
    # revision hidden or None (mask); slots go right after bos: [window | bos | slots | draft | targets].
    # draft_seq/draft_len (level refine): fill the decode_past slot with each group's draft_len
    # preceding tokens of draft_seq (a previous pass's own prediction) instead of real targets.
    # draft_fill: "zero" = out-of-image slots zeroed + masked as keys (variable layout);
    # "mask" = those slots (or ALL slots when draft_seq is None) hold pardec.draft_mask_embed.
    # return_hidden=True: return ONLY the per-position predicted hidden states (before the token
    # head), (batch, n_output_positions*oe, hidden_dim). Causal masking means they do not depend on
    # the target rows at/after their position, so downsampler_rollout passes a dummy target and
    # feeds them to token_ar_rollout instead (needs decode_past==0).
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

    refine = draft_seq is not None or draft_len > 0
    assert draft_fill in ("zero", "mask"), draft_fill
    assert draft_seq is not None or draft_len == 0 or draft_fill == "mask", "an empty draft slot needs draft_fill='mask'"
    assert not (refine and pardec.decode_past > 0), "draft_seq (level refine) and decode_past>0 share one slot"
    decode_past = draft_len if refine else pardec.decode_past
    decode_future = pardec.decode_future
    target_len_per_group = output_group_size * oe
    n_output_positions = n_context_positions * output_group_size // context_group_size
    target_len_per_group_padded = n_groups * output_group_size * oe - n_output_positions * oe
    target_padded = target_seq
    if target_len_per_group_padded > 0:
        target_padded = jnp.pad(target_seq, ((0, 0), (0, target_len_per_group_padded), (0, 0)))

    def tail_windows(padded):
        if decode_future > 0:
            tail_padded = jnp.pad(padded, ((0, 0), (0, decode_future), (0, 0)))
            return jnp.stack(
                [tail_padded[:, g * target_len_per_group:g * target_len_per_group + target_len_per_group + decode_future]
                 for g in range(n_groups)], axis=1)
        return padded.reshape(batch, n_groups, target_len_per_group, *target_seq.shape[2:])
    real_tail_windows = tail_windows(target_padded)
    input_padded, input_windows = target_padded, real_tail_windows
    if input_seq is not None:
        input_padded = input_seq.astype(target_seq.dtype)
        if target_len_per_group_padded > 0:
            input_padded = jnp.pad(input_padded, ((0, 0), (0, target_len_per_group_padded), (0, 0)))
        input_windows = tail_windows(input_padded)
    real_tail_flat = input_windows.reshape(batch * n_groups, target_len_per_group + decode_future, *target_seq.shape[2:])
    real_tail_embedded = code_embed_proj(real_tail_flat, pardec.target_embed, pardec.target_proj)

    if decode_past > 0 and refine and draft_seq is None:
        draft_embedded = jnp.broadcast_to(pardec.draft_mask_embed.astype(real_tail_embedded.dtype),
                                          (batch * n_groups, decode_past, hidden_dim))
        draft_valid_flat = jnp.ones((batch * n_groups, decode_past), dtype=bool)
        target_embedded_flat = jnp.concatenate([draft_embedded, real_tail_embedded], axis=1)
    elif decode_past > 0:
        draft_flat, draft_valid_flat = group_draft_windows(draft_seq if refine else input_padded, n_groups,
                                                           target_len_per_group, decode_past, n_output_positions * oe)
        draft_embedded = code_embed_proj(draft_flat, pardec.target_embed, pardec.target_proj)
        if refine and draft_fill == "mask":
            draft_embedded = jnp.where(draft_valid_flat[:, :, None], draft_embedded,
                                       pardec.draft_mask_embed.astype(draft_embedded.dtype))
            draft_valid_flat = jnp.ones_like(draft_valid_flat)
        else:
            draft_embedded = jnp.where(draft_valid_flat[:, :, None], draft_embedded, 0.0)
        target_embedded_flat = jnp.concatenate([draft_embedded, real_tail_embedded], axis=1)
    else:
        draft_valid_flat = jnp.ones((batch * n_groups, 0), dtype=bool)
        target_embedded_flat = real_tail_embedded

    bos = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch * n_groups, 1, hidden_dim))
    slot_len = 0
    if cycle_ctx:
        slot_rows, slot_valid = cycle_slot_rows(pardec, cycle_ctx, context_group_size, n_groups,
                                                n_context_positions, valid_mask, own_window.dtype)
        slot_len = slot_rows.shape[1]
        row_flat = jnp.concatenate([own_window, bos, slot_rows, target_embedded_flat], axis=1)
    else:
        row_flat = jnp.concatenate([own_window, bos, target_embedded_flat], axis=1)
    per_group_len = window_size + 1 + slot_len + decode_past + target_len_per_group + decode_future

    bos_valid = [jnp.ones((batch * n_groups, 1), dtype=bool)] + ([slot_valid] if slot_len else [])
    key_valid = jnp.concatenate([valid_mask] + bos_valid + [draft_valid_flat,
                                  jnp.ones((batch * n_groups, target_len_per_group + decode_future), dtype=bool)], axis=1)
    rope_bos = jnp.array(group_ends)[:, None]
    if slot_len:
        rope_bos = jnp.concatenate([rope_bos, jnp.stack([end + 1 + jnp.arange(slot_len) for end in group_ends])], axis=1)
    if refine:
        # sequential after bos (window | bos | slots | draft | targets), so pardec_generate's single
        # position counter reproduces it exactly
        rope_draft = jnp.stack([end + 1 + slot_len + jnp.arange(decode_past) for end in group_ends], axis=0)
        tail_start = 1 + slot_len + decode_past
    else:
        rope_draft = jnp.stack([end - decode_past + jnp.arange(decode_past) for end in group_ends], axis=0)
        tail_start = 1 + slot_len
    rope_real_tail = jnp.stack([end + tail_start + jnp.arange(target_len_per_group + decode_future)
                                for end in group_ends], axis=0)
    rope_target = jnp.clip(jnp.concatenate([rope_draft, rope_real_tail], axis=1), 0, None)
    rope_pos_ids_g = jnp.concatenate([rope_ids, rope_bos, rope_target], axis=1)
    rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (batch, n_groups, per_group_len)).reshape(batch * n_groups, per_group_len)

    if pardec.remat_chunks > 1:
        def chunk_stack(x, rp, kv):
            for blk in pardec.blocks:
                x = run_block_pardec(blk, x, rp, kv, pardec.remat)
            return pardec.ln_f(x)
        hidden = remat_row_chunks(chunk_stack, row_flat, rope_pos_ids, key_valid, pardec.remat_chunks)
    else:
        def run_stack(x):
            for blk in pardec.blocks:
                x = run_block_pardec(blk, x, rope_pos_ids, key_valid, pardec.remat)
            return x
        hidden = jax.checkpoint(run_stack)(row_flat) if pardec.remat else run_stack(row_flat)
        hidden = pardec.ln_f(hidden)
    prediction_positions = window_size + slot_len + decode_past + jnp.arange(target_len_per_group)
    predicted_hidden = hidden[:, prediction_positions, :]
    predicted_hidden = predicted_hidden.reshape(batch, n_groups * target_len_per_group, hidden_dim)
    valid_len = n_output_positions * oe
    predicted_hidden = predicted_hidden[:, :valid_len, :]
    target_out = target_seq[:, :valid_len]
    if return_hidden:
        return predicted_hidden
    if pardec.token_head == "linear":
        logits = reshape_pq(predicted_hidden @ pardec.output_head_linear, pardec.output_chunks, pardec.output_vocab)
    else:
        logits = token_ar_teacher_forced(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                          pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                          pardec.token_dim, pardec.output_vocab, predicted_hidden, target_out)

    # decode_future aux NTP loss: standard shifted next-token prediction OVER the widened lookahead
    # span itself (predict future token k from the hidden state produced by processing future token
    # k-1, same causal NTP pattern as CodeLM's own ntp_head) -- caught 2026-09-27: this was
    # completely unimplemented (decode_logits_and_target_multipass hardcoded aux_loss=0 always), so
    # decode_future>0 widened the sequence (real compute cost) but had ZERO effect on training: the
    # extra future tokens were embedded as input/keys but nothing ever queried FROM those positions,
    # and causal masking means they can't influence any earlier (real) prediction either.
    if decode_future > 0:
        # hidden at row idx W+dp+T+k (the embedding of target T-1+k) predicts future_k = target T+k;
        # the old '-1' made k=0 share the hidden that must predict target T-1 (fixed 2026-09-28)
        aux_pred_positions = window_size + slot_len + decode_past + target_len_per_group + jnp.arange(decode_future)
        aux_predicted_hidden = hidden[:, aux_pred_positions, :]
        aux_predicted_hidden = aux_predicted_hidden.reshape(batch, n_groups * decode_future, hidden_dim)
        aux_target = real_tail_windows[:, :, target_len_per_group:, :].reshape(
            batch, n_groups * decode_future, *target_seq.shape[2:])
        # valid only where the future span's absolute output position is real, not artificial
        # end-of-sequence padding (mirrors target_padded/tail_padded's own padding boundary).
        abs_idx = np.array([[(g + 1) * target_len_per_group + k for k in range(decode_future)]
                             for g in range(n_groups)])
        aux_valid = jnp.asarray(abs_idx < valid_len).reshape(1, n_groups * decode_future, 1)
        aux_valid = jnp.broadcast_to(aux_valid, aux_target.shape).astype(jnp.float32)
        if pardec.token_head == "linear":
            aux_logits = reshape_pq(aux_predicted_hidden @ pardec.output_head_linear,
                                    pardec.output_chunks, pardec.output_vocab)
        else:
            aux_logits = token_ar_teacher_forced(pardec.token_in_proj, pardec.token_member_embed,
                                                  pardec.token_norm1, pardec.token_attn, pardec.token_ln_f,
                                                  pardec.token_out_head, pardec.token_dim, pardec.output_vocab,
                                                  aux_predicted_hidden, aux_target)
        logp_aux = jax.nn.log_softmax(aux_logits, axis=-1)
        nll_aux = -jnp.take_along_axis(logp_aux, aux_target[..., None], axis=-1)[..., 0]
        denom = jnp.maximum(jnp.sum(aux_valid), 1.0)
        aux_loss = jnp.sum(nll_aux * aux_valid) / denom
        aux_acc = jnp.sum((jnp.argmax(aux_logits, -1) == aux_target).astype(jnp.float32) * aux_valid) / denom
    else:
        aux_loss = jnp.array(0.0, dtype=logits.dtype)
        aux_acc = jnp.array(0.0, dtype=logits.dtype)
    return logits, target_out, aux_loss, aux_acc


def pardec_generate(pardec: PardecLM, context_h: jnp.ndarray, context_group_size: int,
                     output_group_size: int, rng, greedy: bool = True, temperature: float = 1.0,
                     top_k: int = 0, rate_id: int = 0, output_expansion: int = None,
                     draft_seq: jnp.ndarray = None, draft_len: int = 0, draft_fill: str = "zero",
                     cycle_ctx: list = None) -> jnp.ndarray:
    # Generation counterpart of pardec_score: one growing KV cache (or recurrent state) per group,
    # all groups batched. draft_seq/draft_len/draft_fill/cycle_ctx: same slots as pardec_score,
    # prefilled after bos.
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
    assert draft_fill in ("zero", "mask"), draft_fill
    assert draft_seq is not None or draft_len == 0 or draft_fill == "mask", "an empty draft slot needs draft_fill='mask'"
    Pp = draft_len
    slot_len = 0
    if cycle_ctx:
        slot_rows, slot_valid = cycle_slot_rows(pardec, cycle_ctx, context_group_size, n_groups,
                                                n_context_positions, valid_mask, own_window.dtype)
        slot_len = slot_rows.shape[1]
    total_steps = window_size + 1 + slot_len + Pp + target_len_per_group
    n_output_positions = n_context_positions * output_group_size // context_group_size
    if Pp > 0 and draft_seq is None:
        draft_emb = jnp.broadcast_to(pardec.draft_mask_embed.astype(own_window.dtype), (batch2, Pp, hidden_dim))
        draft_valid_flat = jnp.ones((batch2, Pp), dtype=bool)
    elif Pp > 0:
        draft_flat, draft_valid_flat = group_draft_windows(draft_seq, n_groups, target_len_per_group, Pp,
                                                           n_output_positions * oe)
        draft_emb = code_embed_proj(draft_flat, pardec.target_embed, pardec.target_proj)
        if draft_fill == "mask":
            draft_emb = jnp.where(draft_valid_flat[:, :, None], draft_emb, pardec.draft_mask_embed.astype(draft_emb.dtype))
            draft_valid_flat = jnp.ones_like(draft_valid_flat)
        else:
            draft_emb = jnp.where(draft_valid_flat[:, :, None], draft_emb, 0.0)
    else:
        draft_valid_flat = jnp.ones((batch2, 0), dtype=bool)

    caches0 = [block_cache_init(blk, batch2, total_steps) for blk in pardec.blocks]

    # extra_valid: own_window's leading entries are zero-padding wherever a group's real history
    # is shorter than window_size (see pardec_context_windows' valid_mask) -- idx<=pos causal
    # masking alone can't tell those apart from real keys, so pass this through explicitly (mirrors
    # pardec_score's key_valid, which excludes exactly these positions). bos+target span is always
    # real/self-authored, hence the trailing all-True padding.
    bos_valid = [jnp.ones((batch2, 1), dtype=bool)] + ([slot_valid] if slot_len else [])
    extra_valid = jnp.concatenate(
        [valid_mask] + bos_valid + [draft_valid_flat, jnp.ones((batch2, target_len_per_group), dtype=bool)], axis=1)

    def self_step(x_new, caches, pos):
        new_caches = []
        x = x_new
        for blk, c in zip(pardec.blocks, caches):
            x, c = block_step(blk, x, c, pos, total_steps, extra_valid)
            new_caches.append(c)
        return pardec.ln_f(x), new_caches

    # context prefill: feed the window one position at a time -- blk.step handles a single new
    # position per call (2D x_new, no seq-len axis), so this is a scan over the window.
    def context_step(carry, x_t):
        caches, pos = carry
        _, caches = self_step(x_t, caches, pos)
        return (caches, pos + 1), None

    (caches, pos), _ = jax.lax.scan(context_step, (caches0, jnp.array(0)), jnp.swapaxes(own_window, 0, 1))

    bos_in = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch2, hidden_dim))
    hidden, caches = self_step(bos_in, caches, pos)
    pos = pos + 1

    def prefill_step(carry, x_t):
        caches, pos, _ = carry
        h, caches = self_step(x_t, caches, pos)
        return (caches, pos + 1, h), None

    if slot_len > 0:
        (caches, pos, hidden), _ = jax.lax.scan(prefill_step, (caches, pos, hidden), jnp.swapaxes(slot_rows, 0, 1))
    if Pp > 0:
        (caches, pos, hidden), _ = jax.lax.scan(prefill_step, (caches, pos, hidden), jnp.swapaxes(draft_emb, 0, 1))

    def gen_step(carry, _):
        caches, pos, rng, x_input, hidden = carry
        if pardec.token_head == "linear":
            val, rng = sample_idx(reshape_pq(hidden @ pardec.output_head_linear, pardec.output_chunks,
                                             pardec.output_vocab), rng, greedy, temperature, top_k)
        else:
            val, rng = token_ar_generate(pardec.token_in_proj, pardec.token_member_embed, pardec.token_norm1,
                                          pardec.token_attn, pardec.token_ln_f, pardec.token_out_head,
                                          pardec.output_chunks, hidden, rng, greedy, temperature, top_k)
        x_next = code_embed_proj(val, pardec.target_embed, pardec.target_proj)
        hidden_next, caches = self_step(x_next, caches, pos)
        return (caches, pos + 1, rng, x_next, hidden_next), val

    init_carry = (caches, pos, rng, bos_in, hidden)
    _, vals = jax.lax.scan(gen_step, init_carry, None, length=target_len_per_group)
    vals = jnp.moveaxis(vals, 0, 1)  # (batch2, target_len_per_group, output_chunks)
    vals = vals.reshape(batch, n_groups * target_len_per_group, *vals.shape[2:])
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

    caches0 = [block_cache_init(blk, batch2, total_steps) for blk in pardec.blocks]

    # see pardec_generate's identical comment: own_window's leading entries can be zero-padding
    # (group's real history shorter than window_size) -- mask those out explicitly since idx<=pos
    # causal masking alone can't distinguish them from real keys.
    extra_valid = jnp.concatenate(
        [valid_mask, jnp.ones((batch2, total_steps - window_size), dtype=bool)], axis=1)

    def self_step(x_new, caches, pos):
        new_caches = []
        x = x_new
        for blk, c in zip(pardec.blocks, caches):
            x, c = block_step(blk, x, c, pos, total_steps, extra_valid)
            new_caches.append(c)
        return pardec.ln_f(x), new_caches

    def context_step(carry, x_t):
        caches, pos = carry
        _, caches = self_step(x_t, caches, pos)
        return (caches, pos + 1), None

    (caches, pos), _ = jax.lax.scan(context_step, (caches0, jnp.array(0)), jnp.swapaxes(own_window, 0, 1))

    bos_in = jnp.broadcast_to(pardec.bos_embed[rate_id], (batch2, hidden_dim))
    hidden, caches = self_step(bos_in, caches, pos)
    pos = pos + 1

    def gen_step(carry, step_rng):
        caches, pos, hidden = carry
        logits = reshape_pq(hidden @ pardec.output_head_linear, pardec.output_chunks, pardec.output_vocab)
        code_soft, code_idx = quantize_gumbel(logits, step_rng, temperature, quantize_drop)
        x_next = code_embed_proj(code_soft, pardec.target_embed, pardec.target_proj)
        hidden_next, caches = self_step(x_next, caches, pos)
        return (caches, pos + 1, hidden_next), (code_soft, code_idx)

    step_rngs = jax.random.split(rng, target_len_per_group)
    init_carry = (caches, pos, hidden)
    _, (code_softs, code_idxs) = jax.lax.scan(gen_step, init_carry, step_rngs)
    code_softs = jnp.moveaxis(code_softs, 0, 1).reshape(batch, n_groups * target_len_per_group, *code_softs.shape[2:])
    code_idxs = jnp.moveaxis(code_idxs, 0, 1).reshape(batch, n_groups * target_len_per_group, *code_idxs.shape[2:])
    n_output_positions = n_context_positions * output_group_size // context_group_size
    return code_softs[:, :n_output_positions], code_idxs[:, :n_output_positions]


def bos_rate_map(cfg: "Config") -> tuple:
    # Only meaningful when cfg.share_across_levels=True (the False/per-level-instance case always
    # uses rate_id=0, level identity already being structural -- see LagCodecModel.bos_rate_id).
    # Returns a length-n tuple: level_idx -> rate_id, the row of bos_embed that level reads/writes.
    n = len(cfg.strides)
    if cfg.bos_rate_mode == "absolute":
        return tuple(range(n))
    # "relative": dedup by EFFECTIVE stride value (K()'s own -1-means-1 convention), first-occurrence
    # order -- levels sharing the same real stride share the same bos row. strides=(4,4,4) -> effective
    # (4,4,4) -> one unique value -> (0,0,0), n_rates=1. strides=(3,16,16,-1) -> effective (3,16,16,1)
    # -> (0,1,1,2), n_rates=3 (levels 1&2 share a row, level 3's stride=1 gets its own).
    seen = {}
    out = []
    for i in range(n):
        k = cfg.strides[i] if cfg.strides[i] != -1 else 1
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return tuple(out)


def bos_n_rates(cfg: "Config") -> int:
    return len(set(bos_rate_map(cfg)))


class CodeLM(eqx.Module):
    # The "encoder" of the seq2seq framing (CodeLM=encoder, downsampler/upsampler=two separate
    # decoders). cfg.share_across_levels=True (default): exactly one CodeLM instance for the whole
    # model, shared across all levels via LagCodecModel.codelm_for/bos_rate_id -- NOT one per level.
    # A level's raw byte input (level 0) and a level's code input (level>0)
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
    n_heads: int = eqx.field(static=True)  # needed for KV-cache sizing in _encoder_free_run
    n_kv_heads: int = eqx.field(static=True)
    token_head: str = eqx.field(static=True)  # "linear" | "ar", see Config.codelm_token_head
    token_dim: int = eqx.field(static=True)
    tok_in_proj: jnp.ndarray = None  # AR digit head params, None unless token_head=="ar"
    tok_member_embed: jnp.ndarray = None
    tok_norm1: RMSNorm = None
    tok_attn: Attention = None
    tok_ln_f: RMSNorm = None
    tok_out_head: jnp.ndarray = None

    def __init__(self, key, cfg: Config, level_idx: int = 0):
        # level_idx: which per-level config tuple entry to read architecture from. share_across_levels
        # =True (default): always 0 (Config.__post_init__ asserts codelm_* uniform across ALL levels
        # in that mode, so index 0 is representative of every level). share_across_levels=False: the
        # caller (LagCodecModel) builds one CodeLM per level, each with its own level_idx -- may then
        # have genuinely different d_model/n_layers/etc per level.
        D_enc = cfg.codelm_d_model[level_idx]
        n_layers_enc = cfg.codelm_n_layers[level_idx]
        n_heads_enc = cfg.codelm_n_heads[level_idx]
        n_kv_heads_enc = cfg.codelm_n_kv_heads[level_idx]
        self.remat = cfg.remat if cfg.codelm_remat[level_idx] is None else cfg.codelm_remat[level_idx]
        self.remat_level = cfg.remat_level
        self.attn_lookahead = cfg.attn_lookahead[level_idx]
        self.pq_chunks, self.code_vocab = cfg.pq_chunks[level_idx], cfg.code_vocab[level_idx]
        self.quantize_mode = cfg.quantize_mode
        self.quantize_drop = cfg.quantize_drop
        self.use_codelm_bos = cfg.use_codelm_bos
        self.codelm_bos_prob = cfg.codelm_bos_prob
        self.n_heads, self.n_kv_heads = n_heads_enc, n_kv_heads_enc
        pq_dim = cfg.pq_dim[level_idx]
        own_vocab = self.code_vocab  # raw bytes and codes share one categorical structure (see above)
        ntp_out = self.pq_chunks * self.code_vocab
        keys = jax.random.split(key, 4)

        scheme, use_xsa, use_qknorm = cfg.init_scheme, cfg.use_xsa, cfg.use_qknorm
        self.own_input_embed = init_matrix(keys[0], (own_vocab, pq_dim), scheme)
        self.own_input_proj = init_matrix(keys[1], (self.pq_chunks * pq_dim, D_enc), scheme)
        block_keys = jax.random.split(jax.random.fold_in(key, 3), n_layers_enc)
        enc_window_val = (cfg.encoder_attn_window[level_idx] if cfg.encoder_attn_window[level_idx] is not None
                           else cfg.attn_window[level_idx])
        enc_window = None if enc_window_val == -1 else enc_window_val
        self.blocks = [make_block(k, cfg.codelm_backbone[level_idx], D_enc, n_heads_enc, n_kv_heads_enc,
                                  cfg.mlp_mult[level_idx], cfg.rope_base[level_idx], n_layers=n_layers_enc,
                                  init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, window=enc_window,
                                  lookahead=self.attn_lookahead, use_sink=cfg.use_sink, state_dim=cfg.ssm_state_dim)
                       for k in block_keys]
        self.ln_f = RMSNorm(D_enc)
        self.code_head = init_matrix(keys[2], (D_enc, self.pq_chunks * self.code_vocab), scheme)
        self.ntp_head = init_matrix(keys[3], (D_enc, ntp_out), scheme)
        # CodeLM's own bos is ALWAYS absolute (one row per level index), regardless of
        # cfg.bos_rate_mode -- unlike downsampler/upsampler's bos (which anchors a group's
        # contraction/expansion RATE, so dedup-by-rate is sound), CodeLM's bos anchors "which level
        # am I generating" for encoder_free_run's pure-unconditional rollout. Two levels sharing the
        # same effective stride (e.g. both K=4) are still semantically distinct free-run targets
        # (different resolution/content statistics), so collapsing them onto one shared row would
        # conflate two different anchors -- level IDENTITY, not rate, is what this needs to encode.
        # share_across_levels=False: this instance already belongs to exactly one level structurally,
        # so it only needs its own private row(s) (n_rates=codelm_bos_rates[level_idx], rate_id
        # always 0 at call time -- see LagCodecModel.codelm_bos_rate_id).
        n_bos_rates = (len(cfg.strides) + (cfg.context_source == "codelm_upper") if cfg.share_across_levels
                       else cfg.codelm_bos_rates[level_idx]) \
            if cfg.use_codelm_bos else 1
        bos_keys = jax.random.split(jax.random.fold_in(key, 4), n_bos_rates)
        self.bos_embed = jnp.stack([init_vector(k, D_enc, scheme) for k in bos_keys], axis=0)
        self.token_head = cfg.codelm_token_head
        self.token_dim = cfg.token_dim[level_idx]
        if self.token_head == "ar":
            tk = jax.random.split(jax.random.fold_in(key, 5), 4)
            td, tnh = cfg.token_dim[level_idx], cfg.token_n_heads[level_idx]
            self.tok_in_proj = init_matrix(tk[0], (D_enc, td), scheme)
            self.tok_member_embed = init_matrix(tk[1], (self.code_vocab, td), scheme)
            self.tok_norm1 = RMSNorm(td)
            self.tok_attn = Attention(tk[2], td, tnh, tnh, cfg.rope_base[level_idx], n_layers=1,
                                       init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm)
            self.tok_ln_f = RMSNorm(td)
            self.tok_out_head = init_matrix(tk[3], (td, self.code_vocab), scheme)

    def encode(self, x: jnp.ndarray, target_idx: jnp.ndarray, K: int, rng=None, encode_temperature: float = 1.0,
               layer_drop_prob=None, rate_id: int = 0, force_bos: bool = False) -> dict:
        if force_bos:
            x = x.at[:, 0, :].set(self.bos_embed[rate_id])
        elif self.use_codelm_bos and rng is not None:
            B = x.shape[0]
            do_sub = jax.random.bernoulli(jax.random.fold_in(rng, 6), p=self.codelm_bos_prob, shape=(B,))
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
        code_soft, code_idx = quantize_dispatch(self.quantize_mode, logits, quant_rng, encode_temperature,
                                                 self.quantize_drop)

        probs = jax.nn.softmax(logits, axis=-1)
        p_avg = jnp.mean(probs, axis=(0, 1))
        entropy_loss = jnp.mean(jnp.sum(p_avg * jnp.log(jnp.maximum(p_avg, 1e-9)), axis=-1))

        ntp_shift = 1 + self.attn_lookahead
        if L > ntp_shift:
            tgt = target_idx[:, ntp_shift:]
            ntp_logits = codelm_ntp_logits_tf(self, h[:, :-ntp_shift, :], tgt)
            logp = jax.nn.log_softmax(ntp_logits, axis=-1)
            ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
            ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
        else:
            ntp_loss = jnp.array(0.0, dtype=h.dtype)
            ntp_acc = jnp.array(0.0, dtype=h.dtype)
        util = codebook_utilization(code_idx, self.code_vocab)
        return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                    entropy_loss=entropy_loss, logits=logits)




def codelm_bos_substitute(codelm: CodeLM, x: jnp.ndarray, rate_id: int, rng, group_size: int) -> jnp.ndarray:
    # Single source of truth for CodeLM's own bos substitution -- factored out 2026-09-27 after
    # finding it missing from THREE separate call sites in a row (encode_pardec_downsampler,
    # encode_pardec_downsampler_generate, and the upsampler's context builders
    # decode_logits_and_target_multipass/_decode_generate_pardec_call all build a CodeLM forward
    # pass over their own input x, and each one needs this same substitution applied to ITS OWN x
    # before running it through CodeLM's blocks -- easy to forget per call site, so every caller
    # should route through here instead of reimplementing it inline.
    #
    # REPEAT-PER-GROUP mode (tried 2026-09-27, testing the weight-sharing hypothesis for "train mse
    # low but gen bad": with a SHARED CodeLM across all levels, a group only knew which level it
    # belonged to via PardecLM's own per-group bos_embed[rate_id] row -- CodeLM's OWN context
    # representation h never got a matching per-group anchor, only a global one at absolute sequence
    # position 0). REVERTED same day: user confirmed via real training run (cifar_res_full2) this did
    # NOT fix the bad-generation symptom, and separately confirmed use_codelm_bos=False entirely
    # (run_lagcodec_res_v1.py, pre-bos-substitution code) DID work -- so bos substitution itself (in
    # either form) is implicated, not just its once-vs-per-group granularity. Kept as a comment, not
    # deleted, in case it's revisited:
    # if codelm.use_codelm_bos and rng is not None:
    #     B, T = x.shape[0], x.shape[1]
    #     do_sub = jax.random.bernoulli(jax.random.fold_in(rng, 6), p=codelm.codelm_bos_prob, shape=(B,))
    #     bos_row = codelm.bos_embed[rate_id]
    #     for s in range(0, T, group_size):
    #         xs = jnp.where(do_sub[:, None], bos_row, x[:, s, :])
    #         x = x.at[:, s, :].set(xs)
    # return x
    #
    # ACTIVE (bos-once at position 0 -- original pre-repeat-per-group behavior, restored):
    if codelm.use_codelm_bos and rng is not None:
        B = x.shape[0]
        do_sub = jax.random.bernoulli(jax.random.fold_in(rng, 7), p=codelm.codelm_bos_prob, shape=(B,))
        x0 = jnp.where(do_sub[:, None], codelm.bos_embed[rate_id], x[:, 0, :])
        x = x.at[:, 0, :].set(x0)
    return x


def pardec_context_hidden(codelm: CodeLM, pardec: PardecLM, raw: jnp.ndarray, cfg: "Config",
                           rate_id: int, rng, group_size: int) -> jnp.ndarray:
    # Single source of truth for how the downsampler/upsampler get their CONTEXT, per
    # cfg.context_source (see Config field docstring). `raw` is the pre-embedding representation of
    # this level's own input (hard idx for raw bytes/a previous level's code_idx, or the SOFT
    # code_soft when called from a training path that needs gradient flow through it) -- the exact
    # same value callers already use to build CodeLM's own `x` in the "codelm" mode, generalized so
    # "own_embed"/"shared_embed" modes can bypass CodeLM's own embedding table entirely too.
    if cfg.context_source in ("codelm", "codelm_upper"):  # which CodeLM is the caller's choice
        x = code_embed_proj(raw, codelm.own_input_embed, codelm.own_input_proj)
        # NOT bos-substituted (removed 2026-10-03): every caller of pardec_context_hidden
        # (encode_pardec_downsampler[_generate], decode_logits_and_target_multipass,
        # _decode_generate_pardec_call) always has real content available at position 0 -- none of
        # them is the true "zero real content" unconditional case (that's _encoder_free_run's own,
        # fully independent use_bos path). With codelm_bos_prob=1.0, substituting here permanently
        # discarded real position-0 signal on every encode/decode-context call, train and eval alike
        # -- root cause of total top-level codebook collapse (confirmed: cifar_overfit_3_1layer vs
        # _nobos, identical except use_codelm_bos; bos run: 1 unique code across 8 train samples,
        # gen_byte_acc=0.005 (chance); nobos run: gen_byte_acc=0.88, 7/8 train samples mse=0.00).
        h = x
        def _enc_stack(h):
            for blk in codelm.blocks:
                h = run_block(blk, h, codelm.remat and not codelm.remat_level)
            return h
        h = jax.checkpoint(_enc_stack)(h) if codelm.remat_level else _enc_stack(h)
        return codelm.ln_f(h)
    # "own_embed"/"shared_embed": plain per-position embedding, NO self-attention/contextualization
    # at all -- CodeLM's own block stack is skipped entirely (codelm_bos is therefore also moot here,
    # it only ever modulated CodeLM's own forward pass).
    return code_embed_proj(raw, pardec.own_ctx_embed, pardec.own_ctx_proj)


def pss_inputs(logits: jnp.ndarray, target: jnp.ndarray, cfg: "Config", rng, prob: float) -> jnp.ndarray:
    # parallel scheduled sampling inputs: own prediction per position w.p. prob, else GT, detached. Mask and
    # gumbel noise do not depend on the pass, so one pass per row token reproduces a sequential rollout.
    lg = logits
    if cfg.pss_input_mode == "sample" and rng is not None:
        lg = lg.astype(jnp.float32) / cfg.pss_temperature
        lg = lg + jax.random.gumbel(jax.random.fold_in(rng, 41), lg.shape)
    own_tok = safe_argmax(lg).astype(jnp.int32)  # feeds a gather: avoid TPU argmax bug
    if prob < 1.0 and rng is not None:
        own = jax.random.bernoulli(jax.random.fold_in(rng, 40), p=prob, shape=own_tok.shape[:2])
        own_tok = jnp.where(own[..., None], own_tok, target.astype(jnp.int32))
    return jax.lax.stop_gradient(own_tok)


def pss_n_passes(passes: int, row_tokens: int, decode_past: int) -> int:
    # -1 = one pass per row token; a single-token row with no decode_past has no token input to swap
    if row_tokens <= 1 and decode_past == 0:
        return 1
    return row_tokens if passes == -1 else passes


def rollout_n_passes(passes: int, row_tokens: int) -> int:
    # downsampler_rollout row passes: one per row code (exact) unless downsampler_pss_passes>1 caps it
    return row_tokens if passes in (-1, 1) else min(passes, row_tokens)


def encode_pardec_downsampler(codelm: CodeLM, downsampler: PardecLM, raw: jnp.ndarray, target_idx: jnp.ndarray,
                               flat_bytes: jnp.ndarray, cfg: "Config", pixel_order, label_fn,
                               K: int, rate_id: int = 0, rng=None, downsampler_ncodes: int = 1,
                               encode_temperature: float = 1.0, codelm_rate_id: int = None,
                               pss_passes: int = 1) -> dict:
    # CodeLM forward (same as CodeLM.encode()'s first half), then the SHARED downsampler PardecLM
    # teacher-forced against label_fn's real downsampled-image target (context_group_size=K,
    # output_group_size=1 -- one code per K-block, genuinely autoregressive over context).
    # code_idx/code_soft go through the SAME quantize_mode dispatch as CodeLM.encode() (caught
    # 2026-09-27: this used to always take plain argmax/softmax regardless of quantize_mode --
    # meaning the code_soft fed to the upsampler as context was always "clean"/noiseless during
    # training, never the noisy gumbel-sampled codes quantize_mode="gumbel" is supposed to produce.
    # The upsampler only ever saw idealized context, never the noisy codes a real (non-label_fn-
    # teacher-forced) downsampler actually emits at generation time -- so it never learned to be
    # robust to that noise. Now all three of CodeLM/downsampler/upsampler consistently follow
    # quantize_mode: CodeLM's own naive path already did, this dispatches the SAME way, and the
    # upsampler needs no change since it already just consumes whatever code_soft it's given.
    # Returns the SAME dict shape as CodeLM.encode() so level_forward's surrounding loss code
    # doesn't change.
    # codelm_rate_id: CodeLM's own bos anchor index -- ALWAYS absolute (one row per level, see
    # LagCodecModel.codelm_bos_rate_id), independent of `rate_id` below (the downsampler's OWN bos,
    # which follows cfg.bos_rate_mode). Defaults to `rate_id` for standalone/test callers that don't
    # distinguish the two. `raw` is `target_idx` in the level-0 byte case or the previous level's own
    # code_soft/code_idx -- see pardec_context_hidden for how cfg.context_source dispatches it.
    codelm_rate_id = rate_id if codelm_rate_id is None else codelm_rate_id
    h = pardec_context_hidden(codelm, downsampler, raw, cfg, codelm_rate_id, rng, group_size=K * downsampler_ncodes)
    M, L, D = h.shape
    n_blocks = L // K
    # context_group_size=K*downsampler_ncodes, output_group_size=downsampler_ncodes, rate_id=this
    # level's index -- the SAME shared downsampler weights are called with a different
    # (K, downsampler_ncodes, rate_id) triple per level, modulated via the downsampler's own
    # bos_embed row (see PardecLM), not via separate weights. downsampler_ncodes=1 (default): one
    # code per K-block, computed one at a time. >1: downsampler_ncodes codes' worth of raw input
    # (downsampler_ncodes*K positions) batched into one group/call, producing downsampler_ncodes
    # codes together.
    def _teacher_forced():
        # teacher-forced on label_fn(image) digits (default path; see Config.downsampler_rollout)
        label_tgt = label_fn(flat_bytes, cfg, pixel_order, n_blocks, codelm.pq_chunks, codelm.code_vocab)
        score = lambda **kw: pardec_score(downsampler, label_tgt, h, context_group_size=K * downsampler_ncodes,
                                          output_group_size=downsampler_ncodes, rate_id=rate_id, **kw)[0]
        lg = score()
        for _ in range(1, pss_n_passes(pss_passes, downsampler_ncodes, downsampler.decode_past)):
            lg = score(input_seq=pss_inputs(lg, label_tgt, cfg, rng, cfg.downsampler_pss_prob))
        cs, ci = quantize_dispatch(codelm.quantize_mode, lg, rng, encode_temperature, codelm.quantize_drop)
        return lg, cs, ci

    def _rollout():
        # self-fed: hidden at each row position, then the AR digit head fed its OWN straight-through samples,
        # exactly like inference. ncodes>1: a row code's input is the previous pass's own code (same noise every
        # pass), so rollout_n_passes passes reproduce the sequential rollout; never label_fn tokens.
        dummy = jnp.zeros((M, n_blocks, codelm.pq_chunks), jnp.int32)
        qfn = lambda lg_m, k_: quantize_dispatch(codelm.quantize_mode, lg_m,
                                                  k_ if rng is not None else None,
                                                  encode_temperature, codelm.quantize_drop)
        ci = None
        for _ in range(rollout_n_passes(pss_passes, min(downsampler_ncodes, n_blocks))):
            hid = pardec_score(downsampler, dummy, h, context_group_size=K * downsampler_ncodes,
                               output_group_size=downsampler_ncodes, rate_id=rate_id, return_hidden=True,
                               input_seq=None if ci is None else jax.lax.stop_gradient(ci))
            cs, ci, lg = token_ar_rollout(downsampler.token_in_proj, downsampler.token_member_embed,
                                           downsampler.token_norm1, downsampler.token_attn, downsampler.token_ln_f,
                                           downsampler.token_out_head, downsampler.output_chunks, hid,
                                           rng if rng is not None else jax.random.PRNGKey(0), qfn)
        return lg, cs, ci

    if not cfg.downsampler_rollout:
        logits, code_soft, code_idx = _teacher_forced()
    elif rng is None or cfg.downsampler_rollout_prob >= 1.0:
        logits, code_soft, code_idx = _rollout()
    elif cfg.downsampler_rollout_prob <= 0.0:
        logits, code_soft, code_idx = _teacher_forced()
    else:
        use_roll = jax.random.bernoulli(jax.random.fold_in(rng, 8), p=cfg.downsampler_rollout_prob)
        _canon = lambda t: (t[0].astype(h.dtype), t[1].astype(h.dtype), t[2].astype(jnp.int32))
        logits, code_soft, code_idx = jax.lax.cond(
            use_roll, lambda: _canon(_rollout()), lambda: _canon(_teacher_forced()))

    probs = jax.nn.softmax(logits, axis=-1)
    p_avg = jnp.mean(probs, axis=(0, 1))
    entropy_loss = jnp.mean(jnp.sum(p_avg * jnp.log(jnp.maximum(p_avg, 1e-9)), axis=-1))

    ntp_shift = 1 + codelm.attn_lookahead
    if L > ntp_shift:
        tgt = target_idx[:, ntp_shift:]
        ntp_logits = codelm_ntp_logits_tf(codelm, h[:, :-ntp_shift, :], tgt)
        logp = jax.nn.log_softmax(ntp_logits, axis=-1)
        ntp_loss = -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
        ntp_acc = jnp.mean(jnp.argmax(ntp_logits, -1) == tgt)
    else:
        ntp_loss = jnp.array(0.0, dtype=h.dtype)
        ntp_acc = jnp.array(0.0, dtype=h.dtype)
    util = codebook_utilization(code_idx, codelm.code_vocab)
    return dict(code_soft=code_soft, code_idx=code_idx, ntp_loss=ntp_loss, ntp_acc=ntp_acc, util=util,
                entropy_loss=entropy_loss, logits=logits)


def codelm_ntp_loss(codelm: CodeLM, raw: jnp.ndarray, target: jnp.ndarray, cfg: "Config") -> tuple:
    # next-token loss/acc of `codelm` over its own input sequence (codelm_upper: the CodeLM one level up
    # over the top code -- the decoderless level's only objective)
    h = pardec_context_hidden(codelm, None, raw, cfg, 0, None, group_size=1)
    shift = 1 + codelm.attn_lookahead
    if h.shape[1] <= shift:
        return jnp.array(0.0, dtype=h.dtype), jnp.array(0.0, dtype=h.dtype)
    tgt = target[:, shift:]
    logits = codelm_ntp_logits_tf(codelm, h[:, :-shift, :], tgt)
    loss = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tgt[..., None], axis=-1))
    return loss, jnp.mean(jnp.argmax(logits, -1) == tgt)


def encode_pardec_downsampler_generate(codelm: CodeLM, downsampler: PardecLM, raw: jnp.ndarray, K: int,
                                        cfg: "Config", rate_id: int = 0, rng=None, greedy: bool = True,
                                        temperature: float = 1.0, top_k: int = 0,
                                        downsampler_ncodes: int = 1, codelm_rate_id: int = None) -> dict:
    # Generation-time counterpart of encode_pardec_downsampler: that function is TEACHER-FORCED
    # (needs label_fn's real ground-truth target, via pardec_score), so it can't be used for actual
    # free-run generation (generate_from_prompt's upward-encode loop, no ground truth available at
    # inference). Mirrors _decode_generate_pardec_call's pattern for the upsampler direction, just
    # using pardec_generate (autoregressive, no external target) instead of pardec_score. Caught
    # 2026-09-26: generate_from_prompt was instead calling CodeLM.encode()'s own naive code_head
    # path here, which use_pardec_downsampler=True training never touches/trains at all -- that's
    # what made cascade generation garbage despite good training loss (loss trains the downsampler
    # via encode_pardec_downsampler; generation was reading a permanently-untrained code_head).
    # Same bos substitution as encode_pardec_downsampler/CodeLM.encode() -- keeps generation-time
    # distribution consistent with whatever codelm_bos_prob training actually used. codelm_rate_id:
    # see encode_pardec_downsampler's own docstring -- CodeLM's bos is always absolute, independent
    # of `rate_id` (the downsampler's own bos, which follows cfg.bos_rate_mode).
    codelm_rate_id = rate_id if codelm_rate_id is None else codelm_rate_id
    h = pardec_context_hidden(codelm, downsampler, raw, cfg, codelm_rate_id, rng, group_size=K * downsampler_ncodes)
    code_idx = pardec_generate(downsampler, h, context_group_size=K * downsampler_ncodes,
                                output_group_size=downsampler_ncodes, rng=rng,
                                greedy=greedy, temperature=temperature, top_k=top_k, rate_id=rate_id,
                                output_expansion=1)
    code_soft = jax.nn.one_hot(code_idx, codelm.code_vocab, dtype=h.dtype)
    return dict(code_soft=code_soft, code_idx=code_idx)


def decode_logits_and_target_multipass(model: "LagCodecModel", level_idx: int, target_seq: jnp.ndarray,
                                        ctx_code_soft: jnp.ndarray, upsampler_ncodes: int, rng=None,
                                        encode_temperature: float = 1.0, force_teacher_forced: bool = False,
                                        return_passes: bool = False, cycle_ctx: list = None, **_unused) -> tuple:
    # return_passes=True: list of every level-refine pass's 5-tuple (pass 1 first) instead of the last.
    # cycle_ctx: stack-mode cycle slots (see pardec_score), same in every refine pass.
    # SHARED upsampler, modulated by (output_expansion=K(level_idx), rate_id=level_idx) -- not a
    # separate weight set per level. **_unused absorbs any stale kwargs from callers.
    # No special-casing: the Upsampler's context is built EXACTLY like the Downsampler's, per
    # cfg.context_source (see pardec_context_hidden) -- "codelm" (default): run this level's own
    # code through CodeLM's own embed table + block stack to get real contextualized hidden states.
    # "own_embed"/"shared_embed": a dedicated ctx_embed/ctx_proj lookup shortcut, no CodeLM pass.
    codelm = model.context_codelm_for(level_idx)
    codelm_rate_id = model.context_codelm_rate_id(level_idx)  # CodeLM's own bos: always absolute
    rate_id = model.bos_rate_id(level_idx)  # upsampler's own bos: follows cfg.bos_rate_mode
    upsampler = model.upsampler_for(level_idx)
    cfg = model.cfg
    h_ctx = pardec_context_hidden(codelm, upsampler, ctx_code_soft, cfg,
                                   codelm_rate_id, rng, group_size=upsampler_ncodes)
    oe = model.K(level_idx)

    def _teacher_forced(**dkw):
        return pardec_score(upsampler, target_seq, h_ctx, context_group_size=upsampler_ncodes,
                             output_group_size=upsampler_ncodes, rate_id=rate_id, output_expansion=oe, **dkw)

    def _rollout(**dkw):
        # upsampler_rollout: like downsampler_rollout, but for the upsampler's own digit-AR head --
        # every digit-level AR step is SELF-FED (token_ar_rollout) during training, not teacher-forced
        # (token_ar_teacher_forced via pardec_score) -- mitigates the train/inference mismatch where
        # pardec_generate's sequential self-feeding (used at real generation time) is otherwise never
        # exercised during training at all. hidden doesn't depend on target_seq's own values (causal
        # masking), so the real target_seq can be passed straight through unlike downsampler_rollout's
        # dummy-zeros target (which lacks a real label_fn target handy at this call site anyway).
        hid = pardec_score(upsampler, target_seq, h_ctx, context_group_size=upsampler_ncodes,
                            output_group_size=upsampler_ncodes, rate_id=rate_id, output_expansion=oe,
                            return_hidden=True, **dkw)
        valid_len = hid.shape[1]
        t_out = target_seq[:, :valid_len]
        qfn = lambda lg_m, k_: quantize_dispatch(cfg.quantize_mode, lg_m,
                                                  k_ if rng is not None else None,
                                                  encode_temperature, cfg.quantize_drop)
        _, _, lg = token_ar_rollout(upsampler.token_in_proj, upsampler.token_member_embed,
                                     upsampler.token_norm1, upsampler.token_attn, upsampler.token_ln_f,
                                     upsampler.token_out_head, upsampler.output_chunks, hid,
                                     rng if rng is not None else jax.random.PRNGKey(0), qfn)
        aux_loss = jnp.array(0.0, dtype=lg.dtype)
        aux_acc = jnp.array(0.0, dtype=lg.dtype)
        return lg, t_out, aux_loss, aux_acc

    def _one_pass(**dkw):
        if force_teacher_forced or not cfg.upsampler_rollout:
            return _teacher_forced(**dkw)
        if rng is None or cfg.upsampler_rollout_prob >= 1.0:
            return _rollout(**dkw)
        if cfg.upsampler_rollout_prob <= 0.0:
            return _teacher_forced(**dkw)
        use_roll = jax.random.bernoulli(jax.random.fold_in(rng, 9), p=cfg.upsampler_rollout_prob)
        _canon = lambda t: (t[0].astype(h_ctx.dtype), t[1].astype(jnp.int32),
                             t[2].astype(h_ctx.dtype), t[3].astype(h_ctx.dtype))
        return jax.lax.cond(use_roll, lambda: _canon(_rollout(**dkw)), lambda: _canon(_teacher_forced(**dkw)))

    n_ss = 1 if force_teacher_forced else pss_n_passes(cfg.upsampler_pss_passes[level_idx], upsampler_ncodes * oe,
                                                        upsampler.decode_past)
    _tf_or_rollout = _one_pass

    def _one_pass(**dkw):
        # pss: re-feed the previous pass's own prediction as the row's token inputs; only the last pass is scored
        out = _tf_or_rollout(**dkw)
        for _ in range(1, n_ss):
            out = _tf_or_rollout(input_seq=pss_inputs(out[0], out[1], cfg, rng, cfg.upsampler_pss_prob), **dkw)
        return out

    n_pass = cfg.level_refine_passes[level_idx]
    Pp = cfg.level_refine_window[level_idx] * upsampler_ncodes * oe
    fixed = n_pass > 1 and cfg.level_refine_layout == "fixed"
    fill = "mask" if fixed else "zero"
    ckw = dict(cycle_ctx=cycle_ctx) if cycle_ctx else {}
    passes = [_one_pass(draft_len=Pp, draft_fill="mask", **ckw) if fixed else _one_pass(**ckw)]
    if n_pass > 1:
        for p in range(1, n_pass):
            prev_logits, prev_target = passes[-1][0], passes[-1][1]
            if cfg.level_refine_draft_mode == "sample" and rng is not None:
                lg = prev_logits.astype(jnp.float32) / cfg.level_refine_draft_temperature
                lg = lg + jax.random.gumbel(jax.random.fold_in(rng, 20 + p), lg.shape)
            else:
                lg = prev_logits
            draft = safe_argmax(lg).astype(jnp.int32)  # feeds a gather: avoid TPU argmax bug
            if cfg.level_refine_gt_drop < 1.0 and rng is not None:
                own = jax.random.bernoulli(jax.random.fold_in(rng, 10 + p), p=cfg.level_refine_gt_drop,
                                           shape=draft.shape[:2])
                draft = jnp.where(own[..., None], draft, prev_target.astype(jnp.int32))
            passes.append(_one_pass(draft_seq=jax.lax.stop_gradient(draft), draft_len=Pp, draft_fill=fill, **ckw))
    outs = [(lg, t, None, al, ac) for lg, t, al, ac in passes]
    return outs if return_passes else outs[-1]


def _decode_generate_pardec_call(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                  draft_seq=None, draft_len=0, draft_fill="zero", cycle_ctx=None, rng=None):
    # rng (traced key) overrides PRNGKey(seed) -- for free-run decodes inside a training step
    codelm = model.context_codelm_for(level_idx)
    codelm_rate_id = model.context_codelm_rate_id(level_idx)  # CodeLM's own bos: always absolute
    rate_id = model.bos_rate_id(level_idx)  # upsampler's own bos: follows cfg.bos_rate_mode
    rng = jax.random.PRNGKey(seed) if rng is None else rng
    h_ctx = pardec_context_hidden(codelm, model.upsampler_for(level_idx), ctx_idx, model.cfg,
                                   codelm_rate_id, rng, group_size=upsampler_ncodes)
    ckw = dict(cycle_ctx=cycle_ctx) if cycle_ctx else {}
    return pardec_generate(model.upsampler_for(level_idx), h_ctx, context_group_size=upsampler_ncodes,
                            output_group_size=upsampler_ncodes, rng=rng, greedy=greedy, temperature=temperature,
                            top_k=model.cfg.gen_top_k, rate_id=rate_id, output_expansion=model.K(level_idx),
                            draft_seq=draft_seq, draft_len=draft_len, draft_fill=draft_fill, **ckw)


_decode_generate_pardec_jit = eqx.filter_jit(_decode_generate_pardec_call)


def decode_generate_multipass(model: "LagCodecModel", level_idx: int, ctx_idx: jnp.ndarray, upsampler_ncodes: int,
                               greedy: bool = True, temperature: float = 1.0, seed: int = 0,
                               cycle_ctx: list = None, rng=None) -> jnp.ndarray:
    n_pass = model.cfg.level_refine_passes[level_idx]
    Pp = model.cfg.level_refine_window[level_idx] * upsampler_ncodes * model.K(level_idx)
    fixed = n_pass > 1 and model.cfg.level_refine_layout == "fixed"
    fill = "mask" if fixed else "zero"
    ckw = dict(cycle_ctx=cycle_ctx, rng=rng) if (cycle_ctx or rng is not None) else {}
    if fixed:
        pred = _decode_generate_pardec_jit(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                           None, Pp, "mask", **ckw)
    else:
        pred = _decode_generate_pardec_jit(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed,
                                           **ckw)
    for _ in range(n_pass - 1):
        pred = _decode_generate_pardec_jit(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature,
                                           seed, pred, Pp, fill, **ckw)
    return pred


def cycle_reencode(model: "LagCodecModel", level_idx: int, tokens: jnp.ndarray, rng=None,
                   encode_temperature: float = 1.0) -> tuple:
    # downsampler i's code for level-i tokens, self-fed (never label_fn/GT), digits per quantize_mode (argmax
    # without rng), differentiable via its estimator; dense like training's encode (downsampler_ncodes=1)
    cfg = model.cfg
    codelm, ds = model.codelm_for(level_idx), model.downsampler_for(level_idx)
    K = model.K(level_idx)
    h = pardec_context_hidden(codelm, ds, tokens, cfg, model.codelm_bos_rate_id(level_idx), rng, group_size=K)
    n_blocks = h.shape[1] // K
    hid = pardec_score(ds, jnp.zeros((h.shape[0], n_blocks, ds.output_chunks), jnp.int32), h,
                       context_group_size=K, output_group_size=1, rate_id=model.bos_rate_id(level_idx),
                       return_hidden=True)
    qfn = lambda lg, k_: quantize_dispatch(cfg.quantize_mode, lg, k_, encode_temperature, cfg.quantize_drop)
    if ds.token_head == "linear":
        return qfn(reshape_pq(hid @ ds.output_head_linear, ds.output_chunks, ds.output_vocab), rng)
    if rng is None:
        qfn = lambda lg, k_: quantize_dispatch(cfg.quantize_mode, lg, None, encode_temperature, cfg.quantize_drop)
    cs, ci, _ = token_ar_rollout(ds.token_in_proj, ds.token_member_embed, ds.token_norm1, ds.token_attn,
                                 ds.token_ln_f, ds.token_out_head, ds.output_chunks, hid,
                                 jax.random.PRNGKey(0) if rng is None else rng, qfn)
    return cs, ci


def decode_logits_and_target_cycles(model: "LagCodecModel", level_idx: int, target_seq: jnp.ndarray,
                                    ctx_code_soft: jnp.ndarray, upsampler_ncodes: int, rng=None,
                                    encode_temperature: float = 1.0, force_teacher_forced: bool = False) -> list:
    # one multipass passes-list per cycle (cycle 0 = the plain decode, same rng); between cycles the
    # level_cycle_input tokens are re-encoded into c(t+1): memoryless decodes from it alone, stack adds a slot
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
        rng_t = rng if (t == 0 or rng is None) else jax.random.fold_in(rng, 100 + t)
        passes = decode_logits_and_target_multipass(model, level_idx, target_seq, ctx_t, upsampler_ncodes,
                                                    rng=rng_t, cycle_ctx=slots, **kw)
        cycles.append(passes)
        if t == n_cyc - 1:
            break
        sub = (lambda s: None) if rng_t is None else (lambda s: jax.random.fold_in(rng_t, s))
        if cfg.level_cycle_input == "gt":
            tokens = target_seq
        elif cfg.level_cycle_input == "pss":
            tokens = safe_argmax(passes[-1][0]).astype(jnp.int32)
            if cfg.level_cycle_pss_prob < 1.0 and rng_t is not None:
                own = jax.random.bernoulli(sub(31), p=cfg.level_cycle_pss_prob, shape=tokens.shape[:2])
                tokens = jnp.where(own[..., None], tokens, passes[-1][1].astype(jnp.int32))
        else:  # rollout: free-run decode under this cycle's exact conditioning, as at generation
            greedy = rng_t is None or cfg.quantize_mode == "argmax"
            tokens = decode_generate_multipass(model, level_idx, jax.lax.stop_gradient(ctx_t), upsampler_ncodes,
                                               greedy=greedy, temperature=1.0,
                                               cycle_ctx=jax.lax.stop_gradient(slots), rng=sub(30))
        tokens = jax.lax.stop_gradient(tokens)
        rev_soft, _ = cycle_reencode(model, level_idx, tokens, sub(32), encode_temperature)
        if cfg.level_cycle_detach:
            rev_soft = jax.lax.stop_gradient(rev_soft)
        if stack:
            slots = list(slots)
            slots[t] = pardec_context_hidden(model.context_codelm_for(level_idx), model.upsampler_for(level_idx),
                                             rev_soft, cfg, model.context_codelm_rate_id(level_idx), rng_t,
                                             group_size=upsampler_ncodes)
        else:
            ctx_t = rev_soft
    return cycles


def _cycle_reencode_generate(model, level_idx, tokens, upsampler_ncodes, greedy, temperature, seed, stack):
    # generation-time re-encode: downsampler's own generate path (as gen-eval encodes), + slot hidden
    codelm = model.codelm_for(level_idx)
    out = encode_pardec_downsampler_generate(codelm, model.downsampler_for(level_idx), tokens, model.K(level_idx),
                                              model.cfg, rate_id=model.bos_rate_id(level_idx),
                                              rng=jax.random.PRNGKey(seed), greedy=greedy, temperature=temperature,
                                              top_k=model.cfg.gen_top_k, downsampler_ncodes=1,
                                              codelm_rate_id=model.codelm_bos_rate_id(level_idx))
    if not stack:
        return out["code_idx"], None
    h = pardec_context_hidden(model.context_codelm_for(level_idx), model.upsampler_for(level_idx), out["code_idx"],
                              model.cfg, model.context_codelm_rate_id(level_idx), None, group_size=upsampler_ncodes)
    return out["code_idx"], h


_cycle_reencode_generate_jit = eqx.filter_jit(_cycle_reencode_generate)


def decode_generate_cycles(model: "LagCodecModel", level_idx: int, ctx_idx: jnp.ndarray, upsampler_ncodes: int,
                           greedy: bool = True, temperature: float = 1.0, seed: int = 0) -> jnp.ndarray:
    # generation counterpart of decode_logits_and_target_cycles (gen_level_cycles cycles, free-run)
    cfg = model.cfg
    n_cyc = cfg.gen_level_cycles[level_idx]
    n_slots = cycle_stack_slots(cfg, level_idx)  # trained with slots -> always decode with them (masked until filled)
    if n_cyc <= 1 and n_slots == 0:
        return decode_generate_multipass(model, level_idx, ctx_idx, upsampler_ncodes, greedy, temperature, seed)
    stack = cfg.level_cycle_mode == "stack"
    slots = [None] * n_slots if stack else None
    ctx_t = ctx_idx
    for t in range(n_cyc):
        pred = decode_generate_multipass(model, level_idx, ctx_t, upsampler_ncodes, greedy, temperature,
                                         seed if t == 0 else seed + 100 + t, cycle_ctx=slots)
        if t == n_cyc - 1:
            return pred
        code_idx, h = _cycle_reencode_generate_jit(model, level_idx, pred, upsampler_ncodes, greedy, temperature,
                                                   seed + 1000 * (t + 1), stack)
        if stack:
            slots = list(slots)
            slots[t] = h
        else:
            ctx_t = code_idx


def encoder_hidden(codelm: CodeLM, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for blk in codelm.blocks:
        h = run_block(blk, h, False)
    return codelm.ln_f(h)


def encoder_ntp_logits(codelm: CodeLM, h: jnp.ndarray) -> jnp.ndarray:
    # linear head only: an AR head needs the token's earlier digits, use codelm_ntp_logits_tf /
    # codelm_sample_next for that.
    assert codelm.token_head == "linear", "encoder_ntp_logits is linear-head only"
    return reshape_pq(h @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab)


def codelm_ntp_logits_tf(codelm: CodeLM, h: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    # Teacher-forced NTP logits (..., chunks, V): h (..., D) predicts `target` (..., chunks), the
    # NEXT token. Linear: parallel head. AR: digit m conditioned on the real digits <m of `target`.
    if codelm.token_head == "linear":
        return reshape_pq(h @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab)
    return token_ar_teacher_forced(codelm.tok_in_proj, codelm.tok_member_embed, codelm.tok_norm1,
                                    codelm.tok_attn, codelm.tok_ln_f, codelm.tok_out_head,
                                    codelm.token_dim, codelm.code_vocab, h, target)


def codelm_sample_next(codelm: CodeLM, h_prev: jnp.ndarray, rng, greedy: bool, temperature,
                       top_k: int = 0) -> jnp.ndarray:
    # Sample the next token (B, chunks) from the last hidden state h_prev (B, D).
    if codelm.token_head == "linear":
        return _sample_tokens(reshape_pq(h_prev @ codelm.ntp_head, codelm.pq_chunks, codelm.code_vocab),
                              rng, greedy, temperature, top_k)
    idx, _ = token_ar_generate(codelm.tok_in_proj, codelm.tok_member_embed, codelm.tok_norm1,
                                codelm.tok_attn, codelm.tok_ln_f, codelm.tok_out_head,
                                codelm.pq_chunks, h_prev, rng, greedy, temperature, top_k)
    return idx


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
    # tokens (B,total_len,C) holds the prompt in [:P] (P may be traced); the rest is overwritten.
    # Real incremental KV cache (was: full-buffer recompute every step, O(T^2)) -- same blk.step
    # pattern as pardec_generate. Prefill scans the WHOLE fixed-length buffer once (positions >= P
    # hold junk/placeholder embeddings at that point); each generated position's cache entry is
    # then overwritten with its real embedding via a second self_step call at the same pos, before
    # any later (strictly causal) position ever attends to it -- Block.step/Attention.step write
    # via jax.lax.dynamic_update_slice at the given pos, so a repeat call at the same pos correctly
    # replaces the placeholder rather than appending.
    B, total_len, C = tokens.shape
    x = code_embed_proj(tokens, codelm.own_input_embed, codelm.own_input_proj)
    if use_bos:
        # position 0's discrete `tokens[:,0]` value is meaningless once its embedding is replaced --
        # only the embedding matters for this level's own free-run (P=1, everything after is genuinely
        # free-sampled with zero real content: a true unconditional/free rollout, not just a small
        # real prompt). Downstream re-encoding of the returned tokens is the caller's concern.
        x = x.at[:, 0, :].set(codelm.bos_embed[rate_id])
    # recurrent blocks can't overwrite a placeholder: their prefill only consumes the prompt (pos < P)
    prompt_valid = jnp.broadcast_to((jnp.arange(total_len) < P)[None], (B, total_len))

    def self_step(x_new, caches, pos, prefill=False):
        new_caches = []
        h = x_new
        for blk, c in zip(codelm.blocks, caches):
            valid = prompt_valid if prefill and isinstance(blk, RecurrentBlock) else None
            h, c = block_step(blk, h, c, pos, total_len, valid)
            new_caches.append(c)
        return codelm.ln_f(h), new_caches

    caches0 = [block_cache_init(blk, B, total_len) for blk in codelm.blocks]

    def prefill_step(caches, x_t_and_pos):
        x_t, pos = x_t_and_pos
        h, caches = self_step(x_t, caches, pos, prefill=True)
        return caches, h

    positions = jnp.arange(total_len)
    caches, h_all = jax.lax.scan(prefill_step, caches0, (jnp.swapaxes(x, 0, 1), positions))
    h_prev0 = jax.lax.dynamic_index_in_dim(h_all, P - 1, axis=0, keepdims=False)

    def body(t, carry):
        tokens, caches, h_prev = carry
        tok = codelm_sample_next(codelm, h_prev, jax.random.fold_in(rng, t), greedy, temperature, top_k)
        tokens = tokens.at[:, t].set(tok.astype(tokens.dtype))
        x_new = code_embed_proj(tok, codelm.own_input_embed, codelm.own_input_proj)
        h_new, caches = self_step(x_new, caches, t)
        return tokens, caches, h_new

    tokens, _, _ = jax.lax.fori_loop(P, total_len, body, (tokens, caches, h_prev0))
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
    codelm0 = model.codelm_for(0)
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    K0 = model.K(0)
    B, P, _ = prompt_bytes.shape
    assert P % K0 == 0, f"prompt length {P} must be a multiple of the level-0 stride {K0}"
    tok = byte_pq_fn(prompt_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    for i in range(sample_level):
        codelm_i = model.codelm_for(i)
        tok = encode_pardec_downsampler_generate(codelm_i, model.downsampler_for(i), tok, model.K(i), cfg,
                                                   rate_id=model.bos_rate_id(i),
                                                   codelm_rate_id=model.codelm_bos_rate_id(i),
                                                   rng=jax.random.fold_in(rng, 100000 + i), greedy=greedy,
                                                   temperature=encode_temperature,
                                                   downsampler_ncodes=cfg.downsampler_ncodes[i])["code_idx"]
        assert tok.shape[1] >= 1, "prompt too short to produce a single code at the sampling level"
    ds = 1
    for i in range(sample_level):
        ds *= model.K(i)
    # use_bos: matches training's own position-0 substitution -- with codelm_bos_prob=1.0, training
    # NEVER saw real content at position 0 for ANY level (including sample_level), so generation
    # must substitute bos here too or it's off-distribution exactly where it matters most (caught
    # 2026-09-27: this call previously always fed the real prompt token at position 0 regardless of
    # codelm_bos_prob). For 0<codelm_bos_prob<1, neither True nor False matches training's actual
    # random mix exactly; False (real content) is kept as the closer default since that's what an
    # actual prompted continuation naturally has.
    use_bos = cfg.use_codelm_bos and cfg.codelm_bos_prob >= 1.0
    codelm_L = model.codelm_for(sample_level)
    tokens_L = encoder_free_run(codelm_L, tok, total_positions // ds, model.K(sample_level), rng, greedy,
                                 temperature, top_k, use_bos=use_bos,
                                 rate_id=model.codelm_bos_rate_id(sample_level))

    codes, raw = {}, tokens_L
    for i in range(sample_level, n):
        codelm_i = model.codelm_for(i)
        out = encode_pardec_downsampler_generate(codelm_i, model.downsampler_for(i), raw, model.K(i), cfg,
                                                   rate_id=model.bos_rate_id(i),
                                                   codelm_rate_id=model.codelm_bos_rate_id(i),
                                                   rng=jax.random.fold_in(rng, 200000 + i), greedy=greedy,
                                                   temperature=encode_temperature,
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
    if sample_level == 0:
        res["sampled"] = tokens_L
    else:
        res["sampled"] = cascade(tokens_L, sample_level - 1)
    return res


class LagCodecModel(eqx.Module):
    # Shared mode stores one CodeLM/Downsampler; unshared mode stores one per level.
    codelms: tuple
    downsamplers: tuple
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        n = len(cfg.strides)
        n_instances = 1 if cfg.share_across_levels else n
        codelms, downsamplers = [], []
        for j in range(n_instances):
            level_idx = j  # share=True: n_instances==1, so level_idx is always 0 (the representative
            # index singleton_uniform_fields already enforces uniformity against); share=False:
            # level_idx==j, this instance's own dedicated level.
            codelms.append(CodeLM(jax.random.fold_in(key, 10 + j), cfg, level_idx=level_idx))
            D_enc = cfg.codelm_d_model[level_idx]
            pq_dim, code_vocab, pq_chunks = cfg.pq_dim[level_idx], cfg.code_vocab[level_idx], cfg.pq_chunks[level_idx]
            scheme, use_xsa, use_qknorm = cfg.init_scheme, cfg.use_xsa, cfg.use_qknorm
            # n_rates: share=True needs one bos row per DISTINCT rate (bos_rate_map/bos_n_rates,
            # dedup by effective stride in "relative" mode, one per level index in "absolute");
            # share=False instances are already level-dedicated, so they only need their own row.
            n_rates = bos_n_rates(cfg) if cfg.share_across_levels else 1

            downsampler_remat = cfg.remat if cfg.downsampler_remat[level_idx] is None else cfg.downsampler_remat[level_idx]
            downsampler = PardecLM(
                jax.random.fold_in(key, 20 + j), context_hidden_dim=D_enc, hidden_dim=cfg.downsampler_d_model[level_idx],
                n_heads=cfg.downsampler_n_heads[level_idx], n_kv_heads=cfg.downsampler_n_kv_heads[level_idx],
                n_layers=cfg.downsampler_n_layers[level_idx], mlp_mult=cfg.mlp_mult[level_idx], rope_base=cfg.rope_base[level_idx],
                output_expansion=1, context_window_groups=cfg.downsampler_window[level_idx],
                output_vocab=code_vocab, output_chunks=pq_chunks, pq_dim=pq_dim,
                token_dim=cfg.token_dim[level_idx], token_n_heads=cfg.token_n_heads[level_idx],
                decode_past=cfg.downsampler_decode_past[level_idx], decode_future=cfg.downsampler_decode_future[level_idx],
                n_rates=n_rates, init_scheme=scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, remat=downsampler_remat,
                remat_chunks=cfg.downsampler_remat_chunks[level_idx],
                ctx_vocab=code_vocab, ctx_pq_chunks=pq_chunks, ctx_pq_dim=pq_dim, token_head=cfg.pardec_token_head,
                backbone=cfg.downsampler_backbone[level_idx], state_dim=cfg.ssm_state_dim)
            downsamplers.append(downsampler)

        if cfg.context_source == "codelm_upper" and not cfg.share_across_levels:
            # CodeLM-only context level used by the top encoder.
            codelms.append(CodeLM(jax.random.fold_in(key, 1000), cfg, level_idx=n - 1))
        self.codelms = tuple(codelms)
        self.downsamplers = tuple(downsamplers)

    def codelm_for(self, level_idx: int) -> CodeLM:
        return self.codelms[0] if self.cfg.share_across_levels else self.codelms[level_idx]

    def downsampler_for(self, level_idx: int) -> "PardecLM":
        return self.downsamplers[0] if self.cfg.share_across_levels else self.downsamplers[level_idx]

    def bos_rate_id(self, level_idx: int) -> int:
        # Downsampler's own BOS identifies its contraction rate.
        # share_across_levels=True: rate_id comes from bos_rate_map (relative: dedup by effective
        # stride; absolute: the raw level index). False: each instance already owns exactly one
        # level structurally, always row(s) 0.
        return bos_rate_map(self.cfg)[level_idx] if self.cfg.share_across_levels else 0

    def codelm_bos_rate_id(self, level_idx: int) -> int:
        # CodeLM's OWN bos (anchors "which level am I generating" for encoder_free_run's pure-
        # unconditional rollout) -- ALWAYS absolute (one row per level index), regardless of
        # cfg.bos_rate_mode. Two levels sharing the same effective stride are still semantically
        # distinct free-run targets (different resolution/content statistics), so this must NOT dedup
        # by rate the way bos_rate_id does for downsampler/upsampler (see CodeLM.__init__'s own
        # comment on n_bos_rates). share_across_levels=False: each instance already owns exactly one
        # level structurally, always row(s) 0.
        return level_idx if self.cfg.share_across_levels else 0

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


def weighted_level_mean(losses: list, weights, level_indices=None) -> jnp.ndarray:
    if not losses:
        return jnp.array(0.0)
    stacked = jnp.stack(losses)
    if weights is None:
        return jnp.mean(stacked)
    if level_indices is None:
        selected_weights = weights[:len(losses)]
    else:
        if len(level_indices) != len(losses):
            raise ValueError("level_indices must have one entry per loss")
        if any(i < 0 or i >= len(weights) for i in level_indices):
            raise ValueError(f"level index exceeds configured loss weights: {level_indices}")
        selected_weights = [weights[i] for i in level_indices]
    w = jnp.asarray(selected_weights, dtype=stacked.dtype)
    denom = jnp.maximum(jnp.sum(w), 1e-8)
    return jnp.sum(stacked * w) / denom


def cycle_loss(cycles: list, mode: str) -> jnp.ndarray:
    # mean over refine passes, then over cycles ("all") or the last cycle only
    per = [jnp.mean(jnp.stack([dec_loss_acc(lg, t, m)[0] for lg, t, m, _, _ in passes])) for passes in cycles]
    return per[-1] if (mode == "last" or len(per) == 1) else jnp.mean(jnp.stack(per))


def level_forward(model: LagCodecModel, flat_bytes: jnp.ndarray, phase: int, rng=None,
                   level_gt_drop=None, cascade_rng=None, encode_temperature: float = 1.0,
                   layer_drop_prob=None, label_reg_weight: float = 0.0, label_fn=None,
                   pixel_order=None, byte_pq_fn=None, digit_teacher_force: bool = False,
                   return_recon: bool = False, return_levelwise: bool = False,
                   ctx_ablation: str = None) -> tuple:
    if not model.cfg.encoder_only_pretrain:
        raise ValueError("the pretrain runner supports encoder-only training")
    if return_recon or ctx_ablation is not None:
        raise ValueError("decoder reconstruction and context ablation are unavailable in encoder-only pretraining")
    codelm0 = model.codelm_for(0)
    byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
    # flat_bytes stays raw (fed to label_fn as-is, which interprets literal byte/pixel values);
    # tok0 is the SAME raw bytes converted into CodeLM's own (pq_chunks, code_vocab) categorical
    # representation via byte_pq_fn -- user-settable, defaults to rgb_byte_pq_fn. Level 0's own
    # input/NTP-target and levels>0's own input/NTP-target now share this exact representation,
    # which is what lets a single shared CodeLM process every level.
    tok0 = byte_pq_fn(flat_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    codes, codes_soft = [], []
    enc_losses, enc_accs, utils, entropy_losses, label_losses = [], [], [], [], []
    label_mses, label_mse_losses = [], []
    level_rngs = [None] * phase if rng is None else list(jax.random.split(rng, phase))
    for i in range(phase):
        codelm = model.codelm_for(i)
        out = encode_pardec_downsampler(codelm, model.downsampler_for(i), raw, target, flat_bytes, model.cfg,
                                         pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                         codelm_rate_id=model.codelm_bos_rate_id(i),
                                         rng=level_rngs[i], downsampler_ncodes=model.cfg.downsampler_ncodes[i],
                                        pss_passes=model.cfg.downsampler_pss_passes[i])
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
            raw = out["code_soft"]
            target = out["code_idx"]
    if model.cfg.context_source == "codelm_upper":
        ntp_up, acc_up = codelm_ntp_loss(model.codelm_for(phase), codes_soft[phase - 1], codes[phase - 1], model.cfg)
        enc_losses.append(ntp_up)
        enc_accs.append(acc_up)

    encoder_levels = list(range(phase))
    if model.cfg.context_source == "codelm_upper":
        encoder_levels.append(phase)
    ntp_loss_total = weighted_level_mean(
        enc_losses, model.cfg.encoder_level_loss_weights, encoder_levels)
    entropy_loss_total = jnp.mean(jnp.stack(entropy_losses))
    label_loss_total = jnp.mean(jnp.stack(label_losses)) if label_losses else 0.0
    label_mse_total = jnp.mean(jnp.stack(label_mses)) if label_mses else jnp.array(0.0)
    label_mse_loss_total = jnp.mean(jnp.stack(label_mse_losses)) if label_mse_losses else 0.0
    loss = model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
        + label_reg_weight * label_loss_total + model.cfg.label_mse_weight * label_mse_loss_total
    zero = jnp.array(0.0, dtype=ntp_loss_total.dtype)
    aux = (zero, zero, ntp_loss_total, jnp.mean(jnp.stack(enc_accs)),
           jnp.mean(jnp.stack(utils)), zero, zero, zero,
           label_mse_total)
    if return_levelwise:
        dec_zeros = jnp.zeros((phase,), dtype=ntp_loss_total.dtype)
        aux = aux + (jnp.stack(enc_losses), jnp.stack(enc_accs), dec_zeros, dec_zeros)
    return loss, aux


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


def sample_level_range(py_rng, probs: tuple) -> tuple:
    # cfg.level_select_prob-driven sampler: `probs` has length n-1 (one Bernoulli per transition,
    # the last level never needs its own "stop" flip -- there's nowhere left to walk). Reused for
    # BOTH the start-walk and the end-walk (two independent passes over the same tuple), per the
    # design discussion 2026-10-01: sound and simpler than sample_multires_entry's
    # randrange(1, n_levels-entry_level), which is an EMPTY range (ValueError) whenever
    # entry_level==n_levels-1 and had to pre-exclude that case entirely, artificially disallowing
    # "train the top level alone". Each flip here is an independent, always-valid Bernoulli draw;
    # reaching a boundary without a "stop" draw just forces the decision there, so every (s, e) pair
    # with 0<=s<=e<n -- INCLUDING s==e at any level, including n-1 -- is reachable by construction,
    # never by exclusion. Returns (entry_level, depth) to match sample_multires_entry's own
    # call-site convention (depth = e - s + 1), not (s, e) directly.
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
    # Runs the REAL encoder chain through levels 0..upto_level-1 (upto_level steps), returning the
    # resulting (code_idx, code_soft) -- level `upto_level`'s own real native input, as the shared
    # encoder actually produces it end-to-end, not label_fn's resize-based shortcut. Used only by
    # level_forward_multires's entry_gt_drop blend below; the caller stop_gradients the result (no
    # update onto these levels from this particular usage -- see Config.multires_entry_gt_drop).
    codelm0 = model.codelm_for(0)
    tok0 = byte_pq_fn(flat_bytes, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    level_rngs = [None] * upto_level if rng is None else list(jax.random.split(rng, upto_level))
    code_idx = code_soft = None
    for i in range(upto_level):
        codelm = model.codelm_for(i)
        out = encode_pardec_downsampler(codelm, model.downsampler_for(i), raw, target, flat_bytes, cfg,
                                         pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                         codelm_rate_id=model.codelm_bos_rate_id(i), rng=level_rngs[i],
                                         downsampler_ncodes=cfg.downsampler_ncodes[i],
                                        pss_passes=cfg.downsampler_pss_passes[i])
        code_idx, code_soft = out["code_idx"], out["code_soft"]
        raw, target = code_soft, code_idx
    return code_idx, code_soft


def level_forward_multires(model: LagCodecModel, flat_bytes: jnp.ndarray, entry_level: int, depth: int,
                            rng=None, encode_temperature: float = 1.0, label_reg_weight: float = 0.0,
                            label_fn=None, pixel_order=None, byte_pq_fn=None, entry_gt_drop: float = None) -> tuple:
    # Same idea as level_forward, but the encode cascade starts at entry_level (not always 0) and
    # runs only `depth` further steps. entry_level=0 is equivalent to level_forward(..., phase=depth)
    # in spirit (though the aux tuple shape differs slightly, see below). entry_level>0's input is
    # NOT produced by running the model's own (real) encoder up to that depth -- it's synthesized
    # directly from the real image via the same resize+quantize label_fn already uses for aux
    # targets, standing in for "a native input at this level's resolution". With
    # cfg.share_across_levels=True, every level reuses the SAME shared CodeLM/downsampler/upsampler
    # (see LagCodecModel) -- this is what actually trains those shared parameters across every
    # resolution they need to work at, not just the one fixed depth level_forward always uses.
    if not model.cfg.encoder_only_pretrain:
        raise ValueError("the pretrain runner supports encoder-only training")
    if model.cfg.encoder_only_pretrain:
        codelm_entry = model.codelm_for(entry_level)
        byte_pq_fn = byte_pq_fn or rgb_byte_pq_fn
        if entry_level == 0:
            entry_code = byte_pq_fn(flat_bytes, codelm_entry.pq_chunks, codelm_entry.code_vocab)
            raw, target = entry_code, entry_code
        else:
            n_blocks_entry = n_blocks_for_level(model.cfg, entry_level - 1)
            label_shortcut = label_fn(flat_bytes, model.cfg, pixel_order, n_blocks_entry,
                                       model.cfg.pq_chunks[entry_level - 1], model.cfg.code_vocab[entry_level - 1])
            raw, target = label_shortcut, label_shortcut
        entry_code = target
        codes, codes_soft = [], []
        enc_losses, enc_accs, utils, entropy_losses, label_losses, label_mses, label_mse_losses = \
            [], [], [], [], [], [], []
        level_rngs = [None] * depth if rng is None else list(jax.random.split(rng, depth))
        for d in range(depth):
            i = entry_level + d
            codelm = model.codelm_for(i)
            out = encode_pardec_downsampler(codelm, model.downsampler_for(i), raw, target, flat_bytes, model.cfg,
                                             pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                             codelm_rate_id=model.codelm_bos_rate_id(i),
                                             rng=level_rngs[d], downsampler_ncodes=model.cfg.downsampler_ncodes[i],
                                            pss_passes=model.cfg.downsampler_pss_passes[i])
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
                label_probs_i = jax.nn.softmax(enc_logits, axis=-1)
                label_values_i = jnp.arange(label_probs_i.shape[-1], dtype=label_probs_i.dtype)
                pred_label_soft = jnp.sum(label_probs_i * label_values_i, axis=-1)
                label_mse_losses.append(jnp.mean((pred_label_soft - label_tgt.astype(jnp.float32)) ** 2))
            if d < depth - 1:
                raw = out["code_soft"]
                target = out["code_idx"]
        if model.cfg.context_source == "codelm_upper":
            ntp_up, acc_up = codelm_ntp_loss(model.codelm_for(entry_level + depth), codes_soft[depth - 1],
                                             codes[depth - 1], model.cfg)
            enc_losses.append(ntp_up)
            enc_accs.append(acc_up)
        encoder_levels = list(range(entry_level, entry_level + depth))
        if model.cfg.context_source == "codelm_upper":
            encoder_levels.append(entry_level + depth)
        ntp_loss_total = weighted_level_mean(
            enc_losses, model.cfg.encoder_level_loss_weights, encoder_levels)
        entropy_loss_total = jnp.mean(jnp.stack(entropy_losses))
        label_loss_total = jnp.mean(jnp.stack(label_losses)) if label_losses else 0.0
        label_mse_total = jnp.mean(jnp.stack(label_mses)) if label_mses else jnp.array(0.0)
        label_mse_loss_total = jnp.mean(jnp.stack(label_mse_losses)) if label_mse_losses else 0.0
        loss = model.cfg.ntp_weight * ntp_loss_total + model.cfg.entropy_weight * entropy_loss_total \
            + label_reg_weight * label_loss_total + model.cfg.label_mse_weight * label_mse_loss_total
        bpb = jnp.array(0.0, dtype=jnp.float32)
        byte_acc = jnp.array(0.0, dtype=jnp.float32)
        aux = (bpb, byte_acc, ntp_loss_total, jnp.mean(jnp.stack(enc_accs)),
               jnp.mean(jnp.stack(utils)), jnp.array(0.0, dtype=jnp.float32),
               jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), label_mse_total)
        if model.cfg.log_levelwise_metrics:
            aux = aux + (jnp.stack(enc_losses), jnp.stack(enc_accs), jnp.stack([jnp.array(0.0, dtype=jnp.float32)] * len(enc_losses)),
                         jnp.stack([jnp.array(0.0, dtype=jnp.float32)] * len(enc_accs)))
        return loss, aux

def phase_trainable_filter(model: LagCodecModel, phase: int):
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


def scalar_float(x):
    arr = np.asarray(local_array(x))
    if arr.size == 0:
        return 0.0
    if arr.ndim == 0:
        return float(arr)
    return float(np.mean(arr))


def device_mean_array(x):
    arr = np.asarray(local_array(x))
    if arr.size == 0:
        return arr
    if arr.ndim == 0:
        return arr.reshape(())
    if arr.shape[0] == 1:
        return arr[0]
    return np.mean(arr, axis=0)


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
    eqx.tree_serialise_leaves(ckpt_dir / "encoder.eqx", (model.codelms, model.downsamplers))
    eqx.tree_serialise_leaves(ckpt_dir / "opt_state.eqx", opt_state)
    eqx.tree_serialise_leaves(ckpt_dir / "p_rng.eqx", p_rng)
    (ckpt_dir / "dataloader_state.json").write_text(json.dumps(dict(
        epoch_rng_state=train_iter.epoch_rng.bit_generator.state,
        epoch_seed=train_iter.epoch_seed, pos=train_iter.pos)))
    meta = dict(phase=phase, phase_step=phase_step, step=step, seed=seed)
    if schedule_meta is not None:
        meta["schedule"] = schedule_meta
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))


def load_encoder_only_checkpoint(model, ckpt_path: Path):
    encoder_path = ckpt_path / "encoder.eqx" if ckpt_path.is_dir() else ckpt_path
    if encoder_path.name == "encoder.eqx" and encoder_path.is_file():
        loaded_encoder = eqx.tree_deserialise_leaves(
            encoder_path, (model.codelms, model.downsamplers))
        return eqx.tree_at(lambda m: (m.codelms, m.downsamplers), model,
                           replace=loaded_encoder, is_leaf=lambda x: False)

    model_path = ckpt_path / "model.eqx" if ckpt_path.is_dir() else ckpt_path
    if not model_path.is_file():
        raise FileNotFoundError(f"encoder checkpoint needs encoder.eqx or model.eqx: {ckpt_path}")
    legacy_model = full_runner.LagCodecModel(jax.random.PRNGKey(0), model.cfg)
    loaded_encoder = eqx.tree_deserialise_leaves(model_path, legacy_model)
    return eqx.tree_at(lambda m: (m.codelms, m.downsamplers), model,
                       replace=(loaded_encoder.codelms, loaded_encoder.downsamplers),
                       is_leaf=lambda x: False)


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
    codelm = model.codelm_for(level)
    flat_raw = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    codelm0 = model.codelm_for(0)
    tok0 = rgb_byte_pq_fn(flat_raw, codelm0.pq_chunks, codelm0.code_vocab)
    # For level>0, `level`'s own real input is level (level-1)'s own REAL encoded code output, not
    # the raw image re-fed directly (that was the bug, fixed 2026-09-28: feeding raw 1024-position
    # bytes straight into level1's downsampler -- which was only ever trained on level0's own
    # 256-length code sequence -- ran it wildly out-of-distribution, producing the "hot pixel dot"
    # codegrid artifact). Chain through the real encode cascade (same pattern as level_forward's own
    # encode loop) to get the genuine input this level actually sees during training/generation.
    flat, raw = tok0, tok0
    for i in range(level):
        codelm_i = model.codelm_for(i)
        out_i = encode_pardec_downsampler(codelm_i, model.downsampler_for(i), raw, flat, flat_raw, cfg,
                                           pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                           codelm_rate_id=model.codelm_bos_rate_id(i),
                                           downsampler_ncodes=cfg.downsampler_ncodes[i],
                                        pss_passes=cfg.downsampler_pss_passes[i])
        flat = out_i["code_idx"]
        raw = out_i["code_soft"]
    # Must dispatch exactly like level_forward's own encode call (line ~1809) -- otherwise this
    # diagnostic reads code_idx from CodeLM's own untrained code_head/quantize path while training
    # actually optimizes the separate PardecLM-based encode_pardec_downsampler, silently plotting
    # noise from a never-trained pathway (caught 2026-09-26: loss/label_mse looked fine in
    # run.log, but this plot showed pure random-color noise for pred downsample).
    out = encode_pardec_downsampler(codelm, model.downsampler_for(level), raw, flat, flat_raw, cfg,
                                     pixel_order, label_fn, model.K(level), rate_id=model.bos_rate_id(level),
                                     codelm_rate_id=model.codelm_bos_rate_id(level),
                                     downsampler_ncodes=cfg.downsampler_ncodes[level],
                                        pss_passes=cfg.downsampler_pss_passes[level])
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
        # label_fn ALWAYS resizes from the true raw image (flat_raw), regardless of level -- `flat`
        # is this level's own chained input (level-1's code for level>0, not the raw image), so
        # passing it here crashed for level>0 (wrong length, e.g. 256 instead of img_size**2).
        label_tgt = np.asarray(label_fn(flat_raw, cfg, pixel_order, n_blocks, cfg.pq_chunks[level], cfg.code_vocab[level]))
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


CONFIG_FIELDS = ("img_size", "modality", "seq_len", "audio_sample_rate", "audio_encoding", "codelm_d_model", "codelm_remat", "codelm_n_layers", "codelm_n_heads", "codelm_n_kv_heads",
                  "downsampler_d_model", "downsampler_n_layers", "downsampler_n_heads",
                  "downsampler_n_kv_heads", "downsampler_window",
                  "downsampler_decode_past", "downsampler_decode_future", "downsampler_remat", "downsampler_ncodes",
                  "upsampler_d_model", "upsampler_n_layers", "upsampler_n_heads",
                  "upsampler_n_kv_heads", "upsampler_window",
                  "upsampler_decode_past", "upsampler_decode_future", "upsampler_remat",
                  "downsampler_remat_chunks", "upsampler_remat_chunks",
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
                  "upsampler_pss_passes", "downsampler_pss_passes", "upsampler_pss_prob", "downsampler_pss_prob",
                  "pss_input_mode", "pss_temperature",
                  "codelm_backbone", "downsampler_backbone", "upsampler_backbone", "ssm_state_dim",
                  "precision", "curriculum_mode", "quantize_mode", "quantize_drop",
                  "gumbel_at_inference", "init_scheme", "use_xsa",
                  "use_qknorm", "remat", "remat_level", "attn_window", "attn_lookahead",
                  "encoder_attn_window", "decoder_attn_window", "use_sink",
                  "use_codelm_bos", "codelm_bos_prob", "codelm_bos_rates",
                  "byte_group", "token_head_type", "token_dim", "token_n_heads", "token_mask_prob", "pq_dim",
                  "entropy_weight", "mse_weight",
                  "mse_softmax_tau", "traversal", "label_reg_weight", "label_mse_weight",
                  "encoder_level_loss_weights", "decoder_level_loss_weights",
                  "log_levelwise_metrics", "log_levelwise_eval", "log_levelwise_gen",
                  "encoder_only_pretrain", "load_encoder_checkpoint")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "imagenet64", "imagenet256", "folder"],
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
    p.add_argument("--encoder_only_pretrain", type=lambda x: x.lower() != "false", default=True,
                    help="ignored in this runner: only CodeLM/downsampler parameters are constructed and trained")
    p.add_argument("--load_encoder_checkpoint", type=Path, default=None,
                    help="start fresh from CodeLM/downsampler weights in an encoder.eqx or full checkpoint")
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
    p.add_argument("--encoder_level_loss_weights", type=_float_tuple_arg, default=None,
                    help="optional per-level weights for the encoder codelm losses. If unset, all levels are weighted equally")
    p.add_argument("--decoder_level_loss_weights", type=_float_tuple_arg, default=None,
                    help="optional per-level weights for the decoder upsampler losses. If unset, all levels are weighted equally")
    p.add_argument("--log_levelwise_metrics", type=lambda x: x.lower() != "false", default=False,
                    help="include level-wise encoder/decoder loss and accuracy arrays in training/val logs")
    p.add_argument("--log_levelwise_eval", type=lambda x: x.lower() != "false", default=False,
                    help="log per-level val losses/accuracies in the eval summary")
    p.add_argument("--log_levelwise_gen", type=lambda x: x.lower() != "false", default=False,
                    help="log per-level generation metrics when running gen-eval")
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
    p.add_argument("--modality", type=str, default=Config.modality, choices=list(MODALITIES),
                    help="image | text | audio | binary (1D seq_len x byte_group byte sequences); --dataset folder "
                         "reads every file under --data_root (train/ + val/ subfolders, else every 20th file is val)")
    p.add_argument("--seq_len", type=int, default=Config.seq_len, help="positions per sample, non-image modalities")
    p.add_argument("--audio_sample_rate", type=int, default=Config.audio_sample_rate)
    p.add_argument("--audio_encoding", type=str, default=Config.audio_encoding, choices=["mulaw8", "pcm16"])
    p.add_argument("--codelm_d_model", type=_tuple_arg, default=Config.codelm_d_model)
    p.add_argument("--codelm_n_layers", type=_tuple_arg, default=Config.codelm_n_layers)
    p.add_argument("--codelm_n_heads", type=_tuple_arg, default=Config.codelm_n_heads)
    p.add_argument("--codelm_n_kv_heads", type=_tuple_arg, default=Config.codelm_n_kv_heads)
    p.add_argument("--codelm_remat", type=_opt_bool_tuple_arg, default=Config.codelm_remat,
                    help="CodeLM's OWN per-block remat per level; 'none' (default) falls back to --remat")
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
    for name in ("downsampler_remat_chunks", "upsampler_remat_chunks"):
        p.add_argument(f"--{name}", type=_tuple_arg, default=getattr(Config, name),
                        help="per level: run the pardec stack over this many row chunks, each rematerialized "
                             "(peak activations ~1/chunks, one extra forward). 1 = off")
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
                    choices=("codelm", "codelm_upper", "own_embed", "shared_embed"),
                    help="How downsampler/upsampler get their context. 'codelm' (default): CodeLM's "
                         "own contextualized hidden states (encoder_hidden). 'own_embed': a plain "
                         "per-position embedding table, no self-attention, own table per module. "
                         "'shared_embed': same, but downsampler and upsampler share one table. 'codelm_upper': "
                         "upsampler i's context from CodeLM i+1 (+ a CodeLM-only top level when not shared).")
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
    p.add_argument("--upsampler_pss_passes", type=_tuple_arg, default=Config.upsampler_pss_passes,
                    help="per level: parallel scheduled sampling passes on the upsampler's own token inputs "
                         "(1 = off, -1 = one per row token = exact rollout inputs); loss on the last pass")
    p.add_argument("--downsampler_pss_passes", type=_tuple_arg, default=Config.downsampler_pss_passes,
                    help="same for the downsampler (only matters when downsampler_ncodes > 1)")
    p.add_argument("--upsampler_pss_prob", type=float, default=Config.upsampler_pss_prob,
                    help="pss: per-position prob of own prediction (else GT)")
    p.add_argument("--downsampler_pss_prob", type=float, default=Config.downsampler_pss_prob)
    p.add_argument("--pss_input_mode", type=str, default=Config.pss_input_mode, choices=["argmax", "sample"])
    p.add_argument("--pss_temperature", type=float, default=Config.pss_temperature)
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
    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    label_fn_raw = config_vars.pop("label_fn", None)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    # helper constants (e.g. DEPTH = 4) are allowed: warn and ignore; imports/functions are ignored silently
    consts = sorted(k for k in unknown if not callable(config_vars[k]) and not isinstance(config_vars[k], type(argparse)))
    if consts:
        warnings.warn(f"--config {pre_args.config}: ignoring non-field constant(s) {consts}")
    config_vars = {k: v for k, v in config_vars.items() if k in known}
    p.set_defaults(**config_vars)
    args = p.parse_args()
    if not bool(args.encoder_only_pretrain):
        warnings.warn("--encoder_only_pretrain is ignored in the pretrain runner; encoder-only mode is always forced on.")
    args.encoder_only_pretrain = True
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
    label_fn = resolve_label_fn(label_fn_raw, cfg.modality)
    n_levels = len(cfg.strides)
    top_level_trainable = cfg.strides[-1] != -1
    n_phases = n_levels if top_level_trainable else n_levels - 1
    n_positions = n_positions_of(cfg)
    pixel_order = pixel_order_for(cfg)
    if args.level_select_prob is not None:
        assert len(args.level_select_prob) == n_levels - 1, \
            f"--level_select_prob needs {n_levels - 1} entries (n_levels-1 transitions), " \
            f"got {len(args.level_select_prob)}"
    if args.multires_entry_gt_drop is not None:
        assert len(args.multires_entry_gt_drop) == n_levels, \
            f"--multires_entry_gt_drop needs {n_levels} entries (indexed by entry_level), " \
            f"got {len(args.multires_entry_gt_drop)}"

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

    # "skip-ahead" curriculum check: a phase has nonzero steps while an EARLIER phase has zero --
    # e.g. level_steps=(0,0,0,1_000_000) jumps straight to training all levels jointly, with levels
    # 0..2 never trained standalone/staged first. This is a deliberate, valid pattern (cifar_res_4.py
    # and its forks use it on purpose), so it's only a warning by default -- but it means every level
    # past the first is learning its own conditional generation problem from scratch simultaneously,
    # with no warm-start transfer between levels (especially relevant with share_across_levels=False,
    # where each level is fully independent weights). If generation quality degrades sharply at
    # higher levels while training loss/acc look fine, check here first before suspecting a
    # correctness bug in the generation code itself.
    _phase_steps = args.level_steps if args.level_steps is not None else args.level_epochs
    skip_ahead = any(any(_phase_steps[j] == 0 for j in range(i)) and _phase_steps[i] != 0
                      for i in range(len(_phase_steps)))
    if skip_ahead:
        msg = (f"level_steps/level_epochs={_phase_steps}: a later phase runs with nonzero steps while "
               f"an earlier phase was skipped (0) -- those lower levels are never trained standalone "
               f"before the joint phase trains them all simultaneously (no staged warm start). "
               f"If --require_staged_curriculum is set, refusing to start.")
        if args.require_staged_curriculum:
            raise ValueError(msg)
        warnings.warn(msg)
    if cfg.curriculum_mode == "freeze":
        assert not args.no_curriculum and args.level_select_prob is None, \
            "curriculum_mode='freeze' needs the phase-by-phase curriculum (no --no_curriculum / level_select_prob)"
        if skip_ahead:
            raise ValueError("curriculum_mode='freeze' with a skipped phase would freeze an untrained level")
        if any(g > 0 for g in args.level_gt_drop):
            warnings.warn(f"curriculum_mode='freeze' with level_gt_drop={args.level_gt_drop}: frozen lower "
                          f"decoders get predicted ctx from the new level and can't adapt to it")

    (train_np, train_labels), (val_np, val_labels) = load_dataset(
        args.dataset, Path(args.data_root), cfg.img_size if cfg.modality == "image" else None, cfg=cfg)
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
    if args.load_encoder_checkpoint is not None:
        if args.resume:
            raise ValueError("--resume and --load_encoder_checkpoint cannot be used together")
        ckpt_path = Path(args.load_encoder_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"encoder checkpoint not found: {ckpt_path}")
        model = load_encoder_only_checkpoint(model, ckpt_path)
        logger(f"loaded CodeLM + Downsampler state from {ckpt_path}")

    cfg.encoder_only_pretrain = True
    cfg.load_encoder_checkpoint = str(args.load_encoder_checkpoint) if args.load_encoder_checkpoint is not None else None

    compute_dtype = jnp.bfloat16 if cfg.precision == "bf16" else jnp.float32
    if cfg.precision != "bf16":
        jax.config.update("jax_default_matmul_precision", "highest")
    def token_preview(tokens: np.ndarray, level: int) -> np.ndarray:
        if level == 0:
            return positions_to_image(tokens, cfg, pixel_order)
        count = tokens.shape[1]
        side = math.isqrt(count)
        if side * side != count or tokens.shape[-1] < 3:
            raise ValueError(f"level {level}: cannot render {count} positions with {tokens.shape[-1]} chunks")
        vocab = cfg.code_vocab[level]
        rgb_seq = (tokens[0, :, :3].astype(np.float32) * (255.0 / max(1, vocab - 1))).clip(0, 255).astype(np.uint8)
        order = zorder_pixel_order(side) if cfg.traversal == "zorder" else np.arange(count)
        raster = np.zeros_like(rgb_seq)
        raster[order] = rgb_seq
        grid = raster.reshape(side, side, 3)
        rows = np.linspace(0, side - 1, cfg.img_size).round().astype(int)
        cols = np.linspace(0, side - 1, cfg.img_size).round().astype(int)
        return grid[rows[:, None], cols[None, :]][None]

    def teacher_force_level0(m, tokens: jnp.ndarray) -> jnp.ndarray:
        codelm = m.codelm_for(0)
        h = pardec_context_hidden(m.codelm_for(0), m.downsampler_for(0), tokens, cfg,
                                  m.codelm_bos_rate_id(0), None,
                                  group_size=m.K(0) * cfg.downsampler_ncodes[0])
        logits = codelm_ntp_logits_tf(codelm, h[:, :-1], tokens[:, 1:])
        predicted = jnp.argmax(logits, axis=-1).astype(tokens.dtype)
        return tokens.at[:, 1:].set(predicted)

    def run_qual_eval(eval_model, phase: int, tag: str) -> None:
        m = cast_pytree(eval_model, compute_dtype)
        sources = [("train", train_np[:1]), ("val", val_np[:1])]
        for source_name, images in sources:
            gt_image = images.astype(np.uint8)
            flat = jnp.asarray(images_to_positions(images, cfg, pixel_order))
            raw = rgb_byte_pq_fn(flat, m.codelm_for(0).pq_chunks, m.codelm_for(0).code_vocab)
            level_tokens = []
            for level in range(phase):
                key = jax.random.PRNGKey(args.seed + level * 1009 + (0 if source_name == "val" else 1))
                out = encode_pardec_downsampler_generate(
                    m.codelm_for(level), m.downsampler_for(level), raw, m.K(level), cfg,
                    rate_id=m.bos_rate_id(level), codelm_rate_id=m.codelm_bos_rate_id(level),
                    rng=key, greedy=True, temperature=1.0, top_k=0,
                    downsampler_ncodes=cfg.downsampler_ncodes[level])
                raw = out["code_idx"]
                level_tokens.append(raw)

            for level, tokens in enumerate(level_tokens):
                preview_gt = token_preview(np.asarray(tokens), level)
                for requested_prefix in (1, 128):
                    prefix = min(requested_prefix, tokens.shape[1])
                    generated = encoder_free_run(
                        m.codelm_for(level), tokens[:, :prefix], tokens.shape[1], m.K(level),
                        jax.random.PRNGKey(args.seed + level * 101 + prefix),
                        greedy=True, temperature=1.0, top_k=0,
                        rate_id=m.codelm_bos_rate_id(level))
                    preview_gen = token_preview(np.asarray(generated), level)
                    sample_path = run_dir / f"samples_{tag}_{source_name}_level{level}_prompt{prefix}.png"
                    save_samples(preview_gen, preview_gt, sample_path, cfg)
                    logger(f"[{tag}] QUAL source={source_name} level={level} "
                           f"prompt_positions={prefix}/{tokens.shape[1]} kv_cache=true "
                           f"saved={sample_path.name}")

                if level == 0:
                    tf_tokens = teacher_force_level0(m, tokens)
                    tf_image = positions_to_image(np.asarray(tf_tokens), cfg, pixel_order)
                    mse = pixel_mse(tf_image, gt_image)
                    save_samples(tf_image, gt_image,
                                 run_dir / f"samples_{tag}_{source_name}_level0_teacher_force.png", cfg)
                    logger(f"[{tag}] TF_SANITY source={source_name} level=0 mse={mse:.3f}",
                           tag=tag, source=source_name, tf_sanity_mse=float(mse))

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
        enc_level_losses_sum = None
        enc_level_accs_sum = None
        dec_level_losses_sum = None
        dec_level_accs_sum = None
        for start in range(0, n, bs):
            batch_imgs = val_np[start:start + bs]
            bn = len(batch_imgs)
            batch_flat = jnp.array(images_to_positions(batch_imgs, cfg, pixel_order))
            batch_t0 = time.monotonic()
            loss_b, aux_b = val_eval_jit(m, batch_flat, phase, rng=None,
                                          encode_temperature=args.encode_temperature[phase - 1],
                                          label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                          pixel_order=pixel_order,
                                          return_levelwise=cfg.log_levelwise_eval or cfg.log_levelwise_metrics)
            if not val_jit_timed[0]:
                val_compile_s = time.monotonic() - batch_t0
                val_jit_timed[0] = True
            if cfg.log_levelwise_eval or cfg.log_levelwise_metrics:
                aux_vals = list(aux_b)
                if len(aux_vals) >= 13:
                    enc_level_losses_b = np.asarray(aux_vals[-4])
                    enc_level_accs_b = np.asarray(aux_vals[-3])
                    dec_level_losses_b = np.asarray(aux_vals[-2])
                    dec_level_accs_b = np.asarray(aux_vals[-1])
                    if enc_level_losses_sum is None:
                        enc_level_losses_sum = np.zeros_like(enc_level_losses_b, dtype=np.float64)
                        enc_level_accs_sum = np.zeros_like(enc_level_accs_b, dtype=np.float64)
                        dec_level_losses_sum = np.zeros_like(dec_level_losses_b, dtype=np.float64)
                        dec_level_accs_sum = np.zeros_like(dec_level_accs_b, dtype=np.float64)
                    enc_level_losses_sum += bn * enc_level_losses_b
                    enc_level_accs_sum += bn * enc_level_accs_b
                    dec_level_losses_sum += bn * dec_level_losses_b
                    dec_level_accs_sum += bn * dec_level_accs_b
            sums += bn * np.array([float(a) for a in aux_b[:9]])
            total_loss += bn * float(loss_b)
            total_n += bn
        dec_loss, dec_acc, enc_loss, enc_acc, util, val_mse, _aux_ntp_bpb, aux_ntp_acc, val_label_mse = \
            (sums / total_n).tolist()
        loss = total_loss / total_n
        val_time_s = time.monotonic() - val_t0
        levelwise_suffix = ""
        rec = dict(tag=tag, val_loss=loss,
                   val_dec_loss=dec_loss, val_dec_acc=dec_acc,
                    val_enc_loss=enc_loss, val_enc_acc=enc_acc,
                    val_util=util, val_mse=val_mse,
                    val_d_ntp_acc=aux_ntp_acc, val_label_mse=val_label_mse,
                    val_time_s=val_time_s)
        if cfg.log_levelwise_eval or cfg.log_levelwise_metrics:
            val_enc_level_losses = (enc_level_losses_sum / total_n).tolist()
            val_enc_level_accs = (enc_level_accs_sum / total_n).tolist()
            val_dec_level_losses = (dec_level_losses_sum / total_n).tolist()
            val_dec_level_accs = (dec_level_accs_sum / total_n).tolist()
            rec.update(dict(val_enc_level_losses=val_enc_level_losses,
                            val_enc_level_accs=val_enc_level_accs,
                            val_dec_level_losses=val_dec_level_losses,
                            val_dec_level_accs=val_dec_level_accs))
            levelwise_suffix = (
                " enc_levels=" + ",".join(f"{i}:{x:.3f}" for i, x in enumerate(val_enc_level_losses)) +
                " dec_levels=" + ",".join(f"{i}:{x:.3f}" for i, x in enumerate(val_dec_level_losses))
            )
        msg = (f"[{tag}] VAL loss={loss:.2f} val_dec_loss={dec_loss:.2f} val_dec_acc={dec_acc:.2f} val_mse={val_mse:.4f} "
               f"val_enc_acc={enc_acc:.2f} val_d_ntp_acc={aux_ntp_acc:.2f} "
               f"val_label_mse={val_label_mse:.2f}{levelwise_suffix} val_time={val_time_s:.1f}s")
        if val_compile_s is not None:
            msg += f" (first batch, incl. jit compile: {val_compile_s:.1f}s)"
            rec["val_compile_s"] = val_compile_s
        logger(msg, **rec)
        return dec_loss, dec_acc

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
    step_w = len(str(total_all_steps))
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
    # level_select_prob's any-level training (see sample_level_range): ONE persistent python Random
    # across the whole run (not re-seeded per phase), so distinct phases don't replay the same
    # (entry_level, depth) sequence. Plain python random, not jax -- (entry_level, depth) must be
    # static ints chosen BEFORE jax.jit traces each step, same constraint `phase` already has.
    multires_py_rng = random.Random(args.seed)
    for phase in phase_iter:
        # --level_steps[phase-1]==0 (or --level_epochs[phase-1]==0): skip this phase ENTIRELY before
        # any setup work (BatchIterator, jit) -- e.g. cifar_res_4.py's level_steps=(0,)*4+(100000,)
        # trains only the last phase. (epoch-based 0 already skips the inner step loop today, but
        # still pays for the BatchIterator/pmap setup; the explicit level_steps=0 case is the one
        # that matters for cheaply skipping many phases, so only that one is special-cased here.)
        if args.level_steps is not None and args.level_steps[phase - 1] == 0:
            logger(f"level{phase - 1}: level_steps=0, skipping phase entirely")
            continue
        if args.level_epochs is not None and args.level_epochs[phase - 1] == 0:
            logger(f"level{phase - 1}: level_epochs=0, skipping phase entirely")
            continue
        train_iter = BatchIterator(train_np, train_labels[:len(train_np)], args.batch_size[phase - 1],
                                    n_devices, shuffle=True, seed=args.seed, cfg=cfg)

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
                                     pixel_order=pixel_order,
                                     return_levelwise=cfg.log_levelwise_metrics)

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
            # no gradient tying needed: with cfg.share_across_levels=True (default), codelm_for/
            # downsampler_for resolve to the SAME singleton instance regardless of
            # level index -- level_forward calling it at several different level indices within this
            # one trace already gets correctly-summed gradients from JAX's autodiff, same as any
            # other value used multiple times in one function. This is why LagCodecModel stores a
            # length-1 tuple in that mode rather than the same instance repeated at N positions --
            # duplicating it INTO the pytree would NOT tie weights (see Config.share_across_levels).
            grad_norm = optax.global_norm(grads)
            aux = aux + (grad_norm,)
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            return diff_model, opt_state, rng, loss, aux

        # donate model/opt_state/rng: replaced by the outputs every step (frees the old copies in the update)
        train_step = jax.pmap(train_step, axis_name="d", donate_argnums=(0, 1, 2))

        # --level_select_prob: any-level training (see sample_level_range/level_forward_multires).
        # (entry_level, depth) must be static per trace (same constraint `phase` already has), so
        # each distinct pair gets its own pmap'd closure, built lazily and cached -- repeat pairs
        # reuse the already-compiled step, same amortization `phase` itself already relies on.
        multires_active = args.level_select_prob is not None
        multires_step_cache = {}

        def _build_multires_step(entry_level, depth, static_model=static_model):
            entry_gt_drop_sd = (args.multires_entry_gt_drop[entry_level]
                                 if args.multires_entry_gt_drop is not None and entry_level > 0 else None)

            def loss_fn_sd(diff_model, static_model, flat_bytes, rng, cascade_rng):
                m = eqx.combine(diff_model, static_model)
                m = cast_pytree(m, compute_dtype)
                return level_forward_multires(m, flat_bytes, entry_level=entry_level, depth=depth, rng=rng,
                                               encode_temperature=encode_temperature_phase,
                                               label_reg_weight=cfg.label_reg_weight, label_fn=label_fn,
                                               pixel_order=pixel_order, entry_gt_drop=entry_gt_drop_sd)

            def train_step_sd(diff_model, opt_state, rng, flat_bytes, static_model=static_model):
                rng, level_rng, cascade_rng = jax.random.split(rng, 3)
                (loss, aux), grads = jax.value_and_grad(loss_fn_sd, has_aux=True)(
                    diff_model, static_model, flat_bytes, level_rng, cascade_rng)
                grads = jax.lax.pmean(grads, axis_name="d")
                loss = jax.lax.pmean(loss, axis_name="d")
                aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
                grad_norm = optax.global_norm(grads)
                aux = aux + (grad_norm,)
                updates, opt_state = optimizer.update(grads, opt_state, diff_model)
                diff_model = eqx.apply_updates(diff_model, updates)
                return diff_model, opt_state, rng, loss, aux

            return jax.pmap(train_step_sd, axis_name="d", donate_argnums=(0, 1, 2))

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

        active_desc = f"level{phase - 1}" if not multires_active else f"anylevel(phase{phase})"
        logger(f"=== starting {active_desc} for {phase_total_steps / steps_per_epoch_lr:.3g} "
               f"epochs ({phase_total_steps} steps) ===" +
               (f" -- any-level training active, level_select_prob={args.level_select_prob}"
                if multires_active else ""))

        steps_per_epoch = len(train_iter)
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
                if multires_active:
                    s_lvl, d_lvl = sample_level_range(multires_py_rng, args.level_select_prob)
                    step_fn = multires_step_cache.get((s_lvl, d_lvl))
                    if step_fn is None:
                        step_fn = _build_multires_step(s_lvl, d_lvl)
                        multires_step_cache[(s_lvl, d_lvl)] = step_fn
                    p_diff_model, p_opt_state, p_rng, loss, aux = step_fn(p_diff_model, p_opt_state, p_rng, flat)
                else:
                    p_diff_model, p_opt_state, p_rng, loss, aux = train_step(p_diff_model, p_opt_state, p_rng, flat)
                step += 1
                phase_step += 1
                pbar.update(1)
                loss0 = float(local_array(loss)[0])
                if not jit_timed:
                    logger(f"{active_desc}: first train_step (incl. jit compile) took "
                           f"{time.monotonic() - jit_t0:.1f}s")
                    jit_timed = True
                if cfg.log_levelwise_metrics:
                    scalar_aux = [scalar_float(a) for a in aux[:9]]
                    dec_loss, dec_acc, enc_loss, enc_acc, util, train_mse, _aux_ntp_bpb, aux_ntp_acc, label_mse = scalar_aux
                    enc_level_losses, enc_level_accs, dec_level_losses, dec_level_accs = [
                        np.asarray(device_mean_array(a), dtype=np.float64) for a in aux[9:13]
                    ]
                    grad_norm = scalar_float(aux[13])
                else:
                    dec_loss, dec_acc, enc_loss, enc_acc, util, train_mse, _aux_ntp_bpb, aux_ntp_acc, label_mse, grad_norm = \
                        [scalar_float(a) for a in aux[:10]]
                    enc_level_losses = enc_level_accs = dec_level_losses = dec_level_accs = None
                lr = float(lr_schedule(step - 1))
                lr_str = _fmt_lr(lr)
                pbar.set_postfix(step=step, loss=f"{loss0:.2f}",
                                  acc=f"{enc_acc:.2f}",
                                  lr=lr_str, gnorm=f"{grad_norm:.2f}")
                if step % args.log_every == 0:
                    log_kwargs = dict(level=phase - 1, epoch=epoch_num, step=step, loss=loss0,
                                      enc_loss=enc_loss, enc_acc=enc_acc, util=util,
                                      mse=train_mse, label_mse=label_mse,
                                      lr=lr, grad_norm=grad_norm)
                    if cfg.log_levelwise_metrics:
                        log_kwargs.update(dict(enc_level_losses=enc_level_losses.tolist(),
                                               enc_level_accs=enc_level_accs.tolist()))
                    levelwise_suffix = ""
                    if cfg.log_levelwise_metrics:
                        enc_vals = np.asarray(enc_level_losses).reshape(-1)
                        enc_str = "[" + ", ".join(f"{x:.2f}" for x in enc_vals) + "]"
                        levelwise_suffix = f" enc_losses={enc_str}"
                        enc_vals = np.asarray(enc_level_accs).reshape(-1)
                        enc_str = "[" + ", ".join(f"{x:.2f}" for x in enc_vals) + "]"
                        levelwise_suffix += f" enc_accs={enc_str}"
                    logger(f"l={phase - 1} e={epoch_num} s={step} "
                           f"loss={loss0:.2f} "
                           f"enc_loss={enc_loss:.2f} enc_acc={enc_acc:.2f} "
                           f"util={util:.2f} mse={train_mse:.1f} "
                           f"label_mse={label_mse:.2f} "
                           f"lr={lr_str} grad_norm={grad_norm:.2f}{levelwise_suffix}",
                           **log_kwargs)

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

    global_pbar.update(step - last_global_step)
    global_pbar.close()
    logger("=== all phases done, running final teacher-forced validation metrics ===")
    run_val_eval(model, n_phases, tag="final")
    run_qual_eval(model, n_phases, tag="final")
    logger("training done")


if __name__ == "__main__":
    main()
