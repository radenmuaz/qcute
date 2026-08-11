"""Sanity check, per explicit user request: does v4.5's UNCOND (encode-only) pathway overfit
cleanly when decode gets ZERO loss weight (decode_ntp_weight=0.0)?

Motivation: v4.5's decode Stage 0 reuses encode_lms[i]'s own already-computed `h` directly as
input (see LevelLM.encode's docstring) -- so even under share_level_weights=False (independent
parameter TENSORS for encode vs decode), decode's loss still backprops through that shared `h`
into encode_lms[i]'s parameters, pulling them toward two objectives at once (being good at plain
uncond NTP, AND being a good Stage-0 starting point for decode's conditioned NTP). This is a real
structural difference from v4.4 no-sharing, where decode computes from raw x_list[i] through its
OWN independent embedding -- ZERO shared computation graph with encode at all. Suspected as the
reason v4.5's uncond generation looked uniformly collapsed across every checkpoint tested (see
docs/status.md's "v4.5 uncond generation survey"), even ones whose CONDITIONED generation was
fine. decode_ntp_weight=0.0 makes every decode-derived loss term contribute a zero-scaled
gradient (the forward computation still runs, but 0*loss's backward pass is exactly zero) --
fully decoupling encode_lms's training from decode's forward pass. If uncond generation cleans
up dramatically here vs the normal (decode_ntp_weight=1.0) runs, that confirms the theory.

share_level_weights=False (user: skip the shared-weights variant for this task).
decode_code_ste=False (detach), use_gumbel_noise=False (no gumbel) -- both per explicit request.
Single-level Ks=(4,) twin: qcute_refine_v4_5_uncondonly_k4_1.py (2-level).

    uv run python -m qcute.qcute_refine_v4_5 --config configs/qcute_refine_v4_5_uncondonly_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_uncondonly_k4single/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0
decode_ntp_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
qual_prompt_bytes = 64
