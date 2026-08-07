"""qcute.qcute_refine_v2 config: CLONE of
configs/qcute_refine_v2_byte4_code256_simple.py, one change —
quant_type="identity" instead of "bsq" — run as a diagnostic CEILING
BASELINE, QUEUED (do not launch until both of the following have
happened, in order):

  1. qcute_refine_v2_byte4_code256_simple finishes training (currently
     running — session observation: "current seems flat", i.e. its
     bpb/acc curve looks like it may be plateauing rather than still
     improving — this run exists to check whether BSQ's hard
     discretization is *why*).
  2. scripts/probe_decoder_kv_contribution.py has been run against that
     finished checkpoint (already written and CPU-verified this session,
     not yet run against a real trained checkpoint).

quant_type="identity": EncoderLevel's code_pre output is used AS c_i
directly — no BSQ hypersphere-corner quantization, no STE, unbounded
continuous values. This is the SAME "ceiling baseline" methodology
qcutelm.py's own history already used (docs/status.md: "quant identity
first as ceiling baseline... isolates whether hard discretization is the
slow-convergence bottleneck" for the *original* FSQ/BSQ tokenizer
design) — same question, asked of this fork's own BSQ usage.

Deliberately does NOT also zero code_ntp_weight/tok_weight the way
Config's own docstring flags as the only "sound" combination — that
caution was written for a stricter isolation (removing quantization
*and* every downstream consumer of it at once, so a representation
mismatch could never manifest at all). Here the goal is different: keep
EVERYTHING ELSE — architecture, loss terms, optimizer, steps — identical
to the currently-running baseline, changing only the ONE variable
(quantization) under test, so any bpb difference is attributable to that
change alone. Sanity-checked this session (CPU, small model): forward +
backward run cleanly, no NaN, with code_ntp_weight=tok_weight=1.0 still
on — the "mismatch" the stricter caution warns about is a modeling-
choice concern (predicting sign-of-unbounded-continuous-value instead of
sign-of-a-proper-bounded-bit), not a crash/instability one.

Everything else — Ks=(4,4), context_len=1024, attn_window=(256,64),
tier_d_models=(256,256), byte_repr="embed", code_head_mode="independent",
optimizer/LR/steps matched to bytelm_xs_mtp4_ctx1024 — unchanged from
qcute_refine_v2_byte4_code256_simple.py.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_byte4_code256_identity.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_byte4_code256_identity
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
quant_type = "identity"
cross_attn_rope = True   # explicitly pinned (session ask: "for identity exp, use cross attn rope") —
                          # already Config's own default, pinned here for clarity/reproducibility

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000   # cut from 8000 (session finding: both bytelm_xs_mtp4_ctx1024 and bpelm_32768 hit
                # their own best val_bpb well before step 2000 and are fully overfit/plateaued by
                # step 4000 — running past that just burns wall-clock without adding comparison signal)
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
