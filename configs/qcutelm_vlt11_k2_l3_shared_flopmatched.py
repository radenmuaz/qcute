"""qcute.qcutelm_vlt11 config: FLOP-matched to bytelm_xs_mtp4_ctx1024
(session: "make another config v11 shared that is flop matched closer
with bytelm baseline" — directly tests whether v11's weak bpb vs bytelm
at matching steps is a compute deficit or an architectural one, following
the session discussion "is it reason why val bpb so weak, too few flop").

Forked from configs/qcutelm_vlt11_k2_l3_shared.py (share_across_levels=
True, e_ntp_weight=1.0/e_ntp_every=4/e_ntp_bit_head_mode="independent",
cosine_decay=False/weight_decay=1e-5 — matching baselines exactly per the
earlier "need to be fair" correction). ONE change: tier_d_models 96->192,
lm_d_model 128->256 (width only, n_layers/lm_n_layers unchanged) — using
the same 6*N*tokens FLOPs estimate this session has used throughout:

    orig (tier_d=96, lm_d=128):  ~76 GFLOP/step
    this config (tier_d=192, lm_d=256): ~304 GFLOP/step
    bytelm_xs_mtp4_ctx1024 reference: ~334 GFLOP/step

~91% of bytelm's FLOPs/step — close enough for this comparison's purpose
without being falsely precise (the 6ND estimate is itself approximate).
share_across_levels=True keeps this from being an unreasonably huge
model despite the width increase (ONE shared E/D/codelm/etc. instead of
3 independent copies per role) — params should land well under bytelm's
3.4M despite matching FLOPs/step, since sharing amortizes the width
increase's parameter cost across all 3 levels while bytelm pays its
full param count for a single (wider, deeper) stack. Not yet verified
empirically at construction time — check the run's own logged params
count when launched.

If this STILL plateaus well below bytelm's bpb despite comparable
FLOPs/step, that would point away from "insufficient compute" and toward
either (a) the architecture's specific inductive bias (narrow-then-wide
hierarchy) being a worse fit for this task than dense attention, or (b)
optimization/training-dynamics issues (harder loss landscape from the
multiple simultaneous objectives) rather than raw capacity.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_shared_flopmatched.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (192, 192, 192)   # was (96,96,96)
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 64
share_across_levels = True

lm_d_model = 256          # was 128 — matches bytelm's own d_model exactly
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4
lm_attn_window = 16

code_match_weight = 1.0
e_ntp_weight = 1.0
e_ntp_every = 4
e_ntp_bit_head_mode = "independent"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5        # matches every baseline exactly, same "need to be fair" reasoning
warmup_steps = 500
cosine_decay = False         # matches every baseline exactly
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
