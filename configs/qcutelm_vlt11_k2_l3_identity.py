"""qcute.qcutelm_vlt11 config: ceiling-baseline convergence-speed diagnostic.

Identical to configs/qcutelm_vlt11_k2_l3_full.py except quant_type=
"identity" — disables code discretization entirely (c_i = code_pre's raw
continuous output, unbounded, no rounding/STE). Not a real operating mode
(the resulting "codes" aren't bounded/roundable/transmittable the way
bsq/fsq's codes are, and generate_no_cache/generate_kv_cache do NOT
support it) — this run exists purely to answer one question: how much of
qcutelm_vlt11's slow train/val-bpb convergence (vs. bytelm_xs_mtp4_ctx1024/
bpelm_*, see docs/status.md's "did convergence" comparison) is caused by
the hard-quantization bottleneck itself, vs. other factors (narrow
tier_d_models=96 vs. baseline's 256, competing multi-level/multi-loss
optimization, cold-start codelm/code_pre modules)?

If identity quant converges dramatically faster/closer to baseline bpb,
that points at quantization (or the STE gradient / discrete forward value)
as the dominant bottleneck — worth chasing quant-specific fixes (soft-to-
hard annealing, soft/blended code_match_loss targets, etc., all listed
in docs/status.md's "Strategies to speed convergence"). If it's still
comparably slow, the bottleneck is elsewhere (capacity, multi-objective
competition, cold-start substitution corruption) — points toward the
byte-NTP-only warmup / curriculum staging ideas instead. Queued to run
BEFORE the warmup experiment for exactly this reason (session: "do this
first before the warmup idea... quant identity first as ceiling
baseline... highest repr power").

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_identity.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
context_len = 1024
quant_type = "identity"
fsq_levels = 8   # unused when quant_type="identity", kept for config-shape parity with the other run

n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 64

lm_d_model = 128
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4
lm_attn_window = 16

code_match_weight = 1.0
e_ntp_weight = 1.0
e_ntp_every = 4
e_ntp_bit_head_mode = "independent"   # no-op for the code levels under quant_type="identity" (their
                                        # heads are plain Linear regardless); still applies to head_e0
                                        # (byte-level NTP, unaffected by quant_type)

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
