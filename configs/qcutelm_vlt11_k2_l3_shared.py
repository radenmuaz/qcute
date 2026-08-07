"""qcute.qcutelm_vlt11 config: the "everything combined, run to completion"
attempt (session: fell back to v11 after qcutelm_pyramid showed worse bpb
AND more FLOPs than bytelm on two hyperparameter variants — "should you
fallback to v11 or not or stick with fifo, decide" -> v11, since it has a
PROVEN FLOP reduction pyramid/fifo never delivered).

Forked from configs/qcutelm_vlt11_k2_l3_full.py (Ks=(4,4,4), dqs=(8,8,8),
tier_d_models=(96,96,96), attn_window=64, lm_d_model=128/lm_n_heads=4/
lm_n_layers=3/lm_attn_window=16, e_ntp_weight=1.0/e_ntp_every=4/
e_ntp_bit_head_mode="independent" — the collapse-fix + both amortization
strategies validated earlier this session) — this config has never
actually been run to completion; every prior attempt was stopped early
for a different experiment (identity-quant ablation, byte_only ablation,
v12/pyramid detour). This is the first clean, full attempt.

One change vs. the earlier "full" config: share_across_levels=True (new)
— tier_d_models=(96,96,96) and dqs=(8,8,8) are already uniform, so this
is a free, compatible flag flip: ONE shared E/D/codelm/code_pre/z_proj/
head_code instead of 3 independent copies per role. Verified this
session (gradient-isolation check + param-count check) to work correctly
and preserve the detach-teacher-force fix's guarantees.

An earlier version of this file also added cosine_decay=True and 10x
weight_decay (1e-4), copying a regularization lesson from qcutelm_
pyramid's v1/v2 comparison — reverted after ~1100 steps (session: "need
to be fair") since bytelm/bpelm/v11_full all use cosine_decay=False and
weight_decay=1e-5, and changing the schedule AND the architecture in the
same run means a win or loss can't be cleanly attributed to either. This
config now matches every baseline's training recipe exactly, so
share_across_levels is the ONLY variable under test here. (The aborted
cosine_decay/weight_decay run's own partial trajectory — val_bpb ~4.3-4.4
in the step 500-1100 range, notably behind bytelm's ~2.4-2.5 at the same
steps — is recorded in docs/status.md for the record, not repeated here.)

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_shared.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
context_len = 1024
quant_type = "bsq"
# fsq_levels = 8

n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 16
share_across_levels = True   # new this run — see docstring

lm_d_model = 128
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
weight_decay = 1e-5        # matches every baseline (bytelm/bpelm/v11_full) exactly — session: "need
                            # to be fair" — the earlier 1e-4/cosine_decay=True variant made
                            # share_across_levels not the only thing being tested vs baselines; reverted
                            # so this run isolates JUST the sharing flag, nothing else
warmup_steps = 500
cosine_decay = False        # matches every baseline exactly — see note above
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
