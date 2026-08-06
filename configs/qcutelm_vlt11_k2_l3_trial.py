"""qcute.qcutelm_vlt11 config: session-specified FIRST TRIAL — K=2,
n_levels=3 (matching the worked example: 3 encode levels + 3 mirrored
decode levels), self_consistency_weight=0.0, byte_ntp_weight=0.0 (not yet
implemented), only final_ntp_weight=1.0 active. Small scale, short run —
this is a CPU sanity/quick-test config, not meant to be bpb-competitive.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_trial.py --device cpu
"""
from pathlib import Path

K = 2
n_levels = 3
context_len = 64          # K**n_levels=8, so 8 top-level blocks per example
dq = 8
quant_type = "ifsq"
fsq_levels = 8

d_model = 64
n_heads = 4
n_layers = 2
mlp_mult = 4
pool_n_heads = 4

self_consistency_weight = 0.0
byte_ntp_weight = 0.0
final_ntp_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 500
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 50
cosine_decay = False
constant_steps = 50
eval_every = 50
eval_batches = 10
