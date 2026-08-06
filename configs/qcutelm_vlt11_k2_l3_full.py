"""qcute.qcutelm_vlt11 config: full-scale run of the session-specified
FIRST TRIAL — K=2, n_levels=3, self_consistency_weight=0.0,
byte_ntp_weight=0.0 (not implemented), only final_ntp_weight=1.0 active.
context_len=1024 (must be a multiple of K**n_levels=8) for rough
comparability with the other bsq/baseline runs.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_full.py
"""
from pathlib import Path

K = 2
n_levels = 3
context_len = 1024
dq = 8
quant_type = "ifsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
pool_n_heads = 4

self_consistency_weight = 0.0
byte_ntp_weight = 0.0
final_ntp_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20
