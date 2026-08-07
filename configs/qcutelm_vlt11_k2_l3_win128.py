"""qcute.qcutelm_vlt11 config: attn_window widened 64->128 (session: "make
mod to encoder and decoder both use sliding window attention, make new
config based on _full, test how much iter sec saved, set each level 0 1 2
window 128, do not think about effective context len now").

Forked from configs/qcutelm_vlt11_k2_l3_full.py — every field identical
except attn_window. This is purely an iter/sec probe: does widening the
window from 64->128 change wall-clock throughput at all, given the
dominant cost is the O(tokens*d^2) MLP term rather than the O(tokens*
window*d) attention term (the same FLOPs argument that motivated
qcutelm_mergetoken_v1's level-0 restructure)? Expectation is little to no
change in it/s, since attention was never the bottleneck at this scale.

seq_lens for Ks=(4,4,4)/context_len=1024 are [1024, 256, 64]. window=128
divides level 0 (1024) and level 1 (256) evenly, but level 2's L=64 <
window=128 — CausalSelfAttention.forward already handles this safely by
falling back to dense attention at that level (T <= window skips the
chunked path). The __init__-time assertion was widened this session to
accept L <= window as well as L % window == 0 (previously would have
hard-failed here) — see qcutelm_vlt11.py's own comment at that assert.

Not attempting to extend effective receptive field past 1024 here per
explicit instruction ("do not think about effective context len now") —
this run is scoped to the it/s question only.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_win128.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 128   # was 64 — see docstring

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
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
