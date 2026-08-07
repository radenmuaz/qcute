"""qcute.qcutelm_mergetoken_v1 config: ported from configs/qcutelm_vlt11_
k2_l3_shared.py's settings (session: "new config, from shared") — same
Ks=(4,4,4)/tier_d_models=(96,96,96)/dqs=(8,8,8)/attn_window=64/lm_*/
share_across_levels=True/e_ntp_weight=1.0 etc., and the SAME baseline-
matched training recipe (cosine_decay=False, weight_decay=1e-5 — see
that config's own "need to be fair" note) — the only architectural
difference from qcutelm_vlt11_k2_l3_shared is level 0 itself, which this
file's module docstring covers in full (block-merge + byte-chain MTP
instead of per-byte prediction, cutting level 0's dominant FLOP cost
~4x while still covering the full 1024-byte context).

seq_lens for this config: [256, 256, 64] (level 0's own block count,
level 1's input = c_0 at the same length since level 0's readout doesn't
pool further, level 2's input = c_1 pooled by Ks[1]=4). attn_window=64
divides all three evenly (256/64=4, 256/64=4, 64/64=1).

share_across_levels=True here applies to levels 1/2 only (E_1/D_1/
codelm_1 vs E_2/D_2/codelm_2) — level 0 is architecturally unique in
this file (byte-chain MTP head, block merge), never a sharing candidate.

    uv run python -m qcute.qcutelm_mergetoken_v1 --config configs/qcutelm_mergetoken_v1_shared.py
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
attn_window = 64
share_across_levels = True

fetch_n_heads = 4
fetch_gamma = 1.0
tie_head0 = True

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
weight_decay = 1e-5        # matches every baseline exactly, same "need to be fair" reasoning
warmup_steps = 500
cosine_decay = False         # matches every baseline exactly
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
