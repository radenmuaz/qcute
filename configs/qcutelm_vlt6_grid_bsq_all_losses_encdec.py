"""qcute.qcutelm_vlt6 config: loss-ablation experiment — BSQ quantizer,
all three losses active and equally weighted (main_ntp_weight=
aux_recon_weight=code_match_weight=1.0), separate (not shared)
encoder/decoder weights. Direct BSQ counterpart to qcutelm_vlt6_all_losses.py
(which is ifsq, dq=6, pre-grid architecture, shared encdec) — this one
matches the grid's own BSQ scale (dq=13, exact bpelm-vocab match) and the
grid's no-zerokv base, so it's directly comparable to the grid's
bsq_ntp_encdec cell (val_bpb 2.5872, the best qcute result so far) with
only the loss composition changed.

Same architecture as the grid's bsq cells otherwise: d_model=96/n_layers=2
(tokenizer, separate encoder+decoder), lm_d_model=256/n_layers=3 (codelm),
context_len=1024, attn_window=64, RoPE, use_zero_kv=False.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_grid_bsq_all_losses_encdec.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 13
quant_type = "bsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

use_rope = True
use_zero_kv = False

shared_encoder_decoder = False

main_ntp_weight = 1.0
aux_recon_weight = 1.0
code_match_weight = 1.0

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

gen_every = 1000
gen_prompt_len = 64
gen_new_bytes = 64
