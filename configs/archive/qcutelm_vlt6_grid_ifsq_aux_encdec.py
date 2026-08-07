"""qcute.qcutelm_vlt6 grid cell: ifsq / aux_recon loss / separate enc-dec.
See qcutelm_vlt6_grid_ifsq_aux_shared.py — identical except
shared_encoder_decoder=False (symmetric separate weights, dec_* fields
unset so the decoder mirrors the encoder's own dims).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_grid_ifsq_aux_encdec.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 5
quant_type = "ifsq"
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

main_ntp_weight = 0.0
aux_recon_weight = 1.0
code_match_weight = 0.0

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
