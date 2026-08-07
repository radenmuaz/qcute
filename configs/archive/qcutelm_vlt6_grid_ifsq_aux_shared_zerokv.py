"""qcute.qcutelm_vlt6 grid cell: ifsq / aux_recon loss / shared enc-dec.
ZERO-KV ABLATION (use_zero_kv=True) — see other docstring text below for the rest of this cell's
setup (quant/loss/encdec); this variant additionally re-enables the zero-KV sink (paired with RoPE,
a combination not used in the original design) to isolate use_zero_kv as its own grid dimension.

Part of the 2x2x2 (quant_type x loss_type x shared_encoder_decoder) grid —
see docs/status.md / session notes for the full layout. Same RoPE+no-zero-
KV+windowed(64)+context_len=1024 scheme as qcutelm_vlt6_rope_bpelm_parity.py
(the ifsq/ntp/shared cell), only main_ntp_weight/aux_recon_weight swapped:
decoder trained via aux_recon (decode(code[i]) vs block[i], zero-shift,
bypassing codelm) instead of main_ntp (decode(codepred(codelm(...))) vs
block[i+1]) — no post-codelm decode pass at all, matching "no ntp target
decoder after codelm outs" from earlier in the session.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_grid_ifsq_aux_shared_zerokv.py
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
use_zero_kv = True
shared_encoder_decoder = True

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
