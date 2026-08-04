"""qcute.qcutelm_vlt6 config: 17th grid cell, added after the original 16-cell
sweep was already queued/running — main_ntp_weight=1.0 AND code_match_weight=1.0
jointly active (aux_recon stays 0; main_ntp alone already gives the decoder
gradient, so no starvation risk like the earlier code_match-only design).

Motivation: under main_ntp-only training, codelm's pred_soft is only ever
supervised through the DECODER's mapping to bytes — nothing forces pred_soft
to numerically resemble a real (encoder-derived, quantized) code. That makes
"free" code-only generation (rolling codelm forward in latent space, feeding
pred_soft back as the next step's input via z_proj, decoding only at the end)
unsound: z_proj/codelm were only ever trained to consume true codes as input,
and pred_soft as an output is a different, decoder-mediated object. Adding
code_match_weight directly penalizes pred_soft for deviating from the true
next code in code space, anchoring it to the same manifold z_proj expects —
the property free code-only generation actually needs. None of the original
16 cells (loss axis: ntp vs aux only) cover this.

Otherwise identical to qcutelm_vlt6_rope_bpelm_parity.py / the grid's
ifsq/shared/no-zerokv corner: d_model=96/n_layers=2 (tokenizer, shared
encoder+decoder), lm_d_model=256/n_layers=3 (codelm), context_len=1024,
attn_window=64 (chunked no-sink), RoPE, use_zero_kv=False.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_grid_ifsq_ntp_codematch_shared.py
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

shared_encoder_decoder = True

main_ntp_weight = 1.0
aux_recon_weight = 0.0
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
