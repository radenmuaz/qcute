"""qcute.qcutelm_vlt6 config: main_ntp_weight=0, code_match_weight=1,
aux_recon_weight=1 — codelm is trained ONLY by matching the tokenizer's
own detached next code (factorized BCE-per-bit/CE-per-dim, see
qcutelm_vlt6.py's forward() docstring), no decode pass at all after
codelm's output ("no ntp target decoder after codelm outs"). The decoder
is trained separately by aux_recon (decode(code[i]) vs block[i] directly,
zero-shift, bypassing codelm) so it isn't left with zero gradient (see
qcutelm_vlt6.py's main() startup warning for what happens if you disable
both) — qualitative_gen() stays meaningful.

Same architecture as qcutelm_vlt6_ifsq_2xflops_leanparams.py otherwise —
d_model=96/n_layers=2 (tokenizer), lm_d_model=256/n_layers=3 (codelm),
attn_window=64, context_len=1024 (256 codes, matching bpelm's context).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_codematch_only.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 6
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

main_ntp_weight = 0.0
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

gen_every = 200
gen_prompt_len = 64
gen_new_bytes = 64
