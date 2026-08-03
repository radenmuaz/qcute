"""qcute.qcutelm_vlt6 config: all three losses active, equally weighted
(main_ntp_weight=aux_recon_weight=code_match_weight=1.0). Direct
comparison point against qcutelm_vlt6_codematch_only.py, which found
code_match_loss plateauing (~1.22-1.26, stuck) while aux_recon_acc kept
climbing when trained WITHOUT main_ntp — this run tests whether adding
main_ntp back in (its decode pass gives codelm indirect gradient too,
via pred_soft feeding decode_block) helps code_match/codelm actually
learn to predict useful next-codes, at the cost of the extra main_ntp
compute qcutelm_vlt6_codematch_only.py specifically avoided.

Same architecture as qcutelm_vlt6_ifsq_2xflops_leanparams.py /
qcutelm_vlt6_codematch_only.py otherwise — d_model=96/n_layers=2
(tokenizer), lm_d_model=256/n_layers=3 (codelm), attn_window=64,
context_len=1024 (256 codes, matching bpelm's context).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_all_losses.py
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

gen_every = 200
gen_prompt_len = 64
gen_new_bytes = 64
