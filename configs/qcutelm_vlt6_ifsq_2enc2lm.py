"""qcute.qcutelm_vlt6 config: FLOP-matched split — equal width
(d_model=lm_d_model=128, matching bytelm_xs's width) so the layer-count
ratio directly reflects a FLOP ratio, unlike qcutelm_vlt6_ifsq_bigcodelm.py
(64-wide/2-layer encoder-decoder vs 256-wide/6-layer CodeLM, confounding
width AND depth asymmetry). 2 encoder/decoder layers + 2 CodeLM layers =
4 total, matching bytelm_xs's/bpelm_8192's 4-layer depth for a cleaner
comparison. See also qcutelm_vlt6_ifsq_3enc1lm.py (3+1 split, same total).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_2enc2lm.py
"""
from pathlib import Path

K = 4
context_len = 512
attn_window = 16
dq = 6
quant_type = "ifsq"
fsq_levels = 8

d_model = 128
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 128
lm_n_heads = 4
lm_n_layers = 2
lm_mlp_mult = 4

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

gen_every = 1000
gen_prompt_len = 64
gen_new_bytes = 64
