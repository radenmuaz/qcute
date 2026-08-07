"""qcute.qcutelm_vlt6 config: tokenizer (encoder/decoder) kept cheap
(<1M params) since, unlike bytelm (no tokenizer) or bpelm (non-parametric
BPE tokenizer), qcutelm_vlt6's encoder/decoder is a REAL parameter cost
competing with CodeLM's budget — narrow+shallow (d_model=64, n_layers=2)
keeps that overhead small (0.133M) so CodeLM (d_model=256, n_layers=4,
matching bytelm_xs's own dims almost exactly) carries the actual
"language model" comparison weight. context_len=256 (matching baseline).
Total params 3.318M, close to bytelm_xs's ~3.7M, with the split now
reflecting where the real capacity should go.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_cheap_tokenizer.py
"""
from pathlib import Path

K = 4
context_len = 256
attn_window = 16
dq = 6
quant_type = "ifsq"
fsq_levels = 8

d_model = 64
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 4
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
