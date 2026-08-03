"""qcute.qcutelm_vlt6 config: iFSQ quantizer (sigmoid bound — see
qcutelm_vlt5/status.md's saturation analysis: retains more gradient than
tanh-bound FSQ/BSQ as activations grow, should ease optimization) +
asymmetric capacity — encoder/decoder lighter (the mechanical byte<->code
job), CodeLM bigger (the hard "predict the future code" job this fork's
entire objective rests on).

dq=6, fsq_levels=8 -> 8^6=262144, same codespace size as BSQ's dq=18, for
parity with earlier comparisons.

Now includes periodic qualitative generation (prompt/generated/ground-truth,
fixed seed for comparability across steps) every gen_every steps.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_ifsq_bigcodelm.py
"""
from pathlib import Path

K = 4
context_len = 512
attn_window = 16
dq = 6
quant_type = "ifsq"
fsq_levels = 8

# encoder/decoder: lighter
d_model = 64
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

# codelm: bigger
lm_d_model = 256
lm_n_heads = 8
lm_n_layers = 6
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
