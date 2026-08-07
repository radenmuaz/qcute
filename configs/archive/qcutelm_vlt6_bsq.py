"""qcute.qcutelm_vlt6 first experiment: tokenizer-as-AR-LM, single byte-NTP
loss, BSQ. Same architecture scale as qcutelm_vlt5_codelm.py (encoder +
CodeLM widths) for a direct comparison, but the training objective is
completely different — see qcutelm_vlt6.py's module docstring: no
reconstruction loss, no auxiliary encoder-side NTP loss, only genuine
held-out next-block prediction (codepred predicts the next code from
purely causal past codes, decoder reconstructs a block it was never shown).

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_bsq.py
"""
from pathlib import Path

K = 4
context_len = 512
attn_window = 16
dq = 18
quant_type = "bsq"
d_model = 128
n_heads = 4
n_layers = 4
mlp_mult = 4
code_net_layers = 0

lm_d_model = 128
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
