"""qcute.qcutelm_vlt5 with CodeLM plugged in (use_code_lm=True, the
default): a causal transformer sitting between code production and
decode, operating on the continuous code embedding sequence. No
code-level classification loss, nothing detached — recon_loss's gradient
flows straight through CodeLM into code_net/the tokenizer's own blocks.
Plain pretraining (qcutelm_vlt5_bsq.py) is the literal use_code_lm=False /
identity-at-init special case of this same architecture — see
qcutelm_vlt5.py's CodeLM docstring.

Same tokenizer scale as qcutelm_vlt5_bsq.py; CodeLM sized comparably.

    uv run python -m qcute.qcutelm_vlt5 --config configs/qcutelm_vlt5_codelm.py
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
ntp_loss_weight = 1.0
recon_loss_weight = 1.0

use_code_lm = True
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

recon_target_acc = 0.95
