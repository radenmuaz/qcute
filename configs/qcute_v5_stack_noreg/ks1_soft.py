"""qcute_v5_stack_noreg/ks1_soft: same as ks1.py (Ks=(1,)) but code_hard=False/code_sample=False --
plain softmax relaxation with NO gumbel noise (the "soft" combo that was impossible under the old
single code_sample_mode enum: "soft" always injected noise, "ste" was always hard). Isolates
whether removing hardening alone (holding noise fixed at off) changes training/generation, as a
sibling A/B to ks1.py.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/qcute_v5_stack_noreg/ks1_soft.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_noreg_ks1_soft
"""
from pathlib import Path

decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = False
code_sample = False
quant_type = "simplex"
vocab = 256
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20
full_val_eval = True

qual_gen_bytes = 128
qual_prompt_bytes = 64
