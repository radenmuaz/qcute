"""v5_stack_windowtest/ks221_control: dense-coupling control, 3-level sibling of ks21_control.py
-- same FSQ grid_dq=16/grid_levels=8, code_hard=True/code_sample=False as
v5_stack_fsq_ks221_16x8, on the same 10% data subset (n_bytes=100000) as this folder's other
configs, attn_window left at today's default (-1, fully dense/unbounded).

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_windowtest/ks221_control.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_windowtest_ks221_control
"""
from pathlib import Path

run_name = "v5_stack_windowtest_ks221_control"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "grid"
grid_dq = 16
grid_levels = 8
vocab = 256
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 100000
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
