"""v5_stack_fsq/ks1_16x8: same as ks1_16x4.py but grid_levels=8 (8**16 combinatorial codes)
-- between ks1_16x4.py (4**16) and ks1_16x16.py (16**16). Per-token head cost scales with
dq*L=128, cheap regardless of the huge combinatorial total.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_fsq/ks1_16x8.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_fsq_ks1_16x8
"""
from pathlib import Path

run_name = "v5_stack_fsq_ks1_16x8"
decoder_type = "stack"
Ks = (1,)
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
