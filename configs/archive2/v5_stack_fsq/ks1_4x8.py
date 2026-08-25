"""v5_stack_fsq/ks1_4x8: same as ks1.py but grid_levels=4/grid_dq=8 (4**8=65536 combinatorial
codes, vs ks1.py's grid_levels=8/grid_dq=4 = 8**4=4096) -- bigger grid codebook, sibling A/B.
Per-token cost only scales with dq*L (32 here vs ks1.py's 32 too, same head size) regardless of
the combinatorial total, so this is not meaningfully more expensive per step, just a bigger code
space.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_fsq/ks1_4x8.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_fsq_ks1_4x8
"""
from pathlib import Path

run_name = "v5_stack_fsq_ks1_4x8"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "grid"
grid_dq = 8
grid_levels = 4
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
