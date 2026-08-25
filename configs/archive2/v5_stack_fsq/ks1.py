"""v5_stack_fsq/ks1: Ks=(1,), stack decoder, quant_type="grid" (finite scalar quantization, was
"fsq"), grid_dq=4/grid_levels=8 -- axis-aligned integer-grid code instead of the noreg grid's
categorical simplex, same architecture/schedule as configs/qcute_v5_stack_noreg/ks1.py otherwise
(code_hard=True/code_sample=False i.e. ste-equivalent, entropy_reg_weight=0.0, full val eval).

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_fsq/ks1.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_fsq/ks1
"""
from pathlib import Path

run_name = "v5_stack_fsq_ks1"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "grid"
grid_dq = 4
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
