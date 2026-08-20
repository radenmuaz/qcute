"""v5_stack_windowtest/ks21_control: dense-coupling control for the window-ablation hypothesis
test (see CLAUDE.md's "Latent-AR / parallel-block-local-decode investigation" section) -- same
FSQ grid_dq=16/grid_levels=8, code_hard=True/code_sample=False as the winning
v5_stack_fsq_ks21_16x8 config, but on a 10% data subset (n_bytes=100000, matching this folder's
sibling stage configs) with attn_window left at today's default (-1, fully dense/unbounded) so
the stage A/B ablations below have an apples-to-apples same-subset baseline rather than being
compared against the full-data ks21_16x8 result directly.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/v5_stack_windowtest/ks21_control.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_stack_windowtest_ks21_control
"""
from pathlib import Path

run_name = "v5_stack_windowtest_ks21_control"
decoder_type = "stack"
Ks = (2, 1)
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
