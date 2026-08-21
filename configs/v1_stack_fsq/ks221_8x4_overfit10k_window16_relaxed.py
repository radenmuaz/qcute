"""v1_stack_fsq/ks221_8x4_overfit10k_window16_relaxed: FSQ counterpart to
v1_stack_simplex/ks221_v256_pq1_overfit10k_window16_relaxed.py -- same much-more-generous window
relaxation (both non-top levels' own self-attention windows at 16x their own K, "16 codes worth of
context back": level0's own-byte window 2->32, level1's own-code window 2->32) but quant_type=grid
(FSQ) with grid_dq=8, grid_levels=4 (matching the 8x4 quant setup used in the earlier
window4_relaxed sweep). Pervasive cond_depth=-1, no scheduled sampling -- isolates the window
variable alone. See docs/status.md's 2026-08-20/21 hard-convergence-queue entries.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_fsq/ks221_8x4_overfit10k_window16_relaxed.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_fsq_ks221_8x4_overfit10k_window16_relaxed
"""
from pathlib import Path

run_name = "v1_stack_fsq_ks221_8x4_overfit10k_window16_relaxed"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, -1, -1]),
    (-1, [32, -1]),
    -1,
)
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
n_bytes = 10000
val_frac = 0.1

steps = 3000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
