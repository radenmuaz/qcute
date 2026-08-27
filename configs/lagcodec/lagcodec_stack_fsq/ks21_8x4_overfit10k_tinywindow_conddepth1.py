"""v1_stack_fsq/ks21_8x4_overfit10k_tinywindow_conddepth1: FSQ counterpart to
v1_stack_simplex/ks21_v16_pq4_overfit10k_tinywindow_conddepth1.py -- same tiny-window stress test
(Ks=(2,1), level0 decode's own byte-level self-attention window forced to K0=2, level1 cross-attn
window full, cond_depth=1, a no-op at n_levels=2 but kept for consistency with the paired ks221
config) but quant_type=grid (FSQ) with grid_dq=8, grid_levels=4 (4**8=65536, 16-bit combinatorial
code, matching the PQ variant's total code width) instead of categorical PQ. Naming follows
v5_stack_fsq's own dq-x-levels convention (see configs/v5_stack_fsq/ks1_16x8.py). Baseline sanity
check before trying this quant setup on the harder ks221 config, per docs/status.md's 2026-08-20
hard-convergence-queue entries.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_fsq/ks21_8x4_overfit10k_tinywindow_conddepth1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_fsq_ks21_8x4_overfit10k_tinywindow_conddepth1
"""
from pathlib import Path

run_name = "v1_stack_fsq_ks21_8x4_overfit10k_tinywindow_conddepth1"
decoder_type = "stack"
cond_depth = 1
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, [2, -1]), -1)  # level0: encode full, decode track0 (own-byte self-attn)=K0=2
# (current block only), decode track1 (level1 code cross-attn) full; level1 (top): unchanged full
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

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
