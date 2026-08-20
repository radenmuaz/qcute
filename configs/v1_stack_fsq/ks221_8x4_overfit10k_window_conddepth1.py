"""v1_stack_fsq/ks221_8x4_overfit10k_window_conddepth1: FSQ counterpart to
v1_stack_simplex/ks221_v16_pq4_overfit10k_window_conddepth1.py -- same tiny-window stress test
(Ks=(2,2,1), every non-top level's own self-attention window forced to exactly its own K,
cond_depth=1) but quant_type=grid (FSQ) with grid_dq=8, grid_levels=4 (4**8=65536, 16-bit
combinatorial code, matching the PQ variant's total code width) instead of categorical PQ. Fourth
fallback in the same hard-convergence investigation, after longer-steps, cond_depth=1 (categorical
PQ), and vocab=16/pq_chunks=4 (categorical PQ) all failed to produce coherent real generation --
testing whether continuous scalar quantization (FSQ) rather than categorical PQ changes the
picture, at matched code width. See docs/status.md's 2026-08-20 hard-convergence-queue entries.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_fsq/ks221_8x4_overfit10k_window_conddepth1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_fsq_ks221_8x4_overfit10k_window_conddepth1
"""
from pathlib import Path

run_name = "v1_stack_fsq_ks221_8x4_overfit10k_window_conddepth1"
decoder_type = "stack"
cond_depth = 1
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [2, -1, -1]),  # level0: own-block self-attn (track0)=K0=2, cross-attn to level1/level2 full
    (-1, [2, -1]),       # level1: own-block self-attn (track0)=K1=2, cross-attn to level2 full
    -1,                  # level2 (top): unchanged full
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

qual_gen_bytes = 64  # check_gen_consistency/check_roundtrip_consistency/check_decode_modes all
# skip gracefully at n_levels==3 (StackDecoder's generation-fix work is n_levels==2-only so far,
# chat 2026-08-20) -- check_gen_consistency was missing that guard and crashed until fixed this
# same session; qualitative_generate's uncond/level_gen output still works and is worth seeing.
