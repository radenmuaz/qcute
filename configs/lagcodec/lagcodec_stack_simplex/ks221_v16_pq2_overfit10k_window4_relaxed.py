"""v1_stack_simplex/ks221_v16_pq2_overfit10k_window4_relaxed: attempt #5 of the 5-config random
sweep over quant structure (see ks221_v4_pq4_overfit10k_window4_relaxed.py for the full setup and
rationale). quant_type=simplex, vocab=16, pq_chunks=2 (2 independent 16-way softmaxes, 8-bit
combinatorial code, matching the original width with only 2 chunks instead of 1 or 4). Same
window4_relaxed handicap, pervasive cond_depth=-1, 3000 steps.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq2_overfit10k_window4_relaxed.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq2_overfit10k_window4_relaxed
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq2_overfit10k_window4_relaxed"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [4, -1, -1]),
    (-1, [4, -1]),
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 2
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
