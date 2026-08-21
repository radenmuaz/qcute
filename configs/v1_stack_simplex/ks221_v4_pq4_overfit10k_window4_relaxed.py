"""v1_stack_simplex/ks221_v4_pq4_overfit10k_window4_relaxed: first attempt #1 of a 5-config random
sweep over quant structure, all at the same window4_relaxed handicap (Ks=(2,2,1), pervasive
cond_depth=-1, both non-top levels' own self-attention windows relaxed to 2x their own K, 3000
steps) as ks221_v256_pq1_overfit10k_window4_relaxed.py -- this variant swaps quant_type=simplex,
vocab=4, pq_chunks=4 (4 independent 4-way softmaxes, 8-bit combinatorial code, same total width as
the original vocab=256 pq_chunks=1 baseline but split into 4 small chunks instead of 1 big one).
Tests whether chunk count/structure at fixed code width changes convergence. See
docs/status.md's 2026-08-20 hard-convergence-queue entries. If this fails, a second attempt will
adjust the window; if that fails, a third attempt follows.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v4_pq4_overfit10k_window4_relaxed.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v4_pq4_overfit10k_window4_relaxed
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v4_pq4_overfit10k_window4_relaxed"
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
vocab = 4
pq_chunks = 4
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
