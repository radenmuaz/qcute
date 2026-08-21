"""v1_stack_simplex/ks221_v16_pq8_overfit10k_window16_relaxed_uw: same window16_relaxed +
uncertainty_weighting setup as ks221_v16_pq4_overfit10k_window16_relaxed_uw.py but
vocab=16, pq_chunks=8 (8 independent 16-way softmaxes, 32-bit combinatorial code -- wider than
the v16pq4 sibling's 16 bits, testing whether more codebook capacity per block helps once
uncertainty weighting is already self-balancing the per-level loss scales). See
docs/status.md's 2026-08-20/21 hard-convergence-queue entries and the 2026-08-21 follow-up
diagnostic entry for the full chain this continues.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq8_overfit10k_window16_relaxed_uw.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq8_overfit10k_window16_relaxed_uw
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq8_overfit10k_window16_relaxed_uw"
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
quant_type = "simplex"
vocab = 16
pq_chunks = 8
uncertainty_weighting = True
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
