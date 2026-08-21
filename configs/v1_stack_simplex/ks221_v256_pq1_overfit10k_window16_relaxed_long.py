"""v1_stack_simplex/ks221_v256_pq1_overfit10k_window16_relaxed_long: same as
ks221_v256_pq1_overfit10k_window16_relaxed.py (Ks=(2,2,1), both non-top levels' own
self-attention windows at 16x their own K, pervasive cond_depth=-1, original vocab=256
pq_chunks=1 quant, no scheduled sampling) but steps doubled to 6000 -- that run reached
byte_acc=99.24% at 3000 steps with real generation showing non-coherent but diverse word-salad
(no repetitive token-loop collapse, unlike every narrower-window variant, but still no
grammatical sentences); testing whether more steps alone lets that diversity resolve into
coherence. See docs/status.md's 2026-08-20/21 hard-convergence-queue entries for the full chain
this continues.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v256_pq1_overfit10k_window16_relaxed_long.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v256_pq1_overfit10k_window16_relaxed_long
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v256_pq1_overfit10k_window16_relaxed_long"
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
vocab = 256
pq_chunks = 1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 6000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
