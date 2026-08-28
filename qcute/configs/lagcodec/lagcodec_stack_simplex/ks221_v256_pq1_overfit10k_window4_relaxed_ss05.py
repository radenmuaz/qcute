"""v1_stack_simplex/ks221_v256_pq1_overfit10k_window4_relaxed_ss05: last fallback in the
ks221 hard-convergence investigation chain. Every prior lever -- longer steps, cond_depth=1,
PQ vocab/chunk sweeps (v16pq4, v4pq4, v8pq3, v16pq2), FSQ dq/levels sweeps (8x4, 4x4, 2x16), and
window relaxation to 2x own-K -- converged train byte_acc to 98-99.7% but never broke the
level0_mode1 real-generation collapse into repetitive token loops (see docs/status.md's
2026-08-20/21 hard-convergence-queue entries). This config takes the best-performing setup so far
(window4_relaxed, pervasive cond_depth=-1, original vocab=256 pq_chunks=1 quant) and adds
scheduled_sampling_p=0.5: with probability 0.5, non-top-level decode during training is fed the
level-above's own sampled prediction instead of the real ground-truth code, directly closing the
train/inference exposure-bias gap that's the leading suspect for why train accuracy doesn't
transfer to real autoregressive generation.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v256_pq1_overfit10k_window4_relaxed_ss05.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v256_pq1_overfit10k_window4_relaxed_ss05
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v256_pq1_overfit10k_window4_relaxed_ss05"
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
vocab = 256
pq_chunks = 1
scheduled_sampling_p = 0.5
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
