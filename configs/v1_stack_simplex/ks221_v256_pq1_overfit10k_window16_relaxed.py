"""v1_stack_simplex/ks221_v256_pq1_overfit10k_window16_relaxed: much more generous window
relaxation than ks221_v256_pq1_overfit10k_window4_relaxed.py (2x own-K) -- both non-top levels'
own self-attention windows set to 16x their own K ("16 codes worth of context back"): level0's
own-byte window 2->32, level1's own-code window 2->32. Still bounded (well short of full/
unbounded -- seq_lens are 256 bytes / 128 codes), isolating whether a much larger but still
partial context window is what's needed to fix ks221's real-generation collapse, seen at every
narrower window and every quant-type/cond_depth/scheduled-sampling variant tried so far (see
docs/status.md's 2026-08-20/21 hard-convergence-queue entries). Pervasive cond_depth=-1, no
scheduled sampling, original vocab=256 pq_chunks=1 quant -- isolates the window variable alone.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v256_pq1_overfit10k_window16_relaxed.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v256_pq1_overfit10k_window16_relaxed
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v256_pq1_overfit10k_window16_relaxed"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, -1, -1]),  # level0: own-block self-attn (track0) relaxed to 32 (16 blocks worth of
    # bytes, was 2), cross-attn to level1/level2 full
    (-1, [32, -1]),       # level1: own-block self-attn (track0) relaxed to 32 (16 blocks worth of
    # codes, was 2), cross-attn to level2 full
    -1,                  # level2 (top): unchanged full
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

steps = 3000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
