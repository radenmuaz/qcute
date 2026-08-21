"""v1_stack_simplex/ks221_v64_pq4_overfit10k_window16_relaxed_ss1: same window16_relaxed +
scheduled_sampling_p=1.0 (detach_ss_sample=False, uncertainty_weighting=False) setup as
ks221_v16_pq4_overfit10k_window16_relaxed_ss1.py, but vocab=64 (pq_chunks=4 unchanged -- 24-bit
combinatorial code, vs the v16pq4 sibling's 16-bit) -- both v16pq4 and v16pq8 (wider, 32-bit)
already failed with identical repetitive single-token collapse regardless of uncertainty
weighting or ss=1.0 (see docs/status.md's hard-convergence-queue entries); the curriculum attempt
(max_srcs=2 first-half-of-training ablation) also collapsed by step ~1200/3000, and turned out not
to be a clean ks21-equivalent anyway (level1's own decode still conditions on level2's code
regardless of max_srcs, since max_srcs is a single global cap and level1 has only one upper track
-- see chat 2026-08-21). This tests a bigger per-chunk codebook (64-way vs 16-way softmax per PQ
chunk) at the SAME pq_chunks=4 structure, in case 16-bit total code width itself (not vocab/chunk
split) was the bottleneck.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v64_pq4_overfit10k_window16_relaxed_ss1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v64_pq4_overfit10k_window16_relaxed_ss1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v64_pq4_overfit10k_window16_relaxed_ss1"
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
vocab = 64
pq_chunks = 4
scheduled_sampling_p = 1.0
detach_ss_sample = False
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
