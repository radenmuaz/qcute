"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_uw: same window16_relaxed setup
(Ks=(2,2,1), both non-top levels' own self-attention windows at 16x their own K, pervasive
cond_depth=-1, no scheduled sampling) as ks221_v256_pq1_overfit10k_window16_relaxed.py but with
quant_type=simplex, vocab=16, pq_chunks=4 (16-bit combinatorial code, the PQ variant that
converged cleanly at ks21/n_levels=2 scale) AND uncertainty_weighting=True -- learns one
log-variance per NTP task (Kendall/Gal/Cipolla 2018) instead of the fixed
byte_ntp_weight/code_ntp_weight/decode_ntp_weight scalars, self-balancing each level's own
encode/decode loss scale. Testing whether the GT-code-generation probe's finding (decode is
sound; level1/level2's own code forecast is the actual bottleneck, see
scripts/probe_gt_code_generation.py and docs/status.md's 2026-08-21 follow-up diagnostic entry)
improves once those upper-level forecast losses aren't potentially under-weighted relative to
level0's easier-to-optimize decode loss, combined with the PQ quant structure.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_uw.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_uw
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_uw"
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
pq_chunks = 4
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
