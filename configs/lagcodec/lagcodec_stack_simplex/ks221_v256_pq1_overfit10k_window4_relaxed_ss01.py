"""v1_stack_simplex/ks221_v256_pq1_overfit10k_window4_relaxed_ss01: same real test as
ks221_v256_pq1_overfit10k_window4_relaxed_ss05.py but scheduled_sampling_p=0.1 (was 0.5) --
lower substitution rate, testing whether it keeps the STE-connected scheduled sampling's real-
generation diversity benefit (seen at ks21 scale, see ks21_v256_pq1_overfit10k_tinywindow_ss01.py)
without ss05's training instability. See docs/status.md's 2026-08-20/21 hard-convergence-queue
entries for the full chain this continues.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v256_pq1_overfit10k_window4_relaxed_ss01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v256_pq1_overfit10k_window4_relaxed_ss01
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v256_pq1_overfit10k_window4_relaxed_ss01"
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
scheduled_sampling_p = 0.1
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
