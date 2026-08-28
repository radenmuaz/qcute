"""v1_stack_simplex/ks21_v16_pq4_overfit10k_tinywindow_conddepth1: same tiny-window stress test as
ks21_v256_pq1_overfit10k_tinywindow.py (Ks=(2,1), level0 decode's own byte-level self-attention
window forced to K0=2, level1 cross-attn window full) but with vocab=16, pq_chunks=4 (4
independent 16-way softmaxes, 16-bit combinatorial code -- close in total code width to the
original vocab=256, pq_chunks=1 setup's 8 bits, unlike the older v64_pq4 configs' 24-bit code) and
cond_depth=1 (explicit, though a no-op at n_levels=2 -- included for consistency with the paired
ks221 config, see docs/status.md's 2026-08-20 hard-convergence-queue entries). Baseline sanity
check before trying this quant setup on the harder ks221 config.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v16_pq4_overfit10k_tinywindow_conddepth1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_tinywindow_conddepth1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_tinywindow_conddepth1"
decoder_type = "stack"
cond_depth = 1
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, [2, -1]), -1)  # level0: encode full, decode track0 (own-byte self-attn)=K0=2
# (current block only), decode track1 (level1 code cross-attn) full; level1 (top): unchanged full
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
