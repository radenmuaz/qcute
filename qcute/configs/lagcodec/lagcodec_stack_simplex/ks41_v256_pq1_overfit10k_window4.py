"""v1_stack_simplex/ks41_v256_pq1_overfit10k_window4: tiny-window stress test (see
ks21_v256_pq1_overfit10k_tinywindow.py, docs/status.md's 2026-08-20 entries) generalized to a
harder Ks=(4,1) -- level0's own byte-level self-attention window forced to exactly K0=4 (current
block only, zero raw-byte lookback into any previous block), level0's cross-attention window onto
level1's code stays full/unbounded. Forces level0 to reconstruct/predict purely from its own code
plus level1 conditioning, never from raw-byte history -- see if a coarser 4-byte block still
overfits under the same handicap that worked for Ks=(2,1).

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks41_v256_pq1_overfit10k_window4.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks41_v256_pq1_overfit10k_window4
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks41_v256_pq1_overfit10k_window4"
decoder_type = "stack"
Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, [4, -1]), -1)  # level0: encode full, decode track0 (own-byte self-attn)=K0=4
# (current block only), decode track1 (level1 code cross-attn) full; level1 (top): unchanged full
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

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
