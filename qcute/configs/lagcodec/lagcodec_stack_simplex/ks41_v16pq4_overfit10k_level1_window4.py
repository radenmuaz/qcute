"""v1_stack_simplex/ks41_v16pq4_overfit10k_level1_window4: Ks=(4,1), v16pq4 (current standard
quant recipe, not the abandoned v256pq1 -- see ks41_v256_pq1_overfit10k_window4.py, an older
config using the coupled self/cross window). level0's ONLY cross-attention track is level 1's code
(track0 -- the topmost-plus-one code is already hard-excluded structurally, 2026-08-23). Its window
is set via the new (self_window, cross_window) 2-tuple (2026-08-23, qcute_lagcodec.py's _norm_track0):
self_window=-1 (level0's own byte self-attention stays unbounded), cross_window=4 -- a genuine FIFO
sliding window over level 1's code, RAW units = code count directly for track0 (NOT window=4*cum_K
like cross_attn_stage's upper-track convention -- see encode_like_self_attn_decode's block_lag
comparison). Getting a 5th code pushes the 1st out of visibility.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks41_v16pq4_overfit10k_level1_window4.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks41_v16pq4_overfit10k_level1_window4
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks41_v16pq4_overfit10k_level1_window4"
decoder_type = "stack"
Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [(-1, 4), -1]),  # level0: encode full; decode track0 = (self_window=-1, cross_window=4 codes)
    -1,                    # level1 (top): unchanged, full
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "copy"
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
