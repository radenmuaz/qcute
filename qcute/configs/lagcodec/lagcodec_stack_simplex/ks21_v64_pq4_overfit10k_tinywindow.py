"""v1_stack_simplex/ks21_v64_pq4_overfit10k_tinywindow: same tiny-window stress test as
ks21_v256_pq1_overfit10k_tinywindow.py (level0 decode's own byte-level self-attention window
forced to 2 -- current block only, zero raw-byte lookback -- while the level1 cross-attention
window stays full, forcing decode to depend on level1's code) but on the PQ variant (vocab=64,
pq_chunks=4, see ks21_v64_pq4_overfit10k.py) instead of the single 256-way softmax, and with
steps doubled (2000 instead of 1000) to see whether the extra budget changes the
gt_byte_acc-overfits/pred_byte_acc-stays-weak pattern found in the v256 tinywindow run (see
docs/status.md's 2026-08-20 tiny-window stress test entry).

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v64_pq4_overfit10k_tinywindow.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v64_pq4_overfit10k_tinywindow
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v64_pq4_overfit10k_tinywindow"
decoder_type = "stack"  # StackDecoder -- see ks21_v256_pq1_overfit10k.py
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((-1, [2, -1]), -1)  # level0: encode window full, decode track0 (own-byte self-attn)
# forced to 2 (current block only), decode track1 (level1 code cross-attn) full; level1: unchanged full
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 64
pq_chunks = 4
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 2000  # doubled vs ks21_v64_pq4_overfit10k.py's 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
