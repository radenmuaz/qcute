"""v1_stack_simplex/ks21_v256_pq1_overfit10k_tinywindow: same as ks21_v256_pq1_overfit10k.py but
level0 decode's own byte-level self-attention window (track0) is forced down to 2 (K itself --
current block only, zero visibility into any previous block's raw bytes), while level0's
cross-attention window onto level1's code (track1) stays -1 (full/unbounded). This starves the
byte-level self-attention path of any long-range raw-byte context it could otherwise use to
memorize the 10k-byte corpus directly, forcing level0 to actually depend on level1's code to
reconstruct anything beyond the current block -- a stress test of whether the "gt" path (real
code + real context, see docs/status.md's 2026-08-20 later-session entry) can still overfit when
level1 conditioning is load-bearing rather than optional. attn_window's per-level tuple form:
`((encode_window, [track0_window, track1_window]), level1_window)`.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v256_pq1_overfit10k_tinywindow.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1_overfit10k_tinywindow
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1_overfit10k_tinywindow"
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
