"""qcute_zero/ks21_overfit10k_paralleldecode: same as ks21_overfit10k.py but with
cfg.parallel_decode=True -- trains the shared query_vec to predict a whole Ks[0]=2-byte block at
once from strictly-prior codes only (the "free tier" of the parallel-block-decode brainstorm, see
docs/status.md's "parallel block decode brainstorm" section, 2026-08-22). First real validation:
does parallel_decode_acc climb toward the ordinary cond0_acc/byte_acc, confirming the query vector
can stand in for a real previous-byte hidden state well enough to predict a whole block blind.

uv run python -m qcute.qcute_zero_parallel.qcute_zero_parallel --config configs/qcute_zero_parallel/ks21_overfit10k_paralleldecode.py
"""
from pathlib import Path

run_name = "qcute_zero_parallel_ks21_overfit10k_paralleldecode"
Ks = (2, 1)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8
parallel_decode = True
parallel_decode_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
log_every = 20
eval_every = 50
eval_batches = 5
