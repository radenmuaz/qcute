"""qcute.qcute_bytepool config (variant=v12): full-scale run of the
original K=2 pool -> joint-predict -> BOS-decode pipeline (3 blocks). Same
d_model=256 scale as qcute_bytepool_v13.py for direct comparability;
context_len=1024 (must stay a multiple of 4).

    uv run python -m qcute.qcute_bytepool --config configs/qcute_bytepool_v12.py
"""
from pathlib import Path

variant = "v12"
context_len = 1024

d_model = 256
n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = -1
pool_n_heads = 4
fetch_n_heads = 2
fetch_gamma = 1.0
tie_heads = True
block1_weight = 1.0
block2_weight = 1.0
block3_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20
