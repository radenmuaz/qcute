"""qcute.qcute_bytepool config (variant=v13): full-scale run of the
3-layer cross-attention cascade (byte/pair/quad, coarse-to-fine
speculative-decoding-shaped). d_model=256 uniform across all layers per
the session spec ("all layer must have same hidden dim e.g. 256 for first
trial"). context_len=1024 for rough comparability with the other bsq/
baseline runs (must stay a multiple of 4).

    uv run python -m qcute.qcute_bytepool --config configs/qcute_bytepool_v13.py
"""
from pathlib import Path

variant = "v13"
context_len = 1024

d_model = 256
n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = -1
cross_n_heads = 4
fetch_n_heads = 2
fetch_gamma = 1.0
tie_heads = True
layer1_weight = 1.0
layer2_weight = 1.0
layer3_weight = 1.0

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
