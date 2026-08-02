"""qcute.bytelm config: xs preset on the tiny dataset, extended step budget.

Reproduces the double-descent exploration run — long constant-LR training on
a tiny dataset to see whether val_bpb's rise after the interpolation point
(~step 1500) is the ascending arm of double descent (Nakkiran et al. 2019)
rather than simple monotonic overfitting.

    uv run python -m qcute.bytelm --config configs/bytelm_xs_tiny_longrun.py
"""
from pathlib import Path

preset = "xs"
data = Path("datasets/enwik8_tiny.gz")
val_frac = 0.1
steps = 25000
batch_size = 16
warmup_steps = 200
lr_peak = 6e-4
log_every = 250
eval_every = 500
eval_batches = 20
benchmark_generate_bytes = 128
