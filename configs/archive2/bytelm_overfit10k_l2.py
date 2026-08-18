"""n_layers=2 twin of configs/bytelm_overfit10k_l1.py -- see that file's docstring for full
rationale. Only change here: n_layers 1 -> 2.

    uv run python -m qcute.bytelm --config configs/bytelm_overfit10k_l2.py

    # watch live:
    tail -f logs/bytelm_overfit10k_l2/run.log
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 2
data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1
steps = 1000
batch_size = 16
warmup_steps = 100
cosine_decay = False
lr_peak = 6e-4
log_every = 20
eval_every = 50
eval_batches = 5
