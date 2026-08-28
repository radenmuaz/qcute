"""qcute.bytelm config: same as bytelm_xs4_ctx256.py (xs preset, n_layers=4,
mtp_heads=4 default) but context=1024 (not 256) -- context-length ablation pair,
same seed, longer step budget for the longer context (matches bytelm_xs4_ctx1024_mtp1.py's).

    uv run python -m qcute.bytelm --config configs/bytelm_xs4_ctx1024.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs4_ctx1024
"""
from pathlib import Path

preset = "xs"
context = 1024
n_layers = 4
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 8000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
