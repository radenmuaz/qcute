"""qcute.bytelm config: same as bytelm_xs4_ctx256_mtp1.py (xs preset, n_layers=4,
mtp_heads=1, no multi-token-prediction) but context=1024 (not 256) -- matches
the context=1024 convention used by configs/archive2/bytelm_xs_mtp4_ctx1024.py,
with a longer step budget for the longer context.

    uv run python -m qcute.bytelm --config configs/bytelm_xs4_ctx1024_mtp1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs4_ctx1024_mtp1
"""
from pathlib import Path

preset = "xs"
context = 1024
n_layers = 4
mtp_heads = 1
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
