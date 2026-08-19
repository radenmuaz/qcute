"""qcute.bytelm config: xs preset's own default depth (n_layers=4) and
context (256), stated explicitly rather than left implicit, so it's a
direct layer-count ablation pair against bytelm_xs1_ctx256.py (same
context=256, n_layers=1) with matching step budget/eval cadence.

    uv run python -m qcute.bytelm --config configs/bytelm_xs4_ctx256.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs4_ctx256
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 4
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 4000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
