"""qcute.bytelm config: same as bytelm_xs4_ctx256.py (xs preset, n_layers=4,
context=256) but mtp_heads=1 -- no multi-token-prediction, plain single
next-byte head. Ablation pair against bytelm_xs4_ctx256.py's mtp_heads=4
default to isolate whatever MTP itself contributes.

    uv run python -m qcute.bytelm --config configs/bytelm_xs4_ctx256_mtp1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs4_ctx256_mtp1
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 4
mtp_heads = 1
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
