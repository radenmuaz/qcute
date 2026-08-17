"""qcute.bytelm config: 2-layer variant of the xs preset at context=256 (between
bytelm_xs1_ctx256.py's n_layers=1 and bytelm_xs4_ctx256.py's n_layers=4) -- a fairer
depth-matched baseline against the v5_stack/v5_concat decode designs, whose own
per-block decode adds extra effective depth (self-attention + cross-attention
stages) beyond a plain n_layers=1 encoder, closer to n_layers=2 in raw layer count.

    uv run python -m qcute.bytelm --config configs/bytelm_xs2_ctx256.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs2_ctx256
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 2
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
