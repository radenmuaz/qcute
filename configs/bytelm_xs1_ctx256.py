"""qcute.bytelm config: DIAGNOSTIC — 1-layer variant of the xs preset at its
own default context=256 (not bytelm_xs1_ctx1024.py's context=1024). Same
capacity-floor probe as that config, but at the xs preset's native context
length, for a fair layer-count ablation against bytelm_xs4_ctx256.py
(same context=256, n_layers=4) rather than against a longer-context run.

    uv run python -m qcute.bytelm --config configs/bytelm_xs1_ctx256.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs1_ctx256
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 1
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
