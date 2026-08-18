"""qcute.bytelm config: 3-layer variant of bytelm_xs_mtp4_ctx1024.py — same
d_model=256/n_heads=4/mtp_heads=4/context=1024 (xs preset, ctx overridden
to 1024 same as the existing baseline), only n_layers dropped 4 -> 3.
Session ask: "3 layer bytelm same dim" — a new baseline point to compare
against the qcute_refine lineage's own 2-3 level towers, most of which
land closer to 3 effective transformer-layer-equivalents of depth than
bytelm's usual 4-layer default.

Required a new --n_layers override CLI flag on qcute/bytelm.py (mirroring
the existing --context/--mtp_heads override pattern) — bytelm.py's
PRESETS only exposed n_layers baked into each preset before this, no way
to override it from a config file.

steps=4000, not 8000 — this session's own step-budget finding (see
docs/status.md) found bytelm_xs_mtp4_ctx1024 itself bottoms out around
step 1700 and is fully into its overfit region well before step 8000, so
new comparison runs default to the shorter budget.

    uv run python -m qcute.bytelm --config configs/bytelm_xs3_ctx1024.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs3_ctx1024
"""
from pathlib import Path

preset = "xs"
context = 1024
n_layers = 3
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
