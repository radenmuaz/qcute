"""qcute.bytelm config: DIAGNOSTIC — 1-layer variant of bytelm_xs3_ctx1024.py
(same d_model=256/n_heads=4/mtp_heads=4/context=1024, xs preset), n_layers
dropped 3 -> 1. Not a fair-comparison baseline — a capacity floor probe:
how good can byte-level NTP/MTP get with an almost-trivial transformer (a
single self-attention+MLP block), on this same data/context/step budget?
Gives a lower reference point for "how much does depth actually buy" when
reading the 3-layer and qcute_refine_v2 results — if val_bpb here is close
to the 3-layer number, depth isn't doing much at this scale; if it's much
worse, depth is pulling real weight.

steps=4000, same budget as every other bytelm/bpelm/qcute_refine_v2 run
this session (see docs/status.md's step-budget finding).

    uv run python -m qcute.bytelm --config configs/bytelm_xs1_ctx1024.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs1_ctx1024
"""
from pathlib import Path

preset = "xs"
context = 1024
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
