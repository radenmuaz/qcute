"""qcute.bytelm baseline for the tiny-overfit sanity-check batch (see
configs/qcute_refine_v4_4_overfit10k_k4_1_l1.py for the full rationale): same n_bytes=10000,
context=256, steps=1000 as every qcute_refine_v4_4_overfit10k_* config in the batch, so this run's
train/val bpb is a directly comparable byte-level reference point -- does a plain byte LM at this
tiny scale/budget produce plausible generated text where qcute_refine's collapsed-code variants do
not? n_layers=1 (paired with bytelm_overfit10k_l2.py's n_layers=2, matching the qcute_refine
batch's own 1-layer/2-layer split).

    uv run python -m qcute.bytelm --config configs/bytelm_overfit10k_l1.py

    # watch live:
    tail -f logs/bytelm_overfit10k_l1/run.log
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 1
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
