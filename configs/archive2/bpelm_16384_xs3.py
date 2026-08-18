"""qcute.bpelm config: 3-layer variant at vocab=16384 (bpe_enwik8_1M_16384
already trained this session via scripts/train_bpe.py — see
configs/bpelm_32768.py's own docstring for the vocab-size-vs-bytes/token
measurements, including 16384's 3.53 bytes/token). Same d_model=256/
n_heads=4/context=256 pattern as bpelm_32768.py, only n_layers dropped
4 -> 3 (--n_layers already a supported override on qcute/bpelm.py,
unlike bytelm.py which needed a new flag added this session).

Session ask: "3 layer 16384 bpe" — a new baseline point to compare
against the qcute_refine lineage's own 2-3 level towers, alongside
bytelm_xs3_ctx1024.py's 3-layer bytelm counterpart.

steps=4000, not 8000 — this session's own step-budget finding (see
docs/status.md): new comparison runs default to the shorter budget.

    uv run python -m qcute.bpelm --config configs/bpelm_16384_xs3.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_16384_xs3
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_16384.model")
data = Path("datasets/enwik8_1M.gz")
context = 256
d_model = 256
n_layers = 3
n_heads = 4
val_frac = 0.1
steps = 4000
batch_size = 16
warmup_steps = 500
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
