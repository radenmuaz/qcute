"""qcute.qcutelm config: BSQ bottleneck on the tiny dataset.

Matched-bandwidth companion to configs/bytelm_xs_tiny_longrun.py — same
tiny dataset, same warmup+constant LR schedule, for a head-to-head
comparison at roughly matched param count (~3.9M vs. bytelm's xs ~3.7M).

    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_tiny.py
"""
from pathlib import Path

bottleneck = "bsq"
data = Path("datasets/enwik8_tiny.gz")
val_frac = 0.1
steps = 5000
batch_size = 32
seq_chunks = 32
warmup_steps = 200
lr_peak = 6e-4
log_every = 100
eval_every = 250
eval_batches = 10
qual_gen_bytes = 128
qual_source = "val"
qual_prompt_bytes = 64
