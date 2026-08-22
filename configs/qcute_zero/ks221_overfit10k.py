"""qcute_zero/ks221_overfit10k: Ks=(2,2,1), two fuse stages (periods 2 and 4 bytes) cascading
through the SAME shared LM. No curriculum -- this is the hard case qcute_v1's StackDecoder needed
curriculum_max_srcs/curriculum_step to get working (see docs/status.md's 2026-08-21/22 entry);
qcute_zero's design expects to not need that (see qcute_zero.py's module docstring). Same
scale/methodology as ks21_overfit10k.py for direct comparison.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks221_overfit10k.py
"""
from pathlib import Path

run_name = "qcute_zero_ks221_overfit10k"
Ks = (2, 2, 1)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
log_every = 20
eval_every = 50
eval_batches = 5
