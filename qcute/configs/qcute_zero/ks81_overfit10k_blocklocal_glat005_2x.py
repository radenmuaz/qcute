"""qcute_zero/ks81_overfit10k_blocklocal_glat005_2x: same as ks81_overfit10k_blocklocal_glat02.py
but blocklocal_glat_p=0.05 (much milder swap rate) and steps=2000 (double), to check whether a
lighter, longer-trained GLAT dose changes the inconclusive result from p=0.2/1000 steps (see
docs/status.md 2026-08-24 -- controlled same-seed comparison there came back a wash, with the
apparent win at one prompt actually a repetition-collapse artifact). Paired with
ks81_overfit10k_blocklocal_glat05_2x.py (p=0.5) and a plain baseline at steps=2000, all run with
the same --seed for a controlled comparison.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks81_overfit10k_blocklocal_glat005_2x.py
"""
from pathlib import Path

run_name = "qcute_zero_ks81_overfit10k_blocklocal_glat005_2x"
Ks = (8,)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
input_preset = 8

mtp_heads = 8
mtp_weight = 1.0

blocklocal_seed_weight = 1.0
blocklocal_glat_p = 0.05

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 2000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
log_every = 20
eval_every = 50
eval_batches = 5
