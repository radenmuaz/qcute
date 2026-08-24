"""qcute_zero/ks1_overfit10k_wavefront2: pure byte-level (Ks=(1,), no fuse stage -- generate_wavefront
is byte-level-only for now, bypassing the fuse cascade entirely, see its own docstring) with
mtp_heads=5, the minimum needed to bootstrap a 2-wave wavefront decode's last wave's first token
at K=8 (max_offset = (n_waves-1)*region_len+1 = 1*4+1 = 5). Same overfit10k scale as this
session's other qcute_zero configs (n_bytes=10000, context_len=256). After training, run
scripts/qual_wavefront_check.py against the resulting checkpoint to qualitatively check
generate_wavefront(n_waves=2) output against generate_no_cache/generate_kv_cache.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks1_overfit10k_wavefront2.py
"""
from pathlib import Path

run_name = "qcute_zero_ks1_overfit10k_wavefront2"
Ks = (1,)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
input_preset = 8

mtp_heads = 5
mtp_weight = 1.0

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
