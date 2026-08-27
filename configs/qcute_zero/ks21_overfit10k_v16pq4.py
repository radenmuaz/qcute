"""qcute_zero/ks21_overfit10k_v16pq4: same as ks21_overfit10k.py (Ks=(2,1), byte-level trunk,
input_preset=8) but with the pluggable Quantizer (2026-08-24) set to vocab=16, pq_chunks=4
instead of the default vocab=256, pq_chunks=1 -- product-quantized categorical code, combinatorial
capacity 16**4=65536 (vs the default's flat 256), same "v16pq4" naming convention as this
session's qcute_lagcodec sharing-ablation grid. Tests whether the richer/PQ-structured code
representation trains/overfits comparably to the default flat-256 code at the same overfit10k
scale (n_bytes=10000, context_len=256). mtp_heads=4 added (2026-08-24) so
--eval_decode_mtp_verify can show MTP-drafted speculative decode (verified against NTP) during
training, not just final loss.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k_v16pq4.py --eval_decode_mtp_verify True
"""
from pathlib import Path

run_name = "qcute_zero_ks21_overfit10k_v16pq4"
Ks = (2, 1)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8

quant_type = "simplex"
vocab = 16
pq_chunks = 4

mtp_heads = 4
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
