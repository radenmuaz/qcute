"""bytelm_queryvec/xs_overfit10k: the query_vec/parallel-decode idea (pruned from qcute_zero
2026-08-22, see qcute_zero's own module docstring) preserved as its own standalone testbed on the
plain bytelm trunk -- no fuse stages/codes, query_vec clusters cross-attend directly into the real
trunk's own per-layer K/V. Trains query_vec_n_blocks=4 independent clusters of
query_vec_cluster_len=4 contiguous slots per step, jointly with the ordinary mtp_heads loss.
Validation target: does query_vec_acc climb non-trivially above chance.

uv run python -m qcute.bytelm_queryvec.bytelm_queryvec --config configs/bytelm_queryvec/xs_overfit10k.py
"""
from pathlib import Path

run_name = "bytelm_queryvec_xs_overfit10k"
preset = "xs"
context = 256
mtp_heads = 4
query_vec_cluster_len = 4
query_vec_n_blocks = 4
query_vec_weight = 1.0

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
