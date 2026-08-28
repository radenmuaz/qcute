"""summformer/ks421_overfit10k: Ks=(4,2), a 2-fuse-stage hierarchical-summarization config
(2026-08-25) -- overfit10k testbed (see CLAUDE.md's "Standing methodology"), also exercises
check_kv_cache_consistency across a deeper cascade (run automatically at the end of training).
Updated 2026-08-27 to the current Ks convention (n_fuse = len(Ks), no trailing placeholder) --
was Ks=(4,2,1) under the pre-fix summformer_v1.py semantics.

uv run python -m qcute.summformer.summformer --config configs/summformer/ks421_overfit10k.py
"""
from pathlib import Path

run_name = "summformer_ks421_overfit10k"
Ks = (4, 2)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
mtp_heads = 4

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
