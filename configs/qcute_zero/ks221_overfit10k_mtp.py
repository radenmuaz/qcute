"""qcute_zero/ks221_overfit10k_mtp: same as ks21_overfit10k_mtp.py but Ks=(2,2,1), exercising the
multi-stage cascade. cfg.mtp_heads=4: extra untied linear heads reading the SAME final hidden state
as head0, MTP-style, predicting t+2..t+4 in addition to head0's own t+1. See
ks21_overfit10k_mtp.py's own docstring for the full rationale (supersedes the pruned query_vec/
parallel_decode mechanism).

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks221_overfit10k_mtp.py
"""
from pathlib import Path

run_name = "qcute_zero_ks221_overfit10k_mtp"
Ks = (2, 2, 1)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8
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
