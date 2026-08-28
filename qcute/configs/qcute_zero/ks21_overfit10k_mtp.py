"""qcute_zero/ks21_overfit10k_mtp: cfg.mtp_heads=4 -- extra untied linear heads reading the SAME
final hidden state as head0 (post-cascade cond readout), MTP-style (see qcute.bytelm), predicting
bytes t+2..t+4 in addition to head0's own t+1. Supersedes the pruned query_vec/parallel_decode
mechanism (docs/status.md's "parallel block decode brainstorm" section, 2026-08-22 pruning note) --
that idea's density problem (one query_vec slot = one full attention-stack pass) is fixed by these
heads reusing the SAME per-position hidden state instead, at every position, not just sampled
clusters. Real validation targets: do mtp{2,3,4}_acc climb non-trivially above chance, and
(post-training) does generate_speculative's accept_rate come out non-trivially above chance.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k_mtp.py
"""
from pathlib import Path

run_name = "qcute_zero_ks21_overfit10k_mtp"
Ks = (2,)
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
