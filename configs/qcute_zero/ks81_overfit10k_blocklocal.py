"""qcute_zero/ks81_overfit10k_blocklocal: Ks=(8,1), n_fuse=1 (single fuse stage, matching
generate_free_rollout's own n_fuse==1 requirement), blocklocal_seed_weight=1.0/code_window=1 --
trains the seed mechanism generate_free_rollout was rewritten (2026-08-24) to use instead of its
old causally-invalid ad hoc mask. mtp_heads=8, matching this session's convention. Same overfit10k
scale as this session's other qcute_zero configs.

After training: run generate_free_rollout and check output quality/coherence -- this is the first
checkpoint that can actually exercise the fixed own-block-seed mechanism for real (the earlier
ks21_v16pq4 checkpoint has n_fuse==1 but was never trained with blocklocal_seed_weight>0, so its
seed_embed/fuse_stages[0] own-code cross-attention would be untrained noise).

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks81_overfit10k_blocklocal.py
"""
from pathlib import Path

run_name = "qcute_zero_ks81_overfit10k_blocklocal"
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
