"""qcute_zero/ks221_overfit10k_mtp8: Ks=(2,2,1) -- 3 levels (byte, level-1 code over 2 bytes,
level-2 code over 2 level-1 codes = 4 bytes), mtp_heads=8, blocklocal_seed_weight=1.0/code_window=1/
blocklocal_dual_mode=True (2026-08-24 multi-level own-block seed training, generalized this session
to every fuse stage -- trains both "mask" (stage 1 decoupled from stage 0) and "rollout" (stage 1
assumes stage 0's own recursive rollout succeeded, teacher-forced) modes for stage 1's own-block
decode). Same overfit10k scale as this session's other qcute_zero configs.

After training: (1) generate_early_exit vs generate_no_cache -- does stage-0's confident cond
prediction actually agree with the true (deepest, stage-1) prediction often enough to be useful;
(2) generate_free_rollout -- does it still produce plausible output on a genuine 2-fuse-stage
cascade (n_fuse==2, beyond generate_free_rollout's originally-tested n_fuse==1 case -- its own
assert requires n_fuse==1, so this specifically tests whether that assert needs loosening, not a
working generation path yet).

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks221_overfit10k_mtp8.py
"""
from pathlib import Path

run_name = "qcute_zero_ks221_overfit10k_mtp8"
Ks = (2, 2)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
input_preset = 8

mtp_heads = 8
mtp_weight = 1.0

blocklocal_seed_weight = 1.0
blocklocal_dual_mode = True

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
