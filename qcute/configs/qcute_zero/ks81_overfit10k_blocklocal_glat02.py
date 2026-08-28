"""qcute_zero/ks81_overfit10k_blocklocal_glat02: same as ks81_overfit10k_blocklocal.py but adds
blocklocal_glat_p=0.2 (2026-08-24 GLAT-style scheduled sampling for the block-local decode's
within-block bytes) -- tests whether it closes the train-vs-free-run gap diagnosed in the base
checkpoint (teacher-forced local_acc ~0.68 vs generate_free_rollout's actual free-run local_acc
~0.036). Two batched passes per block: pass 1 teacher-forced (as before), pass 2 reruns the same
block-local decode with each within-block position's true byte swapped w.p. 0.2 for an STE
self-predicted byte (soft-mix gradient into pass 1's local_logits/head, hard argmax forward value).
Loss is additive (both passes' losses always summed), not skip-real -- mirrors encoder_ste_p's
more-stable additive variant. See docs/status.md 2026-08-24.

After training: compare blocklocal0_glat_acc (pass-2, partially self-fed) against
blocklocal0_local_acc (pass-1, fully teacher-forced) on train/val, then rerun generate_free_rollout
vs generate_no_cache on the same prompts as the base run and compare exact-match rate.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks81_overfit10k_blocklocal_glat02.py
"""
from pathlib import Path

run_name = "qcute_zero_ks81_overfit10k_blocklocal_glat02"
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
blocklocal_glat_p = 0.2

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
