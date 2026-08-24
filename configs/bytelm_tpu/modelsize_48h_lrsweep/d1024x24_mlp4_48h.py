"""qcute.bytelm_tpu config: revived model-size sweep on tpu5, config 4 of 4 -- see
d768x12_mlp4_48h.py's docstring for the full sweep rationale (tpu4->tpu5 swap, 48h budget,
random lr_peak/grad_clip sweep).

Config 4 (this file): preset d1024x24_mlp4 (403.2M, the biggest/deepest config in this sweep).
batch_size=2 (known-safe at mtp_heads=1 from the original tpu4 sweep) genuinely OOM'd here at
mtp_heads=8 (2026-08-24): "Attempting to reserve 25.75G ... 25.64G free" -- a ~110MB near-miss,
the extra logits/cross-entropy tensors from 8 heads instead of 1 tipped this already-borderline
config over. Switched to batch_size=1, grad_accum_steps=2 (same effective_batch=2 as before, but
each microbatch only needs the batch=1 memory footprint -- relies on the mark_step()-per-microbatch
fix already in bytelm_tpu.py's training loop, see docs/status.md). lr_peak=8e-4 (highest in the
sweep), grad_clip=10.0 (loosest, pairs with the highest LR since this is also the largest model --
more headroom before instability at this batch/LR combo is untested, flagged for monitoring).
resid_dropout=0.15, layer_drop=0.15, label_smoothing=0.05 (lightest smoothing in this sweep),
beta1=0.9, beta2=0.99 (highest beta2 in the sweep -- more momentum smoothing for the
biggest/noisiest-gradient model).

steps=151000 placeholder (172800s / ~1.14s/it, this preset's previously observed rate at
mtp_heads=1/batch_size=2 -- was the fastest of the four originally despite being the largest
model) -- NOT yet re-measured at the new batch_size=1/grad_accum_steps=2/mtp_heads=8 combo;
correct from a real measured rate once launched.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/modelsize_48h_lrsweep/d1024x24_mlp4_48h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d1024x24_mlp4_48h
"""
from pathlib import Path

preset = "d1024x24_mlp4"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
mtp_heads = 8
resid_dropout = 0.15
layer_drop = 0.15
label_smoothing = 0.05
beta1 = 0.9
beta2 = 0.99
steps = 151000
batch_size = 1
grad_accum_steps = 2
lr_peak = 8e-4
warmup_steps = 3000
cosine_decay = True
constant_steps = 15000
grad_clip = 10.0
weight_decay = 0.01
log_every = 200
eval_every = 8000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
