"""qcute.bytelm_tpu config: power-of-2 model-size + heavier-regularization sweep, config 1 of 4
-- replaces the earlier tpu4 model-size sweep (stopped 2026-08-24 per user request, after
tpu5's bothdrop sibling showed the whole regularization sweep decelerating toward a plateau
around val_bpb~1.13, well short of 1.0 -- see docs/status.md). This sweep instead varies model
size (power-of-2 widths/depths only, unlike the old d768x12_mlp4) with heavier regularization
than the winning tpu5 combo (resid_dropout=0.1/layer_drop=0.1) and tuned AdamW betas.

Config 1 (this file): preset d512x16 (67.3M, 16 layers), the smallest/fastest of the four
siblings -- directly comparable to tpu5's bothdrop (same size, same architecture) but with
heavier regularization (0.15/0.15 vs bothdrop's 0.1/0.1) and label_smoothing=0.1 (bothdrop had
none) to test whether pushing regularization further delays/raises bothdrop's observed plateau.
beta2=0.98 (up from the project default 0.95) per user request to tune AdamW.

batch_size=8 -- known-safe from the earlier tpu5 d512_16L sweep (same preset/context/flash
setup), no grad_accum needed. steps=66000 placeholder (86400s / ~1.3s/it, matching the tpu5
sweep's observed rate) -- NOT yet re-probed on tpu4 specifically; correct from a real measured
rate once launched, per this session's standing lesson about noisy-probe extrapolation.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/power2_heavyreg_sweep/d512x16_heavyreg.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d512x16_heavyreg
"""
from pathlib import Path

preset = "d512x16"
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
label_smoothing = 0.1
beta1 = 0.9
beta2 = 0.98
steps = 132000
batch_size = 8
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 7000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
