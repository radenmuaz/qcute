"""qcute.bytelm_tpu config: power-of-2 model-size + heavier-regularization sweep, config 2 of 4
-- see d512x16_heavyreg.py's docstring for the full sweep rationale.

Config 2 (this file): preset d1024x16_mlp4 (268.7M, 16 layers) -- the anchor-matched size from
the earlier (now-stopped) model-size sweep, here with heavier regularization (resid_dropout=0.15,
layer_drop=0.15, label_smoothing=0.1) and tuned AdamW (beta2=0.98) instead of that sweep's
resid_dropout=0.1/layer_drop=0.1/no smoothing.

batch_size=4 -- known-safe from the earlier model-size sweep (bs=8 OOM'd there). No grad_accum
needed. steps=63000 placeholder (86400s / ~1.36s/it, this preset's previously observed rate on
tpu4) -- correct from a real measured rate once launched.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/power2_heavyreg_sweep/d1024x16_mlp4_heavyreg.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d1024x16_mlp4_heavyreg
"""
from pathlib import Path

preset = "d1024x16_mlp4"
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
steps = 126000
batch_size = 4
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
