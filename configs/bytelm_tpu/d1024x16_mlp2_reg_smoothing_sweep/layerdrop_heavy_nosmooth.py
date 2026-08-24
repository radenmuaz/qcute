"""qcute.bytelm_tpu config: regularization+label_smoothing sweep, config 3 of 4 -- same
d1024x16_mlp2 (168.3M) base as reg_full_smooth.py (see that file's docstring for the full
sweep rationale and mtp_heads=8/batch_size=4 justification).

Config 3 (this file): layer_drop only (no resid_dropout), pushed heavier at 0.15, with
label_smoothing=0.0 (off) -- acts as the sweep's control for smoothing's effect (isolates
layer_drop's own strength/ceiling without any smoothing interaction) and separately tests whether
layer_drop alone, pushed past the 0.1 that won on tpu5's d512_16L grid, still helps or starts
hurting at this larger 168.3M model size.

steps corrected 2026-08-24 to 66000 from a real measured rate on tpu6 (~1.26s/it at step 99
post-compile) -- see reg_full_smooth.py's docstring for the full correction rationale.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/d1024x16_mlp2_reg_smoothing_sweep/layerdrop_heavy_nosmooth.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/layerdrop_heavy_nosmooth
"""
from pathlib import Path

preset = "d1024x16_mlp2"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
mtp_heads = 8
resid_dropout = 0.0
layer_drop = 0.15
label_smoothing = 0.0
steps = 66000
batch_size = 4
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 5000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
