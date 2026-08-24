"""qcute.bytelm_tpu config: regularization+label_smoothing sweep, config 2 of 4 -- same
d1024x16_mlp2 (168.3M) base as reg_full_smooth.py (see that file's docstring for the full
sweep rationale and mtp_heads=8/batch_size=4 justification).

Config 2 (this file): resid_dropout only (no layer_drop) at 0.1, plus a lighter
label_smoothing=0.05 -- probes whether a single regularization knob plus half-strength smoothing
gets most of the combined-regularization benefit more cheaply (no stochastic-depth compute
variance), and whether weaker smoothing is enough to help without over-softening targets.

steps corrected 2026-08-24 to 66000 from a real measured rate on tpu6 (~1.30s/it at step 99
post-compile) -- see reg_full_smooth.py's docstring for the full correction rationale.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/d1024x16_mlp2_reg_smoothing_sweep/resid_light_smooth.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/resid_light_smooth
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
resid_dropout = 0.1
layer_drop = 0.0
label_smoothing = 0.05
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
