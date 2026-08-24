"""qcute.bytelm_tpu config: revived model-size sweep on tpu5, config 3 of 4 -- see
d768x12_mlp4_48h.py's docstring for the full sweep rationale (tpu4->tpu5 swap, 48h budget,
random lr_peak/grad_clip sweep).

Config 3 (this file): preset d1024x16_mlp4 (269.0M, the anchor-matched config). batch_size=4
known-safe. lr_peak=6e-4 (mid-high end of the sweep), grad_clip=5.0. resid_dropout=0.1,
layer_drop=0.25 (heaviest layer_drop in this sweep), label_smoothing=0.1, beta1=0.9, beta2=0.98.

steps=127000 placeholder (172800s / ~1.36s/it, this preset's previously observed rate) -- correct
from a real measured rate once launched.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/modelsize_48h_lrsweep/d1024x16_mlp4_48h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d1024x16_mlp4_48h
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
resid_dropout = 0.1
layer_drop = 0.25
label_smoothing = 0.1
beta1 = 0.9
beta2 = 0.98
steps = 135000
batch_size = 4
lr_peak = 6e-4
warmup_steps = 2500
cosine_decay = True
constant_steps = 12000
grad_clip = 5.0
weight_decay = 0.01
log_every = 200
eval_every = 7000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
