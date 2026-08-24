"""qcute.bytelm_tpu config: revived model-size sweep on tpu5, config 2 of 4 -- see
d768x12_mlp4_48h.py's docstring for the full sweep rationale (tpu4->tpu5 swap, 48h budget,
random lr_peak/grad_clip sweep).

Config 2 (this file): preset d1024x16_mlp2 (168.3M). batch_size=4 known-safe. lr_peak=3e-4
(lighter end of the sweep), grad_clip=2.0. resid_dropout=0.2, layer_drop=0.15,
label_smoothing=0.15 (heaviest smoothing in this sweep), beta1=0.9, beta2=0.95 (unchanged from
project default, the one sibling in this sweep NOT using the tuned 0.98).

steps=138000 placeholder (172800s / ~1.25s/it, this preset's previously observed rate) -- correct
from a real measured rate once launched.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/modelsize_48h_lrsweep/d1024x16_mlp2_48h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d1024x16_mlp2_48h
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
resid_dropout = 0.2
layer_drop = 0.15
label_smoothing = 0.15
beta1 = 0.9
beta2 = 0.95
steps = 146000
batch_size = 4
lr_peak = 3e-4
warmup_steps = 2000
cosine_decay = True
constant_steps = 10000
grad_clip = 2.0
weight_decay = 0.01
log_every = 200
eval_every = 7500
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
