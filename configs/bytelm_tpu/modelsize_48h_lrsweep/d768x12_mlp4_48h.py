"""qcute.bytelm_tpu config: revived model-size sweep on tpu5, config 1 of 4 -- this is the
original tpu4 model-size sweep (d768x12_mlp4/d1024x16_mlp2/d1024x16_mlp4/d1024x24_mlp4, stopped
on tpu4 2026-08-24 when tpu4 switched to the power-of-2 sweep), relaunched here on tpu5 (freed up
after stopping its regularization sweep) so tpu4's power2_heavyreg_sweep can keep running
undisturbed. 48h budget (up from the original 24h), heavier/varied regularization per config
(random-smart-guess style, not a uniform grid), tuned AdamW betas, and a random lr_peak/grad_clip
sweep across the 4 siblings (user request: "use heavier lr like 5e-4, sweep random ... and grad
clip") instead of the uniform lr_peak=1e-4/grad_clip=10.0 used everywhere else this session.

Config 1 (this file): preset d768x12_mlp4 (113.7M, GPT-2-small shape). batch_size=4 known-safe
(from the original tpu4 sweep). lr_peak=5e-4 (5x this session's usual 1e-4), grad_clip=1.0 (tight,
pairs with the higher LR for stability). resid_dropout=0.15, layer_drop=0.2, label_smoothing=0.1,
beta1=0.9, beta2=0.98.

steps=230000 placeholder (172800s / ~0.75s/it, this preset's previously observed rate on tpu4) --
NOT yet re-probed on tpu5 specifically or at this LR; correct from a real measured rate once
launched, per the standing session lesson about noisy-probe extrapolation.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/modelsize_48h_lrsweep/d768x12_mlp4_48h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d768x12_mlp4_48h
"""
from pathlib import Path

preset = "d768x12_mlp4"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
mtp_heads = 8
resid_dropout = 0.15
layer_drop = 0.2
label_smoothing = 0.1
beta1 = 0.9
beta2 = 0.98
steps = 249000
batch_size = 4
lr_peak = 5e-4
warmup_steps = 3000
cosine_decay = True
constant_steps = 15000
grad_clip = 1.0
weight_decay = 0.01
log_every = 200
eval_every = 13000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
