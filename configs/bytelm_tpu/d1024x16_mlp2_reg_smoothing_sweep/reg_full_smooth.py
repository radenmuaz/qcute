"""qcute.bytelm_tpu config: regularization+label_smoothing sweep, config 1 of 4 -- fixed model
size d1024x16_mlp2 (168.3M, same preset across all 4 siblings in this folder), context=8192,
flash-attention, no_zero_kv_sink, batch_size=4, mtp_heads=8 (up from the model-size sweep's
mtp_heads=1 -- per-request, all 4 configs in this folder use 8 parallel next-byte heads; the extra
heads are cheap linear layers (d_model->vocab) reading the same final hidden state, no extra
attention FLOPs, so batch_size=4 wasn't re-bench-probed -- seq_len only grows 8192->8200 and the
head params are a few hundred K vs the model's 168M, both negligible memory deltas).

This is a *random/smart-guess* sweep, not a clean factorial ablation (that was already done on
tpu5's d512_16L regularization grid -- see docs/status.md, winner was resid_dropout=0.1 +
layer_drop=0.1 combined). This folder instead probes 4 distinct, deliberately varied points to
see how label_smoothing interacts with that known-good regularization combo and with lighter/
heavier single-knob variants, rather than a full 2^3 grid.

Config 1 (this file): the tpu5 sweep's known-best regularization combo (resid_dropout=0.1,
layer_drop=0.1) plus standard label_smoothing=0.1 -- tests whether smoothing adds on top of an
already-strong regularization combo or is redundant with it.

steps corrected 2026-08-24 from a real measured rate on tpu6 (fresh v4-8, nightly torch_xla for
flash-attention): ~1.31s/it at step 99 post-compile (mtp_heads=8's extra heads do add a small
measurable slowdown vs the sibling model-size sweep's mtp_heads=1, ~1.25s/it) -- 86400s/1.31s/it
~= 65954, rounded to steps=66000 (down slightly from the 69000 placeholder). eval_every=5000
unchanged (~13 evals over the corrected budget).

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/d1024x16_mlp2_reg_smoothing_sweep/reg_full_smooth.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/reg_full_smooth
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
layer_drop = 0.1
label_smoothing = 0.1
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
