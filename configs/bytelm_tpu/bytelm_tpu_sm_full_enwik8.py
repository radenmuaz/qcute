"""qcute.bytelm_tpu config: small, narrow/deep (not wide) model — sm preset (d_model=256,
n_layers=8, n_heads=4, mlp_mult=2, head_dim=64, ~4.3M params at mtp_heads=1), context bumped to
4096 here (override, not baked into the preset) on the full enwik8 corpus (datasets/enwik8.gz,
100,000,000 bytes) to maximize TPU utilization — more attention/FLOPs work per step keeps the
chip busier than a tiny model at its original context=1024 would. seq_len ends up 4097
(context(4096) + mtp_heads(1)), not itself a power of 2 — context is the architecturally
meaningful, standard-for-comparison quantity (the model's actual attention window; what gets
reported/compared across configs), so it stays the power-of-2 value and the +1 from mtp_heads'
incidental lookahead-target byte is left as-is rather than shaving context to 4095 to round the
sum.

All power-of-2 hyperparams, kept narrow (mlp_mult=2, half the usual mlp_mult=4 default) and deep
(n_layers=8, same depth as the sd/tiny presets) rather than wide — the point of this config is a
fast model, not a param-count target as such; ~4.3M is wherever that lands, not a number aimed
for. (An earlier version of this preset used mlp_mult=8 to hit ~10.6M as a closer match to a 10M
param target — superseded once the actual goal was clarified as "fast, not fat/wide": mlp_mult=2
is what stayed.)

mtp_heads=1: MTP disabled (plain next-byte prediction only), matching
configs/bytelm_tpu/bytelm_tpu_tiny_full_enwik8.py's convention for these architecture-comparison
runs (keeps the comparison clean, not entangled with MTP-specific behavior).

3 epochs over the ~90M-byte train split (val_frac=0.05, test_frac=0.05): seq_len =
context(4096) + mtp_heads(1) = 4097, steps_per_epoch = 90,000,000 / (batch_size(128) * 4097)
=~ 171.6 -> steps = round(171.6 * 3) = 515. warmup_steps=9 =~ 5% of one epoch, then constant
LR (cosine_decay=False), same schedule convention as the tiny_full_enwik8 config. batch_size=128
(not larger) is a memory/utilization tradeoff specific to context=4096 — activation memory scales
with context, so the batch size that maximized utilization at context=1024 doesn't directly
transfer; watch actual TPU memory/compute utilization via `tpu-info` on the first run and retune
batch_size from there if it's leaving the chip underused or OOMing.

    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sm_full_enwik8.py --device xla

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sm_full_enwik8
"""
from pathlib import Path

preset = "sm"
data = Path("datasets/enwik8.gz")   # full 100,000,000-byte corpus
val_frac = 0.05
test_frac = 0.05
mtp_heads = 1                       # MTP disabled — plain next-byte prediction only
steps = 2058                        # 3 epochs over the ~90M-byte train split, see docstring
batch_size = 128
lr_peak = 6e-4
warmup_steps = 34                   # ~5% of one epoch
cosine_decay = False                # constant LR after warmup, no decay
grad_clip = 1.0
log_every = 50
eval_every = 200
eval_batches = 20
save_every_n_evals = 1
