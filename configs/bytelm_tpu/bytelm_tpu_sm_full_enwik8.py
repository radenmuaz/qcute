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

batch_size=8, not larger — this took two OOM iterations to find, both confirmed directly against
the running TPU via its own compile-time error message (not estimated in advance):

  1. batch_size=128: `F.scaled_dot_product_attention` on this torch_xla backend materializes the
     full [B*H, T, T] attention-score matrix rather than using a fused/flash kernel — a single
     f32[512,4096,4096] buffer, 34.36GB, already exceeded the v6e-1's 31.25GB HBM budget on its
     own, before counting anything else.
  2. batch_size=16 (attention buffer alone down to ~4.3GB, should have had headroom): still OOM'd
     — XLA's own error reported the *whole compiled program* needs 37.85GB (used 37.85G of
     31.25G, exceeded by 6.61G), meaning per-layer backward-pass intermediates across all 8
     layers (no gradient checkpointing implemented here) dominate, not just the attention buffer
     alone. That total scales roughly linearly with batch_size (weights are tiny at 4.3M params,
     so batch-dependent activations dominate) — 37.85GB at B=16 implies ~18.9GB at B=8, with
     comfortable headroom under 31.25GB.

No windowed/chunked attention or gradient checkpointing implemented here to relieve this —
context=4096 simply forces a small batch_size for this model, not a target chosen for its own
sake. If a future run wants a larger batch at this context, one of those two would need adding.

Sized for a 12h wall-clock budget, not a fixed epoch count: measured directly on-TPU (this
config, same batch_size/context, `tpu-info` confirmed 19.9GB/31.25GB HBM, 75% duty cycle) at a
steady 4.64 it/s once past the initial XLA-compile step. 12h = 43200s -> 200,448 steps at that
raw rate; reserving ~5% for periodic eval overhead (eval_every below) gives steps=190000. At
batch_size=8, seq_len=4097 that's ~6.24B tokens =~ 69 epochs over the ~90M-byte train split —
heavy repetition for a 4.3M-param model, expect it to overfit well before 12h is up; the
val-driven checkpointer (best.pt) and the held-out final_test_bpb (scored once, from that best
checkpoint, never used to pick anything mid-run) are what make that legible rather than a
problem to avoid — this run is a TPU-utilization/throughput exercise as much as a training run.
warmup_steps=137 (~5% of one epoch's ~2746 steps, unchanged from the epoch-relative definition
used elsewhere in this file) and constant LR after warmup (cosine_decay=False) are both still
epoch-relative choices, not step-count-relative, so they didn't need to change when steps did.

    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sm_full_enwik8.py --device xla

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sm_full_enwik8
"""
from pathlib import Path

preset = "sm"
data = Path("datasets/enwik8.gz")   # full 100,000,000-byte corpus
val_frac = 0.05
test_frac = 0.05
context = 4096                      # bumped from sm preset's own default — see docstring
mtp_heads = 1                       # MTP disabled — plain next-byte prediction only
steps = 190000                      # ~12h at the measured 4.64 it/s, minus ~5% eval overhead margin, see docstring
batch_size = 8                      # memory-limited at context=4096 — see docstring (2 OOM iterations)
lr_peak = 6e-4
warmup_steps = 137                  # ~5% of one epoch
cosine_decay = False                # constant LR after warmup, no decay
grad_clip = 1.0
log_every = 200
eval_every = 800
eval_batches = 20
save_every_n_evals = 1
