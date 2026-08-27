"""qcute.bytelm_tpu config: 24h flash-attention throughput sweep, variant A (lr_peak=6e-4,
constant after warmup — the baseline learning rate already used by
bytelm_tpu_sm_full_enwik8.py). One of 4 variants (see _lr3e4/_lr1e3/_cosine siblings) meant to
run concurrently, one per TPU chip, via `TPU_VISIBLE_CHIPS` — see
docs/tpu_setup.md's "Optional: multiple TPU chips on one host" section. Goal: find which
of these actually converges fastest toward 1.0 bpb, not just which trains fastest per-step.

sm preset (d_model=256, n_layers=8, n_heads=4, mlp_mult=2, ~4.3M params), context=4096,
mtp_heads=1 (MTP disabled) — same architecture as bytelm_tpu_sm_full_enwik8.py.
use_flash_attention=True + batch_size=32: measured directly on this session's v4-8 node,
~1.99-2.0 it/s (vs. plain-SDPA batch_size=8's 3.00 it/s -- 24 samples/s effective) -- ~2.6x more
effective throughput (64 samples/s). batch_size=64 was also tried: ~1.03 it/s, ~65.9 samples/s
effective -- negligible further gain over batch=32, diminishing returns past this point (likely
compute- rather than memory-bound here), so 32 is the sweet spot, not a memory-ceiling-driven
choice like the earlier plain-SDPA batch=8 (that one WAS memory-limited, see
bytelm_tpu_sm_full_enwik8.py's own docstring).

steps=167616 sized for a 24h wall-clock budget at the measured 2.0 it/s, minus ~3% margin for
periodic full-eval overhead: 24h=86400s -> 172800 raw steps -> steps=167616. At batch_size=32,
seq_len=4097 that's ~244 epochs over the ~90M-byte train split -- heavy repetition for a 4.3M
model, expect overfitting well before 24h is up; that's fine, this is a throughput/schedule
exploration as much as a training run, and full_val_eval's checkpointer only ever saves on
val_bpb improvement, never on train loss or step count.

full_val_eval=True: every eval_every steps, BOTH val and test get a full deterministic pass
(chunked through batch_size, never a single giant batch -- see qcute/bytelm_tpu.py's own
comment on that), not just a sampled val estimate. test_bpb is purely observational here (logged,
never drives checkpoint selection) -- watching it throughout training instead of only once at the
end. eval_every=7200 =~ one eval per wall-clock hour at the measured rate (23 evals over the 24h
budget).

warmup_steps=500: NOT scaled to "5% of one epoch" like the shorter sm/tiny configs (an epoch here
is only ~686 steps, so 5% would be an oddly-short ~34-step warmup that doesn't obviously suit a
244-epoch, 24h run) -- picked as a fixed, reasonably-short-relative-to-167616-total value instead.
cosine_decay=False (this variant): constant LR after warmup for the whole run.

    TPU_VISIBLE_CHIPS=0 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sweep24h_lr6e4.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sweep24h_lr6e4
"""
from pathlib import Path

preset = "sm"
data = Path("datasets/enwik8.gz")   # full 100,000,000-byte corpus
val_frac = 0.05
test_frac = 0.05
context = 4096
mtp_heads = 1                       # MTP disabled — plain next-byte prediction only
use_flash_attention = True          # nightly-only, see docs/tpu_setup.md
steps = 167616                      # ~24h at measured throughput, see docstring
batch_size = 32
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False                # constant LR after warmup, no decay
grad_clip = 1.0
log_every = 500
eval_every = 7200                   # ~once per wall-clock hour at measured throughput
full_val_eval = True                # full val AND test pass every eval_every steps, see docstring
eval_batches = 20                   # unused when full_val_eval=True (kept for reference/fallback)
save_every_n_evals = 1
