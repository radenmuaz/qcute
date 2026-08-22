"""qcute.bytelm_tpu config: 24h flash-attention throughput sweep, variant D — same lr_peak=6e-4
as variant A (bytelm_tpu_sweep24h_lr6e4.py), but with cosine_decay=True instead of a constant LR:
warmup(500 steps) -> constant at peak (constant_steps=10000, ~1.4h at the measured rate) -> cosine
decay down to 0.1*lr_peak over the remaining ~157000 steps. Isolates schedule shape from lr
magnitude — compare directly against variant A to see whether decay (not just a different peak
lr) is what actually helps convergence toward 1.0 bpb over a run this long. See
bytelm_tpu_sweep24h_lr6e4.py for the full rationale (architecture, flash-attention/batch_size
throughput measurement, step-count sizing, eval cadence) — everything but the LR schedule is
identical by design.

    TPU_VISIBLE_CHIPS=3 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sweep24h_cosine.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sweep24h_cosine
"""
from pathlib import Path

preset = "sm"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 4096
mtp_heads = 1
use_flash_attention = True
steps = 167616
batch_size = 32
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = True                 # the only difference from bytelm_tpu_sweep24h_lr6e4.py
constant_steps = 10000              # ~1.4h at peak before decay begins
grad_clip = 1.0
log_every = 500
eval_every = 7200
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
