"""qcute.bytelm_tpu config: 24h flash-attention throughput sweep, variant B (lr_peak=3e-4, half
of variant A's 6e-4, constant after warmup). See bytelm_tpu_sweep24h_lr6e4.py for the full
rationale (architecture, flash-attention/batch_size throughput measurement, step-count sizing,
eval cadence) — this file only overrides lr_peak, everything else is identical by design so the
4 variants are a clean apples-to-apples comparison.

    TPU_VISIBLE_CHIPS=1 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sweep24h_lr3e4.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sweep24h_lr3e4
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
lr_peak = 3e-4
warmup_steps = 500
cosine_decay = False
grad_clip = 1.0
log_every = 500
eval_every = 7200
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
