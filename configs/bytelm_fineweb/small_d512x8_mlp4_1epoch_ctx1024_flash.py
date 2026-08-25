"""qcute.bytelm_fineweb config: small starter model on FineWeb-Edu sample-10BT, byte-level, plain
concatenation (no --use_separator). PRESETS["d512x8_mlp4"] (d_model=512, n_layers=8, n_heads=8,
mlp_mult=4, mtp_heads=4, ~34.2M params -- measured, see run log's own params= line), context
overridden to 1024 (from the preset's 2048) and flash-attention turned on to maximize batch size
on a single v4 chip's 31.75GB HBM -- see docs/bytelm_tpu_setup.md's "FineWeb-Edu byte-level
training" section for the full batch-size/context sweep this config's numbers come from:

  batch_size=256 @ context=1024, flash: OOM (49.04G needed)
  batch_size=128 @ context=1024, flash: fits, ~1.13 it/s steady state  <- this config
  batch_size=128 @ context=2048, flash: OOM (49.04G needed)
  batch_size=64  @ context=2048, flash: fits, ~1.4-1.5 it/s steady state
  batch_size=32  @ context=2048, no flash: fits (flash unavailable on the stable venv)

Requires the NIGHTLY torch/torch_xla build (`.venv-nightly`, not `.venv`) -- see
docs/bytelm_tpu_setup.md's "3.5. Nightly build" section -- and MUST launch with
`--no_zero_kv_sink --use_flash_attention` together (baked into this config as no_zero_kv_sink=True
below; use_flash_attention is a CLI-only flag, always pass `--use_flash_attention` explicitly).

warmup_steps=1000 then constant lr_peak=5e-4 (cosine_decay left off -- lr_at's own behavior is
exactly warmup-then-constant, no separate flag needed for this schedule).

steps sized for 1 epoch over train.bin (45,109,723,621 bytes) at batch_size=128,
seq_len=context+mtp_heads=1028:

    steps_per_epoch = 45,109,723,621 / (128 * 1028) ~= 342,821 steps
    at ~1.13 it/s steady state: ~84 hours (~3.5 days)

    uv run python -m qcute.bytelm_fineweb --config configs/bytelm_fineweb/small_d512x8_mlp4_1epoch_ctx1024_flash.py --device xla --use_flash_attention

    # plot after/during training:
    uv run python scripts/plot_run.py logs/small_d512x8_mlp4_1epoch_ctx1024_flash
"""
from pathlib import Path

preset = "d512x8_mlp4"
train_data = Path("datasets/fineweb_edu_10BT/train.bin")
val_data = Path("datasets/fineweb_edu_10BT/val.bin")
context = 1024
no_zero_kv_sink = True
no_torch_compile = True  # kept off -- not yet verified in combination with flash-attention on this node
steps = 342821
batch_size = 128
lr_peak = 5e-4
warmup_steps = 1000
cosine_decay = False
grad_clip = 1.0
weight_decay = 0.1
log_every = 100
eval_every = 2000
eval_batches = 20
save_every_n_evals = 1
