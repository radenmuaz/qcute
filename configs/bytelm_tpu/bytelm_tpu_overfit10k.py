"""qcute.bytelm_tpu config: fast-iteration smoke test — xs preset (~3.7M params) on the standard
n_bytes=10000 slice (see CLAUDE.md's "Standing methodology"). Use this to confirm bytelm_tpu.py
itself (device selection, zero_kv_sink off by default, checkpointing) works — on CPU locally or
on a TPU VM — before launching the full-scale configs/bytelm_tpu/bytelm_tpu_sd_full_enwik8.py run.

    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_overfit10k.py

    # explicit device (auto-detects xla if torch_xla+TPU is importable, else cpu):
    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_overfit10k.py --device xla

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_overfit10k
"""
from pathlib import Path

preset = "xs"
context = 256
n_layers = 4
data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1
steps = 1000
batch_size = 16
warmup_steps = 100
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 10
