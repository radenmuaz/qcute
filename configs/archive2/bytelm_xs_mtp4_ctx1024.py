"""qcute.bytelm config: same xs/mtp4 architecture as bytelm_xs_mtp4.py but
context=1024 (not 256) — matches bpelm_8192's ~1024-byte effective span
(256 BPE tokens x ~4 bytes/token) and qcutelm_vlt6's context_len=1024, so
all three baselines/tokenizer-LM see the same amount of raw text per
example. Everything else (d_model=256, n_layers=4, n_heads=4, mtp_heads=4)
stays the xs preset's default — only --context is overridden.

    uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs_mtp4_ctx1024
"""
from pathlib import Path

preset = "xs"
context = 1024
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 8000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
