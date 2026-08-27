"""qcute.bytelm_fineweb config: first real run on FineWeb-Edu sample-10BT, byte-level (no
tokenizer -- see qcute/bytelm_fineweb.py's own docstring), plain concatenation (no document
separator -- the module's default). sd preset (d_model=1024, n_layers=8, n_heads=16, context=2048,
mtp_heads=8, ~101M params), same shape as qcute.bytelm_tpu's full-enwik8 target run.

Requires scripts/prep_fineweb_edu_bytes.py to have already produced
datasets/fineweb_edu_10BT/{train,val}.bin from the downloaded sample/10BT/*.parquet shards.

steps=50000 * batch_size=128 * seq_len=2056 =~ 13.1B bytes -- a first guess, well under one full
pass over the ~10B-token (much-larger-in-raw-bytes) corpus, NOT yet validated against real TPU
throughput. Watch the first run's actual it/s (tail -f the run.log) and retune --steps from there,
per CLAUDE.md's "long runs have shown unpredictable throughput" caution.

    uv run python -m qcute.bytelm_fineweb --config configs/bytelm_fineweb/sd_fineweb10b_bytes.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/sd_fineweb10b_bytes
"""
from pathlib import Path

preset = "sd"
train_data = Path("datasets/fineweb_edu_10BT/train.bin")
val_data = Path("datasets/fineweb_edu_10BT/val.bin")
steps = 50000
batch_size = 128
lr_peak = 4e-4
warmup_steps = 300
cosine_decay = True
constant_steps = 1000
grad_clip = 1.0
weight_decay = 0.1
log_every = 50
eval_every = 1000
eval_batches = 32
save_every_n_evals = 1
