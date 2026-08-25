"""qcute.bytelm_fineweb config: small starter model on FineWeb-Edu sample-10BT, byte-level,
0x00-separated (see scripts/prep_fineweb_edu_bytes.py --use_separator). PRESETS["d512x8_mlp4"]
(d_model=512, n_layers=8, n_heads=8, mlp_mult=4, context=2048, mtp_heads=4, ~33.6M non-embedding
params -- formula (4+3*mlp_mult)*d_model^2*n_layers from bytelm_fineweb.py's own PRESETS comment).

warmup_steps=1000 then constant lr_peak=5e-4 (cosine_decay left off -- lr_at's own behavior is
exactly warmup-then-constant, no separate flag needed for this schedule).

steps sized for 2 epochs over the real train.bin (45,109,723,621 bytes, written with
--use_separator so this count includes the 0x00 document-boundary bytes) at batch_size=128,
seq_len=context+mtp_heads=2052:

    steps_per_epoch = 45,109,723,621 / (128 * 2052) ~= 171,745
    2 epochs         ~= 343,489 steps

batch_size=128 is a first guess (not yet bench-verified for this shape/context on a single TPU
chip) -- carried over from configs/bytelm_fineweb/sd_fineweb10b_bytes.py's own choice for
consistency; watch actual it/s on the first run and retune batch_size/steps from there, per
CLAUDE.md's "long runs have shown unpredictable throughput" caution.

    uv run python -m qcute.bytelm_fineweb --config configs/bytelm_fineweb/small_d512x8_mlp4_2epoch.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/small_d512x8_mlp4_2epoch
"""
from pathlib import Path

preset = "d512x8_mlp4"
train_data = Path("datasets/fineweb_edu_10BT/train.bin")
val_data = Path("datasets/fineweb_edu_10BT/val.bin")
steps = 343489
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
