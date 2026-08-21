"""qcute.bytelm_tpu config: PIPELINE SANITY CHECK — tiny preset (d_model=128, n_layers=4,
n_heads=4, context=512, ~0.9M params) on the FULL enwik8 corpus (datasets/enwik8.gz,
100,000,000 bytes), not the 10k-byte overfit slice.

Purpose is different from configs/bytelm_tpu/bytelm_tpu_overfit10k.py: that one checks the model
can memorize a tiny slice; this one checks the full-data pipeline itself (load_enwik8 on the real
100MB corpus, batch_iter, eval, checkpointing, device placement) runs end-to-end at real scale
without blowing up. Not meant to reach a good bpb (0.9M params is far too small for that) — just
to confirm nothing about running at full-enwik8 scale is broken before committing to the long
configs/bytelm_tpu/bytelm_tpu_sd_full_enwik8.py run.

mtp_heads=1: MTP disabled (plain next-byte prediction only, no lookahead heads) — keeps this a
clean pipeline/throughput check, not entangled with MTP-specific behavior.

5 full epochs over the ~90M-byte train split (val_frac=0.05, test_frac=0.05 leaves 90% for
train): seq_len = context(512) + mtp_heads(1) = 513, so steps_per_epoch = 90,000,000 /
(batch_size(128) * 513) =~ 1370.6 -> steps = round(1370.6 * 5) = 6850.

warmup_steps=70 =~ 5% of one epoch's steps (0.05 * 1370.6 =~ 68.5, rounded to 70) — a long linear
warmup relative to this tiny model, then held constant (cosine_decay=False) for the rest of
training, per the "long warmup + constant LR" schedule requested for this sanity run specifically
(the sd full-scale config uses warmup -> constant -> cosine instead).

Measured previously at the old (0.83-epoch, mtp_heads=4) step count: ~2.5-3s/it on CPU (Apple
Silicon), ~30-35 it/s on a v6e-1 TPU chip once past initial XLA compile. At 6850 steps that's
roughly 3.5-5.5 hours on CPU (no longer "well under an hour" — this config is now sized for a
TPU run) vs. ~3-4min on TPU. Run on a TPU VM; CPU is fine for a quick partial-step syntax check
only, not a full run at this size.

    # on a TPU VM (intended target):
    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_tiny_full_enwik8.py --device xla

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_tiny_full_enwik8
"""
from pathlib import Path

preset = "tiny"
data = Path("datasets/enwik8.gz")   # full 100,000,000-byte corpus
val_frac = 0.05
test_frac = 0.05                    # exercises the val/test split + final_test_bpb path too
mtp_heads = 1                       # MTP disabled — plain next-byte prediction only
steps = 6850                        # 5 epochs over the ~90M-byte train split, see docstring
batch_size = 128
lr_peak = 6e-4
warmup_steps = 70                   # ~5% of one epoch, long relative to this tiny model
cosine_decay = False                # constant LR after warmup, no decay
grad_clip = 1.0
log_every = 100
eval_every = 500
eval_batches = 20
save_every_n_evals = 1
