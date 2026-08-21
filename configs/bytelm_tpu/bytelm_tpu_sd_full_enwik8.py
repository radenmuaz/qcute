"""qcute.bytelm_tpu config: full-enwik8 target run — sd preset (d_model=1024, n_layers=8,
n_heads=16, context=2048, mtp_heads=8, ~101M params), aimed at sub-1.0 bpb within a 12h budget
on a single TPU chip (see qcute/bytelm_tpu.py's own docstring for the FLOPs-vs-data-budget
reasoning: full enwik8's ~95M-byte train split, not raw compute, is expected to be the binding
constraint at this model size).

steps=8000 * batch_size=128 * seq_len=2056 =~ 2.1B tokens =~ 22 epochs over the ~95M-byte train
split — a first guess, NOT yet validated against real TPU torch_xla throughput. Watch the first
run's actual it/s (tail -f the run.log) and retune --steps for the 12h budget from there, per
CLAUDE.md's "long runs have shown unpredictable throughput" caution — this config's own step
count may need to go up (if 12h leaves headroom) or down (if it/s comes in low).

Before this: confirm the module itself works via
configs/bytelm_tpu/bytelm_tpu_overfit10k.py first.

    uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sd_full_enwik8.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_sd_full_enwik8
"""
from pathlib import Path

preset = "sd"
data = Path("datasets/enwik8.gz")   # full 100,000,000-byte corpus, not the 1M dev slice
val_frac = 0.05                     # ~5M held-out bytes, drives LR-free model selection (checkpointer)
test_frac = 0.05                    # ~5M bytes, standard enwik8-style 90/5/5 split — scored once at the
                                     # end (final_test_bpb) from the val-selected best checkpoint, never
                                     # touched during training/checkpointing itself
steps = 8000
batch_size = 128
lr_peak = 4e-4
warmup_steps = 300
cosine_decay = True
constant_steps = 500
grad_clip = 1.0
weight_decay = 0.1
zero_kv_sink = False                # new option, default off — flip on for an A/B comparison
log_every = 50
eval_every = 500
eval_batches = 32
save_every_n_evals = 1
