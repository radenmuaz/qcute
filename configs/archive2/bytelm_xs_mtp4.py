"""qcute.bytelm config: reproduces the matched-bandwidth (4 bytes/timestep)
byte+MTP baseline on the standard 1M-byte corpus.

Prior result (500K-byte enwik8_tiny.gz, since removed and replaced by the
1M corpus below): best val_bpb 2.52 at step 1300, then mild overfitting drift
by step 2000 (see docs/status.md's comparison table). Needs rerunning on
the 1M corpus to get current numbers — see logs/bytelm_xs_mtp4/bpb.png.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs_mtp4

The config's filename stem (bytelm_xs_mtp4) becomes the run_name
by default, so logs/checkpoints land in the paths above without needing
--run_name. Pass --steps N to extend past 2000 and see more of the
overfitting curve; --run_name something_else to avoid appending into this
run's existing log (see docs/scaffolding_playbook.md §8b for why appending
into the same run_name is handled gracefully by plot_run.py, not something
to avoid on principle).
"""
from pathlib import Path

preset = "xs"  # mtp_heads=4 by default for this preset — see qcute/bytelm.py's PRESETS comment
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 2000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
