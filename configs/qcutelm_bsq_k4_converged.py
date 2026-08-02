"""qcute.qcutelm config: reproduces the matched-bandwidth (K=4) BSQ baseline
on the tiny corpus, **loosely-coupled architecture**.

ARCHITECTURALLY STALE AS OF the tightly-coupled BSQ change: `--bottleneck
bsq` now always trains the tightly-coupled path by default (see
qcute/qcutelm.py's `_forward_bsq_tightly_coupled` and this config's sibling,
configs/qcutelm_bsq_k4_tightlycoupled.py) — re-running *this* config will
NOT reproduce the numbers below anymore, because there's no longer a way to
select the old fully-independent-losses BSQ behavior via CLI (only FSQ still
works that way). Kept as a historical record of what that architecture did.

qcutelm converges much more slowly in step-count than bytelm/bpelm (though
~10x faster per step — chunk-local MLP encoder/decoder vs. a full causal
transformer pass) — 2000 steps isn't enough to see it peak. Our reference
run was stopped early at step ~6300 (still hadn't clearly turned over):
best val_bpb 5.54 at step 5400, val_bpb ~5.56 at the stopping point (see
docs/status.md's comparison table and logs/qcutelm_bsq_k4_converged/bpb.png).
steps=10000 below is what that run was actually launched with — let it run
to completion (only a few minutes) for the full curve.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_tiny.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_converged.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_converged

The config's filename stem (qcutelm_bsq_k4_converged) becomes the run_name
by default.
"""
from pathlib import Path

bottleneck = "bsq"
K = 4  # not the handover doc's K=8 — see qcute/qcutelm.py's Config.K comment
data = Path("datasets/enwik8_tiny.gz")
val_frac = 0.1
steps = 10000
batch_size = 16
seq_chunks = 32
warmup_steps = 200
lr_peak = 6e-4
log_every = 100
eval_every = 200
eval_batches = 20
