"""qcute.qcutelm config: reproduces the tightly-coupled BSQ architecture
(encoder -> z_t -> LM -> raw latent -> bsq_quantize -> decoder -> bytes_{t+1},
graded against the true next bytes) at K=4, auxiliary loss disabled.

Result when this was run: plateaus at val_bpb ~8.0-8.5 after a brief dip to
~7.1 near step 400 — substantially worse than the loosely-coupled
architecture's 5.54 (configs/qcutelm_bsq_k4_converged.py). Train and val bpb
track closely together the whole run (no growing gap) — this is *not*
overfitting, it looks stuck in a poor local optimum instead. See
docs/status.md and docs/architecture.md's qcutelm section for the full story.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_tiny.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_tightlycoupled.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_tightlycoupled

Drop `disable_aux_recon = True` (or flip it to False) to try the auxiliary
encoder-latent->decoder loss enabled — whether that recovers the gap to the
loosely-coupled architecture is an open, not-yet-run experiment.
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
disable_aux_recon = True
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
