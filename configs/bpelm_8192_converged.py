"""qcute.bpelm config: reproduces the matched-bandwidth (~4 bytes/timestep)
BPE baseline on the tiny corpus.

Result when this was run: best val_bpb 2.35 at step 300 — the best of all
three baselines at this scale, but also the fastest to overfit (train_bpb
collapses to ~0.02 by step 2000; see docs/status.md's comparison table and
logs/bpelm_8192_converged/bpb.png). Needs a trained tokenizer first.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_tiny.gz missing
    uv run python scripts/train_bpe.py --data datasets/enwik8_tiny.gz --vocab_size 8192
    uv run python -m qcute.bpelm --config configs/bpelm_8192_converged.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_8192_converged

The config's filename stem (bpelm_8192_converged) becomes the run_name by
default. train_bpe.py hard-fails if the tokenizer isn't verified lossless
(see qcute/bpelm.py's bits_per_byte() docstring) — don't retrain with
different sentencepiece flags without re-checking that.
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_tiny_8192.model")
data = Path("datasets/enwik8_tiny.gz")
context = 256
d_model = 256
n_layers = 4
n_heads = 4
val_frac = 0.1
steps = 2000
batch_size = 16
warmup_steps = 200
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
