"""qcute.bpelm config: reproduces the matched-bandwidth (~4 bytes/timestep)
BPE baseline on the standard 1M-byte corpus.

Prior result (enwik8_1M.gz, steps=2000): best val_bpb 2.3679 (best.pt),
but last.pt badly overfit — train_bpb collapsed to 0.34-0.54 while val_bpb
climbed 3.13->3.24->3.35 over the final 3 evals (see docs/status.md).
steps bumped 2000->8000 here to match bytelm_xs_mtp4_ctx1024.py /
qcutelm_vlt6 grid's budget for a fair wallclock/it-s comparison across all
three baselines/tokenizer-LM at the same step count — best.pt still tracks
the true optimum regardless of how far past it training continues. Needs a
trained tokenizer first.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz --vocab_size 8192
    uv run python -m qcute.bpelm --config configs/bpelm_8192.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_8192

The config's filename stem (bpelm_8192) becomes the run_name by
default. train_bpe.py hard-fails if the tokenizer isn't verified lossless
(see qcute/bpelm.py's bits_per_byte() docstring) — don't retrain with
different sentencepiece flags without re-checking that.
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_8192.model")
data = Path("datasets/enwik8_1M.gz")
context = 256
d_model = 256
n_layers = 4
n_heads = 4
val_frac = 0.1
steps = 8000
batch_size = 16
warmup_steps = 500
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
