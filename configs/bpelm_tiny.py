"""qcute.bpelm config: BPE baseline on the tiny dataset.

Third baseline alongside configs/bytelm_xs_tiny_longrun.py (byte+MTP) and
configs/qcutelm_bsq_tiny.py (BSQ) — same tiny dataset, same warmup+constant
LR schedule. Requires the tokenizer first:
    uv run python scripts/train_bpe.py --data datasets/enwik8_tiny.gz

    uv run python -m qcute.bpelm --config configs/bpelm_tiny.py
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_tiny_8192.model")
data = Path("datasets/enwik8_tiny.gz")
context = 256
d_model = 256
n_layers = 4
n_heads = 4
val_frac = 0.1
steps = 5000
batch_size = 16
warmup_steps = 200
lr_peak = 6e-4
log_every = 100
eval_every = 250
eval_batches = 10
