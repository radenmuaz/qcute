"""summformer/ks21_1M: full-scale (full enwik8_1M, n_bytes=None) run of the simplest
2-level hierarchical-summarization case, following the overfit10k sanity check in
ks21_overfit10k.py. Same architecture/hyperparameter convention as qcute_zero's own full-scale
configs (configs/qcute_zero/ks221_1M.py): d_model=256/n_layers=4/context_len=256, attn_window=None
(unbounded -- no -1 sentinel exists in this codebase's causal_mask, which treats window as a
literal `< window` comparison, so "unbounded" means None, not -1).

uv run python -m qcute.summformer.summformer --config configs/summformer/ks21_1M.py

# plot after training:
uv run python scripts/plot_run.py logs/summformer_ks21_1M
"""
from pathlib import Path

run_name = "summformer_ks21_1M"
Ks = (2,)
d_model = 256
n_layers = 4
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8
mtp_heads = 4

data = Path("datasets/enwik8_1M.gz")
n_bytes = None
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
log_every = 200
eval_every = 500
eval_batches = 20
