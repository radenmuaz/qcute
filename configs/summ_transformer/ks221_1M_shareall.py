"""summ_transformer/ks221_1M_shareall: same as ks221_1M.py but with every weight-sharing flag on --
share_lm (lms[s] all alias lms[0]), share_fuse (fuse_stages[s] all alias fuse_stages[0]), and
weight_tie (head.weight refs embed.weight). Companion to ks21_1M_shareall.py for the 3-level case.

uv run python -m qcute.summ_transformer.summ_transformer --config configs/summ_transformer/ks221_1M_shareall.py

# plot after training:
uv run python scripts/plot_run.py logs/summ_transformer_ks221_1M_shareall
"""
from pathlib import Path

run_name = "summ_transformer_ks221_1M_shareall"
Ks = (2, 2, 1)
d_model = 256
n_layers = 4
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8
mtp_heads = 4
share_lm = True
share_fuse = True
weight_tie = True

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
