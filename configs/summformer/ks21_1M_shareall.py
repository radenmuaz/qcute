"""summformer/ks21_1M_shareall: same as ks21_1M.py but with every weight-sharing flag on --
share_lm (lms[s] all alias lms[0]), share_fuse (fuse_stages[s] all alias fuse_stages[0]), and
weight_tie (head.weight refs embed.weight). Tests whether tying weights reduces the overfitting
seen in ks21_1M (best val_loss=4.225 at step 1499, degrading to 7.77 by step 8000 despite train
byte_acc reaching 0.95).

uv run python -m qcute.summformer.summformer --config configs/summformer/ks21_1M_shareall.py

# plot after training:
uv run python scripts/plot_run.py logs/summformer_ks21_1M_shareall
"""
from pathlib import Path

run_name = "summformer_ks21_1M_shareall"
Ks = (2,)
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
