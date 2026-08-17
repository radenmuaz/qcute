"""
uv run python -m qcute.archive3.qcute_v5_bos --config configs/qcute_v5_bos_1.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_bos_1
"""
from pathlib import Path

Ks = (4,1)
d_model = 256
n_layers = 1
context_len = 1024
attn_window = (16,256)
use_gumbel_noise = False
# gumbel_tau = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
# eval_every = 100
eval_batches = 20

qual_gen_bytes = 256
qual_prompt_bytes = 64
