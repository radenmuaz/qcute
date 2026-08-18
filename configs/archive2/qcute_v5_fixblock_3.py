"""qcute_v5_fixblock_3: Ks=(1,) single-level (n_levels=1, no hierarchy -- self-decode only),
context_len=256, attn_window=(256,) (dense/unbounded) -- fixblock (qcute.qcute_v5_fixblock)
counterpart of configs/qcute_v5_3.py: self track folded into self-attention (no qfb
decode_boundary_query pass), block 0's first K-1 targets excluded from decode's own loss
(empty range here since K=1 -- this config exercises the architecture change, not the exclusion).

uv run python -m qcute.qcute_v5_fixblock --config configs/qcute_v5_fixblock_3.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_fixblock_3
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (256,)
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

qual_gen_bytes = 128
qual_prompt_bytes = 64
