"""qcute_v5_stack_3: Ks=(1,) single-level (n_levels=1, no hierarchy -- self-decode only),
context_len=256, attn_window=(256,) (dense/unbounded) -- skip (qcute.qcute_v5_stack) counterpart
of configs/qcute_v5_fixblock_3.py: same self-attention-folded self track and block-0 exclusion as
fixblock, plus _skip's own buffer pruning (a block's raw bytes are dropped from the self-attention
buffer once that block's code exists). K=1 makes every block a single byte, so this config exercises
_skip's pruning maximally -- every raw byte becomes invisible to everything except its own tied
code's row, roughly halving the effective buffer length vs fixblock.

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_3.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_3
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (256,)
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
