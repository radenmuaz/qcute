"""n_layers=2 twin of configs/qcute_refine_v4_5_overfit10k_k1_k1_l1.py -- see that file's
docstring for full rationale. Only change here: n_layers 1 -> 2.

    uv run python -m qcute.qcute_refine_v4_5 --config configs/qcute_refine_v4_5_overfit10k_k1_k1_l2.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_overfit10k_k1_k1_l2/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = 32
cross_track_source = "decode"
decode_code_ste = False

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
qual_prompt_bytes = 64
