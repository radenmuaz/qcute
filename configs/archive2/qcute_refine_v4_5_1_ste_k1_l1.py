"""qcute_refine_v4_5_1_nosharegrid_k1.py twin: n_layers=2 -> 1, decode_code_ste=False -> True.
See qcute_refine_v4_4_1_ste_k1_l1.py's docstring for the full rationale (same fix, same file pair).

    uv run python -m qcute.qcute_refine_v4_5_1 --config configs/qcute_refine_v4_5_1_ste_k1_l1.py
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,)
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = True
gumbel_tau = 2.0

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
