"""Smoke config: Ks=(4,1), finite attn_window=8 (forces the chunked/banded path), multi_mode_impl=
"single_pass" -- exercises _merged_decode_forward_multimode_chunked end-to-end through real
training, not just isolated forward() calls. Not meant to be trained to convergence.

uv run python -m qcute.qcute_v5_concat_modes --config configs/qcute_v5_concat_modes_chunked_smoke.py
"""
from pathlib import Path

Ks = (4, 1)
d_model = 32
n_layers = 1
context_len = 32
attn_window = 8
multi_mode_impl = "single_pass"

data = Path("datasets/enwik8_1M.gz")
n_bytes = 5000
val_frac = 0.1

steps = 10
batch_size = 8
lr_peak = 6e-4
warmup_steps = 2
cosine_decay = False
log_every = 5
eval_every = 10
eval_batches = 3

qual_gen_bytes = 16
qual_prompt_bytes = 16
