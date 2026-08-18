"""qcute_v5_concat_modes_ks1: Ks=(1,) smoke config for multi_mode_impl="single_pass" -- degenerate
single-track case (T=1 at the only level), so decode_stage_extra_total should stay exactly 0 (no
shallower mode exists below the single self track). Sanity floor of the Ks regression grid
(CLAUDE.md's simplest-to-hardest table, #1).

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_modes_ks1.py
"""
from pathlib import Path

Ks = (1,)
d_model = 64
n_layers = 1
context_len = 64
attn_window = -1
multi_mode_impl = "single_pass"

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 50
batch_size = 8
lr_peak = 6e-4
warmup_steps = 10
cosine_decay = False
log_every = 10
eval_every = 25
eval_batches = 5

qual_gen_bytes = 16
qual_prompt_bytes = 16
