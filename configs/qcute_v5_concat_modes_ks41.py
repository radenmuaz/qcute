"""qcute_v5_concat_modes_ks41: Ks=(4,1) smoke config for multi_mode_impl="single_pass" -- level0
has T=2 tracks (self K=4, +1 K=4 since Ks[1]=1 doesn't widen the span), so decode_stage_extra_total
picks up exactly one shallower mode (self-only) at level0; level1 stays T=1 (no extra mode).
Ks regression grid (CLAUDE.md) #7.

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_modes_ks41.py
"""
from pathlib import Path

Ks = (4, 1)
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
