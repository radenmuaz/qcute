"""qcute_refine_v4_4_1 twin of qcute_refine_v4_4_nosharegrid_nogumbel_k4single.py -- v4.4's single
best config all session (exact verbatim memorization on 3/4 train prompts, fluent held-out
generation), now with path-2 self-code autoencoder decode instead of previous-block-code decode.
See qcute/qcute_refine_v4_4_1.py's module docstring and docs/status.md's "Decode redesign:
self-code autoencoder" section for the full design rationale.

    uv run python -m qcute.qcute_refine_v4_4_1 --config configs/qcute_refine_v4_4_1_nosharegrid_nogumbel_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_1_nosharegrid_nogumbel_k4single/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
decode_pack_mode = "interleave"
decode_chunked = False
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

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
