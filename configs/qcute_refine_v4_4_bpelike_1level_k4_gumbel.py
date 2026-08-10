"""Gumbel-noise counterpart to configs/qcute_refine_v4_4_bpelike_1level_k4.py (the config
bpelike_1level_k4_retry actually used) -- Ks=(4,), n_layers=1, same self-conditioning design, but
use_gumbel_noise=True, gumbel_tau=2.0 (same values as the other two grid cells, for consistency),
and context_len=256 instead of the original 1024 (matching this session's later context_len=1024
-> 256 fix for the dense-decode O((2L)^2) cost -- see docs/status.md; bpelike_1level_k4_retry
itself predates that fix and ran at the slower 1024 setting, but there's no reason to repeat that
cost here since it doesn't change what's being tested).

Third and last cell of the use_gumbel_noise grid, per explicit user request ("repeat with true
later" / "grid") -- every base architecture tested this session with use_gumbel_noise=False (the
Config default, and what EVERY prior config used) now has a matching True counterpart:

    Ks=(4,)   1level:      bpelike_1level_k4_retry (False) / bpelike_1level_k4_gumbel (True, this)
    Ks=(4,1):              bpelike_k4_1 (False)             / bpelike_k4_1_gumbelfix (True, queued)
    Ks=(1,1):              k1_k1_w32 (False)                 / k1_k1_w32_gumbel (True, queued)

bpelike_1level_k4_retry's own entropy probe result (2.62 bits/256, 30/256 active -- collapsed, but
less severely than the Ks=(4,1) variants) makes this cell the most direct single-variable test of
whether gumbel noise alone measurably raises code_0 entropy for Ks[0]=4 block-grouping, without
the cross-level/self-only architecture question also in play.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_1level_k4_gumbel.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_1level_k4_gumbel/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = ((8, 256),)
decode_pack_mode = "interleave"
decode_chunked = False
use_gumbel_noise = True
gumbel_tau = 2.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20

qual_gen_bytes = 64
qual_prompt_bytes = 64
