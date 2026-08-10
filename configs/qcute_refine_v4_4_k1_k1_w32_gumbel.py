"""Gumbel-noise counterpart to configs/qcute_refine_v4_4_k1_k1_w32.py -- identical (Ks=(1,1),
d_model=256, n_layers=2, context_len=256, attn_window=32) except use_gumbel_noise=True,
gumbel_tau=2.0 (same values as bpelike_k4_1_gumbelfix.py, for a consistent grid).

Part of a systematic grid, per explicit user request ("repeat with true later" / "grid"): every
base architecture tested this session with use_gumbel_noise=False (the Config default, and what
EVERY prior config used, including the past-success l1_k1/l2_k1) now gets a matching
use_gumbel_noise=True counterpart --

    Ks=(4,)   1level:      bpelike_1level_k4_retry (False) / bpelike_1level_k4_gumbel (True, new)
    Ks=(4,1):              bpelike_k4_1 (False)            / bpelike_k4_1_gumbelfix (True, queued)
    Ks=(1,1):              k1_k1_w32 (False)                / k1_k1_w32_gumbel (True, this config)

k1_k1_w32's own entropy probe result matters here specifically because k1_k1_w32/l2_k1 (Ks=(1,1))
already showed HEALTHY code_0 teacher-forced entropy (4.46-6+ bits) even WITHOUT gumbel noise --
so this cell of the grid tests whether gumbel noise still helps (e.g. GENERATION-time collapse,
which was observed even for healthy-entropy checkpoints, see docs/status.md round 2) or whether it
matters only for the already-low-entropy Ks[0]=4 case.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_k1_k1_w32_gumbel.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_k1_k1_w32_gumbel/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = 32
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
