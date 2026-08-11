"""v4.5 counterpart of qcute_refine_v4_4_nosharegrid_k4single.py -- see that file's docstring for
full rationale (single-level Ks=(4,), share_level_weights=False, gumbel noise ON). Under v4.5's
staged design with n_levels=1, share_level_weights=False gives exactly 2 independent LMs (encode,
reused for decode's Stage 0 + one independent cross-attn-stage LM for the single self-code
track) -- decode_stage_extra_total is always zero here (no intermediate stages possible).

    uv run python -m qcute.qcute_refine_v4_5 --config configs/qcute_refine_v4_5_nosharegrid_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_nosharegrid_k4single/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
cross_track_source = "decode"
decode_code_ste = False
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
