"""2x2 grid cell (v4.4 x 1-level) twin of qcute_refine_v4_4_nosharegrid_k4_1.py -- see that file's
docstring for full rationale. Degenerate Ks=(1,), share_level_weights=False gives exactly 2
independent LMs here (encode + decode, no cross-level sharing question since there's only 1 level).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_nosharegrid_k1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_nosharegrid_k1/run.log
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (32,)
decode_pack_mode = "interleave"
decode_chunked = True
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
