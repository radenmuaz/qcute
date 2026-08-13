"""v4_5_1 (LM-continuation self-code decode) x k4_1 x gumbel=on grid
cell, share_level_weights=False (no weight sharing, per explicit user request). See
qcute/qcute_refine_v4_5_1.py's module docstring and docs/status.md's "Decode redesign: self-code LM
continuation" section for the design. cross_track_source="decode", decode_code_ste=False (detach).

    uv run python -m qcute.qcute_refine_v4_5_1 --config configs/qcute_refine_v4_5_1_nosharegrid_k4_1.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_1_nosharegrid_k4_1/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False
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
