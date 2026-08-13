"""2x2 grid cell (v4.4 x 2-level), per user request: share_level_weights=False (every level gets
its own independent encode LM and its own independent decode LM -- see qcute_refine_v4_4.py's
Config.share_level_weights docstring), combined with the session's other established "good
settings": use_gumbel_noise=True + gumbel_tau=2.0 (codebook-collapse mitigation, see
bpelike_k4_1_gumbelfix.py), cross_track_source="decode" + decode_code_ste=False (the headline
positive result from the crosstrack_decode ablation). Same overfit10k testbed as the rest of this
session's batch (n_bytes=10000, steps=1000) per the standing "10k slice until bytelm parity"
methodology (see CLAUDE.md / docs/status.md).

Companion grid cells:
  - qcute_refine_v4_4_nosharegrid_k1.py    (v4.4, 1 level)
  - qcute_refine_v4_5_nosharegrid_k4_1.py  (v4.5, 2 level)
  - qcute_refine_v4_5_nosharegrid_k1.py    (v4.5, 1 level)

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_nosharegrid_k4_1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_nosharegrid_k4_1/run.log
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
