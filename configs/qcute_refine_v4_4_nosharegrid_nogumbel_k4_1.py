"""2x2 grid cell (v4.4 x 2-level), no-gumbel-noise variant of qcute_refine_v4_4_nosharegrid_k4_1.py
-- same share_level_weights=False + cross_track_source="decode" + decode_code_ste=False (detach,
UNCHANGED) setup, but per explicit user request: use_gumbel_noise=False, gumbel_tau back to
default 1.0 (i.e. drop the gumbel-noise codebook-collapse mitigation, keep everything else).
Isolates whether the no-sharing architecture's own behavior depends on gumbel noise specifically,
independent of the (unchanged) detach choice.

Companion grid cells:
  - qcute_refine_v4_4_nosharegrid_nogumbel_k1.py    (v4.4, 1 level)
  - qcute_refine_v4_5_nosharegrid_nogumbel_k4_1.py  (v4.5, 2 level)
  - qcute_refine_v4_5_nosharegrid_nogumbel_k1.py    (v4.5, 1 level)
Gumbel-noise counterpart: qcute_refine_v4_4_nosharegrid_k4_1.py

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_nosharegrid_nogumbel_k4_1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_nosharegrid_nogumbel_k4_1/run.log
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
