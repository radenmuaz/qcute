"""New grid: single-level Ks=(4,) -- "ae cond every 4 bytes" (one code per 4-byte block, no
cascade/higher level at all, unlike the k4_1 cells) -- isolates whether the no-sharing
architecture's behavior on 4-byte block grouping specifically depends on having a second
(cascade) level, or holds the same with just one level. share_level_weights=False,
cross_track_source="decode" (no effect here -- no cross track exists at n_levels=1, kept for
consistency), decode_code_ste=False (detach), use_gumbel_noise=True + gumbel_tau=2.0 -- the
gumbel-noise-ON half of this new 2-config pair (see qcute_refine_v4_4_nosharegrid_nogumbel_
k4single.py for the OFF half). attn_window=((8,256),) matches the established bpelike_1level_k4
convention (narrow encode window, wide decode window). decode_chunked=False (decode_K=4 != 1, the
chunked path is decode_K==1-only).

Companion configs:
  - qcute_refine_v4_4_nosharegrid_nogumbel_k4single.py (v4.4, no gumbel noise)
  - qcute_refine_v4_5_nosharegrid_k4single.py           (v4.5, gumbel noise)
  - qcute_refine_v4_5_nosharegrid_nogumbel_k4single.py  (v4.5, no gumbel noise)

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_nosharegrid_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_nosharegrid_k4single/run.log
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
