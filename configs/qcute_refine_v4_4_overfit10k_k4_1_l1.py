"""Tiny-overfit sanity check, per user request ("try to overfit and generate plausible text from
train"). Same architecture family as configs/qcute_refine_v4_4_bpelike_k4_1_crosstrack_decode.py
(Ks=(4,1), cross_track_source="decode", decode_code_ste=False -- the headline positive result from
this session's entropy investigation) but on a MUCH smaller/cheaper slice: n_bytes=10000 (vs the
full ~900k-byte corpus), steps=1000 (vs 4000), n_layers=1 (vs 2 -- this is the 1-layer half of the
1-layer/2-layer restated pair).

Goal: with this little data and this few steps, the model should be able to nearly memorize the
train split. If qualitative generation from a train-set prompt still comes out as word-salad here,
that's strong evidence the collapse/incoherence problem is architectural, not just an
undertrained-at-scale artifact (the interpretation offered for every full-scale run's poor
generation this session). If it DOES produce plausible-looking (even if memorized) text, that
narrows the full-scale problem down to training budget/schedule instead.

Companion configs (same n_bytes=10000, steps=1000, cross_track_source="decode",
decode_code_ste=False throughout):
  - qcute_refine_v4_4_overfit10k_k4_1_l2.py   (this config's n_layers=2 twin)
  - qcute_refine_v4_4_overfit10k_k1_k1_l1.py / _l2.py   (degenerate Ks=(1,1), no compression)
  - qcute_refine_v4_4_overfit10k_k1_l1.py / _l2.py      (degenerate 1-level Ks=(1,))
  - configs/bytelm_overfit10k_l1.py / _l2.py  (qcute.bytelm baseline, same n_bytes/steps/depth)

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_overfit10k_k4_1_l1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_overfit10k_k4_1_l1/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False
cross_track_source = "decode"
decode_code_ste = False

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
