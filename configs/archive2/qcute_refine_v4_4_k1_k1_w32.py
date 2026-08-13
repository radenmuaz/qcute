"""Degenerate Ks=(1,1): neither level compresses at all -- level0 emits one code per raw byte,
level1 emits one code per level0 code, so BOTH levels run at the same full byte-rate sequence
length (no downsampling anywhere in the tower). decode_K=1 for both the self and cross track
(cum_K = Ks[0] = 1 for self, Ks[0]*Ks[1] = 1 for cross) -- the "cheap" decode_K==1 case v4.4's
original chunked decode path was built around, but here exercised as a genuine 2-track
(self+cross) case, which decode_chunked's single-track-only implementation still can't take
(left False, dense reference -- decode_banded is correctness-verified for this shape too, per
scripts/test_v4_4_banded_decode.py, but its own timing hasn't been cleanly re-benchmarked yet
post-training-contention, so not defaulted on here).

attn_window=32 (scalar): broadcasts to EVERY level's encode window AND every decode source's
window (self and cross, for level0; self only, for level1) -- the simplest possible uniform
window choice, no per-track tuning, useful as a clean baseline/sanity point since Ks=(1,1) has no
compression to reason about at all, just plain windowed self+cross attention over two
byte-rate-length streams.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_k1_k1_w32.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_k1_k1_w32/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = 32
decode_pack_mode = "interleave"
decode_chunked = False  # multi-track (self+cross) -- chunked path is single-track K==1 only

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
