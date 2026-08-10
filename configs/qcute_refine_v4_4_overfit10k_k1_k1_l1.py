"""Tiny-overfit sanity check, degenerate architecture: Ks=(1,1) (no compression at all, plain
windowed self+cross attention over two byte-rate-length streams -- same shape as
configs/qcute_refine_v4_4_k1_k1_w32.py) at the same tiny scale as
configs/qcute_refine_v4_4_overfit10k_k4_1_l1.py (n_bytes=10000, steps=1000). See that file's
docstring for the full overfit-sanity-check rationale; this config isolates whether the Ks[0]=4
block-grouping (this session's identified primary driver of code_0 collapse) is itself needed for
plausible-looking generation to fail at this tiny scale, or whether the no-compression Ks=(1,1)
case does any better.

cross_track_source="decode", decode_code_ste=False (same as every config in this batch).
attn_window=32 scalar, matching k1_k1_w32's own uniform-window choice (no per-track tuning needed
since there's no compression to reason about).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_overfit10k_k1_k1_l1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_overfit10k_k1_k1_l1/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = 32
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
