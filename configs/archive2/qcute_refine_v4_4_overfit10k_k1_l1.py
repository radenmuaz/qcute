"""Tiny-overfit sanity check, degenerate 1-level architecture: Ks=(1,) (no code compression, no
higher level at all -- same shape as configs/qcute_refine_v4_4_l1_k1.py) at the same tiny scale as
configs/qcute_refine_v4_4_overfit10k_k4_1_l1.py (n_bytes=10000, steps=1000). See that file's
docstring for the full overfit-sanity-check rationale.

cross_track_source has no effect here (n_levels==1, no cross track exists) but decode_code_ste=
False is still set for consistency with the rest of this batch (governs level0's SELF track
detach, per user instruction to apply the same detach principle throughout). decode_chunked=True,
matching l1_k1.py's own established fast/correct single-track path for this degenerate K==1 case.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_overfit10k_k1_l1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_overfit10k_k1_l1/run.log
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,)
decode_pack_mode = "interleave"
decode_chunked = True
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
