"""v4.4 packed-sequence decode, n_levels=1, Ks=(4,): simpler replacement for
qcute_refine_v4_4_bpelike_k4_1.py's two-level Ks=(4,1) design -- one level is enough to test the
"~4 raw bytes per in-band code, BPE-token-ratio-like" idea (see that file's docstring for the
same BPE caveat: this is a fixed-width grouping, not learned merges). Decode falls back to the
degenerate self-conditioning case (RefineLM._run: n_levels==1, source_c = c_list[0], decode_K =
Ks[0] = 4) -- level 0 conditions its own decode pass on its own just-produced code, one code per
4 raw bytes, via the newly-generalized block-interleave _packed_decode_forward (see
qcute_refine_v4_4_bpelike_k4_1.py's docstring for that mechanism's own design notes).

attn_window is now a per-level (encode_window, decode_window) 2-tuple (Config.attn_window entries
accept either a scalar, applied to both, or an explicit 2-tuple -- see RefineLM.__init__): encode
window 8 (the plain NTP self-attention pass, narrow -- 2 code-blocks' worth of raw bytes), decode
window 256 (the packed encode+code conditioning pass, wide -- most of the 1024-byte context).
decode_chunked left False since decode_K=4 != 1 (chunked decode is decode_K==1-only, see status.md).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_1level_k4.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_1level_k4/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 1
context_len = 1024
attn_window = ((8, 256),)
decode_pack_mode = "interleave"
decode_chunked = False  # decode_K=4 != 1

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
