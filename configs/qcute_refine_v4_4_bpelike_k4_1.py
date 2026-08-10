"""v4.4 packed-sequence decode, Ks=(4,1): level 0 groups raw bytes into fixed-size 4-byte blocks
(one code per 4 bytes) -- a crude, non-learned analogue of BPE's ~4-bytes/token average (see
qcute.bpelm for the actual learned-merge BPE baseline; this is NOT claiming equivalent
segmentation, just a matched code-to-byte ratio for comparison). Level 1 processes that
code sequence at block size 1 (no further compression), so level 0's decode is genuinely
conditioned on level 1's own code -- decode_K = Ks[0]*Ks[1] = 4 (each decode prefix token
covers 4 raw bytes, NOT the decode_K==1 case v4.4's chunked/windowed decode path was built for;
this run exercises the newly-generalized _packed_decode_forward "block-interleave" path, dense
attention only -- decode_chunked is left False since chunking was only derived/verified for
decode_K==1).

attn_window=(8, 256): level 0's raw-byte window is 8 (must divide context_len -- deliberately
narrow, 2 code-blocks' worth of raw bytes); level 1's window is 256, larger than its own sequence
length (context_len // Ks[0] == 64 when context_len=256 -- see docs/status.md's mid-session
context_len=1024->256 fix, dense-decode-was-O((2L)^2) note), i.e. effectively full/dense attention
over the whole code sequence (window >= L falls back to dense with a printed warning -- expected,
not a bug; kept as literal 256 rather than rewritten to 64 to preserve "unambiguously larger than
any L this run will ever see" as the intent, not a number tied to the current context_len).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_k4_1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_k4_1/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False  # decode_K=4 != 1 -- chunked path only implemented/verified for decode_K==1

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
