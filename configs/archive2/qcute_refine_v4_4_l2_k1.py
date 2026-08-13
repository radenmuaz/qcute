"""v4.4 packed-sequence decode: n_levels=2, Ks=(1,1), decode_pack_mode="interleave", chunked
attention. Non-degenerate counterpart to qcute_refine_v4_4_l1_k1.py -- level 0's decode is
genuinely conditioned on level 1's own encode-pass code (RefineLM._run: source_c = c_list[1],
decode_K = Ks[0]*Ks[1] = 1), not self-referential.

context_len=512, same rationale as qcute_refine_v4_4_l1_k1.py (multiple of attn_window=32, kept
below the production 1024 for this first v4.4 training run).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_l2_k1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_l2_k1/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 2
context_len = 512
attn_window = (32, 32)
decode_pack_mode = "interleave"
decode_chunked = True

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
