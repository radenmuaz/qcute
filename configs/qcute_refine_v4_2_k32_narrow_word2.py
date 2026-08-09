"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_word4.py, ONE change: `word_bits=2` instead of 4.

Session: "and word size 2." `n_words=dq//word_bits=4` WORDS, each a
genuine 4-way softmax classifier — a longer chain of cheaper
classifiers than `word4`'s 2-word/16-way version, the opposite end of
the word_bits dial from `word4` at the same dq=8. Direct comparison
point for how the chain-length-vs-per-step-cost tradeoff plays out
empirically (more, cheaper steps vs. fewer, pricier ones).

Everything else identical to word4.py: Ks=(32,32), dq=8, d_model=256,
n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, code_head_mode="word",
code_embed_mode="pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_word2.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_word2
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
code_head_mode = "word"
word_bits = 2
code_embed_mode = "pq_table"

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
