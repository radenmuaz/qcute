"""qcute.qcute_refine_v4_2 config: `code_head_mode="word"`, `word_bits=4`
(new this session — BitPredictHeadWordPredict).

Session: "design another head, wordpredict, which decompose to word like
8 bit, 4 bit, useful for dq more than 8... implement until done complete
with ar gen and config to queue, try word size 4." Decomposes the dq=8
code into `n_words=dq//word_bits=2` WORDS, each a genuine 16-way softmax
classifier (no position-sharing bottleneck within a word, unlike attn/
conv/ssm's own per-bit-position bottleneck) — word 1's classifier
additionally conditions on word 0's own embedding via concatenation
(session: "past chain prob conditioning make simpler but more expensive"
— plain concat, no recurrence/attention machinery). `word_embed_downsample`
left at its default (1, `d_embed=d_model=256`).

Verified this session: fixed/loop consistency (exact/near-exact match)
across word_bits in {8,4,2,1}, gradients, the word_bits==dq degenerate
case matching a plain dense softmax exactly, causality (later words don't
leak into earlier ones' logits), full-model forward+backward, and
`validate_generation` parity (`generate_no_cache` vs `generate_kv_cache`
exact match).

`code_embed_mode="pq_table"` carried over — same fix that resolved
`attn_id4`'s original divergence.

Everything else identical to this session's other `k32_narrow` chain-head
configs: Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_word4.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_word4
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
word_bits = 4
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
