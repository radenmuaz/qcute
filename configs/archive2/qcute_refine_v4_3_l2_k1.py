"""n_levels=2, Ks=(1,1). K=1 at both levels removes all compression as a confound -- level 0
extracts a code at every byte, level 1 (processing that same-length sequence) also extracts a
code at every one of its own positions, so decode's conditioning source c_1 (level 1's own real,
non-degenerate stub -- see qcute_refine_v4_3_l1_k1.py for the 1-level self-conditioning
counterpart) is available at maximal density throughout. Session: "stress test best possible
given some window" -- isolates whether the fixed attention window (32) itself is the bottleneck
on what decode can exploit, independent of any information loss from code-block downsampling.

    uv run python -m qcute.qcute_refine_v4_3 --config configs/qcute_refine_v4_3_l2_k1.py

    # watch live:
    tail -f logs/qcute_refine_v4_3_l2_k1/run.log
"""
from pathlib import Path

Ks = (1, 1)
d_model = 256
n_layers = 2
context_len = 1024
attn_window = (32, 32)

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
