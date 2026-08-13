"""n_levels=1, Ks=(1,). No level above level 0, so decode falls back to the degenerate
self-conditioning case (session: "in extreme degenerate case, actually you can set 1 level, in
decode training, just use query vector to get ids to condition itself during decode") -- level 0
extracts its own code_pool code c_0 from its own just-finished encode-pass k,v, then decodes a
second time conditioned on c_0's own embedded ids.

Ks=(1,) means the code_pool block size is 1 -- a code is extracted at EVERY single byte position
(n_blocks = context_len), the densest possible conditioning signal decode can get ("always use
query every timestep to decode"). Since a length-1 block has nothing to pool over, code_pool's
query is inert here (softmax over one key is always weight 1) -- this isolates whether decode can
exploit ANY self-derived per-position signal at maximal density, independent of code_pool's own
usefulness as an actual poolinq/summarization mechanism (see qcute_refine_v4_3_l2_k1.py for the
non-degenerate, real level-1-stubbed counterpart at the same K=1 density).

    uv run python -m qcute.qcute_refine_v4_3 --config configs/qcute_refine_v4_3_l1_k1.py

    # watch live:
    tail -f logs/qcute_refine_v4_3_l1_k1/run.log
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 2
context_len = 1024
attn_window = (32,)

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
