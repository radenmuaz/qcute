"""qcute_refine_v4_4_1_nosharegrid_k1.py twin: n_layers=2 -> 1, decode_code_ste=False -> True.
Motivation: with decode_code_ste=False (the grid default), the code-producing classify head gets
ZERO gradient for n_levels=1 configs like this one (verified directly: code_head.weight.grad is
None after backward) -- nothing else differentiably consumes the code. decode_code_ste=True fixes
this, and is now well-posed since the LM-continuation fix means code_b conditions block b+1 (a
proper predictive-coding signal), not block b's own reconstruction. See docs/status.md.

    uv run python -m qcute.qcute_refine_v4_4_1 --config configs/qcute_refine_v4_4_1_ste_k1_l1.py
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
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = True
gumbel_tau = 2.0

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
