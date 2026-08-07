"""qcute.qcutelm_pyramid config: byte_only diagnostic — isolates whether
v1/v2's ~3.6 val_bpb plateau (vs bytelm's 2.45 at the same step) is caused
by the merge/BSQ mechanism, or simply by attn_window=80's reduced
receptive field (~2*80*4=640 positions vs bytelm's full dense reach over
1024) — a confound raised in session discussion ("both terrible bpb vs
byte and bpe") that v1/v2 alone can't separate, since both mechanisms are
entangled there.

byte_only=True skips ALL merging/insertion — D_0 (this file's single LM)
trains alone on the raw byte sequence only, no code tokens at all. Same
attn_window=80 as v1/v2, so this isolates windowing's effect specifically:
if this ALSO plateaus around ~3.6 bpb, windowing alone is the bottleneck
(not the merge mechanism). If it does meaningfully better, the merge
mechanism itself is actively hurting relative to a plain windowed LM.

All other dims match v1/v2 exactly for a clean comparison.

    uv run python -m qcute.qcutelm_pyramid --config configs/qcutelm_pyramid_byteonly.py
"""
from pathlib import Path

Ks = (4, 4, 4)
d_model = 256
code_dim = None
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 4
mlp_mult = 4
attn_window = 80
untie_levels = False
bit_head_mode = "independent"
byte_only = True

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 2000            # diagnostic only — don't need the full 8000 to see whether the plateau matches
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
