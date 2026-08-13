"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_simplex.py, ONE change: `n_layers=2` instead of 1 — session:
"queue simplex (code_bits=8) with double layer, at front."

Session rationale: `quant_type="simplex"` at `code_bits=8` is the most
extreme weight-sharing point in the whole v4.2 family — byte level 0 and
every code level literally share ONE `nn.Embedding(256,D)` object, used
weight-tied as both input embed and output classifier at every level
simultaneously (see the parent config's own docstring, and docs/
kv_contribution.md §13's expressivity comparison against `byte256`/
`byte_softmax_head_only`). Doubling `n_layers` gives the shared trunk
itself more capacity to compensate for that lack of per-level private
embed/head parameters, without touching the sharing scheme itself —
tests whether trunk depth is a substitute for per-level privacy, or
whether the two are addressing different bottlenecks.

Everything else identical to qcute_refine_v4_2_k32_narrow_simplex.py:
Ks=(32,32), d_model=256, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, quant_type="simplex",
code_bits=8 (default), gumbel_tau/use_gumbel_noise at their own defaults
(1.0/False — same non-Gumbel default as the parent config), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_l2.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_simplex_l2
"""
from pathlib import Path

Ks = (32, 32)
d_model = 256
n_layers = 2
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
quant_type = "simplex"

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
