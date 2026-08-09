"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_byte_factored.py, ONE change: `byte_head_lowrank=True,
byte_head_rank=16` instead of `byte_head_factored=True`.

Session ask: "how good is factoredsoftmax vs just low rank... analyze
rank." `byte_head_rank=16` deliberately MATCHES `byte_factored`'s own
param/FLOP budget exactly (8,464 params / 16,384 flops vs. factored's
8,224 / 16,384 — same order, measured via FlopCounterMode) so this is a
clean, budget-controlled comparison, not a "bigger head wins" confound.

`LowRankSoftmaxHead` (`h -> Linear(D,rank) -> Linear(rank,vocab)`, the
classic "softmax bottleneck," Yang et al. 2018) is expected to STRICTLY
beat `byte_factored` at this matched budget: `FactoredSoftmaxHead`'s
outer-sum forces every one of the 256 classes into a rigid, ZERO-free-
parameter `w1_i+w2_j` additive template (only 32 total shared "template"
vectors, no per-class freedom at all beyond a discrete row/column
assignment); `LowRankSoftmaxHead` instead gives every class its own FREE
16-dim coefficient vector (a full row of the `rank->vocab` matrix) within
a shared 16-dim subspace of `h`-space — strictly more degrees of freedom
per class at the identical rank/param ceiling. Every outer-sum-
representable logit matrix is also low-rank-representable at rank
`v1+v2=32` (a degenerate all-0-or-1-coefficient special case), so
low-rank's function class is a strict superset of factored's at that
rank. Whether that theoretical edge shows up in actual best_val_bpb is
exactly what this pair of runs tests.

Everything else identical to qcute_refine_v4_2_k32_narrow_byte_factored.py
(and, one level further back, to byte_softmax_head_only.py): Ks=(32,32),
dq=8, d_model=256, n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="independent", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_byte_lowrank.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_byte_lowrank
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
byte_head_lowrank = True
byte_head_rank = 16

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
