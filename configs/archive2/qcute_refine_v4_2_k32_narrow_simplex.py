"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow.py, ONE change: `quant_type="simplex"` instead of the default
`"bsq"`.

Session rationale: generalizes `byte_softmax_head_only`'s own "give every
level an exact softmax classifier" idea to the FULL model, not just
level 0 — session: "generalize with flag to mode where every level is
softmax head 256 way (2**n with currently n fixed to 8), instead of sign
and ste, do gumbel softmax ste... basically no grid assumption that bsq
carries... this mode do not use bsq linear map, but uses shared embedding
table for all level... maintain 2 modes now: bsq, and simplex." BSQ's dq
independent sign-bits form an implicit hypercube GRID (2**dq corners,
factorized into independent/chain bits); `quant_type="simplex"` drops
that structure entirely — every level's code is a flat, unstructured
V=2**code_bits-way CATEGORY (a point on the probability simplex, no bit
factorization), produced via `gumbel_quantize` (default: deterministic
softmax+argmax straight-through, same cheap idiom `bsq_quantize` already
uses for sign()+STE — NOT actual Gumbel-noise sampling by default,
session: "is it ok to have no gumbel, just default argmax and ste like
bsq did... because gumbel is expensive") and predicted via a genuine
V-way softmax classifier at EVERY level, byte included. No separate
`code_pre`/`ntp_head` modules at all — every level's embedding table
IS its own classifier (weight-tied, `F.linear(h, embed.weight)`), and at
the default `code_bits=8` (V=256=vocab), byte level 0's table and every
code level's table are literally the SAME OBJECT — one pool, full
uniform sharing, the most extreme version of this file's own "extreme
weight sharing" lineage yet. Intuition (session): "the model with
end-to-end learn best byte code to downsample longer bytestream" — let
training itself discover the best discrete code/downsampling scheme,
unconstrained by BSQ's hypercube grid.

Everything else identical to qcute_refine_v4_2_k32_narrow.py: Ks=(32,32),
d_model=256, n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, steps=4000. `code_bits`
(default 8) and `gumbel_tau`/`use_gumbel_noise` (default 1.0/False) left
at their own defaults — this run tests the mode's OWN default settings
first, not an ablation of them.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_simplex
"""
from pathlib import Path

Ks = (32, 32)
d_model = 256
n_layers = 1
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
