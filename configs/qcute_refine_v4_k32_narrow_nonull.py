"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_k32_narrow.py,
ONE change: `fuse_use_null_kv=False` instead of the default True — no
learned null KV slot at all in `_fuse`. `fuse_position` stays at the
default "pre".

Session rationale: completes the 2x2 grid this session's own probe
findings motivated — {fuse_position: pre, post} x {fuse_use_null_kv:
True, False} — alongside `qcute_refine_v4_k32_narrow.py` (pre, null),
`qcute_refine_v4_k32_narrow_postfuse.py` (post, null), and
`qcute_refine_v4_k32_narrow_postfuse_nonull.py` (post, no null). This is
the "pre, no null" cell: isolates the null slot's own contribution
specifically in the ORIGINAL (pre-fusion, better-scoring) ordering,
rather than only testing null-slot removal in the already-weaker "post"
setting. See qcute_refine_v4_k32_narrow_postfuse_nonull.py's own
docstring and docs/kv_contribution.md §7 for the full rationale (the
`null_only`/`big_noise` findings that motivated adding `fuse_use_null_kv`
as a real architectural ablation, not just a post-hoc content-corruption
probe).

Everything else identical to qcute_refine_v4_k32_narrow.py: Ks=(32,32),
dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
attn_window=(32,32), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode=
"pq_table", fuse_position="pre" (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_nonull.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_nonull

    # probe once trained:
    uv run python scripts/probe_v4_fusion_contribution.py \\
        --checkpoint checkpoints/qcute_refine_v4_k32_narrow_nonull/best.pt
"""
from pathlib import Path

Ks = (32, 32)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True
code_embed_mode = "pq_table"
fuse_use_null_kv = False   # the actual ablation vs qcute_refine_v4_k32_narrow.py — fuse_position stays "pre"

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
