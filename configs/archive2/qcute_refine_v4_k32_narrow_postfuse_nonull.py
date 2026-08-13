"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_k32_narrow_postfuse.py,
ONE change: `fuse_use_null_kv=False` instead of the default True — no
learned null KV slot at all in `_fuse`.

Session rationale: `qcute_refine_v4_k32_narrow`'s own fusion-contribution
probe (docs/kv_contribution.md §7) found that ~88-92% of fusion's total
benefit survived with the coarser level's REAL content zeroed out or
drowned in noise — strong evidence most of the benefit was `fuse_cross`'s
own extra capacity/parameters, not the content it was designed to carry.
The null KV slot (`Config.fuse_use_null_kv`, new this session) is
implicated directly in that finding: it's a real, NEVER-corrupted learned
parameter that stays fully intact even in the `null_only`/`big_noise`
ablations, and is exactly the kind of "free capacity" the probe's finding
points at. This config asks the direct question: how much of THAT
capacity effect was the null slot specifically, versus `fuse_cross`'s
other parameters (QKV/out/MLP projections)?

Combined with `fuse_position="post"` (testing representational separation
— does self.blocks stay a cleaner, more robust standalone local encoder
when it never sees fused input?), this stacks BOTH ablations from the
same session thread onto one run, rather than requiring two more
sequential configs.

Mechanically verified this session (forward+backward, AND
`validate_generation` across prompt lengths spanning the K=32 boundary,
for all 4 combinations of {pre,post} x {null on,off}): `fuse_use_null_kv=
False` degrades cleanly — F.scaled_dot_product_attention with a
zero-length or fully-masked KV (the genuine early-position/short-prompt
case) produces a well-defined ZERO output, no crash/NaN, in both forward
and backward passes. Not a correctness risk, a real architectural
ablation.

Everything else identical to qcute_refine_v4_k32_narrow_postfuse.py (and
hence qcute_refine_v4_k32_narrow.py before it): Ks=(32,32), dqs=(8,8),
tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
attn_window=(32,32), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode=
"pq_table", fuse_position="post", steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_postfuse_nonull.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_postfuse_nonull

    # probe once trained (note: no_fusion/null_only modes in the probe script
    # are moot here since there's no null_kv to null out — normal vs no_fusion
    # is still the meaningful comparison):
    uv run python scripts/probe_v4_fusion_contribution.py \\
        --checkpoint checkpoints/qcute_refine_v4_k32_narrow_postfuse_nonull/best.pt
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
fuse_position = "post"
fuse_use_null_kv = False   # the actual ablation vs qcute_refine_v4_k32_narrow_postfuse.py

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
