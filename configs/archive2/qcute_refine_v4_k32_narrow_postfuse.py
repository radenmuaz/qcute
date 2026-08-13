"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_k32_narrow.py,
ONE change: `fuse_position="post"` instead of the default "pre".

Session rationale: weighing pre vs. post fusion ordering raised a
specific hypothesis worth testing directly rather than just arguing
about — does "post" (self-attention runs on the RAW embedding first,
fusion only touches the FINAL representation) give level 0's own
self.blocks a more representationally SEPARATE, "unconditional-LM-like"
trunk than "pre" (self-attention runs on the ALREADY-FUSED input, so its
own representation is entangled with cross-level info from the start)?

`qcute_refine_v4_pq` vs `qcute_refine_v4_pq_postfuse` (K=4, pq_table)
already showed post scoring slightly worse (2.4678 vs 2.4565) — but that
pair isn't the right one to probe the "representational separation"
question against `qcute_refine_v4_k32_narrow`'s own fusion-contribution
finding (docs/kv_contribution.md §7: ~90% of fusion's benefit there was
capacity/structure, not content) — this config gives K=32/window=32's
own genuine pre-vs-post pair, so `scripts/probe_v4_fusion_contribution.py`
can be run against BOTH checkpoints under the SAME architecture, isolating
fuse_position as the only variable. Specific question: does POST's
`no_fusion` ablation degrade LESS than PRE's catastrophic +2.42 bpb,
confirming post keeps self.blocks more robust/reusable as a standalone
local encoder?

Everything else identical to qcute_refine_v4_k32_narrow.py: Ks=(32,32),
dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
attn_window=(32,32), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode="pq_table",
steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_postfuse.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_postfuse

    # probe once trained:
    uv run python scripts/probe_v4_fusion_contribution.py \\
        --checkpoint checkpoints/qcute_refine_v4_k32_narrow_postfuse/best.pt
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
fuse_position = "post"   # the actual ablation vs qcute_refine_v4_k32_narrow.py

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
