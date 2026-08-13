"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_pq.py,
ONE change: `fuse_position="post"` instead of the default "pre".

Session ask: add a flag choosing where LevelLM._fuse's cross-attention
sits relative to self.blocks (Config.fuse_position, new this session —
see its own docstring in qcute_refine_v4.py), then queue the "post"
variant based off the best-performing pq config to date.

"pre" (default, what every other fusion config this session used): fuse
THEN self.blocks — every raw-embedded position gets cross-level context
BEFORE positions exchange information with each other via self-attention,
so self-attention can propagate one position's fused context to others
during mixing. "post" (this config): self.blocks THEN fuse — positions
mix purely among themselves first (no cross-level info yet), and only
the FINAL per-position representation looks at the coarser code, with no
further mixing afterward. Both equally causally sound (independent masks
on different axes — see session discussion) but compute genuinely
different functions. Real question: does letting fused context propagate
across positions via self-attention (pre) matter, or does a
direct-to-final-representation fusion (post) do just as well or better?

Everything else identical to qcute_refine_v4_pq.py: Ks=(4,4), dqs=(8,8),
tier_d_models=(256,256), context_len=1024, attn_window=(256,64),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_pq_postfuse.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_pq_postfuse
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True
code_embed_mode = "pq_table"
fuse_position = "post"   # the actual ablation vs qcute_refine_v4_pq.py

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
