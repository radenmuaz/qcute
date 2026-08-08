"""qcute.qcute_refine_v3 config: CLONE of configs/qcute_refine_v3_rope.py,
ONE change: `code_embed_mode="pq_table"` instead of the default "linear".

Session ask: stack this session's two strongest independent findings —
EncoderLevel fusion (v3, lets byte_loss/val_bpb actually depend on the
coarser code — see qcute_refine_v3.py's module docstring) and pq_table
code embedding (`configs/qcute_refine_pq_table.py`, best val_bpb 2.4816,
beat qcute_refine_no_rope's linear-mode 2.5645 by 0.083 and essentially
matched bytelm_xs1_ctx1024's 2.4870 — see docs/status.md) — and see
whether their gains stack or overlap. Both address different parts of
the same "byte_loss was structurally starved" story: fusion gives it a
channel to the coarser code at all; pq_table makes that code's own
embedding more expressive once it arrives. Testing them together is the
natural next step once each has shown a win independently.

Everything else identical to qcute_refine_v3_rope.py: Ks=(4,4), dqs=(8,8),
tier_d_models=(256,256), context_len=1024, attn_window=(256,64),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True
(kept as-is, a faithful single-variable clone — NOT also flipped to
False, to avoid confounding this test with the separate rope finding),
fuse_encoder_levels=True, tok_head_mode="linear", steps=4000.

    uv run python -m qcute.qcute_refine_v3 --config configs/qcute_refine_v3_rope_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v3_rope_pq
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
code_embed_mode = "pq_table"   # the actual ablation: everything else identical to qcute_refine_v3_rope.py

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

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
