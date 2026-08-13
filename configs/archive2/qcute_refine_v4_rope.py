"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_pq.py,
ONE change: `code_embed_mode="linear"` (the default — omitted below) instead
of "pq_table". Equivalently: the v4 (no DecoderLevel) counterpart of
configs/qcute_refine_v3_rope.py — fusion alone, isolated from pq_table,
now on v4's leaner architecture.

Session context: `qcute_refine_v3_rope` (v3, fusion alone, DecoderLevel
still present but inert) reached best val_bpb 2.4302 — the best single
result this session, beating both `bytelm_xs1_ctx1024` (2.4870) and every
fusion+pq_table combination tried (`qcute_refine_v4_pq`: 2.4588,
`qcute_refine_v3_rope_pq`: 2.4639). This config asks the natural next
question: does v3_rope's own 2.4302 reproduce under v4's leaner
architecture (no DecoderLevel at all), or does removing DecoderLevel
change fusion-alone's own result even though DecoderLevel's reads were
always detached and never touched byte_loss? Should reproduce closely if
the detach-based independence reasoning is right; a real gap would be a
signal something else was going on.

Everything else identical to qcute_refine_v4_pq.py (and hence
qcute_refine_v3_rope_pq.py/qcute_refine_v3_rope.py/qcute_refine_rope.py
before it): Ks=(4,4), dqs=(8,8), tier_d_models=(256,256), context_len=1024,
attn_window=(256,64), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_rope.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_rope
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
# code_embed_mode left at default "linear" — the actual ablation vs qcute_refine_v4_pq.py

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
