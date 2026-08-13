"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v3_rope_pq.py
(the best combined result queued this session — EncoderLevel fusion +
code_embed_mode="pq_table" together), running under qcute.qcute_refine_v4
instead of qcute.qcute_refine_v3 — i.e. NO DecoderLevel at all (v4 removed
it entirely, see qcute/qcute_refine_v4.py's own module docstring: fusion
and DecoderLevel turned out to do the literal same job with the same
input requirements, and DecoderLevel never contributed to byte_loss/
val_bpb even in v3 since its reads stayed detached). No tok_d_model/
tok_n_heads/tok_mlp_mult/tok_head_mode/tok_weight fields — meaningless
without DecoderLevel, removed from Config entirely in v4.

Session context this config is the culmination of: `qcute_refine_v3_rope`
(fusion alone) reached best val_bpb 2.4302, beating bytelm_xs1_ctx1024's
2.4870 outright — the first qcute_refine architecture to do so this
session. `qcute_refine_pq_table` (pq_table alone, v2/no fusion) reached
2.4816. This config stacks both, now on v4's leaner architecture (no
wasted DecoderLevel compute), to see how far the combination goes.

v4 also fixes a real gap in v3: generation (generate_no_cache/
generate_kv_cache) now correctly routes through fusion (v3's generation
functions were copied unchanged from v2 and never touched cross-attention
at all) — validated via validate_generation across several architecture
variations (fusion on/off, 3-level, identity quant, mlp code-embed,
prompts shorter than one code block) before this config was written.

Everything else identical to qcute_refine_v3_rope_pq.py (and hence
qcute_refine_v3_rope.py/qcute_refine_rope.py before it): Ks=(4,4),
dqs=(8,8), tier_d_models=(256,256), context_len=1024, attn_window=
(256,64), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode=
"pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_pq
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
