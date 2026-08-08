"""qcute.qcute_refine_v2 config: CLONE of configs/qcute_refine_no_rope.py
(current best real-BSQ v2 result, 2.5645 val_bpb — see docs/status.md),
ONE change: `attn_window = (-1, 4)` instead of `(256, 64)`.

Session rationale: `bytelm_xs1_ctx1024` (1-layer, DENSE causal attention,
full 1024-byte receptive field — bytelm.py's CausalSelfAttention has no
window concept at all) beats every qcute_refine_v2 result. Level 0's own
self-attention here was windowed to 256 bytes — a QUARTER of xs1's own
reach — and per this session's separate finding, v2's `byte_loss`/
`val_bpb` has ZERO access to anything beyond that window anyway (cross-
attention only ever touched the detached `tok_loss` path). So the
xs1-vs-qcute_refine_v2 comparison was never apples-to-apples on
receptive field alone, independent of any hierarchy/fusion question.

This config removes that confound directly: `attn_window[0] = -1`
(dense, level 0's own self-attention now spans the FULL 1024-byte
context, matching xs1 exactly), while DELIBERATELY STARVING level 1's own
window to `attn_window[1] = 4` (4 of its own 256 code positions — a
16-raw-byte-equivalent local reach, far below its previous 64).
Rationale for starving level 1 specifically (not leaving it at 64,
and not also matching it to dense): with level 0 already at xs1's own
full reach, the interesting remaining question is whether the coarser
level (and cross-attention) can add ANYTHING when it's given almost
nothing to work with itself — if this config's val_bpb ties or loses to
`bytelm_xs1_ctx1024`, that's strong evidence the whole hierarchy was
compensating for a self-inflicted windowing limit, not adding real value
of its own; if it meaningfully beats xs1 despite level 1 being nearly
blind, that's evidence the hierarchy/cross-attention mechanism itself is
pulling real weight, not just working around windowing.

Everything else identical to qcute_refine_no_rope.py: Ks=(4,4),
dqs=(8,8), tier_d_models=(256,256), context_len=1024, byte_repr="embed",
code_head_mode="independent", cross_attn_rope=False, tok_head_mode=
"linear", steps=4000. Uses qcute.qcute_refine_v2 (NOT v3/fusion) —
deliberately isolates the windowing/receptive-field hypothesis from the
separate fusion hypothesis being tested by qcute_refine_v3_rope; a v3
companion (same attn_window, fuse_encoder_levels=True) is a natural
follow-up once both this and v3_rope report back, to see whether fusion
adds anything ON TOP of a properly-widened level 0.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_dense0_starve1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_dense0_starve1
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (-1, 4)   # level 0: dense, full 1024-byte reach (matches bytelm_xs1_ctx1024 exactly).
                         # level 1: starved to 4 of its own 256 code positions (~16 raw bytes) —
                         # isolates whether the coarser level/cross-attention adds anything once
                         # level 0 alone already matches xs1's own receptive field.

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = False   # matches qcute_refine_no_rope.py, this session's better rope setting

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
