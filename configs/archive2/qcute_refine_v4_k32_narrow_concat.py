"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_
k32_narrow_nonull_uncond.py (K=32/narrow-window, no null KV, trained
under the new additive PASS1+PASS2 loss scheme), ONE change:
`fuse_position="concat"` instead of the default "pre".

Session ask: "allow levellm does decoding with pre and post cross attn,
add more param. add also if user disable both pre and post, a special
self attn with concat higher level kv at behind, with different masking,
need to concat properly for window attention to work." — this is that
THIRD mode: no separate CrossBlock at all. The level-above's hidden state
(projected to this level's D via the existing `fuse_kv_proj`, same
null-prepend behavior — here off, matching `fuse_use_null_kv=False`) is
appended to the TAIL of every `self.blocks` layer's own K/V, and each
layer does ONE joint windowed-causal + jagged-masked attention call
instead of a separate cross-attention pass. Each layer derives its own
K/V view of the fixed fused tail via ITS OWN qkv weights (no new
cross-attention parameters — see `CausalSelfAttention._fuse_kv_proj`/
`LevelLM._prep_concat` in qcute_refine_v4.py). Mechanically verified this
session: exact causality (perturbing a not-yet-resolved fuse-KV row
changes ONLY the query position where that block first becomes visible,
zero diff everywhere earlier — same boundary condition CrossBlock's own
jagged mask already enforced), and exact `generate_no_cache` vs
`generate_kv_cache` match (`validate_generation`, both null_kv on/off, a
2-level and a 3-level config) — full feature parity with "pre"/"post".

Queued alongside configs/qcute_refine_v4_k32_narrow_both.py (session ask:
"queue concat then both run") — "concat" first since it's the cheaper
mechanism (no new cross-attention params at all).

Everything else identical to qcute_refine_v4_k32_narrow_nonull_uncond.py:
Ks=(32,32), dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1),
context_len=1024, attn_window=(32,32), byte_repr="embed", code_head_mode=
"independent", cross_attn_rope=True, fuse_encoder_levels=True,
code_embed_mode="pq_table", fuse_use_null_kv=False, steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_concat.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_concat
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
fuse_use_null_kv = False
fuse_position = "concat"   # the actual ablation vs qcute_refine_v4_k32_narrow_nonull_uncond.py

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
