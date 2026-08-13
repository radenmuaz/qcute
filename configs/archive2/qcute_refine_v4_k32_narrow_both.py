"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_
k32_narrow_nonull_uncond.py (K=32/narrow-window, no null KV, trained
under the new additive PASS1+PASS2 loss scheme), ONE change:
`fuse_position="both"` instead of the default "pre".

Session ask: "allow levellm does decoding with pre and post cross attn,
add more param." Runs TWO separate `CrossBlock` cross-attention modules
per fusing level — `fuse_cross_pre` (before self.blocks, as in the
default "pre") AND `fuse_cross_post` (after self.blocks, as in "post")
— genuinely more parameters than either alone (see qcute_refine_v4.py's
`LevelLM.__init__`: "both" allocates both CrossBlocks, "pre"/"post" only
one). Mechanically verified this session: forward+backward clean (no
NaN) at 2 and 3 levels, and exact `generate_no_cache` vs
`generate_kv_cache` match (`validate_generation`, null_kv on/off) — full
generation parity with "pre"/"post"/"concat".

Queued alongside configs/qcute_refine_v4_k32_narrow_concat.py (session
ask: "queue concat then both run") — "both" second, as the more
expensive of the two new modes.

Everything else identical to qcute_refine_v4_k32_narrow_nonull_uncond.py:
Ks=(32,32), dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,1),
context_len=1024, attn_window=(32,32), byte_repr="embed", code_head_mode=
"independent", cross_attn_rope=True, fuse_encoder_levels=True,
code_embed_mode="pq_table", fuse_use_null_kv=False, steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow_both.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow_both
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
fuse_position = "both"   # the actual ablation vs qcute_refine_v4_k32_narrow_nonull_uncond.py

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
