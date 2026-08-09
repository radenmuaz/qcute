"""qcute.qcute_refine_v4_2 config: same idea as the deleted
qcute_refine_v4_2_k32_narrow_ssm_id1_pq.py (full `d_model=256` width, no
downsample, `code_embed_mode="pq_table"`), but `bit_head_class="attn"`
instead of `"ssm"` — session: "queue front qcute_refine_v4_2_k32_narrow_
ssm_id1_pq but change to attn, id1."

Session rationale: tests the REVAMPED `BitPredictHeadAttn` (§14,
docs/kv_contribution.md — per-position/concat head via einsum, no BOS
parameter since `h_t` is the concat BOS, Q/K-only attention with no V/
out_proj) at full width, `bit_inner_downsample=1`. Per the session's own
compute analysis, this is the ONE point in the `{indp-8, softmax-256,
attn_id1/4/16}` comparison where `attn` costs MORE than a plain 256-way
softmax classifier on both params (2.1x) and FLOPs (16.6x) — included
anyway as the ceiling-capacity end of the sweep, matching `ssm_id1_pq`'s
own (killed-for-being-slow) role in the SSM family: establishes whether
extra capacity actually helps before conceding it's not worth the cost,
same question `attn_id4`'s divergence (fixed by `pq_table`) and
`attn_id16`'s underfitting (§11) already partly answered for the OLD,
pre-revamp module — this reruns that question on the NEW one.

Everything else identical to the deleted ssm_id1_pq.py / to `attn_id4`'s
own family: Ks=(32,32), attn_window=(32,32), dq=8, d_model=256,
n_layers=1, context_len=1024, fuse_encoder_levels=True,
fuse_use_null_kv=False, code_head_mode="chain", code_embed_mode=
"pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_attn_id1_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_attn_id1_pq
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
code_head_mode = "chain"
bit_head_class = "attn"
bit_inner_downsample = 1
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
