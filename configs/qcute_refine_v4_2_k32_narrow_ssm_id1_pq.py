"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_attn_id1_pq.py, ONE change: `bit_head_class="ssm"` instead of
`"attn"` — session: "queue attn_id1_pq clone but change to ssm, id1
after attn_id1_pq."

Session rationale: full `d_model=256` width (`bit_inner_downsample=1`,
no downsample), on the REVAMPED `BitPredictHeadSSM` this time (§13,
docs/kv_contribution.md — per-position head via einsum, concat instead
of add, trainable `bos_state`) — the direct sibling of `attn_id1_pq`
(same revamp story, `BitPredictHeadAttn`, queued immediately before this
config). An earlier config with this exact name existed and was killed
for being too slow (~0.88 it/s) BEFORE the concat/einsum/bos_state
revamp landed — that einsum fix in particular directly targets the
per-position head's own worst inefficiency (computing a full `[N,dq,dq]`
matrix and discarding every off-diagonal entry), so this reruns that
same full-width test now that the wasteful part of the head is fixed,
rather than assuming the earlier "too slow" verdict still holds
unchanged.

Everything else identical to qcute_refine_v4_2_k32_narrow_attn_id1_pq.py
(and to `ssm_id4_pq_concat`'s own family): Ks=(32,32),
attn_window=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
fuse_encoder_levels=True, fuse_use_null_kv=False, code_head_mode="chain",
code_embed_mode="pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_ssm_id1_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_ssm_id1_pq
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
bit_head_class = "ssm"
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
