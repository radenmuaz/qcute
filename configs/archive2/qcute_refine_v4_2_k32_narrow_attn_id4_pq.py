"""qcute.qcute_refine_v4_2 config: revamped `BitPredictHeadAttn` at
`bit_inner_downsample=4` (session: "id4" — reuses the pre-revamp
`attn_id4_pq` name/width, now on the post-revamp head).

Session rationale: `qcute_refine_v4_2_k32_narrow_attn_id1_pq` (full
`d_model=256` width, no downsample) was killed for being too slow (0.57
it/s — even slower than the `ssm_id1_pq` run that was itself killed for
being too slow at 0.88 it/s). Same fix pattern as `ssm_id4_pq_concat`
before it: downsample to 4x instead of running at full width.

`BitPredictHeadAttn` also gained ONE more change since the original
(pre-revamp) `attn_id4_pq` config — a trainable BOS embed for the
attention sequence's own position-0 "previous bit" slot
(`self.bos_val_emb`, zero-init `nn.Parameter`, session: "make zero_vec
trainable embeds"). This is separate from the head-level BOS question
(`h_t` reaching the head via concat already gives position 0 a distinct
signal there — no separate placeholder needed for that); this new
parameter instead gives `_mha` itself a genuine, distinguishable
start-of-chain signal to attend to/from, instead of an unadorned zero
vector. Verified via fixed/loop consistency (`torch.allclose atol=1e-5`)
and a gradient check confirming `bos_val_emb.grad` is nonzero.

`code_embed_mode="pq_table"` carried over — same fix that resolved the
original `attn_id4`'s divergence.

Everything else identical to qcute_refine_v4_2_k32_narrow_ssm_id4_pq_
concat.py's own family: Ks=(32,32), attn_window=(32,32), dq=8,
d_model=256, n_layers=1, context_len=1024, fuse_encoder_levels=True,
fuse_use_null_kv=False, code_head_mode="chain", bit_head_class="attn",
bit_inner_downsample=4, steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_attn_id4_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_attn_id4_pq
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
bit_inner_downsample = 4
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
