"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_rope.py
(fusion alone, no pq_table, no DecoderLevel — this session's best-known
config family), ONE change: `tier_n_layers=(2, 2)` instead of `(1, 1)`.

Session rationale (reviewer critique: v4's best result, 2.4588, loses to
its closest matched baseline bytelm_xs3_ctx1024 (2.4080) at genuinely
matched params/FLOPs — see docs/status.md's own honest comparison table).
Every config this session used tier_n_layers=(1,1) fixed from the start
for cross-comparison convenience, never actually tested against more
depth — the single most standard capacity lever in deep learning, a
one-line config change, no new mechanism. Before reaching for anything
cleverer, test whether the gap was ever about cross-level information at
all, or just plain undercapacity.

Params/FLOPs (measured via FlopCounterMode, same methodology as every
other comparison this session): 4.152M params, 8.963G flops/fwd. Closest
available baseline by PARAMS: bpelm_8192_ctx448_flopsmatch_rope (4.460M,
+7.4%, val_bpb 2.3559) — not a close FLOPs match though (that baseline's
own FLOPs are 3.993G, since bpelm's FLOPs are dominated by its vocab head
not depth, so this deep qcute_refine config and vocab-heavy bpelm configs
don't track together at matched params here). No baseline in this
session's set is anywhere close to 8.963G FLOPs — bpelm_32768, the
priciest, is 5.906G. This config sits in genuinely uncharted params/FLOPs
territory versus every existing baseline; treat its own result as a
standalone depth-ablation data point, not a matched-baseline comparison.

Everything else identical to qcute_refine_v4_rope.py: Ks=(4,4), dqs=(8,8),
tier_d_models=(256,256), context_len=1024, attn_window=(256,64),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="linear" (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_depth22.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_depth22
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (2, 2)   # the actual ablation vs qcute_refine_v4_rope.py
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True

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
