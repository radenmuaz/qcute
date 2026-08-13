"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_rope.py,
ONE change: `Ks=(2, 2)` instead of `(4, 4)` — finer code blocks (every 2
raw bytes coalesce into one code, instead of every 4).

Session rationale: with fusion, level 1's own hidden state becomes a
direct input to level 0's own prediction (via _fuse) — a finer K means
level 1's sequence is longer (512 positions instead of 256, seq_lens now
[1024, 512]) and each of its own code blocks summarizes less raw content
(2 bytes instead of 4), so fusion gets to read from a HIGHER-RESOLUTION
coarser level, updated more often (every 2 raw bytes instead of every 4).
Tests whether the coarse level's own granularity matters for how useful
fusion's own conditioning signal is.

`attn_window=(256, 64)` deliberately left UNCHANGED (not rescaled to K) —
this is a real, intentional side effect worth flagging: level 1's own
window is still 64 of ITS OWN positions, but since each of its positions
now spans only 2 raw bytes instead of 4, its effective raw-byte reach
SHRINKS to 128 bytes (64*2) instead of 256 (64*4) — a narrower local
context for level 1's own self-attention than in every other config this
session. Kept as-is deliberately (simple, one-variable-changed clone) —
not something to read past without noting.

Params/FLOPs (FlopCounterMode): 2.575M params (Ks doesn't change param
count, only sequence lengths), 5.848G flops/fwd — closest FLOPs match
across the WHOLE session's baseline set is bytelm_xs3_ctx1024 (5.369G,
-8.2%, val_bpb 2.4080), the same closest-matched baseline v4_rope/v4_pq
were already compared against, letting this run's result be read
directly against that same reference point.

Everything else identical to qcute_refine_v4_rope.py: dqs=(8,8),
tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="linear" (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k2.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k2
"""
from pathlib import Path

Ks = (2, 2)   # the actual ablation vs qcute_refine_v4_rope.py
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)   # deliberately unchanged — see docstring's own side-effect note

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
