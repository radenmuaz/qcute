"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_rope.py,
ONE change: `tier_n_layers=(2, 1)` instead of `(1, 1)` — deepen level 0
(the level whose own NTP loss IS val_bpb) only, leave level 1 at its
original single layer.

Session rationale: same as configs/qcute_refine_v4_depth22.py (reviewer
critique that v4's best result loses to a matched bytelm baseline; test
the standard, boring depth lever before anything cleverer) — this variant
isolates depth at level 0 specifically, cheaper than deepening both
levels, and targets the level whose own path is what val_bpb actually
measures (byte_loss's own depth, established earlier this session: fuse
+ self.blocks, so doubling self.blocks here directly deepens that path).

Params/FLOPs (FlopCounterMode): 3.363M params, 8.561G flops/fwd. Closest
available baseline by PARAMS: bytelm_xs_mtp4_ctx1024 — 3.412M, only
+1.5% off, the closest single params match found for ANY config this
session (own val_bpb: 2.3650, the real target to beat). Also close:
bpelm_4096_paramsmatch (3.420M, +1.7%, val_bpb 2.3531). Neither is a
close FLOPs match (bytelm_xs_mtp4: 6.979G, -18.5% vs this config's
8.561G; bpelm_4096_paramsmatch: 2.617G, since bpelm's FLOPs track its
vocab head, not depth) — params is the fairer axis to compare on here.

Everything else identical to qcute_refine_v4_rope.py: Ks=(4,4), dqs=(8,8),
tier_d_models=(256,256), context_len=1024, attn_window=(256,64),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="linear" (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_depth21.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_depth21
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (2, 1)   # the actual ablation vs qcute_refine_v4_rope.py
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
