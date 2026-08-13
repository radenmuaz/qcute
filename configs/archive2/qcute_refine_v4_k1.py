"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_rope.py,
ONE change: `Ks=(1, 1)` instead of `(4, 4)` — the DEGENERATE case, no
compression at all: every single raw byte becomes its own "code block"
(level 1's own sequence length equals level 0's context_len, seq_lens=
[1024, 1024]).

What this actually tests: with K=1, `bsq_quantize`/`code_pre` still run
(every raw position gets BSQ-quantized into its own 8-dim code), but the
hierarchy's whole compression rationale is gone — level 1 becomes, in
effect, ANOTHER byte-level model, one step removed, operating on a
re-encoded (quantized) version of level 0's own per-position hidden
state rather than raw bytes. The jagged causal mask degenerates cleanly
too: with K=1, `n_complete = (t+1)//1 = t+1`, so block b is visible at
position t iff `b <= t` — every position can fuse with every STRICTLY
PRIOR position's own code, immediately, no K-step delay. Mechanically
verified this degenerates without error (mask math, `h_blocks` reshape,
BSQ quantize all handle K=1 as a clean special case, not a crash) before
this config was written.

Purpose: a ceiling/sanity check on fusion itself, orthogonal to Ks=(2,2)'s
question (does finer-but-still-compressed help). If fusion's own benefit
comes from genuinely useful COARSE summarization, K=1 (no summarization
at all, level 1 is just a second differently-parameterized pass over the
same positions) should NOT help as much as K=4/K=2 — if K=1 does just as
well or better, that's evidence fusion's benefit isn't really about
coarseness/compression at all, just about having a second, differently-
initialized pass to condition on.

Cost warning: level 1's own self-attention now runs over the FULL
1024-position sequence (same as level 0), not a compressed 256/512 — real
extra compute, not just a diagnostic technicality. Params/FLOPs
(FlopCounterMode): 2.575M params (unchanged — Ks doesn't affect param
count), 6.866G flops/fwd (vs. Ks=(4,4)'s 5.340G, +28.6%). No baseline in
this session's set is closely matched to this specific FLOPs figure;
treat as a standalone diagnostic result, not a matched-baseline
comparison.

Everything else identical to qcute_refine_v4_rope.py: dqs=(8,8),
tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
attn_window=(256,64) (level 1's own window, now in raw-byte units too
since K=1 — unchanged from other configs, so its effective reach is
literally 64 raw bytes here, far narrower than the 256-byte-equivalent
other K=4 configs got), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode="linear"
(default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k1
"""
from pathlib import Path

Ks = (1, 1)   # the actual ablation vs qcute_refine_v4_rope.py — degenerate, no compression
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
