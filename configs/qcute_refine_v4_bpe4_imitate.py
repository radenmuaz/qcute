"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_rope.py
(fusion alone, no pq_table — isolates this config's own two changes,
doesn't conflate with the code-embed lever), TWO changes: `attn_window=
(8, 256)` (was `(256, 64)`) and `tier_n_layers=(1, 2)` (was `(1, 1)`).

Session rationale: imitate the 4-layer bytelm/bpelm baselines
(`bytelm_xs_mtp4_ctx1024`, `bpelm_8192`, etc. — this session's own
closest-matched baselines throughout docs/status.md) more directly in
DEPTH, not just params/FLOPs:

  - level 0: `Ks[0]=4` (unchanged), `attn_window[0]=8` — narrow local
    window (2 code blocks' worth), forcing more reliance on fusion for
    anything beyond immediate local context, similar in spirit to
    `qcute_refine_tiny_byte_window`/`k32_narrow` but paired with a
    much coarser K=4 (not K=32) and a genuine 2-layer level 1 to fuse
    with, not just a 1-layer one.
  - level 1: `attn_window[1]=256` = its own full sequence length
    (`context_len/K=1024/4=256`), triggering `CausalSelfAttention`'s
    dense-fallback path — level 1 sees its ENTIRE sequence, same "full
    context for the coarse level" pattern as `k32_narrow`. PLUS
    `tier_n_layers[1]=2` — level 1 gets real extra depth, not just width.

Effective depth ~4, matching the baselines being imitated: level 0's own
byte_loss path is `fuse + self.blocks(1 layer)` = 2 layer-equivalents
(established this session — see docs/status.md's own depth-counting
discussion), PLUS level 1 contributes 2 more (`tier_n_layers[1]=2`) via
fusion's KV, for a combined depth in the same ballpark as a plain
4-layer dense trunk — a rougher, cheaper match than exact params/FLOPs
alignment, but directly targets "does depth-matching in this more
literal sense close more of the gap than params/FLOPs-matching alone
has."

Everything else identical to qcute_refine_v4_rope.py: Ks=(4,4), dqs=(8,8),
tier_d_models=(256,256), context_len=1024, byte_repr="embed",
code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="linear" (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_bpe4_imitate.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_bpe4_imitate
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 2)    # the other half of the ablation — level 1 gets real extra depth
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (8, 256)    # level 0: narrow (2 blocks). level 1: its own full sequence length (dense fallback)

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
