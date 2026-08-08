"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_pq.py,
THREE changes: `Ks=(32, 32)` (was `(4, 4)`) and `attn_window=(32, 32)`
(was `(256, 64)`) — pushing level 1 toward genuine token-level
granularity, and testing whether fusion can substitute almost entirely
for level 0's own local context.

Session rationale: with K=32, each code block spans 32 raw bytes —
roughly word/token-scale, not byte-scale — so level 1 (32 code positions
over the full 1024-byte context) is operating at something close to
actual token granularity for the first time this session. Combined with
attn_window=(32, 32):

  - level 0's own self-attention window = 32 = exactly one code block's
    own span — level 0 can only see WITHIN its current block via self-
    attention; anything beyond that can only reach it through fusion.
  - level 1's own window = 32 = its own full sequence length (1024/32),
    so this triggers CausalSelfAttention's dense-fallback path (window >=
    T) — level 1 sees its ENTIRE sequence, dense, every layer.
  - fuse_kv_window (RefineLM.__init__ sets this to windows[1]
    automatically, not separately configurable) = 32 = level 1's own full
    block count too, so fusion's cross-attention reach is effectively
    UNBOUNDED — level 0 can pull in "informative KV from way back",
    exactly as far back as the whole context, via the coarse code.

Net design: level 0 becomes a hyper-local detail processor (32-byte
window only), level 1 becomes a full-context global summarizer (dense
attention over the whole sequence), connected by an effectively-unbounded
fusion cross-attention — a much sharper division of labor than any
earlier config's Ks=(4,4)/attn_window=(256,64) (level 0 already had a
256-byte local window there, cross-attention was more of a bonus, not
level 0's only path to distant context).

Params/FLOPs (FlopCounterMode, same methodology as every other comparison
this session): 2.640M params (Ks/window don't change param count), 4.895G
flops/fwd — close to BOTH bytelm_xs1_ctx1024 (2.147G) and
bytelm_xs3_ctx1024 (5.369G, the closer match, -8.8%), consistent with the
session ask ("flops similar to xs1 and xs3").

MEMORY, measured separately (peak RSS, CPU, forward+backward, isolated
subprocess per model, same methodology across all four): this config
uses MORE memory than either xs1 or xs3 DESPITE comparable/lower FLOPs —
118.3MB (net of the ~150MB python+torch import floor) vs. xs1's 30.1MB,
xs3's 97.7MB, and this session's own K=4/window=(256,64) baseline's
71.7MB (+65% over that baseline despite -8.3% fewer FLOPs). Confirmed
via a separate scaling test that CPU SDPA already uses a memory-
efficient/flash-style kernel here (peak memory scales roughly linearly
with sequence length, not quadratically) — so the gap is NOT an "add
flash attention" fix, it's most likely CausalSelfAttention._forward_
chunked's own per-chunk bookkeeping overhead: K=32/window=32 creates 32
small chunks (vs. K=4/window=256's 4 large chunks) — more distinct
reshape/permute/concat intermediate tensors even though each chunk's own
FLOPs are smaller. See docs/status.md for the full writeup.

Everything else identical to qcute_refine_v4_pq.py: dqs=(8,8),
tier_d_models=(256,256), tier_n_layers=(1,1), context_len=1024,
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True,
fuse_encoder_levels=True, code_embed_mode="pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_k32_narrow.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_k32_narrow
"""
from pathlib import Path

Ks = (32, 32)             # the actual ablation vs qcute_refine_v4_pq.py
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)    # the other half of the ablation — see docstring

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True
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
