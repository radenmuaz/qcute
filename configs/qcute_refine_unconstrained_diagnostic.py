"""qcute.qcute_refine_v2 config: CLONE of configs/qcute_refine_tiny_byte_window.py,
THREE changes, all aimed at removing every non-architectural cap on the
model's own ceiling — session ask: "unconstrained diagnostic" (no budget/
speed concerns, just "how good can this get").

1. `dqs = (256, 256)` (was `(8, 8)`) — dq maximized to equal tier_d_models,
   so EncoderLevel's `code_pre: Linear(D, dq)` is a D->D map (no
   dimensionality reduction) instead of collapsing 256 dims down to 8.

2. `quant_type = "identity"` (was BSQ default) — removes BSQ's hard
   hypersphere-corner discretization entirely; `code_pre`'s output is used
   as the code AS-IS, continuous, no STE. Combined with (1), the
   level0->level1 code channel is (mod one learned linear map) an
   information-lossless passthrough of level 0's own hidden state — no
   quantization bottleneck, no width bottleneck.

3. `code_ntp_weight = 0.0` and a NEW flag `byte_ntp_weight = 0.0` (added
   this session to qcute/qcute_refine_v2.py's Config/CLI — previously
   level 0's own NTP loss, `byte_loss`, was hardcoded into the total loss
   at an unscaled, unconditional weight of 1.0, with no way to turn it
   off) — together these zero out BOTH encoder levels' own NTP losses,
   leaving `tok_weight=1.0`'s DecoderLevel loss (`pair0_tok_loss`, the
   LAST stage: cross-attention-based byte reconstruction from the coarser
   code) as the ONLY term in the training objective. Isolates: what can
   the decoder's cross-attention path alone reach, with zero competing
   gradient from the encoder-side NTP heads and zero information
   bottleneck feeding it?

`byte_ntp_weight=0.0` does NOT skip level 0's own NTP head forward pass
(unlike code_ntp_weight==0.0, which does skip levels>0's ntp_head calls
for real speed) — level 0's hidden state `h` is needed downstream
regardless (feeds the code channel AND the decoder's own Q side), so
`byte_acc`/`byte_loss` are still computed and logged, just excluded from
the backward-relevant total loss.

Everything else identical to qcute_refine_tiny_byte_window.py: Ks=(4,4),
context_len=1024, attn_window=(8,-1) (level 0 crippled to an 8-byte
window — kept as-is, a faithful clone, NOT reverted, since removing the
window constraint too would confound this experiment with that one),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=True
(kept from the base clone as-is; NOT updated to reflect this session's
separate rope-vs-no-rope finding — see docs/status.md — to avoid
confounding two different ablations in one run), tok_head_mode="linear",
steps=4000.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_unconstrained_diagnostic.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_unconstrained_diagnostic
"""
from pathlib import Path

Ks = (4, 4)
dqs = (256, 256)         # maximized: == tier_d_models, no width bottleneck in the code channel
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (8, -1)    # kept from qcute_refine_tiny_byte_window.py, not reverted

byte_repr = "embed"
code_head_mode = "independent"
quant_type = "identity"  # no BSQ discretization
cross_attn_rope = True

code_ntp_weight = 0.0    # level 1's own NTP loss OFF
byte_ntp_weight = 0.0    # level 0's own NTP loss OFF (new flag) -> only the decoder's tok_loss remains

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

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
