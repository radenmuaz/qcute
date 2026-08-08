"""qcute.qcute_refine_v2 config: CLONE of configs/qcute_refine_no_rope.py
(the best real-BSQ qcute_refine_v2 result so far, best val_bpb 2.5645 @
step 3300 — see docs/status.md), ONE change: `code_embed_mode="pq_table"`
instead of the default "linear".

Session hypothesis ("dq is starved"): every level>0 EncoderLevel/
DecoderLevel currently maps its own dq=8-dim BSQ code to a D=256-dim
representation via a single nn.Linear(8, 256) — only 8 additive
directions to work with. BSQ's forward value is actually one of exactly
2**8=256 discrete hypersphere corners (see bsq_quantize), so this config
instead treats the code as an index into a genuine 256-row
nn.Embedding(256, 256) lookup table (code_embed_mode="pq_table", added
this session to qcute/qcute_refine_v2.py — see Config.code_embed_mode and
the new CodeEmbed module for the full rationale and the straight-through
gradient trick it needs to keep training code_pre, the code's own
producer). Applies uniformly to every raw-code-consumption site: level>0
EncoderLevel.embed, and DecoderLevel's kv_pass_through/q_pass_through
embeds (none active in THIS config, since qcute_refine_no_rope uses
default reuse-mode decoding — but the flag is architecture-wide, wired
for when they are).

Queued conditionally this session: only meant to run if
`qcute_refine_unconstrained_diagnostic` (the no-bottleneck/decoder-only-
loss ceiling test) ALSO fails to reach bytelm_xs1_ctx1024's 1-layer
diagnostic result (best val_bpb 2.4870) — if the unconstrained ceiling
test already can't clear that bar, dq-width alone isn't the (or the only)
bottleneck, and this becomes the next most promising lever to test
(expressiveness of the code channel's mapping, not its width).

Everything else identical to qcute_refine_no_rope.py: Ks=(4,4),
context_len=1024, attn_window=(256,64), tier_d_models=(256,256),
byte_repr="embed", code_head_mode="independent", cross_attn_rope=False,
tok_head_mode="linear", steps=4000.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_pq_table.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_pq_table
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = False   # matches qcute_refine_no_rope.py, this session's better rope setting
code_embed_mode = "pq_table"   # the actual ablation: everything else identical to qcute_refine_no_rope.py

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
