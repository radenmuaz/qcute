"""qcute.qcute_refine_v2 config: CLONE of configs/qcute_refine_rope.py,
ONE change: `cross_attn_rope = False`. The direct counterfactual to
`qcute_refine_rope.py`'s own completed run (best val_bpb 2.6310 @ step
3600, 4000 steps) — same architecture, same step budget, same everything
else, so any val_bpb difference between the two is attributable to
cross_attn_rope alone.

(Session correction: an earlier attempt at this ablation compared
against configs/qcute_refine_v2_byte4_code256_simple.py, since deleted —
that file's actual historical run predated the cross_attn_rope feature
entirely, so it was never a reliable cross_attn_rope=True reference
point despite matching Config's CURRENT default. Cloning from
qcute_refine_rope.py instead sidesteps that ambiguity — that run is
confirmed correct under the current codebase.)

With cross_attn_rope=False, DecoderLevel's cross-attention reverts to
position-blind — Q and KV get no positional information at all beyond
the causal-block-visibility mask itself (see docs/qcute_refine_math.md
§7.2). Everything else identical to qcute_refine_rope.py: byte_repr=
"embed", code_head_mode="independent", no BitPredictHead anywhere,
Ks=(4,4), tier_d_models=(256,256), attn_window=(256,64), steps=4000 (the
standard budget every other qcute_refine_v2 ablation this session uses).

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_no_rope.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_no_rope
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
cross_attn_rope = False   # the actual ablation: everything else identical to qcute_refine_rope.py

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000   # matches qcute_refine_rope.py's own budget — the standard budget every other
                # qcute_refine_v2 ablation this session uses
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
