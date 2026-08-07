"""qcute.qcute_refine_v2 config: originally cloned from the now-DELETED
configs/qcute_refine_v2_byte4_code256_simple.py ("v1" of an earlier
session's best architecture — byte_repr="embed", code_head_mode=
"independent", no BitPredictHead anywhere), pinning cross_attn_rope=True
explicitly. CORRECTION (later session): `simple.py` was deleted because
its actual historical run predates the cross_attn_rope feature/default
being added at all — its cross-attention was genuinely position-blind at
the time it trained, making the earlier claim "simple already used
cross_attn_rope=True by default" WRONG (compared against Config's
CURRENT default, not what existed when simple.py actually ran); its
results are no longer a reliable rope=True reference point. This file
(qcute_refine_rope.py) itself is unaffected and remains a genuine,
correctly-labeled cross_attn_rope=True run under the current codebase —
completed, best val_bpb 2.6310 @ step 3600 (4000-step budget). See
configs/qcute_refine_no_rope.py (cloned from THIS file, cross_attn_rope=
False, same 4000 steps) for its direct counterfactual.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_rope.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_rope
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
cross_attn_rope = True

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000   # cut from 8000 (session finding: both bytelm_xs_mtp4_ctx1024 and bpelm_32768 hit
                # their own best val_bpb well before step 2000 and are fully overfit/plateaued by
                # step 4000 — running past that just burns wall-clock without adding comparison signal)
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
