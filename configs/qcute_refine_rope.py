"""qcute.qcute_refine_v2 config: CLONE of
configs/qcute_refine_v2_byte4_code256_simple.py ("v1" of this session's
current best architecture — byte_repr="embed", code_head_mode=
"independent", no BitPredictHead anywhere), pinning
cross_attn_rope=True EXPLICITLY as its own named, documented ablation
entry (already Config's own default — this file exists so the choice is
reproducible/visible in its own right, not because the value differs
from what's already running).

QUEUED — do not launch until qcute_refine_v2_byte4_code256_simple (the
currently-running "v1" baseline) finishes; do not touch that config or
its run.

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
