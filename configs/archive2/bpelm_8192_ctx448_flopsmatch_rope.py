"""qcute.bpelm config: FLOPS-matched fair comparison to qcute_refine_rope
(and qcute_refine_v2_byte4_code256_identity, same params/FLOPs), from
this session's strict-power-of-2 grid search (see qcute/bpelm.py's own
module docstring, "Session notes: vocab-size tradeoffs" section, for the
full grid and methodology).

vocab=8192, context=448, d_model=256, n_layers=3, n_heads=4 -> single
forward pass (batch=1) FLOPs 3.993G vs. qcute_refine_rope's own 3.862G —
+3.4%, the closest FLOPs match available for this specific target.

Note: this does NOT also match params — qcute_refine_rope's 2.706M has
no good power-of-2-vocab match at all (see qcute/bpelm.py's own
docstring: even the smallest vocab tried, 4096, overshoots params by
+26.2%, since the flat 2.367M trunk plus any vocab table already exceeds
2.706M). configs/bpelm_4096_paramsmatch.py is the closest params option
available, but it was chosen to match qcute_refine_rope_3level_curriculum
specifically (near-exact there), not this target.

steps=4000 (this session's step-budget finding).

    uv run python -m qcute.bpelm --config configs/bpelm_8192_ctx448_flopsmatch_rope.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_8192_ctx448_flopsmatch_rope
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_8192.model")
data = Path("datasets/enwik8_1M.gz")
context = 448
d_model = 256
n_layers = 3
n_heads = 4
val_frac = 0.1
steps = 4000
batch_size = 16
warmup_steps = 500
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
