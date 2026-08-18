"""qcute.bpelm config: PARAMS-matched fair comparison to
qcute_refine_rope_3level_curriculum, from this session's strict-power-of-2
grid search (see qcute/bpelm.py's own module docstring, "Session notes:
vocab-size tradeoffs" section, for the full grid and methodology).

vocab=4096, d_model=256, n_layers=3, n_heads=4 -> total params 3.415M vs.
qcute_refine_rope_3level_curriculum's 3.414M — +0.03%, near-exact (the
best params match found across the whole vocab grid for ANY qcute_refine
target). Fairer than configs/bpelm_16384_xs3.py, which wasn't chosen to
match any specific qcute_refine config's params or FLOPs.

context=384 tokens — bytes/token at vocab=4096 measured 2.836 (see
qcute/bpelm.py docstring), so 384 tok ~= 1089 bytes (+6.3% vs the
1024-byte target other baselines use), the closest 64-multiple context
to 1024 bytes at this vocab (320 tok undershoots to 908 bytes, -11.4%).

Note: this config does NOT also match FLOPs (params and FLOPs pull
toward different vocab choices — see the docstring's own caveat) — at
vocab=4096/context=384 FLOPs measure 2.617G vs. rope_3level_curriculum's
own 4.330G, a real gap. Use configs/bpelm_16384_ctx448_flopsmatch.py
instead if FLOPs-fairness matters more for a given comparison.

steps=4000 (this session's step-budget finding).

    uv run python -m qcute.bpelm --config configs/bpelm_4096_paramsmatch.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_4096_paramsmatch
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_4096.model")
data = Path("datasets/enwik8_1M.gz")
context = 384
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
