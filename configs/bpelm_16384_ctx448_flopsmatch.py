"""qcute.bpelm config: FLOPS-matched fair comparison to
qcute_refine_decoder_trunk, from this session's strict-power-of-2 grid
search (see qcute/bpelm.py's own module docstring, "Session notes:
vocab-size tradeoffs" section, for the full grid and methodology).

vocab=16384, context=448, d_model=256, n_layers=3, n_heads=4 -> single
forward pass (batch=1) FLOPs 5.872G vs. qcute_refine_decoder_trunk's own
5.878G — -0.1%, near-exact (the best FLOPs match found across the whole
grid for ANY qcute_refine target). Fairer than configs/bpelm_16384_xs3.py
(same vocab, but context=256 wasn't chosen to match any specific
qcute_refine config's params or FLOPs — this config only differs by
context).

Note: this config does NOT also match params (params and FLOPs pull
toward different vocab choices — see the docstring's own caveat) — at
vocab=16384 total params measure 6.561M vs. decoder_trunk's own 4.424M,
a real 48.3% gap. Use configs/bpelm_4096_paramsmatch.py instead if
params-fairness matters more for a given comparison (there matched
against qcute_refine_rope_3level_curriculum instead, since no single
config matches both axes for the same target simultaneously).

steps=4000 (this session's step-budget finding).

    uv run python -m qcute.bpelm --config configs/bpelm_16384_ctx448_flopsmatch.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_16384_ctx448_flopsmatch
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_16384.model")
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
