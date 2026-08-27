"""qcute_zero/ks221_1M: full-scale (full enwik8_1M, n_bytes=None) run of the hard 3-level case,
following the overfit10k validation in ks221_overfit10k.py (converged cleanly with NO curriculum --
val_byte_acc=0.387, val_cond1_acc=0.387 not degraded vs val_cond0_acc=0.382, coherent generation
from last.pt -- see docs/status.md's qcute_zero section, 2026-08-22). Same architecture (post
RMSNorm/no-bias/merged-FuseStage revision), same Ks=(2,2,1), scaled up d_model/context/steps to
qcute_lagcodec's own full-scale convention (configs/v1_stack_simplex/ks21_v256_pq1.py). NO curriculum --
qcute_zero is designed not to need one.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks221_1M.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_zero_ks221_1M
"""
from pathlib import Path

run_name = "qcute_zero_ks221_1M"
Ks = (2, 2, 1)
d_model = 256
n_layers = 4
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8

data = Path("datasets/enwik8_1M.gz")
n_bytes = None
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
log_every = 200
eval_every = 500
eval_batches = 20
