"""qcute.qcutelm_vlt2 config: 4x wider capacity test (d_model 64 -> 256)
vs. qcutelm_vlt2_4layer.py, which plateaued ~65-70% val_acc at T=4 on the
full 900K-byte corpus — diagnosed (gradient-norm + tiny-subset overfit
checks, see docs/status.md-style session notes) as a capacity ceiling, not
an architecture defect. This config tests whether more capacity alone
closes the gap.

Also uses --no_curriculum (jump straight to T=K=4) since the tiny-subset
check already confirmed the architecture/decoder design converges fine;
no need to re-derive that via curriculum staging on every capacity probe.

    uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer_w256.py
"""
from pathlib import Path

K = 4
d_model = 256
n_heads = 4
n_layers_enc = 4
n_layers_dec = 4
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

no_curriculum = True

steps = 20000
batch_size = 16
lr_peak = 4e-4
weight_decay = 0.1
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

curriculum_target_acc = 0.95
