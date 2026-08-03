"""qcute.qcutelm_vlt3 FSQ ablation: quant_type="fsq", levels=8 (3 bits/dim),
dq=6 -> 8^6 = 262144 = 2^18, exact codespace parity with the BSQ baseline's
dq=18 (qcutelm_vlt3_4layer.py, which reached target_acc — falsely, per a
point-estimate early-stop bug — around ~80% true val_acc at T=4).

Gradient-norm check (see session notes) showed FSQ's pre-quantization
code_proj layer gets ~4.4x stronger gradient than BSQ's at this matched
bit budget, since FSQ has no global L2-normalize step contracting gradient
across all dims the way BSQ does. This run tests whether that translates
into faster/better convergence in practice.

Same architecture/schedule as qcutelm_vlt3_4layer.py otherwise, for a
direct comparison.

    uv run python -m qcute.qcutelm_vlt3 --config configs/qcutelm_vlt3_fsq8.py
"""
from pathlib import Path

K = 4
dq = 6
quant_type = "fsq"
fsq_levels = 8
d_model = 128
n_heads = 4
n_layers = 4
mlp_mult = 4
ntp_loss_weight = 0.5
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

curriculum_target_acc = 0.95
curriculum_max_steps_per_stage = 30000
