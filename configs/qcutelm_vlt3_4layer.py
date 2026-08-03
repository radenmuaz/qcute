"""qcute.qcutelm_vlt3 first experiment: single shared-weight AR tokenizer
(fork of qcute.qcutelm_vlt2 — see that module's docstring for the design:
one causal transformer plays both encoder and decoder roles, joined by a
BSQ code that becomes the decode stage's variable BOS token).

n_layers=4 to roughly match total depth of qcutelm_vlt2_4layer's separate
4-layer encoder + 4-layer decoder in spirit, though here it's ONE 4-layer
stack shared by both stages (half the distinct weights, reused twice per
step) rather than 4+4 distinct layers — d_model bumped to 128 to compensate
somewhat for the shared-capacity trade.

    uv run python -m qcute.qcutelm_vlt3 --config configs/qcutelm_vlt3_4layer.py
"""
from pathlib import Path

K = 4
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
