"""qcute.qcutelm_vlt2 config: same as qcutelm_vlt2_4layer_w256.py (d_model=256,
wd=0.1, no_curriculum straight to T=K=4) but on a 10K-byte subset instead of
the full 900K-byte corpus — sanity check for whether the loss fluctuation
seen on the full corpus is inherent to the arch/optimizer settings at this
scale, or a data-diversity effect that shows up only at full-corpus size.

    uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer_w256_10k.py
"""
from pathlib import Path

K = 4
d_model = 256
n_heads = 4
n_layers_enc = 4
n_layers_dec = 4
data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
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
