"""qcute.qcutelm_vlt2 config: first experiment for the code-prefix decoder
fork (qcute/qcutelm_vlt2.py) — position 0 always holds the BSQ code
embedding, positions 1..T-1 are trainable position embeddings (no more
identical-broadcast-z / NoPE-for-position-differentiation the way
qcutelm_vlt.py's decoder worked). Zero-KV fixed at 1 sink (escape-hatch
role only now, not load-bearing for position differentiation).

Same architecture scale as qcutelm_vlt_4layer_nsink4.py (the best-performing
qcutelm_vlt config so far, though it plateaued ~71-79% at T=4 and never hit
95% — see docs/status.md-style session notes): 4-layer encoder+decoder,
d_model=64 (~1.93x corpus-bit parity), lr_peak=6e-4, 100K-step budget,
same curriculum/replay/forgetting-check machinery.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm_vlt2 --config configs/qcutelm_vlt2_4layer.py
"""
from pathlib import Path

K = 4
d_model = 64
n_heads = 4
n_layers_enc = 4
n_layers_dec = 4
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
