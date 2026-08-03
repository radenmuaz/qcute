"""qcute.qcutelm_vlt config: same as configs/qcutelm_vlt_4layer.py
(4-layer encoder+decoder, d_model=64, ~1.93x corpus-bit parity), but
n_sink=4 instead of the default 1 — see ZeroKVCausalSelfAttention's and
Config.n_sink's docstrings: more zero-KV sink slots give the decoder's
identical-broadcast-input design a richer, higher-rank position-
differentiation basis in the first layer (n_sink=1 -> a single scalar per
position; n_sink=4 -> 4-dimensional), without reintroducing any actual
position embedding.

Also lr_peak lowered 1e-3 -> 6e-4: the previous run's T=4 stage was
visibly noisy (train_recon_acc bouncing ~58-80% step to step, not a clean
trend) — conflates with the n_sink change below, not a clean single-
variable ablation, but worth doing together rather than spending another
long run on a config already flagged as too fluctuating.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm_vlt --config configs/qcutelm_vlt_4layer_nsink4.py
"""
from pathlib import Path

K = 4
d_model = 64
n_heads = 4
n_layers_enc = 4
n_layers_dec = 4
n_sink = 4  # up from the default 1
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000
batch_size = 16
lr_peak = 6e-4  # down from 1e-3 — previous run's T=4 stage was too noisy
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

curriculum_target_acc = 0.95
curriculum_max_steps_per_stage = 30000
