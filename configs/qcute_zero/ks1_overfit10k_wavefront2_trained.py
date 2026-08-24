"""qcute_zero/ks1_overfit10k_wavefront2_trained: same as ks1_overfit10k_wavefront2.py but adds
the 2026-08-24 training-time wavefront loss (Config.wavefront_weight/wavefront_K/wavefront_n_waves)
-- teacher-forces the SAME wavefront_mask lockstep structure generate_wavefront/_wavefront_draft_block
use at generation time, tiled across the whole training sequence in one pass. Tests whether this
closes the gap seen with the untrained mask (unverified generate_wavefront(n_waves=2) output was
garbage after a few tokens despite good plain NTP loss on ks1_overfit10k_wavefront2's checkpoint --
diagnosed as the mask being out-of-distribution at generation time, never exercised during training
without this loss). wavefront_K=8, wavefront_n_waves=2 match generate_wavefront's own qual-check call.

After training, run scripts/qual_wavefront_check.py against this checkpoint and compare
wavefront-ntp output/coherence against the untrained ks1_overfit10k_wavefront2 checkpoint's.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks1_overfit10k_wavefront2_trained.py
"""
from pathlib import Path

run_name = "qcute_zero_ks1_overfit10k_wavefront2_trained"
Ks = (1,)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
input_preset = 8

mtp_heads = 5
mtp_weight = 1.0

wavefront_weight = 1.0
wavefront_K = 8
wavefront_n_waves = 2

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
log_every = 20
eval_every = 50
eval_batches = 5
