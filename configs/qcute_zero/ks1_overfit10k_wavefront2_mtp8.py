"""qcute_zero/ks1_overfit10k_wavefront2_mtp8: same as ks1_overfit10k_wavefront2_trained.py
(training-time wavefront_weight loss, K=8/n_waves=2/region_len=4) but mtp_heads=8 instead of 5 --
covers offsets 2..8, the max needed to ALSO bootstrap the fully-parallel n_waves=8/region_len=1
degenerate case (max_offset=(8-1)*1+1=8) from the SAME checkpoint, not just n_waves=2's
max_offset=5. Motivated by chat 2026-08-24's finding that unverified generate_wavefront(n_waves=2)
free-run quality is bottlenecked by the mtp5 bootstrap head's poor validation generalization
(val_mtp5_acc~0.16 despite train_mtp5_acc~0.97) -- this run checks both (a) whether the extra
head capacity/longer offset coverage changes bootstrap quality at all, and (b) how the fully
degenerate n_waves=8 (pure MTP-bootstrap, no lockstep) mode performs/exact-matches via
generate_wavefront_mtp, compared to n_waves=2.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks1_overfit10k_wavefront2_mtp8.py
"""
from pathlib import Path

run_name = "qcute_zero_ks1_overfit10k_wavefront2_mtp8"
Ks = ()
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
input_preset = 8

mtp_heads = 8
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
