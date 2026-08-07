"""qcute.qcutelm config: sanity check, K=1 — dq=18 bits to encode a single
byte (8 bits of real information) is wildly over-provisioned capacity. If
a healthy pipeline can't reach ~100% train/val recon_acc quickly here,
that's not an architecture/capacity problem the way every other config
this session has been probing — it points at something more fundamentally
broken (data loading, the STE/quantization math, the loss/target wiring,
etc.), since no depth/width lever should matter at this trivial scale.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_sanity_k1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_sanity_k1
"""
from pathlib import Path

bottleneck = "bsq"
K = 1
encoder_layers = 1
encoder_mixer = "conv"
decoder_layers = 1
decoder_mixer = "conv"
d_byte = 32
d_enc = 128
d_dec = 128
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 2000  # trivial task at K=1 — should converge fast, not need a long budget
pretrain_lr = 3e-4
pretrain_cosine_decay = True
pretrain_constant_steps = 50
pretrain_weight_decay = 1e-5
pretrain_eval_every = 50  # finer visibility since this should converge quickly
freeze_after_pretrain = True
pq_groups = 1

warmup_steps = 500

steps = 2000
batch_size = 8
seq_chunks = 256
cosine_decay = False
lr_peak = 6e-4
log_every = 100
eval_every = 200
eval_batches = 20
