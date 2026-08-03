"""qcute.qcutelm config: same as configs/qcutelm_bsq_k4_pq_asym_conv1.py
(BSQ, 1-layer conv+MLP encoder, 4-layer attention decoder), but decoder
2x bigger — d_dec doubled 256->512 (wider, not deeper: decoder_layers
stays at 4, since the earlier gradient-norm diagnostic already found real
per-layer gradient decay across depth in the 8-layer symmetric run, so
adding width is the safer lever to test before adding more depth on top of
an already-decaying gradient signal).

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_asym_conv1_bigdec.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_asym_conv1_bigdec
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
encoder_layers = 1
encoder_mixer = "conv"
decoder_layers = 4
decoder_mixer = "attention"
d_byte = 32
d_enc = 128
d_dec = 512  # 2x qcutelm_bsq_k4_pq_asym_conv1.py's 256
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 20000
pretrain_lr = 3e-4
pretrain_cosine_decay = True
pretrain_constant_steps = 100
pretrain_weight_decay = 1e-5
pretrain_eval_every = 100
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
