"""qcute.qcutelm config: same as configs/qcutelm_bsq_k4_pq_asym.py (BSQ
bottleneck, 4-layer attention decoder, 2x wider decoder MLP), but
encoder_layers=1 with a conv mixer + MLP (MixerBlock's default
mixer_mlp=True, not disabled) instead of the fully linear encoder_layers=0
tried there.

Motivation: qcutelm_bsq_k4_pq_asym.py's linear-only encoder (encoder_layers=0)
was already climbing well (78.66% val_recon_acc at step 13099/20000) before
being stopped for an LFQ comparison — LFQ turned out clearly slower at the
same step count (~50-54% vs ~54% for BSQ, but at 5x the steps), so LFQ was
dropped and this reverts to BSQ. This config isolates the remaining open
question from the earlier diagnostic: does giving the encoder one conv+MLP
layer (instead of none) reduce the encoder/decoder gradient-norm imbalance
enough to matter in practice, now that BSQ (not LFQ) is confirmed as the
better bottleneck choice for this architecture.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_asym_conv1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_asym_conv1
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
d_dec = 256
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
