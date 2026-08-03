"""qcute.qcutelm config: same asymmetric tokenizer as
configs/qcutelm_bsq_k4_pq_asym.py (linear-only encoder: no MixerBlocks,
flatten -> single Linear; 4-layer attention decoder, 2x wider MLP), but
with --lfq — regresses the bottleneck from BSQ to plain LFQ (Lookup-Free
Quantization, Yu et al. 2023): skips BSQ's L2-normalize-onto-the-hypersphere
step, signs the raw projection directly (hypercube corners {-1,+1}^dq,
unconstrained scale) instead of signing a unit vector. Sign bits (targets)
are identical either way; only z_hat's geometry/scale changes.

Motivation: the asym config's own predecessor (encoder_layers=0) was
climbing well before being stopped for this comparison (78.66%
val_recon_acc at step 13099/20000, still trending up) — this isn't a "the
previous approach failed" swap, it's testing whether dropping BSQ's
normalize step changes the gradient dynamics through the quantization
boundary (removing one of the two attenuation sources identified in the
side gradient-norm diagnostic: normalize's Jacobian contraction, alongside
the explicit 1/sqrt(dq) rescale) enough to matter, independent of the
encoder/decoder depth allocation question the asym config is testing.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_asym_lfq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_asym_lfq
"""
from pathlib import Path

bottleneck = "bsq"
lfq = True
K = 4
encoder_layers = 0  # no MixerBlocks at all — flatten -> single Linear (out_proj) -> bottleneck, truly linear
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
