"""qcute.qcutelm config: 1-layer conv encoder AND decoder (symmetric depth,
both shallow), but wide — widths chosen to land the tokenizer's total
fp32-bit size at ~1.0x the training corpus's bit count (900,000 bytes =
7,200,000 bits), i.e. right at the capacity-parity threshold discussed
this session (see qcutelm_bsq_k4_pq.py's docstring trail: 1-layer/halved
width was 0.276x corpus bits, 8-layer/halved width crossed to 1.058x).

A literal "10x wider" (d_byte/d_enc/d_dec all x10 from the halved
baseline, e.g. d_byte=320) massively overshoots this target — ~22x corpus
bits, not ~1x — because out_proj's seq_len*d_byte -> d_enc matrix and the
conv+MLP scale faster than linearly with width. Back-solved instead for
d_byte=70, d_enc=200, d_dec=200 (~ 200/128 ≈ 1.56x the halved-width
baseline's d_enc/d_dec, not literally 10x) landing at 224,344 tokenizer
params = 7,179,008 bits, ratio 0.997x — matching corpus bits almost
exactly with a genuinely shallow (1+1 layer) architecture, isolating width
as the only large lever (no depth, no asymmetric encoder/decoder
allocation) at capacity parity.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_wide1layer.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_wide1layer
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
encoder_layers = 1
encoder_mixer = "conv"
decoder_layers = 1
decoder_mixer = "conv"
d_byte = 70
d_enc = 200
d_dec = 200
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
