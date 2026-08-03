"""qcute.qcutelm config: clone of the best-performing config so far
(qcutelm_bsq_k4_pq_asym_conv1_bigdec_hilr.py — 83.03% final train
recon_acc, best this session), but two changes:

1. encoder_mixer switched from "conv" to "attention" ("full attn", full
   non-causal self-attention over all encoder input positions — matching
   the decoder's mixer, so both sides use the same mixer type now).
2. --context_len 4: the encoder now sees K + context_len = 4 + 4 = 8
   positions total (the chunk's own 4 bytes plus 4 bytes of left context
   from the previous chunk's tail, still causal — see
   gather_left_context()'s docstring). Output is unchanged: the decoder
   still reconstructs exactly K=4 bytes, only the encoder's *input* window
   is extended. This tests whether giving the encoder a look at the
   preceding bytes (real cross-chunk context, previously never available
   to the chunk-local design at all) helps it produce a more informative
   code, independent of the mixer/depth/width levers already explored.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_context4_fullattn.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_context4_fullattn
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
context_len = 4  # encoder sees K+context_len=8 positions total; output stays K=4
encoder_layers = 1
encoder_mixer = "attention"  # switched from "conv" — full attention on both sides now
decoder_layers = 4
decoder_mixer = "attention"
d_byte = 32
d_enc = 128
d_dec = 512
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 20000
pretrain_lr = 1e-3
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
