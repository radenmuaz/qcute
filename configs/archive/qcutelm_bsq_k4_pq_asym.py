"""qcute.qcutelm config: asymmetric tokenizer — shallow encoder (1 conv
layer, "flatten-patch" style — a single conv over K positions approximates
flatten+linear reasonably well), deep(er) decoder (4 attention layers, 2x
wider than the encoder's MLP hidden dim). Same "frozen tokenizer +
PQ/factorized-embedding LM" recipe as configs/qcutelm_bsq_k4_pq.py
otherwise.

Motivation: a side gradient-norm diagnostic (fresh init, same architecture,
20 steps on real data, CPU, run alongside the then-active 8-layer symmetric
run) found encoder gradient norms consistently 10-20x smaller than decoder
gradient norms throughout training — the STE+normalize quantization
boundary attenuates gradient flowing back to the encoder. Every depth
experiment so far (tokenizer_layers 1/2/4/8) scaled the encoder and decoder
*equally*, which doesn't address an asymmetric gradient-starvation problem:
if the encoder is the actual bottleneck, matching its depth to the
decoder's doesn't help it train any faster relative to the decoder. Also
found: decoder's own per-layer gradient norms decayed ~5-8x from block 0 to
the last block across 8 stacked layers — a real (if modest) vanishing-
gradient signature suggesting diminishing returns from decoder depth past
some point too, separate from the encoder issue. No representation
collapse found (0/18 dead BSQ dimensions, healthy per-dim variance) — ruling
out codebook collapse as the explanation.

This config tests the opposite allocation: cheap/shallow encoder (it only
needs to flatten K bytes into a code, not do heavy reconstruction work),
more depth+width concentrated in the decoder (which does the actual
generative/reconstruction work and showed real headroom before its own
gradient decay kicked in around layer 4-5 in the 8-layer run).

UPDATE: a follow-up diagnostic tried encoder_layers=0 (truly linear —
no MixerBlock at all, just flatten -> Linear) instead of the 1-conv-layer
version — counter-intuitively, this made the encoder/decoder gradient
imbalance *worse* (ratio climbed to 12x->28x over 20 steps, vs. 8x->23x for
the 1-conv-layer version), not better. Shrinking the encoder doesn't fix
gradient starvation from the STE+normalize bottleneck — it concentrates
the same attenuated signal onto fewer params, making each one relatively
more starved. Running this variant anyway (default AdamW hparams, no
special encoder LR) to see how it actually trains, not just how its
fresh-init gradients look.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq_asym.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq_asym
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
# dq left at its default (18, set in build_config) — see qcutelm_bsq_k4_pq.py's
# comment for the entropy-matching rationale, unchanged here.
encoder_layers = 0  # no MixerBlocks at all — flatten -> single Linear (out_proj) -> bottleneck, truly linear
decoder_layers = 4
decoder_mixer = "attention"  # heavier, full non-causal self-attention — decoder does the real work
d_byte = 32   # shared embedding dim (encoder+decoder), same as qcutelm_bsq_k4_pq.py's halved value
d_enc = 128   # encoder MLP hidden width, unchanged (encoder is shallow, width matters less here)
d_dec = 256   # decoder MLP hidden width, 2x qcutelm_bsq_k4_pq.py's 128 — decoder gets more capacity
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 20000
pretrain_lr = 3e-4  # cosine peak
pretrain_cosine_decay = True
pretrain_constant_steps = 100
pretrain_weight_decay = 1e-5
pretrain_eval_every = 100
freeze_after_pretrain = True
pq_groups = 1  # single vocab-table + softmax LM path (train_vocab_lm), not PQ/factorized

warmup_steps = 500  # shared between pretrain_ae and the factorized_lm phase (see qcutelm_bsq_k4_pq.py's note)

# factorized_lm phase settings — matched to configs/bpelm_8192.py otherwise
steps = 2000
batch_size = 8
seq_chunks = 256
cosine_decay = False
lr_peak = 6e-4
log_every = 100
eval_every = 200
eval_batches = 20
