"""qcute.qcutelm config: "frozen tokenizer + PQ/factorized-embedding LM"
recipe — the symmetric alternative to configs/qcutelm_bsq_k4_frozen_vocab.py.

Recipe:
  1. --pretrain_ae: train encoder+decoder alone, no LM involved yet, to
     target_acc=0.95 val_recon_acc — high on purpose, because once frozen
     this tokenizer is never touched again: whatever quality it's at when
     frozen is a permanent ceiling on every downstream LM number.

     ARCHITECTURE CHANGE: ChunkDecoder is now code-only (z -> K bytes,
     one-shot, no masked-byte input, no iterative refinement) — like
     regular BSQ/VQ-VAE training, matching MaskGIT's own two-stage
     assumption (Chang et al. 2022: the VQGAN tokenizer is trained to
     convergence and *frozen* before any masked-token scheme is
     introduced; MaskGIT never masks inside tokenizer training). The
     previous MaskGIT-style masked-byte decoder conflated tokenizer
     training with a masking curriculum that was never actually part of
     MaskGIT's own design, and left the decoder receiving no real
     information beyond the code once masking was pushed to 100% (see
     git history / prior session notes for the full trail: maskgit_mask ->
     full_mask -> this). If masking is wanted at all, --pretrain_encoder_mask
     applies it to the ENCODER's raw byte input instead (mask_bytes(),
     MaskGIT's cosine mask-rate schedule reused there) — a denoising-
     autoencoder-style robustness objective, decoder still grades against
     the true uncorrupted bytes. Off by default here.

     2 tokenizer_layers (see docs/status.md's sanity-check: 1 layer never
     reaches 100% train recon_acc; 2 does). AdamW, cosine LR (warmup 1000
     -> constant 1000 -> decay, peak 3e-4), pretrain_weight_decay lowered
     0.1->1e-5 (tiny ~0.3M-param encoder+decoder), over the full corpus via
     batch_iter. (Also tried and reverted: full-batch L-BFGS — never
     plateaued but too slow wall-clock for how little of the corpus its
     fixed sample covered.)
  2. --freeze_after_pretrain + --lm_factorized_input: encoder+decoder
     weights frozen; training switches to train_factorized_lm (see
     qcute/qcutelm.py) — bytes -> (frozen) encoder -> raw per-dim codes ->
     FactorizedCodeEmbedding (PQ-style, compositional, no vocab/UNK) -> LM
     -> per-dim BCE next-code prediction. **No decoder loss anywhere in
     this phase** — the decoder is frozen and only invoked (no_grad) for
     qualitative decode-and-inspect logging, never for gradient.
     This is the architecturally symmetric counterpart to
     qcutelm_bsq_k4_frozen_vocab.py's flat vocab-lookup-table approach: the
     LM's input/output representation mirrors the encoder/decoder's own
     per-dim code structure exactly, instead of collapsing dq bits into one
     arbitrary vocab id.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_pq.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_pq
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
# dq left at its default (18, set in build_config): at K=4, a chunk's raw
# entropy ceiling is log2(256^4)=32 bits, but English text's actual entropy
# is only ~4.5-5 bits/byte (~18-20 bits for 4 bytes) — dq=18 targets that
# real redundancy, not the uniform-byte ceiling, so the bottleneck is a
# tight fit against the corpus's actual information content, not an
# arbitrary squeeze (see build_config()'s comment for the same note).
tokenizer_layers = 8  # doubled from 4 — that run was at 75-76% train/val (tracking closely, no overfitting gap) by step 8999/100000, similar range to every depth tried so far; widths stay halved, only depth is being scaled up now
d_byte = 32   # halved from Config's default 64
d_enc = 128   # halved from Config's default 256
d_dec = 128   # halved from Config's default 256
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 20000
pretrain_lr = 3e-4  # cosine peak
pretrain_cosine_decay = True
pretrain_constant_steps = 100  # halved from 1000 — shorter constant-LR hold before decay, matching the tinier-model experiment
pretrain_weight_decay = 1e-5  # kept low — tiny encoder/decoder
pretrain_eval_every = 100
freeze_after_pretrain = True
pq_groups = 2  # != 1 -> PQ/factorized LM path (train_factorized_lm); see --pq_groups's help for the 1-vs-not-1 semantics

# warmup_steps is a single shared flag used by BOTH pretrain_ae and the
# factorized_lm phase below (no separate --pretrain_warmup_steps exists) —
# halved from 1000 to 500 for this tinier-model experiment; also changes
# the LM phase's warmup from bpelm_8192.py's 500 (coincidentally the same
# value here), a side effect of the shared flag, not a deliberate choice.
warmup_steps = 500

# factorized_lm phase settings — matched to configs/bpelm_8192.py otherwise
steps = 2000
batch_size = 8  # halved from 16, matching the tinier-model experiment
seq_chunks = 256  # LM context, in codes (K=4 bytes each) — same order of magnitude as bpelm's context=256 BPE tokens
cosine_decay = False
lr_peak = 6e-4
log_every = 100
eval_every = 200
eval_batches = 20
