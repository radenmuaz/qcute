"""qcute.qcutelm config: "frozen tokenizer + vocab LM" recipe — the
--freeze_after_pretrain path, architecturally the closest thing to a fair
apples-to-apples comparison against configs/bpelm_8192.py (both are a plain
categorical causal-LM trained by cross-entropy over a fixed vocabulary of
discrete tokens; the only difference is *how* the vocabulary is produced —
sentencepiece BPE over bytes for bpelm, this project's own learned BSQ
tokenizer for qcutelm).

Recipe (see pretrain_autoencoder()'s and main()'s --freeze_after_pretrain
docstrings for the full mechanics):
  1. --pretrain_ae: train encoder+decoder alone (plain MaskGIT reconstruction
     CE, no LM involved yet) to target_acc=0.95 val_recon_acc — high on
     purpose, because once frozen this tokenizer is never touched again:
     whatever quality it's at when frozen is a permanent ceiling on every
     downstream LM number. 2 tokenizer_layers (see docs/status.md's
     sanity-check: 1 layer never reaches 100% train recon_acc no matter how
     else tuned; 2 does). pretrain_steps is a long, generous ceiling, not the
     expected stopping point — pretrain_target_acc should trigger the early
     stop well before it's exhausted; if it doesn't, that's itself a real
     finding (2 layers insufficient at this corpus scale), not just a
     budget problem.
  2. --freeze_after_pretrain: encoder+decoder weights frozen, a discrete
     vocabulary is built from the training data's actual codes
     (build_code_vocab), and training switches entirely to train_vocab_lm —
     a plain nn.Embedding + causal transformer + softmax-over-vocab head,
     trained by cross-entropy on vocab ids only. **No decoder byte-CE loss
     at any point in this phase** — the decoder is frozen and never
     receives gradient again; the only loss is next-code prediction,
     exactly matching bpelm's next-BPE-token prediction setup.
  3. The vocab_lm phase reuses --steps/--batch_size/--warmup_steps/--lr_peak
     below, set to match configs/bpelm_8192.py exactly (steps=2000,
     batch_size=16, warmup_steps=500, lr_peak=6e-4, cosine_decay=False) for
     as close to a controlled comparison as the two architectures allow.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_frozen_vocab.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcutelm_bsq_k4_frozen_vocab
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
tokenizer_layers = 2
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 40000
freeze_after_pretrain = True

# vocab_lm phase settings — matched to configs/bpelm_8192.py
steps = 2000
batch_size = 16
seq_chunks = 256  # LM context, in codes (K=4 bytes each) — same order of magnitude as bpelm's context=256 BPE tokens
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 100
eval_every = 200
eval_batches = 20
