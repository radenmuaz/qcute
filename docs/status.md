# Status

Tracks progress against the phase plan in
[continuous_tokenizer_handover.md §5](continuous_tokenizer_handover.md#5-implementation-phases).

## Phase 0 — infrastructure

- [x] Byte-level data pipeline (`datasets/enwik8.gz`, loaded via `load_enwik8`).
- [x] Reference baseline: byte-softmax LM (`qcute/lm.py`), reports exact BPB.
- [ ] BPE softmax / BPE+MTP baselines — not built.
- [ ] Go/no-go (baselines reproduce literature numbers within ±5%) — not yet
      evaluated; `qcute/lm.py` has only been smoke-tested for correctness (bpb
      starts near 8 and decreases), not trained to convergence.

## Phase 1 — standalone autoencoder

- [x] FSQ bottleneck + causal-byte encoder + NAT decoder (`qcute/tokenizer.py`).
- [ ] Go/no-go (reconstruction > 99.5% on held-out bytes at K=8) — not yet
      run to convergence; smoke-tested only.

## Phase 2 — LM in latent space

Not started.

## Phase 3 — geometric mixers

Not started.
