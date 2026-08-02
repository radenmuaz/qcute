# Status

Tracks progress against the phase plan in
[continuous_tokenizer_handover.md §5](continuous_tokenizer_handover.md#5-implementation-phases).

## Phase 0 — infrastructure

- [x] Byte-level data pipeline (`datasets/enwik8.gz` + `datasets/enwik8_tiny.gz`
      500,000-byte subset via `scripts/prepare_data.py`).
- [x] Reference baseline: byte-level MTP LM (`qcute/bytelm.py`, `xs` preset
      ≈3.7M params, bandwidth-matched to `qcute.qcutelm`'s K=8 via 8 MTP heads),
      reports exact BPB.
- [ ] BPE softmax / BPE+MTP-on-BPE baselines — not built (this is a byte-level
      MTP baseline only).
- [x] **First real training run** (not just smoke-tested): `xs` preset on the
      tiny 500KB subset, 25000 steps, linear-warmup-then-constant LR (6e-4).
      Findings (`logs/qcute_lm_xs_1785657981.{log,jsonl}`, config equivalent
      to `configs/bytelm_xs_tiny_longrun.py`):
  - val_bpb bottomed at **~2.55** around step 1500 (clears the "beats a
    unigram baseline" bar of ~3–4 easily) — the model does learn something
    real from context, not just memorize character frequency.
  - Past that point val_bpb rose, non-monotonically, past the unigram range
    (~4.5–5.0) up to **~5.5+** by step ~11500, while train bpb collapsed to
    **~0.3–0.6** (near-total memorization of 450,000 train bytes by a 3.7M
    param model). Classic overfitting on a tiny dataset, made *visible*
    specifically because LR was held constant rather than decayed (see
    `docs/scaffolding_playbook.md` §7) — this is also the textbook setup
    for epoch-wise double descent (Nakkiran et al. 2019); whether a second
    descent shows up given the full 25000-step budget is still open as of
    this writing.
- [ ] Go/no-go (baselines reproduce literature numbers within ±5%) — not
      applicable yet; no published byte-MTP-on-enwik8-tiny-subset number to
      compare against. The real Phase 0 go/no-go (full enwik8, `sd`/`md`
      presets) hasn't been attempted — everything above is on the 500KB
      tiny subset for fast local iteration.

## Phase 1 — standalone autoencoder

Superseded by an end-to-end approach (see Phase 2 below) before this phase's
go/no-go was cleared. Old streaming-causal-encoder implementation archived
at `archive/tokenizer_phase1_standalone_autoencoder.py` (never trained to
convergence either — smoke-tested only).

## Phase 2 — LM in latent space

- [x] Encoder + FSQ/BSQ bottleneck + latent LM + decoder, trained jointly,
      with a generation loop, interface Option A (`qcute/qcutelm.py`).
      Simplified vs. the doc: non-streaming chunk-local MLP encoder/decoder
      instead of the recommended causal-SSM designs — see
      [architecture.md](architecture.md) for the full list of simplifications.
- [ ] Go/no-go (BPB matches or beats BPE+MTP baseline at matched compute) —
      not evaluated; only smoke-tested (loss decreases, reconstruction and
      latent-prediction accuracy climb on a tiny slice of data over a few
      steps), not trained to convergence on the full corpus.
- [ ] Sub-phase 2a (softmax-only LM) is what's implemented; sub-phase 2b
      (A-native vs. A-grounded comparison) not done — only Option A exists.
- [ ] Reconstruction accuracy vs. the old Phase-1 autoencoder not compared;
      the non-streaming encoder/decoder may reconstruct worse due to the
      chunk-boundary problem the doc warns about (§1.3).

## Phase 3 — geometric mixers

Not started.

## Experiment infrastructure (orthogonal to the phase plan)

Built out alongside the Phase 0/2 runs above, in both `qcute/bytelm.py` and
`qcute/qcutelm.py` — see `docs/architecture.md` for details:

- [x] `--config <file.py>` (see `configs/`), CLI flags override config values.
- [x] Checkpointing: best + last, to `checkpoints/` (gitignored).
- [x] `--eval_only --checkpoint_path ...`: evaluate a saved checkpoint without training.
- [x] `--qual_gen_bytes`: qualitative generation with ground-truth comparison
      and bpb-on-ground-truth, sourced from train/val data or user text.
- [x] Dual logging (raw text + JSONL) with elapsed-time tracking, `logs/` gitignored.
- [x] Same linear-warmup-then-constant LR schedule (`lr_at`) in both scripts.
- [ ] Not yet done: extracting the model-agnostic pieces (`Logger`,
      `Checkpointer`, `load_config_module`, `load_enwik8`, `split_train_val`,
      `lr_at`, RoPE math) into a shared `qcute/utils.py` — identified as safe
      to share (see module docstrings) but deferred as a separate decision.
