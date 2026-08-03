# qcute

("Quantized Continuous Tokenizer") — continuous byte-level tokenizer + LM, per the design in
[docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
See [docs/architecture.md](docs/architecture.md) for how the code maps to that
design, and [docs/status.md](docs/status.md) for phase-by-phase progress.

## Quickstart

```bash
uv sync

# downloads datasets/enwik8.gz (~35MB) and cuts datasets/enwik8_1M.gz
# (1,000,000-byte prefix, for fast smoke/local runs)
uv run python scripts/prepare_data.py

# byte-level baseline LM w/ MTP head (Phase 0), reports bits-per-byte
uv run python -m qcute.bytelm --preset sd
uv run python -m qcute.bytelm --preset xs --data datasets/enwik8_1M.gz  # quick local run

# end-to-end tokenizer + latent LM (encoder + FSQ/BSQ + LM + decoder, jointly trained)
uv run python -m qcute.qcutelm --bottleneck bsq --data datasets/enwik8_1M.gz --qual_gen_bytes 64

# BPE baseline (handover §1.6's BPE half) — train the tokenizer first
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model --data datasets/enwik8_1M.gz

# or via a named, reproducible config (CLI flags still override individual values)
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4.py
uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_k4_frozen_vocab.py     # tightly-coupled BSQ, K=4, 2-layer tokenizer, frozen tokenizer + vocab LM
uv run python -m qcute.bpelm --config configs/bpelm_8192.py

# evaluate a saved checkpoint only, no training
uv run python -m qcute.bytelm --eval_only --checkpoint_path checkpoints/<run_name>/best.pt --data datasets/enwik8_1M.gz
```

All three modules run on CUDA/MPS/CPU automatically.

## So far

- `qcute/bytelm.py`: byte-level causal transformer with RoPE, `mtp_heads` parallel
  next-byte heads bandwidth-matched to `qcute.qcutelm`'s K (handover §1.6's
  BPE+MTP baseline, byte-level), train/val split + periodic eval, exact BPB
  from head 0. Presets `xs` (~3.7M, for quick local runs), `sd` (~101M),
  `md` (~403M). Also includes a self-speculative decoding generator (MTP
  heads as draft, verified against a true causal pass) to benchmark
  generation latency against `qcute.qcutelm`'s K-bytes-per-step decode.
- `qcute/qcutelm.py`: encoder + FSQ/BSQ bottleneck + latent LM + decoder,
  trained jointly end-to-end, train/val split + periodic eval, with an
  autoregressive generation loop. Non-streaming, non-causal chunk-local MLP
  encoder (a causal-TCN variant was tried and reverted — causality belongs
  to the LM, not the chunk-local encoder), simplified vs. the doc's
  causal-SSM design. **FSQ**: loosely coupled, interface Option A (sampled
  codes fed back directly, no re-encoding). **BSQ**: tightly coupled by
  default — the LM's own predicted latent (not the encoder's code) is what
  the decoder learns to decode, with the old loosely-coupled behavior
  available as an optional `--disable_aux_recon`-toggleable auxiliary
  loss. `--lfq` regresses BSQ's quantizer to plain LFQ (Yu et al. 2023).
  The decoder is **MaskGIT-style** (`--maskgit_T`, default 4 refinement
  steps): given a partially-masked byte chunk + the latent, predicts the
  masked positions from the unmasked ones, instead of decoding all K bytes
  independently. `--uncertainty_weighting` (Kendall & Gal 2018) and
  `--entropy_reg_weight` (Yu et al. 2023's LFQ/BSQ entropy term) are two
  further training-loop-only loss-combination options. This is still the
  weakest of the three baselines on this repo's tiny-corpus numbers — see
  [docs/status.md](docs/status.md) for the full trail of variants tried
  (LFQ vs. BSQ, aux on/off, uncertainty weighting, entropy regularization,
  AE pretraining, `dq` sweeps, the MaskGIT decoder change) and what each
  one did or didn't fix. `scripts/diagnose_qcutelm.py` (per-position
  accuracy + per-loss-term gradient norms) and
  `scripts/qualitative_compare.py` (side-by-side generation vs.
  bytelm/bpelm on the same prompts) are the two diagnostic tools that
  actually explained *why*, not just *that*, qcutelm underperforms. See
  [docs/architecture.md](docs/architecture.md) for the full design.
- `qcute/bpelm.py`: the BPE half of handover §1.6's BPE+MTP baseline (no
  MTP here — bytelm covers that half) — a sentencepiece-BPE-tokenized
  causal transformer, same trunk as bytelm, with exact byte-weighted BPB
  (not the naive mean-tokens-per-avg-token-length approximation) so it's
  genuinely comparable to the other two. Needs `scripts/train_bpe.py` run
  first. Has `generate_ar`/`score_continuation_bpb` functions (used by
  `scripts/qualitative_compare.py`), but no `--qual_gen_bytes` CLI flag of
  its own yet — narrower scope than bytelm/qcutelm.
- All three scripts support: a `--config <file.py>` (see `configs/`) with CLI
  flags overriding individual config values; checkpointing (`checkpoints/`,
  gitignored) that keeps the best-so-far and most-recent model, plus
  `--eval_only --checkpoint_path ...` to evaluate without training.
  `qcute.bytelm`/`qcute.qcutelm` additionally support `--qual_gen_bytes` for
  qualitative generation — a prompt (from train/val data or
  `--qual_user_text`) alongside the model's continuation, the real
  ground-truth continuation when available, and the model's bpb on it
  (`qcute.bpelm` doesn't have this yet — narrower scope for now).
- All three modules are self-contained (no shared internal submodules); see
  [docs/architecture.md](docs/architecture.md) for why and when to split further.
- Superseded design (streaming-causal-encoder standalone autoencoder,
  no LM) kept for reference in `archive/`.

Details, gaps, and next steps: [docs/status.md](docs/status.md). For how this
repo itself was scaffolded (reusable for future projects):
[docs/scaffolding_playbook.md](docs/scaffolding_playbook.md).
