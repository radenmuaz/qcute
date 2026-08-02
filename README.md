# qcute

Continuous byte-level tokenizer + LM, per the design in
[docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
See [docs/architecture.md](docs/architecture.md) for how the code maps to that
design, and [docs/status.md](docs/status.md) for phase-by-phase progress.

## Quickstart

```bash
uv sync

# downloads datasets/enwik8.gz (~35MB) and cuts datasets/enwik8_tiny.gz
# (500,000-byte prefix, for fast smoke/local runs)
uv run python scripts/prepare_data.py

# byte-level baseline LM w/ MTP head (Phase 0), reports bits-per-byte
uv run python -m qcute.bytelm --preset sd
uv run python -m qcute.bytelm --preset xs --data datasets/enwik8_tiny.gz  # quick local run

# end-to-end tokenizer + latent LM (encoder + FSQ/BSQ + LM + decoder, jointly trained)
uv run python -m qcute.qcutelm --bottleneck bsq --data datasets/enwik8_tiny.gz --qual_gen_bytes 64

# or via a named, reproducible config (CLI flags still override individual values)
uv run python -m qcute.bytelm --config configs/bytelm_xs_tiny_longrun.py
uv run python -m qcute.qcutelm --config configs/qcutelm_bsq_tiny.py

# evaluate a saved checkpoint only, no training
uv run python -m qcute.bytelm --eval_only --checkpoint_path checkpoints/<name>_best.pt --data datasets/enwik8_tiny.gz
```

Both modules run on CUDA/MPS/CPU automatically.

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
  autoregressive generation loop (interface Option A — sampled codes fed
  back directly, no re-encoding). Non-streaming chunk-local MLP encoder/
  decoder, simplified vs. the doc's causal-SSM design — see
  [docs/architecture.md](docs/architecture.md).
- Both scripts support: a `--config <file.py>` (see `configs/`) with CLI
  flags overriding individual config values; checkpointing (`checkpoints/`,
  gitignored) that keeps the best-so-far and most-recent model, plus
  `--eval_only --checkpoint_path ...` to evaluate without training;
  `--qual_gen_bytes` for qualitative generation — a prompt (from train/val
  data or `--qual_user_text`) alongside the model's continuation, the real
  ground-truth continuation when available, and the model's bpb on it.
- Both modules are self-contained (no shared internal submodules); see
  [docs/architecture.md](docs/architecture.md) for why and when to split further.
- Superseded design (streaming-causal-encoder standalone autoencoder,
  no LM) kept for reference in `archive/`.

Details, gaps, and next steps: [docs/status.md](docs/status.md). For how this
repo itself was scaffolded (reusable for future projects):
[docs/scaffolding_playbook.md](docs/scaffolding_playbook.md).
