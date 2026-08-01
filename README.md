# qcute

Continuous byte-level tokenizer + LM, per the design in
[docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
See [docs/architecture.md](docs/architecture.md) for how the code maps to that
design, and [docs/status.md](docs/status.md) for phase-by-phase progress.

## Quickstart

```bash
uv sync

# dataset — both modules default to datasets/enwik8.gz
mkdir -p datasets
curl -L -o datasets/enwik8.gz \
  https://github.com/lucidrains/memory-transformer-xl/raw/master/examples/enwik8_simple/data/enwik8.gz

# byte-level baseline LM (Phase 0), reports bits-per-byte
uv run python -m qcute.lm --preset sd

# FSQ tokenizer autoencoder (Phase 1), reports reconstruction accuracy
uv run python -m qcute.tokenizer
```

Both modules run on CUDA/MPS/CPU automatically.

## So far

- `qcute/lm.py`: byte-level causal transformer with RoPE, exact BPB. Two
  power-of-2-friendly presets, `sd` (~101M) and `md` (~403M).
- `qcute/tokenizer.py`: standalone FSQ tokenizer autoencoder (causal-byte
  encoder + memoryless NAT decoder) — Phase 1 bottleneck validation, no LM yet.
- Both modules are self-contained (no shared internal submodules yet); see
  [docs/architecture.md](docs/architecture.md) for why and when to split further.

Details, gaps, and next steps: [docs/status.md](docs/status.md). For how this
repo itself was scaffolded (reusable for future projects):
[docs/scaffolding_playbook.md](docs/scaffolding_playbook.md).
