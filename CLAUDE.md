# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                     # install/update env from pyproject.toml + uv.lock
uv run python -m qcute.lm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.tokenizer            # FSQ tokenizer autoencoder (Phase 1)
```

Both modules read `--help` for their full flag list. No test suite, linter, or
CI config exists yet.

## Architecture

`qcute/lm.py` and `qcute/tokenizer.py` are self-contained modules — neither
imports the other, deliberately not factored further yet. Full details,
including which handover-doc section each component implements and known
gaps vs. the design: [docs/architecture.md](docs/architecture.md).

Design source of truth: [docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
Phase-by-phase progress: [docs/status.md](docs/status.md).
