# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                     # install/update env from pyproject.toml + uv.lock
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.qcutelm            # end-to-end tokenizer + latent LM (FSQ/BSQ)
```

Both modules read `--help` for their full flag list. No test suite, linter, or
CI config exists yet.

## Architecture

`qcute/bytelm.py` and `qcute/qcutelm.py` are self-contained modules — neither
imports the other, deliberately not factored further yet. Full details,
including which handover-doc section each component implements and known
gaps vs. the design: [docs/architecture.md](docs/architecture.md).

Design source of truth: [docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
Phase-by-phase progress: [docs/status.md](docs/status.md).

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).
