# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                         # install/update env from pyproject.toml + uv.lock
uv run python scripts/prepare_data.py           # download/cut datasets/enwik8{,_tiny}.gz
uv run python scripts/train_bpe.py --data datasets/enwik8_tiny.gz   # BPE tokenizer for qcute.bpelm
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.qcutelm                  # end-to-end tokenizer + latent LM (FSQ/BSQ)
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_tiny_8192.model   # BPE baseline
uv run python -m qcute.bytelm --config configs/bytelm_xs_tiny_longrun.py   # named, reproducible run
```

All three modules read `--help` for their full flag list; all support
`--config path.py` (see `configs/`), checkpointing to `checkpoints/` (best +
last), and `--eval_only --checkpoint_path ...`; `qcute.bytelm`/`qcute.qcutelm`
additionally support `--qual_gen_bytes` for qualitative generation. No test
suite, linter, or CI config exists yet.

## Architecture

`qcute/bytelm.py`, `qcute/qcutelm.py`, and `qcute/bpelm.py` are self-contained
modules — none import each other, deliberately not factored further yet.
Full details, including which handover-doc section each component
implements and known gaps vs. the design: [docs/architecture.md](docs/architecture.md).

Design source of truth: [docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
Phase-by-phase progress: [docs/status.md](docs/status.md).

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).
