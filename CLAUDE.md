# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                         # install/update env from pyproject.toml + uv.lock
uv run python scripts/prepare_data.py           # download/cut datasets/enwik8{,_1M}.gz
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz   # BPE tokenizer for qcute.bpelm
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.archive.qcutelm          # end-to-end tokenizer + latent LM (FSQ/BSQ) — ARCHIVED,
                                                 # superseded by qcute_refine (see Architecture below); still
                                                 # importable/runnable via its archive path for historical reference
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model   # BPE baseline
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py   # named, reproducible run — the
                                                 # standard byte-level baseline as of this session (context=1024,
                                                 # matching qcute_refine's own context_len); the older
                                                 # configs/bytelm_xs_mtp4.py (context=256) is superseded, kept only
                                                 # for historical reproducibility, not a comparison target anymore
uv run python scripts/plot_run.py logs/<run_name>   # train/val bpb PNG from a run's run.jsonl
uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_byte4_code256_simple.py   # current
                                                 # best qcute prototype — "v1" of the qcute_refine lineage's
                                                 # own best-so-far config (see Architecture below)
```

All three modules read `--help` for their full flag list; all support
`--config path.py` (see `configs/`), `--run_name` (else derived from
`--config`/preset — logs and checkpoints both key off it: `logs/<run_name>/`,
`checkpoints/<run_name>/`), and `--eval_only --checkpoint_path ...`;
`qcute.bytelm`/`qcute.qcutelm` additionally support `--qual_gen_bytes` for
qualitative generation. Tiny-corpus-scale defaults (`xs` preset, `qcutelm`'s
`K`) target ~4 bytes/timestep, not the handover doc's 8 — see
`qcute/bytelm.py`'s `PRESETS` comment for why. No test suite, linter, or CI
config exists yet.

**Only ever run one training job at a time.** All three modules train on
MPS; two concurrent training processes contend for the same GPU and both
slow down (observed directly: a second run caused an already-progressing
job to stall with zero throughput). Kill or wait out the current run
before launching another — never launch a second training process while
one is still active.

**When launching a training run in the background, never redirect its
stdout/stderr to `/dev/null`** — use a file instead (e.g. a scratchpad
path, or `/tmp/<pid>.log` renamed once the PID is known post-launch),
since `/dev/null` silently swallows uncaught-exception tracebacks and
anything not routed through `Logger`, making crashes invisible. Pipe
through `tr '\r' '\n'` before the redirect (`... 2>&1 | tr '\r' '\n' >
/tmp/foo.log &`) — tqdm's progress bar uses `\r` for in-place updates,
which lands as one giant unreadable line in a plain file otherwise; `tr`
turns each update into its own readable line (e.g. `loss=2.1512`). Note
`$!` after a pipe gives the last stage's PID (`tr`), not Python's — use
`pgrep -f "python3 -m qcute.<module>"` to find the actual training
process if you need its PID (e.g. to kill it). **After launching, give
the user two `tail -f` commands**: one on that raw stdout/stderr file,
and one on `logs/<run_name>/run.log` (the structured log `Logger` writes
to at `--log_every`/`--eval_every` intervals) — so they can watch it live
themselves rather than relying on being told the outcome later. Long runs
have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

## Architecture

`qcute/bytelm.py`, `qcute/qcutelm.py`, and `qcute/bpelm.py` are self-contained
modules — none import each other, deliberately not factored further yet.
Full details, including which handover-doc section each component
implements and known gaps vs. the design: [docs/architecture.md](docs/architecture.md).

**`qcute/qcute_refine.py` and `qcute/qcute_refine_v2.py` are the current
active lineage** — considered the project's first success, as of the
session that built them. `qcute_refine.py` ("v1"): pure recursive NTP
tower with BSQ code hand-off between levels, plus a block-local
joint-chain-MTP detokenizer. `qcute_refine_v2.py`: the detokenizer
redesigned into a `DecoderLevel` that cross-attends between adjacent
levels' own `EncoderLevel` hidden states (reused, not recomputed) instead
of running a separate self-attention pass — the actively-developed file,
with a growing set of session-driven flags (`byte_repr`, `code_head_mode`,
`bit_head_class`, `cross_attn_rope`, `decoder_own_trunk`,
`decoder_kv_pass_through`/`decoder_q_pass_through`, `layer_warmup_steps`)
documented in its own `Config` dataclass. Configs live under
`configs/qcute_refine_v2_*.py` and `configs/v1_*.py`. Full narrative:
[docs/status.md](docs/status.md) (session-update sections, newest at the
bottom) — this lineage moves fast and status.md is the only place its
current state is tracked; CLAUDE.md intentionally doesn't duplicate it.

Every earlier qcute-lineage fork — `qcutelm.py`, `qcutelm_vlt*.py`
(`vlt` through `vlt11`), `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py` — is **archived** under `qcute/archive/` (configs
under `configs/archive/`), superseded by the `qcute_refine` lineage.
Kept for historical reference/reproducibility, not part of active work;
`docs/status.md`'s own history of them is untouched. `qcute/bytelm.py`
and `qcute/bpelm.py` are the exception — still the active baseline
comparison points, not archived.

Design source of truth: [docs/continuous_tokenizer_handover.md](docs/continuous_tokenizer_handover.md).
Phase-by-phase progress: [docs/status.md](docs/status.md).

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).
