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
uv run python -m qcute.qcute_refine --config configs/qcute_refine_v4_pq.py   # current best qcute
                                                 # prototype — `qcute.qcute_refine` (no version suffix)
                                                 # always aliases the latest qcute_refine_vN.py, currently
                                                 # v4 (see Architecture below)
```

`qcute.bytelm`, `qcute.bpelm`, and `qcute.qcute_refine` (the active
lineage's own always-latest alias) all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.

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

`qcute/bytelm.py` and `qcute/bpelm.py` are self-contained baseline
modules — none import each other, deliberately not factored further yet.
Full details: [docs/architecture.md](docs/architecture.md) (also covers
`qcutelm.py`, now archived — see below).

**`qcute/qcute_refine_v1.py` through `qcute/qcute_refine_v4.py` are the
active lineage; `qcute/qcute_refine.py` (no version suffix) always
aliases the latest one** — a thin `from qcute.qcute_refine_v4 import *`
re-export, so `uv run python -m qcute.qcute_refine --config ...` always
runs "whatever's current" without needing to track the version number by
hand. Promoting a new version is a one-line change to that alias file.
Historical/comparison work should still target a specific `qcute_refine_
vN.py` directly (every config's own docstring already does this).

- `qcute_refine_v1.py`: pure recursive NTP tower with BSQ code hand-off
  between levels, plus a block-local joint-chain-MTP detokenizer; math:
  [docs/qcute_refine_math.md](docs/qcute_refine_math.md).
- `qcute_refine_v2.py`: the detokenizer redesigned into a `DecoderLevel`
  that cross-attends between adjacent levels' own `EncoderLevel` hidden
  states (reused, not recomputed) instead of running a separate
  self-attention pass. Session-driven flags (`byte_repr`, `code_head_mode`,
  `bit_head_class` with `BitPredictHeadAttn`/`Conv`/`SSM` variants,
  `cross_attn_rope`, `decoder_own_trunk`, `decoder_kv_pass_through`/
  `decoder_q_pass_through`, `code_embed_mode` with `linear`/`mlp`/
  `pq_table`, `layer_warmup_steps`) documented in its own `Config`
  dataclass. Configs live under `configs/qcute_refine_v2_*.py` and
  `configs/qcute_refine_*.py`.
- `qcute_refine_v3.py`: adds EncoderLevel **fusion** — a second forward
  pass per non-top level that cross-attends to the level-above's own
  hidden state *before* its own self-attention runs (`Config.
  fuse_encoder_levels`, `fuse_position` for pre/post self-attention
  ordering), so `byte_loss`/`val_bpb` can finally depend on the coarser
  code — v2's `DecoderLevel` cross-attention never could (its reads were
  detached, a separate scalar `tok_loss`, never compared against
  baselines). `DecoderLevel` still present, unchanged from v2. Configs:
  `configs/qcute_refine_v3_*.py`.
- `qcute_refine_v4.py` (**current alias target**): removes `DecoderLevel`
  entirely — it turned out to do the literal same job as fusion (predict
  a level's own next token, optionally conditioned on the coarser code)
  and never contributed to `byte_loss` even in v3. `EncoderLevel` renamed
  to `LevelLM` (the class now does both the "encode" job — PASS 1,
  produce the code — and the "decode" job — PASS 2, fused/conditioned
  prediction — so "Encoder" alone was misleading). Generation
  (`generate_no_cache`/`generate_kv_cache`) is fusion-aware — v3's own
  generation functions were copied unchanged from v2 and never touched
  cross-attention at all, a real train/inference mismatch v4 fixes.
  `qcute.bytelm`/`qcute.bpelm` also gained matching `generate_kv_cache`/
  `validate_generation` this session, same pattern. Configs:
  `configs/qcute_refine_v4_*.py`.

Diagnostic: `scripts/probe_decoder_kv_contribution.py` (gradient/
ablation/attention-mass analysis of how much cross-attention KV actually
contributes vs. is ignored — run against a saved v2/v3 checkpoint; v4 has
no `DecoderLevel` for this script to target). Full narrative:
[docs/status.md](docs/status.md), [docs/kv_contribution.md](docs/kv_contribution.md),
[docs/bpe_like_boundaries.md](docs/bpe_like_boundaries.md) (session-update
sections, newest at the bottom) — this lineage moves fast and these docs
are the only place its current state is tracked; CLAUDE.md intentionally
doesn't duplicate them.

Every earlier qcute-lineage fork — `qcutelm.py`, `qcutelm_vlt*.py`
(`vlt` through `vlt11`), `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py` — is **archived** under `qcute/archive/` (configs
under `configs/archive/`; their own design docs —
`continuous_tokenizer_handover.md`, `fifo_v2.md`, `vlt12_math.tex` — under
`docs/archive/`), superseded by the `qcute_refine` lineage. Kept for
historical reference/reproducibility, not part of active work;
`docs/status.md`'s own history of them is untouched. `qcute/bytelm.py`
and `qcute/bpelm.py` are the exception — still the active baseline
comparison points, not archived.

Original design source-of-truth for the now-archived lineage:
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md)
(historical — `qcute_refine`'s own design isn't specified by it).
Phase-by-phase progress: [docs/status.md](docs/status.md).

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).
