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
uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1.py
                                                 # ACTIVE lineage: qcute_v1 (qcute/qcute_v1/) is where the
                                                 # latent-AR / parallel-block-local-decode rewrite (see below)
                                                 # is actually implemented -- forked from a verbatim copy of
                                                 # qcute_v5, diverging as of the BOS-interleaved-decode rewrite.
                                                 # Full design doc: docs/qcute_v1_plan.md.
uv run python -m qcute.v5_old.qcute_v5 --decoder_type stack --config configs/v5_stack_fsq/ks1_16x8.py
                                                 # ARCHIVED: qcute_v5 (formerly qcute/qcute_v5*.py, moved to
                                                 # qcute/v5_old/ once qcute_v1 became the active lineage) --
                                                 # still the leaderboard's source of truth for everything
                                                 # already run (docs/status.md), frozen, not receiving further
                                                 # architecture work. `--decoder_type concat` also still works.
```

`qcute.bytelm`, `qcute.bpelm`, `qcute.qcute_v1.qcute_v1`, and
`qcute.v5_old.qcute_v5` all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.
Existing `configs/v5_stack_*/`, `configs/v5_word/`, etc. docstrings still say
`qcute.qcute_v5`/`qcute.qcute_v5_wordlm` (pre-move path) in their copy-pasteable
`uv run` line — substitute `qcute.v5_old.qcute_v5`/`qcute.v5_old.qcute_v5_wordlm`
when actually running them; left as-is rather than bulk-edited, matching how
other archived lineages' configs keep their original invocation text.

**Only ever run one training job at a time.** All four modules train on
MPS; two concurrent training processes contend for the same GPU and both
slow down (observed directly: a second run caused an already-progressing
job to stall with zero throughput). Kill or wait out the current run
before launching another — never launch a second training process while
one is still active.

**When launching a training run in the background, never redirect its
stdout/stderr to `/dev/null`** — use a file instead (e.g. a scratchpad
path, or `/tmp/<pid>.log` renamed once the PID is known post-launch),
since `/dev/null` silently swallows uncaught-exception tracebacks and
anything not routed through `Logger`, making crashes invisible. Do NOT
pipe through `tr '\r' '\n'` to make tqdm's `\r`-updates readable —
confirmed directly that `tr` itself full-block-buffers its own stdout
when writing to a non-tty file, so a `tail -f` on the piped-through file
sits empty for seconds/minutes at a time regardless of how eagerly the
Python process flushes its side of the pipe; it's not a live view, just a
deferred dump. Redirect stdout/stderr straight to a file instead (plain
`... > /tmp/foo.log 2>&1 &`, no pipe) — the tqdm line will look like one
long `\r`-joined blob when catted, but `tail -f` still shows new bytes
arriving in real time, which is the actual goal. Use `pgrep -f "python3
-m qcute.<module>"` to find the training process's PID (e.g. to kill it;
`$!` after a background launch gives the wrapper/shell PID, not
necessarily Python's). **After launching, give the user the PID and two
`tail -f` commands**: one on that raw stdout/stderr file, and one on
`logs/<run_name>/run.log` (the structured log `Logger` writes to at
`--log_every`/`--eval_every` intervals, genuinely real-time since
`Logger` opens and flushes that file directly, no pipe involved) — so
they can watch it live themselves rather than relying on being told the
outcome later. Long runs have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

## Architecture

**`qcute_v1` (`qcute/qcute_v1/`) is the active lineage as of 2026-08-20** — forked from a verbatim
copy of `qcute_v5` (now archived at `qcute/v5_old/`, see Commands above), implementing the
latent-AR / parallel-block-local-decode rewrite: only the top level stays a genuine NTP/AR
decoder; every level below decodes via per-block-seed-token-interleaved self-attention (a
trainable seed token — neither a true single "BOS" (it recurs every block) nor a passive "sink"
(it's a full token, not a fallback key that itself predicts nothing) — prepended before every
`K`-block, not a recurrent self-code chain) plus cross-attention to that same block's own-level
code, with the seed token's own hidden state genuinely reconstructing that block's own first byte
from that block's own code (`own_block_cross_attn_decode`/`own_block_decode_loss` in
`qcute_v1_decoder.py`, fixed
2026-08-20 — an earlier version silently never let a block's own code inform its own
reconstruction, see `docs/status.md`'s session log). Full design narrative, worked examples, and
the staged plan: [docs/qcute_v1_plan.md](docs/qcute_v1_plan.md). Progress/results log:
[docs/status.md](docs/status.md) (reset at the v1 transition — full v5-era history at
[docs/archive4/status.md](docs/archive4/status.md), [docs/archive3/status.md](docs/archive3/status.md),
[docs/archive2/status.md](docs/archive2/status.md)).

**TODO for a fresh session**: `configs/v1_stack_simplex/ks21_v256_pq1.py` and `ks21_v64_pq4.py`
need re-running (full-scale, `--decoder_type stack`) — their existing `best_val_bpb` numbers in
`docs/status.md` predate the own-block-reconstruction fix above and are stale. `ks1_*` configs
(`Ks=(1,)`, no non-top level) are unaffected and don't need re-running.

The rest of this section (below) describes `qcute_v5`'s own architecture — still accurate for
that (frozen, archived) lineage, kept for reference when working in `qcute/v5_old/`.

`qcute_v5_concat.py` (from v4.4.1's packed self-attention decode) and
`qcute_v5_stack.py` (from v4.5.1's staged cross-attention decode) add
`Config.quant_type: "softmax"` (default, unchanged categorical code) or
`"bsq"` (binary spherical quantization, `Config.bsq_bits`-wide sign
code, straight-through) as an alternative to the categorical code
representation. Full detail on all of the above, including an ongoing
slow-convergence investigation (a moving-target/cascade effect for
n_levels=2 configs, compounded in the "stack"/v4.5-style decode by a
much deeper gradient path back to the code producer — decode is now
`n_layers * (1 + n_tracks)` deep sequentially, vs. the "concat"/v4.4-style
decode's flat `n_layers`), and the `decode_code_ste` findings (predates
`qcute_v5.py`/current `qcute_v5_concat.py` hardcoding `decode_code_ste`
to always `True` — see above): [docs/archive4/status.md](docs/archive4/status.md).

**Standing methodology**: use a small (`n_bytes=10000`) slice of the
corpus with a short step budget as the standard fast-iteration testbed
for architecture changes (v5 or v1) — see `configs/*_overfit10k_*.py` — until a config can actually fast-overfit
that slice to a train bpb comparable to `qcute.bytelm`'s own parity
numbers on the same slice (`n_layers=1`: 0.0212 train bpb at step 1000,
~19.7 it/s; `n_layers=2`: 0.0072, ~11.4 it/s — see
`configs/bytelm_overfit10k_*.py`). Full-scale (~900k-byte) runs and
generation-quality comparisons are not trustworthy until that parity
bar is cleared — a config that hasn't even memorized a 10k-byte slice
yet tells you little about its behavior at scale.

**Ks regression grid, simplest to hardest** (for config-writing later): ranked by
`product(Ks)` (compression ratio / minimum warm-up context) first, `n_levels` second,
`max(Ks)` third.

| # | Ks | levels | product(Ks) | max K |
|---|---|---|---|---|
| 1 | `(1,)` | 1 | 1 | 1 |
| 2 | `(1,1)` | 2 | 1 | 1 |
| 3 | `(1,1,1)` | 3 | 1 | 1 |
| 4 | `(2,1)` | 2 | 2 | 2 |
| 5 | `(2,1,1)` | 3 | 2 | 2 |
| 6 | `(2,2)` | 2 | 4 | 2 |
| 7 | `(4,1)` | 2 | 4 | 4 |
| 8 | `(2,2,1)` | 3 | 4 | 2 |
| 9 | `(4,2)` | 2 | 8 | 4 |
| 10 | `(2,2,2)` | 3 | 8 | 2 |
| 11 | `(4,2,1)` | 3 | 8 | 4 |
| 12 | `(4,4,2)` | 3 | 32 | 4 |

Ranks generation/architecture-correctness difficulty (raggedness, warm-up depth), not
training/learnability difficulty — a high-product config may be easier to overfit10k
(fewer effective tokens) despite being harder to verify `check_gen_consistency` on.

**Latent-AR / parallel-block-local-decode investigation**: originated here as a `qcute_v5`
window-ablation/track-dropout curriculum plan (2026-08-19); superseded by the `qcute_v1` rewrite
(2026-08-20, see top of this section) once the investigation concluded that plan's `code_window`/
lag-`D` knobs were redundant with `attn_window` and that the real fix was retargeting decode's
loss (NTP-next-block -> NTP-autoencode) plus a per-block BOS, not a track-sparsity curriculum.
Full narrative: [docs/qcute_v1_plan.md](docs/qcute_v1_plan.md).

Diagnostic: `scripts/probe_decoder_kv_contribution.py` (gradient/
ablation/attention-mass analysis of how much cross-attention KV actually
contributes vs. is ignored — run against a saved v2/v3 checkpoint, now
under `qcute.archive2.qcute_refine_v2`; v4-and-later have no
`DecoderLevel` for this script to target). Historical narrative on the
pre-v4 lineage (v2's `DecoderLevel` KV contribution, boundary-adaptivity
brainstorm, `BitPredictHead` speed comparisons, `torch.compile` on v2):
[docs/archive2/kv_contribution.md](docs/archive2/kv_contribution.md),
[docs/archive2/bpe_like_boundaries.md](docs/archive2/bpe_like_boundaries.md),
[docs/archive2/bitpredict_heads.md](docs/archive2/bitpredict_heads.md),
[docs/archive2/torch_compile.md](docs/archive2/torch_compile.md).

Several older one-off scripts (`scripts/compare_ste_vs_detach.py`,
`probe_code_usage_entropy.py`, `probe_v4_fusion_contribution.py`,
`qual_gen_v4_2.py`, `qual_gen_v4_3_vs_baseline.py`,
`test_v4_4_banded_decode.py`, `test_v4_4_chunked_decode.py`,
`qual_gen_bytelm.py`) still `import` archived `qcute_refine_vN` modules
by their pre-archive top-level path (e.g. `from qcute.qcute_refine_v4_4
import ...` instead of `from qcute.archive2.qcute_refine_v4_4 import
...`) and will `ImportError` if run as-is — not part of the two
maintained diagnostics above, left unfixed since nothing currently
depends on them; fix their import path the same way if you need to
revive one.

Every earlier qcute-lineage fork — `qcutelm.py`, `qcutelm_vlt*.py`
(`vlt` through `vlt11`), `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py` — is **archived** under `qcute/archive/` (configs
under `configs/archive/`; their own design docs —
`continuous_tokenizer_handover.md`, `fifo_v2.md`, `vlt12_math.tex` — under
`docs/archive/`), superseded by the `qcute_refine` lineage. The
`qcute_refine` lineage itself (`v1.py` through `v4_5_1.py`, the pre-v5
history) is separately archived under `qcute/archive2/` — see above — a
different archive directory since it's a later, distinct lineage from
the `qcutelm.py` family. Kept for historical reference/reproducibility,
not part of active work. `qcute/bytelm.py` and `qcute/bpelm.py` are the
exception — still the active baseline comparison points, not archived.

Original design source-of-truth for the now-archived `qcutelm` lineage:
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md)
(historical — `qcute_refine`'s own design isn't specified by it). Chronological one-line summary
of every fork in both the `qcutelm` (`qcute/archive/`) and `qcute_refine` (`qcute/archive2/`)
lineages: [docs/archive/lineage_summary.md](docs/archive/lineage_summary.md).
Current progress: [docs/status.md](docs/status.md).

## Code style

Docstrings and comments must be extremely concise — assume code is self-descriptive; comments
never exceed 2 lines. Only the module-level (top-of-file) docstring is exempt from the length
limit, and should still be kept reasonably tight rather than accumulating restated history.

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).

Keep chat replies terse. State results and next steps directly — no restating context the user
already has, no padding, no multi-paragraph recaps. Save detail for docs/status.md, not the chat.
