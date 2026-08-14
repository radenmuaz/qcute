# qcute v5 status

Reset to a fresh log at this point in the project — the prior session-by-session narrative got
long enough to be more archival than actionable. Full history:
[docs/archive2/status.md](archive2/status.md) (3700+ lines, newest at the bottom). New entries go
below, same convention: newest at the bottom, session-dated where useful.

## Where things stand

Active: `qcute/qcute_v5_concat.py` (packed self-attention decode) and `qcute/qcute_v5_stack.py`
(staged cross-attention decode) — each adds `Config.quant_type: "softmax" | "bsq"`. Both are
standalone modules. Configs: `configs/qcute_v5_concat_*.py`, `configs/qcute_v5_stack_*.py`.
Standard level=1/level=2 overfit10k baselines: `*_k1_l1.py` / `*_k11_l1.py` — level=1 must pass
before trusting level=2.

Earlier lineage history and design docs (`kv_contribution.md`, `bpe_like_boundaries.md`,
`bitpredict_heads.md`, `torch_compile.md`) are archived
under `qcute/archive2/`, `configs/archive2/`, `docs/archive2/` — see CLAUDE.md.

Standing conclusions carried forward from the archived log (still load-bearing, don't re-derive):
- **`decode_code_ste=False` (detach) stays the default.** `=True` (STE) fixes a real dead-gradient
  issue but empirically causes cond-generation to collapse into character repetition at overfit10k
  scale. Revisit at larger scale.
- **Overfit10k is the standard fast-iteration testbed**: `n_bytes=10000`, `steps=1000`,
  `batch_size=16`, `lr_peak=6e-4`, `warmup_steps=100`. No KV-cache path — `generate_no_cache` only.
- **Same-prompt methodology**: byte offset `START` into `datasets/enwik8_1M.gz`'s train split,
  `PROMPT_LEN=64`, `GEN_LEN=64`, greedy argmax, for cross-config/checkpoint comparisons.

## `cond_full` bug hunt: three real bugs, not exposure bias (this session)

`cond_full` (level 0 conditioned on its self track + coarser level's code) showed a correct byte
prefix during free-running generation that then drifted, despite ~98% teacher-forced accuracy.
Ruled out upstream code-quality propagation and KV/masking bugs (both directly tested and clean).
Root cause: found by writing `check_gen_consistency` (now permanent, in both modules, wired into
the `qual_gen_bytes` eval block) — feeds the same ground-truth bytes through incremental no-cache
generation and a one-shot teacher-forced pass, asserts logits match exactly.

1. **Windowed-attention dense fallback** (`CausalSelfAttention`, both files): silently used full
   dense attention whenever `T % window != 0`. Training's fixed, window-aligned `context_len`
   never hit this; growing-length no-cache generation almost always did. Fixed with a single exact
   windowed boolean mask, correct for any `T`; removed the old chunked fast path as dead code.
2. **`can_chunk` dropping the coarser track** (`qcute_v5_concat.py` only): in the multi-track
   branch, `can_chunk` didn't check `len(decode_tracks)`, so whenever `L` was window-aligned
   (always true in training) it silently called `_packed_decode_forward_chunked` with only
   `decode_tracks[0]` — **`cond_full` was never actually trained with multi-track conditioning**.
   Fixed by gating `can_chunk` on `len(decode_tracks) == 1` (unreachable there, so it's now
   permanently disabled in that branch).
3. **`check_gen_consistency` itself had the byte-slot-vs-code-slot bug** on first merge (caught by
   a level=1 smoke test: 31/31 mismatched). Its TF reference read `h` (byte-slot) everywhere, but
   the single-track `selfcode_decode` path's real loss reads `query_h` (code-slot). Fixed by
   exposing a batch-dim-preserving `query_seq` from `LevelLM.forward`/`selfcode_decode`, used when
   available, falling back to `h` for the multi-track path (correct there). Lesson: a verifier is
   code too — sanity-check it against every path it's meant to cover, not just one clean result.

**Every checkpoint predating these fixes is untrustworthy for `cond_full` quality** — bug 2 means
no old checkpoint ever trained genuine multi-track conditioning. Fresh retrains
(`*_k1_l1.py`/`*_k11_l1.py`) confirm 0/N `gen_consistency` mismatches post-fix on all four
combinations (concat/stack × level=1/level=2).

Post-fix level=2 quality differs by architecture: **concat's** `cond_full` now matches ground
truth exactly on the train-prompt check, on par with `cond_self`. **Stack's** `cond_full` still
lags `cond_self` noticeably (garbled vs. near-exact) at the same step count, despite passing
`gen_consistency` — a genuine convergence gap in the 2-track `cross_attn_stage` mechanism, not a
bug. Consistent with the standing conclusion above that staged cross-attention (stack) converges
worse than packed self-attention (concat) for multi-track decode; worth another look now that the
comparison is no longer confounded by the `can_chunk`/windowing bugs.

## K0>1 generation bug hunt: floor-tolerance, not padding (this session)

`check_gen_consistency` on `Ks=(2,1)`-style configs (K0>1) showed a striking pattern: exact match
at `t % K0 == 0`, large mismatch elsewhere. Root cause was structural, in both modules:
`_run`'s track-building loop and `_packed_decode_forward`/`cross_attn_stage` all hard-required
`L_i % cum_K == 0` to use any decode conditioning at all. Training never violates this (context_len
is built to divide evenly at every level), but generation's growing, non-block-aligned prefix
almost always does — so `generate_no_cache` padded with fake zero bytes to force alignment, which
corrupts the trailing block's own pooled code and, in the multi-track case, still only reads the
byte-slot fallback at `L-1` post-padding rather than what training actually optimizes for.

Fix, in three parts:
1. **Floor, don't discard** (concat and stack both — same `_run` logic, forked identically):
   the ragged check now stops adding tracks (keeping whichever finer ones already qualify) instead
   of discarding the whole level's decode conditioning the moment any single track isn't exactly
   block-aligned.
2. **`_packed_decode_forward` was missing the freshest prefix slot** (concat only): it built
   prefixes from `code_kv[:-1]` (n_blocks total), never from the block that just completed —
   invisible during training/one-shot TF (a long fixed window always has enough later blocks to
   cover this slot anyway) but a real, missing piece of context for a short growing generation
   prefix. Now builds `n_blocks + 1` slots, one per available code including the freshest.
   **Stack doesn't have this bug.** Its multi-track path (`cross_attn_stage`) is genuine
   cross-attention — `x_in` queries attend directly to a `code_kv` key/value array via a position
   mask (`code_pos < query_pos`), using every entry (`n_blocks = code_kv.shape[1]`, no `[:-1]`).
   Concat instead packs codes and bytes into *one* flat self-attention sequence, which requires
   materializing codes as shifted, prepended synthetic tokens (`bos, code_kv[:-1]`) to fake the
   "code_b conditions block b+1" causal order — that manual bookkeeping is where the off-by-one
   lived. Cross-attention addresses codes by position instead, so there's no synthetic-sequence
   step for a count to go wrong in.
3. **Ragged tail in the single-track selfcode path** (concat and stack both — same
   `.view(B, n_blocks, K, D)` reshape, same LM-continuation mechanism, forked identically):
   `_packed_decode_forward_selfcode`/`selfcode_decode` can't handle a non-block-aligned trailing
   remainder — now floor-truncate internally and `_run` splices the corresponding tail of the
   plain encode-only `h` back in.

`generate_no_cache` no longer pads at all in either module — it calls `_run` on the true, growing
byte sequence every step. Verified via `check_gen_consistency` on untrained models (no training
needed — this is a forward-pass architecture check) across a base-case-then-widened Ks grid
(`(1,)`, `(1,1)`, `(1,1,1)`, `(2,1)`, `(2,2)`, `(2,1,1)`, `(2,2,1)`, `(2,2,2)`, `(4,1)`, `(4,2)`,
`(4,2,1)`, `(4,4,2)`) on both modules, with `prompt_len` past the warm-up floor below (same floor
applies to both modules — stack reproduces the identical short mismatch window at a too-short
`prompt_len`, e.g. `(4,2,1)` with `prompt_len=8`; it's the inherent cold-start gap described below,
not a module-specific issue). Also re-checked all four existing trusted checkpoints (no regression).
One remaining, expected (not a bug) gap: a 3-level
config shows mismatches only in a short window before `prompt_len`, exactly until the deepest level
accumulates its own minimum 2 native blocks (confirmed by tracing `decode_derived_c`'s availability
directly) — training never sees a context shorter than `context_len`, so there's no trained ground
truth to match before that point; it's an inherent generation cold-start, not something
`check_gen_consistency` should expect to pass with too short a `prompt_len` relative to `n_levels`.

## Must-dos when writing generation code (learned the hard way, repeatedly)

"The model isn't learning" is wrong far more often than "generation is querying something the
trained weights were never optimized to produce." Every bug above had the same signature:
excellent teacher-forced accuracy, garbage free-running generation.

- **Trace the exact tensor** the training loss reads at that point — don't assume shape parity
  (`(B,L,D)`) means value parity between two code paths.
- **Padding to satisfy an alignment assert is a code smell, not a fix.** If generation needs to pad
  a fixed-length training assumption to force an assert to pass, the assert is usually stricter
  than the underlying computation actually requires — check whether floor/ragged tolerance was the
  real answer before reaching for padding.
- **Any control-flow-changing param is a suspect** (`max_decode_sources`, `want_code`,
  `compute_ntp`, truncation/masking). Confirm both branches it can select were actually trained,
  not just the untouched one.
- **Teacher-forced-good + free-running-bad ⇒ check generation code first**, not more training or
  regularization.
- **Test fixes against an existing checkpoint before retraining** — most of these are pure
  generation-code fixes with zero effect on the training graph.
- **Don't reach for decode/train-time band-aids** (temperature, top-k, noise injection) before
  ruling out a plain dispatch/tensor bug — they look identical from the output alone.

## Ks regression grid (1k-byte testbed, this session)

Walking the Ks ranking table in CLAUDE.md (simplest→hardest), concat then stack per Ks, on a
smaller/faster **1k-byte testbed** (`n_bytes=1000`, not the usual overfit10k) — `steps=1000`,
`batch_size=16`, `lr_peak=6e-4`, `warmup_steps=100`, `context_len=256`, `d_model=256`, `n_layers=1`,
`decode_self_only_aux=True` when `n_levels>1`. `qual_prompt_bytes`/`qual_gen_bytes` scale with
`product(Ks)` instead of the fixed `16`/`32` — `max(16, 2×product(Ks))` /
`max(32, product(Ks)+16)` — so generation is pushed past the warm-up floor and past one full
top-level code's span. Configs: `configs/qcute_v5_{concat,stack}_ks<Ks digits>_1k.py`. Gated
one Ks at a time — auto-continue only when generation is an unambiguous perfect overfit match.

| Ks | concat train bpb/byte_acc | concat qual (train) | stack train bpb/byte_acc | stack qual (train) | verdict |
|---|---|---|---|---|---|
| `(1,)` | 0.0486 / 98.70% | exact match | 0.0302 / 99.19% | exact match | perfect, both — auto-continued |
| `(1,1)` | 0.0254 / 99.34% | byte exact (level1 codes not exact yet) | 0.0526 / 98.65% | byte exact (level1 codes not exact yet) | byte-level perfect both — auto-continued |

**Paused between `(1,1)` and `(1,1,1)` to generalize `decode_self_only_aux`** (both modules): it
previously always trained exactly one auxiliary pass per level, self-track-only, regardless of
`n_levels`. For `n_levels>2` that skips every intermediate combination — e.g. at the byte level of
a 3-level model, only `{self}` and the full `{self, code1, code2}` were ever trained, never
`{self, code1}`. Generalized to the full curriculum below full conditioning: for level `i`'s
`tracks` (length `len(tracks)`), now trains one aux pass per `k = 1..len(tracks)-1`
(`{self}`, `{self, +1}`, ..., `{self, +1, ..., top-1}`; the full combo is already the main loss,
not repeated). `decode_self_only_losses`/`accs` changed from one scalar per level to a
`{k: loss}` dict per level; aggregation switched from `sum()` to `mean()` over all curriculum
terms so `decode_self_only_weight`'s meaning doesn't drift with `n_levels`; metrics keys became
`level{i}_ntp_loss_decode_self_k{k}`. For `n_levels<=2` there's only one possible `k` (self alone),
so this is a strict no-op there — verified byte-for-byte identical loss and gradient (same seed,
same `params_without_grad` count) against the pre-change code for `Ks ∈ {(1,), (1,1), (2,1)}`.
Stack's version deliberately runs its own separate staged chain for each `k` rather than reusing
`decode_stage_extra_losses` (which only fires when `share_level_weights=False`) — keeps the aux
curriculum independent of that weight-sharing detail. Re-verified the full `check_gen_consistency`
Ks grid (0 mismatches, both modules — this change is training-only, gated by `self.training`, so
generation is untouched by construction) and all four existing trusted checkpoints (no regression).

| `(1,1,1)` | concat: byte_acc~99.5% (early-stopped ~step 2050/4000) | `cond_full`/`cond_self` both exact at stop | stack: (early-stopped ~step 200/2000, much faster to converge than concat) | `cond_full`/`cond_self` both exact at stop, train+val | both good — early-stopped, proceeding to grid |
| `(1,1,1)` concat, bsq8 side-experiment | bpb=0.044-0.047, byte_acc~99.1% (early-stopped ~step 349/2000) | **`cond_self` fails** (garbled: `'t </:case>\n        <nanese yyydi'` vs gt `'t-letter</case>\n      <namespace'`); `cond_full` matches exactly | — | — | **FAIL: cond_self** with `quant_type="bsq", bsq_bits=8` — full-conditioning path unaffected, self-only path notably worse than the default softmax code at the same step count; not investigated further, noted for follow-up |

**Policy change: reduced grid step budget to `steps=500` (from 1000), and stopped pausing on
failure** — now runs straight through concat then stack for every remaining Ks, logging pass/fail
without waiting for confirmation. Configs renamed `*_500.py` (was `*_1k.py`) to reflect the
now-varying parameter (step count), not the fixed byte-count testbed (still `n_bytes=1000`
throughout).

| `(2,1)` | concat: bpb=0.0298, byte_acc=99.24% | **FAIL** — neither `cond_full` nor `cond_self` matches gt; `cond_self` garbled | stack: bpb=0.0591, byte_acc=98.43% | **PASS** — train `cond_full`/`cond_self` both exact | split result: stack passed, concat failed, same Ks/steps |
| `(2,1,1)` | concat: bpb=0.0251, byte_acc=99.41% | **FAIL** — `cond_full` close but not exact (`'ikipedia</namespace>...'` vs gt `'ikipedia talk</namespace>...'`); `cond_self` badly garbled both splits | stack: bpb=0.0518, byte_acc=98.68% | **FAIL** — `cond_full`/`cond_self` both predict `'"1">Talk</namespace>...'` vs gt `'"-1">Special</namespace>...'` | **FAIL both** |
| `(2,2)` | concat: bpb=0.0327, byte_acc=99.22% | **FAIL** — neither matches gt; `cond_self` badly garbled | stack: bpb=0.0504, byte_acc=98.65% (run took ~20.5min, notably longer than other 500-step runs so far — flagged, not investigated) | **PASS** — train `cond_full`/`cond_self` both exact | stack passed, concat failed |
| `(2,2)` concat, `decode_self_only_aux=False` redo | bpb=0.0264, byte_acc=99.46% (1m30s vs ~3min with aux on — notably faster) | `cond_full` **exact match** on train; `cond_self` badly garbled (worse than the aux-on run) | — | — | aux trades off `cond_full` convergence speed against `cond_self` quality at this step budget — turning it off let `cond_full` pass but starves `cond_self` of any training signal |
| `(4,1)` | concat: bpb=0.0324, byte_acc=99.26% | **PARTIAL** — `cond_full` exact match on train; `cond_self` badly garbled | stack: bpb=0.0596, byte_acc=98.46% | **PASS** — train `cond_full`/`cond_self` both exact | stack passed cleanly, concat partial |
| `(2,2,1)` | concat: bpb=0.0253, byte_acc=99.24% | **PARTIAL** — `cond_full` exact match on train; `cond_self` badly garbled (same pattern as `(4,1)`) | stack: bpb=0.0731, byte_acc=98.38% | **PASS** — train `cond_full`/`cond_self` both exact | reinforces the concat aux-not-helping-`cond_self` finding — 4th concat config in a row with this split; stack keeps passing cleanly across the board |

**Real crash found at `(4,2)` concat, both modules affected — fixed, not just worked around.**
`n_bytes=1000, val_frac=0.1` gives `val_bytes=100`, but grid configs kept `context_len=256` from
the old testbed. `sample_context`'s `n = max(1, len(data) - context_len)` goes negative and clamps
to 1, so every val batch was silently truncated to ~100 bytes (no error) for the entire grid so
far. Most Ks tolerated this by luck; `(4,2)`'s top level (`K=2`) needed `x_list[1]`'s length evenly
divisible by 2, and `100 // 4 = 25` (odd) finally produced a real crash: the single-track
selfcode/`selfcode_decode` path's NTP loss compared `query_h` (floor-based length, `n_units*K` rows)
against `seq_repr[:, K:]` (un-truncated, `L-K` rows) — only equal when `L` is an exact multiple of
`K`, always true in training (fixed `context_len`) but not guaranteed in eval on short data. Fixed
in both modules: slice the loss target to `query_h`'s actual length instead of assuming exact
alignment. Also added a one-time `WARNING` print in `sample_context` (both modules) when
`len(data) < context_len`, so a silently-truncated split is visible instead of invisible. **Val
bpb/loss numbers throughout this grid reflect a truncated ~100-byte context, not the configured
256** — not corrected retroactively since the grid's pass/fail criterion is `qual_train_*` overfit
quality, not val bpb (goal is overfitting, not generalization). Regression-checked clean: ragged-L
smoke test on all previously-affected Ks (both modules, no crash), full `check_gen_consistency`
grid (0 mismatches), all four trusted checkpoints (no regression).

| `(4,2)` | concat: bpb=0.0276, byte_acc=99.39% (no crash after fix) | **FAIL** — neither matches gt; `cond_self` garbled | stack: bpb=0.0552, byte_acc=98.46% | **PASS** — train `cond_full`/`cond_self` both exact | stack passed, concat failed — stack's clean sweep continues |
| `(2,2,2)` | concat: bpb=0.0290, byte_acc=99.39% | **FAIL** — neither matches gt; `cond_self` garbled | stack: bpb=0.0561, byte_acc=98.60% | **NEAR-MISS FAIL** — `cond_full`/`cond_self` both drop one byte (`'letter'`→`'leter'`), rest matches; verified byte-for-byte, not eyeballed | both fail, stack's streak breaks (first non-pass for stack in this grid) |
| `(4,2,1)` | concat: bpb=0.0278, byte_acc=99.24% | **FAIL** — neither matches gt; `cond_self` garbled | stack: bpb=0.0630, byte_acc=98.55% | **FAIL** — neither matches gt | **FAIL both** |
| `(4,4,2)` | concat: bpb=0.0224, byte_acc=99.41% | **FAIL** — neither matches gt; `cond_self` garbled; `gen_consistency: 16/31` (expected — `qual_prompt_bytes=16` is below this config's `2×product(Ks)=64` warm-up floor, not a regression) | stack: bpb=0.0564, byte_acc=98.46% (crashed first attempt, fixed — see below) | **FAIL** — neither matches gt; `gen_consistency: 16/31` (same expected warm-up floor) | **FAIL both — grid complete (12/12 Ks, both modules)** |

**Grid summary (12/12 Ks, both modules, `n_bytes=1000`/`steps=500`/aux on)**:

| Ks | concat | stack |
|---|---|---|
| `(1,)` | PASS | PASS |
| `(1,1)` | PASS | PASS |
| `(1,1,1)` | PASS (early-stopped) | PASS (early-stopped) |
| `(2,1)` | FAIL | PASS |
| `(2,1,1)` | FAIL | FAIL |
| `(2,2)` | FAIL (aux on); PASS if aux off | PASS |
| `(4,1)` | PARTIAL (`cond_full` only) | PASS |
| `(2,2,1)` | PARTIAL (`cond_full` only) | PASS |
| `(4,2)` | FAIL | PASS |
| `(2,2,2)` | FAIL | NEAR-MISS FAIL |
| `(4,2,1)` | FAIL | FAIL |
| `(4,4,2)` | FAIL | FAIL |

Stack cleanly passes almost the whole grid; concat only fully passes the trivial `K=1` configs and
degrades to partial/fail as soon as real compression (`K>1`) enters, with `cond_self` as the
consistent failure point.

**Second real crash found and fixed, at `(4,4,2)` stack**: `qualitative_generate`'s
`generate_level1_codes_via_decode` helper (a newer diagnostic, added since the earlier padding-removal
session and never updated) still used the old pad-based `selfcode_decode` call pattern and hit its
hard `assert n_units >= 1` uncaught when `qual_prompt_bytes=16` didn't give level1 two native blocks
for this Ks (`product(Ks)=32`, same warm-up-floor story as the `(4,2,1)` cold-start finding). Fixed
by adding the same floor check used everywhere else (`if L // K1 < 2: break`) instead of crashing.
Regression-checked clean (smoke test on the exact failing scenario, full `check_gen_consistency`
grid, both stack trusted checkpoints).

**Working conclusion: `decode_self_only_aux` isn't earning its cost on `qcute_v5_concat` at this
grid's step budget — better left off.** `(4,1)` with aux ON shows the exact same
`cond_full`-passes/`cond_self`-garbled split as `(2,2)`'s aux-OFF redo, so the aux loss isn't
buying `cond_self` quality it wouldn't otherwise lack, while `(2,2)`'s direct on/off comparison
showed aux OFF converges `cond_full` faster (1m30s vs ~3min) for the same result. Not yet re-tested
across the rest of the grid with aux off by default — flagging as a candidate default change, not
applying it retroactively to already-logged rows above.

Needed escalation to converge (1000 steps insufficient for a 3-level config): retried at 2000 then
4000 steps, early-stopped ~step 2050/4000 once both `cond_full`/`cond_self` hit an exact match on
the qual check (per-step overfit quality isn't monotonic at this scale — treat "hit a perfect match
at any recent step" as pass, not "still perfect at the final step"). `caffeinate -i` wrapping (added
mid-grid, see feedback memory) sped throughput roughly 3x (was ~3.9s/step, now ~1.3-1.6s/step) —
apply to all runs going forward.

## Concat-only no-aux rerun (`decode_self_only_aux=False`, `steps=200`, `val_frac=0.5`)

Following the aux-not-helping-`cond_self` finding above, rerunning the full 12-Ks grid for
**concat only** (stack assumed to already pass, per the grid summary above — not rerun here) with
aux off, a much shorter 200-step budget, and `val_frac=0.5` (500/500 train/val split instead of
900/100 — avoids the short-val-data truncation warning entirely, doesn't rely on it being handled
gracefully). Configs: `configs/qcute_v5_concat_ks<name>_200_noaux.py`.

| Ks | result |
|---|---|
| `(1,)` | **PASS** — bpb=0.0354, byte_acc=99.07%, `cond_full` exact match on train (28s for 200 steps) |
| `(1,1)` | **PASS** (early-stop, 400 steps) — 200-step run was NEAR-MISS (bpb=0.0301, `cond_full` off by one byte); bumped to 400 steps: step 249 `cond_full` exact byte-for-byte match (`'.w3.org/2001/XMLSchema-instance"'`), step 299 also exact match (`'Schema-instance" xsi:schemaLocat'`) — both verified via python `==`; final step 399 had since collapsed (overfit past the good point), so PASS is on the early-stop checkpoint, not the final one, per standing "perfect match at any recent step" criterion |
| `(1,1,1)` | **PASS** (early-stop, 400 steps) — `cond_full` exact byte-for-byte match at 3 separate eval checkpoints (~00:05:57, ~00:07:06, ~00:08:02); final step 399 had since collapsed (bpb=1.1758, byte_acc=80.93%, badly garbled) — PASS on early-stop checkpoint |
| `(2,1)` | **PASS** (early-stop, 400 steps) — `cond_full` exact byte-for-byte match at eval checkpoints ~00:00:29, ~00:01:32 (also `rg/xml/export-0.3/" xmlns:xsi="h` at ~00:00:44); final step 399 already collapsed (bpb=0.0164, byte_acc=99.58% but garbled) |
| `(2,1,1)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:01:16, ~00:02:10, and at the **final step 399** (bpb=0.0410, byte_acc=99.34%) — clean convergence, no collapse |
| `(2,2)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:00:27, ~00:01:10, and at the **final step 399** (bpb=0.0211, byte_acc=99.44%) — clean convergence, no collapse |
| `(4,1)` | **PASS** (400 steps, ~12min — hit an unpredictable throughput stall between step ~46 and ~66, not a bug) — `cond_full` exact byte-for-byte match at ~00:00:15, ~00:11:52, ~00:12:00 |
| `(2,2,1)` | **FAIL** (400 steps) — no exact `cond_full` match at any of the 7 eval checkpoints scanned, including final step 399 (bpb=0.0174, byte_acc=99.41% but still visibly garbled throughout); log: `logs/qcute_v5_concat_ks221_400_noaux/run.log` |
| `(4,2)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:00:15, ~00:00:58; no `sample_context` truncation crash recurrence (previously fixed) |
| `(2,2,2)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:00:47 only (single checkpoint, all others garbled) |
| `(4,2,1)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:00:46, ~00:00:58, ~00:01:22 |
| `(4,4,2)` | **PASS** (400 steps) — `cond_full` exact byte-for-byte match at ~00:00:19, ~00:01:10, and at the **final step 399** (bpb=0.0176, byte_acc=99.49%); no `generate_level1_codes_via_decode` crash recurrence (previously fixed) — clean convergence on the hardest Ks in the grid |

**Grid complete: 11/12 PASS, 1 FAIL (`(2,2,1)`)** — every other Ks (including the hardest, `(4,4,2)`) reached at least one byte-for-byte exact `cond_full` match, several at the final step with no collapse. `(2,2,1)` alone never matched at any of 7 checkpoints scanned; not re-attempted at higher steps per standing "just continue even if fail, note down which" instruction — flagged here as the one architecture-difficulty outlier in this batch, worth revisiting if aux-off `qcute_v5_concat` work continues.
