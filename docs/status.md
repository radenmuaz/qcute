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

## Must-dos when writing generation code (learned the hard way, repeatedly)

"The model isn't learning" is wrong far more often than "generation is querying something the
trained weights were never optimized to produce." Every bug above had the same signature:
excellent teacher-forced accuracy, garbage free-running generation.

- **Trace the exact tensor** the training loss reads at that point — don't assume shape parity
  (`(B,L,D)`) means value parity between two code paths.
- **Any control-flow-changing param is a suspect** (`max_decode_sources`, `want_code`,
  `compute_ntp`, truncation/masking). Confirm both branches it can select were actually trained,
  not just the untouched one.
- **Teacher-forced-good + free-running-bad ⇒ check generation code first**, not more training or
  regularization.
- **Test fixes against an existing checkpoint before retraining** — most of these are pure
  generation-code fixes with zero effect on the training graph.
- **Don't reach for decode/train-time band-aids** (temperature, top-k, noise injection) before
  ruling out a plain dispatch/tensor bug — they look identical from the output alone.
