# qcute v1 status

Reset at this point — the prior narrative (v5's modular rewrite, FSQ/PQ/GMM leaderboard) is now
archival. Full history: [docs/archive4/status.md](archive4/status.md) (the v5 leaderboard log this
reset supersedes), [docs/archive3/status.md](archive3/status.md), and
[docs/archive2/status.md](archive2/status.md) (older still). New entries go below, same
convention: newest at the bottom, session-dated where useful.

## Where things stand

`qcute_v5` is frozen — moved to `qcute/v5_old/` (formerly `qcute/qcute_v5*.py`), still runnable
(`uv run python -m qcute.v5_old.qcute_v5 ...`), still the source of truth for every leaderboard
number in `docs/archive4/status.md`, not receiving further architecture work.

`qcute_v1` (`qcute/qcute_v1/`) is the active lineage — forked from a verbatim copy of v5, now
diverged. It implements the latent-AR / parallel-block-local-decode investigation (see
`CLAUDE.md`'s "Latent-AR / parallel-block-local-decode investigation" section): the goal is
decoupling decode from the block-to-block recurrent dependency that blocks parallel generation.
Full design narrative, worked examples, and the staged plan: **[docs/qcute_v1_plan.md](qcute_v1_plan.md)**
— this file only tracks results/progress, not design (avoid duplicating that doc here).

### Architecture summary (detail in qcute_v1_plan.md)

Only the top level stays a genuine NTP/AR decoder (self-code recurrence, unchanged from v5).
Every level below top: causal self-attention over its own actual sequence with a trainable
per-block **seed token** prepended before every `K`-block (`bos_interleaved_self_attn`, one fresh
seed token per block, not once per sequence -- not "BOS", it recurs every block; not "sink"
either, it's a full token, not a passive fallback key, see below), then cross-attention to that
*same* block's own-level code (not a coarser level's — `own_block_cross_attn_decode`, separate
`bb_cross` LM instance, NOT `cross_attn_stage`, which is reserved for the top level's coarser-code
case), predicting each block's own K real bytes UNSHIFTED from that block's own code
(`own_block_decode_loss`) — the seed token's own hidden state is a genuine query, not discarded,
and directly predicts its block's own first byte using that block's own code (`c1` reconstructs
`ab`, `c2` reconstructs `cd`, no lag; see 2026-08-20 session log below for the bug this fixed).
`share_encode_decode_self` shares encode's LM (embed/blocks/seed-token) with decode's self-attention
stage at every level; the cross-attention stage always stays separate. `scheduled_sampling_p`
(default 0, one flip per forward pass, training only) swaps the cross-attended code from real to
the level-above's own sampled prediction, closing some of the train/generation exposure-bias gap.

Two new diagnostics (`Decoder.check_roundtrip_consistency`, `Decoder.check_decode_modes`), wired
into `train()`'s existing per-eval diagnostic block (runs every eval step when `qual_gen_bytes>0`,
same as v5's `qualitative_generate`/`check_gen_consistency`):
- `roundtrip_{train,val}_acc`: decode from the real ground-truth code, re-encode, compare —
  baseline round-trip noise floor, no upper-level sampling involved. Printed as accuracy (`.XX`,
  or `1.0` if exact), not a raw count.
- `decode_modes_{train,val}`: `gt_byte_acc` (decode from real code, upper bound) vs.
  `pred_byte_acc` (decode from level1's own sampled code prediction, the real generation-time
  signal) — the actual "can the upper-level LM's forecast support generation" check.

Both are `n_levels>=2`-only (no-op at `n_levels==1`, where level0 is the unchanged top-level
path) and diagnostic-only (print, never gate/halt).

### Not yet implemented

- Generation-loop rewrite (chunk-at-a-time, replacing `generate_no_cache`/`generate_kv_cache`'s
  byte-at-a-time loop) — `qualitative_generate`/`check_gen_consistency` are not reliable for
  `n_levels>=2` configs right now, hence `qual_gen_bytes=0` in most `Ks=(1,)` configs.
- Path (a) generation (draft via uncond LM, encode, decode-refine) and the speculative-decode-like
  verification idea (sample from upper LM, decode, re-encode, check match) — noted in
  `qcute_v1_plan.md`, not built.
- Async self-attention window ablation (`window=K+1`) — implemented and smoke-tested, not yet run
  as a real comparison against the sync default.
- `n_levels>=3` generalization (only immediate own-level code is cross-attended so far).
- `ConcatDecoder` was not converted to the new architecture — still v5's original mechanism.

## Session log

**2026-08-20**: `qcute_v1` created, decoder rewritten (after two false starts — a standalone
learned-query cross-attend-only function, then a wrong "self-attend own code" reading — see
`qcute_v1_plan.md` for the full false-start history), `check_roundtrip_consistency`/
`check_decode_modes` diagnostics added, `share_encode_decode_self` fixed to cover the new
non-top-level self-attention stage (was only wired for top). `qcute_v5` moved to `qcute/v5_old/`.

Validated via CPU smoke tests: `Ks=(1,)` and `Ks=(2,1)`, all 5 quantizer types (simplex, binary,
grid, gmm, gmm_diag), sync and async self-attention windows, `scheduled_sampling_p`,
`share_encode_decode_self` — all train cleanly, no crashes. A 300-step CPU overfit check on
`Ks=(2,1)` showed real learning (train bpb 8.0->5.0, byte_acc 0%->26%).

`configs/v1_stack_simplex/{ks1,ks21}_v{256_pq1,64_pq4}_overfit10k.py` (n_bytes=10000, steps=1000,
per `CLAUDE.md`'s standing overfit10k methodology) ran on MPS: `ks1_v256_pq1_overfit10k` reached
best_val_bpb=4.03 (train bpb overfit to ~0.24 by step 1000, as expected for a 10k slice at
full-scale hyperparameters — this is the fast-iteration sanity bar, not a quality number).

Full-scale (`configs/v1_stack_simplex/{ks1,ks21}_v{256_pq1,64_pq4}.py`, full ~1M-byte enwik8,
steps=8000, matching the v5_stack_fsq/* convention) queue finished:

| config | best_val_bpb |
|---|---|
| ks1_v256_pq1 | 2.694 |
| ks1_v64_pq4 | 2.479 |
| ks21_v256_pq1 | 2.518 |
| ks21_v64_pq4 | 2.526 |

`Ks=(2,1)` beats `Ks=(1,)` at both vocab settings. PQ (`v64_pq4`) wins clearly at `Ks=(1,)` but
loses narrowly at `Ks=(2,1)` (2.526 vs 2.518), unlike the clean win at `Ks=(1,)`.

**These `ks21_*` numbers are now STALE and need re-running** (see below) -- they used a decode
mechanism with a real bug (own-code cross-attention never actually reconstructed its own block, see
next entry), not the corrected one. `ks1_*` numbers are unaffected (`Ks=(1,)` has no non-top level,
so this bug never applied to them).

**Bug found and fixed (same 2026-08-20 session, after the queue above already ran)**:
`cross_attn_stage`'s `code_pos <= query_pos` mask (`code_pos` = a block's own LAST byte position)
meant a block's own code only ever became visible starting at its own last position -- useful only
for predicting *later* blocks' content, never for reconstructing its own. Confirmed empirically
(code-sensitivity probe: perturbing block 3's own code changed nothing about block 3's own
reconstruction, only blocks 4+). This silently reduced non-top decode to "hint from a past
same-level code" (structurally like v5's original coarser-code-hint mechanism, just same-level
instead of coarser), not the intended "reconstruct this block from its own code" autoencoder
framing. Root-caused via a dense back-and-forth (see chat) that also detoured through a
single-global-BOS simplification (reverted -- see `bos_interleaved_self_attn`'s docstring) before
landing on the actual fix: `own_block_cross_attn_decode` + `own_block_decode_loss`
(`qcute_v1_decoder.py`), which keep the per-block seed token's own hidden state as a real query
(not stripped) and set `code_pos` to each block's own seed-token position, so a block's own code is
visible to every position in that block. Full architecture writeup:
`docs/qcute_v1_plan.md`'s "Decoder architecture" section (rewritten to match).

Also fixed in the same pass: `check_gen_consistency`'s truncation bug (a context length not a
multiple of `K0` silently dropped the trailing partial block inside `bos_interleaved_self_attn`'s
reshape, shifting which absolute position the returned last-position query corresponded to, showing
spurious ~50% mismatch rates unrelated to any real inconsistency) -- now restricted to `K0`-aligned
`t` for `n_levels>=2`. Confirmed 0/N mismatches at aligned positions post-fix.

Verified via smoke tests (forward/backward, `check_roundtrip_consistency`, `check_decode_modes`,
`check_gen_consistency`, a real 20-step MPS training run) -- all clean, loss decreasing normally.
Not yet re-run at overfit10k or full scale.

**TODO for a fresh session**: requeue `configs/v1_stack_simplex/ks21_v256_pq1.py` and
`ks21_v64_pq4.py` (full-scale, `--decoder_type stack`) with the corrected decode mechanism --
their existing `best_val_bpb` numbers above are stale. `ks1_*` configs do not need re-running.
