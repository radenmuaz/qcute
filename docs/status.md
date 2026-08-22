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

**2026-08-20 (later session)**: the seed-token-interleaved decoder above (now `StackDecoderV1`,
legacy -- "too memory expensive") got a non-interleaved sibling, `StackDecoder` (was briefly
`StackDecoderV2`/`V3` mid-session, both later renamed away since it's now the default): pass 1
runs plain causal self-attention over real bytes with own-code cross-attention spliced per layer
and saves K/V (`encode_like_self_attn_decode`); pass 2 queries that saved K/V with a trainable
seed token that is never itself a key (`seed_query_decode`) -- avoids V1's sequence-length blowup
(`∝(1+1/K)` vs V1's `∝(1+1/K)²`). Generalizes to multi-level "every upper code" conditioning via
chained `cross_attn_stage` calls (v5-style, gated by new `cfg.cond_depth`: -1 = pervasive, N =
N levels up only). Two more decode variants explored for parallel (non-block-recurrent) decode:
`StackDecoderLocal` (block-diagonal same-level conditioning, seed's self-attention contribution
under same-block+causal is provably always zero so it reduces to pure cross-attention) built and
working; `StackDecoderSync` (synchronized wavefront across blocks) left a `NotImplementedError`
stub with a resumable design-note docstring, gain unclear vs. Local. `--decoder_type` values:
`stack_v1` (legacy), `stack` (default), `stack_local`, `stack_sync` (stub).

**Two real bugs found and fixed on top of this**: (1) a training-time alignment bug --
`decode_level`'s track0 output `h0` is UNSHIFTED (`h0[p]` reconstructs `x_real[p]`, correct for
its own loss) but got fed directly into upper-track `cross_attn_stage`, which assumes SHIFTED
alignment -- corrupted upper-track training signal for any multi-track config; fixed by deriving
a separate `h0_shifted` for the upper-track handoff. (2) a generation-time character-doubling bug
in `generate_no_cache`/`generate_kv_cache` (base `Decoder` class): whenever `query_last is None`
(always true for non-top branches) they fell back to a stale `h_list[0][:, -1, :]` hidden state,
so every other generated byte reused an identical causally-blind hidden state and deterministic
sampling reproduced the same byte verbatim -- caught only by actually reading generated text, not
by accuracy numbers, after being asked "did you check generation output is text legit". Fixed for
both `StackDecoderV1` and `StackDecoder` via `_generate_blockwise`: decode one whole new block at
a time with the same validated batched-decode primitives, teacher-forcing each new byte back in
within the block. `code_source="pred"` (real generation) samples the new block's own code from
level1's genuine NTP (`generate_level_codes`/`sample_next`, never from `decode_level` itself,
which has no mechanism to answer "what's next" for its own autoencoder-style code -- see
`Encoder.forward`'s docstring); `code_source="gt"` (diagnostic) uses the real encoded code. A
mechanics-only check (`check_blockwise_gen_consistency`, reuses the loop's own returned
`code_used` rather than re-deriving it, to guarantee a mismatch can only mean the loop itself is
wrong) verified 0/N mismatches in both modes for both decoders against real checkpoints, with
exact-match/coherent real generated text (no doubling). `StackDecoder` also needed its own
`check_gen_consistency`/`check_roundtrip_consistency`/`check_decode_modes` overrides (base
`Decoder`'s versions hardcode `StackDecoderV1`-specific internals and would crash against
`StackDecoder`'s different `stage_lms` shape) -- built on `_generate_blockwise`, `n_levels==2`
only so far for all of this generation-fix work.

A first real overfit10k run with the fixed, now-default `stack` decoder_type + fixed qual checks
(`ks21_v256_pq1_overfit10k.py`) completed cleanly: bpb 0.073, train byte_acc 98.5% at step 1000,
`blockwise_gen_consistency` 0/64 throughout, and real generated text from a 64-byte prompt
matched ground truth almost exactly (only diverged on the last word of a 64-byte span). But
`decode_modes_train`'s `gt_byte_acc` stayed anomalously low (.02-.29) throughout despite training
byte_acc hitting 98%+ -- traced to `check_decode_modes` using a near-empty 2-byte prompt plus
`_generate_blockwise`'s `gt` mode feeding self-attention context from the model's *own*
previously-generated bytes (only the code was real), so per-block errors compounded
autoregressively across ~63 blocks (0.985^126 ≈ 15%, the right order of magnitude) -- not a
decoder bug, just a metric that stopped being a true code-quality upper bound once
`_generate_blockwise` became the shared implementation. Fixed with a new `_decode_gt_context`
helper (mirrors `StackDecoderV1`'s original `decode_from`: batched decode from real ground-truth
own-code *and* real ground-truth self-attention context throughout, context never overwritten by
predictions) used only by `check_decode_modes`'s gt mode; `_generate_blockwise`/
`check_blockwise_gen_consistency`/`check_roundtrip_consistency` untouched. Re-verified against
the same checkpoint: `gt_byte_acc` jumped to .89 (now properly tracking training byte_acc) while
`pred_byte_acc` stayed at .03 -- revealing the real current bottleneck is level1's own code
forecast, not decode quality: with only a 2-byte prompt it can't yet predict the next block's own
code well enough to drive real autoregressive generation, exactly the risk the diagnostic's own
docstring calls out. (Separately, a cosmetic alignment bug in `qualitative_generate`'s
`level0_mode{N}:` log line padding -- hardcoded instead of computed to match `level0_uncond:`'s
width -- was also fixed.)

**Not yet done**: Variant A (window-setup sweep guideline/script) was planned but not built.
`ks21_v64_pq4_overfit10k.py` still has `qual_gen_bytes=0`, not yet re-enabled like the v256
config. Generation-fix/qual-check work stays scoped to `n_levels==2`.

**Tiny-window stress test** (`configs/v1_stack_simplex/ks21_v256_pq1_overfit10k_tinywindow.py`,
same run 2026-08-20): level0 decode's own byte-level self-attention window (track0) forced to 2
(current block only, zero raw-byte lookback across blocks) while the level1 cross-attention
window (track1) stayed full, to test whether decode can still overfit when forced to depend on
level1's code rather than optionally leaning on long-range raw-byte self-attention. Result: yes
for the code path -- `gt_byte_acc` (true upper bound, via `_decode_gt_context`) reached .90-.97,
tracking the 91.26% final train byte_acc closely, confirming level0's own-code decode doesn't
need self-attention lookback to memorize. But `pred_byte_acc` stayed as low as the full-window
run (.02-.13) -- same bottleneck, level1's own code forecast, not decode capacity. Qualitatively
this matters: with the *same* 64-byte real prompt that gave the full-window run near-perfect
legible generation, this run's `level0_mode1` output visibly mode-collapsed into a repetitive
garble (`"...gosososososososonsilsososososodwinece..."`) -- with no raw-byte fallback to lean on,
generation quality now directly exposes level1's weak code forecast instead of the full-window
run's byte-level memorization masking it. `best_val_bpb=2.63` (worse than the full-window run's,
as expected -- val text isn't memorized either way, and there's no long-range byte context left
to help general val prediction). Net conclusion: level1's code-forecast quality, not decode
capacity or window size, is the real open problem for unconditioned generation.

**Same tiny-window stress test, PQ variant** (`ks21_v64_pq4_overfit10k_tinywindow.py`, vocab=64
pq_chunks=4 instead of the single 256-way softmax, steps doubled to 2000): reached
`gt_byte_acc`=.99-1.0 (repeated exact 1.0s through training) and, notably, `best_val_bpb=1.60` --
clearly better than the single-softmax tinywindow run's 2.63. More importantly, real generation
(`level0_mode1`, `pred` mode, 64-byte real prompt) came out an **exact match** to ground truth on
a train sample (`"    <minor />\n      <comment>Fixing redirect</comment>\n      <te"`), not the
repetitive mode-collapse garble the single-softmax tinywindow run produced under the identical
window handicap -- even though `pred_byte_acc` from the (still 2-byte-prompt) diagnostic stayed
just as low (.03-.13) in both. So the PQ codebook's extra structure seems to give level1 enough
usable signal to avoid visible generation collapse even while its raw single-step forecast
accuracy looks equally weak by the numeric proxy -- worth keeping in mind before trusting
`pred_byte_acc` alone as "generation quality," and a reason to prefer PQ-style codebooks going
forward for any config where level0's window is constrained.

**2026-08-20/21 hard-convergence-queue: `ks221`/`ks441` (n_levels=3) real-generation collapse,
13-config sweep.** Four bugs fixed first: `qualitative_generate`'s `level0_mode{N}:` log padding
(hardcoded, now computed to match `level0_uncond:`'s width); `check_decode_modes`'s "gt" mode
wasn't a true code-quality upper bound (see the tinywindow entry above for the same root cause --
`_generate_blockwise(code_source="gt")` still fed self-attention context from the model's own
prior predictions), fixed via a new `_decode_gt_context` method (batched, real ground-truth
context throughout, `StackDecoderV1.decode_from`-style) used only by `check_decode_modes`;
`StackDecoder.check_gen_consistency` was missing the `n_levels != 2` skip guard present on its
sibling checks, crashing any `n_levels==3` config with `qual_gen_bytes>0` -- fixed; and a
method-name collision (`StackDecoder` and the base `Decoder` both defined their own differently-
designed `_generate_blockwise` -- `StackDecoder.generate_no_cache`'s `n_levels!=2` fallback called
`super().generate_no_cache()`, which internally called `self._generate_blockwise(...)`, but that
polymorphically resolved to `StackDecoder`'s own override, defeating the fallback) -- fixed by
renaming `StackDecoder`'s version to `_stack_generate_blockwise`.

Question investigated: does the `ks41`/`ks81` window-constrained-overfit success (from the
tinywindow entries above) generalize past `n_levels=2`? `configs/v1_stack_simplex/
ks41_v256_pq1_overfit10k_window4.py` and `ks81_v256_pq1_overfit10k_window8.py` (Ks=(4,1)/(8,1),
each level0 window forced to exactly its own K) both **converged cleanly** at 1000 steps
(gt_byte_acc .96-.97, coherent real generation). But `ks221_v256_pq1_overfit10k_window.py` and
`ks441_v256_pq1_overfit10k_window.py` (Ks=(2,2,1)/(4,4,1), n_levels=3, same per-level tiny-window
treatment) **did not converge** -- train byte_acc plateaued at 89-95%, real generation
(`level0_mode1`) stayed repetitive garble even on train data. Longer steps (3000, `_long` variants)
didn't fix it (96-97% train byte_acc, still non-coherent). `cond_depth=1` didn't fix `ks221`
(93.01%, slightly *worse* than pervasive's 96.19%) though it showed a partial improvement for
`ks441` (96.25%, real domain vocabulary like "Anarchism"/"Kropotkin" appearing repeatedly and
correctly, `level1`/`level2_ntp_acc` ~0.30-0.37, notably higher than other configs) -- still no
coherent grammatical sentences.

Quant-structure sweep on `ks221` (all `cond_depth=1`, all converge train byte_acc to 93-99.7%,
**none** fixed real generation -- `level0_mode1` always collapsed into a repetitive token loop):
PQ vocab=16/pq_chunks=4 (99.70%, `"[[an1 ana]] [[an1 ana]]..."`), FSQ grid_dq=8/grid_levels=4
(93.23%), then a further 5-config random sweep all matched to the original 8-bit combinatorial
width but different chunk/dim structure -- PQ vocab=4/pq_chunks=4 (99.43%), vocab=8/pq_chunks=3
(99.56%, `".x.x.x.x..."`), vocab=16/pq_chunks=2 (99.26%); FSQ grid_dq=4/grid_levels=4 (98.33%,
`"histanp]] histanp]]..."`), grid_dq=2/grid_levels=16 (98.70%, `"Med Med Med..."`). Chunk/dim
structure at fixed width made no qualitative difference -- every variant hit the same failure
mode. A pure window-relaxation isolation test (pervasive cond_depth, no quant change, level0/
level1 windows relaxed from exactly-K to 2x-K, "2 blocks worth of context back") also failed
(v256pq1 98.18%, v16pq4 99.70%), as did a much more generous 16x-K ("16 codes worth") relaxation
on both simplex (v256pq1, 99.24%) and FSQ (8x4, 98.77%) -- notably, at 16x-K the failure mode
changed from repetitive token loops to non-coherent but *diverse* word-salad (no more
`"xxx]]xxx]]..."`-style collapse, but still no grammatical sentences), suggesting the window size
does matter for avoiding one specific pathology without yet being sufficient on its own.

`scheduled_sampling_p` (real usage, not just smoke-tested): the last fallback tried.
`sample_next()` originally used raw hard-argmax + `F.one_hot` with no gradient path back to the
level-above encoder that produced the substituted code (confirmed by inspection -- `argmax`/
`one_hot` have no `grad_fn`), unlike `BinaryQuant`/`GMMQuant`'s own `sample_next()`, which already
routed through their STE `bsq_quantize`/`_select` machinery. Changed `SimplexQuant.sample_next` to
use the same `_effective_hard_sample()` hard/sample setting as `quantize()` through
`gumbel_quantize` (STE: hard forward, soft-softmax gradient backward, forward value unchanged from
the old argmax version); `GridQuant.sample_next` similarly but always `sample=False` (no gumbel
noise -- there's no well-defined sampling distribution over FSQ's L levels the way there is for a
softmax, per direct instruction). Added `Config.detach_ss_sample: bool = False` (new default:
gradient connects to the encoder; set `True` for the old fully-detached behavior) threaded through
`make_quant()`. At `scheduled_sampling_p=0.5` with the new STE-connected gradient: `ks21` sanity
check (n_levels=2) train byte_acc **dropped** to 73.92% (vs the old detached version's 99.02%,
`gt_byte_acc` oscillated .59->.87->.55, unstable) -- the extra gradient signal destabilized
training -- but real generation showed more diverse plausible text fragments ("Category",
"namespace", "contributor") instead of pure repetitive collapse. `ks221` at the same p=0.5 still
**failed** (93.97% train byte_acc, `"hatim]] hatim]] hatim]]..."`). Lowered to p=0.1: `ks21` sanity
recovered to 89.64% (still below baseline but more stable, coherent-ish prose fragments), `ks221`
reached 97.61% train byte_acc but still **failed** (`"...in [\net in [\net in [\net in..."`).

**Net conclusion across all 13 configs in this investigation**: every lever tried (longer steps,
`cond_depth`, six PQ/FSQ quant-structure variants at matched code width, window relaxation from
exactly-K up to 16x-K, and STE-connected scheduled sampling at two substitution rates) converges
`ks221`/`ks441` train byte_acc to 93-99.7%, but **none** produce coherent real (non-teacher-forced)
generation -- the failure mode shifts (tight windows/high substitution rates -> repetitive token
loops; generous windows -> non-coherent but diverse word-salad) without ever resolving into
grammatical output. The `n_levels==2` case (`ks21`, `ks41`, `ks81`) converges cleanly and
generates coherently under the same per-level window handicaps -- this appears to be specifically
an `n_levels>=3` (multi-hop cross-attention depth) problem, not a code-width, quant-type, window-
size, or exposure-bias problem in isolation.

**Follow-up diagnostic (2026-08-21): is `ks221`'s failure a decode bug or an upper-level forecast
problem?** `scripts/probe_gt_code_generation.py` (new) runs real sequential byte generation
against the `ks221_v256_pq1_overfit10k_window16_relaxed` checkpoint, but forces EVERY decode
cross-attention code (both level1's and level2's) to the real/ground-truth code -- freshly
recomputed by `encoders[1]`/`encoders[2]` from the true byte stream at each new block, never the
model's own forecast -- while the self-attention byte context still comes from the model's own
just-generated bytes (matching real generation's actual compounding risk; only the cross-attended
codes are cheated). Across 5 different 224-byte spans: byte match 13.39%/82.59%/79.02%/10.71%/
62.95% (avg ~50%), but **every single output is coherent, well-formed real text** (valid XML/wiki
markup, grammatically sound, zero repetitive-loop collapse) -- divergences are isolated to
specific memorized/arbitrary content (a namespace `key="15"` vs `key="-1"`, an article title/id
pair), not garbled grammar; one span matched exactly through a full sentence (141/224 bytes)
before diverging only on the next page's title. Confirms decode itself is sound: given correct
upper-level codes, real incremental generation reconstructs coherent, largely-matching text. The
repetitive-collapse failure seen in every "pred" mode run throughout the whole ks221/ks441
investigation (see the 2026-08-20/21 hard-convergence-queue entry above) is attributable
specifically to level1's/level2's own next-code forecast being wrong at generation time, not a
decode/architecture defect -- any further fix needs to target upper-level code-forecasting
quality (more capacity/data/training signal for those small code-sequence LMs), not decode
mechanics, quant type, or window size.

**2026-08-21/22: uncertainty weighting, a real generation bug, and the curriculum that finally
fixes `ks221`.** Three threads, in order:

1. **Uncertainty weighting** (Kendall/Gal/Cipolla 2018, `Config.uncertainty_weighting`): one
   learnable log-variance per NTP task replacing the fixed `byte_ntp_weight`/`code_ntp_weight`/
   `decode_ntp_weight` scalars, logged as `uncertainty_sigma_<task>`. Implemented as a fully
   isolated `if/else` in `QCuteLM.forward` (trivial to remove). Tested alone
   (`ks221_v16_pq4_..._uw`, `ks221_v16_pq8_..._uw`) and combined with full-strength STE-connected
   scheduled sampling (`..._uw_ss1` vs `..._ss1` isolation pair) — **all four failed identically**
   (98-99.9% train byte_acc, repetitive single-token collapse in real generation); the learned
   sigmas even moved the *wrong* direction (deprioritizing the upper-level forecast loss further).
   Net: uncertainty weighting is not the fix, in any combination tried.

2. **Real generation bug found and fixed**: `StackDecoder.generate_no_cache`/`generate_kv_cache`
   silently fell back to the base `Decoder`'s `_generate_blockwise` for any `n_levels != 2` model
   (i.e. every `ks221`/`ks441` config) — a method hardcoded to a 2-level assumption
   (`self.stage_lms[0][1]`, `model.encoders[1]`) that **takes no `max_srcs` argument at all**.
   Confirmed by direct log inspection: `qual_*_level0_mode1`/`mode2`/`modefull` were byte-identical
   in *every* run this entire session — the mode sweep never actually exercised different
   conditioning depths, and level2 was never exercised by real generation at all, ever, before this
   fix. Every "repetitive collapse" conclusion drawn from real-generation output prior to this
   point (the whole hard-convergence-queue, both uw variants, both ss1 variants) was reading the
   same fixed own+level1-only path regardless of label. Fixed by generalizing
   `StackDecoder._stack_generate_blockwise` to `n_levels>2` (chains `upper_track_step` through
   however many upper-track stages `cond_depth` allocated, capped by a runtime `max_srcs`);
   `generate_no_cache`/`generate_kv_cache` no longer special-case away from it. Verified via smoke
   test (tiny random model, `n_levels=3`) producing genuinely different `ks21` vs `full` output,
   and via `check_gen_consistency`/`check_roundtrip_consistency`/`check_decode_modes` unaffected
   (still correctly gated `n_levels==2`-only, no regression).

3. **`max_srcs` generalized to per-level** (renamed from `max_decode_sources`, same meaning):
   `QCuteLM._run`/`forward` now accept a per-level tuple, not just a scalar. This mattered because
   a *scalar* `max_srcs=2` can't cleanly emulate "as if level2 didn't exist" for `ks221`: level0
   correctly drops level2 (2 upper tracks, keeps the nearer one), but level1 has only *one* upper
   track (level2) to begin with, so a scalar cap of 2 never removes it — level1's decode kept
   conditioning on level2 throughout any scalar-capped "phase 1", contaminating the intended
   ks21-equivalent baseline. A tuple like `(2, 1, None)` fixes this: level0 keeps level1 only,
   level1 keeps nothing above it — genuinely no path from level2's code into decode anywhere.

   With both fixes in place, ran `ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2`
   (`curriculum_max_srcs=(2, 1, None)` for the first half of training, `None`/full for the second
   half, `scheduled_sampling_p=1.0`, no uncertainty weighting): **phase 1 (`level0_modeks21`)
   becomes real, structured Wikipedia-XML text within ~2 minutes and keeps improving. At the
   phase-2 switch (~step 1500), `level0_modefull` — which had been pure noise the entire first half
   (its cross-attn stage never touched) — flips to real words within one eval step and converges to
   coherent text nearly matching `modeks21` within ~200-400 steps, with zero repetitive collapse.**
   This is the first time in the whole investigation that full `ks221` real generation has produced
   coherent output. Train byte_acc ~98%, unremarkable — the qualitative generation is the real
   signal here.

   One caveat surfaced in the same run: `level1_gen`/`level2_gen` (`generate_level_codes` — a
   *free-running* rollout that repeatedly samples from level1's/level2's own NTP head and feeds the
   sample back as its own next input, no byte grounding at all) still collapses to a single
   repeated code, even post-curriculum. This is a *different* generation path from
   `level0_modefull`'s (which recomputes level1/level2's code fresh from the real just-generated
   byte stream every block, never a multi-step free rollout of the code itself) — so the two
   results aren't contradictory: the code is fine when grounded, degenerates immediately under free
   rollout. Consistent with a live hypothesis (see `CLAUDE.md`'s "Non-recurrent upper-level plan"):
   level1/level2's own self-NTP loss rewards a low-entropy/constant code, and it shows up exactly
   where you'd expect — unconstrained free rollout — not in one-step-grounded generation.

   Net conclusion: the `ks221`/`ks441` architecture is sound and the asymmetric-cap curriculum
   (converge a real ks21-equivalent submodel first, only then graft on the coarser level) is a
   working fix for full-model real generation, at overfit10k scale. Not yet tried: full-scale (not
   overfit10k) runs with this curriculum, `ks441`, or dropping non-top levels' self-NTP loss
   entirely (the next planned test, per `CLAUDE.md`).
