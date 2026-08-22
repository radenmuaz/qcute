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
   working fix for full-model real generation, at overfit10k scale.

4. **Isolation: is `scheduled_sampling_p=1.0` actually necessary alongside the curriculum, or just
   carried over from the prior (failed) lever-sweep this config was built on?**
   `ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss` — identical to the run above except
   `scheduled_sampling_p=0.0` (no scheduled sampling at all). **Result: not just as good, strictly
   better.** Train byte_acc 99.73% (vs 98.20%), val byte_acc **0.579** (vs ~0.36-0.40), best val bpb
   **2.73** (vs 4.63) — and the same phase-1-noise/phase-2-graft/coherent-convergence pattern
   replicates cleanly, with the end-of-training val sample even showing a genuine content match
   (`"...Against Anarchists&quot;, 189"` -> both modes correctly continue `"8&lt;/ref&gt;, it has
   also..."`, matching ground truth), not just stylistic coherence. Conclusion: `ss=1.0` was not
   only unnecessary, it was actively worse — the curriculum (`curriculum_max_srcs`/`curriculum_step`
   alone, no scheduled sampling, no uncertainty weighting) is the recipe to keep. This is the
   simplest config that has produced coherent `ks221` generation so far, and the one to carry into
   a full-scale run.

   Not yet tried (superseded by finding 5 below before being run): full-scale (not overfit10k) runs
   with this curriculum, `ks441`, replication across seeds/Ks, or dropping non-top levels' self-NTP
   loss entirely -- that last one addresses the still-unresolved `level1_gen`/`level2_gen`
   free-rollout collapse, not the (now working) grounded generation path.

5. **`kv_lm`: give cross-attention K/V a causal self-attention pass over the embedded code
   sequence first, instead of an isolated per-position embedding** (chat 2026-08-22). Motivation:
   `embed_for_decode(code)` gives each code position a K/V vector with zero interaction between
   code positions -- any context the *producing* level's own self-attention built got thrown away
   at quantization. New `StackDecoder.kv_lm_mode` (`qcute_v1_decoder.py`'s `code_context_pass`/
   `KVContextLM`): `"identity"` (default, unchanged prior behavior), `"fresh"` (one more small LM
   per track, own weights, causally self-attends over the embedded code sequence before it's used
   as K/V), `"shared"` (same causal pass, reusing the producing level's own encoder LM weights
   instead of a fresh module -- cheaper, but ties it to whatever the encoder's self-NTP loss shapes
   those weights toward). Also generalized `max_srcs` (renamed from `max_decode_sources`) to accept
   a per-level tuple, not just a scalar -- a scalar cap can't drop a level's own single nearest
   upper track without also dropping level0's, which is why the *first* curriculum attempt
   (`..._curriculum.py`, superseded by `curriculum2`/`curriculum2_noss` above) was contaminated.

   Three-way comparison, `Ks=(2,2,1)`, `window16_relaxed`, `vocab=16 pq_chunks=4`, no scheduled
   sampling, no uncertainty weighting:

   | run | train byte_acc | val byte_acc | best val bpb |
   |---|---|---|---|
   | curriculum alone (`curriculum2_noss`, finding 4) | 99.73% | 0.579 | 2.73 |
   | **`kv_lm_mode=fresh`, no curriculum** | 99.61% | **0.563** | **1.93** |
   | `kv_lm_mode=fresh` + curriculum together | 99.73% | 0.504 | 3.11 |

   **`kv_lm` alone is the best result of this entire investigation, and combining it with the
   curriculum makes it *worse*, not better.** `ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum`
   produces coherent, exact-matching real generation with no curriculum, no scheduled sampling, no
   uncertainty weighting -- e.g. `qual_train_level0_modeks21`/`modefull` both reproduce
   `"&gt;[[Friedrich Engels|Engels]], Freidrich. ''&quot;[http://www."` identically. This is strong
   evidence the collapse was fundamentally an *expressiveness* problem (isolated per-position code
   embeddings starved of any cross-code context) rather than a training-signal-ordering problem --
   the curriculum was compensating for that expressiveness gap, not fixing an independent issue, so
   once `kv_lm` fixes the gap directly, the curriculum's phase-1 restriction on training signal is
   pure friction. `ks21_v256_pq1_overfit10k_kvlm_fresh` (regression sanity check on the already-easy
   2-level case) ran without incident (val byte_acc 0.347, best val bpb 3.08).

   New recipe to carry forward: `kv_lm_mode="fresh"`, no curriculum, no scheduled sampling, no
   uncertainty weighting. Not yet tried: full-scale (not overfit10k) runs with this recipe, `ks441`,
   `kv_lm_mode="shared"` (cheaper, untested), replication across seeds, or the still-open
   `level1_gen`/`level2_gen` free-rollout-collapse question (unrelated to this fix -- see finding 3's
   caveat, still applies).

## `qcute_zero`: a monolithic single-LM alternative (2026-08-22)

Separate lineage, not a fork of `qcute_v1` -- `qcute/qcute_zero/qcute_zero.py` (single file by
design for now, per direct instruction: "make thing single file first refactor later"). Full design
rationale in the module's own docstring. Core idea: exactly ONE shared transformer LM (embed +
blocks). Every `Ks[s]`-cumulative-period bytes, it summarizes its own hidden state into a discrete
code via the SAME tied embed/output head bytes use (code vocab == byte vocab, so "extracting a
code" is just "predict a byte-shaped distribution, STE-hard-sample it"). That code sequence gets
run through the SAME shared blocks again (a second, shorter forward pass) for a genuine NTP loss on
the code sequence itself (free via weight reuse, no separate per-level encoder) plus contextualized
K/V for a cross-attention ("fuse") stage back into the byte-level query stream. `len(Ks)-1` fuse
stages, one per cumulative period -- each stage (`FuseStage`) owns its cross-attention, its own MLP,
and its own final `ln_out` feeding its own cond-NTP head (revised 2026-08-22: an earlier version
tracked cross-attention and MLP as two separate module lists with the MLP meant to be shared across
stages, but the code actually instantiated one MLP per stage anyway -- merged into one per-stage
module, no pretense of cross-stage MLP sharing left). After a fuse stage's cross-attn+own-MLP, the
byte-level query stream gets ANOTHER pass through the shared self-attn+MLP LM blocks before that
stage's own cond-NTP readout (and before the next stage's cross-attn query input) -- i.e. per stage:
fuse cross-attn+own-MLP -> shared self-attn/MLP -> that stage's own cond head. Also revised: RMSNorm
everywhere (was `nn.LayerNorm`) and every `Linear` is `bias=False`, matching `qcute.bytelm`'s own
bias-free convention (bytelm itself still uses `LayerNorm`, not RMSNorm -- this is a `qcute_zero`-only
change, not applied to the frozen baseline). A mandatory, non-trainable all-zero K/V "sink" is
prepended to every attention call
(self- and cross-, uniformly) so every query row has >=1 visible key even before a periodic code's
causal boundary is reached (avoids NaN from an all-masked softmax row; when the sink is the only
visible key the output is provably exactly zero, a clean no-op, not an arbitrary bias). Default
next-token query is ordinary previous-hidden-state continuation (no seed/BOS token, unlike
`qcute_v1`); an optional, off-by-default `learned_query` mode trains a single generic query vector
(one randomly-sampled position per batch) toward future block-parallel local decode, not required
for or exercised by the default training path.

Designed to need no curriculum at all (unlike `qcute_v1`'s `max_srcs`/`curriculum_max_srcs`): every
fuse stage's code source is the same already-training shared backbone from step 1 (nothing is a
fresh, untouched module the way each `qcute_v1` encoder level was), and the zero-sink lets a
stage's freshly-initialized cross-attention weights self-suppress early (route softmax weight to
the sink) and gradually rely on real codes as those weights improve -- an emergent, learned on-ramp
rather than a hand-scheduled one. Expected, not yet proven.

Causality verified by hand (chat 2026-08-22): every code's visibility boundary must use its
CUMULATIVE byte-span (`cum_K*(block_idx+1)-1`, absolute byte coordinates), never its local index
within whatever intermediate code sequence produced it -- confirmed non-circular under this rule
(a code can only ever inform prediction of bytes strictly after every byte it was itself computed
from).

Implementation smoke-tested (tiny random model, both `Ks=(2,1)` and `Ks=(2,2,1)`, plus the trivial
`Ks=(1,)` no-fuse case): forward+backward+`generate_no_cache` all run without NaN or error, every
parameter receives gradient. `generate_kv_cache` is currently aliased to `generate_no_cache`
(byte-by-byte full recompute) -- matches `qcute_v1`'s own current "incrementally-correct, not yet
KV-cached" state, same precedent; real incremental caching is future work the causal/static-shape
design was built to allow, not a correctness fix needed now.

First real runs (2026-08-22, post RMSNorm/no-bias/merged-FuseStage revision above), both
`configs/qcute_zero/ks21_overfit10k.py` and `ks221_overfit10k.py`, **no curriculum** (matching the
design's own expectation above):

| metric (final, step 1000) | `ks21` (2 levels) | `ks221` (3 levels) |
|---|---|---|
| `val_byte_acc` | 0.391 | 0.387 |
| `val_uncond_acc` (no conditioning) | 0.069 | 0.033 |
| `val_cond0_acc` (level 1) | 0.391 | 0.382 |
| `val_cond1_acc` (level 2, top) | -- | 0.387 |
| `val_fuse0_ntp_acc` | 0.979 | 0.752 |
| `val_fuse1_ntp_acc` | -- | 0.946 |

Key result: `val_cond1_acc` (deepest/top-level cross-attn stage) is not worse than `val_cond0_acc`
for `ks221` -- no degradation cascading up the hierarchy, which is exactly the failure mode
`qcute_v1`'s `StackDecoder` needed `curriculum_max_srcs`/`curriculum_step` to avoid. First 3-level
hierarchical config in this whole investigation (either lineage) to converge cleanly with **zero**
curriculum.

**Checkpoint-selection caveat (important)**: `Checkpointer` saves `best.pt` by lowest *total summed*
val_loss (`uncond_loss + sum(cond_losses) + sum(fuse_ntp_losses)`, per `Config.cond_weight`/
`code_ntp_weight`) -- for both runs this picked a spuriously EARLY checkpoint (step ~200-250) because
`uncond_loss`/`cond1_loss` keep climbing through training even as `byte_acc` keeps improving, dragging
the sum up long before the model is actually done improving. `last.pt` (step 1000, matching the
`val_byte_acc` numbers above) is the trustworthy checkpoint for these overfit10k runs, not `best.pt` --
confirmed by generating from both and comparing (see below). Worth fixing properly later (checkpoint
on `val_byte_acc` or `val_cond{N}_acc` instead of the summed loss), not yet done.

Real generation (`generate_no_cache`, 128 bytes, prompt = 64 real val bytes), from `last.pt`:

- `ks21`: `"arliest times to the present day'', 1945.&lt;/ref&gt; [[Diggers of [[Internation]]] [[Peter Kropotkin|Kropotkin]] foun sond similar sed Intellectuals in Social Change Toward Laissez Faire]&quo"`
- `ks221`: `"arliest times to the present day'', 1945.&lt;/ref&gt; [[Diggers of the Family, Priceter Property Kropotkin, from Encyclopaedia Britannica, 1910]&lt;/ref&gt;\n[[Peter Kropotkin|Kropotkin]] found"`

Both coherent, grammatical, correct wiki-markup structure, no repetitive collapse -- `ks221` (the
hard 3-level case) reads if anything slightly more coherent than `ks21` here. From the same
checkpoints' spurious early `best.pt` instead, both degrade into a repetitive `[[[[[[[...` collapse
(worse for `ks221`, total collapse by ~200 bytes) -- consistent with those being genuinely
under-trained snapshots, not a real architecture regression.

Conclusion: `qcute_zero`'s no-curriculum design premise holds up on first real test, for the exact
case (`ks221`) that broke `qcute_v1` without one.

Full-scale `ks221_1M` run (real `enwik8_1M`, `n_bytes=None`, `configs/qcute_zero/ks221_1M.py`,
`d_model=256, n_layers=4`, 8000 steps) launched 2026-08-22, in progress. `bytelm_xs4_ctx256_mtp1`/
`bytelm_xs4_ctx1024_mtp1` (`mtp_heads=1`, no MTP — isolating whatever MTP itself contributes vs the
xs4/xs preset default of `mtp_heads=4`) queued to follow it (one-job-at-a-time rule).

### The real `qcute_v1` vs `qcute_zero` differentiator: exposure bias at the free-rollout code
substitution, not "autoencoding is bad" (2026-08-22, corrected same day)

Chat discussion converged on a sharper, more fundamental characterization than anything above
(weight-sharing, curriculum mechanics, etc. are all downstream engineering detail by comparison) —
but an initial pass at stating it (below, struck through in spirit) conflated two different things
and needed a same-day correction once challenged directly ("isnt the point [that] decoder must be
autoencoder"):

**Correction: decode-as-autoencoder during training is not the bug — it's the standard, necessary
scheme.** `own_block_cross_attn_decode`'s own docstring already says this plainly
(`qcute_v1_decoder.py:365-370`): reconstructing a block from its own real code during *training* is
"standard VQ-VAE/discrete-autoencoder teacher-forcing," not a leak — there is no other way to train a
reconstruction loss than encode-real-input-to-a-code, then decode-from-that-code, supervised by the
same real input. Calling this "structurally autoencoding, therefore can't do true NTP" (the original,
overstated framing) was wrong: it isn't decode's training-time behavior that's the problem.

**The actual gap is train/inference mismatch at the free-rollout substitution point.** At generation
time, the slot `own_block_cross_attn_decode` cross-attends into has to be filled by the level-above
LM's own free-rollout *prediction* instead of the real per-block code — and decode has never
practiced reconstructing from anything but the ground-truth code during training (except when
`cfg.scheduled_sampling_p` explicitly substitutes one in, see below). This is plain exposure bias,
the same class of train/inference mismatch every teacher-forced seq2seq model faces — not evidence
that reconstruction-based decode is the wrong design. It explains the same findings as before, just
under the corrected diagnosis:
- **Free-rollout collapse** (`level1_gen`/`level2_gen` degenerating to a repeated code): decode was
  never trained against a *self-generated* code standing in for the real one, so nothing during
  training punishes the upper-level LM for collapsing to something decode can't actually use.
- **Why `level0_modefull` "works"**: it sidesteps the substitution entirely by re-encoding from real,
  already-generated bytes every block — always one byte behind, never actually exercising decode
  against a genuinely predicted (not re-derived-from-real) code.
- **Why `curriculum_max_srcs` helped training but not free rollout**: it scaffolds cross-level
  visibility depth, a different axis from the train/inference code-substitution mismatch.

**`qcute_zero`'s generation path has no analogous substitution point.** Verified position-by-position
(chat 2026-08-22, `Ks=(2,1)` worked example): predicting byte `p` always reads `x_cross` at position
`p-1`, and `fuse_mask` only admits codes with `abs_pos <= p-1` — and in ordinary byte-by-byte
generation, every code fed into that mask was *itself* derived from already-committed real bytes,
train and inference alike. There is no point anywhere in the normal decode path where a synthetic/
predicted code stands in for a real one — so this specific mismatch has no seam to live in. (Not
because cross-attention there is somehow structurally different from `v1`'s own-block cross-attn —
both are ordinary reconstruction-style conditioning during training — but because `qcute_zero`'s
inference path never needs to swap in anything decode wasn't trained on.)

**What discipline `v1` would actually need to close this gap** (none currently fully in place):
1. *Structural*: never let decode cross-attend to its own block's code — drop
   `own_block_cross_attn_decode` outright, matching `qcute_zero`'s invariant exactly. Real cost: loses
   the "compress and immediately reuse my own block" capacity — not a free fix, and arguably
   overkill now that the diagnosis is exposure bias, not the reconstruction step itself.
2. *Code-level consistency training, if own-code cross-attn is kept*: `cfg.scheduled_sampling_p`
   **already implements almost exactly this** — one flip per forward pass, training-only, substitutes
   the level-above's own sampled code prediction for the real one across every non-top level
   simultaneously (`qcute_v1.py:114-127`), specifically to close this exact exposure-bias gap
   (correcting an earlier, less careful characterization in chat as "byte-level" — it is not, it's a
   whole-code-sequence substitution). **Empirically this did not help**: `curriculum2_noss` (no SS)
   beat `curriculum2` (`ss=1.0`) on every metric this session. Open question, not yet resolved: is
   `p=1.0` (always-on, all-or-nothing per forward pass) too aggressive vs. an annealed/partial
   schedule, or is a same-loss-different-input substitution insufficient without also weighting a
   dedicated consistency term — i.e. the *mechanism* for #2 exists in the codebase, but the negative
   result found so far doesn't yet distinguish "the idea is wrong" from "this specific tuning of it is
   wrong."
3. *Counter-incentive against the collapse escape hatch*: without a working #2, self-NTP loss alone
   has a free out — go low-entropy/constant, trivially minimizing self-predictability without needing
   to stay useful to decode. A working consistency loss is what punishes that escape (a constant code
   fails to help decode reconstruct diverse blocks when self-sampled); it doesn't currently, since #2
   as tested didn't help.
4. *Or sidestep the joint dynamic entirely*: the already-documented "Non-recurrent upper-level plan"
   in `CLAUDE.md` (train decode-with-real-codes to convergence, freeze, train the upper-level code LM
   purely as an ordinary sequence model over the frozen encoder's real code stream) avoids needing
   #2/#3 at all, since decode and the free-rollout LM never share an optimization loop — a standard,
   well-posed NTP problem has no collapse incentive built in. Given #2's negative empirical result,
   this freeze-then-chain path looks more promising than continuing to tune joint SS.

**Why `qcute_zero`'s basic (already-working, already-tested) generation path needs none of this**:
there is no free-rollout step in ordinary byte-by-byte generation — cross-attn KV always comes from
bytes already committed (real), never an undetermined future code, so the collapse dynamic has no
seam to live in. The one place an analog of #2/#3 *would* become necessary is the speculative
multi-block decode brainstorm below, where a drafted code is genuinely unverified until grounded.

### `qcute_zero`: parallel block decode brainstorm (2026-08-22, not yet implemented)

Motivating question: `qcute_zero`'s generation is currently strictly sequential (byte-by-byte, no
parallel decode at all, unlike `qcute_v1`'s block-parallel-per-level design intent) — but the
existing `learned_query`/`code_kv_cache` scaffolding (single generic trainable query vector, trained
via one random position per batch, currently unused by the default training path) was built exactly
toward this. Worked out concretely for `Ks=(2,1)`:

**Free tier (zero speculation, works today in principle, not yet wired up)**: within a single
`Ks[0]`-period block, parallel decode needs **no code prediction at all**. Predicting byte `p` reads
`x_cross[p-1]`, whose `fuse_mask` only sees codes with `abs_pos <= p-1` — and a block's own code sits
at *its own last byte's* absolute position, so (per the invariant above) neither byte in a
soon-to-be-formed block ever needs that block's own not-yet-real code. Both bytes of a `Ks=(2,1)`
block can be decoded from two `query_vec` slots at once, using only strictly-prior, already-grounded
codes — `B = Ks[0]` bytes per parallel step, entirely for free, no risk.

**Speculative tier (going past one block per parallel step)**: needs the code-level LM
(`fuse{s}_ntp`) to draft codes for blocks not yet grounded — cheap, since it operates at the
compressed code rate (`1` draft token per `K` bytes), even though the drafting itself must stay
sequential (an AR code LM). Procedure: (1) roll `self.blocks`'s own NTP head forward over the *short*
code sequence to draft `ĉ_{m+1}, ĉ_{m+2}, ...`; (2) seed `code_kv_cache` with these at their real
abs positions; (3) run many `query_vec` slots in parallel, each masked exactly per the free-tier rule
(so `ĉ` only becomes visible starting at the block *after* the one that would ground it — never its
own); (4) once those bytes are decided, extract the *real* grounded code and compare to the draft
— match: accept, downstream decode already validated; mismatch: reject and re-decode everything
downstream of the first divergence (standard speculative-decoding accept/reject, but at the code
granularity rather than the byte granularity).

**Comparison to `qcute_v1`'s parallel-decode story — not a strict improvement, a different tradeoff**:
`v1`'s design (only the top/coarsest level genuinely sequential; every level below block-parallel via
a seed token cross-attending to an *already fully-known* upper-level code) is, if it converges,
non-speculative — no accept/reject, no risk of wasted work, block size bounded only by `K` (can be
large, e.g. 32 in real configs). `qcute_zero`'s free tier is bounded by the *smallest* `Ks[0]`
(often tiny, e.g. 2), and going further requires genuine speculative decoding with its usual costs
(payoff gated on the code-LM's acceptance rate, wasted compute on rejection). `qcute_zero` trades a
much easier-to-train decode (see differentiator section above) for a currently unproven,
probabilistic parallel-decode story; `v1` trades brutal training fragility (curriculum, kv_lm,
several collapse bugs this session) for — if it converges — a cleaner, larger, non-speculative
parallel-decode mechanism. Not fixed by removing weight-sharing either way: whether `qcute_zero`'s
per-stage kvlm/heads are tied or fully separate doesn't change this tradeoff, since it depends only
on the code-LM's own NTP accuracy, not on parameter sharing.

**Not yet implemented** at brainstorm time; **built the same session (2026-08-22), then forked off**:
multi-slot `parallel_decode` training (generalized from one random position/batch to `parallel_decode_n_blocks`
independently-sampled `Ks[0]`-blocks per step, folded into an extended batch axis -- "Option B" from
this section, required generalizing `rope_cos_sin_for_positions`/`apply_rope` to per-row positions)
and `generate_blockwise` (the free-tier inference path itself) were both implemented and verified
(no NaN, full gradient coverage, both `Ks=(2,1)`/`Ks=(2,2,1)`). The speculative multi-block extension
and the code-level consistency-training question remain unbuilt.

**This entire query-vec/parallel-decode line of work was then forked off intact into its own
preserved lineage, `qcute/qcute_zero_parallel/`** (module renamed throughout, own `configs/qcute_zero_parallel/`),
rather than continuing to build on top of it in `qcute_zero` itself — a deliberate split, not an
abandonment: `qcute_zero` (this lineage) pivoted instead to building a REAL incremental KV cache for
ordinary sequential generation (see below), a separate, more foundational piece of work that the
query-vec direction doesn't need and shouldn't be entangled with. Revisit `qcute_zero_parallel` on
its own terms later if the parallel-decode direction is worth returning to.

## `qcute_zero`: real incremental KV cache (2026-08-22)

Added `generate_kv_cache` -- a genuine incremental KV cache (previously just aliased to
`generate_no_cache`'s full recompute), verified to produce **bit-exact** the same greedy output as
`generate_no_cache` (315/315 random configs: `Ks` in `{(1,),(2,1),(2,2,1),(4,2,1),(3,3,1),(2,3,1),(4,4,1)}`,
varying `attn_window`/`fuse_window`/prompt length/seed -- see the new `check_kv_cache_consistency`
diagnostic method, the checked-in analog of `qcute_v1`'s `check_roundtrip_consistency`/
`check_gen_consistency` pattern that `qcute_zero` didn't have an equivalent of before this).

Scope: real incremental caching only for the byte-level `self.blocks` self-attention and each fuse
stage's post-cross-attn refinement `self.blocks` pass (`Attn.forward_incremental`/
`Block.forward_incremental`, new) -- the two genuinely `O(L)`-per-step costs. The short
code-sequence self-attention (kvlm) pass and the fuse cross-attention itself are still recomputed
fresh whenever a new code appears (every `Ks[s]` bytes) -- deliberately not cached, since those
sequences are short (`~L/prod(Ks[:s+1])`), not worth the complexity.

Two real bugs found and fixed via direct comparison against `_generate_cascade` (the existing
full-recompute path, refactored out of `_forward_next_byte_logits` to serve as the correctness
reference) before reaching bit-exact:
1. **Skipping a stage's cross-attn+refine pass whenever it had zero real codes yet.** Even with no
   real code, that stage's refine self-attention pass is NOT a no-op (the zero-sink makes
   cross-attn itself contribute a provable zero, but the refine MLP+self-attention afterward still
   transforms its input into something later positions' self-attention needs to have seen) --
   skipping it left `refine_caches[s]` missing entries for every early position. Fixed by always
   running the pass, letting the empty-code case flow through the zero-sink naturally.
2. **Missing "catch-up" priming + wrong `continue` (not `break`) on first activation.** The
   non-incremental cascade has an `if n_blocks<1: break` that skips a stage ENTIRELY (not
   per-position) until enough bytes exist anywhere; matching that exactly requires a stage's FIRST
   activation to prime its refine cache with the FULL backlog of everything it missed (not just the
   newest byte) in one shot -- and, since a deeper stage can never be active while a shallower one
   isn't, that inactive-stage branch must `break` out of the per-stage loop entirely (not
   `continue`), or a later stage's backlog silently double-counts a not-yet-finalized upstream
   value. Both confirmed via direct logit comparison at each prefix length against `_generate_cascade`.

Net result: `generate_kv_cache` now does `O(1)` new attention work per generated byte (for the
dominant byte-level + refinement costs) instead of `generate_no_cache`'s full `O(L)` recompute,
with verified bit-exact equivalence, not an approximation.

## `qcute_zero`: merged `qcute_zero_parallel` back in as an opt-in (2026-08-22)

The query-vec/`parallel_decode`/`generate_blockwise` work (forked off into `qcute_zero_parallel`
earlier this session, see above) was cleanly merged back onto the real-KV-cache baseline as an
opt-in addition (`cfg.parallel_decode=False` by default) -- it was already isolated enough when
originally stripped (separate Config fields, one self-contained training-loss block, one
self-contained generation method, no shared state with `generate_kv_cache`'s caches) that
re-attaching it back onto the bug-fixed baseline was low-risk: re-generalized
`rope_cos_sin_for_positions`/`apply_rope` for batched per-row RoPE positions (needed by the
multi-block training loss, reverted when the fork was stripped), re-added `Config.parallel_decode*`/
`query_vec`/the training loss/`generate_blockwise`/argparse wiring verbatim from the fork.

Verified: (1) default path (`parallel_decode=False`) produces byte-identical behavior to before the
merge; (2) `parallel_decode` training (`Ks=(2,1)`/`(2,2,1)`, `parallel_decode_n_blocks` up to 3) and
`generate_blockwise` both still work; (3) critically, `generate_kv_cache`'s bit-exact equivalence to
`generate_no_cache` is UNCHANGED (0 failures across the same config sweep as before) -- the two
mechanisms are fully independent code paths that don't interact.

Motivation for merging rather than leaving them split: a single codebase now serves as (a) the
trustworthy baseline (exact bpb, exact incremental generation, no bound/approximation caveats) for
ordinary capacity/scaling experiments, while also (b) carrying the machinery needed to later
experiment with MTP-style drafting + speculative-decoding verification (draft `Ks[0]` bytes via
`query_vec` in parallel, then verify by re-running the now-cheap `generate_kv_cache` sequentially
and accept/reject at first divergence -- not yet built, the natural next step for the
`parallel_decode` mechanism, see the "is it just MTP" discussion, 2026-08-22) -- without needing to
maintain two diverging files. `qcute/qcute_zero_parallel/` is now redundant (kept as-is, not
deleted, in case the split-file history is ever useful) but no longer the active home for this work.

## `qcute_zero`: MTP-style clusters + real speculative decoding (2026-08-22)

**Generalized `parallel_decode` from `Ks[0]`-aligned blocks to arbitrary MTP-style clusters.**
`cfg.parallel_decode_cluster_len` (default 2, typical 2/4/8) replaces the hardcoded `Ks[0]` block
size; the cluster's start position is now sampled uniformly at random (`m` in `[1, L-C]`), no
longer required to land on a `Ks[0]` boundary. This is a deliberate loosening from the original
free-tier invariant: a cluster is no longer guaranteed zero-speculation-risk (it may span a code
boundary), which is fine *because* this mechanism is meant to be verified/corrected afterward
(MTP + speculative decoding), not committed outright the way the original free-tier design was.
`cfg.parallel_decode_n_blocks` independent clusters still get folded into one batch per step,
reusing the same `code_kv_cache` (unchanged mechanism from the earlier multi-block generalization).
Memory note (chat 2026-08-22): the query/self-attn side scales cheaply with cluster length
(`O(B·nb·C²·D)` compute, quadratic only in the small `C`); the real cost driver is `nb` itself,
since `h_code.expand(...).reshape(...)` materializes `nb` full duplicates of each stage's entire
code-KV history (not bounded by `C`) -- `fuse_window` bounds this if it becomes a problem.

**Merged `qcute_zero_parallel` back onto the real-KV-cache baseline** (see previous section) before
this generalization, so the cluster/MTP work landed directly on `qcute_zero` itself, not the
now-redundant fork.

**Built `generate_speculative`**: drafts `parallel_decode_cluster_len` bytes per round via
`query_vec` in one parallel shot (same free-tier drafting mechanism as `generate_blockwise`), then
verifies each drafted byte ONE AT A TIME against the real, exact incremental stepper (refactored
`generate_kv_cache`'s inner closure into a reusable `_make_incremental_stepper` factory so both
methods share the identical verification machinery, not a re-derived approximation of it) --
accept while the draft agrees with the model's own true greedy choice, reject-and-use-the-real-byte
at first disagreement, discard the rest of that round's draft (standard speculative-decoding
accept/reject-to-first-divergence). Verified: `generate_speculative`'s final output is **exactly**
identical to `generate_kv_cache`'s own greedy trajectory in every tested config (`Ks` in
`{(1,),(2,1),(2,2,1),(4,2,1)}`, `C` in `{2,4}`, varying prompt length) -- correct by construction,
since the draft can never actually diverge from ground truth, only save compute when it agrees.
Returns an optional `stats={"accept_rate", "n_draft_checks"}` -- `accept_rate` is the direct
empirical measurement of the "code-LM/query-vec acceptance rate" the whole cost/benefit tradeoff in
the original brainstorm depends on; not yet measured on a real trained checkpoint (only smoke-tested
on random-init models so far, where the accept rate is not meaningful).

Two training runs queued (2026-08-22, behind the running `bytelm_xs4_ctx1024_mtp1` baseline):
`ks21_overfit10k_paralleldecode.py` (simple, `Ks=(2,1)`, cluster_len default 2) and
`ks221_overfit10k_paralleldecode.py` (new, `Ks=(2,2,1)`, `parallel_decode_n_blocks=4`,
`parallel_decode_cluster_len=4`) -- results pending. Once trained, the real next step is measuring
`generate_speculative`'s `accept_rate` on an actual checkpoint, the first genuine test of whether
this whole MTP/speculative-decoding direction is worth its complexity in practice (see the earlier
"is complexity worth it" discussion -- deferred as a research priority, but now cheap to actually
measure since the code exists).

## qcute_zero: query_vec pruned, replaced with regular MTP heads (2026-08-22)

Diagnosed flaw in the query_vec/`parallel_decode` mechanism above: it is structurally nothing like
real MTP's density. Real MTP reuses ONE hidden state `h` and adds cheap extra linear heads on top
(zero extra attention FLOPs, pervasive -- every position, every step). `query_vec` instead spent one
full attention-stack pass (cross-attn + self-attn refine, per fuse stage) per drafted position, and
only covered `parallel_decode_n_blocks` sampled clusters per training step -- sparse and expensive
where MTP is dense and cheap.

Fix: pruned `query_vec`/`Config.parallel_decode*`/`generate_blockwise`/the query_vec half of
`generate_speculative` entirely from `qcute_zero.py`. Replaced with `Config.mtp_heads`/`mtp_weight`
(default `mtp_heads=1`, disabled) -- untied `nn.Linear(D, V)` heads reading the SAME final hidden
state (`x_cross`, post-cascade) that head0's own cond/uncond readout already reads, each predicting
a further-future byte (head i predicts offset `i+2`, since head0 already covers offset+1), summed
mean loss weighted by `mtp_weight`. Exactly mirrors `qcute.bytelm`'s own `mtp_heads` pattern.
`_generate_cascade` now also returns the raw final hidden state (`final_h`, third tuple element)
so `generate_speculative` can read it.

`generate_speculative` rewritten to draft via these heads (one forward pass, no per-slot attention
cost) instead of `query_vec`, verifying byte-by-byte against the same exact `generate_kv_cache`
stepper (`_make_incremental_stepper`) as before -- re-verified: output is still **exactly**
`generate_kv_cache`'s own trajectory after the swap (accept_rate=0.5 on an untrained random-init
smoke test, as expected for untrained heads).

The query_vec/cluster mechanism itself is not discarded -- it's preserved as its own standalone
testbed, forked onto the simpler `qcute.bytelm` trunk (no fuse-stage/code complexity to thread
through): `qcute/bytelm_queryvec/bytelm_queryvec.py`. Cluster slots there cross-attend directly
into the real trunk's own per-layer post-RoPE K/V (masked per-row to strictly-prior positions),
computed once per batch and shared across all sampled clusters -- denser real context than
`qcute_zero`'s version had (raw per-position K/V, not compressed codes), since there's no
compression step on this simpler trunk. Config: `configs/bytelm_queryvec/xs_overfit10k.py`.
Smoke-tested (forward/backward/`generate_blockwise` all run cleanly) but not yet trained for real.

Stale configs removed: `configs/qcute_zero/ks21_overfit10k_paralleldecode.py`,
`ks221_overfit10k_paralleldecode.py` (referenced now-removed `Config` fields). Replaced with
`ks21_overfit10k_mtp.py`/`ks221_overfit10k_mtp.py` (`mtp_heads=4`). The previously-queued
`queue_paralleldecode2.sh` training wrapper (waiting on `bytelm_xs4_ctx1024_mtp1` to finish) was
killed since its target configs no longer exist. `qcute/qcute_zero_parallel/` (the original
query-vec fork of `qcute_zero`) is left in place, now doubly superseded, kept only for historical
reference -- not deleted.

Verified via smoke test: default `qcute_zero` training (mtp_heads=1) unaffected; `mtp_heads=4`
training produces `mtp{2,3,4}_loss`/`mtp{2,3,4}_acc` metrics and backprops cleanly;
`generate_no_cache`/`generate_kv_cache`/`generate_speculative` all still produce bit-identical
output; `check_kv_cache_consistency` still reports `match_rate: 1.0`; a real 10-step training run
via `ks21_overfit10k_mtp.py` completes and logs the new metrics correctly.
