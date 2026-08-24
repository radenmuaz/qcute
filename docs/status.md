# qcute status

Pruned 2026-08-23 — the prior narrative (StackDecoderV1's interleaved-seed-token design through
early 2026-08-23 quant/`encoder_ste_p`/`byte_consistency_p` work, and all of `qcute_zero`'s own
foundational 2026-08-22 entries) is now archival, once this file passed 1300 lines with
`qcute_zero` established as a second fully-fledged active lineage alongside `qcute_v1`. Full
prior history: [docs/archive5/status.md](archive5/status.md) (the log this prune supersedes,
verbatim), older still [docs/archive4/status.md](archive4/status.md),
[docs/archive3/status.md](archive3/status.md), [docs/archive2/status.md](archive2/status.md).
For current architecture (not results), see `CLAUDE.md`'s Architecture section; for the
bpb-validity/codec-vs-predictive-LM formal writeup, see [docs/maths.md](maths.md). This file
tracks results/progress going forward, same convention as before: newest at the bottom,
session-dated.

## 2026-08-23: code one level above the top hard-excluded from decode conditioning (structural)

**Naming note (added retroactively, see this date's later "naming fix" entry): this section
originally said "the topmost level's own code" throughout, which is backwards** -- the code
excluded here is `c_list[n_levels-1]`, which under the corrected convention (a code is named by
the level that owns it as input, not the level that produced it) is "level n_levels's code": the
domain one level ABOVE the real topmost level, that nothing actually owns as input since no
`encoders[n_levels]` exists. Left the numbers/results below as originally written; only the
level-naming language is corrected here.

The `notoplevel` curriculum experiments above (`curriculum_max_srcs=(1,None)` for `ks21`,
`(2,1,None)` for `ks221`, held active the whole run) confirmed both hierarchies still overfit
cleanly, and `ks221` generated measurably better, with that one-above-the-top code excluded from
every level's cross-attention conditioning -- motivated by nothing being above the real top to
forecast/validate that code, so conditioning on it trains against a signal free-rollout
generation can't reliably reproduce.

Made this permanent and structural in `StackDecoder.__init__` (`qcute_v1_decoder.py`) rather than
an opt-in curriculum knob: `n_upper` now caps at `n_levels-2-i` (was `n_levels-1-i`), so the
cross-attn-stage LM and kv_lm for that one-above-the-top code are never even allocated -- genuinely
fewer params (`ks221`: 10.482M -> 6.988M), not just an unused capacity. `decode_level`'s
`j_max` capped the same way to match. Renamed `curriculum_max_srcs`/`curriculum_step` ->
`active_srcs_mode`/`active_srcs_until_step` (clearer name now that "exclude the topmost level" is
no longer this flag's job -- it remains for other phased-conditioning ablations). Backup of the
pre-change decoder at `qcute_v1_decoder.py.bak_notoplevel_20260823`.

**Bug caught while wiring this up**: `qcute_v1.py`'s `_run` unconditionally stored
`decode_derived_c[i] = result["code"]` whenever `max_srcs_i is None`, even when `result["code"]`
was `None` -- which now happens for the second-to-top level (it has zero upper tracks left, so
`decode_level`'s `code_final` never gets set). The `None` then got picked up by the level below
instead of falling back to the real ground-truth code (`c_list[j]`), crashing with `TypeError:
linear(): argument 'input' ... NoneType`. Fixed both write sites (main pass + `encoder_ste_p`'s
second pass) to only store the key when `result["code"] is not None`.

Requeued the full `ks221` `encoder_ste_p`/`encoder_ste_skip_real` ablation under the new
hardcoded-exclusion decoder (`_hardexcl` run names) to confirm no regression -- all four completed
without crashing, numbers track the pre-change curriculum-based runs closely (small run-to-run
noise, different code path but same effective masking):

| config | exact matches (train) | final train byte_acc |
|---|---|---|
| base (`encoder_ste_p=0`) | 10/50 (~20%) | 99.80% |
| `encoder_ste_p=0.1` | 21/53 (~40%) | 99.31% |
| `encoder_ste_p=1.0` | 31/51 (~61%) | 98.84% |
| `encoder_ste_p=0.5`, `encoder_ste_skip_real=True` | 18/50 (~36%) | 98.60% |

Same monotonic trend as before: more `encoder_ste_p` -> more exact-match generations, small
byte_acc cost. `skip_real=True` at p=0.5 lands between the two additive endpoints, consistent with
it being a fuller substitution than a mild additive nudge but less aggressive than full p=1.0
additive.

## 2026-08-23: StackDecoderLocal (fully parallel block-local decode) -- architecture question, then a real generation bug

Question raised (chat): given the topmost-level exclusion above, is a FULLY parallel decode
possible -- mask each block's same-level decode so it never sees any OTHER block's bytes at all
(cross-attend only to its own code, plus optionally a small window of neighboring codes from the
level above), so all blocks decode in one parallel batched call instead of sequentially? And does
teacher-forced bpb/loss stay a valid bound under that structure?

Confirmed this already exists (`StackDecoderLocal`, `--decoder_type stack_local`,
`block_local_track0_decode`) and generalizes to both level0 and level1 in `ks221` (only `_track0`
is overridden; `decode_level`'s n_levels handling is untouched, and since the code one level above
the top is now hard-excluded, level1 -- itself the level right below the real top -- already has
zero upper tracks regardless of decoder type). On validity: yes,
still a proper (Gibbs-inequality) bound on the per-byte conditional entropy -- BUT the codebase's
own `bpb`/`bpb_full` metrics only cover `decode_losses[0]` (level0's own reconstruction), never
`encode_losses[1:]` (the code streams' own transmission cost) -- a pre-existing gap that applies
equally to the sequential `StackDecoder`, not introduced by parallelism.

Queued three `ks221` `stack_local` configs (`v16pq4`, `kvlm_fresh`, topmost hard-excluded by
default): window=1 (own block's level1 code only), window=2 (own + 1 neighbor), window=2 +
`encoder_ste_p=0.1`. All three trained fine, no crashes, ~99.5% teacher-forced train byte_acc --
but **all three generated pure repetitive garbage** (`0/52`, `0/54`, `0/48` exact matches; qual
samples like `"srsr    irtrssisssssssssesereeeeeeeeeeeeeeeee"`, `"[[[[[[llii[[[annttiitiniuu"`),
despite byte_acc comparable to or better than the sequential decoder's runs. High teacher-forced
accuracy gave zero warning -- only checking real free-rollout samples caught this (the same
`bpb`/`byte_acc`-isn't-the-whole-story caveat from the validity discussion above, now with a
concrete instance).

**Root cause (confirmed by code inspection, not just capacity/multi-modality speculation)**:
`_stack_generate_blockwise` -- the method every generation path (`generate_no_cache`,
`generate_kv_cache`, and therefore every qual sample) actually calls -- hardcodes
`encode_like_self_attn_decode`/`seed_query_decode` (the sequential mechanism) for track0's
byte-by-byte generation, completely bypassing the `_track0` override point. `StackDecoderLocal`
only overrides `_track0`, which is used by `decode_level` (training) -- generation therefore ran
under a decode mechanism the model was never trained with. `block_local_track0_decode`'s own
docstring proves the seed token's same-block self-attention contribution is provably always
exactly zero during training (block-diagonal, causal, can't see anything before its own block's
start) -- but generation fed the seed real cross-block self-attention K/V from every prior block,
signal the model never learned to produce or consume. A genuine train/inference mismatch, not a
fundamental non-autoregressive-decode limitation.

**Fix**: added `StackDecoderLocal._stack_generate_blockwise`, an override that computes every new
block's track0 hidden state via `block_local_track0_decode` called with `n_blocks=1` on just that
one block's own bytes -- identical computation to training, zero cross-block visibility, matching
`_track0` exactly. Everything else (code sourcing via level1's own NTP, upper-track chaining) is
unchanged from the base class. Smoke-tested (30 steps, no crash, correct dispatch confirmed via
Python's dynamic `self._stack_generate_blockwise` resolution in `generate_no_cache`/
`generate_kv_cache`) -- too few steps to judge generation quality yet. Requeued all three configs
under `_genfix` run names for a full comparison against the `0/52`/`0/54`/`0/48` broken baseline.

**Result: fix confirmed working.** Exact matches went `0/52,0/54,0/48` -> `1/48,2/47,1/41`
(w1/w2/w2_ste01, all clean single-invocation logs), and more importantly the qual samples
themselves flipped from pure repetitive garbage to genuinely coherent English -- e.g.
`b"ers'' are unnecessary and lishZe fKre]erd thene they believent o"` and one exact-length
reconstruction (`b'8029</id>\n      </contributor>\n      <minor />\n      <comment>ad'`), vs. the
pre-fix `b"srsr    irtrssisssssssssesereeeeeeeeeeeeeeeee"`. Still well below the sequential
`StackDecoder`'s 20-60% exact-match range on the same `ks221` setup -- but that gap now looks like
a real capacity difference (block-local decode genuinely has strictly less cross-block information
available by design, per the validity discussion above) rather than a bug: partial matches now
degrade gracefully into plausible-looking word fragments instead of collapsing to repeated
characters. `encoder_ste_p=0.1` (w2_ste01) didn't clearly help here (1/41, similar to w1's 1/48) --
suspect the sequential decoder's `encoder_ste_p` win doesn't transfer cleanly to the parallel case,
but this is a single run each, not an isolated ablation.

## 2026-08-23: naming fix -- a code belongs to the level that owns it as input, not the level that produced it

Chat caught a real, systemic naming bug across this file, `CLAUDE.md`, `qcute_v1_decoder.py`'s
docstrings, and the architecture artifact: `c_list[N]` (the code array at codebase index N,
produced by `encoders[N]`) was being called "level N's own code" throughout. It should be "level
(N+1)'s code" -- provable directly from `Encoder.forward`'s own docstring ("seq_repr -- level0's
own bytes, or level j's own INPUT (= level (j-1)'s code stream) for j>0"): the encoder's own words
say `c_list[j-1]` IS level j's own input, i.e. a code is owned by whoever treats it as their input
domain, not by whoever emitted it. Concretely: level i's decoder never conditions on anything of
its own beyond what it's reconstructing (bytes for i=0) -- every code it cross-attends to,
including what this codebase's variable/function names call "track0"/"own code" (e.g.
`own_block_cross_attn_decode`), is really level (i+1)'s code. That function's "own" is a different,
still-correct concept (this SAME BLOCK's code, temporal/spatial locality within the stream) --
only the LEVEL-ownership claims were wrong, not every "own" in the codebase.

Consequence for the earlier `ks21`/`ks221` cross-attn-track counts discussed in chat: previously
stated as "ks21: level0 has 0 upper tracks (own code only), ks221: level0 has 1 upper track
(level1's own code)" -- both undercounted by one and mislabeled the remaining track. Corrected:
`ks21`'s level0 decoder cross-attends to exactly 1 code (level 1's code, via the always-present
track0 mechanism -- never removed by the 2026-08-23 hard-exclusion, which drops a *second*,
now-nonexistent track to level 2's code). `ks221`'s level0 decoder cross-attends to exactly 2
codes (level 1's and level 2's), with a third track to level 3's code excluded. General rule: level
0's decoder always has exactly as many tracks as there are real levels above it, with one further
track (to the code nothing owns as input) permanently excluded -- matches this date's earlier
"hard-excluded" entry above (now corrected to use this same language) and its `n_upper =
max(0, n_levels-2-i)` formula exactly.

Fixed: `StackDecoder`'s class docstring (`qcute_v1_decoder.py`, `cond_depth`/hard-exclusion
paragraphs rewritten with correct level-ownership language, plus a permanent naming note at the
top of the docstring), this file's two 2026-08-23 entries above, and the published architecture
artifact (`qcute_architecture.html` -- relabeled every `c` superscript, "own code" ->
"level N's code" throughout, corrected exclusion counts in both `ks21`/`ks221` figures and the
footer). Not touched: the ~40 other "own code"/"own-code" occurrences in `qcute_v1_decoder.py` and
`CLAUDE.md` that refer to block-locality (a level's own self-extraction, or "this same block's
code") -- those are a different, correct usage, verified case-by-case before leaving them alone.

## 2026-08-23: track0 window decoupled (self_window, cross_window); found a pre-existing unit mismatch

Chat wanted `ks41` (Ks=(4,1)) with level0's cross-attention into level 1's code capped to a
genuine FIFO window of 4 codes -- but `decode_windows[i][0]` was a SHARED knob (per StackDecoder's
docstring): the same raw number gated both track0's own byte self-attention window AND its
cross-attention window into level (i+1)'s code, no way to set them independently.

Implemented: `decode_windows[i][0]` now accepts either a scalar/None (unchanged, backward
compatible, broadcasts to both) or a `(self_window, cross_window)` 2-tuple (`qcute_v1.py`'s
`_norm_track0`). Threaded through every call site that used to write `track0_window, track0_window`
via a new `split_track0_window` helper (`qcute_v1_decoder.py`) -- `_track0` (training),
`_stack_generate_blockwise` (generation), and both diagnostic functions that build the same
context (`check_blockwise_gen_consistency`, `_decode_gt_context`) -- specifically so training and
generation windowing can't silently diverge, the same class of bug as the `StackDecoderLocal`
generation mismatch found earlier today.

**Pre-existing bug caught while smoke-testing (unrelated to the tuple feature)**: `ks21`-shaped
(n_levels==2) configs now crash in `check_blockwise_gen_consistency`/`_decode_gt_context` --
`self.stage_lms[0][1]` (bb1, the second stage) no longer exists once level0 has zero upper tracks
(the 2026-08-23 hard-exclusion). Both diagnostics assumed a bare 2-level model always has an upper
track, true before the hard-exclusion, false after. Fixed: both now skip gracefully
(`len(self.stage_lms[0]) < 2`) instead of crashing -- `check_roundtrip_consistency` was already
safe (never touches `stage_lms[0][1]` directly).

**Pre-existing SEMANTIC mismatch caught while regression-testing the print fix (predates today,
not introduced by the tuple feature)**: track0's cross-attention window
(`encode_like_self_attn_decode`'s `block_lag < win` mask) compares the raw `decode_windows[i][0]`
value directly against a CODE-COUNT (`block_lag`, a block-index difference) -- unlike every upper
track (`cross_attn_stage`), which compares against a raw BYTE-position difference, requiring
`n_codes = window // cum_K`. The old diagnostic print applied the byte-position formula
uniformly to every track, including track0 -- silently UNDERSTATING track0's real code-visibility
by a factor of `cum_K` for every "windowN" config ever run (e.g. `ks221`'s
`window16_relaxed`-family configs: printed "16codes" for level0's track0, actually 32 -- the
trained models' real behavior was never wrong, only the accounting/mental-model was). Fixed the
print (`qcute_v1.py`) to use the raw value directly for track0's cross-window component. Not
re-audited: whether any config's WINDOW VALUE was originally hand-picked based on the wrong
(understated) code-count -- out of scope for today, flagging for whoever revisits window sizing.

Built `ks41_v16pq4_overfit10k_level1_window4.py` (Ks=(4,1), current-standard `v16pq4`, `decode
attn_window[0] = (self_window=-1, cross_window=4)` -- level0's own byte self-attention stays
unbounded, level1's code visibility capped to the most recent 4 codes, FIFO). Smoke-tested clean
(`decode effective codes level0: K=4:4codes`, confirming the fix); queued for a full run.

## 2026-08-23: maximal encoder/decoder weight reuse -- decode_scope, kv_lm_mode/decoder_own_stage_mode defaults flipped, track0_kv_lm added, byte_head decoupled

Chat pushed the "how much of the decoder is genuinely new vs reused from the encoder" question to
its conclusion, in stages:

**`decode_scope`** (new `Config` field, default `"level0_only"`, alt `"pervasive"`): only level0's
`decode_level` runs by default -- nothing downstream ever consumes level i>0's own reconstruction,
since level0's upper-track cross-attention already conditions on `c_list[j]` (the ENCODER's
output) directly, available regardless of whether level j's own decode runs (the
`decode_derived_c` fallback already handled this). `pervasive` restores the original behavior
(every level's own decode runs, useful as its own diagnostic/training signal). Gated in
`qcute_v1.py`'s `_run` via a shared `decode_levels` range reused by both the main decode pass and
the `encoder_ste_p` second pass.

**`kv_lm_mode`/`decoder_own_stage_mode` defaults flipped to `"shared"`** (were `"identity"` and
`share_encode_decode_self=False`/"copy"): reasoning -- level i's cross-attn stage already has its
own dedicated submodule to consume a code, so kv_lm's/track0's job is just producing a good
*representation*, which the encoder that already models that exact sequence is the natural source
for, rather than training a redundant independent LM. `kv_lm_mode`'s third mode renamed
`"fresh"` -> `"copy"` (same behavior, all 19 existing configs setting it updated). Verified via
direct parameter inspection (not just docstring claims): with both new defaults, the *only*
decoder-namespace parameters not tied to an encoder are `stage_lms[i][1:]` (the `cross_attn_stage`
LMs for genuine upper tracks) -- for `ks21` specifically, that's **zero** parameters (level0's
decoder is entirely the encoder, reused).

**Bug found and fixed**: `make_kv_lm`'s `"shared"` mode reused `encoders[j].lm` to contextualize
`c_list[j]` -- but under the corrected naming (see the earlier 2026-08-23 naming-fix entry),
`c_list[j]` is "level `(j+1)`'s" code, owned by `encoders[j+1]` (which self-attends over it as its
own input), not `encoders[j]` (which self-attends over `c_list[j-1]`, a different sequence
entirely). A direct inheritance of the same pre-fix naming confusion, just embedded in indexing
logic this time instead of prose -- never caught by the earlier naming-fix pass since that pass
only touched docstrings/diagrams. Fixed: `encoders[j+1].lm`.

**Gap found and fixed**: track0's cross-attention target (`c_list[i]`, level `(i+1)`'s code) never
went through any kv_lm at all, regardless of `kv_lm_mode` -- only upper tracks (`stage_lms[i][1:]`)
got the optional causally-contextualized K/V; track0's code embedding went straight from
`quant.embed_for_decode` into cross-attention. Added `StackDecoder.track0_kv_lms` (one per non-top
level, same three modes, sharing `make_kv_lm`'s now-fixed encoder-choice logic) and wired it into
every place that computes track0's `code_embeds0` -- `decode_level` (training), the base
`_stack_generate_blockwise`, `check_blockwise_gen_consistency`, `_decode_gt_context`, and
`StackDecoderLocal`'s `_stack_generate_blockwise` override -- specifically so training and
generation can't silently diverge on this, the same class of bug as the earlier `StackDecoderLocal`
generation mismatch.

**A second, deeper gap found while verifying the above**: even with both `shared` defaults, the
output head used for decode's code-conditioned prediction (`ntp_loss_acc`'s `is_byte_level`
branch) was hardcoded as `F.linear(h, self.embed.weight)` -- so when `decoder_own_stage_mode=
"shared"` ties track0's whole backbone to the encoder, this ALSO ties the encoder's unconditional
byte-NTP head and decode's code-conditioned reconstruction head to the exact same tensor, even
though cross-attention+MLP had already transformed the hidden state into something semantically
different from the unconditional case -- no parameter existed that was specific to "predict a
byte given this code-informed hidden state." Fixed by mirroring the *already-existing* pattern for
the CODE-level head (`cfg.ntp_head_tied`/`self.code_predict`, untouched) with a new, analogous
byte-level one: `cfg.byte_head_tied: bool = False` (default untied) and `LM.byte_head`, a genuine
independent `nn.Linear(D, V, bias=False)` used via a new `LM.byte_output_weight` property (`self.
embed.weight` if tied, else `self.byte_head.weight`) -- the single source of truth every byte-
output call site (loss, generation sampling, the `embed_weight`/`embed_weight_final` dict fields)
now reads from, replacing ~19 hardcoded `.embed.weight` call sites across `qcute_v1.py`/
`qcute_v1_decoder.py`. Verified precisely (not assumed) that `cross_attn_stage`'s own LM never
calls `.embed_input`/`.embed(...)` at all -- its `embed` submodule was already vestigial as an
*input* embedding for that role, only ever touched via the old hardcoded output-tying; `byte_head`
gives it (and track0) a real, dedicated output parameter regardless. Since `byte_head` lives on the
shared `LM` class, this applies uniformly to every LM instance (every encoder, every decode stage)
with no special-casing per role.

**Allowed but warned**: combining `byte_head_tied=True` with `decoder_own_stage_mode="shared"` or
`kv_lm_mode="shared"` collapses track0's conditional and unconditional heads back into one tensor
(the exact gap just fixed) -- `StackDecoder.__init__` now prints a warning recommending
`byte_head_tied=False` in that case, without blocking it.

All changes smoke-tested clean (`ks21` `decode_scope=level0_only` default, `ks221`
`decode_scope=pervasive`, `byte_head_tied` on and off, the warning firing correctly) -- no crashes,
param counts move in the expected direction each time (independent heads/kv_lms add params, shared
ones remove them).

## 2026-08-23: why prefer qcute_v1's own-block reconstruction over qcute_zero's predictive fuse -- moved to docs/maths.md

Moved into `docs/maths.md` Part 10.1 (codec-vs-predictive-LM training objectives, alongside
Parts 9/10's paradigm framing) -- see there for the full writeup (why train as a codec at all
given it only yields an ELBO bound, and why predictive codes can't support "plan then fill"
generation).

## 2026-08-23: `own_code_min_lag` POC -- implementing docs/maths.md Part 8's retargeting, based on v1

Confirmed (chat) it can be implemented as an almost-pure reuse of `qcute_v1`'s existing code, not
a rewrite: `Encoder.forward`'s own per-level NTP was already exact/causal and needs no change at
all (Part 13). The *only* circularity lives in track0's own-code cross-attention mask
(`encode_like_self_attn_decode`/`seed_query_decode`, `qcute_v1_decoder.py`), which currently
admits `block_lag == 0` (a block's own code). Changed both functions' mask construction from
`block_lag >= 0` to `block_lag >= cfg.own_code_min_lag` (new `Config.own_code_min_lag: int = 0`
field, `qcute_v1_common.py`, plus `--own_code_min_lag` CLI flag) -- `min_lag=0` is the unchanged
default; `min_lag=1` excludes a block's own code, admitting only strictly-earlier ones, exactly
Part 8's $c_{i-1}$ retargeting. Both functions already took `cfg = bb.cfg` at the top, so no new
parameter needed threading through any of the 4 call sites (training `decode_level` +
`_stack_generate_blockwise` + `check_blockwise_gen_consistency` + `_decode_gt_context`) -- they
all automatically pick up the new behavior consistently, train and generation alike, closing off
any chance of reintroducing the train/inference mismatch class of bug found earlier this session.
Verified directly (pure tensor check, not just code reading): with `min_lag=1`, block 0 sees no
code at all (no earlier block exists) and block 1 sees only code index 0 (its own index 1 is now
masked out) -- confirmed exactly matching Part 8's intended semantics.

`encoder_ste_p` and the rest of the training-tricks machinery are untouched and compose
automatically: they operate on the code *tensor's values* (real vs. self-sampled), while
`own_code_min_lag` only changes the cross-attention *mask* (which code positions are visible) --
orthogonal concerns, no interaction to account for. Legacy `StackDecoderV1`'s
`own_block_cross_attn_decode` mechanism is untouched (out of scope, superseded lineage).

Smoke-tested clean on CPU (`--device cpu`, to avoid MPS contention with the sharing-ablation grid
still training at the time) for both `min_lag=0` (regression, no crash, byte_acc/bpb numbers
unchanged in kind) and `min_lag=1` (new, no crash, trains).

Queued a first POC pair, `configs/v1_causal_decode_poc/ks21_v16_pq4_overfit10k_minlag{0,1}_*.py`
(`Ks=(2,1)`, `v16pq4`, everything else identical between the pair, `encoder_ste_p=0` to isolate
this one knob cleanly) -- launched via a waiter script that blocks until the sharing-ablation
grid's driver process exits, so only one training job ever runs at a time. Expect `min_lag=1` to
show meaningfully higher train/val bpb and lower byte_acc than `min_lag=0` -- not a regression,
the expected cost (docs/maths.md Part 8/13) of retargeting reconstruction into genuine
next-block prediction. Results and qual-gen comparison TBD once both complete.

## 2026-08-23: fixed `qcute_v1`'s missing bpb term (docs/maths.md Part 4) -- `bpb_valid`/`bpb_full_valid`

Added `add_valid_bpb(result, cfg)` (`qcute_v1_common.py`, right after `add_per_level_bpb`):
converts each upper level's own code-NTP loss (`level{i}_ntp_loss_encode`, nats/code at that
level) to nats/byte via that level's `cum_K = prod(cfg.Ks[0:i])`, sums across levels, and adds the
result (converted to bits) on top of the existing `bpb`/`bpb_full` to produce new `bpb_valid`/
`bpb_full_valid` fields -- the actual valid ELBO-style bound Parts 3/5 derive (decode cost +
code-transmission cost), not just `decode_losses[0]` alone. Deliberately additive, not a
redefinition: `bpb`/`bpb_full` are untouched (Checkpointer and every existing config's historical
`best_val_bpb` still key off the old, decode-only field), the new fields are opt-in reads. Wired
into `eval_model`, `eval_model_full`, and the train-loop's per-`log_every` `train_scalars`.

Smoke-tested clean on CPU across `ks21` (n_levels=2) and `ks221` (n_levels=3, exercises the
multi-level `cum_K` accumulation), and against both `own_code_min_lag=0` and `=1`: e.g. `ks21`
`min_lag=0`: `val_bpb=7.93` vs `val_bpb_valid=11.00`; `ks221` `min_lag=0`: `val_bpb=7.88` vs
`val_bpb_valid=10.88`; `ks21` `min_lag=1`: `val_bpb=8.01` vs `val_bpb_valid=10.01`. In every case
`bpb_valid > bpb`, as expected (Part 4/11: the decode-only number always understates the true
bound by omitting the code's own transmission cost) -- and `min_lag=1`'s `bpb_valid` (10.01) is
notably closer to its own `bpb` (8.01) than `min_lag=0`'s gap (7.93 -> 11.00), consistent with
`min_lag=1`'s decode task already being harder/less-code-dependent so the code contributes
relatively less on top, though these are 3-step smoke-test numbers, not a trained comparison.

## 2026-08-23: added `qcute_zero.generate_free_rollout` -- the missing free-rollout piece (docs/maths.md Part 12)

Implemented the piece Part 12 identified as missing: `qcute_zero`'s code-sequence NTP
(`code_ntp_loss`) was already trained, but nothing sampled from it *ahead of* a chunk's own bytes
existing -- every prior `generate_*` method always extracted a fuse stage's code from the real
trunk hidden state at that chunk's own last byte. New `QCuteZero.generate_free_rollout` (n_fuse==1
only, single fuse stage; deeper cascades are future work): for each new chunk, runs the causal
code-sequence NTP over already-real chunks only, samples the chunk's own code from
`h_code[:, -1, :]` via the same `gumbel_quantize` head used everywhere else, then decodes that
chunk's K bytes one at a time cross-attending to this one pre-sampled code (own-chunk code, never
derived from its own bytes). Full recompute per new byte (like `generate_no_cache`, not the fast
KV-cache path) -- a correctness PoC, not optimized. Requires at least `K0` real prompt bytes (one
whole chunk) to bootstrap the very first sample; a true from-nothing cold start would need an
explicit null-code fallback for chunk 0 (not added here).

Smoke-tested on an untrained model first (shape/crash check only, `configs/qcute_zero/
ks21_overfit10k.py`, CPU) -- ran cleanly alongside `generate_no_cache` with matching output
shapes. Trained the same config for real (1000 steps, CPU, `run_name=freerollout_poc_train`,
`val_byte_acc≈0.38`, `val_fuse0_ntp_acc≈0.76` -- the code-sequence NTP the sampling step depends
on genuinely learned to predict well) to judge output quality on a real checkpoint.

**First real test: degenerate repetitive garbage** (`"EEEEpEEEEEEEEEEEE..."`), while
`generate_no_cache` on the same checkpoint produced plausible wiki-markup English -- the exact
loss-good-but-generation-bad signature this session already has a standing rule for (inspect the
generation code, don't assume an architecture limit). Root cause, found by inspection: a genuine
off-by-one. `logits[:, chunk_start + t, :]` was used to predict `buf[chunk_start + t]` itself --
but per the predict-next convention used everywhere else in this file (`forward()`/
`generate_no_cache`: position `p`'s logits predict byte `p+1`), position `chunk_start + t` is
`buf`'s own not-yet-decided placeholder (still zero-embedded) at exactly the moment its byte is
being chosen -- the model was selecting each new byte from the hidden state produced BY that same
still-undetermined (zero) byte, not from the last real position before it. Fixed:
`logits[:, chunk_start + t - 1, :]`. Retested on the identical checkpoint/prompts: `generate_free_
rollout` now produces plausible continuations comparable in quality to `generate_no_cache`'s real
output (e.g. prompt `"...anarch"` -> `"ism. [[Bertrand Russell]], is his '''&quot;any act that
used vio"`) -- genuinely working, not just non-crashing.

**Constraint found while building this (2026-08-23 chat, "can rollout init byte decode
mid-sentence without prior bytes"): no, not with the current masking, but it degrades gracefully,
not by crashing.** Checked directly: `F.scaled_dot_product_attention` with an attn_mask row that's
entirely `False` (no valid keys at all) returns exact zero, not NaN -- confirmed via a standalone
tensor test. So under `own_code_min_lag=1`, the very first `min_lag` blocks (which structurally
have no valid earlier code to attend to -- confirmed earlier via the direct `block_lag` tensor
check) silently get a zero cross-attention contribution rather than crashing, i.e. an implicit,
untrained "null code" -- not a deliberate, trained representation of "no information yet." A real
from-scratch (no-prompt) generation would want an explicit, *trained* null-code embedding
substituted for these blocks instead of relying on this incidental zero-fallback. Every actual
generation path in this codebase currently requires at least one real block of prompt for exactly
this reason (`all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]`, then asserts
non-empty) -- true zero-context "mid-sentence" init isn't attempted anywhere yet.

## 2026-08-24: `qcute_zero` wavefront lockstep decode — draft (unverified) + drafted-and-verified (`generate_wavefront_mtp`) + training-time loss

**Mechanism**: `wavefront_mask(timestep, region)` (module-level fn) splits a K-byte block into
`n_waves` regions of `region_len=K/n_waves`, decoded in lockstep "timesteps": a query can attend
to any strictly-earlier-timestep key regardless of region, plus same-timestep keys in its OWN
region only (parallel across waves, causal within a wave). Bootstrapped with no seed token: wave
0's first token is the ordinary head0 readout off `h_last`; wave `g>=1`'s first token comes from
`extra_heads[g*region_len-1]` (offset `g*region_len+1`), so `cfg.mtp_heads >=
(n_waves-1)*region_len+1` is required. Degenerates exactly to plain AR when `n_waves=1`
(`check_wavefront_consistency`, `match_rate=1.0`), and — a fact derived and confirmed this
session — **also degenerates exactly to plain MTP-heads drafting when `region_len=1`** (e.g.
`n_waves=K`): with no lockstep loop to run, the "block" is just the bootstrap step, same heads,
same source hidden state (`h_last`), byte-identical draft (confirmed on matched prompts: accept
rate and drafted bytes identical on 15/15 trials). The wavefront-specific independence assumption
only bites once `region_len>1`.

**`generate_wavefront`** (unverified): drafts a full K-byte block via the lockstep mechanism, no
correction against the true model. `buf`'s layout is already true left-to-right byte order (wave
`g`'s slots sit at absolute positions `P+g*region_len..P+(g+1)*region_len-1`), so no reordering
needed. Refactored the per-block draft logic into `_wavefront_draft_block`, shared with:

**`generate_wavefront_mtp`** (drafted-and-verified, the actually-useful mode): drafts a K-byte
block the same way, then verifies it byte-by-byte in true order against the exact incremental
stepper (`_make_incremental_stepper`) — same accept/reject-to-first-mismatch scheme
`generate_speculative` already uses for plain MTP drafts, so output is guaranteed byte-identical to
`generate_no_cache` regardless of draft quality (the speculative-decoding correctness guarantee).
Verified exact on every trial run this session (15/15, both checkpoints, `n_waves` ∈ {2,8}).

**Training-time wavefront loss** (`Config.wavefront_weight/wavefront_K/wavefront_n_waves`,
default off): `wavefront_mask` was previously used ONLY at generation time — `forward()`'s loss
never exercised it, so generation-time lockstep hidden states were out-of-distribution for the
trunk's self-attention. New loss tiles the block-local timestep/region structure across the WHOLE
training sequence in one pass (block `b`'s timestep = `b*region_len+local_j`, keeping blocks
strictly ordered relative to each other) and adds an ordinary next-byte CE loss at every
within-block position except each region's own last local step (whose "next" would cross into a
different region) and excluding each region's own first local step (the MTP-bootstrapped position,
already covered by `mtp_heads`' own loss). Confirmed via direct isolation test (feeding the
lockstep-fill step the TRUE ground-truth bootstrap token rather than a predicted one, scoring
against real val data): lockstep-fill accuracy alone improved **0.354 -> 0.392** after adding this
loss (`ks1_overfit10k_wavefront2` vs `_trained`, `K=8/n_waves=2/mtp_heads=5`, overfit10k scale).

**But this did NOT make unverified `generate_wavefront(n_waves=2)` free-run output exact or even
close to `generate_no_cache`'s own trajectory** (byte-match rate ~0.13-0.14 either way, no
measurable change) — diagnosed as bottlenecked by the BOOTSTRAP step, not the lockstep mechanism:
`val_mtp5_acc≈0.15-0.16` (the far-offset MTP head bootstrapping wave 1's first token) badly
overfits train and doesn't generalize to val, and since wave 1's entire region is built on that one
weak prediction, its error dominates and swamps the lockstep mechanism's own real (if modest)
improvement in the full free-run comparison. Increasing `mtp_heads` 5->8 (`ks1_overfit10k_
wavefront2_mtp8`, also testing whether extra head capacity/offset coverage helps) did NOT fix
far-offset generalization either — `val_mtp2_acc≈0.25` down to `val_mtp8_acc≈0.12`, same decay
shape as the 5-head run.

**Clean same-checkpoint, same-prompt accept-rate comparison** (`ks1_overfit10k_wavefront2_mtp8`,
15 matched trials, all exact 15/15 via the verifier): plain mtp `accept_rate=0.775`, wavefront-mtp
`n_waves=8` (degenerate, `region_len=1`) `accept_rate=0.775` (identical, as derived above),
wavefront-mtp `n_waves=2` (trained lockstep, `region_len=4`) `accept_rate=0.827` — the ONLY real,
reproducible gain from this whole mechanism is the trained lockstep refinement at `region_len>1`;
`n_waves=K` buys nothing over plain MTP since it IS plain MTP. (An earlier same-session in-chat
report of `n_waves=8` beating plain mtp, 0.712 vs 0.682, was a measurement artifact — the two
loops drew different random prompts via separate `torch.randint` calls — corrected once caught.)

**Bottom line**: `generate_wavefront_mtp` is a correct, exact, and (at `region_len>1`, once
trained) genuinely-better-than-plain-MTP draft mechanism. Unverified `generate_wavefront` itself is
not a productive target for exactness at this scale — it's fundamentally gated by the same
far-offset MTP generalization ceiling every bootstrap-style mechanism in this file hits, not by the
wavefront mask/loss itself.

New configs: `configs/qcute_zero/ks1_overfit10k_wavefront2_trained.py` (adds `wavefront_weight`),
`configs/qcute_zero/ks1_overfit10k_wavefront2_mtp8.py` (`mtp_heads=8`). `scripts/
qual_wavefront_check.py` now shows all four modes (ntp/mtp/wavefront-ntp/wavefront-mtp) side by
side with accept-rate and exact-match-vs-ntp for the verified ones.

## 2026-08-24: `generate_early_exit`, `own_block_seed_weight` (stack_local, shift-by-1), `generate_free_rollout` causality fix

**`generate_early_exit`** (`_generate_cascade_early_exit`): confidence-threshold skip of deeper
fuse stages, using a shallower stage's own real cond prediction (not a blind uncond guess) when
its softmax max-prob clears the threshold. No exactness guarantee (documented explicitly, unlike
the verified drafters). On a genuine 2-stage `ks221` cascade (`ks221_overfit10k_mtp8`,
`Ks=(2,2,1)`), `threshold=0.9` skipped stage 1 entirely on 27/40 positions while still matching
`generate_no_cache` exactly on that trial — a real, measured compute saving, not just a PoC.

**`own_block_seed_weight`: stack_local-style block-diagonal decode, generalized to every fuse
stage, then corrected for validity twice.** Initial version (single stage, interleaved
seed+bytes under one FULL causal self-attention mask) let a seed see every real byte from every
earlier block — decoding a block still needed the entire prior byte history, no locality/speed win
at all (chat: "no speedup vs usual causal ntp"). Fixed via `qcute_v1`'s `StackDecoderLocal`
trick (`block_local_track0_decode`): fold `n_blocks` into the batch dimension, run ordinary causal
self-attention within each `cum_Ks[s]`-length block independently — true block-diagonal, LOCAL
positions, zero cross-block byte visibility (directly verified: perturbing block 0's bytes leaves
every block's seed hidden state, including block 0's own, bit-identical *before* cross-attention —
matches `qcute_v1`'s own proof that a block-local seed's self-attention contribution is exactly
zero). `code_window` as a separate hparam was removed — `cfg.attn_window` is reused directly for
the code cross-attention's window (same byte-position units, no conversion).

**But the block-diagonal version still cross-attended to the block's OWN code** (`h_code_s` at its
own position, `code_pos_abs` reused as both query and key for an "offset 0" trick) — which is
circular for a real bpb claim: that code is pooled from the block's own real bytes, including the
very byte being predicted (see chat's `Ks=(2,1)`, `abcdef` walkthrough: `code(e,f)` used to predict
`e` is using the answer, plus more of the future on top, to predict itself — not even a valid
bound, since the conditioning set isn't a subset of the past). Two ideas floated for fixing this
without abandoning validity — STE-on-the-predicted-code (single sample, causally clean but still
Jensen-biased vs. the true marginal) and exact marginalization (sum over the full code
distribution, provably exact but costs `n_blocks × vocab^pq_chunks` extra forward passes per
step — `~2.1M×` for `v16pq4` at `context_len=256`, clearly infeasible; `~8192×` even for flat
`v256pq1` — both ruled out as impractical) — were superseded by a much simpler fix (chat: "cant
just you shift-by-1 this stack local, just like mtp... dont follow v1 as is, mod it so that it is
shift by 1 and causal"): cross-attend to `code_{b-1}` (the PREVIOUS block's real code, offset 1,
exactly what the ordinary non-own-block cond path already uses for a block's first byte) instead
of the block's own code. `code_{b-1}` is a real, already-causal, deterministic quantity — no
sampling, no marginalization needed, and the result is EXACT (not a bound): every conditioning
variable is now a deterministic function of strictly-earlier real data. Block 0 has no `code_{-1}`,
excluded from the seed loss (`n_blocks_s>=2` required). Directly re-verified block-diagonal
isolation AND the shift's forward-only propagation: perturbing block 4's bytes leaves blocks 1-4's
seed predictions bit-identical and changes blocks 5+'s, exactly the expected causal shape.

Confirmed `own_block_losses`/`wavefront_loss` were never part of the reported/canonical bpb metrics
(`final_loss`/`uncond_loss`/`cond_loss`) in the first place — both are auxiliary `total_loss` terms
with their own weights, so neither corrupted the "official" bpb number even before this fix; the
fix matters for the mechanism's OWN validity (own-block-seed-based generation), not for the
headline bpb reported elsewhere.

**`generate_free_rollout` rewritten a third time to match** (causality fix, then true locality,
then shift-by-1): since `code_{b-1}` always already exists once block `b-1` is real, the code-level
NTP's SAMPLING step (`sample_next`) is no longer needed at all -- "free rollout" is a bit of a
misnomer now (nothing is sampled ahead of its own bytes) but the mechanism's actual value survives:
block-LOCAL self-attention (fresh, bounded `K+1`-sized cache every block, thrown away after) +
cheap cross-attention to the short code-level history -- decoding block `b` costs `O(K)` +
`O(n_blocks_so_far)` for the code history, never `O(L)` full self-attention over the raw byte
history. Two real bugs caught via full-recompute consistency checks while building this: (1)
conflating the LOCAL hidden state with the GLOBAL one for code EXTRACTION (the code-level LM was
trained on codes extracted via the ordinary GLOBAL `cur_h[:,K-1::K,:]` convention, a completely
separate computation from the local decode mechanism — mixing them broke consistency 6/8 before
the fix, 8/8 after); (2) `hc_hist`/`cpos_hist` (the cross-attention KV) never grew across the outer
loop after the shift-by-1 rewrite, so every new block was seeing only the prompt's original codes
— caught via a multi-block (6-block) consistency re-check, fixed by appending each new block's own
code-level output/position every iteration, re-verified 8/8.

`n_fuse==1` only; needs `>=2*K0` real prompt bytes now (shift-by-1 needs a real `code_{b-1}`, i.e.
at least one complete prior block beyond the very first). Not yet re-trained/quality-tested on a
real checkpoint under this final design — `ks81_overfit10k_ownblock`'s existing checkpoint used
the pre-shift-by-1 (circular) mechanism and needs a rerun.

## 2026-08-24: `own_block_seed_weight` renamed to `blocklocal_seed_weight`

Caught as misleading (chat: "but is it still ownblock, misleading") once shift-by-1 landed: the
mechanism no longer cross-attends to a block's own code at all (that was exactly the circularity
just fixed) -- it cross-attends to the PREVIOUS block's code. The defining, still-true property is
the block-diagonal self-attention (the actual source of the locality/speed win), not anything
"own". Renamed throughout: `blocklocal_seed_weight`, `blocklocal_dual_mode`,
`blocklocal_losses`/`blocklocal_accs`, `blocklocal{s}_loss/acc` metrics,
`configs/qcute_zero/ks81_overfit10k_blocklocal.py` (renamed from `_ownblock.py`,
`run_name=qcute_zero_ks81_overfit10k_blocklocal`). Re-verified forward()/backward still work
post-rename before relaunching training.

## 2026-08-24: `generate_free_rollout` first real quality check (post shift-by-1)

`ks81_overfit10k_blocklocal` trained 1000 steps, converged as expected for overfit10k:
`train_byte_acc≈0.987`, `train_blocklocal0_acc≈0.78-0.81` (val is low as always at this scale, not
a generalization test). Compared `generate_free_rollout` against `generate_no_cache` (exact
reference) on 2 held-out prompts, 32 new bytes each: `generate_no_cache` exactly reproduces the
memorized training continuation both times (as expected). `generate_free_rollout` diverges from it
early (~55% exact byte match both prompts) but stays qualitatively plausible -- output is still
well-formed wiki-XML-like fragments drawn from elsewhere in the training corpus (e.g.
`<text xml:space...`, `<namespace key="10">`), not garbage/repetition. Verdict: the mechanism is
wired correctly (causally valid per the shift-by-1 fix, block-local decode confirmed working
end-to-end) but not yet high-fidelity at this scale/step budget -- consistent with
`blocklocal0_acc` topping out around 0.8 rather than ~0.99 like the main byte-level NTP loss.
Script: `check_free_rollout.py` (scratchpad, not checked in) -- loads `last.pt`, runs both
generators on the same prompt, reports byte-level exact-match rate.

Follow-up diagnostic (chat: "did you check what if you input the gt codes... to sanity generation
code"): split `generate_free_rollout`'s new bytes into seed (first-of-block, cross-attends to real
`code_{b-1}`) vs local (rest-of-block, autoregressive within the block) and compared against
`forward()`'s own teacher-forced `blocklocal0_seed_acc`/`blocklocal0_local_acc` (added as new
metrics keys this session, same pattern as the existing `blocklocal0_acc`). Free-run seed_acc
(~0.5) tracks teacher-forced seed_acc (~0.57) reasonably well; free-run local_acc collapses to
~0.036 vs teacher-forced ~0.68 -- looked like a generation bug at first. Ruled out by teacher-
forcing the TRUE bytes into `generate_free_rollout`'s own incremental local-decode code path
(same KV cache, same `forward_incremental` calls) instead of its own argmax: reproduced ~0.71,
matching training almost exactly, confirming the code is correct. Root cause is a genuine training
gap: `blocklocal_local_loss` is 100% teacher-forced (always fed ground-truth within-block bytes),
no scheduled sampling, so once free-running commits one wrong byte mid-block every later
prediction in that block is conditioned on a context distribution the model never saw -- the same
compounding-error pattern as `qcute_v1`'s original free-rollout collapse, now reproduced inside a
single block rather than across blocks. Not yet fixed (would need scheduled sampling or a
free-running local-decode loss term analogous to how `own_block_decode_loss`/`code-level` training
addressed the cross-block version).

## 2026-08-24: `blocklocal_glat_p` -- GLAT-style scheduled sampling for the local-decode gap

Implemented the fix for the above: a second, additive local-decode pass (chat: "swap sampling by
input gt teacher force, argmax out all output, then with probability swap ... re input back") --
pass 1 (unchanged, fully teacher-forced) computes `local_logits`; with per-position probability
`cfg.blocklocal_glat_p`, pass 2 reruns the SAME block-local self-attention on an input where each
within-block byte is swapped for pass 1's own argmax prediction at that position, via an STE
(`pred_hard.detach() + pred_soft - pred_soft.detach()`, `pred_soft = softmax(logits) @ embed.weight`)
so pass 2's loss gradient flows back into pass 1's `local_logits`/head -- both passes' losses always
summed (additive, not skip-real, matching `encoder_ste_p`'s more-stable variant per this file's
existing ablation). New metrics: `blocklocal{s}_seed_acc`, `blocklocal{s}_local_acc`,
`blocklocal{s}_glat_acc` (all correctly 0 under `no_grad`/eval, verified). Unlike `encoder_ste_p`
(bridges a genuinely discrete quantized code), this needs no bridge for the *forward* value at all
-- `argmax` already blocks gradient on its own; the STE here exists purely to route gradient
INTO pass 1's logits, which plain `argmax` + embedding lookup would not do.

**Real bug caught before the first real run**: `blocklocal_glat_p` was added to `Config` but never
registered in `build_argparser`'s CLI args nor threaded through `config_from_args` -- `set_defaults`
filters the config file's attributes by `{a.dest for a in p._actions}`, so the config file's
`blocklocal_glat_p = 0.2` was silently dropped and the first `ks81_overfit10k_blocklocal_glat02`
run trained with GLAT fully disabled (`glat_acc=0.0` at every step, config bug not a metric bug --
confirmed via a smoke run with `torch.is_grad_enabled()` explicitly True). Fixed by adding
`p.add_argument("--blocklocal_glat_p", ...)` and `blocklocal_glat_p=args.blocklocal_glat_p` to
`config_from_args`; a 20-step smoke run afterward shows `glat_acc` tracking `local_acc` closely and
rising together, confirming the fix. Re-launched the real run.
