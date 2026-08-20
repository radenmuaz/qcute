# qcute_v1: decoupled-decode rewrite plan

New lineage, forked from `qcute_v5` (copied verbatim into `qcute/qcute_v1/`, files/imports
rewritten `qcute_v5*` -> `qcute_v1*`, smoke-tested standalone). `qcute_v5` stays as-is, frozen,
the leaderboard's source of truth for everything already run. `qcute_v1` is where the latent-AR /
parallel-block-local-decode investigation (see `CLAUDE.md`) actually gets implemented, starting
over from a clean numbering rather than retrofitting v5's decoder further.

## Core reframing

Today (v5): every non-top level's decode is an NTP LM -- predict byte t+1 given bytes <=t plus
cross-attended coarser codes, target shifted by `K` (block b's code+bytes predict block b+1).
Self-code is the mechanism that lets decode see beyond its own block via a recurrent
memory-token chain -- exactly the thing blocking parallel decode, since block b+1 can't start
until block b's actual code exists.

New idea: split responsibility by level.
- **Top level only** stays a genuine NTP/AR LM over codes (cheap, coarsest sequence -- this is
  unavoidably sequential, and that's fine).
- **Every level below top** becomes an autoencoder: given a code that summarizes its OWN chunk
  (not a hint about a past chunk), reconstruct that chunk directly. Training teacher-forces the
  real code; generation feeds the level-above's own causally-drafted code instead (same
  exposure-bias approximation decode already makes today, not new).

**Encode-side NTP loss is kept, unchanged, at every level including non-top ones.** This
reframing only changes what `decode_losses[i]` trains for non-top `i` (NTP -> reconstruction);
`encode_losses[i]` (each level's own next-code prediction over its own sequence, `code_ntp_weight`
-gated, already existing) stays exactly as today and keeps shaping the code space into something
structured/predictable, which is what makes the level-above's own NTP over those codes (and hence
the whole autoencoder-decode chain, since generation feeds the level-above's *predicted* code
into decode) work at all. Both losses stay in the mix for every non-top level; only decode's
objective changes.

Worked example (`Ks=(2,1)`, bytes `abcd`): level0 encodes `ab`->c1, `cd`->c2 (unchanged). Level1
does ordinary NTP over `[c1,c2]`: query at c1, predict c2 (unchanged -- this is already
`level1_ntp_loss_encode`). Level0 decode is then reframed as: given c2 (real in training,
level1-predicted in generation), reconstruct `cd` -- target unshifted, aligned to the same block
the fused code came from, not the next one.

**Causal**: holds. Causality is enforced by where the conditioning code comes from (real at
train time, causally-generated at inference time), not by within-chunk byte ordering.

**KV-cache**: generation loop changes shape (chunk-at-a-time: draft next code at the top level,
reconstruct `K` bytes, append, repeat) rather than byte-at-a-time with a growing cache.
`check_gen_consistency`'s invariant still holds, but `generate_no_cache`/`generate_kv_cache`/
`validate_generation` need restructuring around chunks, not bytes. This is the actual size of
the rewrite (stage 2 below).

## Decoder architecture (implemented)

Went through several false starts before landing here: (1) a standalone learned-query
`autoencode_decode` function cross-attending only to a code, scrapped; (2) "self-attend on own
code, cross-attend upper code," also wrong -- there is no own code in decoding's self-attention
at all; (3) a shift-by-1-NTP + BOS-stripped-from-output version (first "implemented" pass, ran the
four full-scale leaderboard numbers below) that turned out, on inspection prompted by a
`check_gen_consistency` bug hunt, to never actually let a block's own code inform reconstruction of
that SAME block's content -- `cross_attn_stage`'s `code_pos <= query_pos` mask (code_pos = a
block's own LAST byte) meant a block's code only ever became visible starting at its own last
position, i.e. useful only for predicting *later* blocks, functionally reducing decode to "hint
from a past code" (v5's original mechanism, just same-level instead of coarser-level) rather than
genuine reconstruction; (4) briefly tried collapsing BOS to a single global marker (matching v5
exactly) to fix a `check_gen_consistency` truncation bug, but that fix was solving a
diagnostic-only problem at the cost of BOS's real job and was reverted. The actual, current design
(confirmed against the worked example, `Ks=(2,1)`, `abcd`, `c1`/`c2`):

- **Self-attention**: plain causal self-attention over the ACTUAL sequence with a trainable
  **per-block SEED TOKEN prepended before every `K`-block** (`bb.self_code_const`, repurposed --
  neither "BOS" (implies a single sequence-start marker; this recurs every block) nor "sink"
  (implies a passive fallback key that itself predicts nothing; this is a full token -- see below
  -- fits, see chat 2026-08-20) -- `[seed,a,b,seed,c,d]`, not `[a,b,c,d]`. Implemented as
  `bos_interleaved_self_attn` in `qcute_v1_decoder.py`: reshape into `(n_blocks, K)`, concat the
  seed token at the front of each block, flatten, run ordinary RoPE'd causal self-attention
  (`window=None` = sync/unbounded, one continuous chain across every block, today's default;
  `window=K+1` = async ablation, block-local only, reuses `chunked_windowed_attention`).
  `strip_bos=False` keeps the seed-token positions in the output (needed for the next bullet);
  `strip_bos=True` (unused by decode_level as of this revision, kept for any caller that only
  wants the extra-key effect) drops them.
- **Cross-attention**: `own_block_cross_attn_decode`, NOT `cross_attn_stage` (that function is
  still used for the top level's coarser-code case, unrelated to this). The seed token's own hidden
  state is a genuine query here, not discarded -- `code_pos` is set to each block's own seed-token
  position (not its last byte), so a block's own code (`c1` for `ab`, `c2` for `cd`) is visible to
  EVERY position in that block, seed token included. Uses a *separate* LM instance (`bb_cross`) from
  the self-attention stage (`bb_self`), matching v5's `decode_cross_stage_layers` convention.
- **Target**: `own_block_decode_loss` -- UNSHIFTED, one query per real byte of a block (the seed
  token seeds the block's own first byte; each subsequent real byte seeds the next), so
  concatenated across blocks the target is exactly `x_list[i]` itself, not shifted by 1. This is
  the reconstruct-from-own-code framing the design always intended, now actually realized: `c1`
  reconstructs `ab`, `c2` reconstructs `cd`, no lag. Confirmed empirically (code-sensitivity probe,
  2026-08-20): perturbing a block's own code now changes that block's own reconstruction, not just
  later blocks'. Reconstructing a block from its own REAL code during training is standard
  VQ-VAE/discrete-autoencoder teacher-forcing, not a leak -- the real generalization test is the
  gt-vs-pred gap in `check_decode_modes`, i.e. whether a level-above-PREDICTED code (available the
  instant a block starts, no access to that block's real bytes) is ALSO informative enough, not
  whether training on the real code is somehow cheating.
- **`n_levels==1`/top level**: untouched -- still v5's original genuine self-code-recurrent NTP
  decode, `merged_decode_forward`, no seed token involved there.
- **`n_levels>=3`**: only the immediate own-level code is cross-attended so far; whether/how a
  third-level decode additionally cross-attends something coarser than its own code is open, not
  yet generalized.

**Scheduled sampling** (`Config.scheduled_sampling_p`, default 0.0): with this probability (one
flip per forward pass, training only, `torch.is_grad_enabled()`-gated), non-top-level decode's
cross-attended code is swapped from the real `c_list[i]` to level `i+1`'s own sampled prediction
of it (`quant.sample_next` on `h_list[i+1]`, shifted by one to match NTP alignment, position 0
falls back to real since nothing predicts it yet). Closes some of the train/generation
exposure-bias gap on the code decode actually depends on -- directly motivated by the "can the
upper-level LM's code prediction alone support real generation, or is it too vague" concern (see
generation-feasibility discussion below).

**The seed token generalizes to every level except possibly the top** (not just where discussed above) --
every level has its own encode-side NTP model over its own code sequence, and at generation time
that's what has to draft level `i`'s next code before decode can use it (no real bytes exist yet
to run encode on). Whether top is actually exempt from needing one, or has the identical bootstrap
problem, is still open.

## Generation feasibility (not yet implemented, stage 2 concern)

Training is sound either way (autoencoder-flavored decode + unconditional encode-side LM are both
well-posed training objectives). The open question is whether decode conditioned on a code can
actually generate NEW content, since decoding from a block's own REAL code is definitionally
autoencoding (reconstructing something already known) -- true generation needs a code for content
that doesn't exist yet, "always leading one block ahead" (draft `c2` from `c1` before decoding the
`cd` block, never decode `ab` from `c1` as if generating it). Two paths to obtain that code:
- **(b) Upper-level LM predicts it directly** (`c1 -> predict c2`, `generate_level_codes` already
  does this) -- cheap, but risky: asks the coarse level to forecast fine content through a lossy
  bottleneck with no lookahead. Untested whether it's informative enough to beat the unconditional
  baseline; cheap to check once training works (compare decode-from-predicted-c2 against
  `generate_encode_only`'s pure uncond output).
- **(a) Draft via the uncond byte LM, encode the draft, decode-refine** -- grounds code generation
  in something already known to work (plain NTP), sidestepping (b)'s forecasting risk. Not
  circular: the draft is genuinely new content, the code is a real function of it, refine is a
  legitimate second pass. Also more parallel-friendly (draft and encode are both parallelizable
  per block) -- may be the cleaner path to genuine parallel generation rather than just a fallback.
- `Ks=(1,)` (`n_levels==1`) is the degenerate test case: no level above, so path (b) is
  unavailable by construction -- generation can ONLY extrapolate via path (a). Decode there is
  "truly autoencode" and cannot produce anything new on its own. Good minimal test of path (a)'s
  mechanics (causal, static shape both hold -- refining a block using its own model-generated
  draft, including joint information across the block's own positions, is legitimate
  self-consistency revision, not leakage, since nothing involved is true unseen ground truth)
  before testing whether it adds real value on `Ks=(2,1)` where a code can be joint over >1 byte.
- **Eval must run both modes** once implemented: ground-truth code (upper bound on decode quality)
  vs. upper-LM-predicted code (realistic generation proxy) -- needs to generalize to `n_levels>2`,
  which may have more mode combinations (which levels use predicted vs. real codes) than the
  `n_levels=2` binary choice.

## Knobs considered and dropped (redundant on inspection, not implementing)

- `code_window` (how many blocks decoded per parallel pass) -- not a model parameter, a
  generation-time scheduling detail derivable from the self-attention window and `K`.
- Lag `D` / per-distance tuple / fraction-of-K parameterizations for cross-block round-sync --
  all redundant with the self-attention window above, which already spans the full parallel <->
  sequential range via ordinary windowed causal attention. No new mask shape needed for the
  binary sync/async choice.
- `use_self_code` (carried into the `qcute_v1` copy from v5, config field still present):
  moot for non-top levels -- there is no own-code-in-self-attention choice left to toggle.

**Knob kept, orthogonal**: `track_dropout_p0/ramp_steps/schedule` -- cross-*level* sparsity (how
many/which upper levels get cross-attended), only relevant for `n_levels>=3`, a different axis
from the self-attention window question above.

**Later idea, not yet planned in detail**: the sync/async self-attention window choice doesn't
have to stay strictly binary. If the cross-attention window has enough reach/overlap between
adjacent blocks, decode could *pipeline* -- start block `b+1`'s decode early, once enough of
block `b`'s dependency is settled, rather than either waiting for full sync completion or
requiring full independence. Analogous to CPU instruction pipelining: not truly parallel, but a
real speedup over strict sequential, without needing full async's masking machinery. Revisit once
the sync/async quality gap (stage 3) is measured.

## Staged plan

1. **DONE**: seed-token-interleaved self-attention + cross-attention-to-own-code decode, sync
   self-track window default, `scheduled_sampling_p`. Smoke-tested (`Ks=(1,)` sanity, `Ks=(2,1)`
   sync/async/scheduled-sampling all run cleanly on CPU; a 300-step overfit check on `Ks=(2,1)`
   shows real learning, train bpb 8.0->5.0, byte_acc 0%->26%). Originally used a shift-by-1 NTP
   target with a code-visibility lag that turned out to never let a block's own code inform its own
   reconstruction (a real bug, see `docs/status.md`'s 2026-08-20 session log) -- fixed to the
   UNSHIFTED own-block target described in "Decoder architecture" above
   (`own_block_cross_attn_decode`/`own_block_decode_loss`); `ks21_v*` full-scale numbers need
   re-running against the fix (`ks1_*` unaffected, no non-top level). Not yet run as a real
   experiment (overfit10k-scale or full-scale).
2. **Run the actual `overfit10k` validation** (`configs/v1_stack_simplex/ks21_v256_pq1.py`,
   `ks21_v64_pq4.py` -- written earlier, need rerunning since their first partial run used the
   scrapped mechanism) against `CLAUDE.md`'s standing fast-iteration bar before trusting anything
   at scale.
3. **Generation-loop rewrite + feasibility check** (see "Generation feasibility" above): implement
   path (b) first (upper-level-predicts-code), check it beats the unconditional baseline; build
   path (a) (draft+encode+refine) if (b) proves too weak. `check_gen_consistency` needs a matching
   chunk-granularity check once any of this exists.
4. **Async ablation**: self-track window masked to `K+1` (already implemented, `window=K+1`, just
   not yet run as a real comparison) -- measure against the stage-1 sync baseline on the standard
   testbed. Only proceed further if the quality gap is small enough to accept for the parallelism
   win.
5. **Multi-level generalization** (`n_levels>=3`): revisit the "only immediate own-level code is
   cross-attended" limitation and the cross-attention-to-upper-code KIV note above once this is
   actually attempted.

## Open risks

- **Loss changes for every non-top level**: reconstruction loss, not NTP cross-entropy --
  touches `StackDecoderV1`'s core loss computation, not additive on top of what's there.
- **Codebook capacity becomes a hard reconstruction constraint**, not a soft compression-ratio
  knob: `K` bytes needs ~`8*K` bits of real information to reconstruct exactly. `Ks=(2,1)` +
  FSQ 16x8 (~48-bit nominal capacity) is comfortably above the 16-bit target for `K=2`; smaller
  codebooks (e.g. `vocab=8` simplex, 3 bits) would need PQ or a much larger `K` to make sense
  under this framing -- revisit the PQ chunk-count table (see `docs/status.md`) once real
  reconstruction-fidelity numbers exist.
