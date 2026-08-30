# summformer_jax

`summformer.py` (top level) is the canonical, self-contained model file -- `Embedder`, `Encoder`,
`Decoder`, `SummFormer`, `ARHead`, `ClassifierHead` -- shared by `lm/`, `image_gen/`, and
`image_classification/`. No version suffix in the filename; git history is the version record.
Older per-lineage frozen copies and design iterations are archived under `../summformer_old/` for
reference only (`model_summformer_v1.py` is the only old implementation confirmed structurally
correct -- it genuinely chains stage-to-stage stride compounding; the `_v2`-derived per-lineage
copies active through 2026-08-29 had a real bug where fuse stages re-pooled from the full sequence
independently every time instead of chaining).

## Core design idea

Efficient long-sequence modeling via a hierarchy of causal transformers that each summarize (pool,
then self-attend over) the sequence at a progressively coarser timescale -- one stage's summary
becomes the next stage's input, so stride compounds multiplicatively across the chain. Each stage's
summary is cross-attended back into the main decoder sequence. Today's model (self-referential,
early layers shared between encoder and decoder) is the special case of a general encoder-decoder
topology, not a separate code path -- see `SummFormer`'s docstring.

## Choosing attention windows -- read before writing a config

**Don't rely on `window=-1` (auto-derive) for either self-attention or cross-attention in a real
config.** Set both explicitly, and verify with `scripts/check_connectivity.py` first. Summary of
what was found (2026-08-30, full derivation in chat/`docs/status_tpu.md`):

- **Self-attention is the load-bearing parameter for receptive field, not cross-attention.**
  A chain stage's own point-sampling pooling (`cur_h[:, stride-1::stride, :]`) discards everything
  except the sampled position -- self-attention (within Embedder, each chain stage's own
  transformer, and the Decoder) is what lets that sampled position actually absorb history before
  being discarded. Cross-attention window can be as small as **1** without hurting reach, PROVIDED
  self-attention is adequately sized -- a tiny cross-attn window only delivers signal to positions
  that land exactly on a stage's code grid, and the following decoder layers' self-attention
  relays it from there to neighboring positions. If self-attention is undersized, no cross-attn
  window fixes it (tested up to literal dense/unbounded).
- **`window=stride` (self-attention) is provably too small** -- `causal_mask`'s strict `<`
  comparison means a gap of exactly `stride` is never bridged. The real minimum is `window >
  stride`, and even that (`window=stride+1`) is only the exact boundary of connectivity, not a
  safe operating point.
- **The exact minimum self-attn window is NOT a simple closed-form function of stride/cum_stride.**
  For an 8-stage/stride-2/`L=256` chain it was found (empirically, then confirmed via a
  weight-free static connectivity check) to be **window=3** at the absolute margin, but that
  margin is **weight-magnitude-fragile**: confirmed the SAME architecture flips between
  connected/disconnected across random seeds and init scales purely due to softmax weighting a
  real-but-tiny path differently. A comfortably larger window (~7-10+ for that config) gave robust,
  seed-independent connectivity in every test.
- **`CrossAttnSpec`'s `window=-1` auto-derives to that stage's `cum_stride`** -- this is a
  heuristic upper bound (guarantees the nearest code token is always in range), not a proven
  necessary minimum. It's often far more generous (and expensive) than needed.

### Two tools, two different questions

- `scripts/check_connectivity.py` -- **does a path exist at all**, as a pure deterministic
  boolean fact about window/stride arithmetic, no random weights, no floating-point precision
  pitfalls. Use this FIRST for any new `(L, stride, n_stages, window)` combination -- it's exact
  and fast (no model construction, no forward pass).
- `scripts/check_chain_receptive_field.py` -- **is a real (weighted) path strong enough to
  matter**, via perturb-and-diff on an actual model, swept across seeds (float64 required --
  float32 buries real-but-small signals in numerical noise and gives false negatives). Use this
  as a follow-up once connectivity is already confirmed by the first tool, to sanity-check that a
  chosen window isn't sitting right at a fragile margin.

Do not conclude "no receptive field" or "safe minimum" from perturb-testing alone at marginal
windows -- confirm with `check_connectivity.py` first, since a genuinely connected-but-weak path
can look identical to a genuinely disconnected one under naive float32 perturbation testing.

## Architecture validity -- what's been checked and how

Three independent kinds of verification, each answering a different question, all confirmed as of
2026-08-30 for the shared-embedder (self-referential, Case A) topology:

1. **Static/mathematical causality proof** (code-level, no execution) -- proved by induction that
   `code_pos_abs[k] = (k+1)*cum_stride - 1` is an EXACT (not approximate) upper bound on what raw
   positions chain-stage code token `k` can depend on, since every self-attention surface enforces
   `key_pos <= query_pos` unconditionally (independent of window -- `causal_mask`'s `<=` check and
   `chunked_windowed_attention`'s block-diagonal construction both enforce this as a hard
   constraint, not something window size can relax), and cross-attention independently enforces
   `code_pos_abs[k] <= query_pos` as its own hard constraint (both the dense `causal_mask` path and
   the windowed `windowed_cross_attention` gather path AND this in regardless of window). Combining
   both: decoder position `q`'s output depends only on raw positions `<= q`, full stop -- no
   execution needed to establish this, it follows from reading the mask conditions. Also checked
   the incremental (KV-cache) generation path separately (a different code path from the dense
   forward) -- found and fixed one real gap there: the dense (`force_dense`) cross-attn path's
   sentinel-padded (`-10**9`) unwritten future slots aren't excluded by the causal `<=` check alone
   (a very-negative sentinel is trivially `<= query_pos`), so `DecoderLayer.forward_incremental_static`
   explicitly ANDs in `cross_pos_abs >= 0`; the windowed path doesn't need this since its separate
   `< window` check excludes sentinels naturally.
2. **Deterministic connectivity check** (`scripts/check_connectivity.py`) -- does a path exist at
   all, as pure boolean reachability over the same window/stride predicates, no weights, no
   floating-point precision issues. Confirmed to exactly match every known empirical result once
   configured identically (see the script's own sanity checks). This is the tool to run FIRST for
   any new config.
3. **Empirical (weighted, float64) confirmation** (`scripts/check_causal_boundary.py`) -- perturb-
   and-diff on a real model (depth=8, stride=2, `window=32` -- safely above the connectivity
   minimum so this test is purely about causality, not receptive field), `L=512`. Confirmed: every
   "does the future leak backward" probe is an EXACT zero (not just small), every "does the past
   correctly reach forward" probe is genuinely nonzero, and the causal boundary between chain
   blocks is sharp -- perturbing the first token of block 1 (position 256) affects a query inside
   block 1 but has EXACTLY zero effect on a query before block 1 starts. Full output captured in
   that script's own docstring.

Net conclusion: the architecture is causally sound (proven, not just tested) and does achieve real
multiplicative receptive-field compounding through the chain once self-attention windows are
adequately sized (see the window-choice section above) -- the several "cross-attn is broken" /
"causality is broken" hypotheses raised while investigating all turned out to be either measurement
artifacts (float32 noise, single-seed weight coincidences) or a misidentified culprit (self-attn
undersizing, not cross-attn or causality), each run to ground with either a deterministic proof or
a float64 empirical check before being accepted or ruled out.
