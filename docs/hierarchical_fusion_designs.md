# Beyond adjacent-only fusion: DenseNet/MoE/recursive designs for N>2 levels

Cross-referenced from [docs/status.md](status.md). Session brainstorm,
grounded in `qcute_refine_v4.py`'s existing `_encode`/`LevelLM._fuse`
machinery — not yet implemented. Prompted by: how should fusion
generalize to 3+ levels (e.g. `Ks=(2,2,2)`), and can it be made more
"pervasive" (DenseNet-style: every level sees every level above it) or
more selective (MoE-style: a router decides which level(s) to consult)?

## Where things are today: adjacent-only chain

`_encode`'s PASS 2 loop (`for i in range(n_active - 1): fuse_kv=h_list[i+1]`)
already generalizes to N levels, but only as a **chain**: level `i` fuses
with level `i+1` only, and always with `i+1`'s PASS-1 (unfused) hidden
state — never level `i+2` or beyond. For a 3-level `Ks=(2,2,2)` tower
(`seq_lens=[1024,512,256]`), level 0 gets level 1's *raw* state; level
2's information only reaches level 0 by accident, filtered through level
1's own unconditioned representation. Level 0 never directly "sees"
level 2 at all.

## Axis 1: DenseNet-style — pervasive, all-above fusion

Instead of `fuse_kv = h_list[i+1]`, let level `i` cross-attend to the
**union** of every level above it: `h_list[i+1], h_list[i+2], ...,
h_list[N-1]`, concatenated along the KV sequence dimension before one
`fuse_cross` call. Cheap to build — reuses everything already in the
file:

- Each source level's rows get tagged with their own resolved raw-time
  position for `cross_attn_rope`, exactly like the null slot already is
  — extend `k_pos` to include every level's own `(b+1)*K_effective - 1`
  tags, where `K_effective` for level `j`'s own blocks (as seen from
  level `i`) is the PRODUCT `K_i · K_{i+1} · ... · K_{j-1}` (a block at
  level 2 covers `K_0·K_1` raw bytes, not just `K_1`).
- The jagged mask generalizes the same way per source level —
  `jagged_causal_mask_and_positions` already takes `K` as a parameter;
  call it once per source level with its own effective `K`, concatenate
  the resulting `disallow` masks along the KV axis.
- One `null_kv` per source level (or one shared) — matches the existing
  pattern.

Literally DenseNet's "concatenate all preceding feature maps" translated
to cross-attention KV concatenation — no new module type, just a wider
KV tensor built from multiple sources instead of one. Cost: KV length
grows with `sum(seq_lens[i+1:])`, and — per `qcute_refine_v4_k32_narrow`'s
own finding (see docs/status.md) — concatenation/bookkeeping overhead,
not attention FLOPs themselves, is what tends to dominate memory. Worth
remembering before assuming "attend to everything" is free just because
each source is individually cheap.

## Axis 2: MoE-style — gated, level 0 decides

Genuinely different from Axis 1, not a variant of it: instead of
*always* fusing with a fixed set of levels, a small router computes
WHICH level(s) to consult, conditioned on level 0's own current
representation:

```
router_logits = Linear(D_0, N-1)(x_level0)     # one logit per candidate level above
```

Two modes, and the distinction matters for what you actually get:

- **Weighted (soft)**: `weights = softmax(router_logits)`, then
  `fused = Σ_j weights[j] · fuse_cross(x, h_list[j])` — run fusion
  against EVERY candidate level, combine with learned weights. Fully
  differentiable, no tricks needed, but PAYS THE FULL COMPUTE COST OF
  AXIS 1 ANYWAY — softness buys a smoother/differentiable combination,
  not sparsity. Really "Axis 1 with a learned mixing weight instead of
  concatenation," not a distinct compute regime.
- **Discrete (hard)**: top-1 (or top-k) selection — attend ONLY the
  chosen level(s), skipping the rest entirely. This is where MoE's
  actual selling point (sparsity, real compute savings) shows up. Needs
  a straight-through estimator to stay trainable through the discrete
  choice — the exact same trick this file already uses for `CodeEmbed`'s
  `pq_table` mode (`hard + (soft_proxy - soft_proxy).detach()`-style), a
  direct precedent already in this codebase. Real new cost:
  **load-balancing** — without an auxiliary loss encouraging the router
  to spread selections across levels, MoE routers reliably collapse to
  always picking one expert (well-documented failure mode, e.g. Switch
  Transformer's own load-balancing loss) — new machinery this project
  hasn't needed before.

## Axis 3: recursive/cascading refinement — the deep-decode variant

Changes fusion's SEMANTICS, not just its breadth. Today, `fuse_kv` is
always level `(i+1)`'s PASS-1 (unfused) state — deliberately, to avoid a
level's own fusion output feeding back into what feeds it (the "no
infinite regress" note in `_encode`'s own docstring). The recursive
variant inverts this: compute fusion TOP-DOWN — level `N-2` first
(fusing with `N-1`'s PASS-1, nothing above it to cascade from), then
level `N-3` fuses with level `N-2`'s just-computed PASS-2/FUSED state
(not its raw one), down to level 0 — so level 0's fusion transitively
benefits from everything above it, refined through every intermediate
level, not a flat concatenation of raw states.

Requires:
- Reordering `_encode`'s PASS 2 loop to run in reverse (`N-2` down to
  `0`) — each step now depends on the PREVIOUS step's output, no longer
  parallelizable across levels the way the current flat loop is.
- A decision on detach semantics: keep detaching each hop (level `i`'s
  fusion doesn't reshape level `i+1`'s weights, preserving the existing
  "don't reshape the level above" principle) — almost certainly the
  right default, consistent with everything validated this session,
  at the cost of the refinement not carrying gradient back up the chain.

"3 to 2, or 3 to 1" is exactly this — cascading refinement (3→2→1→0,
transitive) versus direct skip (3→1 or 3→0, bypassing intermediate
levels' own refinement, closer to Axis 1's concatenation but for a
single specific pair rather than "everything").

## Axis 4: Level-depth routing — coarser and global, not a segmentation mechanism

Prompted directly by contrasting this file's own design space against
H-Net (Hwang/Wang/Dao et al., "Dynamic Chunking for End-to-End
Hierarchical Sequence Modeling") — see [docs/archive2/bpe_like_boundaries.md]
(archive2/bpe_like_boundaries.md) for the fuller writeup of that comparison.
H-Net's routing decision is **local and pairwise**: a per-position
similarity score between adjacent (smoothed) representations decides
WHERE to cut the byte stream into chunks — genuinely content-adaptive
segmentation, already validated in the literature. Re-deriving that same
mechanism inside `qcute_refine` would be redundant, not novel — the
`Ks`-grid segmentation question is effectively a solved problem
elsewhere.

What `qcute_refine`'s fixed `Ks` grid does NOT yet address, and H-Net's
per-position boundary score doesn't either, is a **coarser, more global**
question: for a given span, HOW MUCH hierarchical depth is worth
engaging at all — level 1 only, or level 1 and 2, or the full tower up to
level `N-1`? Today every level always runs, for every block, regardless
of local complexity — a genuinely predictable run of text pays for the
same depth as a hard-to-predict one. This is a **compute-allocation**
question, not a segmentation question — closer in spirit to Mixture-of-
Depths (Raposo et al. 2024, per-token layer-skip routing in a standard
transformer) but applied along `qcute_refine`'s LEVEL axis instead of a
transformer's LAYER axis.

Two genuinely separate decisions, worth keeping distinct rather than
conflating into one router (matches the two clauses of the prompting
idea):

1. **Compute-allocation routing**: which level(s) actually get computed
   (PASS 1, and PASS 2 fusion into whichever is below them) for a given
   region. Crucially, this is a **producer-side** decision — unlike
   Axis 2's routing (which assumes every level is always computed, and
   only routes level 0's CONSUMPTION of them), a real compute-allocation
   router can skip computing level 2/3/etc. entirely for spans judged not
   to need them, i.e. actual FLOPs savings, not just attention-source
   selection.
2. **Decode-level readout selection**: separately, at prediction/decode
   time, which already-computed level's representation actually feeds
   the byte prediction. Two defaults, matching "given default
   unconstrained implicit in model or constrain inputs":
   - **Unconstrained (implicit)**: no new mechanism at all — whatever
     levels got computed by (1) simply flow through the existing fusion
     cross-attention, and the network learns how much to weight each via
     ordinary gradient descent. This is the cheap default; it only does
     anything interesting in combination with (1) actually skipping
     computation somewhere.
   - **Constrained**: an explicit, externally- or heuristically-derived
     signal forces which level's output is used for a given span — e.g.
     reusing `bpe_like_boundaries.md`'s per-position entropy `H_t` as a
     depth signal (low block-entropy → shallow is enough, force level-0-
     only readout; high block-entropy → force deeper readout) rather than
     letting the model decide. Useful both as a compute-saving heuristic
     and as an inference-time controllability knob (forcing a depth for
     ablation/debugging), independent of whether it's also used to gate
     compute in (1).

**Granularity, per "coarser but global"**: unlike H-Net's per-position
score or Axis 2's per-token/per-block router, this is intended to operate
at a coarser cadence — per coarse block (level 1's own block grid, not
level 0's raw bytes) or even per longer span — trading routing precision
for a much cheaper, simpler routing signal and avoiding the load-
balancing machinery that per-position hard routing requires.

**Shape/batching implications** (same constraint this whole file has
respected throughout): naively skipping a level for SOME blocks but not
others breaks the fixed `n_blocks`-per-level tensor shape. Two ways to
reconcile, differing in whether compute is actually saved:
- **Masked-but-computed** (no real FLOPs savings, but static-shape,
  `torch.compile`-friendly, cheapest to build): always run every level
  for every block, but zero out the loss contribution and/or the fusion
  cross-attention read for blocks the router marks "skip." Good first
  step to validate whether the routing SIGNAL is any good at all before
  paying for real sparsity.
- **Gather/scatter with a capacity factor** (real savings, genuinely
  MoE-shaped): route only the top-`c%` of blocks (by router score) into
  each deeper level's compute, matching Axis 2's own discrete-mode
  caveats — needs the same load-balancing auxiliary loss to avoid
  collapse (e.g. always routing everything to level 1 only), a new
  failure mode this project hasn't had to solve yet.

**Recommendation**: cheapest possible test first — skip the LEARNED
router entirely and reuse `bpe_like_boundaries.md`'s already-derived
per-position entropy as a fixed, non-differentiable HEURISTIC depth
signal (constrained readout selection, masked-but-computed variant) —
zero new trainable parameters, directly answers "does depth-routing even
correlate with anything useful" before investing in a learned
compute-allocation router with its own load-balancing concerns. Only
graduate to a learned router (and eventually real gather/scatter compute
savings) once the heuristic version shows the signal has legs — same
"prefer simple things first" ordering this file already applies to
Axes 1-3.

### Sketch: masked-but-computed variant, grafted onto `_encode`

Grounded directly in `RefineLM._encode`'s existing PASS 1/PASS 2 loop
(`qcute_refine_v4.py`, current code shown for the parts that change):

```python
# PASS 1 (unchanged) — every level still runs for every block; this variant
# saves nothing on the producer side yet, it only tests whether the ROUTING
# SIGNAL correlates with anything, per the recommendation above.
for i in range(n_active):
    c_i, ntp_loss, ntp_acc, h_i = self.encoders[i](seq_repr, compute_ntp=want_ntp)
    ntp_losses_pass1.append(ntp_loss); ntp_accs_pass1.append(ntp_acc)
    h_list.append(h_i); x_list.append(seq_repr)
    seq_repr = c_i

# --- NEW: depth signal, one scalar per level-0 block, no new params ---
# H_blk: [B, n_blocks_0] block-averaged predictive entropy of level 0's own
# NTP head — reuse ntp_head's logits already computed above (detached: a
# routing SIGNAL, not a gradient path, matching bpe_like_boundaries.md's own
# detach convention).
H_blk = entropy_per_block(self.encoders[0].last_logits.detach(), K=cfg.Ks[0])
depth_mask = [None] * n_active            # depth_mask[i][b] = True -> block b uses level i's readout
depth_mask[0] = torch.ones_like(H_blk, dtype=torch.bool)   # level 0 always usable (floor)
for i in range(1, n_active):
    # coarser cadence: pool H_blk up to level i's OWN block grid (product of Ks below it)
    H_i = pool_to_level(H_blk, cfg.Ks[:i])
    depth_mask[i] = H_i > cfg.depth_threshold[i - 1]        # heuristic: only "hard" spans reach level i

# PASS 2 (fusion) — masking happens at READOUT, not at compute (still
# masked-but-computed: self.encoders[i] itself still runs on every block)
ntp_losses_pass2 = [None] * n_active
ntp_accs_pass2 = [None] * n_active
if cfg.fuse_encoder_levels:
    for i in range(n_active - 1):
        c_i2, ntp_loss2, ntp_acc2, h_i2 = self.encoders[i](
            x_list[i], compute_ntp=compute_ntp, fuse_kv=h_list[i + 1].detach()
        )
        # constrained decode-level selection: where depth_mask[i+1] is False for
        # a block, level i's OWN fused-but-should-have-been-shallow prediction is
        # replaced by its PASS-1 (unfused) one — i.e. that span is DECODED as if
        # level i+1 didn't exist, even though it was computed. unconstrained/
        # implicit mode is simply this whole block deleted: h_list[i] = h_i2
        # unconditionally, and the network decides the weighting on its own via
        # ordinary cross-attention gradients (Axis 2 soft mode's own default).
        h_list[i] = torch.where(depth_mask[i + 1][..., None], h_i2, h_list[i])
        ntp_losses_pass2[i] = ntp_loss2   # loss bookkeeping unchanged in this cheap variant;
        ntp_accs_pass2[i] = ntp_acc2      # a stricter version would also mask the LOSS per-block,
                                           # not just the readout h used downstream/at generation
```

`entropy_per_block`/`pool_to_level` are the only new functions — thin
reductions over an already-computed tensor, no new parameters, no change
to `torch.compile`-relevant shapes (`depth_mask` is a boolean tensor of
the SAME shape the block grid already has). The real compute-saving
(gather/scatter) variant replaces the `torch.where` above with actually
skipping `self.encoders[i+1]`'s forward call for masked-out blocks — a
strictly harder, second step, deliberately deferred per the
recommendation.

### Sketch: ratio/budget regularization, targeting a compute/bpb tradeoff

The sketch above uses a fixed hand-set `depth_threshold` — it has no way
to say "route ~20% of blocks to level 2" or "spend more compute until
val_bpb hits X," and a fixed threshold on raw entropy has no natural
units connecting it to either a compute budget or a quality target. H-Net
solves the analogous problem (targeting a specific average chunk length,
i.e. compression ratio `N`) with a **ratio loss**: an auxiliary term that
pushes the mean of a soft, differentiable routing probability toward a
target rate, cross-coupled against the actual hard selection so the
model can't satisfy the loss by decorrelating the soft score from what
actually gets selected (their exact coefficients aren't reproduced here
from memory — what follows is a simplified, self-consistent version with
the same two properties: target-seeking mean, and a soft/hard coupling
that resists collapse).

**1. Make the router soft and differentiable** (replaces the hard
`H_i > depth_threshold[i-1]` comparison in the earlier sketch):

```python
# small linear probe on level i's own block summary — genuinely learned,
# not just a fixed entropy threshold (though H_i can still be a FEATURE
# fed into it, e.g. concatenated into the block summary below)
router_logit_i = self.depth_router[i](block_summary_i)         # [B, n_blocks_i], new tiny nn.Linear
p_i = torch.sigmoid(router_logit_i)                             # soft engagement prob, differentiable
depth_mask[i + 1] = (p_i > 0.5)                                 # hard decision for the forward masking,
                                                                  # via straight-through: p_i + (mask - p_i).detach()
```

**2. Ratio loss, targeting a rate `r_i` (analogous to H-Net's `1/N`)**:

```python
F_i = p_i.mean()                    # soft: average predicted engagement prob
G_i = depth_mask[i + 1].float().mean()   # hard: actual fraction routed to level i+1

# pushes mean engagement toward r_i; the F_i*G_i cross term means the loss
# is only truly minimized when BOTH the soft score's average AND the actual
# hard selection rate sit at r_i together — collapsing one while drifting
# the other (e.g. keeping F_i at target while G_i saturates at 0 or 1
# via a degenerate threshold) still costs loss, unlike a plain (F_i - r_i)^2
# term which only constrains the soft side.
ratio_loss_i = r_i * (1 - G_i) * F_i + (1 - r_i) * G_i * (1 - F_i)
```

Total loss gains `cfg.depth_ratio_weight * sum(ratio_loss_i for i in ...)`
— same additive-term pattern as `Config.fusion_ntp_weight` already
established in `_encode`'s own loss.

**3. Connecting `r_i` to a target BPB, not just a fixed compute budget**:
`r_i` itself doesn't have to be a static hyperparameter — treat it as the
dual variable in a rate-distortion control loop, adjusted from measured
val_bpb rather than fixed in the config:

```python
# outside the training step, e.g. once per eval_every — a dual-ascent /
# PI-controller update, same family as adaptive-KL-coefficient schedules
# used elsewhere (PPO's target-KL, VAE beta-schedulers)
bpb_error = target_bpb - measured_val_bpb        # positive: not good enough yet, spend more
r_i = clip(r_i + controller_lr * bpb_error, 0.0, 1.0)
```

If `measured_val_bpb` is worse than `target_bpb`, `r_i` rises (route more
blocks to deeper levels, buy quality with compute); once bpb meets the
target, `r_i` drifts back down, continuously searching for the CHEAPEST
depth budget that still hits the target — this is what makes it a
genuine "target a compression bpb" knob rather than a fixed compute cap.
Cheapest first step, matching this file's own recommendation ordering:
fix `r_i` as a plain hyperparameter and confirm the ratio loss actually
converges `G_i` to it at all (a static-target sanity check) BEFORE adding
the outer bpb-driven controller loop on top.

## Tiling long documents: exactness and cross-tile continuity (vs. H-Net's causal carry)

A different problem than Axes 1-4 above (those are about breadth/depth of
fusion WITHIN one already-fixed-length sequence) — this is about
processing a document LONGER than `context_len`, split into successive
tiles, given `qcute_refine`'s block grid is fixed-stride and
position-modular rather than causal/streaming the way H-Net is.

**Why H-Net doesn't have this problem the way `qcute_refine` does**:
H-Net's boundary decision is local and pairwise (similarity between
adjacent smoothed representations) — chunks are already variable-length,
so there's no notion of a "trailing partial chunk" at all, and tiling for
long documents just means carrying forward a small O(1) piece of state
(the last position's smoothed representation) across the tile boundary
so the FIRST boundary decision of the next tile is computed exactly as if
the stream had never been cut. `qcute_refine`'s grid is the opposite
tradeoff: block boundaries are POSITION-MODULAR and fixed ahead of time
(`Ks`' nested products) — that rigidity is exactly what makes the
fixed-shape/batched/`torch.compile`-friendly story work at all, but it
also means the grid has a hard, structural notion of "complete" vs.
"incomplete" block that H-Net's own design never has to reckon with.

### The remainder math: mixed-radix decomposition, not just one leftover count

Define `Ktotal_i = Ks[0] · Ks[1] · … · Ks[i]` — the number of raw bytes
spanned by one level-`i` block (`Ktotal_0 = Ks[0]`, and so on up the
tower). Because each `Ktotal_i` is a strict divisor of every coarser
`Ktotal_j` (`j > i`) by construction (it's a nested product), there's a
useful lemma:

> **Padding the raw sequence to the next multiple of the COARSEST active
> level's `Ktotal_{n_active-1}` automatically makes every FINER level's
> blocks complete too** — no separate padding decision is needed per
> level.

Worked example matching the prompt directly — `Ks=(2,4)`, doc length
`L=15`: `Ktotal_0=2`, `Ktotal_1=2·4=8`. `15 = 1·8 + 7` — one complete
level-1 block (8 bytes, itself containing 4 complete level-0 blocks), a
7-byte remainder. That remainder is ITSELF not level-0-clean: `7 = 3·2 +
1` — 3 complete level-0 blocks (6 bytes) plus 1 raw byte that doesn't
even complete its own level-0 pair. This is exactly a mixed-radix / place-
value decomposition with "digits" `Ks[0], Ks[1], …` — but per the lemma,
none of this multi-level bookkeeping needs to be done explicitly: padding
`L=15` up to `⌈15/8⌉·8 = 16` (one pad byte) leaves every level clean
simultaneously (16/8=2 level-1 blocks, 16/2=8 level-0 blocks, both exact).

### Sketch A — pad-to-coarsest-stride + mask (exact, fully parallel, no new sequential dependency)

```python
Ktotal = math.prod(cfg.Ks[:n_active])
pad_len = (-L) % Ktotal                      # 0 if already exact
byte_ids_padded = F.pad(byte_ids, (0, pad_len), value=PAD_BYTE)
# PAD_BYTE: a sentinel outside the normal byte alphabet — same pattern
# v4.2 already uses ("bits valued 0-255 reserved for raw bytes" convention
# for dq>8), just reserve one more out-of-range id here.

pad_mask = torch.arange(L + pad_len) >= L    # [L+pad_len], True on pad positions
```

Two things must respect `pad_mask`, both reusing machinery this file's
Axis 1-4 sketches already lean on:

- **Loss/acc**: exclude pad positions from `byte_loss`/`byte_acc` — same
  crop-before-metric pattern `v4.2`'s dq>8 fix already established
  (compute over the padded shape, mask the metric reduction only).
- **Attention**: PAD keys must never be attended to by REAL positions
  (they'd otherwise pollute the pooled/last-position readout of the
  final, still-incomplete-in-content-terms block) — extend the existing
  `disallow`/null-KV masking machinery (`jagged_causal_mask_and_positions`,
  already parameterized by `K` per level) with one more excluded-key
  class, exactly the same shape of change as the null-KV slot already
  gets threaded through `fuse_position="concat"`.

Fully parallel: one padded tensor, same batched forward call, zero new
cross-step dependency — the only cost is `<Ktotal` wasted positions
(bounded, and negligible relative to any reasonably sized `context_len`).

### Sketch B — cross-tile continuity via a carried KV tail (parallel within a tile, sequential across tiles — same tradeoff windowed attention already accepts)

Sketch A handles the LAST (possibly short) tile of a document exactly.
It does NOT address the separate problem of a windowed self-attention
(`attn_window[i]`) or fusion cross-attention losing context right at
EVERY tile boundary, not just the final one — the first `attn_window[i]-1`
positions of tile `n+1` can't see tile `n`'s tail unless something
carries it forward. This is the direct analog of "carry the segmentation
score to the next non-overlapping chunk" — except `qcute_refine` doesn't
need to carry an ambiguous soft score at all, because the grid tells you
EXACTLY which positions matter: carry the trailing `attn_window[i]-1`
already-resolved KV entries (raw byte positions for level 0; block-scaled
positions for higher levels) from tile `n`'s forward pass into tile
`n+1`'s, prepended as fixed (stop-gradient, matching this file's
established "don't reshape the level above" convention) context —
exactly `generate_kv_cache`'s existing incremental (byte-at-a-time) cache
machinery, generalized from byte-at-a-time to tile-at-a-time increments.

```python
# tile n -> tile n+1, per level i
carried_kv_i = kv_cache_i[:, -(attn_window[i] - 1):]   # tail from tile n, stop-gradient
h_new = level_i.forward_tile(tile_n1_bytes, prefix_kv=carried_kv_i)
```

**Design rule this requires, stated once and satisfied by construction**:
every NON-FINAL tile boundary must land on a multiple of `Ktotal`
(`tile_size % Ktotal == 0`) — this guarantees no block, at any level,
ever straddles a tile edge, so "carry the KV tail" is the ONLY thing tile
boundaries need to fix; block-grid alignment itself is never in question.
Only the document-final tile (possibly short) needs Sketch A's padding on
top.

**Parallelism**: identical tradeoff to `_forward_chunked`'s own windowed-
attention chunking WITHIN a single sequence, just promoted one level up
to the document/tile axis — each tile's own forward pass is still one
fully parallel/batched call over its whole length; only tile `n+1` as a
whole depends on tile `n`'s carried tail, i.e. tiles are sequential
relative to EACH OTHER but not internally serialized. No new parallelism
regime introduced, the same one `attn_window` chunking already relies on,
applied at a coarser grain.

### What this does NOT solve — mid-word/mid-unit cutoffs are a separate, orthogonal problem

Sketch A/B make grid boundaries EXACT (no shape crashes on non-multiple
lengths) and CONTINUOUS (no lost context at tile edges) — they do not
make grid boundaries CONTENT-AWARE. A fixed-stride block can still land
mid-word even with perfect tiling; that's the actual content-adaptivity
question, already explored separately in
[docs/archive2/bpe_like_boundaries.md](archive2/bpe_like_boundaries.md) (soft/entropy-
weighted pooling within the fixed grid) and this file's own Axis 4
(level-depth routing). Tiling exactness/continuity and content-adaptive
segmentation are independent problems with independent fixes — Sketches
A/B don't move the boundary POSITIONS, they just make the fixed positions
well-defined and seamless across tile edges.

## Simulating this in training: recurrent chunked TBPTT, receptive-field-aware carry length, per-chunk depth routing

Sketch B above (carried KV tail across tiles) describes the INFERENCE-time
mechanics. Training it means processing a long document as a sequence of
`chunk_len`-sized chunks (e.g. 256) IN ORDER, carrying detached per-level
state chunk-to-chunk — standard truncated backprop through time (TBPTT),
with one `qcute_refine`-specific wrinkle: not every level's carried state
should hold the same NUMBER of entries, because levels don't have the
same effective receptive field per entry — combined with Axis 4's own
idea that WHICH levels even get exercised can itself vary chunk-to-chunk
("read first chunk, choose level 1 and 2; later chunk, level 0 only").

### Why carry length must be receptive-field-aware, not entry-count-aware

Level `i`'s each KV entry already summarizes `Ktotal_i = Ks[0]·…·Ks[i]`
raw bytes (§ tiling math above). Carrying the SAME NUMBER of entries from
every level therefore buys wildly different amounts of actual lookback —
carrying 32 entries of level 0 (`Ktotal_0` small) reaches only slightly
into the past, while carrying 32 entries of level 2 (`Ktotal_2` large)
can reach an order of magnitude further back, for the same KV-cache
memory/compute cost. So the carry length should be specified in RAW-BYTE
reach and converted per level, not fixed as a flat entry count:

```python
target_reach_bytes = 2048          # e.g. "I want ~2048 bytes of lookback available"
carry_entries = {
    i: max(attn_window[i] - 1, target_reach_bytes // Ktotal[i])
    for i in range(n_active)
}
# worked example, Ks=(4,4,4): Ktotal = [4, 16, 64]
#   level 0: 2048 // 4  = 512 entries carried  (expensive: 512 KV rows)
#   level 1: 2048 // 16 = 128 entries carried
#   level 2: 2048 // 64 =  32 entries carried  (cheap: 32 KV rows, same reach)
```

This is precisely why routing MORE reach through a COARSER level is
attractive, independent of the depth-routing question below: level 2's
carried cache is ~16x cheaper than level 0's for identical raw-byte
lookback, simply because each of its entries already did the compression
work.

### Depth routing per chunk: reuse the existing `n_active_levels(step)` curriculum, drive it by content instead of just step

The codebase already has a mechanism that varies `n_active` — currently
only as a function of TRAINING STEP (a fixed curriculum schedule, see
`Config.layer_warmup_steps`/`n_active_levels`). The natural generalization
here is to let a chunk-boundary router (Axis 4's compute-allocation
router, but evaluated once per CHUNK rather than once per block) also
condition `n_active` on carried state, making the depth decision content-
driven within a single training run rather than only step-driven across
runs:

```python
def n_active_for_chunk(carried_state, cfg, step):
    curriculum_cap = n_active_levels(step)               # existing step-based ceiling, unchanged
    router_logit = depth_router(carried_state.summary)    # small linear probe on carried KV summary —
    n_wanted = 1 + (router_logit.sigmoid() * (curriculum_cap - 1)).round()
    return min(n_wanted, curriculum_cap)
```

**Causality constraint, worth stating explicitly**: the router must
decide chunk `n`'s depth using only carried state FROM chunk `n-1` and
earlier — never chunk `n`'s own content — otherwise the routing decision
itself becomes non-causal (can't be replicated at real generation time,
where chunk `n`'s bytes don't exist yet when the decision is needed).
This is a genuine difference from the single-pass Axis 4 sketch earlier
in this file (which computed `H_blk` from logits already produced within
the SAME forward pass — fine there because byte-level causal masking
already made every individual prediction causal; here the decision is
about an entire UPCOMING chunk, so it needs its own one-chunk-earlier
causal margin, more like Adaptive Computation Time's own halting
mechanism than a per-position entropy readout).

### Handling a level that gets skipped for some chunks — lazy/deferred compute keeps the causal chain alive without paying full compute

If chunk `n` routes to `n_active=1` (level 0 only) and chunk `n+1` routes
back to `n_active=3`, level 2's carried cache has a GAP — it never saw
chunk `n`'s content, because normally `c_i` (level `i`'s output feeding
level `i+1`) requires level `i`'s full forward pass to exist at all. Two
options, and the cheap one is worth defaulting to:

- **Full recompute of skipped levels retroactively** — expensive, defeats
  the purpose of skipping in the first place; rejected.
- **Lazy placeholder code** (recommended default): when level `i+1` is
  not engaged for a chunk, it still receives a CHEAP, non-parametric
  placeholder input for that span — e.g. a plain mean-pool of level `i`'s
  own codes over the skipped span (no transformer forward, no new
  parameters) — just enough to keep the causal chain from having an
  actual hole, deferring the FULL nonlinear compute to whenever the
  router re-engages that level. This is the same masked-but-computed-vs-
  compute-skip distinction Axis 4 already draws, applied across chunks
  instead of within one: level `i+1`'s cache entry for a skipped chunk is
  "cheap and approximate" rather than "absent."

```python
if level_active[i + 1]:
    c_i1, h_i1 = self.encoders[i + 1](c_i, ...)          # real compute
else:
    c_i1 = mean_pool(c_i, window=Ktotal[i + 1] // Ktotal[i])  # lazy placeholder, no grad-bearing compute
    h_i1 = carried_state[i + 1].last_h                        # cache simply doesn't advance its OWN h this step
carried_state[i + 1] = update_cache(carried_state[i + 1], c_i1)
```

### The training loop, put together

```python
state = init_carried_state(cfg)     # per-level detached KV tail, per-level "last real h"
for chunk in document.tiles(chunk_len):           # chunk_len e.g. 256, chunk_len % Ktotal[-1] == 0 (tiling § rule)
    n_active = n_active_for_chunk(state, cfg, step)
    loss_chunk, state = model.forward_chunk(chunk, prefix_state=state, n_active=n_active)
    loss_chunk.backward()                          # gradient scope: THIS chunk only
    state = detach_state(state)                    # TBPTT: carried state stops gradient here,
    optimizer.step(); optimizer.zero_grad()         # matching _encode's existing "don't reshape the
                                                     # level above" detach convention, now applied across
                                                     # chunks too, not just across levels
```

Gradient scope is the standard TBPTT tradeoff, explicitly worth flagging:
each chunk's backward pass only reaches that chunk's own compute plus
whatever's still attached within it — carried KV state from earlier
chunks is detached, so a chunk never gets gradient signal about HOW its
own carried context was produced, only that it existed. Consistent with
this codebase's own existing `fuse_kv.detach()` convention, just widened
in scope from "the level above" to "the chunk before."

### Worked numeric example, tying it together

`Ks=(4,4,4)`, `chunk_len=256`, `target_reach_bytes=2048`. Chunk 1 (start
of doc, entropy/content forces deep routing): `n_active=3`, all three
levels compute fresh codes, caches seeded. Chunk 2 (highly predictable
run — router picks `n_active=1`): only level 0 runs for real; levels 1
and 2 get lazy mean-pooled placeholders, their REAL caches stay as of
chunk 1. Chunk 3 (entropy spikes again, router picks `n_active=3` again):
level 2 fuses against a carried cache whose most recent REAL entry is
still from chunk 1 (chunk 2 only contributed a cheap placeholder) — a
one-chunk-stale but never-absent context, the direct tradeoff this
lazy-placeholder design accepts in exchange for chunk 2 costing almost
nothing at levels 1/2. Level 0's own cache, by contrast, is never stale —
it runs every chunk regardless of the router (the "floor" level, matching
Axis 4's own `depth_mask[0] = always True` convention).

## Worked example: N=3, `Ks=(2,2,2)`, `context_len=1024`

`seq_lens = [1024, 512, 256]`. At level 0's fusion step:

| variant | KV level 0 attends to | KV positions | effective K tagging |
|---|---|---|---|
| chain (current v4) | level 1 only, PASS-1 | 512 | `K=2` |
| dense (Axis 1) | level 1 + level 2, concatenated, both PASS-1 | 512 + 256 = 768 | `K=2` for level 1's rows, `K=4` for level 2's rows (covers `2×2` raw positions) |
| gated-hard (Axis 2) | whichever ONE the router picks | 512 or 256 | matches whichever source |
| recursive (Axis 3) | level 1's PASS-2 (already fused with level 2) | 512 | `K=2`, but level 1's own rows now carry level-2 information transitively |

## Recommendation

Ranked by implementation risk vs. expected signal, given this session's
own "prefer simple things" lesson:

1. **Dense/concatenated (Axis 1) first** — cheapest to build (pure
   plumbing, no new training dynamics), directly tests whether level 0
   benefits from skip-level access at all, before investing in anything
   harder to train.
2. **Recursive cascading (Axis 3) second** — more interesting
   semantically (transitive refinement vs. flat concatenation), same
   training-stability profile as what's already validated (detach-per-
   hop), but real engineering (reordering the loop, losing parallelism
   across levels).
3. **MoE-hard routing (Axis 2, discrete) last** — genuinely the most
   powerful idea (real sparsity, real compute savings, closest to "the
   decision is made at LevelLM 0") but also the only one importing a new
   failure mode this project hasn't had to solve yet (load balancing).
   Worth doing only after 1 and 2 establish whether skip-level
   information even helps — no point building a router to choose between
   levels if the levels being routed to don't move the number.

Not yet implemented — this file records the design space, not a result.

## MoE adaptive-timescale compute: routing by regularity structure, not just depth

Axis 2 (gated fusion) and Axis 4 (level-depth routing) above both route
by "how much compute does this span deserve" along a single fixed `Ks`
tower. A genuinely different framing, prompted by asking what a
byte-level model could exploit across very different data domains (audio,
video, 2D images, 3D voxels/point clouds, text, sensor/actuator control
streams): different spans of the SAME stream, or entire different
streams, have different **characteristic timescales of regularity** —
the raw-byte distance over which a byte becomes predictable from context
— and that timescale is itself a property worth routing on, not just
"hard vs. easy."

**The idea**: instead of one `Ks` tower with a router deciding how deep
to go, instantiate several parallel towers ("timescale experts") at
different fixed `Ks` profiles — e.g. a fast expert `Ks=(4,4)` (short block
spans, good for content that decorrelates quickly: text at the
morpheme/word scale, transient audio onsets, high-frequency sensor
noise) and a slow expert `Ks=(64,64)` (long block spans, good for content
that's redundant over long stretches: sustained audio tones, static
video background, slowly-drifting sensor baselines) — and a lightweight
router picks WHICH expert's tower actually processes a given span. This
is Axis 2's discrete/hard mode generalized from "which LEVEL of one
tower" to "which TOWER entirely," and Axis 4's compute-allocation router
generalized from a depth knob to a genuinely different `Ks` hyperparameter
per expert. Cheapest test, following this file's own "prefer simple
things first" pattern: reuse Axis 4's exact masked-but-computed sketch,
just with `self.encoders[i]` replaced by "tower A vs. tower B" instead of
"level `i` vs. level `i+1`" — same straight-through/load-balancing
caveats apply unchanged.

**Why timescale (not just difficulty) is the right routing signal**:
Axis 4's entropy-based router asks "is this span hard to predict right
now." A pure difficulty signal conflates two different things that call
for different fixes — a span can be locally UNPREDICTABLE (genuinely high
entropy, needs a deeper/wider receptive field to resolve at all — the
existing Axis 4 case) or locally predictable but only over a LONG
horizon (needs a coarser block stride to reach far back cheaply, not more
depth at the current stride). Routing only on entropy conflates these;
a periodic audio waveform's individual samples can be "easy" (low
per-byte entropy, an entropy router would send it to the shallow expert)
while still needing the SLOW expert's long stride to actually capture the
periodicity's ancestor lag — the two signals point in different
directions for exactly the content this idea is meant to help with. A
second, cheap router feature alongside entropy — local autocorrelation
peak lag, computed the same way pitch detection or period-finding does
(cheap: FFT or autocorrelation over a short trailing window, no learned
parameters) — is a more direct proxy for "what timescale is this content's
regularity actually operating at" than entropy alone.

## Limits and feasibility of exploiting byte-level regularity across domains

The byte-level premise this whole codebase is built on — one causal LM
over raw bytes, domain-agnostic — is a genuine strength (no hand-built
tokenizer, one architecture serves any byte stream) but the amount of
exploitable regularity, and the SHAPE of that regularity, differs sharply
by domain. Worth being explicit about where the premise is strong, where
it's weak, and what (if anything) narrows the gap without abandoning
byte-agnosticism.

**What "regularity" byte-level models can actually exploit, in general**:
statistical redundancy (local predictability — the thing cross-entropy
loss directly optimizes), long-range periodicity/self-similarity
(exploitable only if context length and hierarchy reach far enough — this
is exactly what the `Ks` tower and its receptive field, see the tiling
math above, are FOR), and compositionality (recurring substructures —
words, motifs, repeated waveform cycles — which is what the hierarchical
code levels are meant to discover and hand off, functioning as an
implicitly learned tokenizer). None of this requires knowing the domain;
all of it requires enough context and enough hierarchy depth to actually
reach the regularity's timescale, which is a genuine, measurable limit,
not just an engineering inconvenience — a model with `context_len=1024`
structurally cannot exploit a regularity with characteristic period
>1024 bytes no matter how well trained.

**Domain-by-domain feasibility**:

- **Text**: the domain this codebase already targets and the best fit for
  flat byte-stream causal modeling — linguistic structure (morphemes,
  words, phrases) is ALREADY naturally sequential and local in byte order,
  so a 1D causal hierarchy's locality assumption matches the data's own
  structure almost exactly. [docs/archive2/bpe_like_boundaries.md](archive2/bpe_like_boundaries.md)'s
  entropy-boundary finding (learned code boundaries land close to
  BPE/whitespace boundaries) is direct evidence the byte-level premise
  pays off here specifically because text's regularity IS sequential.

- **Raw audio (PCM)**: high sample rate means extreme short-range
  redundancy (adjacent samples are nearly identical) but the perceptually
  important structure — pitch period, phoneme duration, rhythm — lives at
  timescales of hundreds to thousands of samples, i.e. exactly the "needs
  long effective reach, not more depth at the current stride" case the
  adaptive-timescale idea above targets. Feasible with this architecture
  IF context/hierarchy reach the relevant period; a coarse level's block
  stride, chosen to roughly match a domain's typical pitch period, gives
  the tower a usable coarse "cycle" representation for free, matching the
  same mechanism that lets a coarse code level implicitly discover
  word-like units in text.

- **2D images (raster byte streams)**: the sharpest structural mismatch
  in this whole list. Flattening a 2D image into a raster byte order
  destroys the 2D adjacency a native vision architecture (conv/ViT patch)
  gets for free — two vertically adjacent pixels, which are highly
  correlated, sit `width` bytes apart in the flattened stream, often far
  outside a narrow attention window, and even inside `context_len` the
  causal model has to LEARN "distance `width` back is special" purely
  from data rather than being told it structurally. A causal byte model
  over raster order is not fundamentally incapable of learning this (a
  wide enough receptive field plus enough training data can discover the
  row-stride correlation, analogous to how it discovers word boundaries)
  but it's genuinely sample- and compute-inefficient relative to an
  architecture with the 2D prior built in — one of the honest limits of
  domain-agnostic byte modeling, not a bug in this codebase's design.

- **3D voxels / point clouds**: worse than images along the same axis,
  plus a second, more severe problem — extreme sparsity. Most voxels in a
  typical 3D grid are empty; a dense byte-level flattening spends the same
  per-byte compute on empty space as on structured content, unlike a
  sparse-native representation (octree, point list) that pays roughly
  zero cost for emptiness. Point clouds don't even have a fixed grid to
  flatten in the first place — a point's (x,y,z,attrs) tuple carries no
  intrinsic ORDER relative to its neighbors the way a raster image or a
  time series does, so "next-byte prediction" isn't obviously well-posed
  without first imposing SOME order. This is the domain where
  byte-agnosticism arguably breaks down as a design choice rather than
  just costing efficiency.

- **Sensor/actuator control streams**: structurally the most different
  case from text, in a way that matters beyond compute efficiency. Two
  distinct issues: (1) high frame-to-frame redundancy from physical
  continuity (real actuators/sensors have bounded rate of change —
  bandwidth-limited by physics — so successive samples are usually close,
  a strong and genuinely exploitable statistical regularity), but (2) the
  bytes of a structured numeric encoding (e.g. IEEE 754 float32) are NOT
  semantically uniform — the sign/exponent bytes determine
  order-of-magnitude and direction, the mantissa bytes determine
  fine-grained precision, and a plain per-byte cross-entropy loss weighs
  a mantissa LSB flip (usually harmless) the same as a sign-bit flip
  (can invert an actuator command's direction entirely). This is a real
  mismatch between what the loss optimizes (uniform byte-level
  likelihood) and what actually matters for control (bounded, physically
  meaningful error) — not something more context or a bigger model fixes
  on its own.

**Generalization ideas that stay byte-agnostic** (extend the architecture
without hand-building a domain-specific tokenizer, matching this
codebase's own stated philosophy):

1. **Space-filling-curve serialization for 2D/3D** (Hilbert/Morton/
   Z-order curve) as a PREPROCESSING step, not an architecture change:
   reordering pixels/voxels along a locality-preserving curve before
   flattening to bytes means nearby-in-the-stream bytes are also
   nearby-in-space, which is exactly what the causal/windowed-attention
   locality prior already assumes for 1D data. Doesn't manufacture true
   2D/3D equivariance, but narrows the row-stride problem above at zero
   architectural cost — a genuinely cheap, high-leverage first step for
   any domain with implicit spatial structure.

2. **Significance-aware byte structure, reusing this session's own
   `BitPredictHeadWordPredict`/hierarchical-code machinery**: for
   structured numeric formats (IEEE 754, fixed-point sensor readings),
   map a value's semantically MORE significant bytes (sign/exponent) to
   a COARSER level of the existing `Ks` hierarchy and its LESS significant
   bytes (mantissa) to a finer level — the code hierarchy already exists
   to separate "coarse, load-bearing" from "fine, refining" information;
   this just points that same mechanism at engineering byte-significance
   instead of only temporal timescale. A natural loss-side complement:
   weight byte cross-entropy by significance (e.g. scale the sign/exponent
   byte's loss term up) rather than treating every byte position as
   equally costly to get wrong — directly addresses the control-stream
   mismatch above without redesigning tokenization.

3. **Domain/timescale-conditioned expert selection**, tying back to the
   adaptive-timescale MoE idea above: a cheap router (entropy +
   autocorrelation-lag feature) picking among a small set of pre-configured
   `Ks`/`attn_window` profiles per span or per stream is a way to let ONE
   trained model cover several domains' differing native timescales
   without hand-specifying which domain it's looking at — genuinely
   different from training separate domain-specific models, and
   consistent with this file's existing preference for routing/gating
   mechanisms over hard-coded branches.

4. **Physical priors as auxiliary losses, not architecture changes**, for
   control/sensor streams specifically: a cheap regularizer penalizing
   the model's IMPLIED derivative (next-byte prediction's decoded value,
   differenced against the previous decoded value) for exceeding a known
   physical bandwidth limit doesn't touch tokenization or the `Ks` tower
   at all — same additive-loss-term pattern `Config.fusion_ntp_weight`
   and the ratio-loss sketch above already establish, applied as a
   physically-motivated prior instead of a compute-budget one.

**Where the byte-agnostic premise has a real, not-yet-closeable gap**:
none of the ideas above manufacture a hard geometric prior (2D
translation-equivariance, 3D rotation-equivariance) that a domain-native
architecture (conv-net, equivariant GNN) gets by construction — they only
make it CHEAPER for gradient descent to discover an approximation of that
structure from data. For domains where the geometric/physical prior is
strong and well understood (images, 3D geometry, robot dynamics), a
byte-level causal LM should be expected to remain less sample- and
compute-efficient than a structure-native model, full stop — the honest
value proposition of this codebase's approach is a single architecture
that degrades gracefully and improves incrementally (via hierarchy depth,
adaptive timescale routing, significance-aware structure) across many
domains at once, not one that matches domain-specialized architectures on
any single domain's own home turf.

## H-Net's dynamic chunker vs. wider-receptive-field timescale-MoE: does either overcome the domain limits above?

Correction to this file's earlier framing (Axis 4, above): H-Net doesn't
do single-level flat segmentation — its own paper **stacks the dynamic
chunker recursively**, each stage's chunk-pooled output feeding the next
stage's own boundary detector, so it genuinely builds a multi-level
hierarchy, structurally much closer to `qcute_refine`'s `Ks` tower than
"one pairwise cut" suggests. Re-examining the domain limits above with
that correction in mind:

- **2D image raster mismatch — not overcome.** Every stage's boundary
  decision, at every level, is still a pairwise comparison between
  ADJACENT positions in whatever 1D order it's handed. Stacking raises
  effective receptive field (level-2 chunks summarize more raw bytes than
  level-1 ones) but gives no structural shortcut to "distance = image-
  width is special" — a vertically-adjacent pixel is still buried `width`
  positions away at every level, same as a fixed-stride `Ks` tower. Both
  architectures could in principle learn the row-stride correlation given
  enough capacity/data; neither gets it for free from raster-order bytes.

- **3D point clouds/voxels lacking intrinsic order — not overcome,
  orthogonal.** H-Net's chunker needs SOME 1D sequence to walk
  adjacent-pairwise; it doesn't invent an order any more than a
  fixed-stride grid does. This limit is about the absence of an order to
  segment along at all, not about how segmentation is decided once one
  exists.

- **3D voxel sparsity (given some order/serialization already exists) —
  genuinely overcome, a real structural advantage over the fixed `Ks`
  tower.** A long empty run is maximally locally-similar, so H-Net's own
  similarity-threshold mechanism should naturally collapse it into one
  large chunk and spend fine boundaries only where content actually
  varies — adaptive-length compression is built into the segmentation
  itself. `qcute_refine`'s fixed-stride grid only gets an equivalent
  effect by bolting on Axis 4's depth-routing (a separate router, separate
  machinery, separate load-balancing concerns); H-Net gets it as a
  first-class consequence of its core mechanism, no add-on required.

- **Audio periodicity/timescale — partially overcome, and specifically
  BECAUSE of the stacking.** A single H-Net stage's signal (adjacent-
  sample smoothness) targets local continuity, not cyclical repetition —
  consecutive samples within one waveform cycle can differ a lot even
  though the CYCLE repeats, so a bottom-stage boundary detector isn't
  obviously going to find period boundaries directly. Recursive stacking
  is exactly the mechanism that could get there indirectly: if stage 1's
  chunk summaries capture enough within-cycle shape, stage 2's
  adjacent-CHUNK comparison could then operate at a granularity where
  cycle-to-cycle similarity becomes locally visible. Plausible emergent
  property, not a structural guarantee — contrast with the
  adaptive-timescale-MoE section above's explicit autocorrelation-lag
  router feature, which structurally forces the periodicity signal to be
  present rather than hoping it survives multiple stages of learned
  pooling.

- **Control-stream byte significance (sign/exponent vs. mantissa) — not
  overcome, orthogonal.** H-Net's criterion is generic representation
  dissimilarity; nothing in it encodes semantic byte significance.
  Segmentation only decides GROUPING, never per-byte loss weighting or
  which hierarchy depth a byte's importance deserves — needs the
  significance-aware idea above regardless of which chunker (fixed-stride
  or learned) is used underneath it.

**Mechanism-level contrast** — the two designs adapt along genuinely
different axes, not competing solutions to the same problem:

| | H-Net (stacked) | timescale-MoE (above) |
|---|---|---|
| adapts | WHERE boundaries fall, at every level | WHICH of a small fixed menu of stride-profiles handles a span |
| decision granularity | per-position, local, cheap (adjacent-pair comparison) | per-span, coarser, needs a hand-built feature (entropy + autocorrelation lag) |
| expressivity | arbitrary, per-instance chunk lengths | only as flexible as the pre-set menu of `Ks` profiles |
| cost | one tower; boundary placement does all the work | multiple towers (or a shared-weight variant) plus a router |

**They compose rather than compete**: H-Net-style learned per-level
boundaries could serve as `qcute_refine`'s own segmentation mechanism
(the natural graduation path Axis 4 already gestures at — "eventually
replace fixed `Ks` with something content-adaptive"), with a coarse
router still selecting among several such stacked towers pre-biased
toward different characteristic scales, for streams whose content spans
genuinely disparate orders of magnitude in timescale within one document
(e.g. a mixed audio-plus-control-channel log) — answering "where to cut"
and "how much/which scale to spend" as two independent, composable
decisions rather than one mechanism trying to do both.

## Extending H-Net past greedy pairwise causal comparison: windowed/dilated lag detection

H-Net's boundary score bundles two separate limitations into one
mechanism: it's **single-lag** (only ever measures `sim(x_t, x_{t+1})` —
structurally blind to a period `p>1` signal, whose lag-1 similarity can
be low even in a maximally regular region, since adjacent samples inside
one cycle can differ a lot even though the CYCLE repeats) and it's
**greedy** (commits at the first threshold crossing, never reconsiders
once a few more positions of context arrive — the standard failure mode
of greedy change-point/boundary placement, well precedented in speech
segmentation literature). Session question: can a window/lookahead fix
this directly, rather than relying entirely on recursive stacking (this
file's own earlier analysis: stacking can *plausibly* discover
periodicity emergently, but it's not guaranteed — the signal has to
survive several stages of learned pooling).

**Windowed multi-lag comparison, reusing this session's own dilated-conv
machinery.** Instead of one similarity score at lag 1, compute a short
VECTOR of scores across several lags — `sim(x_t, x_{t+k})` for
`k=1..W` — a short-time local autocorrelation, the same technique
DSP pitch/onset detectors use (YIN, short-time ACF). The concrete cheap
implementation is already in this codebase:
`BitPredictHeadConvDilated._dilated_stack` (`qcute_refine_v4_2.py:1386`)
computes exactly this shape — multi-tap, per-channel, `unfold`+`einsum`,
deliberately never `nn.Conv1d` (measured ~300x slower at this scale, see
§17/18 of `docs/archive2/kv_contribution.md`). Two spacing choices, same tradeoff
already established for that class:
- **Flat window** (`k=1..W`): catches periods up to `W`, cost `O(W)`.
- **Dilated/log-spaced taps** (`k∈{1,2,4,8,...}`, `L=log2(period)` taps
  instead of `period`): reaches periods far larger than a flat window
  affords WITHOUT waiting for several stacked levels to accumulate the
  reach — a single stage directly probes candidate periods at multiple
  scales, and the tap with highest similarity doubles as a period
  ESTIMATE, usable to directly set a chunk's target length rather than
  only reacting to a threshold crossing position-by-position.

**Non-greedy refinement, nearly free given the window is already
computed**: instead of committing at the first crossing, pick the LOCAL
MINIMUM of the dissimilarity curve within a small forward span (best of
the next few candidate cuts) before finalizing — buffered-onset-detector
style, trading a few positions of latency for a better-placed boundary.

**Causal cost — a bounded delay, not true lookahead**: both extensions
need `W` (or `2^L`) positions AHEAD of a candidate cut to score it,
genuinely non-causal within that span. Training (teacher-forced, full
sequence visible): free, one batched windowed computation, same as
`attn_window`'s own parallel training path. Streaming/generation: needs
an actual lookahead buffer — position `t`'s boundary isn't finalized
until `t+W` arrives, a bounded, KNOWN latency, not unbounded — identical
shape of tradeoff `attn_window`'s own windowed attention already accepts
in this codebase, just moved from the attention axis to the
segmentation-decision axis.

**Relationship to stacking — complementary, not a replacement.**
Stacking still does something windowed/dilated single-stage detection
can't: genuine representational abstraction (chunk-of-chunks, discovering
compositional units, semantic compression). What this extension fixes is
narrower: it lets the FIRST stage directly detect periodicity that used
to only be discoverable emergently after several levels of stacking
happened to preserve the right information through pooling. Concretely:
fewer levels may be needed to reach a given periodicity, and levels that
do still stack now start from an already period-aware segmentation
instead of only a smoothness-aware one at every stage. Straight-through
selection (hard chunk boundary + soft lag-similarity gradient) follows
the same `pq_table` hard/soft pattern this codebase already established
for `CodeEmbed`.
