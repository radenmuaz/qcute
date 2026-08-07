# `qcute_fifo` v2 — design sketch (not implemented)

Status: algorithm sketch only, discussed in session but not built. Current
`qcute/qcute_fifo.py` (v1) trains by sampling ONE composition per step and
is explicitly NOT KV-cache compatible (merges mutate/replace slot
identity, invalidating old K/V). This doc lays out how a v2 could do
better on both fronts, plus a broader survey of *when-to-merge* scheduling
policies and how they compare on expressivity, speed, and fit for
structured (audio/image-like) data.

## 1. Training many compositions in one pass (compute/memory tradeoff)

v1's bottleneck: each step samples one composition (a length-`window`
sequence of bandwidths), builds one `[B, window, D]` sequence for it, runs
one attention pass. Different compositions never share computation even
though they're all just different slicings of the same underlying byte
range.

**Key observation**: a pyramid "node" (e.g. "the pair-embedding covering
bytes [100,102)") has a hidden state that's a fixed function of *causal
history up to that node* — it does not depend on which composition later
selects it. Compute every node once, and any number of compositions can
read off the nodes they need for free.

**Sketch**:
1. Over the max byte span (`window * max(bandwidths)`), build EVERY
   level's embeddings exhaustively (not just one composition's worth):
   level-1 (every byte), level-2 (every adjacent pair), level-4 (every
   adjacent quad), etc. `~2x` the max span in total nodes (geometric
   series) — a fixed cost regardless of how many compositions get trained
   against this step.
2. Run ONE causal self-attention pass over the union of all pyramid
   nodes, masked so node A may attend to node B iff B's byte span lies
   entirely within A's own already-seen input (generalizes
   `qcute_bytepool.build_cross_mask`'s no-leakage rule). Every node at
   every level now has a well-defined, composition-independent hidden
   state.
3. For each composition to train this step (sample K, or — since the
   enumerated set is often small, e.g. 513 for `window=512, bw=(1,2)` —
   use ALL of them in expectation instead of sampling): gather the
   relevant pyramid nodes (a composition is a monotonic path through the
   pyramid) and run FetchHead on them. Cheap — indexing + small per-node
   MTP heads, no further attention.
4. Total step loss = sum/average over however many compositions were
   served.

**Cost**: one pyramid pass costs `~O((2*window_max)^2)`. K separate
single-composition passes cost `K*O(window^2)`. E.g. `window=256`:
pyramid `~(512)^2=262144` vs. one composition `256^2=65536` — break-even
around K~=4; past that the pyramid approach is strictly cheaper per
composition trained, and near-free once K is in the low tens instead of
relying on random revisits of a 33K-composition set.

## 2. KV-cache compatible revamp

Root cause of v1's incompatibility: merging *destroys identity* — two
1-byte slots collapse into a new 2-byte slot, invalidating both old K/V
entries. Standard KV-caching requires old entries to stay valid forever;
merging violates that by construction.

The Part 1 pyramid fixes this too, because it never merges — only adds:

1. One APPEND-ONLY K/V cache per level: `cache[1]` (bytes) grows by 1 per
   incoming byte; `cache[2]` (pairs) grows by 1 every 2 bytes (built via
   `merge_proj` from the 2 newest byte nodes); `cache[4]` every 4 bytes;
   etc.
2. Each new node's K/V computed once, via normal causal self-attention
   against its own level's existing cache (append, don't rebuild) —
   standard KV-cache append, replicated per level.
3. Cross-level information flow (bytepool's cross-attention cascade, or
   v1's substitution mechanism) becomes ordinary cross-attention against
   an already-cached, already-causally-valid coarser level — no different
   from any encoder-decoder cross-attention KV cache.
4. Bounded memory ("FIFO"-ness) becomes EVICTION, not merging: once a
   level's cache exceeds budget, drop the oldest entries outright.
   Standard sliding-window KV-cache eviction — cheap, well-understood,
   never requires recomputing anything else (unlike v1's merge).
5. Side effect: "one active composition" mostly dissolves at inference —
   every level's cache is simultaneously available and growing in
   parallel, so prediction can cross-attend into whichever levels are
   useful rather than committing to one FIFO shape. Composition-sampling
   becomes training-time-only curriculum (e.g. level-dropout), not
   something inference has to track/mutate.

**Unifying point**: both problems dissolve into the same fix — stop
treating "the FIFO queue" as one mutating sequence; treat it as a fixed,
ever-growing, append-only multi-resolution pyramid, with training-time
composition sampling and inference-time context bounding both implemented
as *read-side selection/eviction* rather than *write-side mutation*.

## 3. When-to-merge scheduling policies (tree-algorithm analogues)

| # | Scheme | Trigger | Analogue | KV-cache compatible? |
|---|---|---|---|---|
| 1 | Greedy / carry-propagate (v1) | merge the instant budget is exceeded, oldest eligible pair first | binary counter carry propagation | No — destructive |
| 2 | Batch / bulk rebuild | accumulate a full window's worth of new input, then pool all at once | LSM-tree compaction / global rebuild | No — still destructive, just batched |
| 3 | Lazy with slack | tolerate overflow up to `window+eps` before merging | B-tree overflow tolerance | No — same destructiveness, lower frequency |
| 4 | Finger-tree / balanced leveling | merge points float to keep structure balanced, not strictly leftmost | finger tree / skip list | No — still a mutating merge |
| 5 | Content-adaptive | merge triggered by a mergeability score (low marginal info between neighbors) | BPE-style greedy pairwise merge | No — data-dependent but still destructive |
| 6 | Fixed per-level sliding bands | no trigger at all — each level has its own fixed-size window, positional not merge-driven | Gaussian/Laplacian pyramid, dilated/banded attention | **Yes — by construction** |
| 7 | Amortized/staggered (+ "keep-raw" hybrid) | at most one merge-step per token, round-robin across levels; hybrid never deletes raw, only adds a compressed summary | incremental/lazy amortized rebuilding | Hybrid variant: **yes** (nothing invalidated, only supplemented); plain variant: no |

Scheme 6 is the only one that's KV-cache compatible *by construction*.
Scheme 7's "keep raw, merge is additive" hybrid is the next best option
if the original "destructive compression" narrative needs preserving
while staying cache-safe — it just spends more memory to get there.
Schemes 1-5 are fundamentally incompatible with KV-caching as long as
"merge" means "replace" — not fixable by adjusting the trigger policy
alone.

## 4. Expressivity ranking (general case)

Axes that determine expressivity:
- **Resolution-decay profile**: how much raw detail survives at distance
  `d` back. Fixed-geometric schemes impose a hard, position-driven decay;
  content-adaptive can reshape it; full dense attention has none — the
  ceiling.
- **Reversibility**: destroyed (schemes 1-5) vs. unselected-but-cached
  (scheme 6, scheme 7-hybrid).
- **Cross-scale mixing richness**: genuine cross-attention (bytepool-
  style, queryable by many finer positions) vs. one-position substitution
  (v1's literal replacement, strictly weaker).
- **Adaptivity**: fixed position (1-4, 6) vs. data-dependent (5) — the
  latter spends a fixed budget more efficiently but is still a hard
  discretization, not dense attention's continuous soft weighting.

Ranking, most to least expressive:
1. Scheme 6, unbounded (no eviction) — closest to full attention, but
   still not equal: information reaches distant positions only through a
   chain of lossy summarizing ancestors, never direct byte-to-byte
   attention.
2. Scheme 7, keep-raw hybrid — ties #1, more memory-hungry.
3. Scheme 5, content-adaptive — best of the genuinely destructive
   schemes, spends its (permanent) budget where redundancy actually is.
4. Scheme 6 bounded (eviction) ~= schemes 1-4 — tied: same steady-state
   resolution-decay profile once band/window sizes match; differ only in
   *when* compression happens (latency/burstiness), not how much
   information survives. (Tie breaks in favor of whichever variant pairs
   with genuine cross-attention — axis 3.)

**Which is closest to full bytelm attention**: none reach it exactly —
that requires zero decay, which by definition means abandoning the
hierarchy and going back to plain `O(L^2)` attention. #1/#2 (unbounded
pyramid / keep-raw hybrid) are the closest practical approximations,
since they're the only ones that don't destructively discard byte-level
information, but the gap to true dense attention is structural: distant
information is only reachable through summarizing ancestors, not direct
long-range attention (the same gap Perceiver/Linformer-style methods have
against a plain transformer).

## 5. Ranking for structured data (audio/image-like)

Real audio/image codecs (wavelets, JPEG's DCT, MP3's subband filters)
universally use a FIXED multi-resolution decomposition — because it
aligns with real signal structure (smooth regions/steady tones compress
trivially at coarse resolution; edges/transients need fine resolution)
*and* because fixed, regular structure is what makes transforms fast
(FFT/DWT owe their speed to a fixed, predictable, vectorizable schedule).
Adaptive *bit allocation* is layered on top of that fixed structure in
practice (MP3's psychoacoustic model, JPEG's quantization tables) —
adaptive *decomposition structure itself* is rare, since data-dependent
control flow kills vectorization/hardware throughput.

Ranking, best to worst combined speed + expressivity for this data type:
1. **Scheme 6** — structurally identical to a Laplacian/wavelet pyramid;
   fully regular/predictable -> maximally vectorizable, and the fixed
   bands genuinely match audio/image's real multi-scale redundancy.
2. **Scheme 1** — same steady-state expressivity as #1, but bursty merge
   timing is less hardware-friendly (still statistically periodic, like a
   binary counter's amortized behavior).
3. **Scheme 2** — regular timing but bursty compute; fine for throughput-
   oriented offline encoding, worse for latency-sensitive streaming.
4. **Scheme 5** — highest theoretical expressivity-per-bit (literally what
   perceptual coding does), but as a standalone *structural* choice
   (adaptive merge boundaries, not fixed bands + adaptive bit-depth) it
   loses the vectorization speed that makes #1 fast.
5. **Scheme 3** — no domain-specific advantage; a dial between #1 and #2.
6. **Scheme 7 (keep-raw hybrid)** — good expressivity, but doesn't exploit
   the domain's genuine compressibility for speed/memory savings — wasteful
   when the data is known to be highly redundant.
7. **Scheme 4 (finger-tree)** — worst fit: its strength (mid-sequence
   rebalancing) is irrelevant to strictly-sequential streaming data, and
   it lacks #1's clean fixed regularity.

**The actual best design isn't on this list**: scheme 6's fixed pyramid
as the backbone (for speed) + scheme 5's adaptivity applied only to
*bit-depth/capacity per node*, not to *where merges happen* — keep the
geometric band structure fixed and regular, but let each node's embedding
dimension or quantization precision vary with local content complexity
(silence/flat regions get a thin representation, transients/edges get a
fuller one). That's the audio/image-codec playbook applied to this
architecture; none of the 7 schemes as described do it — scheme 6 is the
closest starting point to build it from.

## 6. No free lunch: optimizing for audio/image makes text worse

Scheme 6 wins for audio/image *because* those signals are physically
smooth/band-limited — adjacent samples/pixels are highly correlated
almost everywhere, so pooling at FIXED positions is close to lossless at
the coarse level almost regardless of where the boundary falls (the
premise wavelets/subband coding rely on).

Text bytes don't have that property. Redundancy in text is
combinatorial/frequency-driven, not spatially continuous — "th", "ing",
"tion" compress well because they're specific frequent substrings, not
because "byte i and byte i+1 are generally similar." A fixed merge
boundary will, half the time, straddle a meaningful unit (splitting "th"
across two pooled tokens) while gaining nothing from pairs that happen to
land together but aren't actually redundant. Fixed-position pooling is
matched to audio/image's structure and mismatched to text's — this is a
genuine no-free-lunch, not a hedge.

Evidence already in this repo: `qcute.bpelm`'s BPE tokenizer (the
handover doc's "strong baseline") is exactly scheme 5's family
(content-adaptive merge, chosen by corpus frequency, not position).
Nobody uses a fixed-stride byte-pooling tokenizer for text because it
doesn't compress nearly as well — the same structural mismatch showing up
empirically, not a coincidence.

**The resolution BPE already uses**: get scheme 5's expressivity without
scheme 5's runtime speed penalty by decoupling *when* the adaptivity
happens from *when* it's applied — learn merge rules once, offline, from
corpus statistics (`scripts/train_bpe.py`), then apply those learned
rules as a fixed, fast, deterministic algorithm at both train and
inference time. "Adaptive structure, fixed schedule," decoupled in time
rather than fused into every forward pass. This sidesteps the tradeoff
for a SINGLE, KNOWN modality — it does not generalize to genuinely mixed
multimodal streams (see §7): a BPE vocabulary trained on text produces a
domain-specific fixed schedule that's exactly as mismatched to audio/image
bytes as scheme 6 is to text.

## 7. Best tradeoff for multimodal (audio + image + text + raw bytes)

None of the 7 schemes alone are the right answer once the goal is a
SINGLE mechanism that handles audio, image, text, and raw bytes without
knowing in advance which modality it's looking at, and without an
offline per-modality preprocessing step (BPE's resolution in §6 doesn't
generalize here — a text-trained vocabulary is exactly as mismatched to
audio/image bytes as scheme 6 is to text).

### Scheme 8 — gated/routed fixed-schedule merge (new, not in §3's table)

Real-world precedent: ToMe (Token Merging, vision transformers) — merges
only tokens a learned similarity score judges redundant, within a fixed
computational budget, entirely via vectorized similarity/bipartite
matching (no branchy control flow, which is why it stays fast on GPUs
despite being content-dependent).

Mechanism: keep scheme 6's fixed positional schedule as the backbone
(regular tensor shapes, batchable, fast) — but at each node, a
lightweight learned gate decides HOW MUCH to trust the pooled/merged
representation vs. how much residual fine-grained signal to carry
alongside it. Smooth audio/image regions: the gate learns to trust the
pooling fully (behaves like scheme 6, near-optimal there). Text-like/
high-entropy regions: the gate learns to distrust it and preserve more
resolution (behaves closer to scheme 5's outcome, without scheme 5's
ragged/unbatchable merge-boundary search). One mechanism, no offline
per-modality preprocessing, no baked-in assumption about which modality
it's currently looking at.

KV-cache compatibility is NOT automatic here — it requires a real design
constraint: the gate must be causal (only look at history) and its
decisions about past tokens must never be revisited once made ("greedy
causal gating"). ToMe's usual mode (global, non-causal, one shot per
static image) would break append-only caching; the streaming-safe variant
is a genuine restriction on the mechanism, not a free property.

**Relation to H-Net / dynamic chunking**: H-Net replaces fixed
tokenization with a learned, differentiable chunking/boundary mechanism
trained END-TO-END with the language model itself, rather than BPE's
offline-learn-then-fixed-apply two-stage split (§6) — the same core idea
scheme 8 is reaching for: adaptive boundary/degree decisions that stay
part of the trainable model, not precomputed once and frozen. The
precise mechanics (exact boundary-scoring function, how it stays
batchable at scale, hard routing vs. a soft gate) aren't verified against
the source here — noted as the closest named precedent for scheme 8's
category of solution, not a claim of architectural equivalence.

### Per-axis rank tables

**Expressivity / information retention** (1=best)
| rank | scheme | verdict |
|---|---|---|
| 1 | 6, unbounded | closest to full attention; info never destroyed |
| 2 | 7, keep-raw hybrid | ties #1, costs more memory |
| 3 | 8, gated hybrid | adaptive spend beats a fixed budget, but still a bounded budget unless the gate is very permissive |
| 4 | 5, content-adaptive | best of the genuinely destructive schemes |
| 5 | 6 bounded ~= 1-4 (tied) | same steady-state decay profile regardless of scheduling policy |

**KV-cache compatibility**
| rank | scheme | verdict |
|---|---|---|
| 1 | 6 | compatible by construction |
| 1 | 7, keep-raw hybrid | compatible — nothing ever invalidated |
| 2 | 8, gated hybrid | compatible ONLY if gate decisions are causal and permanent — a real design requirement, not free |
| — | 1, 2, 3, 4, 5 | incompatible — "merge" means "replace" in all five |

**Speed / hardware-friendliness**
| rank | scheme | verdict |
|---|---|---|
| 1 | 6 | fully regular schedule, zero data-dependent control flow |
| 2 | 8, gated hybrid | fixed tensor shapes preserved; gate is a cheap vectorized per-node score (ToMe's whole design point), not branchy control flow |
| 3 | 2, batch rebuild | regular timing, big but predictable/batchable bursts |
| 4 | 1 ~= 3 (greedy / lazy-slack) | mostly regular (binary-counter periodicity) but irregular instant-by-instant timing |
| 5 | 4, finger-tree | rebalancing bookkeeping overhead, less regular than fixed bands |
| 6 | 7, keep-raw hybrid | unbounded memory growth — a bandwidth/footprint cost, not a control-flow one |
| 7 | 5, pure content-adaptive | slowest — genuine ragged/data-dependent structural decisions resist batching |

**Content-adaptivity** (does behavior change with content, independent of raw expressivity)
| rank | scheme | verdict |
|---|---|---|
| 1 | 5 | fully adaptive — merge boundary AND degree both content-driven |
| 2 | 8, gated hybrid | boundary fixed, but how much to compress is content-driven |
| 3 | 1, 2, 3, 4, 6, 7 (tied, zero adaptivity) | purely positional schedules, or (scheme 7) unconditional "keep everything" — neither is a content-based decision, just opposite fixed extremes |

**Multimodal generality** (audio + image + text + raw bytes, no per-modality preprocessing)
| rank | scheme | verdict |
|---|---|---|
| 1 | 8, gated hybrid | fixed backbone handles audio/image well, learned gate recovers text-appropriate behavior locally — one mechanism, no modality assumption baked in |
| 2 | 5, content-adaptive (runtime, not offline BPE) | good text fit, can in principle learn audio/image-appropriate behavior too, but real speed cost and no native KV-cache |
| 3 | 7, keep-raw hybrid | never wrong for any modality, but exploits none of their redundancy — safe generalist, not an optimized fit |
| 4 | 6, fixed pyramid | excellent for audio/image, poor for text (see §6) |
| 5 | 1, 2, 3, 4 | same fixed-schedule limitation as 6, destructive/non-cacheable on top, no offsetting benefit here |

### Caveat specific to images

None of these schemes (1-8) model 2D locality at all — they're
inherently 1D/sequential. A raster-scan byte serialization of an image
breaks a lot of the natural 2D neighborhood structure regardless of which
scheme processes the resulting byte stream. This is a real gap none of
the above addresses; a genuinely image-aware version would need a 2D
(or learned-scan-order) notion of "adjacency" before any of this
1D merge-scheduling machinery applies usefully.

## 8. Applying this to `qcutelm_vlt11` — towards KV-cache-efficient inference

`qcutelm_vlt11` (the recursive per-level clockwork sandwich — see that
module's own docstring) is a different architecture from `qcute_fifo`
(fixed tier hierarchy + per-level codelm substitution, not a merge-
scheduling FIFO), but the same KV-cache question applies to it, and the
answer turns out to be encouraging.

### Current state: correct, but not cache-efficient

`generate()`, `plan_coarse_codes`, and `detokenize_from_plan` (built this
session) all recompute every tier from scratch on the FULL sequence at
every generated byte — `model.run_tier(0, ...)` through `run_tier(N,
...)` called fresh each step. That's `O(L)` work per byte, `O(L^2)` total
to generate `L` bytes — the same "no KV cache, consistent with this
lineage's existing simplicity tradeoff" choice every module in this
repo has made so far. `plan_coarse_codes` itself IS cheap (a handful of
small codelm passes over a short code sequence, no tier computation at
all during the rollout) — but `detokenize_from_plan` still pays the full
`O(L^2)` cost, so the free-roll/late-detokenize split doesn't save wall-
clock time as currently implemented (confirmed directly: the expensive
part is untouched, planning is pure overhead on top of it for now).

### The key finding: v11 is NOT destructive, unlike fifo v1

`qcute_fifo` v1's KV-cache incompatibility (§2) has a root cause: merging
MUTATES — two slots collapse into a new one, invalidating old K/V.
`qcutelm_vlt11` never does this. Every tier's hidden state at every past
position is a fixed function of causal history, computed once and never
touched again — appending a new byte only ever adds a new position, it
never rewrites an old one. The substitution mechanism (a block-boundary
position's input comes from `codelm[level]`'s forecast rather than
`local_embed`) is a special-cased INPUT CONSTRUCTION for that one
position, computed once — after that, its resulting hidden state at every
tier is exactly as cacheable as any ordinary position's. This means
v11's current KV-cache incompatibility is a pure IMPLEMENTATION
simplicity choice (recompute everything, because it's easy to get right),
not an architectural constraint the way fifo v1's destructive merging is.

### Concrete plan for a cache-efficient rewrite

1. Give every tier's `CausalSelfAttention` a standard append-only K/V
   cache (ordinary transformer KV-caching, no different from any GPT-
   style decoder — tier_0 alone is already a plain causal byte LM once
   cached this way).
2. Cache each tier's `local_code`/`local_embed` per position too —
   these only ever need recomputing for the NEWEST position each step;
   every past position's value is already fixed (see above).
3. At a block-boundary position, the substituted value
   (`z_proj[level](codelm[level]'s forecast)`) is computed once, the
   moment that position is first processed, then flows into the cache
   exactly like any other position's value from then on — no special
   handling needed after that one-time computation.
4. `codelm[level]` itself operates on a short, append-only code sequence
   (one entry per `periods[level]` bytes) — trivially cacheable the same
   way, and cheap regardless since its sequence length is already
   `L / periods[level]`.

With this, generation cost drops from `O(L^2)` to `O(L)` total (`O(1)`
amortized per byte, plus periodic — not per-byte — codelm calls at each
level). This is also where `plan_coarse_codes`/`detokenize_from_plan`'s
speed payoff would actually materialize: with real caching,
`detokenize_from_plan` could reuse the SAME cache built during
`plan_coarse_codes`'s prompt-encoding pass, rather than recomputing the
prompt's own tier states from scratch a second time — the free-roll
split only pays off once this is built, not before. Not implemented yet;
this section is the design target, not a completed change.
