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
