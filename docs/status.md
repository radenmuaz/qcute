# Status

Tracks progress on the **active** lineage — `qcute/qcute_refine_v1.py` /
`qcute/qcute_refine_v2.py` / `qcute/qcute_refine_v3.py` — going forward
from this session onward.

**`qcute/qcute_refine_v3.py` (new this session)**: clone of v2 plus
EncoderLevel fusion (`Config.fuse_encoder_levels`, default True) — fixes
a finding uncovered this session: v2's `val_bpb`/checkpointing metric
(`byte_loss`) is computed by a purely bottom-up sweep with ZERO access to
the coarser code or cross-attention (only the separate, detached
`tok_loss` path ever benefited from it — see
[docs/kv_contribution.md](kv_contribution.md)). v3 adds a second pass
that re-runs each non-top level with a new cross-attention step BEFORE
its own self-attention blocks, attending to the level-above's own
(detached) hidden state under the same jagged causal mask
`DecoderLevel` already uses — so `byte_loss` itself now depends on, and
must learn to use, the coarser code. `configs/qcute_refine_v3_rope.py`
(queued) clones `qcute_refine_rope.py`'s architecture exactly, isolating
fusion's own effect. Full mechanism/rationale in the module's own
docstring.

**`qcute_refine_v3_rope` RESULT — first `qcute_refine` architecture to
beat `bytelm_xs1_ctx1024` outright.** Best val_bpb **2.4302 @ step 3500**
(mean it/s 1.039, min train bpb 1.4159) vs. `qcute_refine_rope`'s 2.6310
(same architecture, fusion off) — a **0.201 improvement**, by far the
largest single delta of any ablation this session, and it beats
`bytelm_xs1_ctx1024`'s 2.4870 by 0.057. Direct forward-value conditioning
(byte_loss genuinely depends on the coarser code now, not just
shared-weight gradient dynamics like `pq_table` — see
[docs/bpe_like_boundaries.md](bpe_like_boundaries.md) and the session's
own byte-code-causality discussion) is, on this evidence, a materially
stronger lever than anything tried in the pure-v2 family. `qcute_refine_
v3_rope_pq` (queued, stacks fusion + pq_table) will show whether this
and `pq_table`'s gain are additive or redundant.

**Receptive-field confound identified**: `bytelm_xs1_ctx1024`'s
`CausalSelfAttention` (`qcute/bytelm.py`) has no window concept at all —
always dense, full 1024-byte reach on every prediction. Every
`qcute_refine_v2` config's level 0 uses `attn_window=256` (`tiny_byte_
window`: 8) — a QUARTER of xs1's own reach, or worse — and per the v3
finding above, v2's `byte_loss`/`val_bpb` never had a working channel to
anything past that window anyway. So the whole-session xs1-vs-
`qcute_refine_v2` comparison was never apples-to-apples on receptive
field alone, independent of the fusion question. `configs/qcute_refine_
dense0_starve1.py` (queued, third in line after `pq_table`/`v3_rope`)
isolates this directly: `attn_window=(-1, 4)` — level 0 widened to dense/
full-1024 (matches xs1 exactly), level 1 deliberately STARVED to 4 code
positions (~16 raw bytes) so the interesting question stays clean: does
the coarser level/cross-attention add anything once level 0 alone
already matches xs1's own reach? Uses `qcute_refine_v2` (not v3), a
deliberate choice to keep this test orthogonal to the fusion question — a
v3 companion (same windowing + fusion) is a natural follow-up once both
report back.

**Result: RAN — helps, but does not close the gap alone.** Best val_bpb
**2.5947 @ step 3900** (mean it/s 1.964) — better than `qcute_refine_rope`'s
2.6310 (windowing does help some), but still well short of `bytelm_xs1`'s
2.4870. So pure receptive-field widening (matching xs1's dense reach at
level 0, with no cross-attention conditioning of `byte_loss` at all — this
config predates fusion) is NOT sufficient on its own to close the gap.
Combined with `pq_table`'s 2.4816 and `v3_rope`'s 2.4302 (both closing or
beating the gap through entirely different mechanisms — code-embedding
expressiveness and direct forward-value conditioning, respectively — see
[docs/kv_contribution.md](kv_contribution.md) §6), this confirms the real
levers were the ones tried elsewhere this session, not receptive field per
se. The receptive-field CONFOUND was still real and worth ruling out
(the xs1 comparison was never apples-to-apples on window size alone) —
it just wasn't the dominant explanation.

**`qcute/qcute_refine_v4.py` (new this session)**: clone of v3 with
`DecoderLevel` removed entirely — see the session's own architecture
discussion: fusion and `DecoderLevel` turned out to do the literal same
job with the same input requirements (both predict a level's own next
token given that level's own sequence history, optionally conditioned on
the coarser code), and `DecoderLevel` never contributed to `byte_loss`/
`val_bpb` even in v3 (its reads stayed detached) — so removing it costs
nothing measurable and saves real compute (an entire extra `CrossBlock` +
its own embeddings every step). v4 also fixes a real gap v3 left open:
v3's `generate_no_cache`/`generate_kv_cache` were copied unchanged from
v2 and never touched fusion at all (training used it, generation didn't).
v4's generation is fusion-aware — `generate_kv_cache` implements a new
hierarchical caching scheme (a CLEAN self-attention cache per level, plus
one FUSED cache for level 0 specifically, the only level ever sampled
from) — validated via `validate_generation` (exact match against the
slow reference path) across fusion on/off, a 3-level config, identity
quantization, `mlp` code-embedding, and prompts as short as 1 byte.
`configs/qcute_refine_v4_pq.py` stacks v4's leaner architecture with
`pq_table`, the two strongest independent wins this session, to see how
far the combination goes.

**Result: RAN — best val_bpb 2.4588 @ step 1900** (run took much longer
than usual in wall-clock terms due to unrelated system memory pressure
during this session, not a code/architecture issue — see it/s caveat
below). **Clean rerun** (`qcute_refine_v4_pq_rerun`, no swap-thrashing
this time, `caffeinate`-protected): best val_bpb **2.4565 @ step 2400**
— reproduces closely (within 0.002 of the corrupted run, confirming the
VALUE was never actually corrupted, only its wall-clock/it-s measurement
was), mean it/s **1.291** — notably faster than `qcute_refine_v3_rope`'s
1.039, consistent with v4 being cheaper than v3 (no `DecoderLevel`
compute) while still paying more than plain v2 (extra self-attention
pass) — matches the architectural reasoning worked out earlier this
session. Beats `qcute_refine_pq_table` (2.4816, fusion off) and
`bytelm_xs1_ctx1024` (2.4870), but is slightly WORSE than `qcute_refine_
v3_rope` (2.4302, fusion alone, no pq_table) — **stacking fusion +
pq_table did not beat fusion alone.** This is genuinely informative: the
two mechanisms (direct forward-value conditioning via fusion,
shared-weight gradient regularization via pq_table — see
[docs/kv_contribution.md](kv_contribution.md) §6) are not simply
additive, and may be partially redundant or even mildly interfering
routes to the same underlying fix (both, in different ways, are giving
`byte_loss` better access to what the coarser level knows). `qcute_
refine_v3_rope_pq` (v3, same architecture, DecoderLevel still present but
inert) confirms this isn't specific to v4's leaner setup: best val_bpb
**2.4639 @ step 3000** — close to v4_pq's 2.4588 (v4 slightly better),
both still worse than fusion alone (2.4302). Non-additivity holds under
both architectures.

**`qcute_refine_v4_rope` (fusion alone, no pq_table, no DecoderLevel) —
close but not exact reproduction of `v3_rope`.** Best val_bpb **2.4710 @
step 2800**, mean it/s 1.537. `v3_rope`'s own 2.4302 was this session's
best single result; `v4_rope`'s 2.4710 is close (both far better than
every pq_table-stacked or baseline-losing config) but not an exact match
— a real 0.041 gap. No explicit seed-pinning was used between these two
separate training invocations, so this is most plausibly ordinary
run-to-run variance from different random initialization, not evidence
against the detach-based independence reasoning (`DecoderLevel`'s reads
were always detached from `byte_loss`, so removing it shouldn't
systematically change fusion-alone's own result) — but worth recording
as a real, unresolved small gap rather than claiming an exact
reproduction that didn't quite happen.

**`Config.fuse_position` ablation — `qcute_refine_v4_pq_postfuse`
("post") vs. `qcute_refine_v4_pq` ("pre", default).** Best val_bpb
**2.4678 @ step 2100** (mean it/s 1.413) — slightly WORSE than `v4_pq`'s
2.4565, a real but small 0.011 gap. So fusing BEFORE self-attention (the
original design: every position gets cross-level context first, then
self-attention lets that context propagate across positions) beats fusing
AFTER (positions mix purely among themselves first, only the final
representation sees the coarser code, no further propagation) — small
edge, but consistent with "pre" being the more expressive ordering
(gives self-attention a chance to spread fused information around,
"post" doesn't). "pre" stays the default.

**v4 forward-pass cost vs. v2/v3, worked out explicitly**: v4 is cheaper
than v3 (strictly — removing `DecoderLevel` removes compute, nothing
added back; params confirm the direction: `v4_pq` 2.640M vs `v3_rope`
3.563M, not a matched pair since one has `pq_table`, but the point holds
regardless). v4 is NOT cheaper than v2, counterintuitively — v2's default
decoder REUSES `h_prev`/`h_curr` (already computed by PASS 1), paying
only two small linear projections + one lightweight `CrossBlock`, no
self-attention re-run at all. v4's fusion (PASS 2) re-runs level 0's
ENTIRE self-attention trunk on top of the cross-attention — the
"self-attention runs twice" cost established earlier this session. So per
step: v2 pays one self-attention pass + a cheap cross-attn-only decoder;
v4 pays two self-attention passes + a cross-attention fusion, nothing
cheap to offset it. v4 buys real forward-value conditioning v2's decoder
structurally never could — that conditioning isn't free. Note: `v4_pq`'s
own logged it/s (0.216) is NOT a fair speed measurement — that run hit
the swap-thrashing episode; `v3_rope`'s clean it/s (1.039, unaffected) is
the better real-world reference point, and the architectural reasoning
above (not the noisy wall-clock numbers) is what the "v4 vs v2" cost
comparison should rest on.

**`qcute_refine_v4_depth22` (`tier_n_layers=(2,2)`) — closest ANY
`qcute_refine` config has come to a matched baseline all session.** Best
val_bpb **2.4097 @ step 1600** (mean it/s 0.834, min train bpb 0.867) —
within **0.0017** of `bytelm_xs3_ctx1024`'s 2.4080, the strongest
baseline result this whole investigation has been measured against.
Beats every fusion/pq_table/`fuse_position` variant tried (`v3_rope`'s
2.4302, `v4_rope`'s 2.4710, `v4_pq`'s 2.4565) by simply doubling depth —
no new mechanism, no new training dynamics, the single most standard
lever in deep learning, exactly the "test the boring thing first" plan
from a few turns back. Strong validation of the `k32_narrow`
fusion-contribution probe's own finding (docs/kv_contribution.md §7:
~90% of fusion's benefit was capacity/depth, not cross-level content) —
depth alone, with NO cross-attention tricks beyond default fusion,
recovers essentially all of that gap on its own. **Caveat**: not a
params/FLOPs-matched comparison — `depth22` costs 4.152M params/8.963G
FLOPs vs. `bytelm_xs3`'s 2.625M/5.369G (58%/67% more) — so this
specific comparison doesn't prove qcute_refine "wins" at matched
compute, only that depth is likely the dominant lever within its OWN
architecture family, more so than any cross-attention mechanism tried.
A genuinely fair test would need a depth/width-matched dense bytelm
baseline at `depth22`'s own budget — not yet run.

**`qcute_refine_v4_depth21` (`tier_n_layers=(2,1)`) — beats one baseline,
loses to the closer-matched one.** Best val_bpb **2.3957 @ step 1900**
(held through the end — overfit after that point, train bpb collapsed to
~1.0 while val climbed), mean it/s 0.952. Honest framing against both
reference points (params 3.363M, FLOPs 8.561G):

| vs. | params diff | FLOPs diff | val_bpb delta |
|---|---|---|---|
| `bytelm_xs3_ctx1024` (2.4080) | +28.1% | +59.5% | **−0.0123 (beats it)** |
| `bytelm_xs_mtp4_ctx1024` (2.3650) | **−1.4% (closest match)** | +22.7% | +0.0307 (loses) |

So: beats `xs3`, but only by spending 28% more params to do it — not a
fair win. Against `bytelm_xs_mtp4`, the genuinely closest params match
found for ANY config this session (−1.4%), `depth21` still loses, despite
having fewer params (though 23% more FLOPs).

**`qcute_refine_v4_k2` (`Ks=(2,2)`, finer code blocks) — loses at a
genuinely close match.** Best val_bpb **2.4495 @ step 3100** (mean it/s
1.391, min train bpb 1.493). This is the CLOSEST params match of any
config this session:

| vs. | params diff | FLOPs diff | val_bpb delta |
|---|---|---|---|
| `bytelm_xs3_ctx1024` (2.4080) | **−1.9% (closest of any config)** | +8.9% | +0.0415 (loses) |

Fewer params AND a real (if modest) FLOPs premium, still loses by 0.042
bpb — the cleanest "no, this lever alone doesn't close the gap" result
this session. Finer code granularity (K=2 vs. the usual K=4) didn't help
on its own.

**No `qcute_refine` config has yet beaten a baseline at a comparison
that's fair on BOTH params and FLOPs simultaneously** — `depth22` came
closest in absolute score (0.0017 from `xs3`) but at far more compute;
`k2` is the closest in params-matching but loses clearly; `depth21`
beats `xs3` but only at +28% params. Depth remains the strongest lever
found this session, but hasn't closed the gap outright at matched
compute yet.

**`qcute_refine_v4_k1` (`Ks=(1,1)`, degenerate/no compression) — no
close baseline match, but informative anyway.** Best val_bpb **2.4232 @
step 1500**, mean it/s 1.119, params 2.575M, FLOPs 6.866G:

| vs. | params diff | FLOPs diff | val_bpb delta |
|---|---|---|---|
| `bytelm_xs1_ctx1024` (2.4870) | +145.2% | +219.8% | −0.0638 (beats it, unfair — 2.5x params) |
| `bytelm_xs3_ctx1024` (2.4080) | **−1.9% (close)** | +27.9% | +0.0152 (loses) |

Same story as `k2`: close params match to `xs3`, still loses, this time
by a smaller 0.015 bpb — the least-bad "no compression at all" result,
but still a loss at near-matched params. Removing BSQ's whole
compression rationale (every raw position becomes its own code block)
didn't help versus keeping it at `K=4`/`K=2`.

**`qcute_refine_v4_k32_narrow_postfuse` (`fuse_position="post"` on the
K=32/narrow-window architecture) — pre still beats post, same direction
as the earlier `pq_postfuse` ablation.** Best val_bpb **2.5033 @ step
3600** vs. `k32_narrow`'s (pre, default) **2.4926** — post loses by
0.0107, identical params/FLOPs (only fusion's position in the block
differs), consistent in direction and rough magnitude with the earlier
`v4_pq_postfuse` finding (post lost by 0.011 there too). Probed both
checkpoints with `scripts/probe_v4_fusion_contribution.py` — full
writeup in [docs/kv_contribution.md](kv_contribution.md) §8. Headline:
post's standalone trunk (fusion fully removed) IS more robust than pre's
(4.6761 vs. 4.9456 bpb — confirms one half of the "representational
separation" hypothesis), but post's fusion, when present, leans on real
coarser-level CONTENT even less than pre's already content-starved
fusion does (big_noise costs post only 0.0176 bpb vs. pre's 0.0699) — the
trade doesn't net out to a win here. `fuse_use_null_kv=False` runs for
both positions (`k32_narrow_nonull`, `k32_narrow_postfuse_nonull`) are
queued to isolate how much of this is the learned null slot specifically
vs. `fuse_cross`'s other parameters.

**`qcute_refine_v4_k32_narrow_postfuse_nonull` (post + no null KV) — best
of the K=32 family so far.** Best val_bpb **2.4799 @ step 1800** — beats
BOTH `post+null` (2.5033) AND `pre+null` (2.4926, previously the best of
the family). Same params/FLOPs as every other K=32/narrow-window cell
(no null slot removes ~256 negligible params). Probe
(`scripts/probe_v4_fusion_contribution.py`, see
[docs/kv_contribution.md](kv_contribution.md) §9 for the full table):
`null_only`/`big_noise` recover 98.1%/99.0% of fusion's total benefit —
even more capacity-dominated than post+null's 96.4%/99.2%, continuing the
trend. But `unconditional_pass1` (fusion fully removed) is dramatically
WORSE without the null slot — 5.6564 vs. post+null's 4.6761, a +0.98 bpb
jump despite the null slot contributing ~0% to `normal`-mode performance.
Reading: the null slot's real value looks like a TRAINING-time
regularizer (keeps PASS 1's own trunk more standalone-capable, evidently
because `_fuse`'s cross-attention is never "guaranteed available" the way
it is without a null fallback), not a forward-pass content contribution —
those are two different effects, easy to conflate from `normal`-mode
numbers alone.

**2x2 grid complete — `qcute_refine_v4_k32_narrow_nonull` (pre + no null
KV) is the last cell.** Best val_bpb **2.4961 @ step 2700** — essentially
a wash vs. `pre+null`'s 2.4926 (+0.0035), unlike "post" where removing
null improved the trained result by 0.023. Full grid (all four cells,
identical params/FLOPs, only `fuse_position`/`fuse_use_null_kv` differ):

| cell | trained best val_bpb | unconditional_pass1 (no fusion) |
|---|---|---|
| pre + null | 2.4926 | 4.9456 |
| post + null | 2.5033 | 4.6761 |
| **post + no-null** | **2.4799 (best of grid)** | 5.6564 (worst of grid) |
| pre + no-null | 2.4961 | 5.0124 |

Reading (full mechanistic writeup in
[docs/kv_contribution.md](kv_contribution.md) §10): "pre" is robust to
null-slot removal in both directions (trained result AND
unconditional-ablation floor barely move); "post" is sensitive in both
(trained result improves, but the no-fusion floor roughly doubles its
penalty) — because in "pre" mode `self.blocks` always processes fusion's
OUTPUT regardless of whether a null fallback existed, while in "post"
mode `self.blocks` runs entirely independently of `_fuse`, so whatever
fallback existed during training more directly reshapes what the rest of
the network learns to depend on. Best single result of the whole K=32
family is post+no-null (2.4799), narrowly beating `bytelm_xs1_ctx1024`
(2.4870) by 0.007 — within noise, and still well short of `bytelm_
xs3_ctx1024` (2.4080). **The grid's value was mechanistic (what
null_kv/fuse_position actually do), not a competitive result** —
consistent with the session's overall finding that no `qcute_refine`
lever tried has closed the gap to a properly compute-matched dense
baseline.

**`qcute_refine_v4_bpe4_imitate` — worst result of the session relative
to a matched baseline.** K=4/window=8 at level 0 (near-bag-of-8-bytes),
level 1 `window=256` (dense, its own full sequence) + `tier_n_layers=
(1,2)` — a more literal DEPTH-imitation of the 4-layer `bytelm`/`bpelm`
baselines than any params/FLOPs-matched config tried. Best val_bpb
**2.5073 @ step 2600**, params 3.363M, FLOPs 5.742G:

| baseline | params diff | FLOPs diff | val_bpb delta |
|---|---|---|---|
| `bytelm_xs_mtp4_ctx1024` (2.3650) | **−1.4% (closest match)** | −17.7% | **+0.1423 (loses)** |
| `bytelm_xs3_ctx1024` (2.4080) | +28.1% | +6.9% | +0.0993 (loses) |
| `bytelm_xs1_ctx1024` (2.4870, 1-layer diagnostic) | +220.3% | +167.4% | +0.0203 (**loses even to this**) |
| `bpelm_4096_paramsmatch` (2.3531) | −1.7% (close) | +119.4% | +0.1542 (loses) |
| `bpelm_8192` (2.3500) | −36.0% | +113.9% | +0.1573 (loses) |
| `bpelm_8192_ctx448_flopsmatch_rope` (2.3559) | −24.6% | +43.8% | +0.1514 (loses) |
| `bpelm_16384_ctx448_flopsmatch` (2.3438) | −48.7% | −2.2% | +0.1635 (loses) |
| `bpelm_32768` (2.1340) | −70.9% | −2.8% | +0.3733 (loses badly) |

Loses to EVERY baseline, including its own closest params match
(`bytelm_xs_mtp4_ctx1024`, −1.4%) by 0.142 bpb — a far bigger margin than
any other closely-matched pair this session (`k2`/`k1`/`depth21` all lost
by only 0.015–0.041 at similar match quality), and loses even to the
trivial 1-layer `bytelm_xs1` diagnostic despite 3.2x the params and 2.7x
the FLOPs. Reading: literal depth-imitation (narrow level-0 window +
deep level 1) did NOT close the gap the way plain depth WITHIN
`qcute_refine`'s own family did (`depth22` came within 0.0017 of `xs3`)
— crippling level 0's own receptive field to force reliance on fusion
hurt more than the added level-1 depth helped. Consistent with the
session's broader finding: the hierarchical/windowed structure itself,
not depth alone, is the harder constraint for `qcute_refine` to overcome
relative to a dense baseline of equivalent depth.

**This was the last queued config of the session's ablation sweep.**
Overall verdict across everything tried (depth `(2,2)`/`(2,1)`, finer/
degenerate K `2`/`1`, the K=32 narrow-window family and its full 2x2
`fuse_position x fuse_use_null_kv` grid, and this depth-imitation
config): **no `qcute_refine` lever closed the gap to a properly
compute-matched dense `bytelm`/`bpelm` baseline.** Depth remains the
single strongest lever found (`depth22`'s 0.0017 gap to `xs3`, at +58%
params/+67% FLOPs — not a fair win, but the closest approach). Fusion's
own benefit is mostly capacity, not cross-level content (§7-10 above,
~88-99% recoverable with zeroed/noised KV). `fuse_use_null_kv`'s real
value looks like training-time regularization (shapes how standalone-
capable the rest of the network becomes) rather than forward-pass
content. This session's own code change (see `qcute_refine_v4.py`'s
`_encode`/`forward` — PASS 2 no longer overwrites PASS 1, both now
backprop as separate terms, `Config.fusion_ntp_weight`) targets exactly
this: pushing every `LevelLM` toward standalone competence directly
during training, rather than relying on fusion alone and hoping
standalone competence emerges as a side effect. Retraining under this
new scheme is in progress as of this note.

**(Not actually the last config — the session continued substantially
further; superseded by the sections below.)** New lineage members this
session: `qcute_refine_v4_1.py` (extreme weight sharing — one `LevelLM`
trunk shared across every level, `Ks`/`attn_window` stay per-level) and
`qcute_refine_v4_2.py` (further unified: single shared `dq`/embed/head
across every level including byte, concat-only fusion, no cross-attention
module at all). Also added to `qcute_refine_v4.py`: `fuse_position`
gained `"both"` (two independent `CrossBlock`s) and `"concat"` (fusion
folded directly into `self.blocks`' own windowed self-attention, no
separate cross-attention module — see `docs/kv_contribution.md` for the
full mechanism and correctness verification).

**`qcute_refine_v4_k32_narrow_nonull_uncond` retrained under the new
additive-loss scheme: best val_bpb 2.4992** — matches the earlier
`postfuse_nonull_uncond` retrain (2.4967) closely; both land in the same
range as the original (pre-additive-loss) scheme's 2.4926-2.4961, no
regression from adding the PASS1 standalone term to the loss.

**v4.2's fully-unified head shows a genuine, persistent training
instability — not just an efficiency tradeoff.** `qcute_refine_v4_2_
k32_narrow` (K=32/narrow-window, concat-only, single shared head/embed
across every level): best val_bpb **4.0369 @ step 3700**, dramatically
worse than every other K=32 config this session (2.48-2.60) and
`bytelm_xs1_ctx1024` (2.4870) — a gap far too large to be explained by
the independent-bit-BCE-is-an-upper-bound caveat alone. Root cause: the
code level's own loss (`val_level1_bpb_pass1`, computed through the SAME
shared head byte level uses) never stabilizes across the full run —
std over just the second half of training (steps 2000-4000) is still
0.522, no late-training convergence. Full trajectory/isolation analysis
(confirming concat itself trains fine at 2.7150 by step 1100 when
weights are unshared, and testing whether a different shared head type
helps) in [docs/kv_contribution.md](kv_contribution.md) §11.

**Resolves an earlier open question**: `bytelm_xs1_ctx32` (genuine
from-scratch 1-layer bytelm, `context=32`) reaches best val_bpb **2.8664
@ step 4000** — worse than the K=32 family's own standalone/unconditional
`level0_bpb_pass1` values (~2.53-2.60) seen throughout this session.
Confirms that `qcute_refine`'s good standalone byte-level performance at
window=32 reflects a genuine benefit from JOINT training with the fusion
task (multi-task shaping), not simply "32 bytes of raw context is already
enough on its own" — a plain single-task model given the same budget
does meaningfully worse. `bytelm_xs1_ctx8`: best val_bpb 3.357 @ step
2300, already overfitting past that point (unlike ctx32, which was still
improving at the final step) — 8 bytes is a harsher regime than 32,
consistent with earlier findings.

**`qcute_refine_v4_k32_narrow_concat` (plain v4, unshared weights,
`fuse_position="concat"`) finished cleanly: best val_bpb 2.4925**, in the
same 2.48-2.60 range as every other unshared K=32 config, no oscillation
in either level's own trajectory throughout the full 4000-step run. This
confirms concat fusion itself is not what causes v4.2's instability above
— concat trains exactly like `"pre"`/`"post"` fusion do when weights
aren't unified.

**Byte-vs-code task-incompatibility ablation — finished; the two leading
hypotheses turn out to compose rather than compete.**
`qcute_refine_v4_2_k32_narrow_byte256` (v4.2, `byte_head_256way=True`:
level 0 gets its own fully unshared exact 256-way head; TRUNK still
shared across levels) finished at **best val_bpb 2.5660 @ step 3900** —
back in the healthy 2.48-2.60 range, a big recovery from
`v4_2_k32_narrow`'s 4.0369. But `val_level1_bpb_pass1` (the code level's
OWN loss) is still unstable throughout — last-quarter std 0.52, mean
still rising to 3.58 at the final step, essentially unresolved. Reading:
unsharing the byte head doesn't fix the shared trunk's genuinely unstable
code-level predictions, it just stops that instability from LEAKING into
the byte-level readout via shared weights — in the fully-shared run, the
same head produces both predictions, so the code level's bad gradients
directly corrupt level 0's byte prediction too; once separated, `val_bpb`
recovers even though the underlying pathology in the shared-trunk/code
pathway is arguably untouched. `qcute_refine_v4_1_k32_narrow_shared`
(trunk-shared, v4.1's own scheme — head/embed never unified at all, so
nothing for instability to leak through even if present) is now the run
that answers whether `val_level1_bpb_pass1`'s instability is inherent to
trunk-sharing itself or specific to something v4.2 adds on top. Full
detail in [docs/kv_contribution.md](kv_contribution.md) §11.

**`qcute_refine_v4_2_k32_narrow_ssm` (chain/SSM head instead of
independent) killed early at step 1550 — near-total collapse the whole
way, no recovery trend** — `byte_acc` stuck at 0.004-0.012 (near-random)
and `val_bpb` stuck at 7.45-7.59 from step 100 onward, far worse than any
other config this session. Also ~2.5x slower than `byte256` (1.03 it/s
vs. 2.54 it/s), consistent with `BitPredictHeadSSM`'s known heavier
per-step cost. Not yet clear whether this is trunk-sharing-specific or a
more basic issue with this head under this config — no baseline exists
yet for `bit_head_class="ssm"` under v4/v4.1's UNSHARED scheme to compare
against. Two follow-ups queued (both ahead of the rest of the session's
remaining runs): `qcute_refine_v4_2_k32_narrow_byte_softmax_head_only` —
a narrower version of `byte256` (`Config.byte_softmax_head_only`, new
this session: level 0 keeps the shared dq-bit input embedding/`code_pre`,
only its output readout becomes an unshared 256-way head), isolating
whether the shared OUTPUT head specifically was what mattered for
`byte256`'s bpb recovery; and `qcute_refine_v4_2_k32_narrow_attn_id16` — a
reclone of the (now-deleted) `ssm` config with `Ks`/`attn_window` left
UNCHANGED at `(32,32)`, swapping `bit_head_class="attn"` and adding
`bit_inner_downsample=16` (the chain head's own internal working width,
`256 -> 16` — the "16" in the filename refers to this, not `Ks`) —
testing whether the collapse is head-type-specific or resolved by a much
cheaper chain head, without confounding it with a block-grid change.

**`byte_softmax_head_only` finished: best val_bpb 2.7696** — a partial
recovery (better than the fully-shared 4.0369, worse than `byte256`'s
2.5660), `val_level1_bpb_pass1` still just as unstable. Refines the
picture: unsharing the OUTPUT head alone recovers PART of the gap;
unsharing embed/`code_pre` too (as `byte256` does) recovers more — the
shared embed/`code_pre` independently contributes to how much of the
underlying instability leaks into the byte-level metric, not just the
head. Full detail in [docs/kv_contribution.md](kv_contribution.md) §11.

**`quant_type="simplex"` (new this session) — generalizes `byte_softmax_
head_only`'s "give a level an exact softmax classifier" idea to EVERY
level, and drops BSQ's hypercube-grid code representation entirely.**
Session ask: "generalize with flag to mode where every level is softmax
head 256 way... instead of sign and ste, do gumbel softmax ste...
basically no grid assumption that bsq carries... this mode do not use
bsq linear map, but uses shared embedding table for all level... maintain
2 modes now: bsq, and simplex." Where BSQ's `dq` independent sign-bits
form an implicit hypercube grid (`2**dq` corners, bit-factorized),
`quant_type="simplex"` makes every level's code a flat, unstructured
`V=2**code_bits`-way category — no bit factorization at all, a point on
the probability simplex (hence the name). Quantization is
`gumbel_quantize` (new function): **default is a cheap deterministic
softmax+argmax straight-through** (`soft + (hard - soft).detach()`, the
same idiom `bsq_quantize` already uses for `sign()`+STE — session: "is it
ok to have no gumbel, just default argmax and ste like bsq did... because
gumbel is expensive") — `Config.use_gumbel_noise=True` opts into genuine
Gumbel-Softmax sampling instead, at real extra per-step cost. No separate
`code_pre`/`ntp_head` modules exist in this mode at all — a level's own
embedding table IS its classifier too (weight-tied,
`F.linear(h, embed.weight)`), and at the default `code_bits=8`
(`V=256=vocab`), byte level 0's table and every code level's table are
literally the SAME OBJECT (one pool, full uniform sharing — the most
extreme point yet in this file's own weight-sharing lineage).
`code_bits<8` (session: "this mode can generalize to n<8") is also
supported — a genuinely smaller, more heavily compressed code alphabet
for levels above byte, at the cost of splitting the pool (byte's own
vocab=256 table and the code levels' smaller table can't literally be the
same object at different sizes — same split pattern `byte_head_256way`
already uses). Intuition (session): "the model with end-to-end learn best
byte code to downsample longer bytestream" — let training discover its
own best discrete code/downsampling scheme, not one constrained by BSQ's
hypercube structure. Gradient correctness verified directly: the
deterministic STE path's backward gradient through `gumbel_quantize`
matches a PURE (unrounded) softmax's gradient exactly (`torch.allclose`,
`atol=1e-6`), and every parameter in a full forward+backward smoke test
(trunk included) receives a nonzero, finite gradient — confirms the
straight-through trick is doing genuine, correct work, not silently
breaking the gradient path. `configs/qcute_refine_v4_2_k32_narrow_
simplex.py` (K=32/narrow-window, default `code_bits=8`/`gumbel_tau=1.0`/
`use_gumbel_noise=False` — testing the mode's own defaults first, not an
ablation of them) queued.

**`attn_id16` finished: no collapse, meaningfully more stable than every
other shared-head config, still short of the healthy range.** Best
val_bpb **3.6163 @ step 4000**, `val_level1_bpb_pass1` second-half std
**0.333** — clearly better than `ssm` (diverged), `v4_2_k32_narrow`
(0.522), `byte256` (1.14), `byte_softmax_head_only` (~1.0). Also faster
than `ssm` (1.79 it/s vs. 1.03). `BitPredictHeadAttn` is a categorically
better-behaved shared head type, but "more stable" hasn't yet meant
"competitive" — 3.6163 is still well outside the 2.48-2.60 unshared
range. **`qcute_refine_v4_2_k32_narrow_attn_id4` — killed early, genuinely
diverging.** (session: "repeat attn_id16 to clone and make it less
aggressive like x4" — `bit_inner_downsample=4` instead of 16.) `val_bpb`
was actively RISING (5.34 -> 5.66 from step 800 to 900), killed at step
900. Confirmed it used plain defaults, not `pq_table` (session: "check
does it use pq or not").

**`qcute_refine_v4_2_k32_narrow_attn_id4_pq` — `code_embed_mode=
"pq_table"` fixes the divergence and improves both stability and fit.**
(session: "try use pq and rerun".) Treats the dq-bit BSQ code as a table
lookup instead of a linear combination — ~28x more effective degrees of
freedom (rank ≤9 vs. 256) and parameters (2,304 vs. 65,536) than the
default `"linear"` mode. Finished cleanly: **best val_bpb 3.2067**,
`val_level1_bpb_pass1` second-half std **0.196** (even better than
`id16`'s 0.333), train bpb ~2.3-2.6 (better fit than `id16`'s ~2.8-3.5).
The single cleanest positive result in the `chain`-head family this
session — the codebase's own "dq is starved" hypothesis (previously only
validated for the independent-bit head) extends directly to
`BitPredictHeadAttn`'s shared chain head too.

**Gap identified (session: "recheck how v4 BitPredictHeadAttn is more
expressive vs v4.2")**: the `BitPredictHeadAttn` CLASS is byte-identical
between v4 and v4.2 (diffed directly — only difference anywhere is
v4.2's precomputed `h_scale` buffer, a pure efficiency fix, not a
capacity change). The real difference is WIRING: v4 has no sharing
mechanism at all, so `build_bit_head` is called fresh per level — every
level gets its own PRIVATE chain head. v4.2 calls `build_bit_head` once
(only the pool owner); every other level aliases the SAME object
(`self.ntp_head = shared_head.ntp_head`). So `attn_id16`/`attn_id4`/the
deleted `ssm` run all confound "is this chain head TYPE worse than
independent-bit" with "is a SHARED chain head worse than a PRIVATE one"
— no config this session isolates the two. Worth a private-chain-head
control once the current family reports back. Full detail in
[docs/kv_contribution.md](kv_contribution.md) §11.

**v4.2's underfitting is dose-dependent on sharing degree — a TRAIN-side
finding, distinct from §11's val-side instability story.** Session
question: "seems qcute variants underfitting generally vs baselines bpe
and byte at step 4000." Confirmed directly: every UNSHARED config —
`bytelm_xs1_ctx1024`, `qcute_refine_v4_bpe4_imitate`,
`qcute_refine_v4_k32_narrow_concat`, `qcute_refine_v4_k32_narrow_
nonull_uncond` — bottoms out at train bpb ~1.6-2.0 regardless of
architecture family. No v4.2 shared-pool config gets close, and the gap
tracks sharing degree almost monotonically: `byte256` (partial unshare)
≈ the unshared cluster; `byte_softmax_head_only` (narrower unshare)
measurably worse; the fully-shared configs (`attn_id16`, `v4_2_k32_
narrow`) stuck 1-2 full bpb above it — on TRAIN data, not just val. This
is a capacity story, complementary to (not competing with) §11's
stability story: `attn_id16` is simultaneously the MOST capacity-starved
run in this comparison and MORE stable than `byte256`/`v4_2_k32_narrow`
— the two problems don't have to move together.

**Two more baselines added, showing the OPPOSITE failure mode**:
`bytelm_xs3_ctx1024` (more capacity than xs1) drives train bpb to ~1.3
(lower than any unshared qcute config) but its val bpb (~2.70-2.77) is
WORSE than xs1's 2.55 — genuine overfitting, the mirror image of v4.2's
underfitting. `bpelm_32768` (best of every BPE baseline tried this
session — all of `bpelm_8192`/`bpelm_8192_converged`/`bpelm_4096_
paramsmatch`/`bpelm_8192_ctx448_flopsmatch_rope`/`bpelm_16384_ctx448_
flopsmatch` show the same pattern) is far more extreme: train bpb
~0.006-0.009 (essentially memorized), val bpb 3.17-5.02 across every BPE
config — worse than every byte-level baseline and most qcute variants,
including the underfit ones. BPE at this corpus scale overfits
catastrophically, categorically worse than qcute's worst underfitting —
opposite failure directions, neither a "model to beat." Full table and
reasoning in [docs/kv_contribution.md](kv_contribution.md) §12.

**`quant_type="simplex"` needs stochastic exploration; `BitPredictHeadSSM`
gains a per-position head.** `qcute_refine_v4_2_k32_narrow_simplex`
(default `use_gumbel_noise=False`) interrupted at step 1000 with
`best_val_bpb` still improving (4.28->4.12->3.79->3.71) but noisy, `byte_
acc` stuck low (~0.24-0.30) — `qcute_refine_v4_2_k32_narrow_simplex_
gumbel` (`use_gumbel_noise=True`, genuine Gumbel-noise sampling) queued
in its place to test whether stochastic exploration helps. Separately,
`BitPredictHeadSSM`'s `self.head` changed from a single `nn.Linear(d_
inner, 1)` shared across all `dq` bit positions to `nn.Linear(d_inner,
dq)` — a PRIVATE weight row per bit position (session: "let each bit
timestep use different head... similar to independent mode, but has
state"), while keeping the alpha-decayed recurrent state unchanged — a
genuinely new point in the design space (private weights + stateful
conditioning, vs. `"independent"`'s private+stateless or every other
chain head's shared+stateful). Fixed/loop consistency reverified exactly.

**`ssm_id1_pq` (full width, no downsample) killed for being too slow**
(~0.88 it/s) — directly confirming the session's own compute analysis
(`BitPredictHeadSSM` at full width costs ~272x more FLOPs per bit-chain
than a plain independent head). Config deleted. Three more changes made
in response, all reverified (fixed/loop consistency + gradient checks):
**einsum instead of full-matrix+diagonal** (removes an `n`x compute/
memory waste in the per-position head), **concat instead of add**
(`fetched` is now `cat([h_scale*h, state_contrib])`, not their sum —
strictly more information reaches the head), and a **trainable BOS
state** (`self.bos_state`, zero-init, replaces `state_proj(zeros)` at
position 0 with something the model can actually learn away from zero).
`qcute_refine_v4_2_k32_narrow_ssm_id4_pq_concat` (`bit_inner_downsample=
4`, matching `attn_id4`'s width) replaces it, queued to the front —
already confirmed faster (~1.85 it/s vs. the killed run's 0.88 and even
`attn_id4_pq`'s 1.38). Full detail in
[docs/kv_contribution.md](kv_contribution.md) §13.

**Correction: plain `simplex` (no gumbel noise) was never actually
unstable.** The step-1000 "noisy" read above was a snapshot of one
truncated restart segment, not the run's real trend — inspecting its full
uninterrupted stretch (steps 100->2200) shows clean, monotonic
convergence, val_bpb 69.6 -> 3.21, best 3.7105 logged. Re-queued for a
fresh full run to confirm at higher step counts.

**`simplex_gumbel` crashed — genuine MPS-specific numerical bug, fixed at
the source, not a queue artifact.** `torch.AcceleratorError: scatter:
index -1 is out of bounds for dimension with size 256`, found via the raw
piped stdout log (the structured `Logger`-written `run.log` showed no
error, just silently stopped). Root cause: `F.gumbel_softmax`'s internal
`-log(-log(u))` sampling has no epsilon clamp on its uniform draw; a rare
float32 underflow to exactly 0.0/1.0 produces `±inf`, colliding `inf`s
produce `NaN`, and `NaN.argmax()` returns `-1` specifically on MPS (undefined
elsewhere). Fixed by replacing the `F.gumbel_softmax` call inside
`gumbel_quantize` with a manual, epsilon-clamped Gumbel sampling
(`u=torch.rand_like(logits).clamp(eps, 1-eps)`,
`eps=torch.finfo(dtype).tiny`). Verified via gradient check plus a
20,000-iteration x `[256,256]`-per-call stress test producing zero
non-finite outputs, and a full-model smoke test. Re-queued behind
`attn_id1_pq`. Full trajectory/root-cause detail in
[docs/kv_contribution.md](kv_contribution.md) §13.

**`BitPredictHeadAttn` revamped to match; all three `attn_id*` configs
deleted** (session: "delete all attn 4.2 ablation, need revamp"). Same
per-position/concat treatment as SSM, plus one more simplification:
`self.qkv_proj`/`self.out_proj` replaced by Q/K-only (`self.q_proj`/
`self.k_proj`) — attention weights are still learned, but the values
being weighted-summed are the RAW bit-value embeddings, not a learned
V-projection (session: "simplify _mha... remove out proj, v proj, only k
and q proj, basically like weighted sum of h and embeds"). No BOS
parameter needed this time — `h_t` reaches the head via CONCAT (never
summed), so position 0 already gets a distinct signal from `h_t` alone
without a separate learned placeholder (session: "h_t is the concat
bos"). Reverified fixed/loop consistency + gradients.

**Compute analysis** (`D=256`, `dq=8`): vs. the pre-revamp `attn` head,
params drop 48%/22%/6% and FLOPs drop to ~51%/52%/57% of the old cost
across `downsample=1/4/16` (roughly a 2x FLOP speedup, from cutting
projection cost `~4·d²` -> `~2·d²`). Compared against a plain 256-way
softmax classifier and the independent 8-bit linear head: `attn_id16`
(downsample=16) now beats `softmax-256` on BOTH params (13x fewer) and
FLOPs (10x fewer) at once; `attn_id4` undercuts it on params (0.40x) at
roughly FLOP-parity (1.14x); only full-width `attn` costs more than
`softmax-256` on both axes. Full tables in
[docs/kv_contribution.md](kv_contribution.md) §14. Two runs queued on
the revamped modules: `qcute_refine_v4_2_k32_narrow_attn_id1_pq`
(full-width `attn`, the ceiling-capacity point where it costs MORE than
`softmax-256` on both axes — testing whether that extra capacity is
worth it) and `qcute_refine_v4_2_k32_narrow_ssm_id1_pq` (same idea for
`ssm` — an earlier version of this exact config was killed for being too
slow BEFORE the concat/einsum/`bos_state` revamp; the einsum fix
specifically targets that head's own worst inefficiency, so this reruns
the question rather than assuming the old verdict still holds).

**`attn_id1_pq` killed for speed** — 0.57 it/s, actually slower than the
`ssm_id1_pq` run killed earlier for the same reason (0.88 it/s), despite
the revamp's own compute savings. Full-width `attn` at this scale just
isn't worth the ~2hr/4000-step budget as the first of 8 queued jobs.
Config kept (not deleted) for a future downsample requeue.

**`BitPredictHeadAttn` gains a trainable BOS embed inside `_mha` itself**
(session: "make zero_vec trainable embeds") — a different role from the
head-level BOS question already settled above ("h_t is the concat bos"):
this new `self.bos_val_emb` replaces the plain zero vector that stood in
for "no previous bit yet" in the ATTENTION sequence itself (`_mha`'s own
Q/K/V content at position 0), which previously gave attention no way to
distinguish "no previous bit" from "a previous bit that embeds near
zero." Reverified fixed/loop consistency (`atol=1e-5`) and gradient flow.
New config `qcute_refine_v4_2_k32_narrow_attn_id4_pq.py`
(`bit_inner_downsample=4`, `code_embed_mode="pq_table"`, same family as
`ssm_id4_pq_concat`) queued on this updated head.

**`qcute_refine_v4_2_k32_narrow_simplex_l2`** — clone of `simplex` (§13),
`n_layers=2` instead of 1 (session: "queue simplex (code_bits=8) with
double layer, at front"), queued at the front of the training chain.
Tests whether trunk depth can compensate for `simplex`'s own extreme
sharing (byte level and every code level literally share ONE embed/head
table at `code_bits=8` — see [docs/kv_contribution.md](kv_contribution.md)
§13). Measured params/FLOPs (`FlopCounterMode`, batch=1, context=1024):
1.709M params / 6857M flops/fwd, roughly double `simplex`'s own
0.921M/3573M as expected. Against the `bytelm` baselines specifically:
params land closest to `bytelm_xs1_ctx1024` (1.050M, +0.66M away), but
FLOPs land closest to — and actually EXCEED — `bytelm_xs3_ctx1024`'s
5369M, despite `simplex_l2` having fewer params than that 3-layer
baseline (1.709M vs. 2.625M). The two-pass fusion mechanism plus
cross-level trunk sharing cost more per added layer than plain depth
does in a single-tower transformer — the fair FLOPs-matched comparison
target for this run is `xs3`'s 2.4078 best_val_bpb, not `xs1`'s 2.4870.
Full detail in [docs/kv_contribution.md](kv_contribution.md) §15.

Queue as of this writing: `simplex_gumbel` (running) -> `simplex_l2` ->
`attn_id4_pq` -> `simplex` (fresh full run) -> `ssm_id1_pq` (full-width,
revamped) -> `bpe4_imitate_uncond` -> `bpe4_imitate_uncond_l1x4` ->
`v4_k32_narrow_both` -> `v4_1_k32_narrow_shared` (KEY, isolates
trunk-sharing alone vs. v4.2's own additions).

**Queue cleared — several strong results.** `simplex_l2` (n_layers=2)
finished at best_val_bpb **2.5892**, essentially TYING `byte256`'s 2.5660
for best v4.2 result of the session (0.023 apart, within noise) —
confirms trunk depth is a real (if FLOP-costly, see
[docs/kv_contribution.md](kv_contribution.md) §15) substitute for
`simplex`'s missing per-level embed/head privacy. `simplex` itself, on a
clean rerun after the queue-leak contamination (below), finished at
2.8687 — far better than the earlier interrupted read (3.7105),
reinforcing the earlier correction that plain `simplex` was never
actually unstable. `simplex_gumbel` finished at 2.9443 — respectable, but
the extra stochastic exploration still doesn't beat the clean
non-Gumbel rerun.

**Regression found: revamped `attn_id4_pq` (with the new `bos_val_emb`)
scores 3.5659 — WORSE than the OLD pre-revamp `attn_id4_pq`'s 3.2067**,
despite the revamp's own FLOP/param savings (§14) and the theoretical
motivation for `bos_val_emb` (§15). Flagged as a genuine regression, not
noise — left as an open question (per-position weights may be
data-starved at downsample=4, or Q/K-only attention with no learned V
may be a real expressivity cut the FLOP savings don't compensate for).
`ssm_id1_pq` (revamped, full-width) also finished this time (unlike its
pre-revamp version, killed for being too slow) — 3.7708, the weakest
finished chain-head variant this session, confirming full-width chain
heads of either flavor aren't worth their compute at this scale.

**`v4_k32_narrow_both` (`fuse_position="both"`, v4 lineage not v4.2) is
the best v4/v4.2-lineage result of the ENTIRE session: best_val_bpb
2.4443**, nearly matching `bytelm_xs3_ctx1024`'s 2.4078 (a 3-layer
single-tower baseline) despite using only one self-attention layer per
tier. `bpe4_imitate_uncond`/`_l1x4` (2.5533/2.5493) and
`v4_k32_narrow_concat` (2.4925) all finished strong too. Full ranking
table and detail in
[docs/kv_contribution.md](kv_contribution.md) §16.

`v4_1_k32_narrow_shared` (the KEY isolator run) is now running — the
last item in the queue.

**`v4_1_k32_narrow_shared` finished: best_val_bpb 2.5254 — ANSWERS the
long-open instability question.** Close to `byte256`/`simplex_l2`
(2.5660/2.5892), far better than the fully-shared `v4_2_k32_narrow`
baseline (4.0369). **Trunk-sharing ALONE (v4.1's own design — one trunk
reused across levels, nothing else unshared) is NOT what causes v4.2's
instability/underfitting.** v4.2 additionally shares the embed table,
NTP head, AND `code_pre` across every level — it's specifically that
EXTRA layer of sharing, not trunk-sharing itself, that's responsible.
Every v4.2 config that partially/fully unshares embed/head recovers most
or all of the gap to this healthy result — consistent across the whole
session's ablation family. Full detail in
[docs/kv_contribution.md](kv_contribution.md) §17.

**Structured/cheap alternatives to a dense V-way softmax classifier**
(session: "use structured matrix... replace dense linear map to 2**n way
output softmax... some loss in repr ok for params saving"): new
`FactoredSoftmaxHead` (outer-sum of two small D->v1/D->v2 projections,
`v1*v2==vocab`) — 8x fewer params/FLOPs than dense at vocab=256 (8,224 vs
65,792 params). Session then asked to compare against plain low-rank
("how good is factoredsoftmax vs just low rank... analyze rank") — new
`LowRankSoftmaxHead` (classic softmax bottleneck) is THEORETICALLY MORE
EXPRESSIVE at matched budget: factored forces every class into a rigid
zero-free-parameter `w1_i+w2_j` template, while low-rank gives every
class its own free coefficient vector within the same rank ceiling — a
strict superset. Both queued (`byte_factored`/`byte_lowrank`, rank=16
matched budget) for a direct empirical test. Also implemented (from an
earlier ask): `BitPredictHeadHSoftmax`, classic hierarchical softmax
over the dq-bit tree — gives every one of `2**dq-1` tree NODES its own
weight vector (unlike attn/conv/ssm's one-direction-per-POSITION,
shared-across-every-prefix bottleneck) — measured FLOP savings vs dense
softmax grow from 31x (dq=8) to 3,855x (dq=16), though params stay tied
to dense at every scale. Full tables in
[docs/kv_contribution.md](kv_contribution.md) §17.

**`BitPredictHeadConv` made ~171x/228x cheaper via a depthwise
`conv_impl`** (session: "consider making bitpredictconv more efficient,
last time huge compute, maybe try group conv or depthwise") — the
existing "conv1d"/"matmul" impls are fully dense across channels,
costing 525,313 params / 8,392,704 FLOPs at full width, the actual "huge
compute" referenced. New `conv_impl="depthwise"` (per-channel K-tap
filters, no cross-channel mixing, via `einsum` not `nn.Conv1d`): 3,073
params / 36,864 FLOPs — cheap enough to finally test `conv` at full
width for the first time this session. Also built `BitPredictHeadConvDilated`,
a WaveNet-style dilated conv stack — hit and then FIXED a real ~300x
wallclock regression along the way: the first version used `nn.Conv1d`
directly per layer (assumed safe since only called in the parallel/
batched path), measured at 298ms/fwd; swapping to `unfold`+`einsum` (no
`nn.Conv1d` at all, same fix `BitPredictHeadConv`'s own impls already
use) brought it to 0.53ms/fwd — the FASTEST chain-head variant
benchmarked this session. Generation support (`_forward_loop`) added
afterward and verified (`validate_generation` parity, exact match); now
wired into `Config.bit_head_class="conv_dilated"`/`build_bit_head`/CLI
and queued for training (`conv_dilated`, `mode="depthwise"`). A `"dense"`
mode (full cross-channel mixing per layer, verified to exactly reproduce
a real `nn.Conv1d(groups=1)` stack) was also finished — ~25% fewer
params/FLOPs than a single big dense kernel, but actually SLOWER in
wallclock at this scale (per-layer overhead dominates) — not queued for
training, a useful negative data point. Full writeup in
[docs/kv_contribution.md](kv_contribution.md) §17.

**`v4_1_k32_narrow_shared` finished: best_val_bpb 2.5254.** Close to
`byte256`/`simplex_l2`, far better than the fully-shared `v4_2_k32_narrow`
baseline (4.0369) — answers the long-open instability question: trunk-
sharing ALONE is NOT what causes v4.2's instability/underfitting. v4.2
additionally shares the embed table, NTP head, and `code_pre` across
every level — it's specifically that extra sharing, not trunk-sharing
itself, responsible. Every v4.2 config that partially/fully unshares
embed/head recovers most or all of the gap. Full detail in
[docs/kv_contribution.md](kv_contribution.md) §17.

**`downsample` decoupled from `h`'s own dimension for Attn/SSM** (session:
"can the downsample flag only be applied on embeds, h maintains full
dim") — feasible for concat-based heads (Attn, SSM: drop `in_proj`, keep
`h` full-width in the concat) but NOT possible for `BitPredictHeadHSoftmax`
(its `h @ node_weight` is a genuine dot product, fundamentally requiring
matching dims) and not attempted yet for Conv/ConvDilated (add-based,
would need an add->concat conversion first). While implementing this,
**`BitPredictHeadAttn` was reverted to v4's original design** (full QKV
self-attention + `out_proj`, single shared head) — the session's own
earlier revamp (concat/per-position/Q-K-only/`bos_val_emb`) was found to
REGRESS empirically (§16/§17: 3.5659 vs. the original's 3.2067), so
rather than retrofit the h-decoupling onto an already-worse design, this
starts from the design that actually worked. The revamped version is
preserved as a commented-out reference block, not deleted. Full detail
in [docs/kv_contribution.md](kv_contribution.md) §18.

Follow-up (session: "queue more experiments to test this hypothesis,
repr loss because of downsample h," and "allow indp heads for each
timestep... on by default"): added `downsample_h`/`per_position_head`
flags to both classes so the pre-decoupling behavior can be A/B'd
directly against the new one at the same downsample ratio. Four configs
queued: `attn_id4_hfull`/`attn_id4_hds`, `ssm_id4_hfull`/`ssm_id4_hds`.

**`BitPredictHeadWordPredict`** (new — session: "design another head,
wordpredict, which decompose to word like 8 bit, 4 bit... implement
until done complete with ar gen and config to queue"). Decomposes the
dq-bit code into `n_words=dq//word_bits` WORDS, each a genuine
`2**word_bits`-way softmax — a middle ground between hierarchical
softmax's per-bit tree and one expensive flat `V=2**dq` softmax. Word
`i`'s classifier conditions on `cat([h, embed(word_0),...,embed(word_
{i-1})])` (session: "past chain prob conditioning make simpler but more
expensive" — plain concat, growing linearly, no recurrence/attention).
`word_bits==dq` degenerates to a single flat softmax, verified numerically
identical to a plain `nn.Linear`. Fully wired end-to-end (new `code_head_
mode="word"`, own loss, own generation path) and verified (fixed/loop
consistency, gradients, causality, full-model integration,
`validate_generation` parity). `_forward_fixed` batches all words into
ONE kernel launch via padding (session: "find way to parallel launch
kernel, maybe pad") instead of `n_words` separate matmuls. Two configs
queued: `word4` (`word_bits=4`) and `word2` (`word_bits=2`). Full detail
in [docs/kv_contribution.md](kv_contribution.md) §19.

Full queue as of this writing: `byte_lowrank` (running) -> `conv_depthwise`
-> `conv_dilated` -> `hsoftmax` -> `attn_id4_hfull` -> `attn_id4_hds` ->
`ssm_id4_hfull` -> `ssm_id4_hds` -> `word4` -> `word2`.

**Housekeeping (this session)**: `EncoderLevel` renamed to `LevelLM`
throughout `qcute_refine_v4.py` — the class does both the "encode"
(PASS 1, produce the code) and "decode" (PASS 2, fused/conditioned
prediction) jobs now, so "Encoder" alone was misleading; "LevelLM"
matches the module's own "N-level recursive NTP tower" framing (each
level runs its own LM). Also added `qcute/qcute_refine.py` — a thin
alias (`from qcute.qcute_refine_v4 import *`) with the convention that
the no-suffix file always points at the latest version; promoting to a
future v5 is a one-line change there, not a duplicated file.

**Everything prior** (Phase 0-3 of the original `continuous_tokenizer_
handover.md` plan, and the full `qcutelm`/`qcutelm_vlt`/`qcutelm_pyramid`/
`qcute_fifo`/`qcute_bytepool` fork-by-fork narrative that plan produced) is
archived at [docs/archive/status_archive.md](archive/status_archive.md) —
that lineage itself is archived under `qcute/archive/`
(`configs/archive/`), superseded by `qcute_refine`. Read the archive for
historical context/reproducibility; nothing there is still being acted on.

`bytelm.py`/`bpelm.py` remain the active baseline comparison points (not
archived) — see `docs/archive/status_archive.md` for their own original
setup narrative, still valid.

**This file was split this session** (had grown to 472 lines mixing
several long-running investigative threads) — it now stays lean:
baseline numbers, params/FLOPs, lineage pointers, and the main
ablation-family results table. Deep-dive threads moved to their own
docs, cross-referenced below:
- [docs/kv_contribution.md](kv_contribution.md) — does `DecoderLevel`'s
  cross-attention KV actually matter? Three probes/experiments.
- [docs/torch_compile.md](torch_compile.md) — `--compile` on
  `qcute_refine_v2`: root cause, fix, MPS speed verdict.
- [docs/bitpredict_heads.md](bitpredict_heads.md) — `BitPredictHead`
  speed (linear vs. attn/conv/ssm, matmul reparam, inner-downsample).
- [docs/bpe_like_boundaries.md](bpe_like_boundaries.md) — is
  `qcute_refine` doing anything BPE-like? Brainstorm + math for
  content-adaptive pooling within the fixed-K grid (not yet implemented).
- [docs/hierarchical_fusion_designs.md](hierarchical_fusion_designs.md) —
  generalizing fusion beyond adjacent-only chains for N>2 levels:
  DenseNet-style pervasive/dense fusion, MoE-style gated routing
  (weighted vs. discrete), recursive/cascading refinement (not yet
  implemented).

## Baseline numbers (current reference point for every `qcute_refine` comparison)

All 8000-step runs unless noted, batch_size=16, `datasets/enwik8_1M.gz`.
**mean it/s recomputed this session** for every row here and in the
`qcute_refine_v2` table below, all via the SAME consistent method —
`last_logged_step / elapsed_s_at_that_row` from each run's own
`run.jsonl` (includes periodic-eval overhead, not train-only throughput;
this is why some numbers here differ from earlier informal in-session
estimates — those used inconsistent/unspecified methods, these don't).
Best val_bpb/step computed from each log's final (non-stale) segment.

| run | context | best val_bpb | @ step | mean it/s | wall time |
|---|---|---|---|---|---|
| `bytelm_xs_mtp4_ctx1024` | 1024 bytes | **2.365** | 1700 | 0.904 | 2:27:31 (8000 steps) |
| `bpelm_8192` | 256 tok (~845 byte-equiv) | **2.350** | 800 | 3.878 | 0:34:25 (8000 steps) |
| `bpelm_32768` | 256 tok (~973 byte-equiv) | **2.134** | 500 | 1.971 | 1:07:41 (8000 steps) |
| `bytelm_xs3_ctx1024` (3-layer, this session) | 1024 bytes | **2.408** | 2100 | 1.563 | 0:42:39 (4000 steps) |
| `bpelm_4096_paramsmatch` (params-matched to `rope_3level_curriculum`) | 384 tok | **2.3531** | — | — | 0:14:51 (4000 steps) |
| `bpelm_16384_ctx448_flopsmatch` (FLOPs-matched to `decoder_trunk`) | 448 tok | **2.3438** | 400 | — | killed @ step ~2750/4000 |
| `bpelm_8192_ctx448_flopsmatch_rope` (FLOPs-matched to `rope`/`identity`) | 448 tok | **2.3559** | 500 | — | killed @ step ~2900/4000 |
| `bytelm_xs1_ctx1024` (DIAGNOSTIC: 1-layer, this session) | 1024 bytes | **2.4870** | 3000 | 3.429 | 0:19:26 (4000 steps) |
| ~~`qcute_refine_v2_byte4_code256_simple` ("v1")~~ | 1024 bytes | ~~**2.485**~~ | 5600 | 2.42 | 1:00:53 (8000 steps) |

**This config/run was later DELETED this session** (see the "Full
qcute_refine_v2 ablation-family comparison" section's own CORRECTION
note below for why — its actual cross_attn_rope status at training time
was ambiguous, not the confirmed value this row implied). Left struck
through, not removed, so this table's own history stays intact; treat
`qcute_refine_rope` (in the later table) as its replacement reference
point instead.

**Both `flopsmatch` bpelm configs (16384 and 8192 vocab) overfit
catastrophically and were killed early**: same pattern each time — val_bpb
bottoms out early (2.3438 @ step 400 for the 16384 config, 2.3559 @ step
500 for the 8192 one) then climbs monotonically past 4.0 by step ~2700-
2900 while train bpb collapses to ~0.01-0.03 — large vocab heads have
enough capacity to memorize the ~237K-token train set outright once past
a few hundred steps, regardless of exact vocab size in the thousands
range. `best.pt` was saved before the divergence each time, so the
reported val_bpb is unaffected; only the wasted remaining steps were cut
(both runs killed manually around step 2750-2900/4000).
`bpelm_4096_paramsmatch`, by contrast, trained cleanly to its full 4000
steps (see the ablation-family table below for full detail on all
three).

`qcute_refine_v2`'s "v1" run: worse best-bpb than either bpelm variant,
better than plain `bytelm`, but at ~2.1x `bytelm`'s throughput and ~4.3M
fewer params (2.706M vs. bytelm's 3.412M). Every baseline here overfits
well before 8000 steps (see the step-budget finding below) — these
best-bpb numbers, not the final-step ones, are the actual comparison
target.

## Params/FLOPs comparison table (this session)

Two new closer-param-matched baselines added this session:
`configs/bytelm_xs3_ctx1024.py` (3-layer variant of `bytelm_xs_mtp4_ctx1024`
— same d_model=256/n_heads=4/mtp_heads=4/context=1024, only n_layers 4->3;
required adding a `--n_layers` override flag to `qcute/bytelm.py`, which
previously only exposed `--context`/`--mtp_heads` as preset overrides) and
`configs/bpelm_16384_xs3.py` (3-layer, vocab=16384, same d_model=256/
context=256 pattern as `bpelm_32768.py` — `--n_layers` was already a
supported override on `qcute/bpelm.py`). Both `steps=4000` (this session's
step-budget finding). Reason: want baselines closer in params/depth to
the `qcute_refine` lineage's own 2-3 level towers than the original
4-layer/8000-step bytelm/bpelm baselines were.

FLOPs = single forward pass, batch_size=1, real op count via
`torch.utils.flop_counter.FlopCounterMode` (same methodology as
`scripts/bench_forward.py`), CPU, using each config's own context
length. Sorted by params:

| run | params | flops/fwd (batch=1) |
|---|---|---|
| `qcute_refine_v1` | 1.244M | 2.916G |
| `bytelm_xs3_ctx1024` | 2.625M | 5.369G |
| `qcute_refine_pass_through` | 2.642M | 3.695G |
| `qcute_refine_rope` | 2.706M | 3.862G |
| `qcute_refine_v2_byte4_code256_identity` | 2.706M | 3.862G |
| `qcute_refine_rope_3level_curriculum` | 3.414M | 4.330G |
| `qcute_refine_decoder_trunk` | 4.424M | 5.878G |
| `qcute_refine_v4_pq` (fusion + pq_table, no DecoderLevel) | 2.640M | 5.340G |
| `bpelm_16384_xs3` | 6.561M | 3.355G |

**`qcute_refine_v4_pq` vs. every baseline, params/flops-sorted** (all
FLOPs measured the same way — `FlopCounterMode`, single forward pass,
batch=1, CPU, each config's own context length; `%diff` is vs. `v4_pq`'s
own 2.640M params):

| baseline | params | flops/fwd | %diff params | val_bpb | vs. `v4_pq` (2.4588) |
|---|---|---|---|---|---|
| `bytelm_xs1_ctx1024` | 1.050M | 2.147G | −60.2% | 2.4870 | +0.0282 (v4_pq wins, unfair — 2.5x fewer params) |
| **`bytelm_xs3_ctx1024`** | **2.625M** | **5.369G** | **−0.6% (closest)** | **2.4080** | **−0.0508 (beats v4_pq)** |
| `qcute_refine_v4_pq` | 2.640M | 5.340G | — | 2.4588 | — |
| `bytelm_xs_mtp4_ctx1024` | 3.412M | 6.979G | +29.2% | 2.3650 | −0.0938 |
| `bpelm_4096_paramsmatch` | 3.420M | 2.617G | +29.5% | 2.3531 | −0.1057 |
| `bpelm_8192_ctx448_flopsmatch_rope` | 4.460M | 3.993G | +68.9% | 2.3559 | −0.1029 |
| `bpelm_8192` | 5.253M | 2.684G | +99.0% | 2.3500 | −0.1088 |
| `bpelm_16384_ctx448_flopsmatch` | 6.560M | 5.872G | +148.5% | 2.3438 | −0.1150 |
| `bpelm_32768` | 11.544M | 5.906G | +337.3% | 2.1340 | −0.3248 |

**Honest read**: `bytelm_xs3_ctx1024` is an almost exact double-match to
`v4_pq` — within 0.6% on params AND 0.5% on FLOPs simultaneously, the
closest/fairest single comparison in this table. At that genuinely
matched compute, `bytelm_xs3_ctx1024` **beats** `v4_pq` by 0.051 bpb.
Every larger baseline beats `v4_pq` too. The only baseline `v4_pq` beats
outright is `bytelm_xs1_ctx1024` — but that's at 2.5x FEWER params, not
a fair comparison in `v4_pq`'s favor. So despite fusion+pq_table being
the strongest `qcute_refine` architecture built this session (beats every
other variant, including `xs1` specifically), it has NOT closed the gap
to a plain dense 3-layer bytelm at genuinely matched compute — worth
stating plainly rather than only citing the comparisons that flatter it.

**FLOPs and memory diverge — `qcute_refine_v4_k32_narrow.py`.** Session
question: can `qcute_refine` match xs1/xs3's FLOPs while pushing level 1
toward real token-granularity (`Ks=(32,32)`, ~32-byte blocks) and giving
level 0 an extremely narrow local window (`attn_window=(32,32)` — level 0
sees only 32 raw bytes via self-attention, level 1 gets dense/full
attention over its own 32 positions, fusion's own reach becomes
effectively unbounded since `fuse_kv_window` inherits `windows[1]=32 ≥`
level 1's own block count)? Params/FLOPs: 2.640M / 4.895G — close to BOTH
`bytelm_xs1_ctx1024` (2.147G) and, more closely, `bytelm_xs3_ctx1024`
(5.369G, −8.8%).

**Memory does NOT track FLOPs, measured directly** (peak RSS, CPU,
forward+backward, isolated subprocess per model, net of the ~150MB
python+torch import floor):

| model | flops/fwd | net peak memory |
|---|---|---|
| `bytelm_xs1_ctx1024` | 2.147G | 30.1MB |
| `qcute_refine_v4_pq` (K=4, baseline) | 5.340G | 71.7MB |
| `bytelm_xs3_ctx1024` | 5.369G | 97.7MB |
| `qcute_refine_v4_k32_narrow` (K=32) | 4.895G | **118.3MB** |

`k32_narrow` uses MORE memory than every other row despite having the
LOWEST FLOPs of the three `qcute_refine`/`xs3` entries — +65% memory over
the K=4 baseline it's cloned from, despite −8.3% fewer FLOPs. Checked
whether this is a "needs flash attention" gap: a separate scaling test
(`F.scaled_dot_product_attention`, CPU, varying L from 256→2048) shows
peak memory grows roughly LINEARLY with sequence length, not
quadratically (~1.75x memory when L quadruples, not ~16x) — CPU SDPA is
already using a memory-efficient/flash-style kernel here, so this ISN'T
a "swap in flash attention" fix. Most likely explanation:
`CausalSelfAttention._forward_chunked`'s own per-chunk bookkeeping
overhead — K=32/window=32 creates 32 small chunks (vs. K=4/window=256's
4 large chunks); more distinct reshape/permute/concat intermediate
tensors even though each individual chunk's own FLOPs are smaller. A
real, somewhat counter-intuitive finding: shrinking the window doesn't
just trade accuracy for speed, it can trade FLOPs for MEMORY via the
chunking implementation's own overhead — worth knowing before assuming
"finer window = cheaper" holds on every axis.

**`k32_narrow` RESULT: best val_bpb 2.4926 @ step 2800, mean it/s 2.259**
(the fastest of any `qcute_refine_v4` config trained this session,
consistent with its cheap FLOPs). Essentially tied with `bytelm_xs1_
ctx1024` (2.4870 — k32_narrow is 0.006 worse, within noise) and clearly
behind `bytelm_xs3_ctx1024` (2.4080). So the design (level 0 hyper-local,
level 1 dense/full-context, fusion effectively unbounded) trains fast and
lands in a reasonable place, but doesn't beat either matched baseline —
consistent with this session's broader finding that no `qcute_refine`
variant has yet beaten a properly compute-matched dense bytelm. The
real, distinct value of this run was methodological: it's the config
that exposed the FLOPs-vs-memory divergence above, independent of how
its own val_bpb landed. **Fusion-contribution probe** (new script,
`scripts/probe_v4_fusion_contribution.py` — see
[docs/kv_contribution.md](kv_contribution.md) §7 for the full writeup):
removing fusion entirely costs +2.42 bpb (catastrophic, since
`attn_window=32` exactly equals `K=32` here — level 0 has zero local
context beyond one block without it), but ≈88-92% of that recovery comes
from just having the `fuse_cross` module's own extra capacity/parameters
present — confirmed by TWO independent controls (content zeroed out,
`null_only`; content drowned in 10x-magnitude noise, `big_noise`, which
scores even worse than zeros — consistent with each other) — only
≈8-12% is attributable to the coarser level's actual content. Real
nuance on "direct forward-value conditioning is the strongest lever" — at least
for this narrow-window config, most of it was capacity, not information.

`bytelm_xs3_ctx1024` (2.625M) lands almost exactly on top of most
2-level `qcute_refine_v2` configs (2.6-2.7M) — a much closer param match
than the original 4-layer bytelm baseline (3.412M) ever was, at roughly
similar FLOPs too (5.4G vs. 3.7-3.9G — bytelm still costs more per
forward pass at matched params, consistent with its dense full-context
attention every layer vs. `qcute_refine`'s windowed/hierarchical
attention). `bpelm_16384_xs3` (6.561M) is far larger than everything
else here — its 16384-vocab embed+unembed tables dominate — so it's not
a fair params-matched comparison to anything in this table despite its
FLOPs/fwd being the lowest overall (context=256 tokens is a much shorter
sequence than the byte-level runs' context=1024). Fuller grid searches
(strict power-of-2 vocab/context, params-matched vs. FLOPs-matched picks
per `qcute_refine_v2` ablation target) live in `qcute/bytelm.py`'s and
`qcute/bpelm.py`'s own module docstrings ("Session notes" sections), not
duplicated here.

## `qcute/qcute_refine_v1.py` / `qcute/qcute_refine_v2.py` — new fork lineage (this session)

A second, independent fork lineage alongside `qcutelm_vlt*`/`qcutelm_pyramid*`
(self-contained-module convention, same as the rest of this project — full
per-version design rationale lives in each file's own module docstring, not
duplicated here; this entry is a pointer + one cross-cutting finding, not a
full recap). `qcute_refine_v1.py`: pure recursive NTP
tower with BSQ code hand-off between levels, plus a block-local joint-chain-
MTP detokenizer. `qcute_refine_v2.py`: detokenizer redesigned into a
`DecoderLevel` that cross-attends between adjacent levels' own
`EncoderLevel` hidden states (reused, not recomputed) instead of running a
block-local self-attention pass; grew a number of session-driven flags
(`byte_repr`, `code_head_mode`, `bit_head_class` with `BitPredictHeadAttn`/
`Conv`/`SSM` variants, `cross_attn_rope`, `decoder_own_trunk`,
`decoder_kv_pass_through`/`decoder_q_pass_through`, `layer_warmup_steps`)
plus a real MPS-specific bug fix (`nn.MultiheadAttention`'s backward
produced NaN gradients at `d_model=256`, resolved by switching to manual
`F.scaled_dot_product_attention` throughout, matching every other attention
op in the file). Configs live under `configs/qcute_refine_v2_*` and
`configs/qcute_refine_*`.

**Baseline step-budget finding, applies project-wide, not just to this
fork**: checked `bytelm_xs_mtp4_ctx1024`'s and `bpelm_32768`'s own full
8000-step val_bpb curves (excluding each log's earlier stale/restarted
segment). Neither one benefits from the full 8000-step budget —
`bytelm_xs_mtp4_ctx1024` bottoms out at **step 1700 (val_bpb 2.365)** then
overfits almost monotonically to 4.43 by step 8000, no flat region at all;
`bpelm_32768` bottoms out at **step 500 (val_bpb 2.134)**, overfits sharply
through ~step 3000-4000, then genuinely plateaus (noisy, no further trend)
through 8000. Since both baselines are already fully into their
overfit/plateaued state by step 4000, comparison runs gain no signal from
the second half of an 8000-step budget — **new `qcute_refine_v2` ablation
configs default to `steps=4000`** going forward. Also worth adopting
project-wide: report **best-checkpoint val_bpb** (`checkpoints/<run>/
best.pt`, already tracked by `Checkpointer`), not final-step val_bpb, as
the headline comparison number — final-step numbers on these
small-corpus runs mostly measure how overfit a run got, not how good its
best state was.

Documentation note: `CLAUDE.md`'s own Commands section previously pointed
its `bytelm` example at `configs/bytelm_xs_mtp4.py` (`context=256`) —
updated to `configs/bytelm_xs_mtp4_ctx1024.py`, the actual standard
baseline as of this session (`context=1024`, matching `qcute_refine`'s own
`context_len`). The old `context=256` config is kept (historical
reproducibility) but is no longer the comparison target for new work.

**Housekeeping**: every earlier qcute-lineage fork (`qcutelm.py`,
`qcutelm_vlt*.py`, `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py`) archived to `qcute/archive/` (configs to
`configs/archive/`, their own design docs — `continuous_tokenizer_
handover.md`, `fifo_v2.md`, `vlt12_math.tex` — to `docs/archive/`), 93 old
log directories cleared, 4 scripts' broken imports fixed
(`qcute.archive.*`). `bytelm.py`/`bpelm.py` are the explicit exception,
still active baselines. The `v1_*` ablation configs renamed to
`qcute_refine_*` for naming consistency with the rest of the lineage.
`qcute/qcute_refine.py` itself later renamed to `qcute/qcute_refine_v1.py`
(configs/docs updated to match) for naming consistency with
`qcute_refine_v2.py`.

## New closer-matched baselines + full ablation-family comparison table (this session)

**`configs/bytelm_xs3_ctx1024.py`** and **`configs/bpelm_16384_xs3.py`**
added as closer-param-matched baselines to the `qcute_refine_v2` 2-level
configs than the original 4-layer/8000-step `bytelm`/`bpelm` baselines
were (see the params/FLOPs table above). Three new fair-comparison bpelm
configs came out of the fuller grid search documented in `qcute/bpelm.py`'s
own docstring: `configs/bpelm_4096_paramsmatch.py` (params-matched to
`rope_3level_curriculum`, near-exact), `configs/bpelm_16384_ctx448_flopsmatch.py`
(FLOPs-matched to `decoder_trunk`, near-exact),
`configs/bpelm_8192_ctx448_flopsmatch_rope.py` (FLOPs-matched to
`rope`/`identity`) — queued to run after `bytelm_xs3_ctx1024`.

**Full `qcute_refine_v2` ablation-family comparison** (params/flops =
single forward pass batch=1, `FlopCounterMode`; best val_bpb/step and min
train bpb from each run's own `run.jsonl`; mean it/s = same recomputed
method as the baseline table above, `last_logged_step / elapsed_s`,
includes eval overhead):

| run | params | flops/fwd | best val_bpb | @ step | min train bpb | mean it/s |
|---|---|---|---|---|---|---|
| `qcute_refine_v1` (module, stopped early @1050/8000) | 1.249M | 2.916G | 4.2202 | 1000 | 3.9936 | 0.165 |
| `qcute_refine_rope` | 2.706M | 3.862G | 2.6310 | 3600 | 1.8923 | 0.656 |
| `qcute_refine_pass_through` | 2.642M | 3.695G | 2.5575 | 3300 | 1.9614 | 2.149 |
| `qcute_refine_decoder_trunk` | 4.424M | 5.878G | 2.5793 | 2800 | 1.8086 | 1.407 |
| `qcute_refine_v2_byte4_code256_identity` | 2.706M | 3.862G | 2.5868 | 3800 | 1.8957 | 2.545 |
| `qcute_refine_rope_3level_curriculum` | 3.414M | 4.330G | 2.6463 | 3300 | 1.9569 | 0.484 |
| `qcute_refine_tiny_byte_window` | 2.706M | ~3.86G (narrower window, not separately remeasured) | 2.6206 | 2600 | 1.7551 | 2.408 |
| `qcute_refine_no_rope` | 2.706M | 3.862G (identical arch to `rope`, flag-only diff) | **2.5645** | 3300 | 1.7582 | 0.996 |
| `qcute_refine_pq_table` (v2, `code_embed_mode="pq_table"`) | 2.772M | ~3.86G (not separately remeasured) | **2.4816** | 3300 | — | 2.017 |
| `qcute_refine_v3_rope` (v3, `fuse_encoder_levels=True`) | 3.563M | not separately remeasured | **2.4302** | 3500 | 1.4159 | 1.039 |

**Every `qcute_refine_v2` result so far loses to a trivial 1-layer bytelm
diagnostic**: `bytelm_xs1_ctx1024` (see baseline table above — one
self-attention+MLP block, 1.1M params, same `d_model=256`/`context=1024`
as every `qcute_refine_v2` config here) reaches best val_bpb **2.4870**,
beating even the best `qcute_refine_v2` result (`no_rope`, 2.5645) by
0.078 — a bigger margin than any ablation delta *within* the
`qcute_refine_v2` family itself (best-to-worst spread here is only 0.082,
2.5645 to 2.6463). At roughly half the params of the smallest
`qcute_refine_v2` config (1.1M vs. 2.6-2.7M) and none of the hierarchy,
BSQ quantization, or cross-attention machinery. Session hypothesis
(unconfirmed, under active investigation): the full architecture may not
yet be earning its own complexity over the cheapest possible baseline —
see [docs/kv_contribution.md](kv_contribution.md) for the KV-usefulness
probes this bears on, and `configs/qcute_refine_unconstrained_diagnostic.py`
(queued) for a diagnostic aimed directly at this: BSQ quantization
removed (`quant_type="identity"`), code width maximized (`dqs=(256,256)`,
no dimensionality reduction), and the encoder-side NTP losses zeroed out
(`code_ntp_weight=byte_ntp_weight=0.0`, the latter a new flag added this
session) so the DECODER's own cross-attention-based reconstruction loss
is the sole training signal — isolates what the architecture's own
ceiling looks like with every non-architectural constraint removed.

**Result: RAN — worse, not better, and a metric caveat worth recording.**
The training script's own `val_bpb`/`bpb` fields are computed from
`byte_loss` (level 0's own NTP head) unconditionally — with
`byte_ntp_weight=0.0` that head is never trained, so those fields read a
near-random-init ~8.2 bpb, a metric artifact, NOT the architecture's real
output quality. The real signal is `val_pair0_tok_loss` (the decoder path
that WAS trained, since `tok_weight=1.0`) — converting nats to bits gives
the actual result: best **2.884 bpb-equivalent @ step 3400** (44.5%
decoder token accuracy). Still worse than `bytelm_xs1_ctx1024` (2.4870)
and worse than every real `qcute_refine_v2` config (2.56-2.65 best-val
band) — removing BOTH the info bottleneck (dq=256, identity quant) AND
the encoder-side auxiliary NTP losses did NOT raise the ceiling, it
lowered it. Suggests those auxiliary losses were doing real, load-bearing
regularization/shaping work, not just diluting the decoder's own
gradient as the "unconstrained" framing hypothesized. Triggered the
conditional queue (see `configs/qcute_refine_pq_table.py`) as designed —
the trigger's threshold check used the (misleading) raw val_bpb field,
but the corrected 2.884 number still clears the same conclusion (worse
than bytelm_xs1), so the triggered run is still the right call, just for
the numerically correct reason. **Caveat for future use of
`byte_ntp_weight=0.0`**: don't trust `val_bpb`/checkpointer's
best-selection (which also keys off this same field) under this flag —
compare via `val_pair0_tok_loss` (or whatever loss term IS being trained)
instead.

**`code_embed_mode="pq_table"` RAN and WINS.** Clone of `qcute_refine_no_rope`
(2.5645, `code_embed_mode="linear"` implicitly), only the code-channel
mapping changed to a genuine 256-row lookup table (dq=8 → 2**8=256 exact
BSQ corners) instead of a single `nn.Linear(8, 256)`. Best val_bpb
**2.4816 @ step 3300** — beats `no_rope` by 0.083 (~3.2%), and now
essentially matches `bytelm_xs1_ctx1024`'s 2.4870 (within noise). The
"dq is starved" hypothesis (a linear map over an 8-dim ±1 vector can only
express 8 additive directions; an arbitrary function of a genuinely small
256-state space needs a table or nonlinearity) is the first lever tried
this session that closes essentially the ENTIRE gap to the 1-layer
bytelm diagnostic, on its own, without touching receptive field or
fusion. Strongest single result of the session so far.

**Rope-vs-no-rope, the genuine ablation (same arch, same 4000-step
budget, only `cross_attn_rope` differs)**: `no_rope` **beats** `rope` —
2.5645 vs. 2.6310 best val_bpb (no_rope wins by 0.067, ~2.5%), and
no_rope's min train bpb is also lower (1.7582 vs. 1.8923). Counter to the
intuition that giving cross-attention explicit relative-position
information should help — on this data/step budget it doesn't, and
mildly hurts. Doesn't overturn `rope`'s original design rationale
outright (one seed, one step budget), but it's now a clean, unconfounded
data point: `cross_attn_rope`'s default should not be assumed beneficial
without re-checking.

Notable it/s spread despite `rope`/`pass_through`/`identity` sharing near-
identical params (2.6-2.7M): 0.656 vs. 2.149 vs. 2.545 it/s — a ~3.9x
range. `identity` (no BSQ quantization) and `pass_through` (no encoder-
hidden-state reuse on either decoder side) are both cheaper per-step than
`rope`'s full BSQ+reuse path despite near-identical FLOPs/fwd and params,
suggesting real per-step overhead outside the counted forward-pass FLOPs
(quantization op, extra backward-graph complexity from hidden-state
reuse) — worth a closer look if throughput matters more than architecture
purity for future runs. `decoder_trunk` (private trunk copies, most
params/flops) and `rope_3level_curriculum` (3 levels, more sequential
work) are slowest, as expected from their own higher params/flops.
`qcute_refine_v1` is by far the slowest (0.165 it/s) — see
[docs/bitpredict_heads.md](bitpredict_heads.md) for why (chain-mode NTP
heads throughout, unlike v2's `code_head_mode="independent"` runs).

**CORRECTION (superseding this section's original text): `configs/
qcute_refine_v2_byte4_code256_simple.py` and its results (logs/
checkpoints) were DELETED from this table and from the repo.** The
original claim here — that `qcute_refine_rope` was "functionally a
duplicate" of `simple` because both used `cross_attn_rope=True` — was
wrong. It compared `simple`'s config text against `Config`'s CURRENT
default, but `simple`'s actual historical run happened BEFORE the
`cross_attn_rope` feature (and its default) existed in the codebase at
all — at the time it trained, cross-attention had no RoPE option, period,
so that run's cross-attention was genuinely position-blind (i.e. closer
to a `cross_attn_rope=False` run than a `True` one), not the confirmed
`True` run this table previously implied. Rather than keep a config/
result pair whose actual historical rope status is ambiguous, it was
deleted outright. The real rope-vs-no-rope ablation is now
`configs/qcute_refine_rope.py` (confirmed `cross_attn_rope=True`, in the
table above) vs. `configs/qcute_refine_no_rope.py` (cloned directly from
`rope.py`, only `cross_attn_rope=False` changed, same 4000-step budget)
— queued, not yet run as of this note.

Other reads from the table: `qcute_refine_v1` stopped at step 1050 of a
planned 8000 (superseded early when work moved to v2) — not a fair
endpoint comparison, shown for completeness only. Among the remaining
4000-step same-family runs (`rope`/`pass_through`/`decoder_trunk`/
`identity`/`rope_3level_curriculum`), results cluster tightly
(2.5575-2.6463) — no ablation here produced a dramatic swing;
`pass_through` (cheapest architecture, zero encoder-hidden-state reuse on
either side of the decoder) actually edges out the others slightly
despite being the most stripped-down. `decoder_trunk` is the most
expensive (4.424M params, 5.878G flops — private trunk copies aren't
free) without a proportionate quality win. See
[docs/kv_contribution.md](kv_contribution.md) for the deeper investigation
into why KV contribution varies so much across these configs.

## `qcute_refine_v4_3.py` — new fork, full weight sharing only, encode/decode renamed (2026-08-09 session)

New file `qcute/qcute_refine_v4_3.py`, cloned from v4.2 and stripped to
one fixed configuration only (no comments/docstrings, no per-ablation
flags): `quant_type="simplex"` only, `dq=8`/`code_bits=8` fixed (one
uniform 256-way pool shared by every level including byte level 0),
concat-only fusion (no `CrossBlock`/cross-attention module at all), no
`untie_levels`/`untie_fusion_pass`/layer-warmup curriculum — always full,
unconditional weight sharing across every level's trunk/embed/head.
`Config.Ks`/`d_model`/`n_layers`/`attn_window`/`context_len` and the
usual training/data/logging flags are all that remain.

**Terminology renamed**: v4.2's "PASS 1"/"PASS 2" become **encode**
(bottom-up, each level's own standalone NTP loss) and **decode** (a
second, conditioned forward pass). `RefineLM._encode` → `RefineLM._run`.

**New code-extraction mechanism.** v4.2 read a level's upward code
directly from `h_t` at the block's last position — the SAME hidden state
also used for that position's own NTP prediction, forcing "predict next
byte" and "summarize this block for the level above" to share one
representation. v4.3 replaces this with a `CodePool` module: a single
shared, learned query vector `[n_heads, head_dim]` cross-attends over the
block's own `K` local self-attention keys/values (reused directly from
the last `Block`'s own already-computed `k,v` — no new projection),
producing a fresh `h_code` decoupled from `h_t`. Fully batched across all
blocks in one `scaled_dot_product_attention` call via a reshape-blocks-
into-batch trick (`[B,H,T,hd]` → `[B*n_blocks,H,K,hd]`), no mask needed
(the reshape itself is the block boundary). One shared `CodePool`
instance, aliased across every level (full weight sharing).

**BUG, caught and fixed same session: decode was self-referential.**
First implementation had level *i*'s decode condition on `c_i` — its
OWN just-computed output, the exact same code level *i+1* consumes as
input. This has zero genuine dependency on anything hierarchically
above level *i*. Confirmed empirically: in the first "narrow" 2-level
configs trained under this design (`Ks=(32,16)` etc.), `level1_ntp_acc_
encode` collapsed to `1.0000`/loss≈`0.0000` almost immediately (by step
~300 of 4000) — level 1's own code degenerated to a single constant
value it could trivially "predict" — and correspondingly, `qual_*_
level0_uncond` and `qual_*_level0_cond` were near-byte-identical garbage
throughout training (both "the the the..."-style repetition), since
decode's conditioning signal carried no information once the code
collapsed. This happened well before any comparison of code COUNT
(sparse vs. dense) could matter — the collapse was upstream of that
variable entirely.

**Fix**: decode at level *i* now conditions on `c_{i+1}` — level *i+1*'s
own `CodePool` output (already computed every forward pass and
previously discarded, since nothing consumed it in a 2-level model) —
not `c_i`. This is described as needing a "level *i+2*" to exist as
`c_{i+1}`'s real consumer (by the file's own input/output naming, `c_j`
is level `j+1`'s own input), but no level *i+2* `LevelLM`/weights/NTP
loss are added — it's a "stub," reusing an already-shared, already-
computed value. Makes decode genuinely wait on level *i+1*'s encode pass
finishing (real dependency, not the previous accidental non-dependency).
Degenerate special case: `n_levels==1` has no level above to stub from,
so level 0 falls back to conditioning on its own `c_0` (the ORIGINAL
self-referential design) — the one place that mechanism is intentional
rather than a bug, since there is no alternative in a single-level model.
`decode_K` (the raw-byte span one decode-KV row represents, needed for
the jagged causal mask's block-resolution boundary) becomes `Ks[i]*
Ks[i+1]` in the stubbed case, or just `Ks[i]` in the self-conditioning
case.

**Rolling `kv_window` added to decode's jagged mask.** Previously decode
saw the ENTIRE history of resolved codes with no limit, unlike local
self-attention's own bounded, rolling receptive field (`_forward_chunked`'s
own current+previous-chunk pattern, `2×window` raw bytes). Fixed by
restricting `jagged_causal_mask_and_positions`'s visibility to the
`kv_window` most-recently-resolved blocks, sized as `ceil(2×window /
decode_K)` — mirrors self-attention's own `2×window` reach in code-block
units instead of an ever-growing window.

**Configs**: `configs/qcute_refine_v4_3_l1_k1.py` (`n_levels=1`,
`Ks=(1,)` — maximal-density self-conditioning, a code extracted at
EVERY byte position, "always use query every timestep to decode") and
`configs/qcute_refine_v4_3_l2_k1.py` (`n_levels=2`, `Ks=(1,1)` — same
maximal density, but decode's conditioning code is now the genuine
level-1 stub, not self-referential) — isolates whether the fixed
`attn_window=32` itself, independent of any code-compression granularity,
is the bottleneck on what decode can exploit. Both queued sequentially
(single-MPS-job convention); as of this note, `l1_k1` is running and
noticeably slow to produce its first logged step — plausibly because
`CodePool` at `K=1` reshapes into a `B*context_len = 16*1024 = 16384`
batch dimension for its `scaled_dot_product_attention` call, which MPS
may handle poorly; not yet confirmed as the actual cause. Results/
resolution not yet in as of this note.

**`CodePool` removed; code extraction reverted to reading straight from
`h`, then made pluggable (still 2026-08-09 session).** The `Ks=(1,)`
slowness above WAS `CodePool`'s own `B*n_blocks=16384`-batch attention
call — confirmed by removing it entirely (code extraction reverts to
v4.2's original `h_blocks[:,:,K-1,:]` readout) and re-measuring: MPS
forward+backward dropped from 70-80s/it to 0.7-1.0s/it, ~100x. On top of
that plain readout, four **interchangeable extraction modes** were added
(`Config.code_extract_mode`): `"last_h"` (the readout, now default, paired
with `code_head_tied=False` — a private, untied classifier, since reusing
NTP's own tied classifier on the exact same `h` risks the code degenerating
into a redundant copy of the NTP distribution), `"softmax_pool"` (no new
params — self-attends over the block using `h_{K-1}` as an implicit query;
provably degenerates to `"last_h"` at `K=1` since softmax over one item is
always weight 1, so disallowed there), `"light_query_attn"` (a genuine
learned query + its own `out_proj`, non-degenerate even at `K=1` since
that extra learned transform still runs), and `"query_embed"` (a learned
token spliced into the actual trunk sequence per block, densely masked —
"most expensive," not used for real training, kept for completeness).

**Same-position decode leak found and fixed (critical correctness bug,
present since this lineage's very first fusion/concat mechanism, not
just this session's code).** `jagged_causal_mask_and_positions`'s
`n_complete = (t+1)//K` made a code block visible to a query at the exact
position that PRODUCED it — but that code is extracted from the same `h`
NTP already uses to predict the NEXT token, so the query effectively got
to see a smeared copy of its own upcoming answer before answering it (see
the `docs/hierarchical_fusion_designs.md`-adjacent chat trace, not yet
written up separately: traced concretely through the string `"abcd"` —
`code[t]` structurally approximates `predicted byte[t+1]`, and the old
mask made `code[t]` visible starting at query `t` itself, not `t+1`).
Fixed by changing the formula to `n_complete = t//K` (block only visible
starting ONE position after it resolves) plus matching fixes in the
decode_K==1 fast path (shift `decode_kv` by one raw position before use)
and `generate_kv_cache`'s own incremental step timing (use the PREVIOUS
step's resolved code, never the current step's). Verified via direct
perturbation test (`decode_kv[10]` no longer affects position 10's own
output, only position 11 onward) and re-run `validate_generation`
equivalence (still exact `torch.equal` under the corrected boundary).
`generate_kv_cache` was also newly built this session (didn't exist
before) specifically for `code_extract_mode=="last_h"`/`decode_K==1` —
~2x faster than `generate_no_cache` at `qual_gen_bytes=64` scale on CPU,
validated bit-identical to it.

**Known remaining rough edges in the (pre-v4.4) concat mechanism**, not
yet fixed as of the v4.4 rewrite below: (1) the `decode_K==1` fast path's
RoPE tag is wrong — it tags the shifted decode slot with the QUERY's
position instead of the code's true content-origin position (the general,
non-fast path gets this right via `block_pos`); not a leak, just a
suboptimal-but-consistent convention the model has to learn around. (2)
Position 0's "nothing resolved yet" placeholder was a literal zero vector
fed as a real, attendable (softmax-competing) key — diluting attention
mass slightly versus the general path's cleaner full-exclusion at that
position; the cheap fix (special-case the one mask entry) was identified
but not applied before the design moved to v4.4 instead.

**Performance dead-end investigated and abandoned: `l2_k1`
(`Ks=(1,1)`) was still catastrophically slow (~83s/it) even after the
`CodePool` fix**, isolated via direct profiling to NOT be the core
per-step forward/backward (confirmed fast in isolation, ~1.0s/it,
matching `l1_k1`) — the actual cost is almost certainly
`qualitative_generate`'s own `generate_no_cache`, which recomputes the
WHOLE model from scratch at a NEW, ever-growing sequence length every
single generated byte; for `n_levels=2` specifically this means THREE
separately-shaped forward passes (level 0 encode, level 1 encode, level 0
decode) per generated byte instead of `l1_k1`'s one, and MPS appears to
pay a large one-time-per-novel-shape compilation cost (measured directly:
a fresh model's very first call took 120s, then steady-state dropped to
~1s/it) — with 128 total qual-gen steps per eval round (64 train + 64
val), most at DISTINCT lengths, this plausibly compounds into the
observed multi-minute-per-eval-round cost. Not fully root-caused or fixed
(the investigation was abandoned mid-profiling in favor of moving to
v4.4); the concrete, not-yet-applied fix would be swapping
`qualitative_generate` from `generate_no_cache` to the newly-built
`generate_kv_cache` (fixed-shape single-token updates, shouldn't trigger
the same per-shape recompilation).

## `qcute_refine_v4_4.py` — packed-sequence decode, replacing concat entirely (2026-08-09 session)

New file, cloned from v4.3. Removes the ENTIRE concat/KV-injection
mechanism (`CausalSelfAttention` no longer has any `decode_kv`-related
parameters at all — back to plain self-attention) in favor of **splicing
the code embedding directly into the input sequence at the embedding
level**, before `self.blocks` runs at all, then running the combined
(byte+code) sequence through completely ordinary self-attention. Two
interchangeable layouts (`Config.decode_pack_mode`): `"interleave"`
(`code,byte,code,byte,...`) and `"prepend"` (all resolved codes for a
region bunched before the bytes they condition) — both implemented for
direct A/B comparison, cost-per-mode analysis deferred to after some
training data exists.

**Causality is enforced by one shared, position-based rule** (not a
jagged block-mask anymore): for any query/key pair, `key_true_pos <=
query_true_pos AND NOT(key_is_code AND key_true_pos == query_true_pos)`
— ordinary inclusive causality, except a code is excluded at the EXACT
position that produced it (the same boundary the leak-fix above
established), regardless of where in the packed sequence it physically
sits. This one formula is layout-agnostic — verified: perturbing a single
byte only changes that position's own output and everything strictly
after it, identically for both `interleave` and `prepend` (accounting for
a `nonzero()`-on-2D-tensor display artifact in the very first check,
which initially looked like a leak into position 0 and wasn't one — the
"0" entries were the batch index, not a leaked position).

**RoPE is applied once, after packing**, to each token's own TRUE
timeline (byte tokens get their real raw position; code tokens get their
true content-origin position, i.e. one less than the byte they precede)
— fixes the RoPE-tag inconsistency flagged above for the old fast path,
by construction (there's only one packing step now, not a separate
projection-then-shift).

**Trainable BOS, not a zero vector.** Position 0 (nothing resolved yet)
gets a genuine learned `nn.Parameter` (`LevelLM.decode_bos`, shared/
aliased like everything else) instead of a hardcoded zero — chosen
specifically because it keeps the packed-sequence construction uniform
for incremental generation (every step concatenates one more token,
real code or BOS, no special-cased masking branch needed for the first
position).

**Implementation status: dense only, not yet windowed/chunked.** The
packed sequence is `2L` long (one code per byte at `decode_K==1`, the
only case implemented) and attention over it is currently computed as a
single `O((2L)^2)`-ish masked `scaled_dot_product_attention` call per
layer — correct (verified via the causality/gradient checks above) but
NOT the efficient windowed/chunked form the rest of this file uses
elsewhere; at `context_len=1024` this is expected to be impractically
slow for real training. A chunked version (mirroring `_forward_chunked`'s
own `kc_prev`/`kc` trick, but on `2×window`-sized combined chunks) is
real, identified, deferred work — not started, given the added
complexity of getting chunk-local masking right for `"prepend"`
specifically (sequence order and true-causal order diverge within a
prepend-packed chunk, unlike `"interleave"` where they coincide "for
free"). `generate_kv_cache`/`validate_generation`/the old `_step_block`
incremental-generation machinery were all REMOVED (not ported) for this
version, since they were built entirely around the mechanism this file
just deleted — `generate_no_cache` (recompute-from-scratch reference)
still works unchanged, since it only calls `RefineLM._run`/
`LevelLM.forward`, agnostic to how decode is implemented inside.

**Chunked/windowed decode, `"interleave"` only.**
`LevelLM._packed_decode_forward_chunked` (`Config.decode_chunked=True`,
gated to `decode_pack_mode=="interleave"` in `LevelLM.forward`'s
dispatch — `"prepend"` still falls back to the dense path, for the reason
already noted above: prepend's sequence order and true-causal order
diverge, so the chunk-contiguous trick below doesn't apply to it
directly without extra work, not attempted this round). Key fact that
makes interleave chunkable at all: its packed sequence
(`code_0,byte_0,code_1,byte_1,...`) has **true_pos non-decreasing in
sequence order** (each code's true_pos is one less than the byte it
precedes, so the sequence reads `...,-1,0,0,1,1,2,2,...`), so it can be
chunked contiguously exactly like ordinary windowed self-attention,
reusing the "previous chunk + current chunk" trick already used
elsewhere in this file — just with two adjustments: (1) chunk size is
`sc = 2*W` slots (one byte's window-worth `W` covers `2*W` combined
slots, since each byte position occupies 2 slots); (2) the per-key
causal/window/same-position-exclusion test can't use the older
offset-based `_causal_window_mask` (which assumes uniform 1-unit-per-slot
spacing) since interleave's slot-to-true_pos spacing isn't uniform (some
adjacent slots share a true_pos) — so masking is computed from the
actual gathered `true_pos`/`is_code` values per chunk instead (still
cheap: one small `[chunk_size, key_context_size]` mask per chunk, not
global).

The needed reach is `R = 2*W` (`decode`'s window is double the byte-level
one, per the original spec). Using `n_prev_chunks` previous `sc`-sized
chunks of extra key context (plus the current chunk) as a safety margin,
`n_prev_chunks=1` was empirically **insufficient** (`max_diff=0.26`
against dense, i.e. a real correctness gap, not a rounding difference) —
`n_prev_chunks=2` matches dense **exactly** (`max_diff=0.0` at
`d_model=256, context_len=256`, `~7e-7`, i.e. float rounding, at smaller
scale) and was hardcoded as the default margin. Verified via
`scripts/test_v4_4_chunked_decode.py`: exact match at `Ks=(1,)` and
`Ks=(1,1)`, multiple `d_model`/`context_len`/`window` combinations,
including the `n_chunks==1` degenerate edge case
(`context_len==attn_window`, previous-chunk padding all zero/masked-out).

**Benchmark (MPS, full train step: forward+backward+`opt.step()`,
`d_model=256, n_layers=2, attn_window=32`):**

| context_len | chunked | dense |
|---|---|---|
| 256 | 73.8ms | 60.1ms |
| 512 | 129.5ms | 133.6ms |
| 1024 | 249.2ms | skipped (extrapolated well over 500ms, quadratic) |

Chunked is *not* faster at small `context_len` (gather/reshape/padding
overhead dominates when the dense call is already cheap) but crosses
over by `context_len=512` and its linear-in-`L` scaling wins decisively
as `context_len` grows — exactly the tradeoff expected given dense is
`O((2L)^2)`-ish and chunked is `O(L)`. `context_len=1024` (the
production target) was not measured for dense (extrapolated
impractical); chunked measured directly at `249.2ms/iter`, practical for
real training.

**First v4.4 training runs launched**: `configs/qcute_refine_v4_4_l1_k1.py`
(`Ks=(1,)`, degenerate self-conditioning) and
`configs/qcute_refine_v4_4_l2_k1.py` (`Ks=(1,1)`, genuine level-1-stubbed
conditioning) — both `decode_pack_mode="interleave"`,
`decode_chunked=True`, `context_len=512` (not yet the production 1024,
kept lower for this first real run; only constraint is being a multiple
of `attn_window=32`, per windowed attention's own requirement). Queued
sequentially (`l1_k1` first, `l2_k1` auto-starts after via a wrapper
script polling `l1_k1`'s PID), consistent with the project's one-MPS-job-
at-a-time rule. `"prepend"` mode and `context_len=1024` remain
unbenchmarked in a real training run as of this note.

**Two real bugs found once actual training runs were attempted (neither
is chunked-decode-specific — both present in the dense path too, and the
init-scale one is present in `qcute_refine_v4_3.py` as well, confirmed by
re-reading `logs/qcute_refine_v4_3_{l1_k1,l2_k1}/run.log`'s own first
log lines):**

1. **`nn.Embedding` default init (`std=1.0`) is far too large for a
   tied classifier head** — `LevelLM.embed` was never given an explicit
   init (unlike `qcute/bytelm.py`'s own `_init_weights`, which sets
   `std=0.02` for every `nn.Linear`/`nn.Embedding`). With `std=1.0` and
   `d_model=256`, `logits = h @ embed.weight.T` has variance ~`d_model`
   (std ~16), producing wildly peaked/near-random logits at init and
   catastrophic initial cross-entropy: **measured bpb at step 1 was
   ~230-245** (both `qcute_refine_v4_4_l1_k1` and its v4.3 predecessor)
   instead of the expected `log2(256)=8.0` uniform-random floor.
   Confirmed by isolated test (`model(x)` on a fresh model, `embed.weight`
   std ≈0.999 by default) and by reproducing exactly with `torch.nn.init.
   normal_(embed.weight, std=0.02)` applied post-hoc — bpb dropped to
   `8.07`, matching the analytic floor. **Fixed in `qcute_refine_v4_4.py`**:
   `LevelLM.__init__` now calls `nn.init.normal_(self.embed.weight,
   std=0.02)` and the same for `code_head.weight` when untied. `qcute_
   refine_v4_3.py` (and earlier) were NOT patched — left as accurate
   historical runs, consistent with the project's "target a specific
   vN file for historical work" convention; this is a lineage-wide latent
   bug, not new to v4.4, and anyone re-running v4.3 configs should expect
   the same over-inflated initial bpb.

2. **`_packed_decode_forward_chunked` crashed during
   `qualitative_generate`** — `generate_no_cache` grows the sequence one
   byte at a time (`T=65,66,...`), which isn't a multiple of
   `attn_window`; the plain `CausalSelfAttention.forward` already handles
   this gracefully (falls back to dense with a printed warning) but the
   new chunked-decode path had a hard `assert L % W == 0`, killing both
   training runs at their first eval (`step=100`). **Fixed**: `LevelLM.
   forward`'s dispatch now checks `L % self.window == 0` before choosing
   the chunked path, falling back to `_packed_decode_forward` (dense)
   otherwise — mirrors the existing base-attention fallback pattern
   exactly.

Both `qcute_refine_v4_4_l1_k1` and `qcute_refine_v4_4_l2_k1` were
relaunched (same queued-sequential setup) after these fixes; `l1_k1`
confirmed at `step=49: bpb=6.18` (below the 8.0 floor and dropping),
correcting the pre-fix run's `step=49: bpb=231.5`.

**`qualitative_generate` now also prints an annotated `level0_cond_codes`
line** — the conditioned generation with each byte's own in-band code
shown inline as `char{code_id}`, plus the full generated code sequence
trailing in `<...>` (re-derives the codes via a new helper,
`_decode_source_codes`, which replays `RefineLM._run`'s own
`source_c`-selection logic — `c_list[1]` if `n_levels>1` else
`c_list[0]` — against the final generated byte sequence). Only affects
freshly-launched processes (Python doesn't hot-reload a running training
job), so the in-progress `l1_k1` run predates this and won't show it;
`l2_k1` and later runs will.

**`_packed_decode_forward` generalized to arbitrary `decode_K`
(previously hardcoded to `decode_K==1`).** Motivating case: `Ks=(4,1)`
(one code per 4 raw bytes, a crude fixed-width analogue of BPE's
~4-bytes/token average) gives `decode_K = Ks[0]*Ks[1] = 4`, which the
original implementation flatly asserted against. Generalized via
**block-interleave**: instead of one prefix token per byte, there's one
prefix token (BOS or a code) per `K`-byte block — prefix `b` is
`decode_bos` for `b==0`, else `code_kv[b-1]` (the PRECEDING block's own
code; the last block's own code is still never consumed, same
end-effect as the `K==1` case). Each prefix's `true_pos` is set to
`b*K - 1` (the last raw byte position of the block it summarizes) —
combined with the *existing*, unchanged same-position-exclusion rule,
this reproduces "a block's code becomes visible only strictly after its
own last covered byte" at any `K`, and provably reduces to the original
`K==1` formulas exactly (`code_true_pos = byte_pos - 1` was the `K=1`
special case of `b*K-1` with the block/byte index identification
`b=t-1`). Verified: forward+backward runs cleanly for both
`decode_pack_mode`s at production scale (`d_model=256, context_len=1024,
Ks=(4,1)`), init bpb ≈8.06-8.08 (correctly reflects the embed-init fix
above), and a perturbation/causality test at `Ks=(4,1)` shows byte `t`'s
perturbation affects exactly positions `>= t` (itself included, via
ordinary self-attention — not a leak) and nothing before it.

**The chunked decode path (`_packed_decode_forward_chunked`) was NOT
generalized** — it remains `decode_K==1`-only by design (`LevelLM.
forward`'s dispatch now checks `decode_K == 1` explicitly before
considering the chunked path, falling back to the newly-general dense
path otherwise). Not just laziness: at `decode_K=4`, block-interleave's
packed length is only `L + L/K = L*1.25` (one prefix per 4 bytes, not
per byte) — much closer to `L` than the `decode_K==1` case's `2L`, so
dense attention is proportionally cheaper here even before chunking;
confirmed practical directly (`context_len=1024` forward+backward ran
without issue, no need to extrapolate).

**New config**: `configs/qcute_refine_v4_4_bpelike_k4_1.py` (`Ks=(4,1)`,
`attn_window=(8,256)`, `decode_chunked=False` since decode_K=4).
Queued third in the sequential chain (`l1_k1` → `l2_k1` → this), still
respecting the one-MPS-job-at-a-time rule. Caveat carried into the
config's own docstring: this is a fixed-width 4-byte grouping, NOT
learned BPE-style merges — a matched code-to-byte ratio for a rough
comparison against `qcute.bpelm`, not a claim of equivalent
segmentation. (Later superseded/reordered — see below.)

**Split encode/decode window sizes.** `Config.attn_window`'s per-level
entries now accept either a scalar (applied to both encode and decode,
the original behavior) or an explicit `(encode_window, decode_window)`
2-tuple, parsed in `RefineLM.__init__` into two separate lists
(`self.windows`, `self.decode_windows`), each passed into its own
`LevelLM.window`/`LevelLM.decode_window` attribute. `_packed_decode_
forward`/`_packed_decode_forward_chunked` and the chunked-path dispatch
in `LevelLM.forward` all switched from reading `self.window` to reading
`self.decode_window`. Motivated by wanting a byte-level encode pass with
a narrow window (cheap, local) alongside a decode pass with a much wider
window (needs more reach to usefully condition on coarser codes).
**Caveat worth remembering**: the decode window is measured in raw-byte
`true_pos` units, same as everything else in the packed-decode masking
(`windowed = (ti-tj) < 2*W`) — NOT in code-count units. A decode window
of `256` gives a reach of `2*256=512` raw bytes, not "256 codes" /
"full 1024-byte context" — covering the full context at `context_len=
1024` requires `decode_window=512`, not `256`.

**Terminology: `stream_i` / `code_i` replaces "in-band code."** Level
`i`'s own input is `stream_i` (raw bytes for `i=0`; `code_{i-1}` for
`i>0`), and its own output — the thing it hands up to the level above,
or feeds back to itself in the `n_levels==1` degenerate case — is
`code_i`. The recursive relation is just `stream_{i+1} = code_i`. Chosen
over "in-band" (this session's earlier term, still findable in a couple
of older docstrings/comments not yet swept) because "in-band" only
described the shared-vocabulary property, not the actual recursive
structure the user was asking to name clearly.

**Found: decode's loss does NOT backprop into the code producer — `code_
ids = source_c.argmax(-1)` is non-differentiable** (confirmed directly:
`code_ids.requires_grad` is `False` even though `source_c.requires_grad`
is `True`). This applies identically to both the `n_levels==1` self-
conditioning case (level 0 conditioning on its own `code_0`) and the
`n_levels==2` cross-level case (level 0 conditioning on level 1's `code_
1`) — in both cases, whatever produced the code_i being consumed gets
*zero* gradient signal from decode's loss, only from that level's own
independent NTP loss (`encode_losses[i]`, `code_ntp_weight`-weighted).
Practical consequence: decode's window size, decode's loss weight, and
even whether decode is enabled at all, provably cannot affect the code-
producing level's own training signal — they're fully decoupled via
gradient (though of course NOT decoupled through data: decode still
reads whatever codes that level happens to produce, good or bad).

**Fixed** (by explicit user choice, after being presented as an option):
`RefineLM._run`'s decode loop now uses the *same* straight-through
mechanism the encode chain already uses internally (`x = seq_repr @
self.embed.weight`) instead of the hard lookup: `code_embeds = source_c
@ self.encoders[i].embed.weight` (gated by new `Config.decode_code_ste`,
default `True`; `False` reproduces the original detached behavior via
`source_c.detach() @ embed.weight`). Forward VALUE is provably identical
either way — `gumbel_quantize`'s straight-through estimator already
makes `source_c`'s forward value the exact one-hot `hard` tensor, so
`source_c @ embed.weight` equals `embed.weight[argmax(source_c)]`
numerically; only the backward path changes. Verified: (1) `code_head.
weight.grad` is now non-`None` and nonzero from `decode_loss.backward()`
alone, confirming the new gradient path actually carries signal; (2) the
`decode_code_ste=False` flag reproduces the original fully-detached
behavior exactly (`grad is None` when isolated to `decode_loss` alone);
(3) chunked-vs-dense equivalence and the causality/perturbation test
were both re-run post-change across `Ks=(1,)`, `Ks=(1,1)`, `Ks=(4,1)`
(`"prepend"`), `Ks=(4,)` (`"interleave"`) — all still exact/clean, since
this change only touches *how* `code_embeds` is computed in `_run`, not
`_packed_decode_forward*`'s own masking/packing logic. **Ablation
deferred** (explicit user instruction: "ablate later") — `decode_code_
ste` exists as a switch but no run has been queued to compare the two
settings yet.

**Confirmed (not new, just re-verified on request): `"prepend"` already
produces the expected bunched-prefix layout.** For byte sequence `abcd`
with self-codes `x,y,z,w` (`w` unused, same as always), `"prepend"`
gives combined slots `[BOS, x, y, z, a, b, c, d]` with `true_pos =
[-1,0,1,2, 0,1,2,3]` — directly inspected via a small script, matches
the general block-interleave-derived formula exactly, no code changes
needed (it already generalized correctly when `_packed_decode_forward`
was generalized to arbitrary `decode_K` earlier this session).

**Third bug found once `bpelike_1level_k4` actually ran: `_packed_
decode_forward`'s `assert L % K == 0` crashed during `qualitative_
generate`** — same root cause as the earlier chunked-path crash
(`generate_no_cache` grows the sequence one byte at a time, so `L`
generally isn't a multiple of `decode_K`), just newly surfaced in the
dense path once `decode_K` became `4` instead of always `1` (`decode_K
==1` divides any `L`, so this never showed up before). **Fixed**: `Refine
LM._run`'s decode loop now checks `x_list[i].shape[1] % decode_K != 0`
and skips decode entirely for that call (falls back to the encode-only
`h_i` already in `h_out[i]`) rather than asserting — same graceful-
degradation pattern as the base-attention and chunked-decode dense
fallbacks elsewhere in this file. This only triggers during generation;
training always uses `context_len`, guaranteed divisible by every level's
`Ks[i]` per `RefineLM.__init__`'s own asserts. Also fixed a related
display bug this surfaced: `_decode_source_codes`/`_annotate_bytes_with_
codes` (the `level0_cond_codes` qualitative-gen line) assumed one code
per byte (`decode_K==1`) when slicing/indexing — silently produced an
empty/wrong annotation at `decode_K>1` instead of crashing. Generalized
both to group `decode_K` bytes per code tag (verified across `Ks=(1,)`,
`Ks=(4,)`, `Ks=(1,1)`, `Ks=(4,1)`).

Both fixes required `bpelike_1level_k4` to be re-queued (as `qcute_
refine_v4_4_bpelike_1level_k4_retry`) — it had already run and crashed
at its first eval before the fix landed. Since `bpelike_k4_1` (still
pending) shares the same `decode_K=4`, it would have hit the identical
crash; the fix covers it automatically since it re-imports the file
fresh at launch. Current queue: `l2_k1` (running) → `bpelike_1level_k4_
retry` → `bpelike_k4_1`.

## Cumulative multi-level decode conditioning

Generalized decode from "level `i` conditions on level `i+1`'s code
only" to **cumulative**: level `i` conditions on its own code (`code_i`,
self) PLUS every coarser level above it (`code_{i+1}`, `code_{i+2}`,
..., up to the top), each as its own independently-windowed track. The
top level now also gets a decode pass (self-conditioning only, same
degenerate case the old `n_levels==1` self-conditioning always was, now
just the general "nothing coarser exists" case for whichever level
happens to be on top).

**Terminology used throughout this section**: `stream_i` (level `i`'s
own input — raw bytes for `i=0`, `code_{i-1}` for `i>0`) and `code_i`
(level `i`'s own output), per the `stream_i`/`code_i` convention adopted
earlier this session in place of "in-band code."

**Config**: `attn_window`'s per-level `decode_window` (the second slot
of the `(encode_window, decode_window)` 2-tuple) is now itself either a
scalar (broadcast to ALL of that level's decode sources) or an explicit
tuple of length `n_levels - i`, ordered `[self, +1, +2, ..., top]`. Two
distinct per-source sentinel values, not to be confused: `-1` means
unbounded/full-context (the track stays ACTIVE, just with no window
limit) and `0` means the track is EXCLUDED from decode entirely (not
computed at all). Divisibility assertions apply to both encode and every
active decode-source window independently.

**`_packed_decode_forward(x0, tracks)`** replaces the old single-source
`(x0, code_kv, decode_K)` signature — `tracks` is a list of `(code_kv,
K, window)` triples, one per active decode source. Packing: multiple
tracks only support `decode_pack_mode="prepend"` (all prefix streams
concatenated coarsest-to-finest, self track immediately before the
bytes — order doesn't affect correctness, causality is governed by
`true_pos` values not physical position, but this is the more readable
convention); single-track (`len(tracks)==1`) still supports
`"interleave"` too, needed by the chunked decode path. The `windowed`
mask term is now per-KEY (`window_of_key`, gathered per-track) rather
than a single scalar `W`, since different tracks can have different
windows within the same forward call.

**`_packed_decode_forward_chunked`** unchanged in mechanism, just takes
`window` as an explicit parameter now (was reading `self.decode_window`,
which no longer exists as a single value) — still single-track,
`decode_K==1`-only, gated in `LevelLM.forward`'s dispatch.

**Verification**: multi-track layout directly inspected for a 2-track
case (`Ks=(4,1)`, `L=8`) — prefix `true_pos` came out exactly
`[-1,3,-1,3]` (coarsest track's 2 block-prefixes, then self track's 2
block-prefixes, matching the documented coarsest-to-finest convention).
Causality/perturbation-tested across `Ks=(1,)`, `Ks=(1,1)`, `Ks=(4,1)`,
and a 3-level `Ks=(2,2,1)` case — level 0's own `h` always showed the
perturbed position as the exact earliest affected one, nothing before
it. Level 1's `h` showed the affected range starting one position later
than the theoretical earliest point — **not a bug**: level 1's input is
level 0's *discretized* (`gumbel_quantize`/argmax) code, and a
perturbation that doesn't flip the discrete class at the theoretically
susceptible position produces zero downstream effect there, a normal
property of a quantization bottleneck (crucially, nothing *before* the
theoretical earliest point was ever affected, in any run). Gradient
flow verified into BOTH the self track's classifier and the +1 track's
classifier from a single `decode_loss.backward()` call (both nonzero,
and — expected, given full weight sharing — numerically identical,
since `encoders[1].code_head` is literally the same aliased parameter
tensor as `encoders[0]`'s own, not a coincidence).

**`generate_no_cache`'s ragged-length handling generalized** to the
multi-track case: `RefineLM._run` now checks divisibility per-track
(`L_i % cum_K`) while building each level's track list, skipping
individual disabled/misaligned tracks rather than an all-or-nothing
single check.

### Blockwise parallel decoding: built, tested, found NOT valid for this architecture

Followed through on "queue it, because can catch bugs" — built
`generate_blockwise_parallel` (drafts level1's codes autoregressively
via its NTP head, batches all new K0-byte blocks along the batch
dimension, aiming for `O(gen_len/K0 + K0)` instead of `generate_no_
cache`'s `O(gen_len)`) and tested it for EXACT equivalence against
`generate_no_cache`. It caught two real problems, the second of which
is structural, not a fixable bug:

1. **Off-by-one, fixed**: predicting byte `t+1` uses `h[t]`, and `h[t]`'s
   decode prefix is *its own* block's code (`code_kv[block(t)-1]`) — so
   the very FIRST new byte after the prompt is actually predicted using
   the SECOND-to-last prompt block's code, not the last one (which only
   starts conditioning the byte after that). The original implementation
   uniformly assigned each new K0-byte group to `code_kv[block_index-1]`
   assuming clean block alignment; the true alignment is shifted by
   exactly one byte relative to that. Fixed by special-casing the first
   new byte as an ordinary single step, then generating the group-batched
   remainder shifted by one position, then trimming the one extra byte
   the shifted grouping overproduces at the end.

2. **Structural, NOT fixed — the whole premise doesn't hold**: even
   with (1) fixed, outputs still didn't match. Root cause, confirmed
   directly: `c_list[1][p]` depends on `c_list[0][p]` at the SAME
   position (causal self-attention always includes the current position
   as a key for itself) — meaning a new block's conditioning code
   (`code_1` at that block's index) literally **cannot exist** before
   that same block's own bytes are already generated. `RefineLM._run`'s
   decode conditions on `c_list[i+1]`, level `i+1`'s DERIVED self-code —
   a **reconstruction** of the span it summarizes ("here's a compressed
   recap of the K bytes you just finished, use it as extra context for
   the next K"), not an independently-forecastable quantity. This is a
   genuinely different signal from what `generate_level1_codes`
   produces (a next-token PREDICTION via level1's own NTP head, which
   *is* independently forecastable, ahead of the underlying bytes
   existing) — `generate_blockwise_parallel` drafted the latter but
   `_run` actually conditions on the former, so they structurally
   cannot agree. Confirmed the user's own framing of this directly:
   self/adjacent-level conditioning in this architecture is
   reconstruction-shaped, not draft-shaped — exactly why it's
   sequential across blocks, matching the direct test
   (`c_list[1][p]` genuinely responds to a perturbation at `c_list[0][p]`
   itself, not just earlier positions).

**Conclusion**: `generate_blockwise_parallel` is kept in the file
(marked "NOT CURRENTLY VALID" in its own docstring) as a reference for
what was tried and why it doesn't work, not as usable generation code.
Genuine block-parallel decoding would need decode to condition on a
DRAFTED signal (e.g. level `i+1`'s own NTP-predicted continuation of
`stream_i`) instead of its derived self-code — a real architecture
change, not a generation-function fix. Not attempted this session.

### The fix: two-stage latent-variable model (`.detach()`, not STE) — resolves the dead end above, not yet implemented

Follow-up design discussion (analysis only, no code changes yet) that
directly resolves the "structural, NOT fixed" blocker above.

**The pattern**: train this as the standard two-stage discrete latent-
variable setup (VQ-VAE-2 / DALL-E / Jukebox's own decoder-then-prior
training, not novel to this session) — a decoder trained on TRUE codes
with `.detach()` (no gradient into the code producer), and a SEPARATE
autoregressive prior over that same code space, trained purely on its
own NTP loss, zero gradient coupling to the decoder.

**Why `decode_code_ste=False` (detach), not `=True` (STE, this
session's DEFAULT), is the right choice for this purpose**: STE lets
decode's loss shape the code producer toward "more decode-useful"
codes — good for decode's OWN quality in isolation, but it actively
fights having an INDEPENDENT predictor track that code distribution
well, since STE continuously deforms the distribution via a signal the
predictor never sees. `.detach()` leaves the code producer shaped by
nothing but its own objective (self next-code accuracy), which is
exactly what an independent prior CAN learn to match closely.

**Why it must be SELF-conditioning (`code_i` on its own past, the
`self` track), not the cross-level `+1` track built earlier this
session**: `code_{i+1}` (the coarser self-code, via `_classify`/
`code_head`) is a DIFFERENT projection from level `i+1`'s own NTP head
(`embed.weight`) whenever `code_head_tied=False` — no clean "predict
this" target exists for it. But `code_i` (level `i`'s OWN code) and
level `i+1`'s NTP prediction of "the next `code_i` token" ARE the same
object, same embedding space — `generate_level1_codes` already computes
exactly this prediction, completely unmodified.

**The mechanism**: train decode SELF-conditioned on `code_i`
(`decode_code_ste=False`); separately, level `i+1` (or, for `n_levels==
1` configs with no level above to draft with, a small dedicated
auxiliary LM — this is precisely the `aux_code_lm` idea parked earlier
this session, now with a clear purpose) learns to predict that SAME
`code_i` stream via its own NTP loss, untouched by decode's gradient.
At GENERATION time, substitute the DRAFTED `code_i` predictions for the
true reconstructed ones in decode's self-conditioning slot — since
decode was never trained to expect anything about WHERE the code came
from (no gradient ever shaped it around decode's specific needs), this
substitution is a small, well-behaved distribution shift, not the
structural mismatch the cross-level/reconstruction path had (that path
required `code_{i+1}[p]` to depend on `code_i[p]` at the exact SAME
position, an unavoidable circular dependency; self-conditioning +
independent-drafter has no such cycle, since the drafter never needs
`code_i[p]` itself to predict it).

**Status: design written up in full as `docs/two_stage_latent_decode_math.md`
(math spec, portable to code without more context), and a first test
config queued.**

**`AuxCodeLM` built, then reverted.** First implementation attempt built
a bespoke, non-shared-weight drafter module (`AuxCodeLM`, own
`embed`/`blocks`, for `n_levels==1` configs with no natural level above
to reuse). User feedback: this solves the wrong problem — the
`n_levels==1` self-conditioning base case is already correct and needs
no special handling; "no drafter exists for the top level" is just "no
`Ks[-1]=1` entry appended," already fully covered by the EXISTING
shared-weight `LevelLM`/`generate_level1_codes` machinery, no new class
needed. `AuxCodeLM` was actually addressing a DIFFERENT, unconfirmed
concern (does the drafter's SHARED weights get pulled off-target by the
main tower's other objectives, even with `decode_code_ste=False`
severing the direct gradient path through the code tensor?) — a real
question, but one to measure before building infrastructure for it.
Reverted (`Config.aux_code_lm`/`aux_code_lm_layers`, the `AuxCodeLM`
class, and `RefineLM.forward`'s use of it all removed) in favor of the
simpler default: reuse `Ks=(K0, 1)` directly.

**New config**: `configs/qcute_refine_v4_4_selfcond_detach_k4.py`
(`Ks=(4,1)`, `decode_code_ste=False`) — level 0 decodes SELF-conditioned
only (its cross-level `+1` track to level1's code is explicitly disabled,
window value `0`); level1's own decode/self-conditioning is also disabled
(irrelevant to this experiment); level1 exists purely as an independent
NTP model over the `code_0` stream, i.e. exactly the drafter role,
reusing `generate_level1_codes` unmodified. Verified directly before
queuing: `enc0.code_head.weight.grad is None` when isolated to
`decode_losses[0].backward()` alone (confirms the detach), and
`generate_level1_codes` still runs unmodified against this config.
Queued behind `bpelike_k4_1` (`caffeinate -i -w <wrapper pid>` covering
both jobs, so the machine won't sleep through either).

**What this run will tell us**: once trained, compare
`val_level1_ntp_acc_encode` (level1's own accuracy at predicting the
NEXT `code_0` token) against `val_level0_ntp_acc_decode`/`val_bpb`
(decode's own quality using the TRUE `code_0`) — if level1 predicts
`code_0` well, its drafted continuation is a credible substitute for
decode's true self-conditioning signal at generation time (§7 of the
math doc). The actual drafted-substitution GENERATION function itself
is still **not implemented** — this config only tests whether training
produces drafts good enough to make building it worthwhile.

**Machine reboot mid-`bpelike_k4_1` run, silent kill, no resume
capability.** Around 8:28am the machine rebooted outright (confirmed via
`uptime` showing only ~6h40m elapsed, not the days-long uptime expected)
— NOT a training-time crash. This silently killed the training process,
its queue wrapper script, and the `caffeinate -i -w <wrapper pid>`
watching it, all simultaneously; `bpelike_k4_1`'s raw stdout log
(`/tmp/v4_4_bpelike_k4_1.log`) was gone entirely post-reboot.
`qcute_refine_v4_4.py`'s `main()` has no checkpoint-resume path — it
always starts training from step 0 — so the run had to be relaunched
from scratch (it had reached step 1699/4000 before the reboot; that
progress is lost, not resumed). Relaunched
(`--run_name qcute_refine_v4_4_bpelike_k4_1`, wrapper PID 4685→python
PID 4688), queue wrapper rebuilt (`run_v4_4_selfcond_after_bpelike_k4_1_v2.sh`,
polls `kill -0 4688`), `caffeinate -i -w <new wrapper pid>` relaunched
watching it. Confirmed via fresh `[00:00:00]` entries appended to
`logs/qcute_refine_v4_4_bpelike_k4_1/run.log` (that file is append-mode,
so it now contains both the killed run's history up to step 1699 AND the
new run's entries concatenated — read with that in mind). `caffeinate`
prevents idle sleep, not a full reboot; there is no mitigation in place
against this happening again besides noticing it promptly.

**Post-restart throughput degraded sharply, then root-caused and fixed**:
by step 349/4000 (1h28m elapsed) the observed rate had degraded to
~17.83s/it average, with individual steps as bad as 129.83s/it and
climbing over time (not a stable slow rate — actively worsening),
against a nominal ~30min budget. `vm_stat` showed only ~64MB free RAM at
the time (`Pages free: 4111` × 16KB), consistent with the ~60MB-free
pressure flagged earlier this session, though `ps aux -m` found no
single runaway process (just cumulative load from an active VS Code +
Chrome session) — the memory pressure looks incidental, not the root
cause.

**Root cause: `context_len=1024` decode passes are dense O((2L)²), not
windowed.** Level 0's own `attn_window=8` encode pass genuinely takes
the efficient chunked path (`T=1024 % window=8==0` and `T>window`
satisfied). But: level 1's encode window (256) exactly equals its own
sequence length (`context_len/Ks[0]=256`), so `T>window` is false and it
silently falls back to dense (expected/flagged by the code's own
warning, just easy to overlook the cost). More importantly, **both
levels' decode passes ran fully dense** — `decode_chunked=False` was
forced off in this config specifically because the chunked decode path
(`_packed_decode_forward_chunked`) is only implemented/verified for
`decode_K==1`, and this run has `decode_K=Ks[0]*Ks[1]=4`. Dense decode
attention over a packed multi-track sequence at `context_len=1024` is
the dominant cost, and it scales as `O((2·context_len)²)`.

**Fix applied**: dropped `context_len` 1024→256 in both
`bpelike_k4_1` and `selfcond_detach_k4` (attn_window's dense-fallback
level scaled down proportionally, 256→64, keeping the same
window==seqlen relationship). Killed the degraded run (PID 4688) and
its queue chain, relaunched fresh. Result: **4.41 it/s** immediately
after restart (was 17.83s/it average, worse and worsening before) — a
~2-orders-of-magnitude improvement, confirming dense-decode-at-1024 was
the actual bottleneck, not thermal/memory drift. New `bpelike_k4_1` ETA
~15min for the full 4000 steps. `selfcond_detach_k4` still queued behind
it (new wrapper/caffeinate chain rebuilt, watching python PID 8973).
Longer-term fix, not done here: extend `_packed_decode_forward_chunked`
to support `decode_K>1` so `context_len=1024` runs don't need this
workaround.

**`LevelLM._packed_decode_forward_banded` built: general O(L*sum(windows))
alternative to dense's O((2L)²), any track count/K/window/pack_mode.**
Supersedes the "extend `_packed_decode_forward_chunked`" TODO above --
this handles `decode_K>1` and multiple simultaneous cross-level tracks by
construction, and doesn't special-case `decode_pack_mode` at all (see
below). `Config.decode_banded: bool = False` (default off; dense
`_packed_decode_forward` remains the reference implementation).

Design: rather than choosing a packing order and hoping it stays
true_pos-monotonic (only true for `decode_K==1` interleave), build the
sequence once canonically (all prefixes, then bytes) and explicitly SORT
by `true_pos`, making it monotonic unconditionally. Then reuse
`_packed_decode_forward_chunked`'s existing chunk-with-margin banding
trick, generalized to a PER-KEY window (`window_of_key`) instead of one
shared scalar, since different tracks can have different windows. The
`allow` mask formula (causal & same_pos_code_excluded & windowed) is
copied verbatim from dense, just evaluated on a small gathered
`(sc, Kc)` window per query-chunk instead of the full `(Le, Le)` matrix
-- still an exact row-wise mask, just minimal in extent, not absent.

**Bug found and fixed: tie-break order at equal true_pos.** A track's
code sits at `true_pos = b*K-1`, one unit below its block's first byte
-- so a code and a byte routinely share the same `true_pos` (a "tie").
The mask is ASYMMETRIC at ties: a code query CAN see a same-true_pos
byte key (`same_pos_code_excluded` only excludes CODE keys), but a byte
query CANNOT see a same-true_pos code key. A backward-only gather over
one sorted sequence can only realize that asymmetry if ties are ordered
bytes-before-codes (so codes, sorted later, can look back and find
them; bytes, sorted earlier, never look far enough forward to find
same-true_pos codes at all). The original sort (stable, inheriting
codes-before-bytes from the packing order) had this backwards --
produced small-but-real errors (~0.04-0.08 max diff) that were EASY TO
MISS: they didn't show up with `n_layers=1` (byte outputs matched
exactly), only with `n_layers>=2`, because code positions' own hidden
states diverged first and only leaked into byte outputs via the next
layer's cross-attention read. A second, unrelated bug (insufficient
margin sizing -- `ceil(R/sc)` undercounts required sorted-array reach
when ties let true_pos advance by 0 for several consecutive sorted
steps; fixed by scaling margin by `len(tracks)+1`, the max ties possible
per true_pos value) was found and fixed first, before the tie-break bug
was isolated by comparing full (code+byte) hidden states layer-by-layer
against dense rather than just the extracted byte output.

**Correctness: verified exact.** New `scripts/test_v4_4_banded_decode.py`
-- 8 configs (single-track K=1, single-track K>1 decode_K, 2-track and
3-track cumulative cross-level conditioning with differing K/windows,
production scale) all match dense to float precision (`ALL MATCH`).

**Timing: RE-BENCHMARKED CLEANLY (queue fully idle, no MPS contention)
-- verdict: banded is consistently SLOWER than dense on MPS, sometimes
dramatically so.**

| config | dense | banded | banded/dense |
|---|---|---|---|
| K4 W16 L=256 | 7.2ms | 31.1ms | 4.3x slower |
| K4 W16 L=1024 | 34.4ms | 116.9ms | 3.4x slower |
| K4 W16 L=4096 | (impractical, skipped) | 458.9ms | n/a |
| Ks=(4,1) two-track L=1024 | 44.6ms | **1223.6ms** | **27x slower** |

Correctness re-confirmed (`ALL MATCH`, same 8 configs as before). The
multi-track case is dramatically worse -- likely the combination of
this session's conservative `(len(tracks)+1)*R` margin sizing (derived
for CORRECTNESS, not tuned for speed) inflating `Kc`, plus MPS's known
weakness at advanced-indexing/gather ops (the sort + gather-based
banding mechanism leans on exactly that). **Conclusion: `decode_banded`
stays `False` by default (already is) -- implemented and verified
correct, but not currently worth using on MPS.** Would need either a
CUDA target (where gather/indexing is much cheaper relative to dense
matmul) or a reimplementation that avoids the heavy sort+gather
machinery to realize the asymptotically-better complexity class in
practice on this hardware. Not pursued further this session --
correctness was the requested deliverable, and it's met; the
performance question is now answered (not favorably) rather than left
open.

**Bug found via verification: `generate_no_cache` silently falls back to
encode-only (unconditioned) prediction on 3 out of every 4 generation
steps for any `decode_K>1` config (e.g. `Ks=(4,1)` or single-level
`Ks=(4,)`).** Found while verifying (with running code, not just static
reasoning) whether generation reproduces training's packing/masking
exactly. `generate_no_cache` calls `RefineLM._run` fresh on the growing
byte sequence at every step and reads the LAST position's hidden state
(`h_list[0][:, -1, :]`) -- but `_run`'s own decode-activation logic
requires `L_i % cum_K == 0` for EVERY active track (`RefineLM._run`'s
`ragged` check), and SKIPS decode entirely for that level if not met
(falls back to `h_out[i] = h_i`, the plain encode-only hidden state,
computed with zero code-conditioning at all -- not a degraded/partial
version, a completely different, weaker signal). During training this
never triggers, since `context_len` is asserted divisible by every
track's cumulative stride at `RefineLM.__init__` time. During
byte-by-byte generation, though, the TOTAL sequence length only revisits
a multiple of `cum_K` once every `cum_K` steps -- for `Ks=(4,1)`
(`cum_K=4` for both the self and cross track), that means generation is
code-conditioned on step 1 of every 4, and completely unconditioned
(ignoring the code entirely) on the other 3.

Verified directly (`scripts/` scratch repro, not committed -- see
session transcript): built a tiny `Ks=(4,1)` model, ran `_run` once
teacher-forced over a full 16-byte sequence (decode active throughout,
by construction), then ran `_run` again on each growing prefix `L=8..16`
the way `generate_no_cache` does. Result: `L%4==0` steps exactly
reproduce the teacher-forced hidden state (diff `0.0000`); `L%4!=0`
steps diverge substantially (diff `0.44-1.02`, same order of magnitude
as the hidden state's own scale) -- i.e. NOT a small approximation
error, a qualitatively different prediction using none of the
code-conditioning signal the model was actually trained with.

**Practical impact**: `qualitative_generate`'s `level0_cond` field
(logged every eval as `qual_train_level0_cond` / `qual_val_level0_cond`
in every run this session with `decode_K>1` -- `bpelike_1level_k4_retry`,
`bpelike_k4_1`, and the currently-running `selfcond_detach_k4`) calls
`generate_no_cache` directly, so every one of those printed samples this
session has been ~75% unconditioned generation mislabeled as
conditioned. TRAINING METRICS (loss/bpb/accuracy) are NOT affected --
those only ever come from full-context teacher-forced `forward()`
passes, never from `generate_no_cache`. Only the qualitative eyeball
samples are compromised. Not yet fixed -- needs a generation loop that
tracks code state incrementally (buffer bytes into blocks, emit/condition
per-block rather than re-deriving decode-readiness from the raw growing
byte count every single step) rather than `generate_no_cache`'s current
"just re-run `_run` on the growing prefix" approach, which implicitly
assumed `decode_K==1` (always active) and was never re-examined when
`decode_K>1` configs were introduced.

**FIXED.** Simpler than the incremental-state-tracking approach sketched
above: pad the growing byte sequence up to the next multiple of
`decode_K` before each `_run` call, then read off the REAL last
position (index `L-1`, not the padded tail) for the next-byte
prediction, instead of always reading `h[:, -1, :]`. Pad value is
irrelevant. Why this is exact, not an approximation: causal attention
means position `L-1`'s hidden state can only depend on positions
`<= L-1`; the padding is appended strictly after it, so it literally
cannot be attended to, at any layer, at any level. The one subtlety --
the FINAL block straddles real content and padding (since
`pad_len < decode_K` always) -- doesn't matter either: that straddling
block's own code is only ever used as the PREFIX for the NEXT block
(strictly after `L-1`), never for its own bytes; position `L-1`'s
decode computation only ever reads PREVIOUS, fully-real blocks' codes,
at every level (padding is sized to the full product `decode_K`, so
every active track's block boundary aligns simultaneously). Verified
directly with the same methodology that found the bug: teacher-forced
reference vs. the fixed `generate_no_cache`-style padded call, for
every `L=8..15` in a `Ks=(4,1)` toy model -- ALL steps now match
exactly (diff `<1e-6`), where previously only `L%4==0` steps matched
and the rest diverged by `0.44-1.02`. Regression-checked: existing
`scripts/test_v4_4_chunked_decode.py` correctness suite still `ALL
MATCH`, and `qualitative_generate` runs cleanly end-to-end for both
`cross_track_source` settings post-fix. `generate_self_only_cond`
(built on top of `generate_no_cache` via `max_decode_sources=1`)
inherits the fix automatically, no separate change needed.

**`Config.decode_self_only_aux` built: level0's decode was only ever
trained on ONE fixed track combination, never "self-only."** Follow-up
finding to the `generate_no_cache` conditioning-gap bug above and the
checkpoint-generation verification (both this same session): confirmed
in code that `RefineLM._run`'s decode loop builds exactly one track
combination per level per step (self + every coarser level with a
nonzero window) and calls decode with it ONCE -- there was never a code
path training decode with the coarser tracks dropped (self-only). Two
of the three natural "how much conditioning does decode have" modes
already got gradient signal from different loss terms (uncond via
`encode_losses[0]`, the byte NTP loss on the plain encode pass with zero
code conditioning at all; self+level1/"full" via `decode_losses[0]`,
every training step since `context_len` is always divisible) -- but
self-only had none. This matters because self-only is exactly the
regime a graceful ragged-length generation fallback would need instead
of the current full-cumulative-or-nothing jump, AND exactly the regime
decode ends up in for real if a coarser level's own AR generation
degenerates (which was directly observed: `bpelike_k4_1`'s checkpoint
collapsed to a single repeated level1 code during generation, see the
checkpoint-generation verification section above).

Considered and rejected: random per-step dropout (truncate the track
list to a random prefix length each step, classifier-free-guidance
style, one mode active per step). Rejected per explicit user direction
("do it no dropout way, just a new loss path as if trained with 1
level") in favor of an ALWAYS-ON additional loss term: `Config.
decode_self_only_aux: bool = False` (opt-in). When True, `RefineLM._run`
runs a SECOND decode forward pass every step using only `tracks[:1]`
(the self track), alongside -- not instead of -- the existing
full-cumulative pass; both contribute to the loss every step. Weighted
by `Config.decode_self_only_weight` (default 1.0), reported as
`level{i}_ntp_loss_decode_self` / `level{i}_ntp_acc_decode_self` per
level and `decode_self_only_total` in the loss breakdown.
`decode_losses[i]` (full-cumulative, what `byte_loss`/`val_bpb` are
computed from) is untouched -- this is purely additive.

Implementation mechanism: `RefineLM._run` gained a `max_decode_sources:
int | None = None` param that truncates every level's track list to at
most that many sources before use. `max_decode_sources=1` forces
self-only; this is the SAME mechanism used both for the aux loss (called
internally with `tracks[:1]`) and for a new generation function,
`generate_self_only_cond` (`generate_no_cache` with `max_decode_sources=1`
forced at every step, sidestepping the ragged-length fallback for the
self track specifically -- there's no coarser track to be inconsistently
available when only the self track is requested).

**Naming scheme for the three modes** (used consistently in metrics and
qualitative output): "uncond" (zero code conditioning), "cond_self"
(self track only), "cond_full" (self + every coarser level, the
previous behavior, previously just called "cond"). `qualitative_generate`
renamed its existing `level0_cond`/`level0_cond_codes` fields to
`level0_cond_full`/`level0_cond_full_codes` and added
`level0_cond_self`/`level0_cond_self_codes` (annotated with level 0's
OWN code via `_decode_source_codes(..., level=0)`, not the topmost level
`_decode_source_codes` shows by default for the "full" case) --
verified all three modes print correctly and the aux loss trains/
backprops correctly via a small script before launching a real run.

**Verified via running code** (not committed, scratch repro): built a
tiny `Ks=(4,1)` model with `decode_self_only_aux=True`, confirmed (a)
`forward()` in train mode returns a nonzero `decode_self_only_total`
with a working `.backward()`, (b) the SAME model in eval mode returns
`decode_self_only_total=0.0` (aux only applies when `self.training`),
and (c) `qualitative_generate` prints all three modes plus two code-
annotation lines without error. Both `scripts/test_v4_4_chunked_decode.py`
and `scripts/test_v4_4_banded_decode.py` (whose `_run(...)` unpacking
needed updating for the new 8-tuple return, since `decode_self_only_
losses`/`decode_self_only_accs` were appended) re-verified `ALL MATCH`
after the change -- this feature is orthogonal to the banded/chunked
decode-attention paths, doesn't touch masking, just which tracks reach
them.

**New config**: `configs/qcute_refine_v4_4_bpelike_k4_1_selfonly_aux.py`
-- identical to `bpelike_k4_1.py` (`Ks=(4,1)`, `context_len=256`,
`attn_window=(8,256)`) plus `decode_self_only_aux=True`. Launched
immediately after `selfcond_detach_k4` finished (queue was idle, no
contention) -- first real run to exercise the self-only aux loss and
the three-mode qualitative print in practice, ~4.4 it/s, ETA ~15min.

**Bug found (via the user asking "does this train run actually include
all 3 modes"), significant: `main()`'s argparse/Config wiring silently
dropped FOUR Config fields added this session -- `decode_code_ste`,
`decode_banded`, `decode_self_only_aux`, `decode_self_only_weight`
(`vocab` too, though nothing had tried overriding it).** `main()`
constructs `Config(...)` from an explicitly hand-written kwargs list
that was never updated when these fields were added to the `Config`
dataclass, AND none of them had a matching `p.add_argument("--...")`
registered -- so `p.set_defaults(**{k: v for k, v in
load_config_module(...).items() if k in {a.dest for a in
p._actions}})` silently filtered them out at the FIRST stage already
(no matching argparse dest), before even reaching the `Config(...)`
call. Net effect: a config `.py` file setting any of these had that
setting completely ignored, no error, no warning -- the dataclass
DEFAULT was used instead, always.

**This invalidates `selfcond_detach_k4`'s entire premise.** That run's
whole point was `decode_code_ste=False` (detach, no STE, required per
`docs/two_stage_latent_decode_math.md` for the drafted-substitution
generation scheme) -- but the flag never reached the model. It trained
with the DEFAULT `decode_code_ste=True` (straight-through) the entire
time, which is a materially different training setup (gradient DOES
flow from decode into the code producer, the opposite of what the
experiment needed to isolate). The run completed successfully and
produced real numbers (`best_val_bpb=2.4369`), but those numbers don't
answer the question the run was built to answer. Needs a genuine rerun
now that the wiring is fixed -- todo list updated to reflect this
(item was "compare val_level1_ntp_acc_encode vs
val_level0_ntp_acc_decode from `selfcond_detach_k4`"; now blocked on a
rerun, not just interpretation of the existing checkpoint).
`bpelike_k4_1` and `bpelike_1level_k4_retry` did NOT set any of the
four broken fields, so those results are unaffected and still valid as
reported.

Also explains directly why `qcute_refine_v4_4_bpelike_k4_1_selfonly_aux`
(the run launched right after building `decode_self_only_aux`) showed
`decode_self_only_total=0.0000` in BOTH train-step and val logs --
looked at first like it might be the `self.training`-gating logic
itself being wrong, but the real cause was one level up: the flag never
reached `cfg.decode_self_only_aux` at all, so the gate's own condition
(`cfg.decode_self_only_aux and self.training and ...`) was always
false regardless of train/eval state. Killed that run immediately on
finding this (not measuring anything real).

**Fix applied**: added the five missing `p.add_argument(...)` calls and
wired all five into the `Config(...)` constructor call. Verified
directly: a 5-step smoke run now shows `decode_self_only_total=5.617`
on an actual train-step log entry (was `0.0000` before the fix).

**Permanent safety net added**, since this bug class (a Config field
quietly added without updating the hand-maintained CLI-wiring list) can
recur any time a new `Config` field is added: `main()` now asserts, at
argparse-setup time, that every `Config` dataclass field name has a
matching registered `--arg` dest (`{f.name for f in
dataclass_fields(Config)} <= {a.dest for a in p._actions}`, with an
explicit-but-currently-empty `_cli_excluded_config_fields` escape hatch
for any field that's deliberately CLI-unreachable in the future). A
future field added to `Config` without a matching `add_argument` call
now fails LOUDLY at startup with a clear message, instead of silently
training with the wrong config forever.

**Relaunched, chained**: `bpelike_k4_1_selfonly_aux` (now actually
exercising the self-only aux loss and 3-mode qualitative print) first,
`selfcond_detach_k4` rerun (now actually using `decode_code_ste=False`)
queued behind it.

**Live confirmation of the collapse, from `bpelike_k4_1_selfonly_aux`'s
own qualitative output (step ~2500)**: `level0_cond_full_codes` had
collapsed to a SINGLE repeated code value (`{24}` for every block in
one observed sample) and `level1_gen` (level1's own AR code generation)
had ALSO collapsed to a single repeated value, while that SAME run's
`level0_cond_self` showed healthy, varied codes
(`{5,8,103,141,166,62,213,...}`) for the identical prompt. Directly
motivates the next experiment below: isolate whether the cross-track/
level1 dependency is *the* cause of the collapse, not just a
contributing factor, by removing it entirely rather than adding a
parallel signal alongside it.

**New config**: `configs/qcute_refine_v4_4_bpelike_k4_1_selfonly_only.py`
-- same base numbers as `bpelike_k4_1.py`, but level0's cross track to
level1 is disabled entirely (`decode_window=0` for that source, not
just de-prioritized), verified via `decode_windows[0] == [8, 0]`. Since
the cross track never gets built, the ONE decode pass level0 runs IS
self-only by construction -- distinct from `decode_self_only_aux`
(which runs self-only as an ADDITIONAL pass alongside the full
cumulative one); this config never computes "self+code1" NTP at all.
Level1 itself is left otherwise unchanged (still trains its own encode
NTP over the `code_0` stream) purely so its qualitative output stays
available for comparison, even though level0 no longer depends on it.

**New script**: `scripts/compare_v4_4_checkpoints.py` -- loads six
checkpoints (`bpelike_1level_k4_retry`, `bpelike_k4_1` (old two-track),
`bpelike_k4_1_selfonly_aux`, `bpelike_k4_1_selfonly_only`, and the two
pre-existing "past-success" checkpoints `qcute_refine_v4_4_l1_k1`/
`l2_k1` for an external reference point), CPU-only (no MPS contention
with the training queue), runs `qualitative_generate` on the SAME real
validation-set prompt for each, prints `best_val_bpb` alongside. Not
run standalone -- auto-runs as the last step of the training queue
chain below, output to `/tmp/v4_4_checkpoint_comparison.log`.

**New config**: `configs/qcute_refine_v4_4_k1_k1_w32.py` -- degenerate
`Ks=(1,1)` (neither level compresses at all; both levels run at full
byte-rate sequence length), `attn_window=32` scalar (broadcasts to
every level's encode window AND every decode source's window
uniformly). A genuine 2-track (self+cross) `decode_K=1` case --
`decode_chunked` stays False since its single-track-only implementation
still can't take this shape (verified `decode_windows[0]==[32,32]`,
`decode_windows[1]==[32]`, decode active both levels before queuing).
Simplest possible uniform-window baseline, no per-track tuning, useful
as a clean sanity point.

**Full training queue as of this update** (each stage waits on the
previous via a `kill -0 <pid>` polling wrapper, `caffeinate -i -w
<wrapper pid>` layered on every stage so the machine can't sleep through
any of it):
1. `bpelike_k4_1_selfonly_aux` -- near-finished (step 3999/4000) as this
   was written.
2. `selfcond_detach_k4` rerun -- queued.
3. `bpelike_k4_1_selfonly_only` (the collapse-isolation test) -- queued.
4. `scripts/compare_v4_4_checkpoints.py` -- auto-runs after #3.
5. `k1_k1_w32` (degenerate sanity baseline) -- queued.

Per user direction ("auto research 12 hours, queue relevant hparams,
test generation, probe gradients iff needed, then always update docs"):
this session is now running as a self-paced autonomous loop, checking
back in via scheduled wakeups roughly matched to each stage's expected
duration (~15-20min/training stage observed so far), deciding follow-up
experiments from each stage's results, and updating this file after
every stage, not just at the end.

**Round 1 results: `bpelike_k4_1_selfonly_only` (the collapse-isolation
test) does NOT fix the collapse -- self-only collapses too, just more
slowly.** Live qualitative samples across its own training run: early
steps show varied codes, but by step ~4000 `level0_cond_self_codes` is
dominated by 2-3 repeating values (`{213}`/`{170}`/`{58}`, occasionally
alternating pairs like `{212,250}`) -- the classic signature of
codebook/index collapse, not a self-vs-cross-conditioning-specific
symptom. Ruled out: cross-level dependency is not "the" cause of
`bpelike_k4_1`'s original collapse (`{24}` constant) after all --
disabling it entirely still lets level0's own code collapse
independently, just on a different (slower) timeline.

**Follow-up diagnostic, `scripts/probe_code_usage_entropy.py`**: rather
than eyeballing more single-prompt qualitative snippets, measured code
usage entropy directly over ~5000 validation-set code tokens per
checkpoint (bits, active-index count, top-5 mass). Chose this over a
gradient probe as the first diagnostic step per the user's "probe
gradients iff needed" framing -- entropy/histogram analysis fully
explained the pattern without needing one. Results, most important
finding of this research round:

| checkpoint | Ks | code_0 entropy (max 8 bits) | code_0 active/256 | code_1 entropy |
|---|---|---|---|---|
| 1level | (4,) | 2.62 | 30 | -- |
| bpelike_k4_1 (old) | (4,1) | 2.01 | 13 | 0.00 (1 active) |
| +decode_self_only_aux | (4,1) | 1.76 | 16 | 0.00 (1 active) |
| self-only-ONLY | (4,1) | 1.76 | 16 | 0.11 (3 active) |
| selfcond_detach (real detach) | (4,1) | 0.92 | 14 | 0.03 (3 active) |
| **l1_k1 (past-success)** | **(1,)** | **6.03** | **219** | -- |
| **l2_k1 (past-success)** | **(1,1)** | **4.46** | **59** | **0.66 (23 active)** |

**The collapse tracks `Ks[0]=4` (grouping 4 raw bytes into one
discrete code), not the self-vs-cross architecture variant at all** --
EVERY `Ks[0]=4` config tested, including the plain 1-level one, shows
low code_0 entropy (0.9-2.6 bits); every `Ks[0]=1` config (one code per
raw byte, no block-grouping) shows healthy, actively-used codebooks
(4.5-6.0 bits, 59-219/256 indices active). `code_1` (level1's own
quantizer, `Ks=(4,1)` only) is far worse still -- essentially a single
constant symbol in every variant -- consistent with it trying to
further quantize an already near-degenerate ~2-bit `code_0` stream into
another 256-way codebook and finding nothing left to encode.

None of the `Ks[0]=4` configs tested so far touch `gumbel_tau`/
`use_gumbel_noise` (all left at the defaults: `tau=1.0`, no noise) --
both are standard collapse mitigations for discrete bottlenecks (noise
encourages exploration, higher tau softens assignment so more of the
codebook receives gradient). **New config**:
`configs/qcute_refine_v4_4_bpelike_k4_1_gumbelfix.py`
(`use_gumbel_noise=True`, `gumbel_tau=2.0`, otherwise identical to
`bpelike_k4_1.py`) -- queued to test whether this is a fixable
optimization issue or an inherent property of `Ks[0]=4` block-grouping.
`scripts/probe_code_usage_entropy.py` re-runs automatically after it
finishes, output to `/tmp/v4_4_entropy_probe_round2.log`.

**Current full queue** (7 stages, each `caffeinate`-protected):
1-3. `bpelike_k4_1_selfonly_aux`, `selfcond_detach_k4` rerun,
   `bpelike_k4_1_selfonly_only` -- done.
4. `compare_v4_4_checkpoints.py` -- queued.
5. `k1_k1_w32` (degenerate `Ks=(1,1)`, uniform window=32 sanity
   baseline) -- queued.
6. `bpelike_k4_1_gumbelfix` + entropy re-probe -- queued.
7. `selfcond_ste_k4` (decode_code_ste=True counterpart to
   `selfcond_detach_k4_rerun`) + `scripts/compare_ste_vs_detach.py` --
   queued.

**Ablation added per explicit user request: STE vs detach, properly
this time.** `selfcond_detach_k4`'s ORIGINAL run (pre-wiring-fix)
unintentionally WAS the straight-through condition despite its name --
so the "ablation" that mattered (does detach vs STE actually change
anything for this self-conditioning-only design) was never genuinely
run on both sides. New config: `configs/qcute_refine_v4_4_selfcond_ste_
k4.py` -- byte-for-byte identical to `selfcond_detach_k4.py` except
`decode_code_ste=True`. New script: `scripts/compare_ste_vs_detach.py`
-- diffs val metrics (`val_level0_ntp_acc_decode`, `val_level1_ntp_
acc_encode` i.e. the drafter's own accuracy at predicting `code_0`,
`val_bpb`) AND code usage entropy for both `code_0`/`code_1` side by
side between the two runs, auto-runs after `selfcond_ste_k4` finishes.

**Round 2: `compare_v4_4_checkpoints.py` output reveals a THIRD
mechanism, refining (not replacing) the `Ks[0]=4` finding.** Side by
side on the same real prompt, `k1_k1_w32` (Ks=(1,1), no block-grouping
at all -- currently training, ~52% done at this check) showed
`level0_cond_full_codes` collapsed to a single repeated value (`{211}`)
during GENERATION, despite `code_0` having HIGH entropy (4.46-6.03
bits) under the entropy probe's TEACHER-FORCED measurement (real
ground-truth bytes fed in, not self-generated). That's a real
distinction: the probe measures how diverse the trained model's code
assignment is GIVEN REAL DATA; generation feeds the model's OWN
predictions back autoregressively, which is a different regime.
`level0_cond_self` in that same sample showed much more code variation
(`{122,18,8,36,21,144,136,239,209,27,181,16,22}`) than `cond_full`
(`{211}` almost exclusively) -- fewer compounding self-referential
tracks fed back seems to reduce (not eliminate) the repetitive
collapse.

Traced the actual mechanism: EVERY generation function in this codebase
(`_sample_next_byte`, `generate_level1_codes`, `generate_blockwise_
parallel`) uses pure `argmax(-1)` -- no temperature, no top-k/nucleus
sampling anywhere. Pure greedy autoregressive decoding is a
well-documented cause of repetitive degenerate loops, independent of
any training-time codebook issue -- a real candidate explanation for
WHY generation specifically (not the entropy probe) shows collapse even
for healthy-entropy `Ks[0]=1` checkpoints.

**Tested directly, cheaply, no retraining needed** (scratch script,
temperature-sampled generation vs greedy on two EXISTING checkpoints,
`l2_k1` and `bpelike_k4_1`): temperature sampling (0.7, 1.0) changes
which specific tokens come out, but does NOT rescue overall text
coherence -- temp=0.7 is still word-salad for both, temp=1.0 is
actively WORSE (produces non-ASCII garbage bytes for `l2_k1`). **This
argues against "just add sampling" as a fix**: greedy decoding is a
real contributing factor to the specific visually-repetitive-code
pattern, but not the dominant explanation for the underlying quality
gap. The honest conclusion across all three investigated mechanisms so
far: (1) `code_0` teacher-forced entropy genuinely tracks `Ks[0]=4`
block-grouping (training-time, real effect, confirmed); (2) generation-
time repetitive collapse is partly a generic greedy-decoding artifact
(confirmed, but sampling alone doesn't fix quality); (3) these models
are simply undertrained at this scale (~1600-2600 steps, d_model=256,
~1M-byte corpus) -- likely the dominant factor underneath both (1) and
(2), not fully separable from them with the experiments run so far.

**Round 3: `k1_k1_w32` finished, confirms round 2's dual-mechanism
finding with a second independent data point.** `best_val_bpb=2.3943`
-- the best of every config tested this session (Ks=(4,) and Ks=(4,1)
included). Qualitatively: `level1_gen` STARTS varied
(`[21,8,27,8,8,131,...]`) then LOCKS INTO a repetitive loop (`240`
repeated ~58 times straight) mid-generation -- the textbook signature
of greedy-decoding degenerate-attractor collapse (a model wanders into
a locally-self-reinforcing token and never escapes it under pure
argmax), not a training-time codebook-entropy problem, since this same
checkpoint's `code_0` entropy under teacher forcing was already
measured healthy (round 1 table). `level0_cond_self` stayed richly
varied throughout the same sample (dozens of distinct code values) vs.
`level0_cond_full`'s total collapse to `{211}` -- consistent with round
2's "fewer compounding self-referential tracks fed back reduces (but
doesn't eliminate) the repetitive collapse" observation. Two
independent Ks=(1,1) checkpoints (`l2_k1`, now `k1_k1_w32`) both show
this exact pattern -- level1-generation-collapse-via-greedy-decoding
looks universal across `Ks[0]` values, not specific to block-grouping.

**Gumbel-noise grid added, per explicit user request ("repeat with true
later" / "grid").** User asked to hold off on temperature-annealing
(explicitly: "no annealing first, past training used no gumbel") and
instead first systematically test plain (static-temperature) Gumbel
noise across every base architecture already tested with the Config
default `use_gumbel_noise=False` -- confirmed precisely (grepped every
config file) that EVERY prior run this session, including the
past-success `l1_k1`/`l2_k1`, used the default `False`; the only
config setting `True` is `bpelike_k4_1_gumbelfix`. Completed the grid
with two new configs, both `use_gumbel_noise=True, gumbel_tau=2.0`
(matching `gumbelfix`'s values for a controlled comparison):
`configs/qcute_refine_v4_4_k1_k1_w32_gumbel.py` (Ks=(1,1) counterpart
to `k1_k1_w32`) and `configs/qcute_refine_v4_4_bpelike_1level_k4_
gumbel.py` (Ks=(4,) counterpart to `bpelike_1level_k4_retry`, run at
`context_len=256` instead of the original's 1024 -- no reason to repeat
the pre-context_len-fix O((2L)^2) slowdown here). Full grid:

| Ks | gumbel=False | gumbel=True |
|---|---|---|
| (4,) 1level | bpelike_1level_k4_retry | bpelike_1level_k4_gumbel (queued) |
| (4,1) | bpelike_k4_1 | bpelike_k4_1_gumbelfix (queued) |
| (1,1) | k1_k1_w32 | k1_k1_w32_gumbel (queued) |

`scripts/probe_code_usage_entropy.py`'s `CHECKPOINTS` list extended to
cover the whole grid; re-runs automatically as the LAST stage of the
queue, output to `/tmp/v4_4_entropy_probe_grid.log`.

**Round 4 results, two stages landed.**

**`bpelike_k4_1_gumbelfix` (static `tau=2.0`, `use_gumbel_noise=True`)
partially helps -- `code_1`, not `code_0`.** `code_0` entropy: 0.90 bits
-- essentially unchanged from every non-gumbel `Ks=(4,1)` baseline
(0.92-2.01 bits), gumbel noise did NOT rescue the primary Ks[0]=4
block-grouping collapse. `code_1` entropy: **0.85 bits** -- a real,
substantial jump from every non-gumbel `Ks=(4,1)` variant's ~0.00-0.11
bits (essentially one constant symbol), though still far below the
healthy 4-6 bit range `Ks[0]=1` configs show untouched. Verdict so far:
gumbel noise measurably helps the SECONDARY, more severely collapsed
quantity (level1's own code) but not the primary one (`code_0`'s
Ks[0]=4-driven collapse) -- a partial, not a full, fix.

**STE vs detach ablation (`compare_ste_vs_detach.py`), a genuinely
counterintuitive result.** The drafter-accuracy metric this whole
experiment exists to measure (`val_level1_ntp_acc_encode` -- how well
level1 predicts `code_0`) is dramatically BETTER under STE than detach:
**66.4% (STE) vs 47.5% (detach)**. By the letter of `selfcond_detach_
k4`'s own stated success criterion ("if level1 predicts code_0 well,
its drafted continuation is a credible substitute"), STE looks like the
winner. BUT: `code_0`'s own entropy is WORSE under STE (0.27 bits vs
detach's 0.92 bits) -- STE achieves its higher drafting accuracy partly
BECAUSE decode's gradient pushes `code_0` toward a MORE collapsed,
more trivially-predictable distribution (a near-constant target is
inherently easy to "predict"), not because level1 got better at
modeling a genuinely informative signal. `code_1` entropy also jumps
under STE (0.75 bits vs detach's 0.03) -- consistent with the same
"decode's gradient reduces collapse severity, but by making the target
less informative" pattern seen with gumbel noise above, though via a
completely different mechanism (gradient-path change, not
temperature/noise). `val_level0_ntp_acc_decode` (decode's own quality)
and `best_val_bpb` are roughly comparable between the two (STE
best_val_bpb=2.4258 vs detach 2.4320 -- within noise). Net read: the
higher "drafting accuracy" under STE is a somewhat hollow signal --
easier to predict a code that carries less information isn't the same
as level1 successfully modeling a rich, useful `code_0` distribution.
The two-stage-latent-variable design's ORIGINAL premise (code_0 stays
informative AND independently predictable) isn't cleanly validated by
either setting so far; the underlying `code_0` collapse (present
regardless of STE/detach) is still the more fundamental problem to
solve first.

**Optimization + conceptual clarification, per explicit user request:
the detach path's matmul replaced with a plain embedding lookup.**
`RefineLM._run`'s decode-conditioning path (`code_embeds = src @
self.encoders[i].embed.weight`) previously always used a full
`vocab x D` matmul regardless of `decode_code_ste`. When
`decode_code_ste=False` (detach), that's unnecessary: `source_c.
detach() @ embed.weight`'s forward value is mathematically a
one-hot-row selection, IDENTICAL to `embed.weight[source_c.argmax(-1)]`
-- and since no gradient into `source_c` is wanted in the detach case
anyway, a plain index lookup gives the exact same forward value AND the
exact same gradient w.r.t. `embed.weight` (index-select's gradient
scatters into the selected row, same as `one_hot@W`'s), for less
compute. Verified directly (scratch script): forward values match
exactly, `embed.weight.grad` matches exactly between the two
formulations, and ~1.6x faster at production-ish scale (vocab=256,
d_model=256, CPU). In the STE case (`decode_code_ste=True`), the matmul
is still REQUIRED and was left untouched -- `source_c`'s value is the
hard one-hot but its GRADIENT behaves as the soft distribution
(`gumbel_quantize`'s straight-through construction); that gradient
estimate only exists via the matmul, since index-select has no gradient
w.r.t. which index was chosen. Regression-checked: both existing
correctness suites (`scripts/test_v4_4_chunked_decode.py`,
`scripts/test_v4_4_banded_decode.py`) still report `ALL MATCH` after
the change (their timing sections were killed early to avoid MPS
contention with the live training queue -- correctness sections, which
run on CPU/small scale, completed first and are unaffected).

Also affirmed the user's framing directly, since it explains WHY this
optimization is not just a speed win but the conceptually correct
reading of what detach is FOR: the decoder should condition on the code
as a fixed discrete query/embedding lookup -- latent-variable /
Markov-chain style, where the decoder doesn't get to reshape which
latent it's conditioning on -- not as a soft mixture it could partially
steer. This directly reframes the STE-vs-detach ablation result just
above: STE's higher drafting accuracy (66.4% vs 47.5%) looks less like
a genuine win once you see it's achieved partly by letting decode's
gradient collapse `code_0` into something more trivially predictable
(lower entropy under STE, 0.27 vs detach's 0.92 bits) -- i.e. STE
blurs exactly the boundary detach is designed to keep clean. Detach
remains the architecturally-motivated choice for the two-stage
latent-variable design, independent of this specific ablation's raw
numbers; the `code_0` collapse itself (present under BOTH settings) is
still the more fundamental unsolved problem.

**New capability + ablation, per explicit user request: cross-track
conditioning from decode instead of encode.** User's question ("where
did you extract code_1 to pass to level0, is it h from uncond encode or
h from cond decode") led to confirming precisely: cross tracks always
came from `c_list[j]`, populated ONCE in the encode-only loop, never
from a coarser level's own decode output. User's follow-up argued
detach makes decode's role cleanest read as reconstruction-from-latent,
and that RECURSIVELY sourcing a lower level's cross track from the
level above's OWN decode pass is more consistent with that generative
structure than pulling from an uncond-NTP-focused encode pass -- except
for the degenerate `n_levels==1` case, where no higher-level decode
exists to source from at all (must stay on encode there, unconditionally).

Turned out to be directly implementable with much less new code than
expected: `LevelLM.forward` ALREADY computes a fresh code from
whichever `h` it produced (same pooling+classify+quantize pipeline
either way) and returns it as its first value -- `RefineLM._run`'s
decode loop was simply discarding it (`_, loss_i2, acc_i2, h_i2 = ...`).
New `Config.cross_track_source: str = "encode"` (`"encode"` | `"decode"`).
When `"decode"`: cross tracks (`j>i`, a coarser level's code) source
from that level's captured decode-derived code instead of `c_list[j]`,
falling back to `c_list[j]` if unavailable (e.g. level `j`'s decode was
itself ragged/disabled -- keeps correctness rather than crashing). SELF
tracks (`j==i`) are UNCHANGED regardless of this setting -- a level
can't condition its own decode on its own not-yet-decoded output, so
self always sources from encode. Requires top-down decode iteration
(`reversed(range(n_levels))`) so a coarser level's decode-derived code
exists before a finer level needs it; `_run` now always iterates this
way (provably a no-op for `"encode"`, since `c_list` is fully built
before the decode loop starts regardless of iteration order -- verified
via the existing `scripts/test_v4_4_chunked_decode.py` correctness
suite, still `ALL MATCH` after the change).

Verified directly: both settings train/backward without error and
produce genuinely different results (not a silent no-op), safety-net
assertion passes with the new field wired through argparse, and a real
end-to-end 2-step training run via the actual config-file-loading path
completes cleanly.

**New config**: `configs/qcute_refine_v4_4_bpelike_k4_1_crosstrack_
decode.py` -- identical to `bpelike_k4_1.py` except
`cross_track_source="decode"`. Updated in place (before it started
running -- wrapper still waiting on the gumbel grid at the time) per
explicit user follow-up ("with detach"): also sets `decode_code_ste=
False`, pairing the decode-sourced cross track with the same detach
principle already established for the self-conditioning experiments --
without it, decode's gradient could flow back through the RECURSIVELY-
SOURCED decode-derived code into level1's own code producer, compounding
across levels in a way that's especially hard to reason about once
decode's own output feeds the next level's conditioning input. Verified
the combination trains/backprops correctly (scratch smoke test) -- the
detach dispatch (`if cfg.decode_code_ste: matmul else: index`) applies
uniformly regardless of whether `source_c` came from `c_list[j]` or the
newly-captured decode-derived code, so no extra plumbing was needed.
Queued as the final stage of the current chain (after the gumbel grid),
`scripts/probe_code_usage_entropy.py`
extended with this checkpoint and re-runs automatically after it,
output to `/tmp/v4_4_entropy_probe_final.log`.

**Round 6: full gumbel grid landed, revises round 5's preliminary
read.** All three architectures now have both cells:

| Ks | gumbel | code_0 entropy | code_1 entropy |
|---|---|---|---|
| (4,) 1level | False | 2.62 | -- |
| (4,) 1level | **True** | **2.21 (WORSE)** | -- |
| (4,1) | False | 2.01 | 0.00 |
| (4,1) | **True** | **0.93 (WORSE)** | **0.85 (better)** |
| (1,1) | False | 3.72 | 0.00 |
| (1,1) | **True** | **5.00 (better)** | 0.13 (marginal) |

Round 5's "helps whichever code was already less collapsed" hypothesis
doesn't survive the third cell: `1level` had NO second code to be
"more collapsed" for comparison, yet its `code_0` still got WORSE under
gumbel (2.62->2.21) -- the same direction as `Ks=(4,1)`'s `code_0`. The
pattern that actually holds across all three cells: **gumbel noise
hurts `code_0` for BOTH `Ks[0]=4` architectures (1level AND 2level)
and HELPS it for the one `Ks[0]=1` architecture** -- correlates with
`Ks[0]` (task difficulty: compressing 4 raw bytes into one code is a
much harder discretization problem than 1-to-1), not with "which code
happened to be less collapsed already." Plausible mechanism (not
verified further this round): stochastic exploration during training
of an intrinsically HARD compression target may push optimization
toward a low-entropy "safe" solution faster, while the same noise is
pure beneficial regularization for an EASY, already-tractable target.
`code_1` (present only in the two 2-level configs, already the more
severely collapsed quantity in both) improved somewhat under gumbel in
BOTH cases (0.00->0.85 for `Ks=(4,1)`, 0.00->0.13 for `Ks=(1,1)`) --
this part of round 4/5's finding holds.

**Final verdict on static gumbel noise (`tau=2.0`, no annealing)**:
NOT a reliable fix. It actively makes the FLAGSHIP problem this whole
investigation started with (`Ks[0]=4`'s `code_0` collapse) slightly
WORSE, while helping a secondary quantity (`code_1`) and an already-
healthy architecture (`Ks[0]=1`'s `code_0`) that didn't need fixing as
badly. Static noise + fixed elevated temperature is not the answer;
proper annealing (start high, decay down over training -- the standard
Gumbel-Softmax recipe, deliberately NOT tested this round per explicit
user direction to test static noise first) remains untested and could
behave differently, but that's a separate future direction, not
something this round's results extrapolate to.

**Round 6, headline result: `cross_track_source="decode"` + detach is
the first genuinely clean win of the whole collapse investigation.**
`bpelike_k4_1_crosstrack_decode` (level0's cross track sourced from
level1's OWN cond decode pass instead of its uncond encode pass, paired
with `decode_code_ste=False` per user follow-up) vs the `bpelike_k4_1`
baseline (encode-sourced, default STE):

| metric | baseline (encode, STE) | crosstrack_decode (decode, detach) |
|---|---|---|
| `code_0` entropy | 2.01 bits | **2.37 bits (better)** |
| `code_1` entropy | 0.00 bits | **0.72 bits (much better)** |
| `val_level1_ntp_acc_encode` | 24.9% | **47.5% (nearly doubled)** |
| `best_val_bpb` | 2.4223 | 2.4246 (statistically unchanged) |

Both `code_0` AND `code_1` improved SIMULTANEOUSLY -- something no
other single intervention this session achieved (gumbel noise always
traded one off against the other, or actively hurt `code_0` for
`Ks[0]=4`; `decode_self_only_aux` and `self-only-ONLY` didn't move
`code_0` at all). Level1's own accuracy at modeling `code_0` nearly
doubled, essentially for free (`best_val_bpb` unchanged within noise).
Plausible mechanism: recursively sourcing the cross track from level1's
OWN reconstruction-style decode pass (rather than a plain uncond
encode pass) gives level1's decode objective a genuine downstream
consumer/purpose -- previously level1's decode ran and contributed a
loss term, but nothing else in the architecture actually USED its
output, whereas now level0 directly depends on the quality of what
level1's decode produces, which appears to pull level1 toward a richer,
more useful representation instead of collapsing into whatever's
locally easiest to fit its own isolated NTP loss.

**Final verdict on this round's collapse investigation, addressing the
open question directly**: NOT diminishing returns -- ends on a real,
positive, actionable architectural finding
(`cross_track_source="decode"` + detach), a better outcome than either
gumbel-noise path tested. Recommended as the new reference direction
for `Ks=(4,1)`-style configs going forward; a longer-training follow-up
of specifically this config (not the gumbel variants) would be the
natural next step if this investigation continues, but per the
session's own prioritization, pivoting NOW to the two long-deferred,
well-scoped engineering items below (queue is finally idle) rather than
opening a new research thread.

**Round 5 preliminary: gumbel noise helps DIFFERENT code depending on
architecture -- not a uniform effect.** Checked `k1_k1_w32_gumbel`'s
in-progress checkpoint (77% through training, `best.pt` already usable)
against its non-gumbel counterpart `k1_k1_w32`: `code_0` entropy
**3.72 -> 5.00 bits** (real, meaningful jump), but `code_1` stayed
collapsed (**0.00 -> 0.11 bits**, marginal). This is the OPPOSITE
pattern from `Ks=(4,1)`'s `gumbelfix` result (round 4: `code_1` jumped
0.00-0.11 -> 0.85 bits, `code_0` stayed flat at ~0.9 bits). So gumbel
noise isn't uniformly "fixing collapse" -- it seems to help whichever
code was CLOSER to being learnable already (`Ks=(1,1)`'s already-healthy
`code_0` gets pushed higher; `Ks=(4,1)`'s already-more-active `code_1`,
relatively speaking, gets pushed up more than its severely-collapsed
`code_0`). Preliminary -- full grid (including `bpelike_1level_k4_gumbel`,
the third cell) still running; will confirm/revise once all three
architectures' gumbel-vs-no-gumbel pairs are in.

**Full queue as of this update (10 stages)**: stages 1-5 done
(`bpelike_k4_1_selfonly_aux`, `selfcond_detach_k4` rerun,
`bpelike_k4_1_selfonly_only`, `compare_v4_4_checkpoints.py`,
`k1_k1_w32`); 6. `bpelike_k4_1_gumbelfix` + entropy re-probe; 7.
`selfcond_ste_k4` + `compare_ste_vs_detach.py`; 8. `k1_k1_w32_gumbel`;
9. `bpelike_1level_k4_gumbel`; 10. `probe_code_usage_entropy.py` (full
grid). All `caffeinate`-protected end to end.

**Also addressed directly (user question): does level1 also run its
own self-code decode, analogous to a 1-level config's level0
self-decode?** Yes, confirmed via both code (`RefineLM._run`'s decode
loop is unconditional over `range(n_levels)`, top level always gets
exactly one track -- its own code) and log evidence
(`bpelike_k4_1`'s own val log has `val_level1_ntp_loss_decode=2.2937
val_level1_ntp_acc_decode=0.2495` populated). ONE exception found:
`selfcond_detach_k4`(_rerun) deliberately disables level1's own decode
(`attn_window`'s level1 entry is `(64, 0)`) by original design -- level1
was meant to be a pure NTP drafter with no self-conditioning confound.
Asked the user whether to change this given the collapse research;
answered "leave as-is" -- that isolation stays intact, not touched.

**Answers the "does packing scheme affect efficiency" question**: no,
by construction, for the banded path -- it ignores `cfg.decode_pack_mode`
entirely (always builds prepend-style pre-sort order, then explicitly
sorts by true_pos), generalizing dense's own docstring observation that
"physical packing order doesn't affect correctness, only true_pos does"
into the efficiency question too. Packing mode only ever mattered for
the OLD chunked path (`_packed_decode_forward_chunked`), which needed
physical interleave order as a substitute for an explicit sort -- a
shortcut that only worked because `decode_K==1`, single-track made
physical order and true_pos order coincide already.
