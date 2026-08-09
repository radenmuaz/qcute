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
