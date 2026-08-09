# Does `DecoderLevel`'s cross-attention KV actually matter?

Cross-referenced from [docs/status.md](status.md). Running thread across
this session: does the cross-attention KV (`EncoderLevel[level+1]`'s own
hidden states — the coarser code) in `qcute_refine_v2`'s `DecoderLevel`
actually contribute to predictions, or is it something the model has
learned to mostly ignore? Three probes/experiments, in order.

## 1. `qcute_refine_v2_byte4_code256_simple` ("v1") — mixed/inconclusive

*(This config and its results were later DELETED — see
[docs/status.md](status.md)'s own correction note for why. Findings kept
here as historical record; don't cite this run as a live reference
point.)*

`scripts/probe_decoder_kv_contribution.py` (gradient-norm ratio,
KV-ablation loss/acc delta, null-slot attention mass — three independent
signals since no one alone is trustworthy) against its `best.pt`:
`grad_ratio_curr_over_prev` **0.01-0.02** (KV side's gradient is ~1-2% of
Q side's), `null_slot_attn_mass` **~0.29** (vs. ~0.004 uniform over 257
KV positions — the model has learned to substantially opt out of
attending to real code content), and on val data specifically, **ablating
KV entirely (forcing null-only) *lowers* loss** (1.687 vs. 1.731 with
it) despite accuracy being very slightly *worse* without it (0.532 vs.
0.538). Cross-checked against the FULL `run.jsonl` trajectory (not just
one checkpoint): `pair0_tok_acc` (with KV) is consistently ≥
`level0_ntp_acc` (without KV) from step ~1600 onward, small but
persistent (+0.001 to +0.016) — so KV *does* give a real, if small,
accuracy edge across most of training. Read together: **overfit
calibration, not overfit accuracy** — the decoder's cross-attention head
grows more confident in ways that help top-1 accuracy marginally but hurt
held-out cross-entropy (a classic overconfidence signature, not a "KV is
useless" one). Genuinely unresolved which effect dominates in practice —
motivated experiment 2 below.

**Side findings from this run**: real bug found and fixed —
`DecoderLevel`'s cross-attention KV window was unbounded (the causal
mask enforced *that* a KV block must be complete before being visible,
but never capped *how far back*, inconsistent with the encoder's own
windowed self-attention at that level). Fixed: `DecoderLevel` now also
takes `kv_window = Config.attn_window[level+1]` and requires `b >=
n_complete(t) - kv_window` too (`None`/`-1` preserves unbounded reach) —
see `docs/qcute_refine_math.md` §7.1. Also added `Config.cross_attn_rope`
(default `True`): Q gets its own raw-byte-time RoPE position, each KV
slot gets the raw-byte-time position it becomes causally resolved at —
gives the cross-attention actual relative-distance information instead
of only the boolean visible/blocked mask. See `docs/qcute_refine_math.md`
§7.2.

## 2. `qcute_refine_pass_through` — UNAMBIGUOUS (opposite read)

Full 4000-step run (`decoder_kv_pass_through=True` +
`decoder_q_pass_through=True`, `cross_attn_rope=True`, windowed KV):
**best val_bpb 2.5575 @ step 3300**.

`scripts/probe_decoder_kv_contribution.py` **adapted** to support
non-default decoder modes (`_compute_qkv` now mirrors
`DecoderLevel.forward`'s own own_trunk/q_pass_through/kv_pass_through
dispatch instead of hardcoding `h_prev`/`h_curr` reuse — the original
script crashed outright against this checkpoint, since `q_embed` is an
`nn.Embedding` and pass-through mode feeds it raw byte ids, not a
continuous `h_prev` hidden state). Gradient-norm signal is skipped
(`NaN`) on any side using pass_through/own_trunk, since there's no
continuous encoder hidden state on that side to take a gradient w.r.t. —
ablation and attention-mass remain fully meaningful and are the ones
that matter here anyway.

Re-run against `qcute_refine_pass_through`'s `best.pt` (8 batches x 16,
train + val): **`delta_loss_from_kv` ≈ −0.23 to −0.24** (ablating KV
entirely *raises* loss substantially — opposite sign from experiment 1's
val-only finding), **`delta_acc_from_kv` ≈ +0.036 to +0.038** (consistent,
positive, same sign on train AND val — no train/val split this time), and
**`null_slot_attn_mass` ≈ 0.05–0.06** (vs. ~0.29 in the reuse-mode run —
attention is overwhelmingly on real code content, not opting out to the
null fallback).

**Reading the two experiments together**: when Q/KV are reused encoder
hidden states already carrying most of the signal, cross-attention has
less left to add and drifts toward overconfident-but-not-more-accurate
null-heavy attention (experiment 1's "overfit calibration, not accuracy"
read); when Q/KV are stripped to raw pass-through with no such shortcut,
the model has no alternative but to genuinely rely on the cross-attention
KV, and does — consistently, causally, on both loss and accuracy. **Net:
KV contribution depends heavily on what else is available to lean on,
not a fixed property of the cross-attention mechanism itself.**

## 3. Mask correctness re-verification (side effect of torch.compile debugging)

While debugging `torch.compile` (see [docs/torch_compile.md](torch_compile.md)),
`DecoderLevel`'s cross-attention KV mask (`n_complete = (t_idx+1)//K`,
`visible = b_idx < n_complete`) was re-verified correct — a "jagged
staircase" pattern, empirically confirmed for K=4: visible block count
steps +1 every K raw positions, but the step boundary lands at
`t=(b+1)*K-1` (not a K-multiple) — block b's code depends on
`EncoderLevel`'s hidden state at that block's LAST position, so it isn't
causally available until that byte has been seen; the formula gets this
exactly right (no off-by-one, no label leakage). Documented directly in
the mask's own code comment in `qcute_refine_v2.py`. This matters for
interpreting experiments 1/2 above — the mask itself was never the
confound, only what's on each side of it (reused hidden states vs. raw
pass-through) was.

## 4. `configs/qcute_refine_tiny_byte_window.py` — queued, forcing the question further

Clone of `qcute_refine_rope.py` with `attn_window` changed `(256, 64)` ->
`(8, -1)`: level 0 (byte encoder) window shrunk to an extremely tiny 8
raw bytes (self-attention alone can see almost nothing), level 1 (code
encoder) set to dense/full attention over its own 256 code positions (its
own effective receptive field now spans the WHOLE 1024-byte context).

Rationale: if experiments 1/2's mixed-to-unambiguous split was partly
because level 0 in the reuse-mode runs already had plenty of local
context (its own `attn_window=256` already covers a full 256-byte
lookback) to lean on instead of the cross-attention, this config removes
that alternative almost entirely — level 0 is nearly a bag-of-8-bytes
model on its own, so genuine cross-attention KV contribution (if any)
should show up starkly in a `probe_decoder_kv_contribution.py` re-run
against its checkpoint, in REUSE mode this time (not pass-through), which
would isolate the "does level 0 have an alternative to lean on" variable
specifically. Smoke-tested (3 CPU steps, no shape/divisibility errors, no
dense-fallback warnings) before queueing.

**Result: RAN — best val_bpb 2.6206 @ step 2600 (close to `qcute_refine_rope`'s
2.6310 despite the crippled window, itself notable — the cross-attention
KV substitutes for local context almost fully). Probe re-run against
`best.pt` in REUSE mode (128 batches x 16, train + val):
`delta_loss_from_kv` mean **−0.094 (train) / −0.103 (val)** (ablating KV
consistently RAISES loss, both splits — same sign as experiment 2's
pass-through result, unlike experiment 1's val-only sign flip),
`delta_acc_from_kv` mean **+0.0147 (train) / +0.0135 (val)** (consistent,
positive, both splits), `null_slot_attn_mass` **≈0.51** (between
experiment 1's ~0.29 reuse-mode value and experiment 2's ~0.05-0.06
pass-through value — attention is split roughly evenly between the null
fallback and real code content, not overwhelmingly either way),
`grad_ratio_curr_over_prev` **≈0.07-0.08** (KV side's gradient ~7-8% of Q
side's — small but nonzero, unlike experiment 1's ~1-2%).

**Confirms the hypothesis**: with level 0's local receptive field
crippled to 8 raw bytes, cross-attention KV contribution becomes
unambiguous and consistent (same sign, both train and val) — much closer
to experiment 2's pass-through finding than experiment 1's reuse-mode
mixed result, even though this run (unlike experiment 2) still uses
REUSE mode for Q/KV. This isolates the variable experiment 1 left
confounded: it's not reuse-vs-pass-through mode per se that determines
whether KV matters, it's whether level 0 has enough of its OWN local
context to lean on instead. Strengthens the session's net conclusion:
**KV contribution depends on what else is available to lean on, not a
fixed property of the mechanism** — genuinely gradable, not binary,
confirmed now by a third independent lever (window size) pointing the
same direction as reuse/pass-through mode did.

## 5. Related: `cross_attn_rope` doesn't help (`docs/status.md`'s own ablation)

Worth noting alongside the above: `configs/qcute_refine_no_rope.py`'s
completed run (see [docs/status.md](status.md) for the full result) shows
`cross_attn_rope=False` beating `cross_attn_rope=True` — 2.5645 vs. 2.6310
best val_bpb, same architecture, same budget. Positional information on
top of the cross-attention KV doesn't help here, even though the KV
content itself demonstrably does (experiments 2 and 4 above). Suggests
the cross-attention mechanism's value is in WHAT it retrieves (the
coarser code's content), not WHERE it's retrieved from — the jagged
block-visibility mask (§3 above) may already carry enough positional
structure on its own, making the explicit RoPE signal redundant or
mildly distracting.

## 6. Every self-attn/cross-attn stacking combination tried, and whether `byte_loss` actually gets conditioned

The KV-contribution probes above all measure `tok_loss`/`pair0_tok_acc` —
a metric that turned out to be structurally disconnected from
`val_bpb`/`byte_loss` (the metric every baseline comparison in
[docs/status.md](status.md) actually uses) in every v2 decoder mode,
regardless of which one: `DecoderLevel` always runs AFTER the full
bottom-up encoder sweep finishes, and both sides of its cross-attention
read (`h_list[i]`/`h_list[i+1]`) are `.detach()`'d before use — so
`tok_loss` can genuinely depend on the coarser code (as these probes
show), while `byte_loss` never can, in ANY v2 mode. See the module
docstring's own comment: "this decoder's loss must not reshape either
EncoderLevel's own hidden state."

Two distinct senses of "conditioned" matter here: **forward-value**
conditioning (does `byte_loss`'s own computation literally use level 1's
values?) vs. **gradient-only** conditioning (do the WEIGHTS producing
`byte_loss` get shaped by level 1's task, even with no forward path?).

| scheme | self-attn structure | cross-attn structure | `byte_loss` conditioned? | mechanism | it/s | best val_bpb |
|---|---|---|---|---|---|---|
| `bytelm_xs1` (reference, not qcute) | 1 layer, dense full-context | none | — | — | 3.429 | 2.4870 |
| v2 `rope`/`no_rope`/`identity`/`curriculum`/`tiny_byte_window` | windowed, bottom-up, shared weights | 1 `CrossBlock`, detached reads, separate pass | No | — | 0.48-2.55 | 2.5645-2.6463 |
| v2 `pass_through` | none (Q/KV are raw embeddings) | 1 `CrossBlock`, own fresh embeddings, detached | No | — | 2.149 | 2.5575 |
| v2 `own_trunk` | separate full trunk copy, then 1 `CrossBlock` | same, detached | No | — | 1.407 | 2.5793 |
| v2 `pq_table` | same as `no_rope`, code-embed changed | detached (unchanged) | **Indirectly** | shared-weight gradient dynamics — `code_pre`/`self.blocks` are shared between `byte_loss` and level 1's own (now better-conditioned) task; no forward path, but a real, measured effect | 2.017 | **2.4816** |
| v3 fusion (`v3_rope`) | windowed pass 1, then a 2nd self-attn pass fused with cross-attn before it | cross-attn INSIDE the encoder, undetached for the fusing level's own weights | **Yes, directly** | real forward-value path — `byte_loss` literally depends on level 1's hidden state, not just shared parameters | 1.039 | **2.4302** — beats `bytelm_xs1` outright, largest single delta of the session (+0.201 vs. `rope`) |

`v3_rope`'s result is the clearest evidence yet that direct forward-value
conditioning is a materially stronger lever than the shared-weight/
gradient-only channel `pq_table` exploits — though `pq_table` alone
already closed nearly the whole gap to `bytelm_xs1`, suggesting the two
mechanisms may be capturing overlapping, not fully independent, sources
of improvement. `configs/qcute_refine_v3_rope_pq.py` (queued, stacks
both) is the direct test of whether they're additive.

## 7. `qcute_refine_v4_k32_narrow` — most of fusion's benefit is capacity, not content

v2/v3's probes above all measure `tok_loss`, structurally disconnected
from `byte_loss`. v4 fixes that disconnection — fusion feeds `byte_loss`
directly — so probing it no longer needs gradient-norm/attention-mass
proxies: just re-evaluate the SAME trained checkpoint under different
ablations and read `val_bpb` off directly.
`scripts/probe_v4_fusion_contribution.py` (new this session — the old
`probe_decoder_kv_contribution.py` targets `DecoderLevel`, which v4
doesn't have) does exactly this, four ways, same weights throughout:

1. **normal** — fusion as trained.
2. **null_only** — PASS 2 still runs (same `fuse_cross` module, same
   null slot), but `fuse_kv` is replaced with zeros before use — isolates
   whether the model needs REAL coarser-level content, or whether just
   having a KV tensor (any tensor) to cross-attend to is most of the
   benefit.
3. **big_noise** — `fuse_kv` = real content + large-magnitude (10x its
   own std) i.i.d. Gaussian noise — a STRONGER control than null_only:
   zeros are a degenerate, structured input a model could plausibly
   handle specially; large additive noise keeps real variance/structure
   partially present but swamps the actual signal, separating "needs a
   varying signal to attend to" from "needs the SPECIFIC content."
4. **no_fusion** — `fuse_encoder_levels=False`, full ablation, matching
   v2's own `byte_loss` computation (PASS 1 only).

Run against `qcute_refine_v4_k32_narrow`'s `best.pt` (20 val batches):

| mode | val_bpb | Δ from normal |
|---|---|---|
| normal | 2.5528 | — |
| null_only | 2.7582 | +0.2053 |
| big_noise | 2.8437 | +0.2909 |
| no_fusion | 4.9759 | +2.4230 |

**Reading**: removing fusion ENTIRELY is catastrophic (+2.42 bpb, near
the ~8-bit-per-byte random-guessing ceiling) — expected for this specific
config, since `attn_window=(32,32)` means level 0's own self-attention
window exactly equals one code block (`K=32`); without fusion, level 0
has essentially NO access to anything beyond its current 32-byte chunk.
But the decomposition is the real finding: **both controls recover the
vast majority of that 2.42 bpb with NO real coarser-level content** —
null_only recovers 2.22 (≈92%), big_noise recovers 2.13 (≈88%) — the
`fuse_cross` module's own weights (extra cross-attention QKV/MLP
parameters, plus the always-real, never-corrupted learned `null_kv`
slot) apparently function as genuine extra capacity/depth for level 0's
own forward pass, largely independent of what they're attending to. Only
≈8-12% (0.21-0.29 bpb) is attributable to the coarser level's ACTUAL
content. Consistency check: big_noise scores worse than null_only
(2.8437 vs. 2.7582) — active noise actively disrupts the computation,
worse than a clean, predictable zero input the model can partially learn
to route around — the two controls agree with each other, not just
individually with the headline finding.

**Follow-up queued**: `Config.fuse_use_null_kv` (new — see
`qcute_refine_v4.py`'s own docstring) lets the null KV slot be removed
architecturally, not just content-ablated post-hoc. `configs/qcute_
refine_v4_k32_narrow_postfuse_nonull.py` stacks this with `fuse_position=
"post"` (testing whether `self.blocks` stays a cleaner, more robust
standalone local encoder without ever seeing fused input) on the same
K=32/narrow-window architecture — asks directly how much of the
capacity-not-content finding above was the null slot specifically, versus
`fuse_cross`'s other parameters. Not yet run.

This adds real nuance to `qcute_refine_v3_rope`'s "direct forward-value
conditioning beats everything" reading from §6 above: for THIS
narrow-window config at least, most of that forward-value benefit isn't
really about the CONTENT being conditioned on — it's about the extra
processing capacity the fusion mechanism happens to add along the way.
Consistent with, and motivates, the `tier_n_layers=(2,1)`/`(2,2)` depth
ablations queued this session (see docs/status.md) — if plain depth
matches or beats fusion's own contribution, that would confirm capacity,
not cross-level information, was the dominant lever all along.

Note: the probe's `no_fusion` mode above was later renamed
`unconditional_pass1` (§8's numbers use the new name; same computation,
just a clearer label — it's PASS 1's own byte_loss, matching v2 exactly).

## 8. `qcute_refine_v4_k32_narrow_postfuse` — pre vs. post, and the representational-separation hypothesis

Same K=32/narrow-window architecture as §7, one change:
`fuse_position="post"` (cross-attention runs AFTER `self.blocks`, not
before). Trained result: best val_bpb **2.5033 @ step 3600** vs. §7's
pre-fusion **2.4926** — post is slightly worse at matched params/FLOPs
(identical architecture, only the fusion's position in the block
differs), consistent with `docs/status.md`'s earlier `fuse_position`
ablation on the K=4 pq_table config (post also lost there, 2.4678 vs.
2.4565).

Probed the same way as §7 (`scripts/probe_v4_fusion_contribution.py`,
20 val batches, `checkpoints/qcute_refine_v4_k32_narrow_postfuse/best.pt`):

| mode | pre (`k32_narrow`) val_bpb | post (`k32_narrow_postfuse`) val_bpb |
|---|---|---|
| normal | 2.5757 | 2.5208 |
| null_only | 2.7895 | 2.5994 |
| big_noise | 2.8496 | 2.5384 |
| unconditional_pass1 (no fusion) | 4.9456 | 4.6761 |

(pre-mode numbers here are a clean rerun with the renamed probe script,
not §7's original run — same checkpoint, same config, numbers differ
slightly from §7's 2.5528/2.7582/2.8437/4.9759 only due to eval-batch
sampling noise; both readings support the same conclusion.)

**Two separate findings, pulling in different directions:**

1. **Post's standalone trunk IS more robust, as the representational-
   separation hypothesis predicted.** `unconditional_pass1` (fusion fully
   removed, level 0 running as a pure local encoder) scores 4.6761 for
   post vs. 4.9456 for pre — post is 0.2695 bpb BETTER with no fusion at
   all. This makes sense structurally: in "post" mode, `self.blocks` runs
   first and produces a real, self-contained local representation before
   cross-attention ever touches it; in "pre" mode, cross-attention runs
   first and `self.blocks` learns to expect an already-fused input,
   so stripping fusion leaves `self.blocks` operating on an
   out-of-distribution input it never saw during training.

2. **But post relies EVEN MORE heavily on the null/structural slot for
   its OWN normal-mode performance, not less.** Of fusion's total
   contribution (`normal` − `unconditional_pass1`), null_only alone
   recovers 91.0% of it for pre vs. **96.4%** for post; big_noise
   recovers 88.4% for pre vs. **99.2%** for post. In post mode, real
   coarser-level content is responsible for essentially none of
   `normal`'s advantage over a noised/null KV (big_noise is only 0.0176
   bpb worse than normal for post, vs. 0.0699 for pre) — almost the
   entire post-fusion benefit is the extra `fuse_cross` capacity/null-slot
   itself, even more so than §7's already-capacity-dominated pre-mode
   finding.

**Reading**: these aren't contradictory — they're about two different
counterfactuals. Removing fusion ENTIRELY (`unconditional_pass1`) hurts
post less, because post's trunk never became dependent on fusion being
present. But GIVEN that fusion runs, post's cross-attention branch itself
contributes almost nothing beyond raw capacity (content barely matters),
more so than pre's. Net: post buys a cleaner fallback/more modular trunk
at the cost of an even less content-driven fusion mechanism, and overall
scores slightly worse at matched compute (2.5033 vs. 2.4926) — the
representational-separation trade doesn't pay for itself here. Still
motivates finishing the 2x2 grid (`fuse_use_null_kv=False` in both
positions) to isolate how much of post's null-slot dependence is the slot
itself vs. `fuse_cross`'s other parameters.

## 9. `qcute_refine_v4_k32_narrow_postfuse_nonull` (post + no null KV) — best trained result of the grid so far, and a genuine surprise

Third of four 2x2 cells: `fuse_position="post"` + `fuse_use_null_kv=False`
(no learned null slot at all — `_fuse`'s cross-attention has ONLY the
real coarser-level KV rows to attend to, nothing else). **Trained result:
best val_bpb 2.4799 @ step 1800** — better than BOTH `post+null`
(2.5033) and, more surprisingly, better than `pre+null` (2.4926, the best
of this whole K=32/narrow-window family until now). Removing the null
slot in "post" mode isn't just neutral, it's the best config in the
K=32 family tried this session.

Probed the same way (`scripts/probe_v4_fusion_contribution.py`, 20 val
batches, `checkpoints/qcute_refine_v4_k32_narrow_postfuse_nonull/best.pt`):

| mode | pre+null | post+null | post+no-null |
|---|---|---|---|
| normal | 2.5757 | 2.5208 | 2.5005 |
| null_only | 2.7895 | 2.5994 | 2.5596 |
| big_noise | 2.8496 | 2.5384 | 2.5314 |
| unconditional_pass1 (no fusion) | 4.9456 | 4.6761 | **5.6564** |
| null_only/big_noise recover... | 91.0%/88.4% | 96.4%/99.2% | 98.1%/99.0% |
| of fusion's total benefit | | | |

Two findings, one confirming the trend, one genuinely unexpected:

1. **Confirms the "capacity, not content" trend continues to strengthen**:
   with no null slot to fall back on at all, `null_only`/`big_noise`
   still recover 98-99% of fusion's benefit — essentially ALL of it, even
   more than post+null's already-dominant 96-99%. Real coarser-level
   content is doing almost nothing measurable in this config, with or
   without a null slot present.

2. **Unexpected: `unconditional_pass1` gets dramatically WORSE without
   the null slot (5.6564 vs. 4.6761 — +0.98 bpb), even though the null
   slot barely mattered for `normal`-mode performance.** The null slot's
   real value, on this evidence, isn't what it contributes to the forward
   pass when present — it's a TRAINING-time effect: without it,
   `self.blocks` (PASS 1) apparently never learns to be self-sufficient,
   because `_fuse`'s cross-attention (now real-content-only, no fallback)
   is always "available" during training, so gradients never push PASS 1
   toward standalone competence the way `null_only`'s own robustness
   might. With the null slot present, evidently something about having
   an always-visible, low-information anchor slot in the KV set makes the
   REST of the network (particularly PASS 1's own trunk) train into a
   more standalone-capable state — a genuine regularization effect,
   opposite in direction from what its ~0% content-recovery share in
   `normal`-mode would suggest on its own.

**Net reading of the grid so far (3 of 4 cells)**: null_kv's role is
almost entirely about TRAINING dynamics/robustness, not about what it
contributes to a normal forward pass (which is consistently ~0-10%,
across every cell). Removing it can still IMPROVE the trained result
(post+no-null beats post+null on `normal`-mode val_bpb, 2.4799 vs.
2.5033) even while making the model dramatically MORE fragile to full
fusion ablation (5.6564 vs. 4.6761) — two different things, easy to
conflate. `qcute_refine_v4_k32_narrow_nonull` (pre+no-null, the last
cell, training as of this writing) will show whether this same pattern
(no-null beats null, unconditional_pass1 gets much worse) holds in "pre"
mode too, or is specific to "post."

## 10. Full 2x2 grid `{pre,post} x {null,no-null}` — consolidated

All four cells now trained and probed (`checkpoints/qcute_refine_v4_
k32_narrow{,_postfuse}{,_nonull}/best.pt`, same K=32/narrow-window
architecture, identical params/FLOPs across all four — only
`fuse_position`/`fuse_use_null_kv` differ):

| cell | trained best val_bpb | @ step | null_only recovers | big_noise recovers | unconditional_pass1 |
|---|---|---|---|---|---|
| pre + null (`k32_narrow`) | 2.4926 | 2800 | 91.0% | 88.4% | 4.9456 |
| post + null (`postfuse`) | 2.5033 | 3600 | 96.4% | 99.2% | 4.6761 |
| **post + no-null (`postfuse_nonull`)** | **2.4799 (best of grid)** | 1800 | 98.1% | 99.0% | 5.6564 (worst of grid) |
| pre + no-null (`nonull`) | 2.4961 | 2700 | 95.9% | 97.4% | 5.0124 |

Four findings, read together:

1. **Content contribution shrinks monotonically toward ~0 as you move
   away from pre+null.** null_only/big_noise recovery climbs from
   91/88% (pre+null, the "most content-dependent" cell) up to 98/99%
   (post+no-null, the "most capacity-dependent" cell) — real
   coarser-level content matters least exactly where the architecture is
   least conservative (post ordering, no fallback slot).

2. **"pre" is robust to null-slot removal in BOTH directions; "post" is
   sensitive in both.** Removing null_kv barely moves pre's trained
   result (2.4926→2.4961, +0.0035, a wash) or its unconditional_pass1
   floor (4.9456→5.0124, +0.067, mild) — but for post it improves the
   trained result substantially (2.5033→2.4799, −0.023) while roughly
   DOUBLING the unconditional_pass1 penalty (4.6761→5.6564, +0.98).
   Structural explanation: in "pre" mode, `self.blocks` always runs on
   fusion's OUTPUT (with or without a null slot, it's still post-
   cross-attention data self.blocks can normalize/adapt to), so the
   null slot's presence is a smaller perturbation to what self.blocks
   sees. In "post" mode, `self.blocks` runs FIRST, entirely independent
   of whatever `_fuse` does afterward — so whether `_fuse` had a null
   fallback available during training more directly reshapes what
   downstream layers (and `self.blocks`'s own gradients, since fusion's
   loss still backprops through the whole model) come to rely on.

3. **Best single trained result of the whole K=32 family is post+no-null
   (2.4799)** — but it is also the LEAST robust to fusion being fully
   ablated (worst unconditional_pass1 of all four, 5.6564). A genuine
   score-vs-robustness trade-off, not a free win: whether "best trained
   val_bpb" or "graceful degradation under corrupted/missing coarser
   context" matters more for a given use case decides which cell is
   actually "best."

4. **None of these four numbers close the gap to `bytelm_xs1_ctx1024`
   (2.4870) or `bytelm_xs3_ctx1024` (2.4080)** — post+no-null (2.4799)
   is the only cell that beats `xs1`, and only by 0.007, well within
   run-to-run noise, and still loses clearly to `xs3`. Consistent with
   this session's overall finding: no `qcute_refine` lever tried,
   including this whole null-kv/fuse-position grid, has closed the gap
   to a properly matched dense baseline. The grid's real value was
   mechanistic (what null_kv and fuse_position actually do), not a
   competitive result in its own right.

## 11. v4.2's fully-unified head — genuine training instability, not just an efficiency tradeoff

`qcute_refine_v4_2_k32_narrow` (first real v4.2 training run — K=32/narrow-window, concat-only
fusion, `dq=8`, single shared `embed`/`ntp_head`/`code_pre` across EVERY level including byte,
`code_head_mode="independent"`): **best val_bpb 4.0369 @ step 3700**, full 4000-step run. This is
dramatically worse than every other K=32 config this session (2.48-2.60 range) or even
`bytelm_xs1_ctx1024`'s 2.4870 — a ~1.55 bpb gap far larger than the "independent-bit BCE is an
upper bound on true bits-per-byte" caveat alone could explain (that would predict a few tenths of
a bpb at most, not 1.5+).

**Root cause, confirmed by the per-level trajectory**: `val_level1_bpb_pass1` (the code level's
own loss, computed through the SAME shared head/embed byte level uses) never converges across the
full 4000-step run — min 0.556, max 5.700, mean 3.705, and critically the **standard deviation
over just the SECOND HALF of training (steps 2000-4000) is still 0.522** — no stabilization late
in training, genuinely persistent oscillation, not a slow-to-settle transient. Meanwhile
`val_level0_bpb_pass1` (byte level) DOES show a real if slow and noisy downward trend (6.03 →
4.94 → 4.66 → 4.38 → 4.27 → 4.20 → 4.07) — byte level is learning something, just far worse and
slower than every unshared config, while the code level's own loss is essentially thrashing the
entire time. Reading: forcing ONE head/embed to serve both byte-level prediction (256-way
discrimination over raw byte identity) and code-level prediction (a continuous BSQ code with very
different target statistics — already >85% bit-accuracy in every other config this session, a much
easier task) creates a genuine optimization conflict, not merely a capacity/efficiency tradeoff —
the shared head's gradient updates from one task actively destabilize the other, and never settle
into a shared solution good for both.

**Isolating what specifically causes it — the "does concat itself train fine" baseline requested
directly**: `qcute_refine_v4_k32_narrow_concat` (plain v4, UNSHARED weights, same K=32/narrow-window
architecture, same additive-loss training scheme, `fuse_position="concat"`) finished its full
4000-step run at **best val_bpb 2.4925** (final-step val_bpb 2.5661) — converged cleanly and
normally throughout, no oscillation in either level's own bpb trajectory, matching every other
unshared K=32 config this session (2.48-2.60 range). This directly confirms concat fusion itself
is not the problem — it trains exactly like `"pre"`/`"post"` fusion do; the instability is specific
to v4.2's head/embed unification, not the concat mechanism it happens to also use.

**Byte-vs-code task-incompatibility ablation — finished, and the result is more nuanced than the
step-2900 snapshot suggested.** `qcute_refine_v4_2_k32_narrow_byte256` (v4.2, `byte_head_256way=
True` — level 0 gets its OWN unshared exact 256-way softmax head, levels 1+ form a separate shared
pool of their own; TRUNK `self.blocks`/`ln_f` still shared across both levels) finished its full
4000-step run at **best val_bpb 2.5660 @ step 3900** (final-step val_bpb 2.5709) — squarely in the
healthy 2.48-2.60 range, a dramatic recovery from `v4_2_k32_narrow`'s 4.0369. Taken alone this looks
like task-incompatibility WAS the cause after all. But `val_level1_bpb_pass1` (the code level's own
loss) tells a different story: over the full run it never stabilizes — min 1.02, max 9.76, and the
**LAST quarter of training (steps 3000-4000) still has std 0.52, mean actually rising to 3.58** (the
run ends on an uptrend, not a plateau) — essentially the same persistent thrashing as the fully-
shared `v4_2_k32_narrow` baseline, not resolved at all.

**The reconciliation**: unsharing the byte head doesn't fix the code level's own instability — it
just fixes the FUSED byte-level bpb from being contaminated by it. In `v4_2_k32_narrow`, the SAME
head produces both byte and code predictions, so the code level's unstable gradients feed directly
into the weights level 0's byte prediction also depends on, dragging `val_bpb` up to 4.0369. In
`byte256`, level 0's head is fully private — the shared TRUNK still produces genuinely unstable code
predictions (level 1's own loss), but that instability no longer has a direct WEIGHT-SHARING path
into the byte-level readout, so `val_bpb` recovers to normal even though the underlying pathology in
the shared trunk/code-head pathway is still there, arguably undiminished. **This means the original
"byte-vs-code task incompatibility" and "shared trunk" hypotheses aren't actually competing
explanations — they compose**: the trunk-sharing (or something in the code level's own
shared-pool-of-one pathway) is the thing that's genuinely unstable, and byte/code head-sharing is
what let that instability LEAK into the metric that matters (fused val_bpb) in the original
`v4_2_k32_narrow` run. `qcute_refine_v4_1_k32_narrow_shared` (trunk-shared, but v4.1's OWN scheme —
head/embed never unified to begin with, so nothing FOR the code level's instability to leak through
even if it's still there) is now the run that answers the remaining precise question: is
`val_level1_bpb_pass1`'s own instability inherent to `shared=True` trunk-sharing itself (v4.1 would
show it too, just harmlessly, same as `byte256` shows it harmlessly), or is it specific to something
v4.2 adds beyond trunk-sharing (e.g. the level-1-owns-a-shared-pool-of-one construction, or an
interaction with `code_head_mode="independent"`) that v4.1 doesn't have at all? Not yet resolved —
`v4_1_k32_narrow_shared` hasn't started as of this writing.

`qcute_refine_v4_2_k32_narrow_ssm` (same architecture, `code_head_mode="chain"` +
`bit_head_class="ssm"` instead of `"independent"`) **killed early at step 1550, never recovering
from a near-total training collapse** — `byte_acc` stayed at 0.004-0.012 (near-random for a 256-way
task) and `val_bpb` stayed at 7.45-7.59 from step 100 through the kill point, no downward trend at
all, not merely oscillation around a mediocre value like the independent-head instability elsewhere
in this section. Also notably slow: **1.03 it/s vs. `byte256`'s 2.54 it/s**, a ~2.5x slowdown
consistent with `BitPredictHeadSSM`'s heavier per-step cost (flagged in this session's own earlier
efficiency review of that head). Whether this is genuine evidence that chain/SSM heads are
categorically worse under trunk-sharing, or a config/LR issue specific to this run/head, isn't
resolved — no baseline exists yet for `bit_head_class="ssm"` under v4/v4.1's UNSHARED scheme to
compare against. Two follow-up ablations queued to dig further:
**`qcute_refine_v4_2_k32_narrow_byte_softmax_head_only` — finished: partial recovery, not full.** A
NARROWER version of the `byte256` ablation above (session: "no byte embedding, assume byte as bits 0
to bits 255, only head is softmax"): level 0 keeps the SHARED dq-bit input embedding and `code_pre`
(unlike `byte256`, which unshared all three of embed/head/`code_pre` together), only its output
readout becomes an unshared 256-way head (`Config.byte_softmax_head_only`, new this session).
**Best val_bpb 2.7696** (final step 4000) — better than the fully-shared `v4_2_k32_narrow` (4.0369),
but a smaller recovery than `byte256`'s full unsharing (2.5660), and still outside the healthy
2.48-2.60 range every unshared config lands in. `val_level1_bpb_pass1` is just as unstable as ever
(last-quarter std 0.995, mean 4.64 — if anything slightly worse than `byte256`'s own last-quarter
figures). **Reading**: unsharing the OUTPUT head alone recovers PART of the gap, but unsharing
embed/`code_pre` too (as `byte256` does) recovers more — so the shared embed/`code_pre` isn't just
inert plumbing riding along with the head, it's independently contributing to how much of the
underlying code-level instability leaks into the byte-level metric. This refines last section's
"leak, don't cause" framing: it's not one single component leaking the instability, more of it leaks
through with more sharing, less leaks through with less — consistent with, not contradicting, the
trunk being the actual SOURCE either way.
**`qcute_refine_v4_2_k32_narrow_attn_id16` — finished: no collapse, meaningfully more stable, still
short of the healthy range.** Reclone of the (now-deleted, superseded) `qcute_refine_v4_2_k32_narrow_
ssm.py`, keeping `Ks`/`attn_window` UNCHANGED at `(32,32)` — swaps `bit_head_class="attn"`
(`BitPredictHeadAttn`) for `"ssm"`, and sets `bit_inner_downsample=16` (session: "reclone from
qcute_refine_v4_2_k32_narrow_ssm... find bitpredictattn to downsample bit embeds to 16x") — the "16"
in this file's name refers to that INNER chain-head-width downsample factor (`d_model=256 ->
256//16=16` working width inside the head), not to `Ks`. **Best val_bpb 3.6163 @ step 4000**, `byte_
acc` well above random throughout (no `ssm`-style collapse), and `val_level1_bpb_pass1`'s second-half
std is **0.333 — clearly more stable than every other shared-head config this session** (`ssm`
diverged outright; `v4_2_k32_narrow` 0.522; `byte256` 1.14; `byte_softmax_head_only` ~1.0). So
`BitPredictHeadAttn` (even this heavily downsampled) is a categorically better-behaved shared head
type than either the plain independent-bit head or `BitPredictHeadSSM` — but its fused bpb (3.6163)
is still well short of the 2.48-2.60 range every unshared/byte256/byte_softmax_head_only config
reaches, so "more stable" hasn't yet translated into "competitive." Also notably faster than `ssm`:
**1.79 it/s vs. `ssm`'s 1.03** (though still slower than the independent head's 2.54), consistent
with `bit_inner_downsample` doing real work.

**`qcute_refine_v4_2_k32_narrow_attn_id4` — killed early, genuinely DIVERGING, not just underfitting.**
Same architecture as `attn_id16`, `bit_inner_downsample=4` instead of 16 (`d_model=256 -> 256//4=64`
working width, vs. `id16`'s 16) — session: "repeat attn_id16 to clone and make it less aggressive
like x4," testing whether `id16`'s own aggressive downsampling was ITSELF limiting how much useful
chain-conditioning `BitPredictHeadAttn` could do. Result was the opposite of "more capacity helps":
`val_bpb` was actively RISING (best 5.3405 @ step 800 -> 5.6594 @ step 900, `byte_acc` stuck near
0.11-0.14) — killed at step 900, well before completion. Checked directly (session: "check does it
use pq or not"): confirmed NEITHER `code_embed_mode="pq_table"` nor any `quant_type` override was
set — plain defaults (`quant_type="bsq"`, `code_embed_mode="linear"`), identical to `id16`.

**`qcute_refine_v4_2_k32_narrow_attn_id4_pq` — `code_embed_mode="pq_table"` fixes the divergence
AND meaningfully improves both stability and fit.** Same as `attn_id4`, `code_embed_mode="pq_table"`
instead of the default `"linear"` (session: "try use pq and rerun") — treats the dq-bit BSQ code as
an INDEX into a `2**dq`-row `nn.Embedding` table instead of a linear combination of its `±1`
components (`CodeEmbed`'s own "dq is starved" hypothesis: a linear map over an 8-dim `±1` vector can
express only 8 additive directions regardless of `D`; at this config's scale, `linear` has rank ≤ 9
— `D×(dq+1) = 2,304` params — vs. `pq_table`'s full `min(256,D)=256` rank — `256×D = 65,536`
params, ~28x more effective degrees of freedom AND parameters). Finished cleanly at step 4000, no
divergence: **best val_bpb 3.2067**, `val_level1_bpb_pass1` second-half std **0.196** — even more
stable than `id16`'s already-good 0.333 — and train bpb ~2.3-2.6, meaningfully better fit than
`id16`'s ~2.8-3.5 (still short of the unshared cluster's ~1.6-2.0, but a real improvement on both
axes at once). **This is the single cleanest positive result in the whole `chain`-head family this
session**: the "dq is starved" hypothesis, previously only validated for the independent-bit head
(`docs/status.md`'s own `pq_table` ablation), extends directly to `BitPredictHeadAttn`'s shared
chain head too — the linear code-embedding bottleneck was silently capping BOTH capacity and
stability, not just capacity.

**Gap identified this session, not yet controlled for anywhere in the `ssm`/`attn_id16`/`attn_id4`
family: `code_head_mode="chain"` heads are SHARED across levels in v4.2 the exact same way the
independent-bit `ntp_head` is, but v4 (no sharing mechanism at all) never had to pay that cost.**
Session ask: "recheck how v4 BitPredictHeadAttn is more expressive vs v4.2." Checked directly — the
`BitPredictHeadAttn` CLASS itself is byte-identical between the two files (diffed; the only
difference anywhere in the class body is v4.2's precomputed `h_scale` buffer, a pure efficiency fix
from earlier this session, not a capacity change). The real difference is in how `LevelLM.__init__`
WIRES it: v4 has no `shared_head` concept at all — `build_bit_head` is called fresh for every
level, so each level gets its own PRIVATE `BitPredictHeadAttn` with dedicated weights. v4.2's
`LevelLM.__init__` calls `build_bit_head` exactly ONCE (only when `shared_head is None`, i.e. only
the pool owner); every other level does `self.ntp_head = shared_head.ntp_head` — the literal SAME
object. In `attn_id16`/`attn_id4`/the deleted `ssm` run, ONE `BitPredictHeadAttn` instance must
serve both level 0's byte-chain prediction AND level 1's code-chain prediction simultaneously, with
the same weights — a genuine multi-task capacity constraint v4's own (never directly compared)
per-level-private chain heads never faced. **This means the whole `ssm`/`attn_id16`/`attn_id4`
family confounds two separate questions that were never isolated: "is this chain head TYPE worse
than the plain independent-bit head" vs. "is a SHARED chain head worse than a PRIVATE one" — every
run in this family tests both at once.** No config this session gives any level a private
(unshared) chain head to serve as the missing control — worth queueing once the current family
reports back, to know whether `attn_id16`'s relatively good stability (§ above) reflects
`BitPredictHeadAttn` genuinely being a better-behaved shared head, or would be even better/different
again without the sharing burden at all.

**Side finding, resolves an open question from earlier this session**: `bytelm_xs1_ctx32`
(plain 1-layer dense bytelm, `context=32` — the genuine from-scratch single-task baseline for "how
good can a model with only 32 bytes of context get") reaches best val_bpb **2.8664 @ step 4000**
(still improving at the final step, not yet plateaued) — notably WORSE than the K=32 family's own
`level0_bpb_pass1` (unconditional/standalone) values of ~2.53-2.60 seen throughout this session's
probes. This resolves the "is qcute_refine's ~2.5-2.6 uncond bpb at window=32 normal" question
asked earlier: **no, it's better than what a genuinely single-task 32-context model achieves**,
confirming the earlier hedge was right to flag — `self.blocks`' good standalone performance in the
`qcute_refine` family reflects benefit from JOINT training with the fusion task (multi-task
regularization/shaping), not simply "32 bytes of context is already enough on its own." A plain
model given only that same 32-byte budget and nothing else reaches 2.87, meaningfully worse.
`bytelm_xs1_ctx8` reaches best val_bpb 3.357 @ step 2300 (and, unlike ctx32, already overfitting/
degrading past that point rather than still improving at step 4000) — consistent with the
session's earlier finding that 8 bytes is a harsher regime than 32.

**Level 0's uncond (standalone, no-fusion) bpb, at each run's own best-fused-checkpoint step, so
far:**

| run | fused val_bpb | level0 uncond bpb | @ step |
|---|---|---|---|
| `qcute_refine_v4_k32_narrow_postfuse_nonull_uncond` (post+no-null) | 2.4967 | 2.5768 | 2000 |
| `qcute_refine_v4_k32_narrow_nonull_uncond` (pre+no-null) | 2.4992 | **2.4782 (beats fused)** | 3500 |
| `qcute_refine_v4_k32_narrow_concat` (v4, unshared, finished) | 2.4925 (best), 2.5661 (final) | 2.5363 | 2500 |
| `qcute_refine_v4_2_k32_narrow` (v4.2, shared head, unstable) | 4.0369 | 4.0563 | 3700 |
| `qcute_refine_v4_2_k32_narrow_byte256` (v4.2, unshared byte head, shared trunk, recovers fused bpb, code level still unstable) | 2.5660 | — (not probed) | 3900 |
| `bytelm_xs1_ctx32` (genuine single-task, no fusion concept at all) | 2.8664 | — (n/a, no fusion) | 4000 |
| `bytelm_xs1_ctx8` (genuine single-task) | 3.357 | — (n/a) | 2300 |

Notable: `nonull_uncond`'s (pre) own uncond value (2.4782) is actually SLIGHTLY BETTER than its
own fused value (2.4992) at that checkpoint — level 0's standalone path briefly outperforms the
fused prediction, a real instance of the additive-loss scheme pushing standalone competence high
enough to occasionally beat the fused path itself, not just approach it. `k32_narrow_concat`'s own
uncond and fused values are nearly identical (2.5363 vs 2.5366) — fusion is barely adding anything
measurable at that checkpoint, consistent with the earlier "most of fusion's benefit is capacity,
not content" finding (§7-10) extended to the additive-loss regime. `v4_2_k32_narrow`'s uncond
(4.0563) tracks its own fused value (4.0369) closely too — the instability is shared-head-wide,
not something fusion is uniquely responsible for or masking.

**Is `nonull_uncond`'s fused-losing-to-uncond result (2.4992 vs. 2.4782) actually counter-
intuitive?** On its face yes — fusion gives level 0 strictly MORE information (a coarser code
built from a much larger receptive field), so a naive expectation is fused ≥ uncond always. Two
things resolve it, neither undermining the architecture:

1. **The gap (0.021 bpb) is small and likely just checkpoint noise, not fusion actively hurting.**
   In `"pre"` mode, `self.blocks` is the SAME shared weights serving both the PASS 1 (raw, unfused)
   input and the PASS 2 (fused) input — both loss terms backprop into it, so it has to generalize
   across two different input distributions at once. That's a small multi-task tension inherent to
   the additive-loss design itself (much milder than v4.2's full head-sharing pathology above, but
   the same flavor) — the two paths trading places by a hair at any single snapshot, rather than one
   strictly dominating throughout training, is expected, not alarming.

2. **The "32 bytes → 2.48 bpb" number itself isn't secretly free — already checked directly.**
   `bytelm_xs1_ctx32` (a genuinely SEPARATE model, trained ONLY on the 32-byte-window task, no
   fusion anywhere) reaches best val_bpb **2.8664** — clearly worse than `qcute_refine`'s own
   uncond 2.4782 at the identical 32-byte cap. So level 0's strong standalone number isn't "32
   bytes turns out to be plenty" — it's `self.blocks` being trained JOINTLY with the fusion task
   (even though fusion is switched OFF at this particular eval) that shapes it into a better
   standalone 32-byte model than training on the 32-byte task in isolation ever produces. A real,
   if slightly surprising, training-time regularization/shaping effect — "trained with X available"
   and "evaluated with X available" are different axes, and weights carry information from how they
   were trained even when an input channel is removed at inference time. Not a violation of "more
   information should help," just a reminder that the two questions aren't the same question.

## 12. v4.2's underfitting is dose-dependent on sharing degree — a train-side finding, not just val-side instability

Session question: "seems qcute variants underfitting generally vs baselines bpe and byte at step
4000." Checked directly by comparing TRAIN bpb (not just val) across every relevant run at step
~4000:

| run | train bpb | val bpb | sharing degree |
|---|---|---|---|
| `bytelm_xs3_ctx1024` (bigger byte baseline) | ~1.23-1.33 | ~2.70-2.77 | none (dense baseline, more capacity than xs1) |
| `bpelm_32768` (best-performing BPE baseline, step 8000) | ~0.006-0.009 | ~3.17-3.28 | none (BPE tokenization) |
| `bytelm_xs1_ctx1024` (baseline) | ~1.9-2.0 | ~2.55 | none (dense baseline) |
| `qcute_refine_v4_bpe4_imitate` | ~1.6-1.8 | ~2.55 | none (v4, unshared) |
| `qcute_refine_v4_k32_narrow_concat` | ~1.7-1.85 | ~2.55 | none (v4, unshared) |
| `qcute_refine_v4_k32_narrow_nonull_uncond` | ~1.6-1.75 | ~2.55 | none (v4, unshared) |
| `qcute_refine_v4_2_k32_narrow_byte256` | ~1.7-2.0 | ~2.6 | embed+head+`code_pre` unshared for byte only |
| `qcute_refine_v4_2_k32_narrow_byte_softmax_head_only` | ~2.0-2.5 | ~2.8 | only output head unshared for byte |
| `qcute_refine_v4_2_k32_narrow_attn_id16` | ~2.8-3.5 | ~3.6 | fully shared, tiny shared chain head |
| `qcute_refine_v4_2_k32_narrow` (original) | ~3.2-3.7 | ~4.1 | fully shared |

**Every unshared config — the genuine baselines AND plain v4 — bottoms out at train bpb ~1.6-2.0,
tightly clustered regardless of architecture family.** No v4.2 shared-pool variant gets anywhere
close, and the degree of degradation tracks the degree of sharing almost monotonically: `byte256`
(partial unshare — embed/head/`code_pre` all private for byte) matches the unshared cluster almost
exactly; `byte_softmax_head_only` (narrower unshare — only the head) is measurably worse; the fully
shared configs (`attn_id16`, `v4_2_k32_narrow`) are stuck 1-2 full bpb above the unshared cluster on
TRAIN data, not just val.

**The two new top-row baselines show the OPPOSITE failure mode, useful as a contrast.**
`bytelm_xs3_ctx1024` (more capacity than xs1) drives train bpb down to ~1.3 — lower than any unshared
qcute config — but its val bpb (~2.70-2.77) is actually WORSE than xs1's own 2.55: genuine
overfitting, more capacity buying better train fit but worse generalization, the mirror image of
v4.2's capacity-starved underfitting. `bpelm_32768` (the best of every BPE-tokenized baseline tried
this session — `bpelm_8192`/`bpelm_8192_converged`/`bpelm_4096_paramsmatch`/`bpelm_8192_ctx448_
flopsmatch_rope`/`bpelm_16384_ctx448_flopsmatch` all show the identical pattern) is far more extreme:
train bpb collapses to ~0.006-0.009 (essentially memorized) while val bpb sits at 3.17-5.02 across
every BPE config tried, all WORSE than every byte-level baseline and most qcute variants including
the underfit ones. **BPE tokenization at this corpus scale overfits catastrophically, categorically
worse than qcute's worst underfitting** — the two tokenization families fail in opposite directions,
and neither of this session's two new reference points is "the model to beat" in the way the tight
~1.6-2.0/~2.55 unshared cluster already is.

**Why this matters, distinct from everything else in this section**: §11's whole investigation has
been about `val_level1_bpb_pass1`'s STABILITY (does the code level's own loss ever converge) — a
val-side, optimization-dynamics framing. This is a different, complementary axis: even setting
convergence/stability aside entirely, the sharED models are failing to fit their OWN TRAINING data
as well as any unshared config does. That's a capacity story, not (only) an optimization-dynamics
story — consistent with, and probably a major contributor to, `attn_id16`'s specific finding of a
5,377-parameter shared chain head (§11) being asked to do byte-chain AND code-chain prediction at
once. The two framings aren't competing: a model can be BOTH capacity-starved (this section) AND
unstable in how what little capacity it has gets used (§11) — `attn_id16` shows meaningfully BETTER
stability than `byte256`/`v4_2_k32_narrow` (§11's own std numbers) while still being clearly the
most capacity-starved run in this table, so the two problems don't have to move together.

## 13. `quant_type="simplex"` needs stochastic exploration; `BitPredictHeadSSM` gains a per-position head

`qcute_refine_v4_2_k32_narrow_simplex` (the new `quant_type="simplex"` mode's own default-settings
run, `use_gumbel_noise=False`) was interrupted at step 1000 (queue reprioritization, not a kill-for-
cause) with `best_val_bpb` still actively improving (4.28 -> 4.12 -> 3.79 -> 3.71) but noisy —
`byte_acc` stuck low (~0.24-0.30) and bouncing rather than climbing steadily. Not runaway divergence
like `attn_id4`, but concerning enough (session: "simplex run loss and bpb train increasing" at one
point mid-run) to test the mode's OTHER lever before re-running the plain default further.

**Update: the deterministic default was fine all along — genuinely, monotonically converging, no
divergence anywhere.** Restarted from scratch multiple times by later queue reorderings (no
checkpoint-resume between runs), its best uninterrupted stretch reached step 2200 with `val_bpb`
declining cleanly the whole way: 69.6 (step 100) -> 4.39 (500) -> 3.73 (1000) -> 3.49 (1500) -> 3.30
(2000) -> **3.21 (2200)**, `byte_acc` climbing steadily alongside it (0.30 -> 0.38) — the earlier
"loss and bpb train increasing" observation was a local, checkpoint-level blip (visible in the
step-800 row of any given restart), not a real trend; the FULL trajectory across a longer
uninterrupted stretch shows no such pattern. Never finished a full 4000-step run (always cut short
for queue reordering, not for cause) — requeued to run to completion.

**`qcute_refine_v4_2_k32_narrow_simplex_gumbel` (`use_gumbel_noise=True`) — crashed for a real
reason, not queue reordering: `torch.AcceleratorError: scatter: index -1 is out of bounds for
dimension with size 256` at step ~829.** Root cause: `gumbel_quantize`'s stochastic branch
originally called `F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)` directly — that function's
OWN internal Gumbel sampling (`-log(-log(u))`, `u ~ Uniform(0,1)`) has no epsilon clamp. Over enough
steps x a 256-way softmax x however many codes per batch, `u` eventually underflows to exactly `0.0`
or `1.0` in float32, sending `-log(-log(u))` to `±inf`; two `inf`s landing in the same softmax row
produce `NaN`, and `NaN.argmax()` returned `-1` on MPS specifically (not necessarily on other
backends), which `F.one_hot` then can't scatter into a size-256 dimension — the observed crash.
**Fixed**: `gumbel_quantize` now samples Gumbel noise manually, clamping `u` to
`[torch.finfo(dtype).tiny, 1 - tiny]` before the double-log — the standard numerically-safe form,
which torch's own built-in simply doesn't apply. Verified directly: STE gradient still correct
(matches a plain-softmax gradient check, nonzero and finite on a non-degenerate loss), and a
20,000-iteration x `[256, 256]`-per-call stress test (deliberately wider-range logits than training
ever produces, to make the rare underflow event more likely to surface) produced **zero** non-finite
outputs, vs. reliably crashing within ~800 real training steps before the fix. Requeued (right after
the plain `simplex` run finishes) to test whether stochastic exploration in the quantization step
gives smoother/more monotonic convergence than the deterministic default, now that it can actually
complete.

**`BitPredictHeadSSM` gained a per-position head this session** (session: "edit bitpredictssm to let
each bit timestep use different head... similar to independent mode, but has state"). Previously
`self.head` was a single `nn.Linear(d_inner, 1)` broadcast identically across all `dq` bit
positions — every chained bit read the SAME output weights, unlike `code_head_mode="independent"`'s
per-bit-private `nn.Linear(D, dq)`. Now `self.head` is `nn.Linear(d_inner, dq)`: `_forward_fixed`
selects position `j`'s own logit via `torch.diagonal` (each position's hidden vector run through
every position's weight row, keep only the matching diagonal entry); `_forward_loop` slices
`self.head.weight[j:j+1]` directly at each step. Verified fixed/loop consistency still holds exactly
(`torch.allclose`, `atol=1e-5`) and every parameter still receives gradient. This gives
`BitPredictHeadSSM` a genuinely new point in the design space: PRIVATE per-bit weights (like
`"independent"`) combined with STATEFUL cross-bit conditioning (the alpha-decayed cumulative sum,
unchanged) — previously every chain head in this file was shared-weights-and-stateful, and
`"independent"` was private-weights-and-stateless; nothing sat at "private weights, stateful
conditioning" before.

**`qcute_refine_v4_2_k32_narrow_ssm_id1_pq` (full width, no downsample) killed for being too slow**
(session: "stop train too slow") — ~0.88 it/s at step 950, confirming the session's own compute
analysis directly: `BitPredictHeadSSM` at full `d_model=256` width costs ~272x more FLOPs per
bit-chain than a plain independent linear head (dominated by the decay-cumsum's `O(dq²·d_state)`
term, worse still with the just-added per-position head's own `O(dq²·d_inner)` term computed via a
wasteful full-matrix-then-diagonal — see the FLOPs table below). Config deleted, superseded.

**Three more changes to `BitPredictHeadSSM` this session, in response to that same analysis and
direct architectural feedback**, all reverified via fixed/loop-consistency (`torch.allclose`,
`atol=1e-5`) and full gradient checks (including the new `bos_state` parameter):
- **Einsum, not full-matrix+diagonal** (session: "use einsum") — `_forward_fixed`'s per-position
  head used to compute the entire `[N,dq,dq]` matrix via `self.head(fetched)` then discard every
  off-diagonal entry via `torch.diagonal` — an `n`x compute/memory waste. Now
  `torch.einsum("njd,jd->nj", fetched, self.head.weight) + self.head.bias`, computing only the
  diagonal directly. `_forward_loop`'s own per-step weight-row slice was already this cheap.
- **Concat, not add** (session: "make concat mode default... h_t always concat not add with
  current embed") — `fetched` used to be `h_scale*h + state_contrib` (summed into one `d_inner`-dim
  vector); now `torch.cat([h_scale*h, state_contrib], dim=-1)` (`2*d_inner`-dim, `self.head`'s input
  width doubles accordingly) — strictly more information available to the head (summing is a lossy
  special case concat can still learn to approximate).
- **Trainable BOS state** (session: "consider a trainable bos token init zero at dq 0") — position
  0's `state_contrib` used to be `state_proj(zeros)` (an accident of that layer's own bias init, not
  a deliberately chosen "no state yet" representation); now `self.bos_state` (zero-initialized
  `nn.Parameter`) stands in directly, free to move away from the old zero-equivalent behavior.
- The recurrence itself (alpha-decayed cumulative sum over past bits) is UNCHANGED — session:
  "retain cumsum with decay but each timestep must concat with original h_t."

**`qcute_refine_v4_2_k32_narrow_ssm_id4_pq_concat`** (session: "for this better downsample or make
embeds smaller dim, like 4x" — `bit_inner_downsample=4`, matching `attn_id4`'s own working width,
`code_embed_mode="pq_table"` carried over) replaces the killed full-width run, queued to the front.
Confirmed meaningfully faster already: **~1.85 it/s at step 50**, faster than both the killed
`downsample=1` run (~0.88 it/s) and `attn_id4_pq` itself (1.38 it/s).

## 14. `BitPredictHeadAttn` revamped to match — all three `attn_id*` ablation configs deleted, superseded

Session: "delete all attn 4.2 ablation, need revamp." `attn_id16`/`attn_id4`/`attn_id4_pq` configs
deleted — their own results (§11/§13 above) still stand as historical record, but the module
they trained is gone, replaced by a substantially revamped `BitPredictHeadAttn`, mirroring
`BitPredictHeadSSM`'s own revamp (§13) plus one further architectural simplification. All changes
reverified via fixed/loop-consistency (`torch.allclose`, `atol=2.4e-7`) and full gradient checks.

- **Per-position head, concat input** (session: "use indp head each timestep... use concat style
  like ssm") — same change as `BitPredictHeadSSM`: `self.head` is `nn.Linear(2*d_inner, dq)`, one
  weight row per bit position, read via `torch.einsum("njd,jd->nj", fetched, self.head.weight)`
  (no full-matrix-then-diagonal waste); `fetched = torch.cat([h_scaled, attn_out], dim=-1)`, not
  summed.
- **No BOS parameter — `h_t` itself is the concat BOS** (session: "use h_t as bos token, remove bos
  param... h_t is the concat bos"). The trainable `bos_val_emb` added earlier this session (in
  response to "is it like bos token") was REMOVED again — position 0's "previous bit" slot is a
  plain zero vector once more, but this is now sound: since `h_t` is CONCATENATED (never summed)
  into `fetched`, position 0 still receives a distinct, non-vanishing signal from `h_t` regardless
  of what the attention half contributes — concatenation can't blend/erase it the way addition
  could, so no separate learned placeholder is needed. Verified directly: perturbing `h` changes
  position 0's output even with the zero placeholder in place.
- **Q/K-only attention, no V or `out_proj`** (session: "simplify _mha, comment current one, make new
  _mha, remove out proj, v proj, only k and q proj, basically like weighted sum of h and embeds") —
  `self.qkv_proj`/`self.out_proj` replaced by `self.q_proj`/`self.k_proj` alone; attention weights
  are still learned, but what gets weighted-summed is the RAW (unprojected) bit-value embeddings
  themselves — a genuine soft causal average over actual embedding content, not a learned value
  transform of it. Old implementation kept as a commented-out `_mha_old_full_qkv` for reference,
  never called.

**Compute analysis** (session: "analyze how much compute savings with this," then "and params," then
"how much flops vs softmax 256 vs indp 8," then "merge flops param count") — `D=256`, `dq=8`
throughout; `softmax-256` (`byte_head_256way`'s own `nn.Linear(D,256)`) and `indp-8`
(`code_head_mode="independent"`'s `nn.Linear(D,8)`) have no downsample knob, always full-`D`:

| head | params | vs indp-8 | vs softmax-256 | FLOPs | vs indp-8 | vs softmax-256 |
|---|---|---|---|---|---|---|
| `indp-8` (`D→8`) | 2,056 | 1x | 0.03x | 4,096 | 1x | 0.03x |
| `softmax-256` (`D→256`) | 65,792 | 32x | 1x | 131,072 | 32x | 1x |
| **attn**, `downsample=16` | 5,080 | 2.5x | 0.077x | 12,800 | 3.1x | **0.098x** |
| **attn**, `downsample=4` | 26,440 | 12.9x | 0.40x | 149,504 | 36.5x | 1.14x |
| **attn**, `downsample=1` | 138,248 | 67.2x | 2.1x | 2,170,880 | 530x | 16.6x |

Versus the PRE-revamp `attn` head (shared single-output head, full QKV+out_proj — the version
`attn_id16`/`attn_id4`/`attn_id4_pq` actually trained): params drop 48%/22%/6% (downsample
1/4/16) and FLOPs drop to ~51%/52%/57% of the old cost (~2x speedup), since removing `out_proj` and
`V` cuts the projection cost from `~4·d²` (QKV's 3 + out_proj) to `~2·d²` (Q and K only) — the
dominant term whenever `D` isn't tiny relative to `dq`.

**Notable finding from the merged table**: `attn_id16` (downsample=16) now beats a plain 256-way
softmax classifier on BOTH axes at once — ~13x fewer params, ~10x fewer FLOPs — while `attn_id4`
(downsample=4) undercuts `softmax-256` on params (0.40x) at roughly FLOP-parity (1.14x). Only the
full-width (`downsample=1`) config is more expensive than `softmax-256` on both axes. No training
run yet on the revamped module as of this writing — the next `attn_id*`-family config should target
`downsample=4` or `16` given this.

## 15. `attn_id1_pq` killed for speed; `BitPredictHeadAttn` gains a trainable BOS embed; `simplex_l2` tests trunk depth as a substitute for per-level privacy

**`attn_id1_pq` (full-width, revamped `BitPredictHeadAttn`) killed — 0.57 it/s, actually
*slower* than the `ssm_id1_pq` run killed earlier this session for being too slow (0.88 it/s)**,
despite the revamp's own §14 FLOP savings. At 0.57 it/s the full 4000-step run would have taken
~2 hours, first of 8 queued jobs — not worth it at full width; config kept for a future `downsample`
requeue rather than deleted outright.

**`BitPredictHeadAttn` gains a trainable BOS embed for `_mha`'s own position-0 slot** (session:
"make zero_vec trainable embeds") — NOT a reversal of §14's "no BOS parameter, `h_t` is the concat
BOS" decision, which is about a different role entirely. §14's removed `bos_val_emb` stood in for
the HEAD's own missing signal at position 0 (moot once `h_t` reaches `self.head` via concat, since
concat can't blend/erase `h_t`'s contribution the way addition could). This new `self.bos_val_emb`
(`nn.Parameter(torch.zeros(d_inner))`) is internal to `_mha` itself: position 0's "previous bit"
slot in the attention key/value sequence was still a plain unadorned zero, giving attention no way
to distinguish "no previous bit exists yet" from "the previous bit's embedding happened to be near
zero." Replaces `zero_vec = val_embeds.new_zeros(...)` in both `_forward_fixed` and `_forward_loop`.
Verified: `torch.allclose(fixed, loop, atol=1e-5)` (max diff 1.19e-7) and `bos_val_emb.grad` nonzero
after `.backward()`.

New config `configs/qcute_refine_v4_2_k32_narrow_attn_id4_pq.py` — same family as
`ssm_id4_pq_concat` (`bit_inner_downsample=4`, `code_embed_mode="pq_table"`, everything else
matching the `k32_narrow` defaults), on the new `bos_val_emb`-equipped head. Queued behind
`simplex_l2` (below).

**`simplex_l2`** (`configs/qcute_refine_v4_2_k32_narrow_simplex_l2.py`) — clone of
`qcute_refine_v4_2_k32_narrow_simplex.py` (§13), `n_layers=2` instead of 1 (session: "queue simplex
(code_bits=8) with double layer, at front"). Tests whether trunk depth can compensate for
`quant_type="simplex"`'s own §13 finding: at the default `code_bits=8`, byte level 0 and every code
level literally share ONE `nn.Embedding(256,D)` object — weight-tied as both input embed and output
classifier at every level simultaneously (`LevelLM.__init__`, `qcute_refine_v4_2.py:1420-1433`) —
the most extreme sharing point in the whole v4.2 family, and (per §12's dose-dependent-sharing
finding) the least private per-level capacity of any variant tried. Doubling `n_layers` adds trunk
capacity without touching that sharing scheme at all — separates "does the shared trunk need more
depth" from "does the byte level need its own private embed/head" as two independently-testable
bottlenecks.

**Measured params/FLOPs** (`torch.utils.flop_counter.FlopCounterMode`, CPU, batch=1,
`context_len=1024`, matching every `k32_narrow` config's own context):

| model | params | flops/fwd |
|---|---|---|
| `simplex` (n_layers=1) | 0.921M | 3573M |
| **`simplex_l2` (n_layers=2)** | **1.709M** | **6857M** |
| `byte256` | 0.995M | 3557M |
| `byte_softmax_head_only` | 0.927M | 3565M |
| `v4_2_k32_narrow` (fully shared baseline) | 0.861M | 3306M |
| `bytelm_xs1_ctx1024` (1-layer) | 1.050M | 2147M |
| `bytelm_xs3_ctx1024` (3-layer) | 2.625M | 5369M |

Doubling `n_layers` roughly doubles both params (0.921M→1.709M) and FLOPs (3573M→6857M) relative to
`simplex`, as expected. Against the `bytelm` baselines specifically: **params land closest to
`bytelm_xs1_ctx1024`** (1.709M vs. 1.050M, +0.66M away — closer than to `xs3`'s 2.625M, +0.92M
away), but **FLOPs land closest to `bytelm_xs3_ctx1024`** (6857M vs. 5369M, +1.49G away) — and
actually *exceed* the 3-layer bytelm's own FLOPs, despite `simplex_l2` having fewer params than
it (1.709M vs 2.625M). The two-pass PASS1+PASS2 fusion (§7-10) plus cross-level trunk-sharing
multiply per-layer cost faster than plain depth does in a single-tower transformer — `n_layers=2`
here isn't "2x one bytelm layer," it's closer to paying `xs3`'s 3-layer compute bill while carrying
`xs1`'s 1-layer parameter budget. Honest bar for this run once finished: beating `bytelm_xs1_ctx1024`'s
2.4870 best_val_bpb would be a modest result given the FLOP spend already exceeds `xs3`'s
(best_val_bpb 2.4078) — the real comparison point on a FLOPs-matched basis is `xs3`, not `xs1`.

Queue as of this writing: `simplex_gumbel` (running) → `simplex_l2` → `attn_id4_pq` → `simplex`
(fresh full run) → `ssm_id1_pq` (full-width, revamped) → `bpe4_imitate_uncond` →
`bpe4_imitate_uncond_l1x4` → `v4_k32_narrow_both` → `v4_1_k32_narrow_shared` (KEY, isolates
trunk-sharing alone vs. v4.2's own additions).

## 16. Queue clears — `simplex_l2` near-ties `byte256`, revamped `attn_id4_pq` regresses, `v4_k32_narrow_both` nearly matches `bytelm_xs3`

Every job queued in §15 ran to completion (results below). Only `v4_1_k32_narrow_shared` (§10's
long-deferred KEY isolator) remains, now running.

**`simplex_l2` (n_layers=2) — best_val_bpb 2.5892, essentially tying `byte256` for best v4.2 result
of the session** (2.5660, 0.023 apart — within noise). Confirms §15's own hypothesis: trunk depth
IS a real substitute for `simplex`'s missing per-level embed/head privacy, at least partially — a
2x deeper shared trunk closes nearly the entire gap to the config that instead gives the byte level
its own fully-private embed+head pair. Doesn't resolve which lever is "better" per FLOP (§15 already
showed `simplex_l2` spends more FLOPs than even `bytelm_xs3_ctx1024`), but it's a genuine second
path to the same destination.

**`simplex` (clean full rerun after the queue-leak contamination) — best_val_bpb 2.8687**, far
better than the earlier interrupted/contaminated read (3.7105, §13). Retroactively confirms §13's
own correction (the plain-simplex run was never actually unstable) even more strongly — under a
clean, uncontended run it lands closer to `byte_softmax_head_only` (2.7696) than to anything in the
"broken" tier.

**`simplex_gumbel` — finished at best_val_bpb 2.9443** (step 4000), extending its earlier
trajectory (3.145→3.058 mid-run, §15) to a respectable finish, though still behind the
non-stochastic `simplex` rerun (2.8687) — the extra Gumbel-noise exploration doesn't pay for itself
at this step budget, on this data.

**`attn_id4_pq` (revamped, with the new `bos_val_emb`) — best_val_bpb 3.5659, WORSE than the OLD
pre-revamp `attn_id4_pq`'s 3.2067**, despite the revamp's own §14 FLOP/param savings and despite
`bos_val_emb`'s theoretical motivation (giving `_mha` a genuine start-of-chain signal instead of an
overloaded zero). This is a real regression, not noise-level — worth flagging plainly: "cheaper and
more principled" did not translate to "better" here. Candidate explanations, untested: per-position
(private) weights at `downsample=4` may have less data per position to fit than the old shared
single head did; or Q/K-only attention (no learned V) may be a genuine expressivity cut relative to
full QKV+out_proj that the FLOP savings don't compensate for. Not chased further this session —
noted as an open question for whoever revisits `BitPredictHeadAttn` next.

**`ssm_id1_pq` (revamped, full-width) — actually completed this time** (unlike the pre-revamp
version killed in §13 for being too slow) — 6944s (~1h55m) for 4000 steps, ~0.58 it/s, essentially
the same throughput class as the `attn_id1_pq` that got killed in §15 for being "too slow." Finished
at best_val_bpb 3.7708 — the weakest of every `simplex`/`attn`/`ssm` variant finished this session,
confirming full-width chain heads (of either flavor) just aren't a good use of compute at this
scale regardless of the per-position/concat/einsum revamp.

**v4-lineage (not v4.2) results, all strong**:

| config | fuse_position / tiers | best_val_bpb | params |
|---|---|---|---|
| `qcute_refine_v4_bpe4_imitate_uncond` | tier_n_layers=(1,2) | 2.5533 | 3.363M |
| `qcute_refine_v4_bpe4_imitate_uncond_l1x4` | tier_n_layers=(1,4) | 2.5493 | 4.941M |
| `qcute_refine_v4_k32_narrow_concat` | fuse_position="concat" | 2.4925 | — |
| **`qcute_refine_v4_k32_narrow_both`** | fuse_position="both" | **2.4443** | 3.430M |

**`v4_k32_narrow_both` (`fuse_position="both"` — both pre- and post-self-attention `CrossBlock`
cross-attention modules, more params than either "pre" or "post" alone) is the best v4/v4.2-lineage
result of the ENTIRE session** — 2.4443 best_val_bpb, nearly matching `bytelm_xs3_ctx1024`'s
2.4078 (a 3-layer single-tower baseline) despite `v4_k32_narrow_both` using only `tier_n_layers=
(1,1)` — one self-attention layer per tier. Strongest evidence yet that fusion capacity (running
BOTH cross-attention placements rather than picking one) buys real quality, not just redundant
compute — consistent with, and extending, §7-10's "fusion's benefit is capacity, not [just]
content" finding.

**Updated final `qcute_refine_v4_2_*` ranking** (best_val_bpb, ascending):

| rank | run | best_val_bpb |
|---|---|---|
| 1 | `byte256` | 2.5660 |
| 2 | `simplex_l2` (NEW) | 2.5892 |
| 3 | `byte_softmax_head_only` | 2.7696 |
| 4 | `simplex` (clean rerun) | 2.8687 |
| 5 | `simplex_gumbel` | 2.9443 |
| 6 | `attn_id4_pq` (OLD, pre-revamp, deleted config) | 3.2067 |
| 7 | `attn_id4_pq` (NEW, revamped) | 3.5659 |
| 8 | `attn_id16` | 3.6163 |
| 9 | `ssm_id4_pq_concat` | 3.7569 |
| 10 | `ssm_id1_pq` (revamped, full-width) | 3.7708 |
| 11 | `v4_2_k32_narrow` (original, fully shared) | 4.0369 |
| 12 | `ssm_id1_pq` (OLD, pre-revamp, killed) | 4.3298 |
| 13 | `attn_id4` (diverged, pre-pq) | 5.3405 |
| 14 | `ssm` (collapsed) | 7.5483 |

`byte256` and `simplex_l2` are now a genuine near-tie at the top — two structurally very different
levers (private embed+head vs. deeper shared trunk) landing at essentially the same result.

`v4_1_k32_narrow_shared` (the KEY run isolating whether trunk-sharing ALONE, without any of
v4.2's other extreme-sharing additions, causes the instability documented in §11) is now running —
the last item in the queue, no downstream jobs after it.

## 17. `v4_1_k32_narrow_shared` finishes — trunk-sharing alone is NOT the instability's cause; plus structured-softmax heads and a depthwise `BitPredictHeadConv`

**`v4_1_k32_narrow_shared` finished at best_val_bpb 2.5254** — close to `byte256`/`simplex_l2`
(2.5660/2.5892) and far better than the fully-shared `qcute_refine_v4_2_k32_narrow` baseline
(4.0369). This answers §11-§16's long-open question directly: **trunk-sharing alone (v4.1's own
"shared" mode — one trunk reused across every level, nothing else unshared) is NOT what causes
v4.2's instability/underfitting.** v4.2 additionally shares the embed table, the NTP head, AND
`code_pre` across every level (byte included) — it's specifically THAT additional layer of sharing,
not trunk-sharing per se, that produces the val-side instability and underfitting documented
throughout §11-§12. Every v4.2 config that partially or fully unshares embed/head (`byte256`,
`byte_softmax_head_only`, `simplex_l2`) recovers most or all of the gap to `v4_1`'s own healthy
result — consistent, convergent evidence across the whole session's ablation family.

**Structured/cheap replacements for a dense `V`-way softmax classifier** (session: "use structured
matrix but to replace dense linear map to 2**n way output softmax... some loss in repr ok for
params saving"):

- **`FactoredSoftmaxHead`** (new) — outer-sum over two small projections: `logits[i,j] =
  f1(h)[i] + f2(h)[j]`, `v1*v2==vocab`. Trivially valid (an ordinary softmax over a structured
  logit vector, no chain-rule/teacher-forcing needed — parallel/one-shot, unlike every
  BitPredictHead*). At `vocab=256, D=256, v1=v2=16`: params `8,224` vs. dense's `65,792` (8x
  fewer), FLOPs `16,384` vs. `131,072` (8x fewer). Cost: the reshaped `[v1,v2]` logit matrix is
  additively separable (row-effect + column-effect only) — the effective `V×D` weight matrix has
  rank `≤v1+v2`.
- **`LowRankSoftmaxHead`** (new, added after session asked "how good is factoredsoftmax vs just low
  rank... analyze rank") — the classic softmax bottleneck (Yang et al. 2018): `h -> Linear(D,rank)
  -> Linear(rank,vocab)`. At matched budget (`rank=16`, same param/FLOP order as factored above:
  `8,464`/`16,384`), **theoretically STRICTLY more expressive than `FactoredSoftmaxHead`**: every
  outer-sum-representable logit matrix is also representable by low-rank at rank `v1+v2` (a
  degenerate, zero-free-coefficient special case), but low-rank additionally gives every one of the
  `V` classes its own FREE `rank`-dim coefficient vector, while factored's classes get NO per-class
  freedom at all beyond a discrete row/column slot in a rigid `w1_i+w2_j` template — a strictly
  narrower function class at the same rank ceiling. Queued as `byte_factored`/`byte_lowrank`
  (both cloned from `byte_softmax_head_only`'s narrow ablation shape — embed/code_pre stay shared,
  only the readout is private/structured) for a direct, budget-controlled empirical test of this
  rank argument.
- Real butterfly/Monarch matrices (the general structured-matrix family both of the above are
  shallow/2-stage special cases of) would push params/FLOPs down further (`O(log V)`/`O(√V)`
  stages instead of 2) at real implementation-correctness risk not attempted this session — noted
  as a documented future direction, not built.
- Compositional bit-path codes (small per-depth/per-bit embeddings summed along the tree path,
  discussed as an alternative to `BitPredictHeadHSoftmax`'s own `O(V)` per-node table) were
  proposed but not implemented — same reasoning, deferred as a documented direction.

**`BitPredictHeadHSoftmax`** (new, from the earlier "find something that satisfies chain probs
validity and cheap and same repr power as large softmax head" ask) — classic hierarchical softmax
over the same `dq`-depth binary tree every BitPredictHead* factorizes, but unlike attn/conv/ssm
(which reuse ONE classifying direction per bit POSITION, shared across every prefix reaching that
position — the diagnosed geometric bottleneck behind why those heads underperform, even
pre-revamp), gives every one of the `2**dq-1` tree NODES its own private weight vector. Verified:
fixed/loop consistency (`atol=5.7e-6`), gradients, full-model smoke test, `validate_generation`
parity — all clean. Measured (not estimated) params/FLOPs vs. dense `softmax(2**dq)`:

| dq | V | hsoftmax params | hsoftmax FLOPs | dense softmax params | dense softmax FLOPs | FLOP ratio |
|---|---|---|---|---|---|---|
| 8 | 256 | 65,535 | 4,224 | 65,536 | 131,072 | 31x |
| 13 | 8,192 | 2,105,087 | 6,994 | 2,097,152 | 4,194,304 | 600x |
| 16 | 65,536 | 16,842,495 | 8,704 | 16,777,216 | 33,554,432 | 3,855x |

Params stay tied to dense softmax at every scale (both `O(V·D)`); FLOPs diverge explosively in
hsoftmax's favor (`O(log V)` vs `O(V)`). Real caveats before scaling `dq` up in this codebase: (1)
params don't shrink, so `dq=16` alone costs ~16.8M params for this one head, dwarfing every model
in this session's ablation family; (2) the tree is plain heap-indexed (not usage-weighted like
classic Huffman-tree hierarchical softmax), so at `dq=13/16` with `enwik8_1M`'s modest training set
(further shrunk by `K`-downsampling at coarser levels), many of the deep node weights risk being
under-trained from sheer data scarcity. Not run this session — `dq=8` (matching every existing
config) was judged the safer test; `hsoftmax` wired into `Config.bit_head_class`/`build_bit_head`/
CLI, ready for a `dq=8` config whenever queued.

**`BitPredictHeadConv` depthwise fix** (session: "consider making bitpredictconv more efficient,
last time huge compute, maybe try group conv or depthwise") — both existing `conv_impl` options
("conv1d"/"matmul") are FULLY DENSE across channels: every output channel reads every input channel
at every window position, costing `K*d_inner^2` params/FLOPs. At full width (`d_model=256,
kernel_size=dq=8`, measured via FlopCounterMode): **525,313 params / 8,392,704 FLOPs for this one
head** — the actual "huge compute" being referenced, and the reason every prior `conv`-family
config this session used `bit_inner_downsample>1`, never full width. New `conv_impl="depthwise"`:
each channel gets its own private `K`-tap filter (no cross-channel mixing at all), implemented via
plain `einsum` rather than `nn.Conv1d` (preserving "matmul"'s own loop-overhead-avoidance property
for the sequential `_forward_loop` decode path) — **3,073 params / 36,864 FLOPs at the same full
width: a 171x/228x reduction.** Verified: fixed/loop consistency (exact match, max diff 0.0),
gradients, full-model smoke test, `validate_generation` parity. Cheap enough to finally test `conv`
at full width for the first time this session (`configs/qcute_refine_v4_2_k32_narrow_conv_
depthwise.py`, `code_embed_mode="pq_table"` carried over from the `attn_id4` divergence fix).

**`BitPredictHeadConvDilated`** (new, session: "do this stacked small kernel, then check memory
usage vs single large conv") — a WaveNet-style dilated depthwise-separable causal conv STACK
(kernel=`dilation_base`, dilation=`dilation_base^l` per layer `l`, `L=ceil(log_b(dq))` layers deep)
as a further compute lever beyond the single-layer depthwise fix above. PURELY LINEAR, no
activation between stacked layers (session: "i mean for memory and param save even though
linear") — composing linear filters stays linear, so this is representationally a *subset* of what
a single full-width kernel can express (a real expressivity cost flagged explicitly, not a free
win); it exists to test the params/FLOPs/wallclock side of the tradeoff in isolation.
Initially TRAINING-ONLY (session: "no need ar gen yet for this stacked") — later completed (session:
"then check ar gen conv code, then train this"): `_forward_loop` reuses the same `_dilated_stack`
helper `_forward_fixed` calls, recomputed on the growing bit-history each step (no WaveNet FIFO
cache needed — the "just recompute the window read" tradeoff `BitPredictHeadConv`'s own
`_forward_loop` already makes, cheap enough at `dq=8`). Verified fixed/loop consistency (exact/
near-exact match, both `depthwise` and `dense` modes), a standalone greedy-decode smoke test, full
`RefineLM` integration (forward+backward clean, no missing grads), and `validate_generation` parity
(`generate_no_cache` vs. `generate_kv_cache` exact match) — all clean. Now wired into `Config.
bit_head_class="conv_dilated"`/`build_bit_head`/CLI (`conv_dilated_base`, `conv_dilated_mode`), and
queued for training (`configs/qcute_refine_v4_2_k32_narrow_conv_dilated.py`, `mode="depthwise"`,
`code_embed_mode="pq_table"`).

Also finished (session: "finish impl conv dilated dense", "make it like conv1d groups=1, dense"): a
`mode="dense"` variant with full cross-channel mixing per layer (weight `[D_out,D_in,K]`, the
dilated-stack analogue of `BitPredictHeadConv`'s own dense `"matmul"` — same stack structure, just
without the per-channel restriction). Its `unfold`+`einsum` computation was verified to EXACTLY
reproduce a real `nn.Conv1d(groups=1)` stack (weights copied over layer-by-layer, `torch.allclose`
exact) — confirms the einsum correctly implements standard dense dilated-conv semantics. At
`dq=8, d_model=256, dilation_base=2`: `394,753` params / `100,728,832` FLOPs — a genuine ~25%
reduction vs. the single big dense kernel's `525,313`/`134,283,264` (matching the tap-count math:
`3*2=6` "dense tap-layers" vs. `8`), but wallclock (`0.47ms/fwd`) is actually SLOWER than the
single-layer dense kernel's own `0.30ms/fwd` — per-layer overhead (3 separate `unfold`+`einsum`
calls, intermediate allocations) outweighs the FLOP savings at this small scale, a genuinely
different result from `depthwise` mode (which won on wallclock too). Not currently queued for
training (the `depthwise` mode above is the config actually running) — a real, honest data point
that "fewer FLOPs" and "faster wallclock" don't always move together at small scale.

At `dq=8, d_model=256, dilation_base=2` (3 layers, dilations 1/2/4, receptive field exactly 8,
verified via a causality check — flipping bit 5 leaves logits 0-4 unchanged but changes logit 6):
params are a near-wash vs. single-layer `depthwise` (`3,073` — identical to single-layer at
`b=2`, since 3 layers' extra bias terms exactly cancel the tap savings at this small scale;
`dilation_base=3` does slightly better via fewer layers: `2,817`), while FLOPs are modestly better
(`28,672` vs. `36,864`, 22% fewer) — both confirm the earlier prediction that dilation's real
payoff is at larger `dq` (16+), not `dq=8`.

**A real implementation pitfall surfaced and fixed along the way, independent of dilation itself**:
the first implementation used `nn.Conv1d` directly for each layer (reasoned, at the time, that its
per-call overhead only mattered inside `_forward_loop`'s sequential decode — not here, since this
path runs once per training step). Measured wallclock (CPU, `_forward_fixed`, batch=16) showed this
was wrong: **298ms/fwd** — ~300x slower than a plain single-layer `depthwise` (0.96ms). A diagnostic
(session: "not dilated depthwise, but dilated full dense kernel") — swapping `groups=d_inner` for
`groups=1` (fully dense per layer) while keeping everything else identical — dropped this to
21.3ms, isolating the true cause as `nn.Conv1d`'s own per-call dispatch overhead on this backend
(present regardless of grouping), not the multi-layer/dilation structure itself. This is the exact
same issue `BitPredictHeadConv`'s own `"matmul"`/`"depthwise"` impls were already built to dodge
via `unfold`+`einsum` instead of `nn.Conv1d` — applying that same fix to the dilated stack (`torch.
Tensor` weights, `unfold`+`einsum` per layer, no `nn.Conv1d` anywhere) brought it down to
**0.53ms/fwd — the FASTEST of every chain-head variant benchmarked** (dense `matmul` 0.64ms,
single-layer `depthwise` 0.96ms, revamped `BitPredictHeadAttn` 1.34ms), confirming this session's
own long-standing "avoid `nn.Conv1d`, even in the parallel/batched path" lesson applies more
broadly than just the original sequential-decode-loop case it was first diagnosed for. Reverified
fixed/loop-consistency-equivalent checks (causality, gradients) after the rewrite — all clean.

Independent of that implementation detail, the general dilation-vs-depth math (total taps scale
`~b*log_b(dq)` vs. a single layer's `dq`, minimized near `b=e≈2.7`) still holds and matters more at
larger `dq`: at `dq=16`, `b=2` gives `8` taps vs. a single layer's `16` (2x fewer) — real savings,
just not yet visible at this session's actual `dq=8` configs. A NON-dilated deep stack (`L≈dq`
plain kernel-2 layers, linear receptive-field growth) remains strictly worse on both params
(`~2*dq` taps) and sequential depth than either the single-kernel baseline or the dilated version —
confirmed by the same math, not separately re-verified in code.

## 18. `downsample` decoupled from `h`'s own dimension (Attn/SSM); `BitPredictHeadAttn` reverted to v4's original design

Session: "for each bitpredict* can the downsample has flag only be applied on embeds, h maintains
full dim." Every `BitPredictHead*`'s `downsample>1` previously ran an `in_proj: nn.Linear(d_model,
d_inner)` on `h` itself before anything else — shrinking `h`'s own information, not just the
embed/attention/state machinery's width. Feasibility check per class, by mechanism:

- **Attn, SSM (concat-based)** — `h` only ever reaches the head via `torch.cat([h_scaled, ...])`,
  which doesn't require matching dims. **Easy**: drop `in_proj`, keep `h` at full `d_model` in the
  concat, resize `self.head`'s input from `2*d_inner` to `d_model+d_inner`. Implemented and
  verified (fixed/loop consistency, gradients, full-model integration, `validate_generation`
  parity — both `downsample=1` and `downsample=4`) for both classes.
- **Conv, ConvDilated (add-based)** — `fetched = h_scale*h + conv_out` REQUIRES matching dims.
  **Medium**: would need converting add->concat first (the same "concat is strictly more
  information than add" argument already used for Attn/SSM's own earlier revamp). Not implemented
  this pass — noted as the next step if these two heads need the same treatment.
- **HSoftmax (dot-product-based)** — `h @ node_weight[node_idx]` is a genuine inner product,
  mathematically REQUIRING `h` and `node_weight` to share a dimension. **Not possible** to decouple
  without reintroducing a projection (i.e. `in_proj` under a different name) — this is a hard
  constraint of the mechanism, not an engineering gap. `BitPredictHeadHSoftmax`'s own earlier
  finding (§17: downsampling INCREASES its FLOPs/wallclock, since `in_proj`'s own cost dominates
  its otherwise-tiny node-read) is the concrete symptom of this same constraint.

**`BitPredictHeadAttn` reverted to v4's original design** while implementing the above (session:
"for attn, comment current impl, revert to v4") — the session's own earlier revamp (concat/
per-position head/Q-K-only attention/trainable `bos_val_emb`, §14) was found to REGRESS empirically
(`attn_id4_pq` on the revamped head: 3.5659 best_val_bpb, worse than the original's 3.2067 — §16/
§17), so rather than retrofit the h-decoupling onto a design already known to underperform, this
reverts to v4's original mechanism: full QKV self-attention + `out_proj`, a single SHARED head
(not per-position). The revamped implementation is preserved as a commented-out reference block in
the class itself (matching the file's existing `_mha_old_full_qkv` convention), not deleted.

One deliberate departure from v4's EXACT original, needed to satisfy the h-decoupling goal: v4's
version mixed `h` directly into the attention's own INPUT (`x = h_scale*h + shifted + pos`), which
forces `h` and the bit-embeds to share a dimension — incompatible with keeping `h` at full width
while downsampling the embeds. Here, `h` only enters at the final concat step; attention's own
Q/K/V machinery runs purely on bit-embeds/position at `d_inner`. Verified identically to SSM above
(fixed/loop consistency exact/near-exact, gradients, full-model integration, `validate_generation`
parity at `downsample=1` and `downsample=4`). Not yet re-queued for training — the reverted design's
own best_val_bpb (with or without the h-decoupling change) hasn't been re-measured this session;
worth a fresh `attn_id*`-family run once the current queue clears.

## 19. `downsample_h` A/B configs queued; `BitPredictHeadWordPredict` designed, implemented, and wired end-to-end

**`downsample_h`/`per_position_head` flags added** (session: "queue more experiments to test this
hypothesis, repr loss because of downsample h," and "also for each self attn and ssm, allow indp
heads for each timestep different head, on by default") — since §18's decoupling REPLACED the old
in_proj-based behavior rather than keeping it as an option, testing "does downsampling h itself
(not just the embeds) cause a real quality loss" required restoring it as an explicit, orthogonal
flag on both `BitPredictHeadAttn` and `BitPredictHeadSSM`:

- `downsample_h: bool = False` (default: h stays full width, §18's behavior). `True`: restores the
  original in_proj-based behavior (h also projected to `d_inner`) for direct A/B at the same
  downsample ratio.
- `per_position_head: bool = True` (default: dq separate weight rows, einsum-read). `False`: a
  single SHARED head (v4's original design) applied via broadcasting.

Both flags verified across all 2x2 combinations for both classes (fixed/loop consistency,
gradients, full-model integration, `validate_generation` parity) plus the `downsample=1` default
case. Four new configs queued: `attn_id4_hfull`/`attn_id4_hds` and `ssm_id4_hfull`/`ssm_id4_hds`
(all `bit_inner_downsample=4`, `bit_per_position_head=True`, differing only in `bit_downsample_h`)
— direct, budget-controlled tests of the hypothesis.

**`BitPredictHeadWordPredict`** (new — session: "design another head, wordpredict, which decompose
to word like 8 bit, 4 bit, useful for dq more than 8... can degenerate to single softmax for
compatibility... implement until done complete with ar gen and config to queue"). Decomposes the
dq-bit code into `n_words=dq//word_bits` WORDS, each a genuine `2**word_bits`-way softmax — a
middle ground between `BitPredictHeadHSoftmax`'s per-bit binary tree (many cheap steps) and a
single flat `V=2**dq` softmax (one expensive step). Conditioning (session: "past chain prob
conditioning make simpler but more expensive"): word `i`'s classifier reads
`cat([h, embed(word_0),...,embed(word_{i-1})])` — plain concatenation, no recurrence/attention
machinery, but the classifier input GROWS linearly with position (genuinely simpler, genuinely more
expensive per step than attn/ssm's fixed-size conditioning).

Returns a LIST of per-word logit tensors (not `[N,dq]` per-bit logits like every other
`BitPredictHead*`), so it needed a new `code_head_mode="word"` pipeline path end-to-end: `Config.
word_bits`/`word_d_embed`/`word_embed_downsample`, `LevelLM.__init__`/`forward` (own loss —
`self.ntp_head.loss`, sum of per-word cross-entropies), and `_sample_next_byte` (own
`logits_to_word_ints`/`word_ints_to_bits` round-trip back to the shared dq-bit representation).

`word_bits==dq` (`n_words=1`) DEGENERATES to a single flat softmax with no chain/embed machinery at
all — verified numerically IDENTICAL to a plain `nn.Linear(D,vocab)` (weights copied over,
`torch.allclose` exact) — the "compatibility" the session asked for.

**Parallel kernel launch for `_forward_fixed`** (session: "find way to parallel launch kernel,
maybe pad, idk"): since teacher-forcing means every word is known upfront, all `n_words` steps'
context vectors can be built in parallel (unlike generation, where each word's context depends on
the GREEDILY decided previous one). Pads every word's context to the largest word's own width
(zero-filling the unused tail — the padding weight columns simply never receive gradient, since
their input is always 0) and reads all words via ONE batched einsum against a single
`[n_words, word_vocab, max_dim]` weight tensor, instead of `n_words` separate `nn.Linear` calls —
"find a way to parallel launch kernel" satisfied directly. `_forward_loop` (generation) stays
genuinely sequential, reusing slices of the same weight tensor via `F.linear`.

Fully verified: fixed/loop consistency across `word_bits` in {8,4,2,1} (exact/near-exact match),
gradients, the degenerate-case exact match, causality (a later word's bits don't leak into an
earlier word's logits — confirmed by flipping word 2's bits and checking words 0/1's logits are
byte-for-byte unaffected while word 3's do change), full-model forward+backward, and
`validate_generation` exact parity. Two configs queued: `word4` (`word_bits=4`, `n_words=2`, each a
16-way softmax) and `word2` (`word_bits=2`, `n_words=4`, each a 4-way softmax) — the two ends of the
word_bits dial at `dq=8`, both `code_embed_mode="pq_table"`.

Full queue as of this writing: `byte_lowrank` (running) -> `conv_depthwise` -> `conv_dilated` ->
`hsoftmax` -> `attn_id4_hfull` -> `attn_id4_hds` -> `ssm_id4_hfull` -> `ssm_id4_hds` -> `word4` ->
`word2`.
