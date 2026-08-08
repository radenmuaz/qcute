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
architecture, same additive-loss training scheme, `fuse_position="concat"`) reaches **val_bpb
2.7150 already by step 1100** — better than v4.2's entire 4000-step best (4.0369) in barely a
quarter of the training budget. This directly confirms concat fusion itself is not the problem;
the instability is specific to v4.2's head/embed unification. `qcute_refine_v4_1_k32_narrow_shared`
(trunk-shared, head/embed NOT unified — v4.1's own scheme) is queued to isolate the remaining
question: is trunk-sharing alone (self.blocks/ln_f tied across levels) fine, with the instability
coming specifically from ALSO tying the embed/head (v4.2's addition on top), or does trunk-sharing
alone already show some of this pathology? Not yet resolved as of this writing.

`qcute_refine_v4_2_k32_narrow_ssm` (same architecture, `code_head_mode="chain"` +
`bit_head_class="ssm"` instead of `"independent"`) is also queued — tests whether a different
SHARED head type (exact chain-rule factorization via `BitPredictHeadSSM`, rather than independent
per-bit logits) reduces or reproduces the same instability, which would further localize the cause
to "any shared head, regardless of type" vs. "specifically the independent-linear head's own
optimization landscape."

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
| `qcute_refine_v4_k32_narrow_concat` (v4, unshared, in progress) | 2.5366 so far | 2.5363 | 2500 |
| `qcute_refine_v4_2_k32_narrow` (v4.2, shared head, unstable) | 4.0369 | 4.0563 | 3700 |
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
