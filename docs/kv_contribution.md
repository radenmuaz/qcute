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

This adds real nuance to `qcute_refine_v3_rope`'s "direct forward-value
conditioning beats everything" reading from §6 above: for THIS
narrow-window config at least, most of that forward-value benefit isn't
really about the CONTENT being conditioned on — it's about the extra
processing capacity the fusion mechanism happens to add along the way.
Consistent with, and motivates, the `tier_n_layers=(2,1)`/`(2,2)` depth
ablations queued this session (see docs/status.md) — if plain depth
matches or beats fusion's own contribution, that would confirm capacity,
not cross-level information, was the dominant lever all along.
