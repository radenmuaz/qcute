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
