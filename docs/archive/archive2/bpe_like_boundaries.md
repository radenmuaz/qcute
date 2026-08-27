# Is qcute_refine doing something like BPE? Making the fixed grid content-adaptive

Cross-referenced from [docs/status.md](status.md). Session discussion,
not yet implemented — brainstorm + math for the most promising
directions, no training run yet.

## Is it BPE-like?

No, and it's worth being precise about why. BPE merges are
**content-adaptive**: frequent substrings collapse into single tokens,
rare ones stay unmerged — the number of raw bytes per token varies with
what the text contains. `qcute_refine`'s downsampling is the opposite:
`Ks=(4,4)` pools exactly every 4 raw positions into one code,
unconditionally, regardless of content. Closer to a fixed-stride
conv/patch embedding than to BPE — the block boundary is fixed in the
SEQUENCE, not learned from the DATA.

## The hypothesis

The "inner LM" (level 0's own per-position NTP head) already computes a
signal BPE's merge criterion effectively approximates for free:
per-position predictive entropy/surprisal. Low local entropy roughly
corresponds to "this looks like the inside of a token BPE would have
already merged"; high entropy roughly corresponds to "this is near a
natural boundary." `code_pre` currently reads only
`h_blocks[:, :, K-1, :]` (the block's LAST position) — it never weights
by anything the inner LM itself learned about that block's internal
structure. That's a discarded signal, not a missing one.

## Constraint

True BPE produces a *variable* token count per byte span — incompatible
with fixed tensor shapes, batching, `torch.compile`. Every direction
below keeps the code count fixed at exactly `context_len/K` (unchanged)
and only makes *which bytes matter within each fixed block*
content-dependent — soft/differentiable adaptivity inside a rigid grid,
not variable-length tokenization.

## 1. Entropy-weighted pooling

Notation: level `i`'s post-self-attention hidden states `h_t ∈ R^D`,
`t=0..L-1`. Block `b` spans raw positions `t ∈ [bK, bK+K-1]`,
`b=0..n_blocks-1`. Current readout: `c_b = code_pre(h_{bK+K-1})`.

Per-position entropy from the inner LM's own prediction (predicting
position `t+1` from `h_t`):

- `byte_repr="embed"` (256-way softmax): `p_t = softmax(ntp_head(h_t))`,
  `H_t = -Σ_v p_t(v) log p_t(v)`.
- `byte_repr="bits"`/chain head (dq independent-ish bits): sum of binary
  entropies, `H_t = Σ_{j=1}^{dq} H(bit_j | h_t)`.

Softmax pooling weights over the K positions in block `b`, using
NEGATIVE entropy (low entropy = confident = stable "interior" of a
token = more pooling weight — high entropy near the end of a block means
the model was uncertain about what comes next, i.e. `t` is often where a
segment would naturally end, matching BPE's own implicit criterion):

```
w_{b,k} = exp(-H_{bK+k} / τ) / Σ_{k'=0}^{K-1} exp(-H_{bK+k'} / τ),   k = 0..K-1
```

`τ`: temperature. `τ→0` approaches hard argmin-entropy position
selection; `τ→∞` degenerates to uniform mean pooling.

Pooled vector and code (same `code_pre`/BSQ machinery downstream, only
the INPUT changes from a hard single-position read to a soft
combination):

```
h̃_b = Σ_{k=0}^{K-1} w_{b,k} · h_{bK+k}
c_b  = quantize(code_pre(h̃_b))
```

**Shape**: always exactly `n_blocks` vectors of dim `D` — only the
convex-combination weights vary with content. Fully static.

**Gradient path**: `H_t` is a differentiable function of `h_t` (via
`ntp_head`), so `w_{b,k}` is differentiable too — a proper
attention-like pooling, no STE needed, UNLIKE `bsq_quantize`/
`CodeEmbed`'s `pq_table` mode.

**Real risk worth flagging**: if `H_t` is used BOTH as a training target
(the NTP loss wants it low where genuinely predictable) AND as a
differentiable pooling gate feeding forward into the SAME model's future
loss, there's a feedback-loop risk — the model could learn to report
artificially low entropy everywhere just to win better pooling weight,
gaming the routing rather than reflecting genuine predictive difficulty.
Mitigation, matching this codebase's own established convention
(`bsq_quantize`, `CodeEmbed`'s `pq_table`): **detach `H_t`** before using
it as a pooling weight — treat entropy as an observed statistic for
routing, not a differentiable gate. Safer starting point; revisit
whether to let gradient flow through routing only if the detached
version shows promise.

**Cost**: negligible — entropy is a cheap reduction over logits already
computed for `ntp_loss` when `compute_ntp=True`.

## 2. Learned soft-assignment pooling (Charformer/GBST-style)

Prior art worth naming directly: Charformer's Gradient-Based Subword
Tokenization (Tay et al. 2021) computes scores over candidate
subword widths/positions and does a soft, fully-differentiable
combination, producing a FIXED-length downsampled sequence — exactly
this "imitate BPE, stay static-shape" goal, already solved in the
literature, just needs adapting to this file's fixed-`K` constraint
(GBST varies block WIDTH; here `K` is fixed for shape reasons, so we vary
the POOLING FUNCTION instead — a narrower but shape-compatible version of
the same idea).

Define a small FIXED set of `J` candidate poolers `φ_j`, each still
producing exactly one `D`-dim vector per block (so output shape is always
`[B, n_blocks, D]` regardless of which candidate wins):

- `φ_1`: read last position, `h_{bK+K-1}` (today's default)
- `φ_2`: mean over the full block, `(1/K) Σ_{k=0}^{K-1} h_{bK+k}`
- `φ_3`: mean over the last `K/2` positions
- `φ_4`: entropy-weighted pool, per §1

```
X_b^{(j)} = φ_j(h_{bK}, ..., h_{bK+K-1}),   j = 1..J
```

Score each candidate with a small learned gate (e.g. a linear/MLP over
the block's own pooled statistics):

```
score_b^{(j)} = g_j(mean(h_{bK:bK+K}))
π_b            = softmax(score_b^{(1)}, ..., score_b^{(J)}) ∈ Δ^{J-1}
X̃_b           = Σ_{j=1}^{J} π_b^{(j)} · X_b^{(j)}
c_b            = quantize(code_pre(X̃_b))
```

§1 is the special case `J=K`, one candidate per raw position, scored by
`-H_t`. GBST-style widens this to different POOLING SHAPES, not just
candidate positions — lets the model learn e.g. "this block is better
summarized as a mean" vs. "this block is defined by its last position"
vs. "this block is defined by its first position" (start-of-word-like
blocks) per block, content-dependently.

**Compile-friendliness**: `J` is a small fixed integer (hyperparameter,
e.g. 3-5) — `J` parallel poolings + one small softmax gate + weighted
sum, all static-shape tensor ops, no data-dependent control flow.

## 4. Boundary signal as an explicit extra channel (not just a pooling weight)

Rather than only using the boundary/entropy signal to WEIGHT the
pooling (§1/§2, where it can get blurred into an average), feed it
upward as an explicit extra input dimension so level `i+1` can condition
on it directly.

Per-block boundary score, using the LAST position's entropy specifically
(most indicative of whether the hand-off point was a genuine
predictability drop or an arbitrary cut through a predictable run):

```
β_b  = H_{bK+K-1}                    (or averaged: (1/K) Σ_k H_{bK+k})
β̃_b = β_b / log(V)                  (bytes: V=256, log(256)=8 nats → β̃_b ∈ [0,1] roughly)
```

Level `i+1`'s own input becomes the code AUGMENTED with this scalar (or
a small learned embedding of a coarsely-binned `β̃_b`, dimension `m`):

```
x_b^{(i+1)} = [c_b^{(i)} ; β̃_b^{(i)}]  ∈  R^{dq_i + 1}
```

Wiring: bump `in_dq` for level `i+1`'s `CodeEmbed` by `+1` (or `+m`) —
a compile-time-constant shape change (fixed at model-definition time,
not data-dependent), same shape-friendliness principle as everything
else here. One wrinkle: `code_embed_mode="pq_table"` (session-added,
see `docs/status.md`) indexes by treating the code as `2**in_dq`
discrete corners — a continuous `β̃_b` isn't one of BSQ's ±1 corners, so
`pq_table` mode would need a hybrid `CodeEmbed`: table-lookup the
discrete `dq_i`-bit code as now, PLUS a small continuous linear read of
`β̃_b`, summed — not a single enlarged table.

**Complementary to §1/§2, not a replacement**: could combine — use
entropy-weighted or GBST-style pooling to form the code CONTENT itself,
and separately pass the boundary score forward as explicit metadata via
this channel, giving level `i+1` both a better-summarized code AND
explicit information about how sharp/confident that summary's own
boundary was.

## Recommendation

§1 first — smallest, cheapest, no new modules, directly tests whether
the hypothesis has legs (entropy-weighted pooling vs. fixed-position
readout on val_bpb) before investing in §2's new scored module or §4's
shape/CodeEmbed changes. Not yet implemented — queue is currently three
runs deep (`pq_table` [done, 2.4816], `v3_rope`, `dense0_starve1`);
natural next slot once those report back.
