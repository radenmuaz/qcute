# `qcute_refine` — math

Companion to `qcute/qcute_refine_v1.py`'s module docstring. Notation mirrors
the code directly (`Config.Ks`, `Config.dqs`, etc.) so each equation below
can be matched line-for-line against a function in the file. Written to be
checked, not just read — flag anything that doesn't type-check.

## 1. Setup and notation

$n$ levels, $i = 0, \dots, n-1$, local compression factors
$K_0, \dots, K_{n-1}$ (`cfg.Ks`), per-level widths $d_i$ (`cfg.tier_d_models[i]`)
and emitted code widths $q_i$ (`cfg.dqs[i]`).

Level $0$'s own input is the raw byte sequence; level $i>0$'s own input is
level $i-1$'s emitted code sequence. Define the **own input width**
$$
\rho_i = \begin{cases} 8 & i = 0 \quad \text{(byte\_to\_bits)} \\ q_{i-1} & i > 0 \end{cases}
$$
(`in_dqs[i]` in code — level $i$'s own sequence lives in $\{-1,+1\}^{\rho_i}/\sqrt{\rho_i}$,
whether that's a literal byte-bit encoding or a BSQ code from the level below).

Sequence lengths shrink geometrically:
$$
L_0 = \texttt{context\_len}, \qquad L_i = L_{i-1} / K_{i-1} \quad (i \ge 1),
$$
and each level's own **code sequence length** (number of blocks) is
$$
n^{(i)}_{\text{blk}} = L_i / K_i.
$$
Note $n^{(i)}_{\text{blk}} = L_{i+1}$ for $i < n-1$ — level $i$'s emitted
code sequence literally *is* level $i+1$'s own input sequence, by
construction (`seq_repr = c_i` in `RefineLM.forward`).

Let $x^{(i)} \in \{-1,+1\}^{L_i \times \rho_i}/\sqrt{\rho_i}$ denote level
$i$'s own input sequence (`seq_repr` at the top of iteration $i$), and
$c^{(i)} \in \mathbb{R}^{n^{(i)}_{\text{blk}} \times q_i}$ its emitted code
sequence.

## 2. BSQ quantizer

Parameter-free, hypersphere-corner quantization with a straight-through
estimator (`bsq_quantize`), applied to a raw pre-quantization vector
$v \in \mathbb{R}^{q}$:
$$
\hat v = \frac{v}{\lVert v \rVert_2}, \qquad
z = \frac{\hat v + \operatorname{sg}\!\big(\operatorname{sign}(\hat v) - \hat v\big)}{\sqrt q},
$$
where $\operatorname{sg}(\cdot)$ is stop-gradient. Forward pass: $z =
\operatorname{sign}(\hat v)/\sqrt q \in \{-1/\sqrt q, +1/\sqrt q\}^q$ (a
hypersphere corner). Backward pass: gradient flows as if $z = \hat v /
\sqrt q$ (identity through the sign). This is exactly `byte_to_bits`'s
target geometry at $q=8$ — the "pure" design choice (message: *"by
default use bsq, first block bsq 8 dq"*) is that every level's own
representation, own emitted code, **and** its own input's native
encoding all live in the same $\{-1,+1\}^{(\cdot)}/\sqrt{(\cdot)}$ family,
with no separate discrete-vocabulary/embedding-table special case
anywhere in the tower.

## 3. Encoder level $i$ — `EncoderLevel.forward`

$$
h^{(i)} = \operatorname{LN}\!\Big(\operatorname{Block}^{(i)}_{1:L^{(i)}_{\text{layers}}}\big(W^{(i)}_{\text{embed}}\, x^{(i)}\big)\Big) \in \mathbb{R}^{L_i \times d_i},
$$
a plain causal transformer of depth `cfg.tier_n_layers[i]` (default $1$)
over $x^{(i)}$ — nothing else feeds in.

**Own NTP loss** (always on, own head, own target — this is the "targets
stable" property): a `BitPredictHead` chain-decodes $\rho_i$ bits from
each position's hidden state to predict that level's own next input
element. Writing the joint chain-rule factorization explicitly, for
position $t$ predicting $x^{(i)}_{t+1} \in \{-1,+1\}^{\rho_i}$ from
$h^{(i)}_t$:
$$
P\big(x^{(i)}_{t+1} \mid h^{(i)}_t\big) = \prod_{j=1}^{\rho_i} P\big(x^{(i)}_{t+1,j} \mid h^{(i)}_t,\, x^{(i)}_{t+1,<j}\big),
$$
each factor a Bernoulli logit from `BitPredictHead`'s causally-masked
self-attention chain over the $\rho_i$ bit positions (teacher-forced with
the true preceding bits during training — the `_forward_fixed` batched
path). The per-level loss is the summed negative log-likelihood, averaged
over positions and batch:
$$
\mathcal L^{(i)}_{\text{ntp}} = \frac{1}{B(L_i-1)} \sum_{b,t} \sum_{j=1}^{\rho_i} \operatorname{BCE}\Big(\ell^{(i)}_{t,j},\ \mathbb 1[x^{(i)}_{t+1,j} > 0]\Big),
$$
where $\ell^{(i)}_{t,j}$ is the chain head's raw logit for bit $j$
(`chain_bce_loss`, sum over bits then mean over positions/batch — *not*
`reduction='mean'` over everything, which would silently average over
bits too and misreport nats-per-unit as nats-per-bit).

**Code emission**: at block-end positions only,
$$
c^{(i)}_b = \operatorname{BSQ}\!\Big(W^{(i)}_{\text{code}}\, h^{(i)}_{bK_i + K_i - 1}\Big), \qquad b = 0, \dots, n^{(i)}_{\text{blk}}-1,
$$
i.e. `code_pre` reads only the *last* hidden state in each $K_i$-sized
block, matching the encoder's "only ever reads every $K$-th position"
property (as opposed to the detokenizer below, which uses every code
position).

## 4. Detokenizer level $i$ — `Detokenizer.forward`

The structural inverse of §3's code-emission step: given $c^{(i)}$ alone
(detached — `c_i.detach()` in `RefineLM.forward`, so this loss cannot
reshape the encoder), reconstruct the exact block of $\rho_i$-dim units
that produced each code element. Runs its **own** causal transformer, of
depth `cfg.detok_n_layers`, over the *code* sequence itself (length
$n^{(i)}_{\text{blk}}$, every position used — no skipping):
$$
g^{(i)} = \operatorname{LN}\!\Big(\operatorname{Block}^{(i)}_{1:\text{detok\_n\_layers}}\big(W^{(i)}_{\text{code-embed}}\, c^{(i)}\big)\Big) \in \mathbb{R}^{n^{(i)}_{\text{blk}} \times d_{\text{detok}}}.
$$
Because this trunk is causal over $b$, $g^{(i)}_b$ can draw on code
context from blocks $0, \dots, b$ (past + current — the long-range
channel: coarse code carries compressed history further than the block
itself would alone).

**Joint chain MTP head**: at *every* block $b$ (no skipping — this is the
"detokenizer trunk uses every position" half of the earlier "no timestep
skip" framing, now applied at the code granularity rather than the raw
one), predict the *entire* $K_i$-element target block
$y^{(i)}_b = \big(x^{(i)}_{bK_i}, \dots, x^{(i)}_{bK_i+K_i-1}\big) \in \{-1,+1\}^{K_i \rho_i}$
**jointly**, via one chain-rule factorization over all $K_i \rho_i$ bits
at once (not $K_i$ independent per-step guesses, and not $K_i$ separate
per-step chains — one continuous chain spanning the whole block):
$$
P\big(y^{(i)}_b \mid g^{(i)}_b\big) = \prod_{m=1}^{K_i \rho_i} P\big(y^{(i)}_{b,m} \mid g^{(i)}_b,\, y^{(i)}_{b,<m}\big).
$$
This is exactly `BitPredictHead` again, instantiated with `dq = K_i *
rho_i` instead of `rho_i` — the flatten-then-chain trick that lets the
same chain machinery serve both an $8$-bit single-byte NTP head (§3) and
a $K_i \rho_i$-bit whole-block MTP head (§4) with no new class. Loss:
$$
\mathcal L^{(i)}_{\text{mtp}} = \frac{1}{B\, n^{(i)}_{\text{blk}}} \sum_{b} \sum_{m=1}^{K_i \rho_i} \operatorname{BCE}\Big(\ell^{(i)}_{b,m},\ \mathbb 1[y^{(i)}_{b,m} > 0]\Big).
$$

**Recursive stacking.** Detokenizer $i$ only ever decodes $c^{(i)} \to
x^{(i)}$ — the block that *inputs* $c^{(i)}$, per level, nothing dense or
cross-level. A full top-down decode of a given top code sequence
$c^{(n-1)}$ back to raw bytes is the composition
$$
c^{(n-1)} \xrightarrow{\text{Detok}_{n-1}} x^{(n-1)} = c^{(n-2)} \xrightarrow{\text{Detok}_{n-2}} \cdots \xrightarrow{\text{Detok}_0} x^{(0)} = \text{bytes},
$$
i.e. **first** applied = `detokenizers[n-1]` (decodes the *last*/coarsest
code), **last** applied = `detokenizers[0]` (decodes bytes) — matching
the stack-order framing verbatim. `self.detokenizers` is still indexed
$0, \dots, n-1$ in the code (paired 1:1 with `self.encoders[i]`); the
reversal is in execution order for a full decode, not storage order.

## 5. Total loss

$$
\mathcal L = \sum_{i=0}^{n-1} \mathcal L^{(i)}_{\text{ntp}} \;+\; \lambda_{\text{detok}} \sum_{i=0}^{n-1} \mathcal L^{(i)}_{\text{mtp}}, \qquad \lambda_{\text{detok}} = \texttt{cfg.detok\_weight}.
$$
`byte_loss := L^{(0)}_ntp` is the only term with a direct bits-per-byte
reading (`bpb = byte_loss / ln 2`); every other term is an auxiliary
signal at some coarser grain (code-level NTP, or block reconstruction) —
`mtp_loss_total` in particular sums nats over $K_i \rho_i$-bit *blocks*,
not per-byte units, so it is not bpb-comparable without an explicit
$1/(K_i \rho_i)$ rescale (not currently applied — logged as a raw
diagnostic, matching every other fork's "some accumulated metrics are
mechanically scaled by $dq$/$K$ and not directly bpb-comparable" caveat).

## 6. Generation — greedy chain decode, and its KV-cache equivalence

Only $\text{EncoderLevel}_0$'s own NTP head is generative (§3 at $i=0$
predicts the literal next byte from causal byte history alone — every
other component either looks at a not-yet-known future code, like
levels $i>0$, or decodes an *already fully-known* code's *already
fully-known* children, like every detokenizer, which is a reconstruction
map, not a next-token map).

Greedy byte decode at step $t$: draw $8$ bits by walking `BitPredictHead`'s
chain with `true_bits=None` (`_forward_loop`, since future bits within
the same byte are only defined by the head's own already-emitted earlier
bits, unlike training's teacher-forced batched path):
$$
\hat b_j = \operatorname{sign}\big(\ell_j(h^{(0)}_t,\ \hat b_{<j})\big), \qquad j = 1,\dots,8,
$$
then `bits_to_byte(\hat b)` recovers the byte id. `generate_no_cache`
recomputes $h^{(0)}_{1:t}$ from scratch (dense causal attention over the
whole growing sequence) before every draw — obviously correct, $O(t^2)$
per new byte. `generate_kv_cache` instead grows each attention layer's
$K/V$ tensors by exactly one entry per step via `Block.forward_step`, and
never recomputes a past position's hidden state — valid **iff** $h^{(0)}_t$
is a pure function of causal history alone and never mutated by anything
computed later, which holds here by construction (§3: nothing downstream
of level $0$ ever feeds back into level $0$'s own hidden states). Formally,
for both procedures, at every shared step $t$:
$$
h^{(0),\,\text{no-cache}}_t = h^{(0),\,\text{kv-cache}}_t \quad \Longrightarrow \quad \hat b^{\text{no-cache}}_t = \hat b^{\text{kv-cache}}_t
$$
(same hidden state $\Rightarrow$ same head $\Rightarrow$ same greedy
chain draw, since both routes call the identical `_sample_next_byte`).
`validate_generation` checks the terminal claim directly — full output
byte-sequence equality — rather than re-deriving the per-step hidden-state
equality above; a real run (`Config` at $n=2$, `context_len=32`, random
init) confirmed `torch.equal(out_no_cache, out_kv_cache) == True`.

## 7. `qcute_refine_v2` — cross-attention `DecoderLevel`, replacing §4's `Detokenizer`

Encoder tower (§1–§3) is **unchanged** in v2 — same `EncoderLevel`,
same recursion, same always-on per-level NTP loss, same BSQ hand-off.
The only thing v2 replaces is §4's `Detokenizer` (a *self*-attention
pass over the code sequence, decoding a whole $K$-block jointly). In its
place: `DecoderLevel_i`, one per adjacent level pair $(i, i{+}1)$ — not
one per level (there are $n{-}1$ of them, since there's no decoder above
the top level), and it *reuses* $h^{(i)}$/$h^{(i+1)}$ (already computed
by the encoder tower — zero extra trunk compute by default) instead of
running any self-attention of its own.

**Algorithm** (`DecoderLevel_i.forward`, teacher-forced training):

```
Input:  h_prev  ∈ ℝ^{B×L_i×d_i}        (EncoderLevel_i's own hidden states, DETACHED)
        h_curr  ∈ ℝ^{B×n_{i+1}×d_{i+1}} (EncoderLevel_{i+1}'s own hidden states, DETACHED)
        x_i     — level i's own true input sequence (decode target)
Output: loss, acc

 1. q  ← q_proj(h_prev)                          # ℝ^{B×L_i×D}
 2. kv ← kv_proj(h_curr)                          # ℝ^{B×n_{i+1}×D}
 3. kv ← concat(null_kv, kv)                      # prepend 1 learned always-visible slot
 4. for t in 0..L_i-1, b in 0..n_{i+1}-1:
        n_complete(t) ← ⌊(t+1) / K_i⌋              # blocks 0..n_complete(t)-1 are causally resolved by t
        visible[t, b] ← n_complete(t) - W_{i+1} ≤ b < n_complete(t)   # §7.1: causal AND within KV window
    visible[t, null] ← True  ∀t                    # null slot always visible (avoids all-masked rows)
 5. if cross_attn_rope:
        rope_q[t]     ← RoPE-angles(position = t)
        rope_k[null]  ← RoPE-angles(position = 0)
        rope_k[b]     ← RoPE-angles(position = (b+1)·K_i - 1)     # block b's own raw-time resolve point
        q, kv ← apply_rope(q, rope_q), apply_rope(kv, rope_k)     # (applied inside CrossBlock, per-head)
 6. h_dec ← CrossBlock(q, kv, mask = ¬visible)     # cross-attn sublayer + MLP sublayer, both pre-norm+residual
 7. (loss, acc) ← Head(h_dec[:, :-1, :], target = x_i[:, 1:])   # softmax (level 0, byte_repr="embed")
                                                                   # or chain-BCE (bit-shaped target), else
return (loss, acc)
```

**§7.1 KV window.** Step 4's `visible` predicate has two independent
constraints, not one: block $b$ must be *causally resolved*
($b < n_{\text{complete}}(t)$, same rule as v1's own past-block
reasoning) **and** *recent enough*
($b \ge n_{\text{complete}}(t) - W_{i+1}$), where $W_{i+1} =$
`Config.attn_window[i+1]` (`None`/$-1$ ⇒ no second constraint, full
unbounded causal reach — the original, pre-fix behavior). Before this
was added, the cross-attention could reach arbitrarily far back in block-
index terms with no cap at all, inconsistent with the encoder's own
windowed self-attention at that same level — $W_{i+1}$ makes the two
consistent (same units: level $i{+}1$'s own block/position scale).

**§7.2 RoPE positions.** Q lives at level $i$'s raw-time resolution
($0, \dots, L_i-1$); KV lives at level $i{+}1$'s block resolution — the
two can't share one contiguous rotary range, so each side gets its own
explicit position *tag*, both expressed in the same raw-byte-time units
so relative distances are meaningful across the boundary: query
position $t$ literally is raw-time $t$; KV block $b$ is tagged at
$(b{+}1)\cdot K_i - 1$, the exact raw-time index at which block $b$
*becomes* fully resolved (matches §7's own `n_complete` cutoff exactly —
by construction, block $b$ is visible from query $t$ iff its resolve-time
tag is $\le t$ *and* within the window). The null slot is tagged at a
fixed reference position $0$ (arbitrary but constant — it carries no
temporal meaning, only a well-defined fallback). `Config.cross_attn_rope`
defaults `True`; `False` restores plain (position-blind) cross-attention.

**§7.3 Two more structural options**, both defaulting to the §7
algorithm above (`decoder_own_trunk = decoder_kv_pass_through =
decoder_q_pass_through = False`), each swapping out exactly one line:

- `decoder_own_trunk=True`: replace step 1/2's *reuse* of
  $h^{(i)}$/$h^{(i+1)}$ with a **private, separate-weight**
  `EncoderLevel` copy run fresh over raw $x_i$/$c_i$ — restores a real
  self-attention trunk on both sides, at the cost the reuse design was
  built specifically to avoid (session estimate: this level pair's own
  params/FLOPs roughly double).
- `decoder_kv_pass_through=True` (Q unaffected) / `decoder_q_pass_through=True`
  (KV unaffected) — independently swap step 1 or step 2 for a **direct,
  trunk-free projection**: $q \leftarrow \texttt{q\_embed}(x_i)$ (a
  fresh `Embedding`/`Linear` straight to width $D$) or
  $kv \leftarrow \texttt{code\_proj}(c_i)$ (a fresh `Linear(dq_i, D)` on
  the raw code, bypassing $h^{(i+1)}$ entirely) — the "how much does this
  side's own contextualization actually matter" floor probe.

## 8. Loss and total (v2)

Same shape as §5, renamed: $\lambda_{\text{detok}} \to
\lambda_{\text{tok}} =$ `Config.tok_weight`, and `Detokenizer` →
`DecoderLevel` throughout. `Config.code_ntp_weight` additionally scales
levels $>0$'s own NTP terms (level 0's `byte_loss` is never scaled) —
$0.0$ **skips** that level's own `ntp_head` call entirely (not just
zero-weights it), same "skip the expensive call, don't just multiply by
zero" principle `tok_weight=0.0` already applied to §7's `DecoderLevel`
in v1.
