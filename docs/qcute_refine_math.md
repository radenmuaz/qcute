# `qcute_refine` — math

Companion to `qcute/qcute_refine.py`'s module docstring. Notation mirrors
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
