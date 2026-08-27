# Why teacher-forced bpb is still a valid bound under code-conditioned block-local decode

Question this answers (chat, 2026-08-23): `StackDecoderLocal` decodes every block's bytes in
parallel, conditioned only on that block's own code (plus optionally a small window of
neighboring codes one level up) -- no cross-block same-level attention at all. Is the resulting
teacher-forced `bpb`/loss still a valid bound on the true per-byte entropy, the way a normal
autoregressive LM's chain-rule log-likelihood is? Below: yes, via the same cross-entropy
(Gibbs') inequality that makes *any* teacher-forced LM loss valid, generalized to the
encode/decode (discrete-latent) case via the standard ELBO argument for deterministic encoders
(as used in VQ-VAE-style models). The AR-LM case is the special case where that argument's
inequality collapses to an *equality*.

## 1. The one fact everything else reduces to: cross-entropy is a valid upper bound

For any two distributions $p$ (true) and $q$ (model) over the same space $X$:

$$
H(p, q) \;=\; \mathbb{E}_{x \sim p}[-\log q(x)] \;=\; H(p) + D_{\mathrm{KL}}(p \,\|\, q) \;\ge\; H(p)
$$

with equality iff $q = p$ almost everywhere. This is Gibbs' inequality ($D_{\mathrm{KL}} \ge 0$,
itself Jensen's inequality applied to $-\log$, a convex function). It holds for **any** $q$,
regardless of how $q$ is computed or parameterized -- the only requirement is that $q$ is a
genuine, correctly-normalized probability distribution over $X$. Training a model by minimizing
$\mathbb{E}_{x\sim p}[-\log q(x)]$ (cross-entropy / NLL) therefore always minimizes a valid upper
bound on the true entropy $H(p)$; the reported loss can only overstate the true bits/byte, never
understate it (in expectation, over the true data distribution, for a properly normalized $q$).

`bpb` is exactly this quantity in bits: $\mathrm{bpb} = H(p,q)/\ln 2$ per byte.

## 2. The plain autoregressive case (equality, no approximation)

An ordinary byte-level LM factorizes $q(x)$ by the chain rule over positions $1, \dots, L$:

$$
q(x_1, \dots, x_L) = \prod_{t=1}^{L} q(x_t \mid x_{<t})
$$

This is an **exact identity**, true for *any* joint distribution -- the chain rule is not an
approximation, so plugging it into Part 1 gives:

$$
\mathrm{bpb}_{\text{AR}} = \frac{1}{L\ln 2}\sum_{t=1}^{L} \mathbb{E}[-\log q(x_t \mid x_{<t})] \;\ge\; \frac{H(p)}{L}
$$

with the *only* source of looseness being how well the model $q(x_t\mid x_{<t})$ matches the true
conditional $p(x_t\mid x_{<t})$ -- there is no separate compression/bottleneck step. This is the
familiar case; the question is whether introducing a discrete code $c$ that a block's own decode
depends on (rather than raw byte history) breaks this.

## 3. The encode/decode case: a deterministic-encoder ELBO, not a chain-rule identity

Let $c = f(x)$ be the (hard, `code_hard=True`, `code_sample=False`) code the encoder computes
from the bytes of an entire block -- so $c$ is a **deterministic** function of $x$, not sampled,
matching every config in this session's ablation grid. Decode produces a conditional model
$q(x \mid c)$ (block-local, parallel across blocks, but a well-defined distribution once $c$ is
fixed), and a separate code-level model $p(c)$ (the encoder's own self-NTP over the code stream,
sequential/causal across code positions). The claim to prove: training on
$-\log p(c) - \log q(x\mid c)$ (encode loss + decode loss, evaluated at the real $c=f(x)$) still
upper-bounds $-\log p(x)$ for the true marginal $p(x)$, i.e. is still a valid `bpb` bound, even
though $c$ was computed *from* $x$ itself.

Start from the marginal likelihood of any latent-variable model, and introduce an arbitrary
"encoder" distribution $q(c\mid x)$ (below, the hard/deterministic one) purely as an
importance-sampling device:

$$
\log p(x) = \log \sum_{c} p(x, c) = \log \sum_{c} q(c\mid x)\,\frac{p(x,c)}{q(c\mid x)}
\;\ge\; \sum_{c} q(c\mid x) \log \frac{p(x,c)}{q(c\mid x)}
$$

by Jensen's inequality (concavity of $\log$). Expanding $p(x,c) = p(c)\,q(x\mid c)$ (code first,
then decode) and $q(c\mid x)$'s own entropy term:

$$
\log p(x) \;\ge\; \underbrace{\mathbb{E}_{q(c\mid x)}[\log p(c)]}_{\text{code/encode term}} + \underbrace{\mathbb{E}_{q(c\mid x)}[\log q(x\mid c)]}_{\text{decode term}} + \underbrace{H(q(c\mid x))}_{\text{encoder entropy}}
$$

This is the standard ELBO (evidence lower bound), exactly the objective used to justify
VQ-VAE-style discrete autoencoders. Now specialize to a **hard** encoder: $q(c \mid x)$ is a
point mass at $c = f(x)$ (exactly `code_hard=True, code_sample=False`). A point mass has zero
entropy, $H(q(c\mid x)) = 0$, and every expectation collapses to evaluating at the single value
$c = f(x)$:

$$
\log p(x) \;\ge\; \log p\big(c{=}f(x)\big) + \log q\big(x \mid c{=}f(x)\big)
$$

Negating and switching to bits-per-byte (dividing by $L\ln 2$):

$$
\boxed{\;\mathrm{bpb} \;=\; \frac{-\log p(c) - \log q(x\mid c)}{L\ln 2} \;\ge\; \frac{-\log p(x)}{L \ln 2}\;}
\qquad c = f(x)
$$

i.e. encode loss (`encode_losses[1:]`, the code stream's own transmission cost under $p(c)$) plus
decode loss (`decode_losses[0]`, $x$'s reconstruction cost given the real code) together still
upper-bound the true per-byte entropy -- exactly the same guarantee as the plain AR case in Part
2, just derived via Jensen/ELBO instead of via an exact chain-rule identity, because a lossy
discrete bottleneck ($c=f(x)$ compressing a whole block down to one code) sits in between.

## 4. Where the codebase's current metric falls short of this bound

The proof above needs **both** terms: $p(c)$ (code transmission cost) and $q(x\mid c)$
(reconstruction cost given that code). The codebase's own `bpb`/`bpb_full` metrics currently only
accumulate `decode_losses[0]` -- i.e. only $-\log q(x\mid c)$, dropping $-\log p(c)$ entirely
(`docs/status.md`'s 2026-08-23 `StackDecoderLocal` entry, "a pre-existing gap"). This is **not**
a validity problem with the argument above -- $q(x\mid c)$ alone is not claimed to bound
$-\log p(x)$, only $-\log p(c) - \log q(x\mid c)$ together is. The currently-reported `bpb` is
therefore best read as a *conditional* reconstruction cost given a free/uncosted code, not yet
the full valid bound Part 3 establishes; getting an apples-to-apples number against a real AR
baseline (`qcute.bytelm`) requires adding the missing `encode_losses[1:]` term in. This gap
applies identically to the sequential `StackDecoder` -- it is not introduced or worsened by
`StackDecoderLocal`'s parallelism, which only changes *how* $q(x\mid c)$ is computed (block-local
vs. cross-block self-attention), not whether the encode term is being counted.

## 5. The default sequential `StackDecoder`: same ELBO status as Part 3, just a tighter $q(x\mid c)$

Question: `decoder_type="stack"` (the default) additionally lets each block's seed token
self-attend across *previous* blocks (`encode_like_self_attn_decode`/`seed_query_decode`'s
cross-block channel) -- real sequential AR context that `stack_local` deliberately drops. Does
that extra machinery make sequential `StackDecoder`'s bpb an *exact* chain-rule quantity like
Part 2's plain AR LM, or is it still only a bound like Part 3?

**Still only a bound -- same ELBO status as `stack_local`, for the identical reason.** Partition
$x$ into blocks $B_1,\dots,B_M$ of size $K$, and let $c_i = f(x_{B_i})$ be block $i$'s hard code,
computed from block $i$'s *own* bytes (same `code_hard=True, code_sample=False` deterministic
encoder as Part 3 -- nothing about the decode mechanism changes what $c_i$ is or how it was
produced). Sequential `StackDecoder` decodes byte $r$ of block $i$ as

$$
q\big(x_{i,r} \mid x_{<B_i},\, x_{i,<r},\, c_i\big)
$$

conditioning on *all* previously decoded bytes globally ($x_{<B_i}$: every earlier block in full,
via cross-block self-attention) plus this block's own prefix ($x_{i,<r}$), plus this block's own
code $c_i$ (cross-attention, constant across the whole block). Chaining this within a block, then
across blocks, is a chain-rule identity **of $q(\cdot \mid c_{1:M})$ itself** -- the chain rule is
always an exact way to write *any* joint as a product of conditionals, regardless of what extra
variables ($c_i$) appear in the conditioning:

$$
q(x \mid c_1,\dots,c_M) = \prod_{i=1}^{M}\prod_{r=1}^{K} q\big(x_{i,r}\mid x_{<B_i}, x_{i,<r}, c_i\big)
$$

This is exactly Part 3's $q(x\mid c)$, merely factorized more expressively (with genuine
cross-block AR structure) instead of `stack_local`'s pure block-local factorization. Since
$c_i = f(x_{B_i})$ is still a *deterministic hard code derived from $x$ itself*, the identical
Jensen/ELBO step from Part 3 applies verbatim:

$$
\log p(x) \;\ge\; \sum_{i=1}^{M} \log p(c_i) \;+\; \log q(x \mid c_{1:M})
$$

i.e. `bpb` $= \big(-\sum_i \log p(c_i) - \log q(x\mid c_{1:M})\big)/(L\ln 2)$ is **still only an
ELBO-style bound** on the true entropy, not an exact AR identity -- exactly Part 3's conclusion,
unchanged. The cross-block self-attention only makes $q(x\mid c_{1:M})$ a strictly richer,
better-fitting conditional family than `stack_local`'s block-diagonal one (so it can only tighten
the achievable bound / lower the loss, never invalidate it or turn it into an equality) -- it does
not remove the need for the $p(c_i)$ code-transmission term, and it does not change which class of
guarantee applies. The two decoder types differ only in Part 2's sense (how well $q$ can fit $p$),
never in Part 3's sense (whether the guarantee is an equality or merely a bound).

The one place an exact identity genuinely *does* apply, in either decoder: the topmost level's own
decode (`is_top`, no code cross-attention at all -- plain NTP, Part 2's case exactly) and each
encoder's own NTP loss (`Encoder.forward`, non-circular by construction, predicting $x_{t+1}$ from
real causal history alone). It is specifically the non-top *decode* levels' own-code
cross-attention -- present in both `stack` and `stack_local` -- that requires Part 3's ELBO
argument instead of Part 2's exact chain rule.

## 6. `qcute_zero`: genuinely causal code-conditioning $\Rightarrow$ exact AR chain rule, not just a bound

`qcute_zero` (`qcute/qcute_zero/`) is architecturally different from `qcute_v1`'s `StackDecoder`/
`StackDecoderLocal` in exactly the one respect that matters for Parts 3 and 5: there is a single
shared LM doing the byte pass, and each `Ks`-stage periodically *fuses* its own code sequence back
into the byte stream via cross-attention -- but (verified position-by-position by direct code
inspection, `docs/status.md`'s 2026-08-22 "real differentiator" entry) **a byte's fuse
cross-attention mask never admits a code derived from that same byte, or from any byte at or after
it -- no exception.** Every admissible code $c_j$ at byte position $t$ is $c_j = g(x_{a_j:b_j})$
for some chunk with $b_j < t$, i.e. computed purely from bytes strictly earlier than the one being
predicted.

This is the crucial difference from `qcute_v1`: there, block $i$'s code $c_i = f(x_{B_i})$ is
computed from block $i$'s *own* bytes -- so for a byte $x_{i,r}$ being predicted, $c_i$ depends on
other bytes of the *same* block ($x_{i,r+1},\dots,x_{i,K}$) that haven't been generated yet at
inference time. Formally, $c_i \notin \sigma(x_{<t})$ for $t$ inside block $i$ -- the code carries
information outside the byte's own causal history, which is exactly what forced the Jensen/ELBO
step in Parts 3 and 5.

`qcute_zero`'s fuse codes have no such problem. Since every admissible $c_j$ satisfies
$c_j = g(x_{a_j:b_j})$ with $b_j < t$, each one is a **deterministic function of a subset of
$x_{<t}$** -- i.e. $c_j \in \sigma(x_{<t})$, the same sigma-algebra already generating the ordinary
AR conditioning set. Folding a variable that is already a deterministic function of your
conditioning set *into* that conditioning set changes nothing measure-theoretically -- no new
randomness is introduced, so there is nothing to marginalize over and no Jensen step is needed:

$$
q\big(x_t \mid x_{<t}\big) \;=\; q\Big(x_t \mid x_{<t},\, \{c_j = g(x_{a_j:b_j}) : b_j < t\}\Big)
$$

exactly (the fuse codes are redundant information, already implied by $x_{<t}$ -- cross-attending
to them is just a computational/parameterization convenience for the model, the same way any AR
LM's hidden state is itself a deterministic function of its own causal history). The chain rule
therefore stays an **exact identity**, precisely Part 2's case:

$$
\log q(x) = \sum_{t=1}^{L} \log q(x_t \mid x_{<t})
$$

so

$$
\mathrm{bpb}_{\text{qcute\_zero}} = \frac{1}{L\ln 2}\sum_{t=1}^{L}\mathbb{E}\big[-\log q(x_t\mid x_{<t})\big] \;\ge\; \frac{H(p)}{L}
$$

with the gap coming *only* from model fit, exactly like a plain byte-level AR LM (`qcute.bytelm`)
-- **not** an ELBO-style bound like `qcute_v1`'s `stack`/`stack_local` (Parts 3, 5). This also
explains why `qcute_zero` doesn't inherit `qcute_v1`'s "bpb only counts `decode_losses[0]`,
missing `encode_losses[1:]`" gap (Part 4): there is no separate code-transmission cost to add in
the first place, because the fuse codes carry no information beyond what $x_{<t}$ already
contains -- nothing is being "sent" that isn't already paid for by the ordinary byte NTP loss.
The teacher-forced `bpb` reported for `qcute_zero` is therefore already the complete, valid
quantity as-is, with the same validity guarantee as a standard autoregressive LM -- the fuse
mechanism is best understood as extra internal *feature engineering* for the conditional, not a
lossy encode/decode split.

## 7. Summary: what parallels the AR chain rule, and what doesn't

| | AR LM (Part 2) | `stack_local` decode (Part 3) | `stack` (sequential) decode (Part 5) | `qcute_zero` (Part 6) |
|---|---|---|---|---|
| Factorization | chain rule, exact identity | ELBO via Jensen, an inequality | ELBO via Jensen, an inequality | chain rule, exact identity |
| Bound tightness | equality up to model fit only | equality up to model fit **and** encoder losslessness | same as `stack_local` | equality up to model fit only (same class as plain AR LM) |
| What must be counted | $-\log q(x_t\mid x_{<t})$ | $-\log p(c)$ **and** $-\log q(x\mid c)$, both | $-\log p(c_i)$ **and** $-\log q(x\mid c_{1:M})$, both | $-\log q(x_t\mid x_{<t})$ only -- fuse codes add nothing to count |
| Why | n/a | $c_i = f(x_{B_i})$ depends on same-block bytes not yet generated: $c_i \notin \sigma(x_{<t})$ | same as `stack_local` -- own-block code, same circularity | fuse codes are strictly past-derived: $c_j \in \sigma(x_{<t})$ always, so folding them in is free |
| $q(x\mid c)$'s expressiveness | n/a | block-diagonal only (no cross-block visibility) | cross-block self-attention (richer, tighter potential bound) | n/a (no encode/decode split at all) |
| Validity class vs. plain AR LM | -- | bound only (ELBO) | bound only (ELBO), same as `stack_local` | **identical to plain AR LM** |

## 8. Would retargeting `qcute_v1`'s decode from own-block reconstruction to next-block prediction make it exact, like an AR LM?

Question (chat, 2026-08-23): instead of decoding block $i$'s bytes from block $i$'s *own* code
$c_i = f(x_{B_i})$ (the autoencoder-style target Parts 3/5 identify as the source of the
ELBO-only bound), retarget decode to predict block $i$'s byte *content* from the **upper level's
code at block $i-1$** -- i.e. $c_{i-1}$, produced by the upper-level encoder's own causal,
unconditional NTP pass over blocks $1,\dots,i-1$ only (exactly `Encoder.forward`'s own mechanism,
already documented as non-circular -- see its docstring's "path (b): upper-level LM predicts the
next code directly"). Does this make `qcute_v1`'s bpb an exact chain-rule identity rather than
just a bound?

**Yes.** The entire reason Parts 3 and 5 needed the Jensen/ELBO step was that $c_i = f(x_{B_i})$
depends on bytes of the *same* block being predicted -- $c_i \notin \sigma(x_{<t})$ for $t$ inside
block $i$. Swap the conditioning code to $c_{i-1}$, produced purely from blocks strictly before
block $i$:

$$
c_{i-1} = h\big(x_{B_1}, \dots, x_{B_{i-1}}\big)
$$

Then for *every* byte position $t$ in block $i$, $c_{i-1}$ is a deterministic function of bytes
strictly before $t$ (all of blocks $1,\dots,i-1$ finish before block $i$ starts), i.e.
$c_{i-1} \in \sigma(x_{<t})$ always -- exactly Part 6's condition for `qcute_zero`'s fuse codes.
The identical argument applies verbatim: folding a variable already computable from the
conditioning set into that set is measure-theoretically free, so

$$
q\big(x_t \mid x_{<t}\big) = q\big(x_t \mid x_{<t},\, c_{i-1}\big) \quad\text{exactly, no Jensen step needed}
$$

and the chain rule over $t$ stays an exact identity, giving `bpb` the same validity class as a
plain AR LM (Part 2) / `qcute_zero` (Part 6) -- tightness purely from model fit, no separate
$p(c)$ transmission term required, no missing-term gap (Part 4) to worry about.

**What this costs -- updated 2026-08-23 after actually implementing it: the switch itself is
cheap, only the statistical cost is real.** This was originally written as if retargeting meant
abandoning a mechanism -- it doesn't. Implemented directly as `Config.own_code_min_lag` (0 =
current own-block default, 1 = this retargeting): the *only* code that changes is
`encode_like_self_attn_decode`/`seed_query_decode`'s cross-attention mask, whose lower bound moved
from `block_lag >= 0` to `block_lag >= min_lag` -- a two-line diff, no new call sites, no
mechanism removed (`own_block_cross_attn_decode`/`own_block_decode_loss` is `StackDecoderV1`'s
separate, legacy mechanism, untouched either way). Both functions already read `cfg.own_code_min_lag`
internally, so every one of the 4 call sites (training `decode_level` plus all 3 generation/
diagnostic paths) picks up the new behavior automatically and consistently -- switching between
autoencoder and predictive-LM training is a single config flag on an existing model class, not a
separate lineage or a rewrite (`docs/status.md`'s 2026-08-23 `own_code_min_lag` POC entry has the
full implementation writeup, and `configs/v1_causal_decode_poc/` a running A/B comparison).

So the real, remaining cost is purely statistical, not architectural:

- The training signal becomes strictly weaker per block: $c_i$ trivially carries near-complete
  information about $x_{B_i}$ (it was computed from those exact bytes), whereas $c_{i-1}$ only
  carries whatever the upper-level encoder's own NTP forecast could extract about block $i$'s
  content from *prior* blocks -- genuinely predictive, but only as good as adjacent-block
  correlation actually is (the same "is the upper LM earning its keep over just re-encoding"
  question CLAUDE.md's "Non-recurrent upper-level plan" already raises, 2026-08-21).
- With `min_lag=1`, `qcute_v1`'s decode becomes a special case of exactly what `qcute_zero` already
  does (Part 6): a byte's cross-attention target is always strictly past-derived. Run that way, the
  two lineages no longer differ in the property this whole document is about -- only in incidental
  details (shared vs. per-stage weights, curriculum, etc.) -- but nothing stops running the *same*
  `qcute_v1` codebase in either mode, config to config.

So: provably closes the bpb-validity gap, cheap to turn on (one flag), but a genuine statistical
tradeoff remains (own-block reconstruction signal for AR-LM-valid bpb) -- not a strictly-better
free change, just no longer an expensive one either.

### 8.1 Restated: does this retargeting break "free rollout, then decode" (Part 9.2), or not?

**No -- the two-stage pipeline itself survives unbroken; only decode's per-block fidelity
degrades.** Separate the mechanism into its two independent halves:

1. **Rolling out the upper-level code sequence** ($c_1, c_2, \dots$, via the upper encoder's own
   causal NTP, `Encoder.forward`, unchanged) never depended on which code decode happens to
   consume -- it only ever reads *previously produced* codes, exactly as before. This half is
   untouched by the retargeting.
2. **Decoding block $i$'s content from a code** is the only thing that changes -- and only in
   *which* code is used, not in whether the call can still happen. Under the old scheme, block $i$
   is rendered from $c_i$; under the retargeted scheme, from $c_{i-1}$. In *both* cases that code
   was already available by the time block $i$ is reached in the same causal rollout (the old
   scheme's $c_i$ is produced at exactly the point the upper LM's own rollout reaches index $i$,
   same timing as $c_{i-1}$ one step earlier) -- so mechanically, decode still consumes exactly one
   code per block either way (per Part 9.2's correction: decode itself still runs its normal
   sequential per-byte loop *within* the block, using that one code throughout -- the index shift
   changes which code that is, not how many are needed). The pipeline "cheaply roll out codes,
   reusing one per block's worth of sequential decode" is not structurally broken by the index
   shift.

**What genuinely changes is fidelity, not mechanism.** $c_i$ was trained to be *sufficient* for
block $i$'s specific content (near-total leakage by construction); $c_{i-1}$ only ever carries
whatever the upper LM's forecast could extract about block $i$ from strictly earlier content --
weaker, genuinely predictive, not reconstructive. So decode(sampled $c_{i-1}$) will tend to
produce a blurrier, more generic rendering of "whatever's typical given the recent gist" rather
than a faithful realization of one specific pre-authored block -- the same fidelity cost already
flagged in the main Part 8 discussion above, just now framed against Part 9.2's free-rollout
mechanism specifically rather than against training-signal strength in general. This also likely
erodes how well a *fully parallel*, one-shot block render (`stack_local`'s style) can work
specifically -- a code that only forecasts a block, rather than one custom-derived to determine
it, gives parallel same-block positions much less to jointly agree on, so good quality plausibly
needs *more* real within-block causal structure (byte-by-byte, using $c_{i-1}$ as coarse
conditioning) than a pure one-shot parallel render can offer -- eroding, not eliminating, the
$K\times$ amortization advantage, rather than removing the free-rollout mechanism altogether.

**A side benefit worth noting**: the *old* scheme's free rollout has its own quiet, easy-to-miss
validity gap that the retargeted scheme removes. Decode was only ever trained on *real* own-block
codes ($c_i = f(x_{B_i})$, never a merely-plausible sampled stand-in, absent `encoder_ste_p`
specifically forcing otherwise) -- so feeding it a *sampled* code at generation time is a genuine
train/generation distribution shift specific to this architecture, the exact thing
`encoder_ste_p`/scheduled sampling exists to paper over, and CLAUDE.md already records that
scheduled sampling "empirically hurt, not helped" in this codebase -- an open, unresolved problem.
The retargeted scheme instead only inherits *ordinary* exposure bias (train on real
$x_{B_1},\dots,x_{B_{i-1}}$, generate from the model's own previously-generated blocks) -- the same
universal, well-understood issue every autoregressive model already has, not a bespoke mismatch
unique to sampling own-block codes.

## 9. Two generative-modeling paradigms: LM-with-autoencoder vs. LM-with-context-compression

Parts 3/5/6/8 all reduce to one structural choice: where is a code $c$ allowed to draw its
information from, relative to the content it conditions? That choice splits every code-augmented
byte model in this codebase into exactly two paradigms:

- **LM with autoencoder** (`qcute_v1`'s `stack`/`stack_local`): $c_i = f(x_{B_i})$, drawn from the
  *same* content it later helps decode -- code and content are entangled by construction, a real
  encoder/decoder codec sitting inside the generative model.
- **LM with context compression** (`qcute_zero`): $c_j = g(x_{a_j:b_j})$ with $b_j < t$ always --
  drawn only from content strictly earlier than anything it conditions. The code is a lossy
  summary of the *past*, never of what it is used to predict.

### 9.1 Circularity

Autoencoder: $c_i \notin \sigma(x_{<t})$ for $t$ inside block $i$ -- the code carries information
from bytes not yet generated at that point, forcing the Jensen/ELBO step (Parts 3, 5, 8); bpb is
only a bound, and a valid one requires separately counting $p(c)$ (Part 4's currently-missing
term).

Context-compression: $c_j \in \sigma(x_{<t})$ always -- the code is redundant given the causal
history, so folding it into the conditioning set is free (Part 6); bpb is an exact chain-rule
identity, same validity class as a plain AR LM, with nothing extra to count.

### 9.2 Free rollout ("plan the block, then render it") -- and what this amortizes, precisely

**Correction (2026-08-23, after directly checking every `qcute_v1` generation path -- all 8
occurrences of `for t in range(K)` across `_stack_generate_blockwise` and its `StackDecoderLocal`
override): "render an entire block in one shot, no per-byte stepper" overstated what actually
runs.** Every generation path, `stack_local` included, still generates a block's $K$ bytes one at a
time, teacher-forcing each sampled byte back in before predicting the next -- `stack_local`'s
"block-local, parallel" property is about *training* (all blocks computed in one batched matrix op,
since they don't depend on each other there), not about skipping per-byte sampling *at generation
time*. What genuinely *is* amortized, precisely: the upper-level code is sampled once per $K$
bytes (`Encoder.forward`'s "path (b)", `generate_level_codes`, one cheap upper-LM step) and then
*reused* across all $K$ of that block's sequential decode steps, rather than needing a fresh
upper-level sample at every byte. That is the real, verified saving -- not a zero-per-byte-cost
one-shot render.

Autoencoder: because $c_i$ was trained to be *sufficient* to reconstruct block $i$'s specific
content (a genuine codec), decode's $c \to \text{content}$ mapping works on a *sampled* code too,
not only the real one -- nothing ground-truth-dependent in `decode`'s use of $c$ once $c$ exists.
This licenses the two-stage generation just described: roll out the coarse code cheaply, one per
$K$ bytes, *then* run the (still sequential, per-byte) decode using that single code throughout.

Context-compression has no analogous shortcut, but the shortfall is *exactly* this
one-code-per-$K$-bytes reuse, not "one-shot rendering" (which, per the correction above, was
never real anywhere in this codebase). $c_j$ only ever exists as a function of bytes *already
produced* -- there is no code that stands in for a block's content ahead of generating that block,
because such a code would need to summarize bytes that don't exist yet, exactly the circularity
Part 9.1 shows context-compression codes are built to avoid. Generation is therefore necessarily
byte-by-byte causal throughout, with fresh fuse codes re-derived from whatever has already been
produced as generation proceeds -- there is no way to amortize even the *coarse-code* sampling
into one cheap upstream step the way the autoencoder paradigm can (`qcute_zero`'s fuse mechanism
does already reuse one *already-derived* code across many later bytes' cross-attention, same as
the autoencoder side -- what it lacks is specifically the ability to *sample a code ahead of its
own chunk's content*, confirmed missing by direct inspection in Part 12).

### 9.3 Speculative decoding -- available to both, and *not* the same axis as 9.1/9.2

MTP-style speculative decoding (draft several next bytes cheaply from linear heads reading one
already-computed hidden state, then verify each one against the real exact incremental stepper,
accepting up to the first disagreement) is a *different, orthogonal* mechanism from 9.2's
block-level free rollout -- it drafts individual next tokens from a hidden state, not whole blocks
from an upstream code, and it always verifies against the model's own true output rather than
committing to a draft unconditionally. It requires nothing more than a well-defined,
causal-enough per-position hidden state to draft from and verify against -- a requirement both
paradigms satisfy equally: `qcute_zero`'s shared-LM hidden state, or `qcute_v1`'s track0 hidden
state within a block (itself conditioned on that block's own code, whatever its bpb-validity
status). `qcute_zero` already implements this exactly (`generate_speculative`, `mtp_heads` drafting
verified against `generate_kv_cache`'s `_make_incremental_stepper`, `qcute_zero.py`). `qcute_v1`
has no equivalent implemented, but nothing about the circularity/free-rollout distinction blocks
adding one -- the same MTP-head-draft-then-verify scheme could equally sit on top of track0's
per-byte generation within a block, since verification is always against the model's own real
output regardless of whether that output's own bpb happens to be exact or only a bound.

**So**: circularity (9.1) and free rollout (9.2) are genuine, structural differentiators between
the two paradigms -- a code either can or cannot stand in for content that doesn't exist yet, and
that single fact determines both properties together. Speculative decoding (9.3) is not on that
axis at all: it is equally available (implemented, for `qcute_zero`; straightforward to add, for
`qcute_v1`) to either paradigm, because it only needs a causal hidden state to draft and verify
from -- not any particular relationship between a code and the content it summarizes.

## 10. Which paradigm fits which domain (text / audio / image / video)

Corollary of Part 9 (chat, 2026-08-23): is `qcute_v1`'s autoencoder paradigm mainly suited to
non-text domains, and `qcute_zero`'s context-compression paradigm right for text but too slow
there unless paired with speculative decoding? **Broadly yes**, and Part 9's two structural
properties (9.1, 9.2) are exactly why:

- **Autoencoder (`qcute_v1`) fits domains with high local redundancy** -- image patches, audio
  frames, short video clips -- where a block's raw samples are largely determined by a much
  lower-dimensional underlying signal, so one learned code genuinely *can* be close to sufficient
  for reconstructing that block (9.2's premise holds well). This is exactly the standard toolkit
  those domains already use successfully (VQ-VAE/RVQ + an autoregressive code prior, EnCodec-style
  audio codecs, VQGAN-style image tokenizers) -- fields where rate-distortion bpb bounds (Part 4)
  and perceptual/reconstruction quality, not exact per-symbol entropy, are already the accepted
  measure of success. The ELBO-only bpb (9.1) isn't a real cost there; it's the native framing.
- **Context-compression (`qcute_zero`) fits text specifically** because text's dominant successful
  paradigm *is* exact per-token density estimation -- perplexity/bpb genuinely correlates with
  downstream quality there (Part 6's guarantee is the whole point), and natural-language bytes
  don't have image/audio's kind of "a few learned numbers nearly determine this whole chunk"
  redundancy -- committing early to a block-level code tends to be a much weaker prior for text
  than for pixels.
- **The cost, precisely per 9.2**: `qcute_zero` has no block-amortization shortcut, so text
  generation is necessarily byte-by-byte causal -- a genuine throughput cost `qcute_v1`-style
  domains don't pay (their code-then-render step amortizes a whole block, roughly a $K\times$
  speedup, in one cheap upstream sample). Ordinary MTP-style speculative decoding (9.3) is the
  realistic mitigation, but it only buys a *constant-factor* speedup bounded by how often the
  cheap draft agrees with the model's own true greedy choice -- not the compression-ratio-scale
  ($K\times$) speedup block amortization gives when it works. For text this constant factor is
  often decent (next-token distributions are frequently locally confident), which is exactly why
  "too slow unless speculative decoding" is the right characterization rather than "unusably slow."
- **Caveat -- not a strict either/or.** `qcute_v1` runs on text today (this whole codebase's
  `enwik8` testbed), it just can't claim an exact bpb without the missing `p(c)` term (Part 4).
  Symmetrically, `qcute_zero`'s context-compression codes could in principle be built for
  image/audio/video too -- but forfeiting within-patch/within-frame block amortization is a much
  larger cost there than for text, since those domains' raw sample counts (millions of pixels per
  image/frame) make pure byte-by-byte causal generation far less tractable than it is for text,
  even with speculative decoding's constant-factor help. Domain redundancy structure, not the
  input modality label itself, is what actually determines which paradigm fits.

### 10.1 For/against, restated in objective terms: codec training vs. predictive-LM training

Two follow-on questions (chat, 2026-08-23), moved here from `docs/status.md`'s 2026-08-23 entry of
the same title: given Part 8 shows `qcute_v1`'s own-block reconstruction only yields an ELBO
bound while `qcute_zero`'s predictive fuse gives an exact bpb, why ever train as a codec at all?
And can predictive codes still support "plan then fill" generation?

**Why ever prefer `qcute_v1`'s own-block reconstruction, given it only yields an ELBO bound, not
an exact bpb?** Because the two designs are training the code for genuinely different purposes,
and bpb-exactness isn't the only thing worth optimizing for:

- `qcute_v1`'s $c_i = f(x_{B_i})$ is trained so that $q(x\mid c_i)$ can reconstruct that *specific*
  block -- a real rate-distortion/codec objective. The code is guaranteed (by the training
  objective itself) to be a faithful, self-contained, standalone representation of its own block:
  decode it in isolation and you get that block's content back. This is exactly the property the
  original project goal (`docs/archive/continuous_tokenizer_handover.md`, "qcute builds a
  tokenizer") needs -- a code meant to be reused downstream (indexed, edited, fed to a separate
  coarser model) has to actually mean something about its own block, not merely be "whatever
  helped forecast the future."
- `qcute_zero`'s fuse codes have no such guarantee or objective -- they are trained purely to be
  *useful for predicting bytes that come after them*. There is no dedicated code-to-content
  decode step at all, so a fuse code cannot be meaningfully decoded in isolation; its only
  property is "helps the shared LM's next-token prediction," which is weaker and less
  interpretable than "faithfully represents this block."
- So the choice is precisely the Part 8 tradeoff restated in objective terms: exact-bpb-validity
  and pure forecasting utility (`qcute_zero`) vs. a genuine, inspectable, reconstructible codec
  (`qcute_v1`), at the cost of the bpb number only being a valid bound (Parts 3/5), not an exact
  quantity, unless the missing $p(c)$ term (Part 4) is added in.

**Can predictive (`qcute_zero`-style) codes still support "roll out the upper code LM, then
reconstruct/decode a whole block from the sampled code," the way `qcute_v1`'s
`generate_level_codes` + `decode_level` does?** No -- genuinely different, not just an
implementation gap, for the same reason as the bpb question. `qcute_v1`'s decode can do this
because $c_i$ was trained specifically to be *sufficient* to reconstruct block $i$'s real content
(it was computed from that content) -- so even a *sampled* (not ground-truth) code, once decode
treats it as if it were real, still gets turned into a fully-committed, specific byte sequence via
a real code$\to$content mapping. `qcute_zero` has no such mapping: a fuse code is only ever a
function of bytes *already decided* (Part 6's whole point, and the reason its bpb is exact), so by
construction it can never summarize or stand in for content that hasn't been generated yet -- there
is nothing to "reconstruct from" a not-yet-real block's code, because that code doesn't exist
before the block does. Generation in `qcute_zero` is therefore necessarily byte-by-byte causal
(the shared LM's ordinary forward pass, periodically re-deriving fuse codes from whatever has
already been produced), not a two-stage "decide the gist, then render it" scheme.

This is the same tension underlying Part 8's tradeoff, restated for generation: a code cannot
simultaneously (a) depend only on the past, which is what exact bpb validity and genuine
forecasting require, and (b) be sufficient to determine/reconstruct content that is strictly in
its own future, which is what a "plan the block, then fill it in" generation scheme requires.
`qcute_v1` chose (b); `qcute_zero` chose (a); no single code can have both properties at once.

## 11. How to actually interpret `qcute_v1`'s reported train/val bpb

Question (chat, 2026-08-23): given `qcute_v1`'s reported train and val bpb numbers, how should
they be read against a model with a genuinely exact bpb (a plain AR LM, `qcute.bytelm`)?

**They are not the same quantity, and are not directly comparable as reported.** Per Part 4, the
codebase's `bpb`/`bpb_full` currently only accumulate `decode_losses[0]` -- i.e.
$-\log q(x\mid c{=}f(x))$, evaluated at the *real* code computed from that same block. Per Part 4,
a valid bound needs $-\log p(c)$ added in too. Without it, what's reported is not "$\mathrm{bpb}
\ge H(p)/L$" at all -- it's the *conditional* reconstruction cost given a free, uncosted oracle
code, which will look artificially low purely because $c$ was computed from the very bytes it's
reconstructing (near-total information leakage by construction, scaling with how much capacity
`vocab`/`pq_chunks` give the code). A very low decode-only bpb says "this codec's decoder is doing
its job well given its code," not "this is a strong density model of the data" -- the two claims
sound similar but are not the same statement.

**Making it comparable requires the missing term, and even then it's only a bound.** Report
$\mathrm{bpb}_{\text{total}} = (\text{decode\_losses[0]} + \sum_{i\ge 1}\text{encode\_losses}[i]) /
(L\ln 2)$ (properly normalized per byte, not per code) to get Part 3/5's actual ELBO quantity:
$\mathrm{bpb}_{\text{total}} \ge H(p)/L$, a genuine upper bound, directly comparable in the same
units and same *direction* of inequality as `qcute.bytelm`'s own reported bpb (which is Part 2's
plain cross-entropy bound, tight up to model fit only). But note the two bounds are not equally
tight in general: `qcute_v1`'s bound carries an *extra*, structural gap beyond ordinary model-fit
looseness -- forcing a **hard/deterministic** encoder ($c=f(x)$, a point mass, `code_hard=True`)
is itself a specific, generally suboptimal choice of the importance-sampling distribution $q(c\mid
x)$ in Part 3's derivation; the true posterior $p(c\mid x) \propto p(c)\,q(x\mid c)$ implied by the
model is not, in general, a point mass, so committing to a single hard code discards information a
softer encoder could have kept. This is the same "amortization/approximation gap" familiar from
discrete-VAE literature -- it doesn't vanish with more training, only with a better choice of
encoder distribution. So even a fully-counted $\mathrm{bpb}_{\text{total}}$ should be *expected*
to sit somewhat above a well-trained AR LM's bpb on the same data, not merely equal to it; a
`qcute_v1` number that comes in *lower* than `qcute.bytelm`'s on the same slice is a red flag that
the comparison still isn't fair (almost certainly the missing-encode-term undercount, Part 4),
not evidence `qcute_v1` is winning.

**Train vs. val, read separately:**

- **Train bpb** (decode-only, as currently reported) mostly measures whether the fixed-capacity
  code+decoder pair has enough functional capacity to memorize/reconstruct this specific training
  slice -- closer to a lossy-codec reconstruction-error curve than a language-modeling perplexity
  curve. A very low value reflects codec capacity more than generalizable predictive quality.
- **Val bpb** (decode-only, held-out data) tests whether the *codec* generalizes -- given the val
  block's own real code (computed by the trained encoder from that unseen block's own bytes), can
  decode reconstruct it -- i.e. a rate-distortion generalization check, still circular per block,
  just on unseen data. It says nothing about the encoder's own forecasting quality
  (`encode_losses`, still excluded), which is exactly the piece free-rollout generation actually
  depends on.
- This is precisely why low teacher-forced bpb/high byte_acc coexisted with pure-garbage free
  rollout earlier this session (`StackDecoderLocal`, `docs/status.md`'s 2026-08-23 entry):
  decode-only bpb, however low, simply doesn't measure the thing free-rollout generation quality
  depends on.

**Practical rule of thumb**: use `qcute_v1`'s reported train/val bpb only for *relative*
comparisons among `qcute_v1` variants (e.g. this session's sharing-ablation grid, comparing codec
quality config-to-config), never read as an absolute cross-entropy number, and never compared
directly against `qcute.bytelm`'s bpb unless the encode-loss term is added in on `qcute_v1`'s side
first. For anything resembling "does this generate well," check free-rollout qual-gen samples
directly -- bpb/byte_acc alone, at any value, is not sufficient evidence either way.

## 12. Could `qcute_zero` get `qcute_v1`-style free rollout / `stack_local`, and would it stay valid?

Three related questions (chat, 2026-08-23): what structurally blocks `qcute_zero` from getting
Part 9.2's free-rollout feature; is a `stack_local`-style block-local parallel decoder feasible
inside `qcute_zero`; if built, would its bpb be exact or only a bound; given `qcute_zero`'s
`FuseStage` already supports fully independent, unshared per-stage parameters (confirmed directly
in code -- `FuseStage`'s own docstring: "own weights throughout (no cross-stage sharing)"), is
capacity/parameter-sharing actually the limiting factor?

**What's blocking free rollout -- specific to `qcute_zero`'s current wiring, and it's the same
mathematical fact as everywhere else in this document, not a missing feature.** Part 9.2
established free rollout needs a code that can *stand in for* a block's content before that block
exists -- $c_i$ sufficient to reconstruct $x_{B_i}$, sampled *ahead of* generating $x_{B_i}$
itself. `qcute_zero`'s exact-bpb guarantee (Part 6) is built on the opposite property by
construction: every fuse code satisfies $c_j \in \sigma(x_{<t})$, derived *only* from bytes already
generated -- there is no code anywhere in `qcute_zero`, *as currently trained*, that summarizes a
future block, because building one would require computing it from bytes that don't exist yet,
exactly the circularity Part 6 shows the architecture is designed to avoid. This isn't a
missing-feature gap that more engineering closes -- it's the direct structural cost of the same
property that makes `qcute_zero`'s bpb exact in the first place (Part 9's "can't have both"). This
confirms the fourth question's premise directly: independent per-stage `FuseStage` weights don't
change this at all -- capacity/sharing was never the limiting factor, causality is. (There's also a
concrete empirical precedent: `qcute_zero_parallel`, the original query-vec fork that *did* attempt
something in this direction -- a query slot standing in for an upcoming chunk's gist -- was found
weaker than plain MTP-head drafting per position covered and per attention-stack cost paid, and was
pruned from `qcute_zero` proper on 2026-08-22 in favor of `generate_speculative`.)

**Correction (2026-08-23, after actually reading `forward`/`generate_no_cache`'s code path rather
than assuming symmetry with Part 8): this direction is NOT equally cheap, and splits into two
genuinely different-difficulty pieces.** Loosening the *mask* alone -- letting byte $t$ in chunk
$i$ see chunk $i$'s own code, not just earlier chunks' -- really is cheap and symmetric to
`own_code_min_lag`: `code_pos_abs = (block\_idx{+}1){\cdot}cum\_K - 1` and
`fuse_mask = causal_mask(byte_pos, code_pos_abs, window)` already exist, and a variant admitting
same-chunk visibility reuses `FuseStage` unchanged. But that alone only gets you option (A)'s
teacher-forced circularity (still bound-only, still requires the chunk's real bytes to compute
the code) -- it does *not* get you free rollout. The free-rollout half is genuinely missing, not
just masked off: in *both* `forward` (training) and `generate_no_cache`/`generate_kv_cache`
(generation), every code is extracted from `cur_h` -- the trunk's real hidden state at that
chunk's own last byte position -- so it always requires that chunk's real bytes to already exist,
at training time *and* generation time alike. The code-sequence NTP pass (`h_code =
self._run_blocks(code_embeds, ...)`) is already a genuinely causal model over the code stream
(same mechanism as `qcute_v1`'s `Encoder.forward`), but nothing currently *samples* a code from it
ahead of its own chunk's bytes existing -- that sampling routine (read `h_code[:, -1, :]`, sample
the next code, exactly `qcute_v1`'s `sample_next` pattern) plus a generation-time wiring to feed it
into the loosened-mask `FuseStage` before that chunk's bytes are generated, is new code, not a
flag. So: moderate, scoped, and buildable from existing load-bearing pieces (`FuseStage`, the code
NTP, `causal_mask`) -- but not the one-line change `own_code_min_lag` turned out to be.

**Is `stack_local` inside `qcute_zero` feasible?** Mechanically yes -- nothing stops bolting a
per-block parallel decode head onto the shared trunk, using `FuseStage`'s already-supported
independent weights. But there are exactly two ways to source the code it conditions on, and each
one reduces to something that already exists in this codebase rather than a new capability:

- **(A) Use an own-block code**, $c_i = f(x_{B_i})$, the only way to get `stack_local`'s actual
  value (good decode fidelity, since the code was custom-derived to determine that specific block
  -- note per Part 9.2's correction this is fidelity of a still-sequential-within-block decode, not
  a literal one-shot render, which nothing in this codebase currently does). This reintroduces
  exactly Part 3's circularity ($c_i \notin \sigma(x_{<t})$) --
  the resulting bpb is only an ELBO bound, not exact, for the identical reason `qcute_v1`'s
  `stack_local` is a bound. This is just re-deriving `qcute_v1`'s `StackDecoderLocal` inside
  `qcute_zero`'s shell -- no advantage over using `StackDecoderLocal` directly.
- **(B) Use only strictly-past-derived codes**, to preserve exactness. Bpb stays exact (Part 6's
  argument still applies), but you lose real one-shot fidelity: jointly predicting several
  not-yet-decided positions from only stale context is conditionally-independent/non-autoregressive
  sampling, the well-known weaker regime (the multimodality/incoherence problem familiar from
  non-autoregressive translation) -- at best this reduces to `generate_speculative`'s existing
  parallel MTP drafting (still verified byte-by-byte against the truth), not a new "block-local
  decode" capability.

**So**: it can be built, but whichever branch is taken lands back on a mechanism that already
exists (`qcute_v1`'s `stack_local`, bound-only; or `qcute_zero`'s own `generate_speculative`,
exact) rather than a genuinely new combination of exact bpb *and* `stack_local`-quality one-shot
block rendering. This is Part 9's "a code can't have both properties at once" restated once more,
now specifically against this proposed feature -- consistent with every other angle this document
has checked it from.

## 13. Stepping back: yes, `qcute_v1`'s own cascade already contains a way to do both -- constructively

Question (chat, 2026-08-23): `qcute_v1`'s cascade already has each level's encoder modeling the
level below's code stream with a genuinely exact, causal NTP (`Encoder.forward`, confirmed by
direct inspection: pure self-attention over its own input stream, no cross-level attention at
all -- level $j{+}1$'s NTP never reads anything from level $j{+}2$ or above). Given that, is there
a way to free-roll-out the hierarchy *and* decode all the way back to level-0 bytes *and* keep the
resulting bpb exact, rather than the ELBO-bound status Parts 3/5/8's *unmodified* decode gives?

**Yes -- and it's Part 8's construction, generalized recursively up the whole cascade rather than
applied once at the bottom.** The key realization: nothing about `Encoder.forward`'s own NTP is the
problem anywhere in this cascade -- every level's own self-attention-only NTP over its own input
stream is already Part 2's exact case, standalone, at every level, today, unmodified. The *only*
place circularity ever enters is decode's own-block conditioning choice ($c_i = f(x_{B_i})$, Parts
3/5). So the fix is not to touch the encoder cascade at all -- it's to apply Part 8's index shift
*at every level's decode step*, not just level 0's:

1. **Top level down**: roll out the coarsest code stream via the top encoder's own exact NTP
   (`generate_level_codes`, already exact, unmodified -- this is Part 2's case already, today).
2. **Each level below**: instead of that level's decode reconstructing chunk $i$ from *that
   chunk's own* just-computed code (own-block, circular), condition it on the *coarser* level's
   code from strictly *before* chunk $i$ -- i.e. Part 8's $c_{i-1}$ substitution, applied at every
   level boundary in the cascade, not only at level 0.
3. **Bottom (level 0)**: decode renders that block's actual bytes from the causally-available
   coarser forecast, exactly Part 8's construction.

Because every step in this chain only ever conditions on strictly-earlier information (the top
rollout is already causal; each intermediate substitution is Part 8's $\sigma(x_{<t})$ argument,
applied recursively level by level), the *composition* stays exact end-to-end -- Part 1's Gibbs
argument doesn't care how many levels the exact chain-rule factorization is nested through, only
whether circularity was introduced anywhere along the way, and by construction it never is. And
because every level still operates at *its own* coarser rate (the top level takes one step per its
own block, decode still renders a whole finer block per call), genuine multi-level amortized free
rollout survives throughout the cascade, not just as a two-level toy case -- this is the full,
recursive version of Part 8.1's two-stage pipeline, not a restricted special case of it.

**What this costs, stated precisely (not newly, just now generalized):** every level's rendering
becomes a genuine *forecast* from a coarser, strictly-past context rather than a lossless-by-
construction *reconstruction* of its own chunk -- so content specificity degrades level by level as
more forecasting-based conditioning stacks up, exactly Part 8's fidelity cost, now compounding
across the whole hierarchy rather than incurred once. Nothing about the encoder cascade needs to
change to get here (it already does its part correctly) -- the entire fix is confined to decode's
choice of *which* code to condition on, applied consistently at every level rather than left at
`qcute_v1`'s current own-block default.

## 14. `qcute_v1` (maximally shared) vs. `qcute_zero`: feature parity or not?

Question (chat, 2026-08-23): given `qcute_v1` can now share weights heavily
(`kv_lm_mode`/`decoder_own_stage_mode="shared"`), achieve exact bpb (`own_code_min_lag=1`, Part 8)
and report it correctly (`bpb_valid`, Part 4's fix), and given `qcute_zero` just gained free
rollout (Part 12's missing piece, now built), is either now a strict subset/superset of the
other? **No** -- both lineages converge on the same *mathematical* point from either side (Part
8/9), but three structural differences remain, none of them a missing flag:

| Dimension | `qcute_v1` | `qcute_zero` |
|---|---|---|
| Default bpb validity | ELBO bound only (`own_code_min_lag=0` default, Parts 3/5) | Exact by construction (Part 6) |
| Can reach exact bpb | Yes -- `own_code_min_lag=1` (Part 8) | Yes -- native, always |
| bpb metric completeness (as logged) | Fixed 2026-08-23: `bpb_valid`/`bpb_full_valid` additive fields (Part 4) -- `bpb` itself still decode-only, kept for `Checkpointer` compatibility | Always complete -- one unified NTP loss, nothing to add |
| Free rollout (code sampled ahead of its own content) | Built-in from the start (`generate_level_codes`, `code_source="pred"`) | Added 2026-08-23 (`generate_free_rollout`) -- single-fuse-stage PoC only, not yet the fast/cached path |
| Weight-sharing scope | Decode reuses the *same-level* encoder only (`kv_lm_mode`/`decoder_own_stage_mode="shared"`) -- still $O(\text{n\_levels})$ distinct trunks, confirmed no cross-level tying exists anywhere | One trunk reused for the byte pass *and* every fuse stage's own code NTP -- $O(1)$, always |
| Per-level capacity | Heterogeneous `d_model`/`n_layers`/`vocab` per level, independently configurable (`Encoder(cfg, d_models[i], n_layers_list[i], vocabs[i])`) | Forced homogeneous -- single shared trunk means every stage uses identical dimensions |
| Code vocabulary | Own PQ-structured combinatorial space (`vocab`, `pq_chunks`), independent of the byte alphabet | Code space *is* the byte alphabet -- one tied embed/output table, flat softmax, no PQ |
| Incremental KV-cache generation | Yes -- `generate_kv_cache`, verified bit-exact against full recompute | Yes -- `generate_kv_cache`/`_make_incremental_stepper`, verified bit-exact across 315 configs |
| Speculative decoding | Not implemented (Part 9.3: nothing blocks adding it) | Implemented and verified (`generate_speculative`, MTP heads) |
| Block-local/parallel same-level decode | Yes -- `StackDecoderLocal` | Not implemented (Part 12's options A/B: buildable, but would cost exact bpb the same way `qcute_v1`'s does if own-block-code) |

**Reading the table**: rows 1-4 are the ones this whole document is actually about (bpb validity,
free rollout) -- both sides can now reach the same point, so neither dominates there anymore.
Rows 5-7 are foundational *design choices*, not gaps: how much to unify weights across levels, and
what a "code" even is (an independent combinatorial alphabet vs. the byte alphabet itself) --
closing either would mean one lineage becoming a rewrite of the other, not a config change. Rows
8-9 are genuine, currently one-sided capabilities (speculative decoding only in `qcute_zero`,
block-local decode only in `qcute_v1`) that plausibly *could* be ported either direction with
moderate effort (like `generate_free_rollout` just was) -- but haven't been.
