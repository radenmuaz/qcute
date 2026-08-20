# Symmetric hierarchy: generalizations beyond qcute_v1

Speculative, forward-looking design notes — not yet implemented, not yet staged. `qcute_v1`
(see [docs/qcute_v1_plan.md](qcute_v1_plan.md)) remains the doc for what's actually built. This
file is where these generalizations live until (if) they get folded into an actual plan.

Throughline across all four: pushing the architecture from "top level is special, everyone else
follows one rule" toward a genuinely uniform/symmetric per-level structure, and re-examining a
few pieces (BOS, the code-prediction target) that currently work but carry more machinery than
they need to.

## 1. A third path to "what should the next code be"

Today there are two ways to obtain a code for content that doesn't exist yet: (a) draft via the
uncond byte LM, encode the draft, decode-refine; (b) ask the level-above's own NTP to predict it.
A third, cheaper option: give decode's own hidden state (after processing a block) a linear head
that predicts the *next* code directly — architecturally the same job level `i+1`'s NTP already
does, just computed from a more grounded hidden state (decode has seen actual reconstructed
content; level `i+1` only ever sees the coarse code sequence).

**Not circular.** Encode's `code_head` (bytes -> code, compression) and this new forecast head
(context -> next code, prediction) are different functions that happen to share a codomain, the
same way `code_head`/`code_predict` already coexist on every `LM` without conflict.

**One shared head is sound, but only between the two forecasters — never with encode.** Decode's
forecast and level `i+1`'s NTP are doing the *same kind* of task (predict the code with
less-than-full information) and can reasonably share weights, which also forces a consistency
constraint between them (similar in spirit to the round-trip check, but built into training
rather than measured after the fact). Encode must stay separate: it has strictly more information
(real bytes) than either forecaster, and tying its head to a forecaster's would force one function
to do two incompatible jobs. Encode also has to remain the single, stable ground-truth producer —
if the target itself were defined by a head that's being trained, it would be a moving target,
recreating the same slow-convergence instability already documented in the archived v5 history
(n_levels=2 cascade effect, `docs/archive4/status.md`).

## 2. Adaptive dynamic chunking (H-Net-style)

Today `K` is a fixed architectural hyperparameter per level. Generalizing to learned, variable
chunk length needs the length signal to flow *through* the model's own computation graph — not
as external side info, a heuristic, or a non-differentiable control channel.

**Sound mechanism**: fold it into the same quantized-code machinery every other discrete decision
here already uses. Add a second small head off the same pooled hidden state that produces the
code, outputting logits over a small *fixed* set of candidate lengths (e.g. `{1,2,4,8}` —
literally the values in `CLAUDE.md`'s Ks regression-grid table), quantized via the same Gumbel/STE
trick as the code itself. Cheapest version: treat it as one more PQ chunk appended to the existing
code (`SimplexQuant`'s `_chunked`/`to_ids` already generalize to this) — the length becomes part
of the same code tensor decode already cross-attends to, no new plumbing.

**Bounding is what keeps this feasible.** An unbounded/continuous length prediction would break
static-shape batching, which the whole parallel-decode goal depends on. A bounded discrete choice
among a small candidate set, computed at `K_max` width and masked down, stays static-shape and
differentiable — the standard pad+mask pattern for variable-length sequences, not a genuinely
variable tensor shape.

**Open**: whether the length choice should be made by the *encoder* (content-driven boundary
detection, H-Net's actual role) or by whatever *drafts* the code at generation time (level `i+1`
deciding how far ahead to draft) — different points in the pipeline, only one of which is
causally available during real generation. Not resolved.

## 3. Removing the special-cased top level: a uniform stack-cross-attn-parallel architecture

Hypothetical: every level (including the current top, and level 0) has the identical structure —
self-attention over its own code history + cross-attention to the level above *and* the level
below, simultaneously + MLP + NTP head + code head. No level is structurally special.

**Removing the top level's special self-code-recurrent path is a real simplification, not just a
neutral change.** The top level's own sequence already *is* a sequence of discrete codes. Plain
unbounded self-attention over that sequence already gives direct access to every earlier code
value — there's no need for the old mechanism's extra step of additionally quantizing a "self-code
summary" on top of values that are already the summary. Unifying it into the same structure as
every other level should be strictly equivalent or better.

**Cross-attending up (to the coarser level) is fine, given the lag already established** ("lead
one block ahead" — level `i+1`'s code for a past span was derived from level `i`'s own
already-produced content there; settled information, no new risk).

**Cross-attending down (to the finer level) is the one place a genuine cycle is possible, not just
a soft risk.** If level `i` attends to level `i-1`'s content for the *same* span it's currently
producing a code for, and level `i-1` needs level `i`'s code for that same span before it can
decode it at all, that's two things depending on each other at the same timestep — a true cycle,
not resolvable by reordering. The fix is mandatory, not a tunable knob the way the sync/async
self-attention window is: downward cross-attention must be restricted to strictly-past spans of
the level below. That restores a valid DAG (level `i`'s prediction for span `X` only ever depends
on strictly-past information from itself and both neighbors), but it also means downward
attention can only ever supply *historical local texture* — never grounding for the exact span
being decided, since that would be exactly the circular case being excluded.

**Open**: what "own level code" means at level 0 specifically — the real quantized `c_list[0]`
(as everywhere else), or the raw bytes treated as a trivial/identity code. Determines whether
level 0 is a genuine base case of a uniform recursion or still a special case wearing the same
interface as everyone else.

## 4. BOS without interleaving: register/sink tokens

The current mechanism (`bos_interleaved_self_attn`) inserts a real BOS token before every
`K`-block, physically inflating the sequence (`(n_blocks, K)` -> `(n_blocks, K+1)`, stripped back
out afterward). It works, but couples two things that don't need to be coupled: seeding the first
query of a block (the actual "position -1" problem) and marking block boundaries. Boundary
marking turns out not to be load-bearing at all — that's already fully determined by the static
`(n_blocks, K)` reshape, independent of whether a BOS token is physically present. So BOS is only
really needed for the seeding role, and interleaving is an expensive way to provide it — it
inflates the content stream, which is specifically awkward for unconditional generation, where
you want a clean stream of real content, not one periodically interrupted by synthetic markers
that have to be inserted and stripped procedurally.

**Register/sink tokens provide the same seeding role without living in the sequence at all.** A
small fixed set of learned KV pairs (parameters, like `self_code_const` already is, just used
differently) concatenated once as extra always-available keys/values for the whole self-attention
computation — never reshaped, never indexed per-block, never part of the positional sequence.
Block-start queries with no real history fall back to attending only to the sink; later positions
attend to the sink plus real history. No token insertion, no stripping, content stream stays
exactly as long as the actual content.

Causal (fixed parameter, no content dependence, safe to attend from anywhere) and static-shape
(actually simpler than today: plain `(n_blocks, K)` content reshaping with a constant number of
extra sink slots appended once, no `K+1`/strip dance).

**A query-only BOS variant (BOS used only as a query, never re-attended as a key) was considered
and is a valid middle point, but inherits more of the original awkwardness than it avoids** — it
still occupies a per-block sequence slot (same `K+1` bookkeeping as today) and adds an asymmetric
masking rule (real positions must never attend back to a BOS slot). The register/sink version
gets the identical seeding effect without either cost, since the sink is symmetric by
construction — it was never part of the positional sequence, so no special rule is needed to keep
anything from attending "back" to it.
