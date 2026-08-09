# Two-stage latent-variable decode (self-conditioning + `.detach()` + independent drafter) — math

Design spec for the direction discussed in-session (not yet implemented in
`qcute/qcute_refine_v4_4.py`). Notation mirrors that file's actual names
(`Config.Ks`, `decode_windows`, `code_kv`, `gumbel_quantize`, `decode_code_ste`)
so each equation below maps directly onto a change to a specific function.
Written to be checked, and to be portable to code without re-deriving
anything — flag anything that doesn't type-check.

## 1. Setup and notation

$n$ levels, $i = 0, \dots, n-1$, local compression factors $K_0, \dots,
K_{n-1}$ (`cfg.Ks`), shared vocabulary size $V$ (`cfg.vocab`), shared model
width $D$ (`cfg.d_model`) — full weight sharing across all levels (one
`embed`, one stack of `blocks`, one `ln_f`, aliased across every `LevelLM`).

Level $0$'s own input **stream** is the raw byte sequence; level $i>0$'s
own input stream is level $i-1$'s own emitted **code**. Using this
session's terminology:
$$
\text{stream}_0 = \text{byte\_ids}, \qquad \text{stream}_i = \text{code}_{i-1} \quad (i>0).
$$
Sequence lengths shrink geometrically: $L_0 = \texttt{context\_len}$,
$L_i = L_{i-1}/K_{i-1}$ for $i \ge 1$ (`RefineLM.seq_lens`). Level $i$'s own
code sequence $\text{code}_i \in \{0,1\}^{L_i/K_i \times V}$ (one-hot rows,
`c_list[i]` in code) has length $L_{i+1}$.

This document concerns **only self-conditioning**: level $i$'s decode pass
conditions on $\text{code}_i$ (its own code), never on $\text{code}_{i+1}$
or above (the "cumulative cross-level" tracks built earlier this session
are **not used** by the mechanism below — see §5 for why).

## 2. Encode pass (unchanged from current `LevelLM.forward`, non-decode branch)

For level $i$, with $x_i \in \mathbb{R}^{B\times L_i \times D}$ the embedded
stream ($x_i = \texttt{embed}(\text{stream}_i)$ if $i=0$, else $x_i =
\text{stream}_i \,@\, \texttt{embed.weight}$):
$$
h_i = \texttt{ln\_f}\big(\texttt{blocks}(x_i)\big) \in \mathbb{R}^{B \times L_i \times D},
$$
ordinary causal (optionally windowed) self-attention, window
$w^{\text{enc}}_i$ (`LevelLM.window`).

**Encode NTP loss** (`encode_losses[i]`, unchanged):
$$
\mathcal{L}^{\text{enc}}_i = \text{CE}\Big(h_i[:, :-1, :] \,@\, \texttt{embed.weight}^\top,\ \ \text{stream}_i[:, 1:]\Big).
$$

**Code extraction** (`code_extract_mode`, unchanged), pooling $h_i$ over
each $K_i$-sized block into $\text{pooled}_i \in \mathbb{R}^{B \times
L_i/K_i \times D}$, then
$$
\text{pre\_q}_i = \texttt{\_classify}(\text{pooled}_i) = \begin{cases}
\text{pooled}_i \,@\, \texttt{code\_head.weight}^\top & \texttt{code\_head\_tied=False} \\
\text{pooled}_i \,@\, \texttt{embed.weight}^\top & \texttt{code\_head\_tied=True}
\end{cases}
$$
$$
\text{code}_i = \texttt{gumbel\_quantize}(\text{pre\_q}_i, \tau) = \text{soft} + \operatorname{sg}(\text{hard} - \text{soft}),
$$
$\text{soft} = \text{softmax}(\text{pre\_q}_i/\tau)$, $\text{hard} =
\text{one\_hot}(\arg\max \text{soft})$, $\operatorname{sg}(\cdot)$ = stop-gradient
(`.detach()`). **Forward value of $\text{code}_i$ is always exactly
$\text{hard}$** regardless of $\tau$ — this is why substituting a drafted
code for a reconstructed one at inference (§6) changes nothing about the
*type* of value decode receives, only which discrete class it is.

## 3. Self-conditioned decode pass — block-prefix packing (unchanged mechanism from `_packed_decode_forward`, restricted to a single self track)

Block $b \in \{0, \dots, L_i/K_i - 1\}$ covers raw stream positions
$[bK_i,\, bK_i+K_i-1]$. Its **prefix** is
$$
\text{prefix}_i[b] = \begin{cases} \texttt{decode\_bos} & b = 0 \\ \text{code\_kv}_i[b-1] & b \ge 1 \end{cases}, \qquad \text{code\_kv}_i = \text{src}_i \,@\, \texttt{embed.weight},
$$
where $\text{src}_i = \text{code}_i$ if `decode_code_ste=True`, else
$\text{src}_i = \operatorname{sg}(\text{code}_i)$ if `decode_code_ste=False`
(§5). Prefix $b$'s position is $t_b = bK_i - 1$ (`prefix_true_pos`).

Packed sequence (prepend layout, generalizes to any $K_i$):
$$
\xi_i = [\text{prefix}_i[0], \dots, \text{prefix}_i[L_i/K_i-1],\ x_i[0], \dots, x_i[L_i-1]] \in \mathbb{R}^{(n_{\text{blk}}+L_i) \times D}, \quad n_{\text{blk}} = L_i/K_i.
$$
True positions: prefixes get $t_b = bK_i-1$; bytes get their own raw
position $t = 0,\dots,L_i-1$. Attention mask, for query true-pos $t_q$ and
key true-pos $t_k$ (key is a code iff it's one of the prefixes):
$$
\text{allow}(t_q, t_k) = \underbrace{(t_k \le t_q)}_{\text{causal}} \ \land\ \underbrace{\lnot(\text{key\_is\_code} \land t_k = t_q)}_{\text{same-position exclusion}} \ \land\ \underbrace{(t_q - t_k < 2w^{\text{dec}}_i)}_{\text{windowed}},
$$
$w^{\text{dec}}_i$ = the self track's decode window
(`decode_windows[i][0]`). RoPE applied once, post-packing, using each
token's own true position.

$$
\eta_i = \texttt{ln\_f}\big(\texttt{blocks}(\xi_i)\big), \qquad h^{\text{dec}}_i = \eta_i[n_{\text{blk}}:] \in \mathbb{R}^{B \times L_i \times D} \quad \text{(drop the prefix rows)}.
$$

**Decode NTP loss** (`decode_losses[i]`):
$$
\mathcal{L}^{\text{dec}}_i = \text{CE}\Big(h^{\text{dec}}_i[:,:-1,:] \,@\, \texttt{embed.weight}^\top,\ \ \text{stream}_i[:,1:]\Big).
$$

## 4. Total loss (unchanged structure)

$$
\mathcal{L} = \beta \mathcal{L}^{\text{enc}}_0 + \gamma \sum_{i\ge1} \mathcal{L}^{\text{enc}}_i + \delta \sum_i \mathcal{L}^{\text{dec}}_i
$$
(`byte_ntp_weight`, `code_ntp_weight`, `decode_ntp_weight`).

## 5. The detach requirement, precisely

`decode_code_ste` (`Config` flag, existing) controls $\text{src}_i$ in §3:

- **`True` (STE)**: $\partial \mathcal{L}^{\text{dec}}_i / \partial(\text{pre\_q}_i)$
  is nonzero (flows through $\text{code}_i$'s straight-through path into
  `_classify`'s own weights). Decode's loss reshapes level $i$'s own code
  distribution toward whatever is easiest to *decode from*.
- **`False` (detach, required for this design)**: $\text{src}_i =
  \operatorname{sg}(\text{code}_i)$, so $\partial \mathcal{L}^{\text{dec}}_i
  / \partial(\text{pre\_q}_i) = 0$ identically. Level $i$'s own code
  distribution is shaped **only** by $\mathcal{L}^{\text{enc}}_i$ (and
  $\mathcal{L}^{\text{enc}}_{i+1}$, since $\text{code}_i$ feeds level
  $i+1$'s encode pass via straight-through there — that path is separate
  and untouched by this flag).

**Required**: set `decode_code_ste=False` for every level whose self track
is meant to support the drafted-substitution generation scheme in §6.
(Forward numerics of decode are unaffected either way — only backward.)

## 6. The drafter — an independent LM over $\text{code}_i$

For level $i < n-1$: level $i+1$ **already is** this drafter — no new
module needed. Level $i+1$'s own NTP head, applied to $\text{stream}_{i+1}
= \text{code}_i$:
$$
\text{logits}^{\text{draft}}_i[t] = h_{i+1}[t] \,@\, \texttt{embed.weight}^\top \in \mathbb{R}^V, \qquad \widehat{\text{code}_i}[t+1] = \text{one\_hot}\big(\arg\max \text{logits}^{\text{draft}}_i[t]\big),
$$
exactly `generate_level1_codes`'s existing mechanism, unmodified. This
prediction depends only on $\text{code}_i[0..t]$ — **never** on
$\text{stream}_i$'s raw content beyond block $t$, so it is computable
strictly ahead of any of level $i$'s own future decode work.

For the top level ($i = n-1$, or any $n_{\text{levels}}=1$ config): no
level $i+1$ exists. Needs a **dedicated auxiliary LM**
(`Config.aux_code_lm`, new — this is the item parked earlier this session,
now with a concrete spec): a small causal LM over $\text{code}_{n-1}$
(or $\text{code}_0$ in the $n=1$ case), architecturally identical to a
`LevelLM`'s encode-only path (own `blocks`/`embed`, may or may not share
weights with the main tower — open parameter, not fixed by this doc),
trained via the same NTP loss form as $\mathcal{L}^{\text{enc}}_{i+1}$
above but with $\text{stream}_{\text{aux}} = \text{code}_{n-1}$ as both
input and target.

## 7. Generation-time substitution + block-parallel decode

Given a prompt of $n_{\text{prompt}}$ complete blocks
($n_{\text{prompt}} \ge 2$, see §7.1) and $g = \texttt{gen\_len}/K_i$ new
blocks wanted:

**7.1 — Draft codes** (sequential, $O(g)$ steps, cheap — level $i+1$'s own
sequence has length $L_{i+1} \ll L_i$):
$$
\widehat{\text{code}_i}[n_{\text{prompt}}], \ \widehat{\text{code}_i}[n_{\text{prompt}}+1], \ \dots,\ \widehat{\text{code}_i}[n_{\text{prompt}}+g-1]
$$
via §6's autoregressive loop, seeded with the prompt's own true
$\text{code}_i[0..n_{\text{prompt}}-1]$ (computed once via the ordinary
encode pass over the prompt).

**7.2 — The off-by-one** (real, found and fixed this session for the
now-shelved cross-level version — same fix applies here identically):
predicting raw byte $t+1$ uses $h^{\text{dec}}_i[t]$, whose governing
prefix is $\text{prefix}_i[\lfloor t/K_i \rfloor]$ — i.e. **block $b$'s
prefix conditions bytes $[bK_i+1,\, bK_i+K_i]$, not $[bK_i,\, bK_i+K_i-1]$**.
So:
- Byte $P = n_{\text{prompt}} K_i$ (first new byte): predicted via
  $h^{\text{dec}}_i[P-1]$, block $n_{\text{prompt}}-1$, prefix
  $\text{code\_kv}_i[n_{\text{prompt}}-2]$ — a TRUE (already-known) code,
  needs no draft. One ordinary single-block decode step.
- Bytes $[P+1,\dots,P+K_i]$: block $n_{\text{prompt}}$, prefix
  $\text{code\_kv}_i[n_{\text{prompt}}-1]$ — TRUE (prompt's own last
  block), still no draft needed yet.
- Bytes $[P+K_i+1,\dots,P+2K_i]$: prefix $\widehat{\text{code}_i}[n_{\text{prompt}}]$
  — first DRAFTED prefix.
- In general, group $g'=0,\dots,g-1$ covers bytes $[P+1+g'K_i,\, P+(g'+1)K_i]$
  using prefix $\text{code\_kv}_i[n_{\text{prompt}}-1+g']$ (true for
  $g'=0$, drafted for $g' \ge 1$).

**7.3 — Batch across groups.** Since every drafted prefix in step 7.1 was
computed without needing ANY of level $i$'s own new raw bytes (unlike the
shelved cross-level version, where $\text{code}_{i+1}[p]$ provably
depended on $\text{code}_i[p]$ at the *same* position — see
`docs/status.md`'s "Blockwise parallel decoding" section for that proof),
groups $g'=1,\dots,g-1$ share no dependency on each other's raw bytes.
Stack them along the batch dimension and run $K_i$ synchronized
micro-steps (§7.4), giving $O(g) + O(K_i)$ total steps instead of
$O(g \cdot K_i) = O(\texttt{gen\_len})$.

**7.4 — Per-micro-step** (mirrors `LevelLM._block_decode_step`, unchanged
mechanism, just fed a drafted `prefix_embed` for $g' \ge 1$): for local
step $\tau = 0,\dots,K_i-1$, with $\text{block\_bytes}$ initialized to
placeholders,
$$
\xi = [\text{prefix\_embed},\ \texttt{embed}(\text{block\_bytes})], \quad \eta = \texttt{ln\_f}(\texttt{blocks}(\xi)), \quad \text{logits} = \eta[\tau] \,@\, \texttt{embed.weight}^\top,
$$
$$
\text{block\_bytes}[\tau] \leftarrow \arg\max(\text{logits}).
$$
Local true-positions $(-1, 0, \dots, K_i-1)$ suffice (not each block's real
absolute offset) — exact under RoPE's relative-position property (identical
relative layout in every block, see `docs/status.md`).

**7.5 — Assemble.** Concatenate: prompt bytes, then the byte $P$ special
step (7.2), then groups $g'=0,\dots,g-1$'s outputs, trimming the one extra
byte the last group overproduces (bytes $[P+1,\dots,P+g\cdot K_i]$ from the
groups is $g\cdot K_i$ bytes; combined with byte $P$ that's
$g\cdot K_i + 1$ bytes; keep the first $\texttt{gen\_len} = g\cdot K_i$ of
them, i.e. byte $P$ through $P+\texttt{gen\_len}-1$).

## 8. What this does **not** claim

This substitutes a *drafted* $\widehat{\text{code}_i}$ for the *true*
$\text{code}_i$ starting at group $g' \ge 1$ — output will **not**
bit-for-bit match `generate_no_cache` (unlike the shelved cross-level
attempt, which was designed to be exact and failed to be). Fidelity
depends entirely on level $i+1$'s (or the aux LM's) own NTP accuracy
against the true $\text{code}_i$ stream — an empirical question, not
addressed by this document. No verify/accept-reject step is specified
here (see the earlier draft-and-verify discussion in the session log for
that extension, also not implemented).
