# `qcute_refine_v4_4_1` / `qcute_refine_v4_5_1` — math

Direct translation of `LevelLM._packed_decode_forward_selfcode` (v4.4.1) /
`LevelLM.selfcode_decode` (v4.5.1) — the two are the same math, same variable roles, just
duplicated per file. Everything else (encode, `RefineLM._run` dispatch, multi-track fallback,
generation loop) is v4.4/v4.5-identical; only the self-track decode changed. Symbols match code
names 1:1 where possible.

## 0. Given

Level input $x = (x_0,\dots,x_{L-1})$ (`seq_repr`), block size $K$, $n_{\text{blk}} = L/K$,
$d$ = `cfg.d_model`, $H$ = `cfg.n_heads`, $h_d = d/H$, $V$ = `cfg.vocab`.

## 1. Encode → per-block code (unchanged code path, stated exactly)

$x_0 = \text{embed}(x)$ if byte-level else $x \,W_{\text{embed}}$ (`x @ self.embed.weight` — for
$i{>}0$, $x$ is `c_list[i-1]`, an STE tensor, forward-value one-hot but soft-gradient; this matmul
is unconditional, never `.detach()`'d, NOT gated by `decode_code_ste` — that flag only gates the
separate `code_kv`/$e_b$ embedding built in §3, a different embedding of the same code tensor for
decode's conditioning input, not this level's own representation). Plain causal self-attention
(§3 below, without any code interleaving) over $x_0$ gives
$h^{\text{enc}} \in \mathbb{R}^{L\times d}$. Per block $b=0,\dots,n_{\text{blk}}-1$:
$$
\phi_b = h^{\text{enc}}_{(b+1)K-1} \quad (\texttt{code\_extract\_mode="last\_h"}, \text{ the only mode used in this grid — no pooling, a single indexed row})
$$
$$
\text{logits}_b = \phi_b W_{\text{head}}^\top, \quad W_{\text{head}} = \begin{cases} W_{\text{code\_head}} & \texttt{code\_head\_tied=False (default)} \\ W_{\text{embed}} & \texttt{code\_head\_tied=True} \end{cases}
$$
$$
c_b = \text{soft}_b + \text{sg}(\text{hard}_b - \text{soft}_b), \quad \text{soft}_b=\operatorname{softmax}(\text{logits}_b/\tau), \ \text{hard}_b=\text{onehot}(\arg\max \text{soft}_b)
$$
(`gumbel_quantize`; $\text{sg}$ = stop-gradient/`.detach()`; Gumbel noise added to `logits_b` pre-softmax iff `use_gumbel_noise=True`, omitted from above since our grid runs both settings). $c_b\in\{0,1\}^V$ forward-valued, straight-through gradient.

## 2. Corrected decode semantics

$c_b$ conditions block $(b{+}1)$, not block $b$ (an earlier draft paired $c_b$ with its own block —
an autoencoder; not what's implemented). $b$ ranges $0,\dots,n_{\text{blk}}-2$; $n_{\text{blk}}-1$
values get used, $c_{n_{\text{blk}}-1}$ is computed but unconsumed (nothing follows it).

## 3. Decode packing + attention (`code_kv[:, :n_units, :]`, `x0_blocks[:, 1:, :, :]`)

$n_{\text{units}} = n_{\text{blk}} - 1$. Caller passes `code_kv` already embedded (from
`RefineLM._run`): $e_b = W_{\text{embed}}[\arg\max c_b]$ (`decode_code_ste=False`, all grid
configs) or $e_b = c_b W_{\text{embed}}$ (`decode_code_ste=True`, unused here).

$$
X_e = \big(e_0, x_{0,K},\dots,x_{0,2K-1},\ e_1, x_{0,2K},\dots,x_{0,3K-1},\ \dots,\ e_{n_{\text{units}}-1}, x_{0,n_{\text{blk}}K-K},\dots,x_{0,n_{\text{blk}}K-1}\big)
$$

where $x_{0,j}$ is `x0`'s $j$-th row (`x0_blocks[:, 1:, :, :]`, i.e. blocks $1,\dots,n_{\text{blk}}-1$'s raw embeddings). $X_e \in \mathbb{R}^{L_e \times d}$, $L_e = n_{\text{units}}(K{+}1)$, `.view(B, n_units*(K+1), D)`. Positions: $t = 0,1,\dots,L_e{-}1$ (`torch.arange(Le)` — plain sequential, code and byte slots counted identically, no $-1$ offset).

**Attention mask** (`ti, tj = true_pos`; NOT the chunked-window path `CausalSelfAttention` uses elsewhere — a separate dense boolean mask built inline):
$$
\text{allow}[i,j] = (t_j \le t_i) \ \wedge \ \big(\texttt{window is None} \lor (t_i - t_j) < \texttt{window}\big).
$$

**Per layer** ($\ell = 1,\dots,$ `n_layers`, weights `block.attn.qkv`/`.out`, `block.ln1/ln2/mlp`):
$$
\tilde x = \text{ln1}(X_e), \quad [Q;K;V] = \tilde x\, W_{qkv}^\top \ \text{(no bias)}, \ \text{split into } H \text{ heads of dim } h_d
$$
$$
Q \leftarrow Q\odot\cos + \text{rot}(Q)\odot\sin, \quad K \leftarrow K\odot\cos + \text{rot}(K)\odot\sin \quad (\text{RoPE, } \cos,\sin \text{ from plain positions } 0..L_e{-}1)
$$
$$
\text{rot}(z) = (-z_{[d_h/2:]},\, z_{[:d_h/2]}), \qquad A = \operatorname{SDPA}(Q,K,V;\ \text{mask}=\text{allow}), \qquad X_e \leftarrow X_e + A\,W_{out}^\top
$$
$$
X_e \leftarrow X_e + \text{MLP}(\text{ln2}(X_e)), \quad \text{MLP}(z) = \text{GELU}(z\,W_1^\top)\,W_2^\top
$$
After the last layer: $\tilde h = \text{ln}_f(X_e)$ (`self.ln_f`).

## 4. NTP loss (`he_blocks = he.view(B, n_units, K+1, D)`)

Reshape $\tilde h$ into $n_{\text{units}}$ groups of $(K{+}1)$ rows. Group $u$'s **query** rows are
its first $K$ ($\tilde h^{(u)}_0,\dots,\tilde h^{(u)}_{K-1}$ = code row + block's first $K{-}1$
bytes — `he_blocks[:,:,:-1,:]`, dropping row $K$ = the block's own last byte). **Targets**:
block $(u{+}1)$'s $K$ real bytes/codes, `seq_repr[:, K:]` reshaped to match:
$$
\mathcal{L}_{\text{decode}} = \frac{1}{n_{\text{units}}K}\sum_{u=0}^{n_{\text{units}}-1}\sum_{k=0}^{K-1} \text{CE}\Big(\tilde h^{(u)}_{k}\,W_{\text{embed}}^\top,\ x_{(u+1)K+k}\Big).
$$
Row $K$ of each group (the block's own last byte) is computed (feeds later positions' causal
attention) but is never itself a query — its "next" would be the following group's code row, a
different target space.

## 5. Returned $h$ (`h = torch.cat([x0[:, :K, :], he_blocks[:, :, 1:, :].reshape(...)], dim=1)`)

$$
h^{\text{dec}} = \big(\underbrace{x_{0,0},\dots,x_{0,K-1}}_{\text{block 0, raw, no gradient from } \mathcal{L}_{\text{decode}}},\ \underbrace{\tilde h_{\text{row }1},\dots,\tilde h_{\text{row } n_{\text{units}}(K+1)-1}}_{\text{group rows }1..K\text{ of each group, i.e. reconstructed bytes}}\big) \in \mathbb{R}^{L\times d}.
$$
Block 0's slice exists only to keep shape $[L,d]$ (consumed by `_sample_next_byte`/code
extraction elsewhere) — untrained.

**Guard** (`n_units >= 1` asserted in the method; enforced one level up in `RefineLM._run`):
`if len(tracks) == 1 and L_i // tracks[0][1] < 2: continue` — same treatment as a ragged length,
$h^{\text{dec}}_i \to h^{\text{enc}}_i$ for that step.

## 6. Multi-level (`len(full_track_specs) > 1`)

Unchanged from v4.4/v4.5: applies only when level $i$ has a coarser level above it. That path
still uses the original previous-block-code / cross-attention mechanism (already causal, no
per-code query — out of scope here, see `docs/qcute_refine_math.md`).

## 7. Generation

`generate_no_cache` only (no KV-cache path exists in `qcute_refine`, any version). Same
pad-to-`decode_K`-then-read-`h[L-1]` loop as v4.4/v4.5; `_run` now resolves through §1–§5 above
whenever `len(tracks)==1`. Causally exact by construction: $c_b$ is computable only once block
$b$ is complete and only ever conditions block $(b{+}1)$ onward — no circularity for the block
being generated, unlike the discarded self-reconstruction draft (§2).
