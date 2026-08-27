# Archived qcute lineage, chronological one-liners

Every fork under `qcute/archive/` (the `qcutelm` family) and `qcute/archive2/` (the `qcute_refine`
family), in the order they were written. Both families are fully archived — see CLAUDE.md's
Architecture section for the design docs that go with each (`docs/archive/*.md` for the `qcutelm`
family, `docs/archive2/*.md` for the `qcute_refine` family). This file is just the chronological
index; read each file's own module docstring for the full rationale behind a given step.

## `qcute/archive/` — `qcutelm` lineage

| # | File | One-liner |
|---|---|---|
| 1 | `qcutelm.py` | End-to-end latent AR: MLP encoder/decoder over fixed K-byte chunks + FSQ/BSQ bottleneck + LM, pure latent autoregression, jointly trained. |
| 2 | `qcutelm_vlt.py` | Variable-length causal-transformer pooler encoder + NoPE broadcast-z decoder (all T positions get the identical code, order recovered only via causal mask). |
| 3 | `qcutelm_vlt2.py` | Same encoder as vlt; decoder input changed to code-at-position-0 + real trainable per-position embeddings (breaks vlt's uniform-attention NoPE degeneracy). |
| 4 | `qcutelm_vlt3.py` | Collapses separate encoder/decoder into one shared-weight causal transformer with an appended trainable "code-query" token and a content-dependent BOS decode. |
| 5 | `qcutelm_vlt4.py` | Replaces the code-query token with a strided readout off a regular large-context causal LM — codes read for free from an unconstrained-by-bottleneck LM. |
| 6 | `qcutelm_vlt5.py` | Sliding-window attention as the real default + continuous (not per-block-reset) reconstruction path across the whole context. |
| 7 | `qcutelm_vlt6.py` | Tokenizer IS an AR latent LM: single byte-level NTP loss only, no reconstruction/code-classification losses, decoder never sees its own target bytes. |
| 8 | `qcutelm_vlt7.py` | Narrow tokenizer at the edges + separate wide codelm sandwiched in the middle operating on the short code sequence — restores the width/length compute split lost in earlier interleaved-single-stack designs. |
| 9 | `qcutelm_vlt8.py` | Same hybrid as vlt7, fixes `attn_window` to respect the (K+1)-periodic interleaved-block structure instead of chunking mid-block. |
| 10 | `qcutelm_vlt9.py` | True encode/decode symmetry: encode also consumes codelm's forecast as conditioning via a genuine block-by-block AR bootstrap loop. |
| 11 | `qcutelm_vlt10.py` | Clockwork-RNN-style N-level tiered stack — codelm becomes a literal sparse middle layer (invoked every K bytes) instead of a side-module or bootstrap loop. |
| 12 | `qcutelm_vlt11.py` | Recursive Pass1/Pass2 sandwich: unconditional encoder + separate conditioned decoder at every level, genuine sequence-length compression recursively applied. |
| 13 | `qcutelm_pyramid.py` | Single shared LM over a flat multi-resolution sequence, codes merged FIFO-style via a cheap non-attentional linear merge instead of forecast-substitution. |
| 14 | `qcutelm_mergetoken_v1.py` | vlt11 with level 0 restructured for multi-token prediction — cuts the dominant MLP/projection FLOP cost while keeping full 1024-byte receptive field. |
| 15 | `qcute_bytepool.py` | No BSQ/FSQ at all: independent byte/pair/quad LMs connected by cross-attention, coarse-to-fine speculative-decoding-shaped cascade (two variants, v12/v13). |

## `qcute/archive2/` — `qcute_refine` lineage

| # | File | One-liner |
|---|---|---|
| 16 | `qcute_refine_v1.py` | Pure recursive NTP tower (always-on per-level NTP loss, no moving-target problem) + a mirrored joint-chain-MTP detokenizer tower. |
| 17 | `qcute_refine_v2.py` | Same encoder tower; detokenizer replaced by cross-attention reusing the encoder's own hidden states instead of running a fresh self-attention trunk. |
| 18 | `qcute_refine_v3.py` | Adds `EncoderLevel` fusion on top of v2 — bottom-up sweep followed by a fused pass so cross-attention can actually move `byte_loss`/`val_bpb`, not just an invisible `tok_loss`. |
| 19 | `qcute_refine_v4.py` | Removes `DecoderLevel` entirely (redundant with fusion, never contributed to `byte_loss`); makes both generation paths fusion-aware (v3's generation had silently skipped it). |
| 20 | `qcute_refine_v4_1.py` | Extreme weight sharing — one `LevelLM` (`share_levellm`) reused across every level, `window` moved to a forward-time argument so shared weights can still serve per-level windows. |
| 21 | `qcute_refine_v4_2.py` | Concat-only fusion (removes the separate cross-attention `CrossBlock`/`fuse_position` choice entirely) + a single shared `dq`/embed/head across every level including byte level. |
| 22 | `qcute_refine_v5_old.py` | Byte-identical clone of v4_2 carrying an unimplemented design sketch (H-Net-style variable-stride boundary detection) — never built, later superseded by the real v5 line. |
| 23 | `qcute_refine_v4_3.py` | Switches the code representation from BSQ/per-bit-BCE to categorical softmax + Gumbel — the ancestor of `quant_type="softmax"` in the later v5 modules. |
| 24 | `qcute_refine_v4_4.py` | Packed-sequence multi-track cumulative decode (self + every coarser level's code interleaved/prepended into one sequence, trainable BOS, correct RoPE timing). |
| 25 | `qcute_refine_v4_4_1.py` | Fixes the self track to genuine LM continuation (code_b conditions block b+1, never its own block b — v4.4 was an accidental autoencoder there). |
| 26 | `qcute_refine_v4_5.py` | Replaces v4.4's packed-sequence decode with explicit staged cross-attention through the same shared weights (one full-depth stage per conditioning track). |
| 27 | `qcute_refine_v4_5_1.py` | Same self-track LM-continuation fix as v4.4.1, applied to v4.5's staged design — self track packed inline, coarser tracks still staged cross-attention. |

`qcute_refine.py` (not numbered above, 24 lines) is a thin always-latest alias
(`from qcute.qcute_refine_v4 import *`), not an architecture step itself — it just re-exports
whichever `vN` was current when last updated.

This is exactly the lineage the active v5 modules descend from: `v4_4_1` → `qcute_v5_concat.py`
(later rewritten with chronological merged-interleave decode, see docs/status.md), `v4_5_1` →
`qcute_v5_stack.py` → `qcute_v5.py` (efficient-windowed-attention rewrite, also see docs/status.md).
