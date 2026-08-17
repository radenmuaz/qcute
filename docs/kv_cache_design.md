# Towards a true incremental KV cache for qcute_v5 / qcute_v5_concat (2026-08-17)

## Current state

Both `qcute/qcute_v5.py` and `qcute/qcute_v5_concat.py` ship a `generate_kv_cache` that is FIFO-window
*re-truncation*, not an incremental K/V tensor cache: every new-byte step slices the trailing
`context_len` bytes and reruns the full windowed forward from scratch over that window. This bounds
per-step cost to `O(context_len)` instead of `O(current total length)`, but recomputes every
projection/attention/MLP op every step — no attention K/V state persists across steps.

## RoPE positions are per-level LOCAL, not a shared global timestep

Checked directly in both files: every level's `encode`/`cross_attn_stage`/`_merged_decode_forward`
call generates its own position ids as `torch.arange(L)` (or the `true_pos`/`real_pos` equivalent in
`qcute_v5_concat.py`), where `L` is *that level's own* current sequence length in *that level's own*
granularity units — bytes for level 0, codes for level 1, etc. A level-1 code at code-index 5 gets
RoPE position `5`, not `5*K0=20` (its corresponding byte offset). Positions reset to 0 fresh on every
call; there is no shared, cross-level absolute clock. This holds identically in `qcute_v5.py`'s
`cross_attn_stage` (`query_pos = torch.arange(L)`, `code_pos` counted in the calling level's own
frame) and `qcute_v5_concat.py`'s `_merged_layout` (`byte_true_pos = torch.arange(L)`, `code_true_pos`
built relative to the same `L`).

## Why this matters for KV caching

**The good news**: both files already precompute their buffer/attention addressing (which slot holds
which token, which rows get boundary-patched, which code keys a given query window can see) as a
*pure function of shape* (`(L, tracks_meta)` for `_merged_layout`, `(L, track_K)` for
`cross_attn_stage`'s `code_pos`/`bq_pos`), cached across calls of the same shape — exactly the
prerequisite an incremental cache needs to know where a new token/code lands without recomputing
structure every step.

**The catch**: resetting positions to 0 every window call is *why* nothing can be cached today — a
token cached with RoPE angle computed at window-relative position 5 becomes wrong the moment the
window slides and that same token is now at position 4. Fix: switch each level's own local position
counter from "reset to 0 every call" to "absolute, monotonically increasing, sliced to the current
window" — per level, at that level's own rate (level 0 counts absolute byte position, level 1 counts
absolute code position, etc.). This is a free correctness win, not a new approximation: RoPE attention
scores depend only on the *relative* offset `(pos_q - pos_k)`, never the absolute value, so a uniform
shift applied to an entire window (today's reset-to-0 vs. an absolute counter) produces bit-identical
attention output. A cached K/V vector, once computed at its true absolute position, stays valid
forever until evicted — no recomputation needed as the window slides past it.

## What's still real work beyond that

- **Actually persisting K/V tensors per layer** with circular-buffer eviction — today nothing is
  stored; `generate_kv_cache` re-slices raw bytes and reruns the whole forward.
- **Multi-rate append cadence**: bytes append every generation step, but a level's own code only
  finalizes every `K` bytes, and each level's cache lives at its own (slower) rate — a problem plain
  single-stream (bytelm-style) KV caching never has to solve.
- `qcute_v5.py`'s qfb boundary-query pass is a *second*, separate append stream (one entry per
  completed block, not per byte), needing its own cache bookkeeping alongside the main one.

## Status: implemented for n_levels==1, both files

`generate_true_kv_cache` added to both `qcute/qcute_v5.py` and `qcute/qcute_v5_concat.py` — scoped
to `n_levels==1` (single level, one self track) and `code_extract_mode='last_h'`; general
multi-level/multi-track caching remains future work (the multi-rate append cadence and, for
`qcute_v5.py`, the qfb boundary-query stream's own bookkeeping, described above).

**Correctness**: verified exactly (`torch.equal`, token-for-token) against `generate_no_cache` across
`Ks in {(1,),(2,),(4,)}` x `attn_window in {small, dense, unbounded}`, including generation runs
exceeding `context_len` (stress-testing the absolute-position claim) and both `quant_type in
{softmax, bsq}`. One expected non-match: `use_gumbel_noise=True` diverges between separate calls of
even the SAME generation function (confirmed pre-existing, not a regression — `generate_kv_cache` vs
`generate_no_cache` already diverge under gumbel noise, independent of this new code).

**Two real bugs found and fixed during this verification** (both outside the new cache code itself):
- `qcute_v5_concat.py`'s chunked windowed decode silently narrowed its effective attention window
  whenever codes crowded the merged buffer (small `K`) — see
  [chunked_decode_window_bug.md](chunked_decode_window_bug.md), now fixed. `generate_true_kv_cache`
  was built against the dense reference (the intended behavior) throughout, which is exactly how this
  surfaced — it "diverged" from the (buggy) `generate_no_cache` until the underlying bug was fixed,
  not because the new cache was wrong.
- `qcute_v5_concat.py`'s `generate_true_kv_cache` itself had a real bug on ragged prompts: `_run` has
  a "self-code decode needs 2+ complete blocks, else fall back to pure encode output" guard
  (`len(tracks)==1 and L_i//K < 2: continue`) that the new cache initially didn't replicate, so a
  prompt ending mid-block (e.g. `prompt_len=5`, `K=4`, only 1 complete block) used the decode-derived
  query a step too early. Fixed by gating the reported query on `(pos+1)//K >= 2`, falling back to
  `h_enc` (pure encode output) below that floor — buffer state is still built unconditionally
  underneath so it's ready the moment the floor is crossed, only the *reported* query differs. Note
  `qcute_v5.py` needed no equivalent gate: its `cross_attn_stage` already dropped this guard entirely
  during the qfb rewrite (confirmed safe there since SDPA on an all-False mask returns deterministic
  zeros — see [status.md](status.md)'s qfb section), so its cache never needed one either.

After both fixes: **exact match verified across `Ks in {(1,),(2,),(4,)}` x `attn_window in {small,
dense, unbounded}` x `prompt_len in {5, 12}` (18 combinations) for both files**, plus `quant_type in
{softmax, bsq}` and generation exceeding `context_len`.

**Speedup** (`Ks=(1,)`, `d_model=256`, `n_layers=1`, CPU, 200 new bytes, vs. the existing FIFO
`generate_kv_cache`):

| file | context_len | window | FIFO recompute | true KV cache | speedup |
|---|---|---|---:|---:|---:|
| `qcute_v5.py` | 256 | 16 | 0.936s | 0.205s | 4.57x |
| `qcute_v5.py` | 1024 | 16 | 1.056s | 0.210s | 5.02x |
| `qcute_v5.py` | 1024 | 64 | 1.215s | 0.256s | 4.75x |
| `qcute_v5_concat.py` | 256 | 16 | 0.932s | 0.246s | 3.79x |
| `qcute_v5_concat.py` | 1024 | 16 | 0.836s | 0.183s | 4.57x |
| `qcute_v5_concat.py` | 1024 | 64 | 0.933s | 0.232s | 4.03x |

True-cache wall-clock stays roughly constant as `context_len` grows (bounded by `window`, not `L`),
while FIFO recompute cost grows with `context_len` — the gap widens exactly as expected, similarly
for both files.
