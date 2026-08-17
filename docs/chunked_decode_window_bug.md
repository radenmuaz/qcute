# Bug (now fixed): qcute_v5_concat.py's chunked/banded windowed decode didn't match its own dense reference (2026-08-17)

**Status: FIXED** in `qcute/qcute_v5_concat.py`'s `_merged_layout` (see "Fix" below). Scope was
narrower than first thought — see "Correction" below: `qcute_v5.py` was never affected.

## Summary

Found while validating the new `generate_true_kv_cache` (see [kv_cache_design.md](kv_cache_design.md))
against `generate_no_cache`: **`qcute_v5_concat.py`'s `_merged_decode_forward` computed a numerically
different result via its chunked/banded attention path than via its own dense-masked fallback, for
the same window**. Not a crash, not NaN — a silently narrower attention window than configured,
whenever `window < L` (chunking triggers) AND the merged buffer has more than one entry per `true_pos`
unit (i.e. whenever a decode track's `K` is small enough that codes tie with or crowd near byte
positions). This was NOT a bug in the new KV-cache code — the new incremental cache was built
against, and verified to match, the dense reference exactly.

## Correction: qcute_v5.py was NOT affected — my first report was wrong

An earlier version of this doc claimed `qcute_v5.py`'s `cross_attn_stage` had the same bug, based on
comparing its chunked output against a dense reference at the sequence's LAST row. That comparison
was invalid: the last row (`L=32`, `K=4`) happens to be a block boundary, and `cross_attn_stage`
deliberately OVERWRITES boundary rows with the qfb boundary-query pass's own output (see
`qcute_v5.py`'s own qfb docstring) — so I was comparing the boundary-patched value against the
unpatched main-path dense reference, not an apples-to-apples check. Redone correctly: comparing
chunked vs. dense at NON-boundary rows (`0, 16, 29, 30`) — exact match. Comparing the boundary-query
pass itself (chunked vs. a manual dense reimplementation of the boundary formula) — also exact
match. `qcute_v5.py`'s chunked path is correct; its own `check_gen_consistency`-style self-check
(built into training/eval) is why this never surfaced there. **`qcute_v5.py` needed no fix.**

## Reproduction

**`qcute_v5_concat.py`**, `Ks=(1,)`, `attn_window=(8,)`, `d_model=32`, `n_layers=2`:
```python
c_dense = enc(prompt[:8].unsqueeze(0), level=0, window=8, compute_ntp=False)[0]
code_embeds = dec.quant.embed_for_decode(dec, c_dense)
h_out, q_last = dec._merged_decode_forward(dec.embed(prompt[:8].unsqueeze(0)), [(code_embeds, 1, 8)], extra_query=True)
# q_last = [-0.8421, 0.1049, -0.4863, 0.9664]   <- chunked path (real forward())

# force the SAME call through the dense-mask fallback by stripping the cached chunk fields:
# (patched dict without 'sc'/'n_chunks'/'idx'/'chunk_mask'/etc.)
# q_last = [-0.5209, -0.0423, -0.3468, -0.0326]  <- dense-masked fallback, SAME window=8
```
Confirmed on the actual queued-job shape too: `Ks=(4,1)`, `attn_window=(16,64)`, `context_len=64` —
`next_query[0]` differs between chunked (real) and dense-forced (`[-1.5132,-1.7070,0.5318,-0.3281]`
vs `[-1.4312,-1.6861,0.6601,-0.1573]`).

## Root cause (`qcute_v5_concat.py`'s `_merged_layout`)

`_merged_layout` computes `n_prev_chunks = ceil(W_max / sc)` — how many PRIOR buffer-index chunks
(each of size `sc`, the smallest configured window) a query needs gathered to guarantee full window
coverage. This formula implicitly assumes each buffer-index chunk of size `sc` spans `sc` units of
`true_pos` — true only when there's exactly one buffer entry per `true_pos` value. But the merged
buffer interleaves CODE entries at the same (or adjacent) `true_pos` as their boundary byte — for
small `K` (e.g. `K=1`, every byte completes its own block), there are up to 2 buffer entries per
`true_pos` unit, so a window of `W` `true_pos` units can require looking back up to `2*W` BUFFER
SLOTS, not `W`. `n_prev_chunks` undercounts this, so queries near a chunk boundary silently lose
visibility into keys that should be within window — the chunked path computes a *narrower* effective
window than configured, not the intended `reach < window` semantics.

## Impact

Every windowed (`window < context_len`/`L`) `qcute_v5_concat` training run this session —
`qcute_v5_concat_1`, plus the just-killed `qcute_v5_concat_2*`/`qcute_v5_concat_3` batch (logs
deleted, to be rerun with the fix) — trained and evaluated through this chunked path, so the model
saw a somewhat narrower effective attention window than its `attn_window` config states.
Training/eval used the SAME (buggy) path consistently both times, so *relative* comparisons between
`qcute_v5_concat` configs trained this way remain internally meaningful, but the absolute
windowed-attention behavior didn't match what the configs claimed. `qcute_v5.py`/`qcute_v5_1`'s
results are unaffected (see Correction above).

## Fix

`_merged_layout`'s `n_prev_chunks` now multiplies `W_max` by `density = len(tracks_meta) + 1` (max
possible buffer entries per `true_pos` unit: 1 byte + at most one code per track tying there) before
computing the chunk-lookback count, guaranteeing the gathered range covers the full window in
`true_pos` terms regardless of how densely codes crowd the buffer. Verified: the `Ks=(1,)`/`window=8`
repro above now matches the dense-mask fallback exactly; `check_gen_consistency` and finite-loss
checks pass across `Ks in {(1,),(2,),(4,),(2,2),(4,2),(2,1,1)}`; the pre-existing (unrelated)
`Ks=(4,2)` `check_gen_consistency` gap (8/39 mismatches, a ragged-prefix harness issue, not caused by
this bug) reproduces identically before and after the fix, confirming no regression.

Existing test suite (`scripts/test_v5_concat.py`) has a gap worth closing separately: its
`independent_dense_reference`/`test_dense_reference_mask_shape` only checks buffer STRUCTURE
(ordering, length), never numerically compares the chunked path's actual output against a dense
reference at matching window — exactly why this slipped through. Also currently broken independent
of this fix (`Config.__init__() got an unexpected keyword argument 'share_level_weights'` — a stale
reference to an already-pruned config field).
