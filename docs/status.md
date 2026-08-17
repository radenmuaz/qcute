# qcute v5 status

Reset to a fresh log at this point in the project — the prior session-by-session narrative got
long enough to be more archival than actionable. Full history:
[docs/archive2/status.md](archive2/status.md) (4300+ lines, newest at the bottom). New entries go
below, same convention: newest at the bottom, session-dated where useful.

## Where things stand

Default v5 modules: `qcute/qcute_v5_concat.py` (chronological merged-interleave decode) and
`qcute/qcute_v5.py` (staged cross-attention decode, efficient windowed attention, "query first
byte"/qfb boundary-query fix). Both hardcode `decode_code_ste=True`, `cross_track_source="decode"`,
and have removed `decode_self_only_aux` (single decode NTP loss term only); `quant_type: "softmax"
| "bsq"` dispatch is a `QuantScheme` strategy-class pair, not scattered `if` branches;
`share_level_weights`/`decode_separate_stage0` pruned to their always-`False` default. Reference
variants kept for comparison, none with the qfb fix: `qcute_v5_concat_slow.py`, `qcute_v5_slow.py`,
`qcute_v5_ws_slow.py`. Both files also ship `generate_true_kv_cache` — a real incremental per-layer
K/V cache (not FIFO recompute), scoped to `n_levels==1`, verified exact vs. `generate_no_cache`,
4.5-5x speedup over the older FIFO `generate_kv_cache`. Full design:
[kv_cache_design.md](kv_cache_design.md).

**Known-fixed correctness bug**: `qcute_v5_concat.py`'s chunked/banded windowed decode undercounted
its chunk lookback whenever codes crowded the merged buffer (small `K`), silently narrowing the
effective attention window below what `attn_window` configured. Fixed (density-aware chunk-lookback
count); confirmed to have measurably affected training loss, not just generation (real-data val bpb
2.6700→2.6604 on the same checkpoint, pre-fix vs. post-fix code). `qcute_v5.py` was never affected
(an earlier report to the contrary was a false positive from an invalid comparison — see the doc's
own "Correction" section). Full repro/root-cause/fix:
[chunked_decode_window_bug.md](chunked_decode_window_bug.md). **`qcute_v5_concat_1`'s checkpoint
predates this fix and is suspect** — not yet rerun.

Earlier lineage history and design docs are archived under `qcute/archive2/`, `configs/archive2/`,
`docs/archive2/` — see CLAUDE.md.

Standing conclusions carried forward (still load-bearing, don't re-derive):
- **`decode_code_ste=False` (detach) stays the default** — `=True` (STE) fixes a dead-gradient issue
  but empirically causes cond-generation to collapse into character repetition at overfit10k scale.
- **Overfit10k is the standard fast-iteration testbed**: `n_bytes=10000`, `steps=1000`,
  `batch_size=16`, `lr_peak=6e-4`, `warmup_steps=100`. No KV-cache path — `generate_no_cache` only.
- **`FlopCounterMode` (torch 2.13.0) does not count `scaled_dot_product_attention` FLOPs** — every
  "flops" figure in this doc set is linear/MLP-only, not true total compute. See
  [v5_1_flops_breakdown.md](v5_1_flops_breakdown.md).
- **RoPE positions are per-level LOCAL** (reset to 0 each call, in that level's own granularity
  units), never a shared cross-level absolute clock — see [kv_cache_design.md](kv_cache_design.md).
- Neither `qcute_v5_concat_1` nor `qcute_v5_1` clearly beats the plain deeper `bytelm_xs4_ctx256`
  baseline (2.356 bpb) despite 2-3x the flops/byte — see [baseline_vs_v5_bpb.md](baseline_vs_v5_bpb.md).

## Quantization-scheme + degenerate-case sweep (2026-08-17, running)

10-job queue, `qcute.qcute_v5` then `qcute.qcute_v5_concat`, same 5 configs each (`Ks=(4,1)` except
the last, `context_len=256`, `attn_window=(16,64)` except the last, 8000 steps): `_bsq8`, `_bsq16`,
`_gumbel`, plain (`qcute_v5_2`/`qcute_v5_concat_2`), and `_3` (`Ks=(1,)`, dense, out-of-family sanity
point). Relaunched fresh from step 0 after the chunked-decode-window fix above (all logs/checkpoints
from the pre-fix run deleted).

| job | best_val_bpb | gen_consistency |
|---|---:|---|
| `qcute_v5_2_bsq8` | 3.0225 | clean |
| `qcute_v5_2_bsq16` | 2.9261 | clean |
| `qcute_v5_2_gumbel` | 2.7736 | 83/127 mismatched (expected — gumbel noise diverges run-to-run even between two calls of the same generation function, pre-existing, not a regression) |
| `qcute_v5_2` | 2.9135 | clean |
| `qcute_v5_3` | 2.7093 | clean |
| `qcute_v5_concat_2_bsq8` | pending | — |
| `qcute_v5_concat_2_bsq16` | pending | — |
| `qcute_v5_concat_2_gumbel` | pending | — |
| `qcute_v5_concat_2` | pending | — |
| `qcute_v5_concat_3` | pending | — |

Final table (with `bytelm_xs1_ctx256`/`bytelm_xs4_ctx256` baselines, params/flops) still pending
once all 10 jobs finish.

## `_skip` (buffer-pruning) promotion + full-val-set eval (2026-08-17)

`qcute_v5_skip.py`/`qcute_v5_concat_skip.py` (buffer pruning: once a block's code exists, its raw
bytes are dropped from the decode buffer -- static mask-only for training, real KV-cache eviction
for generation) promoted to the new defaults `qcute/qcute_v5_stack.py` / `qcute/qcute_v5_concat.py`.
Prior qfb-based defaults archived as `qcute/archive3/qcute_v5_bos.py` /
`qcute/archive3/qcute_v5_concat_bos.py`. `qcute_v5_fixblock.py`/`qcute_v5_concat_fixblock.py`
(pre-`_skip`, qfb removed via decode_bos + block-0 exclusion) kept as intermediate reference forks.

Added a true full-val-set `--eval_only` mode to `qcute.bytelm`, `qcute.qcute_v5_stack`,
`qcute.qcute_v5_concat`, `qcute.qcute_v5_fixblock`, `qcute.qcute_v5_concat_fixblock`
(`eval_bpb_full`/`eval_model_full`): non-overlapping windows walked in fixed chronological order
from byte 0, no random sampling/replacement, every val byte scored exactly once -- unlike the
in-training `eval_batches`-sampled `best_val_bpb` (~82% of val set, fresh random sample each call,
source of the run-to-run bpb bounce noted earlier in this doc).

Also queued `bytelm_xs2_ctx256` (xs preset, `context=256`, `n_layers=2`) as a depth-matched
baseline between `bytelm_xs1_ctx256` (`n_layers=1`) and the `n_layers=4` default, for a fairer
comparison against v5's per-block decode (extra effective depth beyond a plain encoder).

**`Ks=(1,)`, `context_len=256`, 8000-step comparison — sampled `best_val_bpb` (in-training) vs.
deterministic full-val-set `val_bpb` (`--eval_only`):**

| run | best_val_bpb (sampled) | val_bpb (full valset) |
|---|---:|---:|
| `bytelm_xs1_ctx256` (n_layers=1) | 2.5586 | 2.5889 |
| `bytelm_xs2_ctx256` (n_layers=2) | 2.4383 | 2.4827 |
| `qcute_v5_fixblock_3` (stack, no skip) | 2.5043 | 2.5179 |
| `qcute_v5_concat_fixblock_3` (no skip) | 2.5024 | 2.5389 |
| `qcute_v5_skip_3` (stack, pruned) | 2.5810 | 2.6076 |
| `qcute_v5_concat_skip_3` (pruned) | 2.6450 | 2.6182 |

`bytelm_xs2` (plain 2-layer baseline) still wins outright. Both `_skip` (buffer-pruned) variants
are clearly worse than their `_fixblock` parents -- pruning raw bytes and relying solely on the
code as summary costs more accuracy than the compute/memory savings buy at this scale, and it
hurts the concat design's chronological merged-buffer somewhat more than stack's self-attention-
folded one. Full-valset numbers run consistently a bit higher than the sampled `best_val_bpb`
(expected -- that number is the best of several noisy sampled evals during training, this is one
honest full pass), but the ranking/gaps are unchanged.

**Qualitative generation, same train prompt, greedy/argmax** (`start=200000` byte offset into the
train split, `prompt_len=64`, `gen_len=64`, `qcute_v5_concat_fixblock_3` and `qcute_v5_concat_skip_3`
via `generate_no_cache`, bytelm via its own greedy `generate_no_cache`):

```
prompt:       ]]\n[[mk:Алабама]]\n[[ms:Alabama]]\n[[mo:Алабама]]\n[[
ground_truth: nl:Alabama]]\n[[ja:アラバマ州]]\n[[no:Alabama]]\n[[nn:Alabama]
```

| model | generated continuation | byte_acc vs GT |
|---|---|---:|
| `bytelm_xs1_ctx256` | `sk:Alabama]]\n[[sk:Алхия]]\n[[sk:Algeria]]\n[[sk:Algeria]]\n[` | 0.219 |
| `bytelm_xs2_ctx256` | `sl:Alabama]]\n[[sl:Alabama]]\n[[sl:Alabama]]\n[[sl:Alabama]]\n[[sl:A` | 0.234 |
| `qcute_v5_concat_fixblock_3` | `ms:Alabama]]\n[[da:Alabama]]\n[[da:Algeria]]\n[[da:Algeria]]\n[[da:A` | 0.234 |
| `qcute_v5_concat_skip_3` | `sv:Algeria]]\n[[sv:Alabama]]\n[[th:` + repeated Thai-script bytes | 0.156 |

All four pick up the `[[xx:Alabama/Algeria]]` interwiki-link structure from the prompt but never
predict the exact right language code (near-random target). `concat_fixblock` matches `bytelm_xs2`
on byte_acc; `concat_skip` is visibly worse and degenerates into repeated non-Latin-script bytes
partway through -- consistent with its full-valset bpb gap (2.539 vs 2.618) above.

## `code_sample_mode` sweep (2026-08-17, done)

Unified `Config.code_sample_mode: "ste" | "sample" | "soft"` added to `qcute_v5_stack.py`/
`qcute_v5_concat.py`, replacing the separate `use_gumbel_noise`/`bsq_use_bernoulli_sample` bools
-- `ste` (default, unchanged): plain softmax/sign(), hard forward, straight-through backward.
`sample`: stochastic hard forward (Gumbel noise for softmax, Bernoulli(sigmoid(v_unit)) for BSQ),
still straight-through backward. `soft`: no hard forward at all -- plain Gumbel-Softmax relaxation
(Jang et al. 2016) for softmax, raw normalized vector for BSQ. `check_gen_consistency` mismatches
under `sample`/`soft`-with-noise are expected (same structural RNG-stream-divergence reason as the
earlier `qcute_v5_2_gumbel` precedent); BSQ's `soft` mode is a deterministic function of `v`, so it
still matches exactly (0/127) unlike softmax's noisy `soft`. `Ks=(1,)`, `context_len=256`, 8000-step
family, same as the `_fixblock`/`_skip` comparison above:

| job | mode | best_val_bpb |
|---|---|---:|
| `qcute_v5_stack_gumbel` | softmax, `sample` | 2.6762 |
| `qcute_v5_concat_gumbel` | softmax, `sample` | 2.6331 |
| `qcute_v5_stack_bsq16` | bsq16, `sample` | 3.8748 (full valset: 3.9047) |
| `qcute_v5_concat_bsq16` | bsq16, `sample` | 3.8782 (full valset: 3.9007) |
| `qcute_v5_stack_bsq16_ste` | bsq16, `ste` | 2.8518 |
| `qcute_v5_concat_bsq16_ste` | bsq16, `ste` | 2.7986 |
| `qcute_v5_stack_soft` | softmax, `soft` | 2.4886 |
| `qcute_v5_concat_soft` | softmax, `soft` | 2.4802 |

`sample`-mode softmax runs land worse than their `ste` `_fixblock` counterparts (2.50/2.50) and
even their `_skip` counterparts (2.58/2.65) -- stochastic hard-forward sampling adds training noise
on top of the pruning cost, as expected. BSQ16 is weak across the board (~3.87-3.88 sampled,
~3.90 full-valset for `sample` mode) -- `ste` helps a lot vs `sample` (2.80-2.85 vs 3.87-3.90) but
BSQ16's underlying weakness isn't primarily the stochastic sampling; something more fundamental
about the 16-bit BSQ setup itself (also note: `bsq_bits=16` inflates params to 35.3M via
`CodeEmbed`'s `2**16 x D` lookup table, vs 1.84M for the softmax variants -- not an
apples-to-apples parameter budget). **`soft` mode is the best v5 result of the whole session**
(2.4886/2.4802, beating every `ste`/`fixblock`/`_skip` variant, edging close to
`bytelm_xs2_ctx256`'s 2.4383).

**But `soft`'s generation quality doesn't match its bpb** -- qualitative greedy generation (same
`start=200000`/`prompt_len=64`/`gen_len=64` train-prompt methodology as the `_fixblock`/`_skip`
comparison) shows `soft`-mode generation is genuinely *worse* than every other variant compared
this session, including the plain `bytelm` baselines:

```
prompt:       ]]\n[[mk:Алабама]]\n[[ms:Alabama]]\n[[mo:Алабама]]\n[[
ground_truth: nl:Alabama]]\n[[ja:アラバマ州]]\n[[no:Alabama]]\n[[nn:Alabama]
```

| model | mode | byte_acc vs GT | notes |
|---|---|---:|---|
| `bytelm_xs1_ctx256` | -- | 0.219 | |
| `bytelm_xs2_ctx256` | -- | 0.234 | |
| `qcute_v5_stack_soft` | native (soft, stochastic) | 0.062 | loops on "Abraham Lincoln" |
| `qcute_v5_stack_soft` | argmax override (`ste`, same weights) | 0.125 | less repetitive |
| `qcute_v5_stack_soft` | high-tau override (soft, tau=4.0) | 0.031 | worst -- more noise, no benefit |
| `qcute_v5_concat_soft` | native (soft, stochastic) | 0.172 | loops on "American Samoa" |
| `qcute_v5_concat_soft` | argmax override (`ste`, same weights) | 0.141 | slightly worse than native |
| `qcute_v5_concat_soft` | high-tau override (soft, tau=4.0) | 0.078 | worst |

Root cause: `code_sample_mode` only affects `quant.mode` at call time, not the weights, so a
`soft`-trained checkpoint can be generated from under any mode by just overriding `Config` before
loading `state_dict` -- used here to isolate "does the noise hurt at generation time specifically."
Forcing deterministic argmax (`ste`) at generation time roughly DOUBLES `stack_soft`'s byte_acc
(0.062->0.125, visibly less repetitive), but slightly HURTS `concat_soft` (0.172->0.141, more
repetitive) -- no universal fix, the right decoding strategy is itself decode-architecture-
dependent (stack's staged cross-attention benefits from decoupling train-time noise from
generation-time determinism; concat's flat merged-buffer self-attention doesn't). High temperature
(more noise) makes both strictly worse in every case tried. Overall: `soft` mode's excellent
held-out bpb does not transfer to generation quality -- a genuine train/inference mismatch, not
just a caveat from the Jang et al. literature review earlier this session, now empirically
confirmed on our own checkpoints.

## `qcute_v5_concat_modes.py`: multi-mode decode loss (2026-08-17)

Forked from `qcute_v5_concat.py` to add `Config.multi_mode_impl: "off" | "multipass" |
"single_pass"` -- `qcute_v5_stack.py`'s staged cross-attention decode gets a loss at every
conditioning depth (self-only, self+track1, self+track1+track2, ...) for free, as a byproduct of
its sequential per-track stages (`decode_stage_extra_losses`); `qcute_v5_concat.py`'s decode is one
flat self-attention pass over a merged buffer with no such intermediate readout to tap. This fork
adds it:
- `"off"` (default): unchanged behavior, verified bit-exact no-op vs. plain `qcute_v5_concat.py`.
- `"multipass"`: naive reference -- calls the per-level decode once per mode `m=1..T` (`T` =
  number of available tracks), each with `tracks[:m]`.
- `"single_pass"`: one shared pass through `self.blocks` gets every mode via a BLOCK-DIAGONAL
  attention mask -- each mode is its own independent segment (its own
  `_merged_layout(L, tracks_meta[:m], device)` buffer), concatenated with zero cross-segment
  attention. Provably exact (isomorphic to `multipass`, not an approximation). An earlier design
  (duplicate query-only rows reading from a shared backbone via masking, cheaper in principle) was
  considered and rejected during design: it doesn't reproduce a shallower mode's true behavior at
  points where a lower-category code ties with a higher one at the same timestep -- the tied code's
  own recomputed hidden state matters, not just a masked read of it. The corrected block-diagonal
  design costs the SAME total compute as `multipass`; its real value is fewer Python/kernel launches
  (batched into one pass), not fewer FLOPs -- a smaller win than originally scoped, worth being
  explicit about. Per-mode losses feed `decode_stage_extra_total`, mirroring `qcute_v5_stack.py`'s
  field name/semantics for its own staged intermediate losses. Dense (non-chunked) attention only --
  chunked/banded configs fail loudly via `AssertionError`, not silently wrong.

Verified (`scripts/test_v5_concat_modes.py`) across `Ks=(1,)`, `(4,1)`, `(2,2,1)`: `off == plain`
and `single_pass == multipass` both exact (`torch.allclose`, all per-level metrics). End-to-end
training smoke-tested for all three (`configs/qcute_v5_concat_modes_ks{1,41,221}.py`, 50 steps):
`decode_stage_extra_total` correctly `0.0` for `Ks=(1,)` (T=1, no shallower mode exists), non-zero
for the other two (`Ks=(4,1)`: 1 extra mode at level0; `Ks=(2,2,1)`: 2 extra modes at level0, 1 at
level1), `check_gen_consistency` clean (0/15) in all three. No real (non-smoke) training queued yet.
