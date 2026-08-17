# qcute_v5_1 FLOPs vs. bytelm: matched-context comparison + a methodology correction (2026-08-17)

## Correction to the FLOP numbers used everywhere else in this doc set

**`torch.utils.flop_counter.FlopCounterMode` (torch 2.13.0) does not count attention
(`scaled_dot_product_attention`) FLOPs at all** — confirmed by dumping `fc.get_flop_counts()`: every
entry is `aten.mm`/`aten.addmm` (QKV/out projections, MLP, code/embed heads); no `sdpa`/attention op
ever appears, even though `torch.utils.flop_counter` does define `sdpa_flop`/`sdpa_flop_count`
formulas — they're evidently not hooked for the plain `F.scaled_dot_product_attention` call path used
here. Direct proof: forcing `qcute_v5_1` to dense attention (`attn_window=(-1,-1)`, no windowing at
all) produces the **exact same flop count, bit-for-bit** (`7213547520`) as the trained windowed
config (`attn_window=(16,256)`). Attention cost cannot possibly be identical between dense and
`window=16` at `L=1024` — the measurement is simply blind to it.

**Consequence**: every FLOP number in [baseline_vs_v5_bpb.md](baseline_vs_v5_bpb.md) and the
previous version of this doc is a *linear/MLP-only* FLOP count, not true total compute. That doesn't
invalidate the relative comparisons between architectures at a *fixed context length* (linear/MLP
FLOPs are still real, still scale with `L*n_layers*d^2`, and still dominate wall-clock at this
`d_model=256` scale in practice) — but the previous version of this doc's explanation ("windowing
only saves the attention term, which is small anyway") was correct in conclusion by coincidence, not
for the reason stated: the measurement can't see the attention term changing at all, windowed or not,
so it never validly demonstrated the term is small. Treat all "flops" figures in this repo's docs as
**linear/MLP FLOPs**, not full transformer FLOPs, until a methodology that hooks SDPA is used instead.

## Matched-context comparison (linear/MLP FLOPs only, batch=1, CPU)

| model | context | layers | params | flops (linear/MLP only) |
|---|---:|---:|---:|---:|
| `bytelm_xs4_ctx256` (current default xs preset) | 256 | 4 | 3.412M | 1744.83M |
| `bytelm_xs4_ctx1024` (`configs/bytelm_xs_mtp4_ctx1024.py`'s architecture) | 1024 | 4 | 3.412M | 6979.32M |
| `qcute_v5_1` (dense or windowed — identical under this methodology) | 1024 | 1 (x5 stacks) | 4.603M | 7213.55M |

`bytelm_xs4_ctx1024`'s flops are exactly `bytelm_xs4_ctx256`'s x4 (`1744.83 * 4 = 6979.32`), confirming
linear/MLP cost scales linearly with context length as expected, independent of windowing.

**At matched context (1024), `qcute_v5_1` costs only 1.034x `bytelm_xs4_ctx1024`'s linear/MLP FLOPs**
(7213.55M vs. 6979.32M) — not the 4.13x gap reported earlier against `bytelm_xs4_ctx256`. Almost all
of that earlier gap was simply the 4x context-length mismatch (1024 vs 256), not hierarchy overhead:
`qcute_v5_1` runs 5 separate transformer stacks (2 encode levels + 3 decode tracks, see the per-stage
breakdown below) at `n_layers=1` each, while `bytelm_xs4_ctx1024` runs 1 stack at `n_layers=4` — a
similar total `stacks * n_layers` depth budget, so the two land close together once context is held
fixed. `qcute_v5_1` also uses ~1.35x the params (4.603M vs 3.412M — the extra `code_head`/
`decode_boundary_query`/duplicate embed tables per stage) for that near-parity FLOP cost.

## Per-stage breakdown (`qcute_v5_1`, `context_len=1024`, `attn_window=(16,256)` — same under dense)

Recomputed with each stage's `compute_ntp`/`want_code` flags matching exactly what `RefineLM._run`
actually passes (earlier version of this table used slightly different flags per stage, hence a
larger apparent gap):

| stack | seq len (L) | flops |
|---|---:|---:|
| `encode_lms[0]` (byte-level encode) | 1024 | 1778.25M |
| `encode_lms[1]` (level-1 encode, over codes) | 256 | 503.19M |
| `decode_stage_lms[0][0]` (level0 self track) | 1024 | 1946.03M |
| `decode_stage_lms[0][1]` (level0 cross track → level1) | 1024 | 1979.58M |
| `decode_stage_lms[1][0]` (level1 self track) | 256 | 905.84M |
| **sum of components** | | **7112.88M** |
| **actual total forward** | | **7213.55M** |

(~100.66M / 1.4% gap is unattributed loss-head/misc `F.linear` calls not captured by the isolated
per-stage calls — see the raw `fc.get_flop_counts()` module dump for why: query/key/value
projections inside `cross_attn_stage`/`forward_cross` are computed via `F.linear` on sliced
`self.qkv.weight` rows rather than calling `self.qkv(...)` as a submodule, so `FlopCounterMode`'s
module-attribution hook doesn't tag them to a specific submodule the way it does `attn.out`/`mlp`.)

**Why `qcute_v5_1` still costs more than `bytelm_xs4_ctx1024` despite a similar `stacks * n_layers`
depth budget** (5 stacks x 1 layer vs. 1 stack x 4 layers, same `L=1024`): 3 of `qcute_v5_1`'s 5
stacks pay costs `bytelm` never does. `decode_stage_lms[0][0]`/`[0][1]` (the two level-0 decode
tracks) each run **two** full attention+MLP passes per stage, not one — the main cross-attention pass
over the `L=1024` byte queries, plus the qfb boundary-query pass over the 256 block-boundary
positions, both through the same MLP — so each of those two stacks pays the linear/MLP cost of
`1024+256=1280` effective positions, not 1024. `decode_stage_lms[1][0]` similarly pays for
`256+256=512` positions instead of 256. Every one of the 5 stacks also carries its own separate
`code_head` classifier matmul (4 total, ~33.5M each) and its own NTP-loss output projection, where
`bytelm` has one shared embed/head pair for the whole model. That's the entire 1.034x gap
(7213.55M vs. 6979.32M).

## `qcute_v5_concat` vs. `qcute_v5` on the identical config (2026-08-17)

Same config as `qcute_v5_1`'s checkpoint (`Ks=(4,1)`, `d_model=256`, `n_layers=1`,
`context_len=1024`, `attn_window=(16,256)`), run through each decode mechanism:

| module | params | flops (linear/MLP only) |
|---|---:|---:|
| `qcute_v5_concat` (chronological merged-interleave decode) | 3.682M | 6299.32M |
| `qcute_v5` (staged cross-attention decode, qfb) | 4.603M | 7213.55M |

`qcute_v5_concat` uses **12.7% fewer flops** (914.23M less) and **20% fewer params**, consistent with
the per-stage breakdown above: `qcute_v5`'s staged decode pays for a *separate* qfb boundary-query
cross-attention+MLP pass at every decode stage plus a separate `code_head`/embed table per stage;
`qcute_v5_concat`'s single merged-buffer decode does one pass per level with shared buffer indexing
instead, avoiding both extra costs by construction.

See [baseline_vs_v5_bpb.md](baseline_vs_v5_bpb.md) for the full params/flops/bpb table this
breakdown feeds into.
