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

| stage | seq len (L) | flops |
|---|---:|---:|
| `encode_lms[0]` (byte-level) | 1024 | 1644.2M |
| `encode_lms[1]` (level-1 codes) | 256 | 469.8M |
| decode level0 stage0 (self track) | 1024 q x 256 kv | 1946.0M |
| decode level0 stage1 (cross track to level1) | 1024 q x 256 kv | 1979.6M |
| decode level1 stage0 (self track) | 256 q x 256 kv | 905.8M |
| **sum of components** | | **6945.6M** |
| **actual total forward** | | **7213.6M** |

(~268M gap is code-head/loss overhead not isolated in the per-stage calls.)

See [baseline_vs_v5_bpb.md](baseline_vs_v5_bpb.md) for the full params/flops/bpb table this
breakdown feeds into.
