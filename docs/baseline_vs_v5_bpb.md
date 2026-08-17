# Baseline vs. qcute_v5 family: params, FLOPs, val bpb (2026-08-17)

Params via `sum(p.numel() for p in model.parameters())`, FLOPs via `torch.utils.flop_counter.FlopCounterMode`
(single forward, batch=1, CPU — same methodology as `scripts/bench_forward.py`). bpb is each run's
checkpointed best val bpb (`checkpoints/<run_name>/best.pt`'s save point, from `logs/<run_name>/run.jsonl`).

| model | context | layers | params | flops/fwd | flops/byte | best val bpb |
|---|---:|---:|---:|---:|---:|---:|
| `bytelm_xs1_ctx256` | 256 | 1 | 1.050M | 536.9M | 2.10M | 2.5586 |
| `bytelm_xs4_ctx256` | 256 | 4 | 3.412M | 1744.8M | 6.82M | 2.3560 |
| `qcute_v5_concat_1` | 1024 | 1 (x2 tracks) | 3.682M | 6286.7M | 6.14M | 2.5848 |
| `qcute_v5_1` | 1024 | 1 (x2 tracks) | 4.603M | 7213.6M | 7.05M | 2.5676 |

Configs: `configs/bytelm_xs1_ctx256.py`, `configs/bytelm_xs4_ctx256.py` (new this session — plain
byte-level baselines at the xs preset's native context=256, differing only in `n_layers`, for a
layer-count ablation pair); `configs/qcute_v5_concat_1.py`, `configs/qcute_v5_1.py` (`Ks=(4,1)`,
`n_layers=1`, `context=1024`, the two current-default hierarchical decode mechanisms — chronological
merged-interleave vs. qfb-fixed staged cross-attention, see [status.md](status.md)'s Architecture
summary — run back-to-back on the same data/step budget for a fair comparison).

Findings:
- **Depth matters for the plain baseline.** `bytelm_xs4_ctx256` (4 layers) beats `bytelm_xs1_ctx256`
  (1 layer) by 0.20 bpb (2.356 vs 2.559) at context=256 — depth buys real capacity at this scale, not
  just a marginal gain.
- **Both v5 hierarchical decoders land within noise of each other and of the weakest baseline.**
  `qcute_v5_concat_1` and `qcute_v5_1` — same `Ks`, `n_layers`, context — differ by only 0.02 bpb
  (2.585 vs 2.568) from each other, and neither clearly beats plain `bytelm_xs1_ctx256` (2.559)
  despite spending 3.5-4.4x the params and ~3x the flops/byte.
- **Neither v5 variant beats the deeper plain baseline.** `bytelm_xs4_ctx256`'s 2.356 bpb is the best
  number in this table, achieved with the fewest params (3.412M) and lowest flops/byte (6.82M) of any
  non-`xs1` entry — at comparable or greater compute, the hierarchical architecture doesn't yet pay
  for its added complexity on this corpus/step budget.
- `qcute_v5_1` edges out `qcute_v5_concat_1` slightly (2.568 vs 2.585) at somewhat higher cost (4.60M
  vs 3.68M params, 7.05M vs 6.14M flops/byte) — consistent with qfb's extra boundary-query
  cross-attention pass adding real compute for a real (if small) accuracy gain.
- Both hierarchical checkpoints are their run's *best* point, not their final one — both overfit past
  step 2000 (see [status.md](status.md)'s "Code-conditioning ablation" note and
  [ablate_v5_concat_1.md](ablate_v5_concat_1.md)), so these are the most favorable bpb numbers either
  architecture reaches in an 8000-step run, not a snapshot mid-improvement.
