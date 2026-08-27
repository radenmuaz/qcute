# Code-conditioning ablation on `qcute_v5_concat_1` checkpoint (2026-08-16)

`scripts/ablate_v5_concat.py` — loads `checkpoints/qcute_v5_concat_1/best.pt` (`configs/qcute_v5_concat_1.py`:
`Ks=(4,1)`, `d_model=256`, `n_layers=1`, `context_len=1024`, step 2000) and measures val bpb under
level-0 decode with different code inputs, 20 batches x batch_size=16, `device=cpu`:

| mode | bpb | what it feeds level-0 decode |
|---|---|---|
| `uncond` | 2.685 | `encode_lms[0]` alone, no cross-attention to any code |
| `full_cond` | 2.616 | normal forward: self track = `c_list[0]`, cross track = `decode_derived_c[1]` (teacher-forced, ground-truth-conditioned) |
| `ar_level1_cond` | 3.156 | cross track replaced by `encode_lms[1]`'s own autoregressive rollout (block 0's code is the real encode output, every later code is `enc1`'s own argmax prediction fed back in — no ground truth after block 0) |
| `perturb_self` | 3.092 | self track (`c_list[0]`) replaced with uniform-random ids |
| `perturb_cross` | 3.945 | cross track (`decode_derived_c[1]`) replaced with uniform-random ids |
| `perturb_both` | 4.296 | both replaced with random ids |

Two findings:
- **The model actively relies on both code streams, it doesn't ignore them.** Randomizing either
  track makes bpb *worse than `uncond`* (3.092/3.945 vs. 2.685) — a genuinely useless code would
  leave bpb near `uncond`, not push it higher. `full_cond`'s edge over `uncond` is modest (2.616 vs
  2.685) given how much worse things get when either code is corrupted, meaning the model has grown
  dependent on codes that individually carry a fairly small net benefit — consistent with
  [qcute_refine_v4_4_1_v4_5_1_math.md](qcute_refine_v4_4_1_v4_5_1_math.md)'s self-code LM
  continuation being easy to overfit onto teacher-forced ground truth.
- **Exposure bias in the level-1 code LM is severe.** Feeding level-0 decode the SAME cross-track
  role but with genuinely autoregressive (not teacher-forced) level-1 codes (`ar_level1_cond`,
  3.156) is worse than not conditioning on level-1 codes at all (`uncond`, 2.685) — `encode_lms[1]`'s
  own next-code predictions drift far enough from the ground-truth code distribution that level-0
  decode, trained only ever having seen ground-truth-derived codes, is actively misled by them. This
  is the same "moving-target/cascade effect for n_levels=2" flagged in the Architecture section of
  [CLAUDE.md](../CLAUDE.md) and earlier in [status.md](status.md) — this ablation is concrete
  quantitative evidence of it on a real (if lightly trained, 2000-step) checkpoint, not just a
  training-curve symptom.
