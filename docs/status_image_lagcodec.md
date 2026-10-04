# image_lagcodec status

Living doc — check here rather than assuming CLAUDE.md is current. Update in place (don't append) on every run start/stop/change.

## What it is

`image_lagcodec/run_lagcodec.py` (+ shared primitives in `image_lagcodec/eqx_common.py`) — a JAX/Equinox hierarchical PQ-VQ-VAE codec for CIFAR-10, built as a faithful, generalized port of the `qcute_lagcodec` torch reference's `StackDecoder`. Unlike `image_gen_cifar` (a from-scratch hard-fork), this port deliberately tracks the reference's architecture and diverges only where explicitly decided (see below).

## Legacy architecture (StackDecoder, superseded — see Current below)

- **Encoder** (`HierEncoder`/`EncoderLevel`): N hierarchical levels (`strides` tuple, e.g. `(3,4,4)`), each a causal-LM level producing PQ categorical codes (`code_vocab` per chunk, `pq_chunks` chunks) via `quantize_hard` (softmax→argmax→one-hot STE), plus an auxiliary per-level NTP loss/accuracy.
- **`StackDecoder`** (`decoder_type="stack"`, the only actively-trained decoder): generalized to any depth/stride. Deliberate divergences from the torch reference, all explicitly approved:
  - **BOS token**: a separate trainable `(D,)` param (not part of the byte embedding table), prepended once — replaces the reference's dual h0/h0_shifted seed mechanism with plain shifted NTP.
  - **3 norms per layer** (`Level1Layer`: `norm1` self-attn, `norm_cross` cross-attn, `norm2` mlp) vs. the reference's 2 (shared `ln1` for self+cross).
  - **Every level self-attends** (self-attn → cross-attn-to-that-level's-code → mlp, per level per layer) vs. the reference's self-attn-once-at-track0-only.
  - **`lag` parameter** (invented this session, generalizes the reference's `own_code_min_lag`): `lag_bytes = (lag+1) * prod(strides[:-1])`; cross-attn mask is `code_pos <= query_pos + lag_bytes` — a pure forward-shift of the cross-attn mask, self-attention always fully causal/unbounded. `lag=0` = own-code-only visibility, higher = more codes visible earlier. `lag<0` (true causal NTP) is `assert`ed invalid — would need encoder-like shifted targets, judged too complex for now.
- **`StageLocalDecoder`** (`decoder_type="self_attn_local"`) and **`StageLagDecoder`** (`decoder_type="self_attn_lag"`): older/alternate decoders, kept but not used in active experiments. `StageLagDecoder` has a known divergence (code-as-token self-attention, no real cross-attention) — not fixed, out of scope.
- **Generation** (`StackDecoder` only): `reconstruct_full_recompute` (O(T²) diagnostic reference), `reconstruct_kv_cache` (incremental, un-jitted), `reconstruct_kv_cache_scan` (jit + `lax.scan`, ~12x faster than the un-jitted version, the one actually used in training's periodic qual-gen) — all three verified to agree byte-exact at toy scale.

## Current (2026-09-21)

- `run_lagcodec.py`: hierarchical enc/dec levels (`strides`). Causal encoder per level → PQ codes pooled at the last position of each K-block, + NTP head on its own input.
- Decoder = parallel block-local decode: groups of `decoder_ncodes` codes are independent rows `[own ctx | extra ctx | BOS | draft | targets]`.
- Decoder rows use a per-row key-valid mask (no pad embeddings): `[extras | own window | BOS | draft | own tokens]`, built by one shared `_pardec_ctx_rows` for training + generation. Non-existent slots are masked as keys; invalid draft slots input zeros.
- Knobs: `stream_chunks` (0 = per-group streaming, 1 = wait once, n = n chunks; replaces `streaming`) + `ncodes_window`; `cond_depth`/`cond_window`/`cond_drop` (strictly causal "complete" alignment, `decoder_ncodes` may be 1); `decode_past`/`decode_future`; `level_refine_passes`/`window`/`gumbel`/`gt_drop`/`drop` (refine early exit: `lax.cond`, same draw on all devices); `attn_window` (encoder); `remat`/`remat_level`/`refine_remat`.
- Data: `dataset` = `cifar`/`imagenet64`/`imagenet256`, `data_root`; shards from `scripts/imagenet/download_imagenet*.py`. Off-training scripts load via `dataset_from_config`.
- Multi-host TPU slice: `multihost=True`, run the same command on every host; batch is per device, each host reads its slice of the global batch.
- Eval: cascade gen per `top` (argmax + sampled, `gen_temperature=1`, `gen_top_k=8`); final eval runs every `top`. Prompted generation through the encoder NTP: `generate_from_prompt`, `scripts/prompt_generate.py` (CPU only).
- Gate: after editing `run_lagcodec.py`, `scripts/pardec_v2_consistency_check.py` must end `PASS all`.
- Gotchas: (1) TPU jit `argmax`→gather bug, use `safe_argmax`. (2) fp32 needs `jax_default_matmul_precision=highest` (set unless bf16). (3) Train OK but generation bad → suspect generation code first.
- Config parser: non-field constants (e.g. `DEPTH = 4`) warn and are ignored.

## res_denoise fork + torch port (2026-10-03)

- `run_lagcodec_res_denoise.py`: fork of `run_lagcodec_res.py`; all defaults bit-identical to it (regression harness: loss/aux/grads/eval/gen).
- Same-level cycles: `level_cycles` (per level), `level_cycle_mode` memoryless (default) | stack, `level_cycle_input` rollout (default, free-run, leak-free) | pss (`level_cycle_pss_prob`) | gt (overfit sanity), `level_cycle_detach`, `level_cycle_loss`, `gen_level_cycles`.
- Parallel scheduled sampling on a pardec row's own token inputs (2026-10-04, both files): `upsampler_pss_passes` / `downsampler_pss_passes` (per level, 1 = off, -1 = one pass per row token), `*_pss_prob` (own vs GT per position), `pss_input_mode` argmax | sample, `pss_temperature`.
  - Pass 1 teacher-forces GT; later passes re-feed the previous pass's detached prediction; loss on the last pass only. Context is untouched (that is `level_gt_drop`). Independent of refine passes.
  - n passes make the first n tokens of each row equal a real greedy rollout; -1 is exact (checked vs generation, linear head and ar head + `upsampler_rollout`). ar head without `upsampler_rollout`: digits inside a token stay GT-forced, so approximate.
  - Downsampler side is a no-op unless `downsampler_ncodes > 1` (one token per row has no token input). Eval always feeds own prediction. Cost: one extra forward per pass, backward only on the last.
  - TODO: CodeLM pss (its inputs are true codes from the prefill; only makes sense for free generation).
  - Fixed vs `run_lagcodec` cycle: TF-argmax GT leak, revision granularity (finer code windowed with the coarser stride/table), slot-0 train/gen mismatch, order-blind shared slot table, dropped empty slots.
- Backbones: `codelm/downsampler/upsampler_backbone` transformer | gru | linear_gru | ssm (`ssm_state_dim`); recurrent = fixed state, no KV cache.
- `context_source="codelm_upper"`: upsampler i reads CodeLM i+1; unshared adds a CodeLM-only top level (+ bos rows); that CodeLM also gets NTP on the top code; freeze trains CodeLM p in phase p.
- Data: `dataset="folder"`, `modality` image | text | audio | binary, `seq_len`, `audio_sample_rate`, `audio_encoding` mulaw8 | pcm16. Labels: `byte_mean_label_fn`, `byte_interp_label_fn`, `audio_resample_label_fn`, `bpe_label_fn` (stub, `HierarchicalBPE`), or `label_fn="pkg.module:fn"`. Example configs `configs/folder_*_overfit.py`.
- Gate: `scripts/res_denoise_consistency_check.py` must end `PASS all` (name args run a subset).
- `run_lagcodec_res_denoise_torch.py`: single-file PyTorch port, `--device auto|cpu|cuda|mps`, `--threads`; one device, bf16 autocast on CUDA only. Config/data/argparse sections are verbatim copies of the JAX file (re-sync after editing those).
  - Gate: `scripts/res_denoise_torch_parity_check.py` (JAX weights copied in; loss ~1e-7, grads ~5e-6, generation exact) must end `PASS all`. Label resize differs by ±1 at exact .5 values (~0.1%, XLA summation order).
- Results (tpu1, cifar overfit, freeze curriculum, 60k steps, one seed each; final gen mse val / train, n=8):
  - baseline `cifar_overfit_2stage_freeze`: top1 2489 / 2511, top0 2202 / 1529.
  - memoryless `_denoise` (1 gen cycle): top1 2928 / 1312, top0 2817 / 638 -- fits train far better, val worse; every extra gen cycle hurts (top1 train 1312 -> 2349 -> 3355): the re-encode loses information.
  - stack `_denoise_stack` (2 gen cycles): top1 2752 / 2530, top0 2355 / 1548 -- about baseline; 1 vs 2 gen cycles nearly equal.
  - Eval script: `scripts/denoise_cycle_eval.py <run> --cycles 1,2,3 --top N`.
- Fixed in the fork and (2026-10-04) ported to `run_lagcodec_res.py`; no-op for unshared configs, old shared checkpoints are already untied: `share_downsampler_upsampler_lm` / `context_source="shared_embed"` were tied only at init -- jax.grad gave each pytree position its own partial, adamw untied them after step 1. `tie_shared_grads` sums them (torch shares one Parameter).
- Memory flags: `downsampler_remat_chunks` / `upsampler_remat_chunks` (per level, 1 = off): pardec rows run in that many sequential checkpointed chunks (same loss/grads, ~1e-7). `codelm_remat` (per level, None -> `remat`). pmapped train steps donate model/opt_state/rng.
- Memory audit (2026-10-04, CPU saved-residual counts, small config): reinmax_limit's dense 256x256 surrogate per digit position was 55% of what backward keeps (81% with upsampler_rollout) -- now an exact O(K) form (`quantize_reinmax_limit`; slow reference `quantize_reinmax_limit_dense`): same forward, grads ~1e-6, saved memory 1791 -> 838 MiB (remat on: 1104 -> 152, upsampler_rollout: 4378 -> 886); rollout_prob<1 (lax.cond) keeps both branches' residuals; stack cycles recompute c(0)'s CodeLM context every cycle.
- Known: every pmap run (res too) recompiles once at step 2 (~2 min on TPU); `Attention.step` ignores `use_sink` (CodeLM free-run only).

## TODO — tiling / seam artifacts (2026-09-21 discussion, untested)

- Cause: tile g sees coarse codes ≤ its group + previous-pass drafts only; groups decode in parallel. Measured (imagenet64_5, 4 imgs): |dx| at tile edge 15.4 vs 8.6 inside (GT 9.5/10.0).
- Encoder must stay causal (NTP rollout); decoder may look at more parent codes (rollout can run ahead) — only latency.
- Config only: `streaming=False` + `ncodes_window=-1` on the top level; refine passes 3–4 + wider `level_refine_window` (z-order 4×4 tiles ≈6, raster 1); `level_refine_gt_drop`~0.5; smaller strides / ≥48 bits per finest code; z-order has the shortest seams.
- Done (2026-09-21): `stream_chunks` (chunked parent visibility) + key-valid mask refactor; tests: equivalence, leak, dense-vs-incremental, independent row reference. Not yet trained: see `configs/imagenet64_7.py`.
- Done (2026-09-22): `decode_generate_pardec_sync` (real cross-wave sync, generalizes the never-built `sync` flag) -- splits generation into `wave_groups` sequential waves (default = stream_chunks' own chunk size), each wave's decode_past draft reads the REAL previous wave's output off a running cache (not a private per-group redecode). wave_groups=n_groups is bit-identical to plain decode_generate_pardec; wave_groups=1 is the fully-causal group-by-group extreme, verified to reproduce itself under teacher-forced re-scoring at 100% (vs ~6% for vanilla generation's private redraft, on a toy config) -- tests: run_sync_wave. Scope: cond_depth<=1 tested (cond_depth=2 spot-checked, composes via the shared ends/chunk math, not covered by a gated test); cycle_refine_passes is orchestrated above this function, untouched. level_refine (intra-wave coherence) is orthogonal and still needed for wave_groups>1 -- NOT yet composed with sync in one call. NOT wired into any training/eval call site (opt-in function only, zero risk to running jobs); training-side (teacher-forced multi-wave) not built.
- Minor code: (bounded parent lookahead → covered by `stream_chunks`); future draft (Pf tokens from prev pass, placed before own tokens); sequential/wavefront generation (`sync=True` stub, `gen_decode_future` exists); scheduled sampling / masking on `decode_past`; fixed-point early exit at inference, more passes than train.
- Not recommended: blending overlaps at inference (discrete tokens); extra early codes (causal encoder can't add info at the prefix).
- Measure per-tile error at eval on ~512 images (current evidence: 4 imgs, no early/late trend).
- Open: codebook `util` 0.47 in imagenet64_6 vs ~0.85 in imagenet64_5 — cause not isolated.
- Open: `configs/run5.py` (`DEPTH=4`) asserts on `attn_window`/`attn_lookahead` length.

## Attention backend (2026-09-08/09)

Both self-attention (encoder + decoder) and cross-attention (decoder's `cross_attn_own_code`, the batched training path only) run on the **Pallas TPU `splash_attention`** kernel, not `jax.nn.dot_product_attention` (confirmed its TPU `xla` backend doesn't fuse/save memory) and not the plain `flash_attention` kernel (hit a `vmem` tiling limit at real batch size).

- `eqx_common.py`: `splash_attention()` (causal/full self-attn, native GQA) and `splash_cross_attention()` + `LagCrossMask` (rectangular q×kv, the lag-shifted mask as a custom `_ComputableMask`).
- **Not** wired to splash: `cross_attn_own_code_single`/`self_step` (incremental single-query decode) — padding a length-1 query to the 128-lane minimum would only add overhead, no memory win at that scale.
- Gotchas hit and fixed, in case they recur: (1) `BlockSizes` needs the backward-pass fields (`block_q_dkv` etc.) set too, or `grad` raises `"Need to specify backward blocks"` — forward-only smoke tests won't catch this. (2) Never `functools.lru_cache` a built `SplashAttentionKernel` — it holds `jnp.array`-converted `MaskInfo` bound to whichever trace first built it; reusing it in a later, separate trace raises `UnexpectedTracerError`. Rebuild fresh every call. (3) Splash's `block_kv_compute` must be a multiple of 128 (TPU lane width) — pad q/k/v to a 128-multiple and slice back down, rather than shrinking the block size like `flash_attention` allowed.
- Real-scale (batch=64/device, d_model=512, `SEQ_LEN=3072`) forward: ~227-260MB peak per chip (was 100GB+ OOM with plain einsum attention).

## Precision

`Config.precision` (default `"bf16"`): casts the model to `bfloat16` transiently inside the training/eval loss and `run_reconstruct`'s forward calls; master weights/grads/optimizer state stay `float32` (JAX's `astype` VJP upcasts the cotangent automatically, no extra plumbing needed). `"fp32"` disables the cast — diagnostic correctness mode (e.g. for exact-match checks against `reconstruct_full_recompute`).

**Caveat (2026-09-09): measured zero speedup from bf16** at real scale (1.59-1.61s/it vs. fp32's 1.60s/it) — likely because JAX's default TPU matmul precision already runs bf16-like reduced-precision passes for fp32 inputs. Not yet root-caused further; the real per-step bottleneck is still unidentified (candidates: splash's block-routing overhead, sinkgd's iterative steps, non-matmul dispatch).

## Batch size ceiling (structural, not precision-related)

`batch_size` in configs is **per-device**; `n_devices` resolves to `jax.local_device_count()` (4 on the v4-8 nodes in use), so effective global batch = `batch_size * 4`. At the current model size (d_model=512×3 levels, `SEQ_LEN=3072`), the backward pass hits a sharp, nonlinear XLA rematerialization-strategy cliff between per-device batch 64 (fine, ~400MB) and 128 (72.1G fp32 / 66.1G bf16, both OOM against the 30.75G/chip budget) — bf16 barely moves the number, confirming it's an XLA scheduling decision (store-vs-recompute), not raw bit-width. Fixing this for real would need explicit `jax.checkpoint`/remat, not yet done. **Current ceiling: `batch_size=64` (256 effective).**

## Config / invocation

```
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_lag0.py
```
`configs/cifar10_stack_lag{0,4,16,max}.py`: the 4 active full-CIFAR-10 run configs (only `lag` differs: 0/4/16/255). `d_model=(512,512,512)`, `strides=(3,4,4)`, `code_vocab=16`, `pq_chunks=4`, `batch_size=64`, `optimizer="sinkgd"` (1 sinkhorn iter), `lr=1e-3` with `warmup_cosine` schedule (1000 warmup steps), `epochs=100`, `eval_every_epochs=10` (runs val + qual-gen reconstruction PNGs + checkpoint). Train qual-gen: greedy. Val qual-gen: sampled, `temperature=0.01`. Standard CIFAR-10 train/val split (`train_subset_n=None`).

Logging: `run.jsonl`/`run.log` in `image_lagcodec/logs/<run_name>/`. Per-step (every `log_every`) and tqdm-live: `train_loss` (the actual combined objective — decode loss + `ntp_weight`·ntp loss — was silently discarded before 2026-09-09), `byte_bpb/acc`, `ntp_bpb/acc`, `util`. Per-epoch: `train_loss_epoch_avg` (new 2026-09-09, mirrors val's existing per-eval aggregate). `BatchIterator` now has `__len__` so tqdm shows a real total/ETA (was missing before 2026-09-09).

## Run status

- **2026-09-21**: `imagenet64_6` (d_model 512, strides (16,16), `decoder_ncodes` 16, `level_refine_drop` 0.5) running on tpu34 (v4-16, 2 hosts), batch 4/device, 80,072 steps (2 epochs), ~3 it/s, first eval at step 40,036. `imagenet64_5` stopped at step 10,008 (val loss 8.34, cascade mse 446). `imagenet64_1` (tpu2) stopped early. Details: [status_tpu.md](status_tpu.md).
