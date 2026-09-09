# image_lagcodec status

Living doc — check here rather than assuming CLAUDE.md is current. Update in place (don't append) on every run start/stop/change.

## What it is

`image_lagcodec/run_lagcodec.py` (+ shared primitives in `image_lagcodec/eqx_common.py`) — a JAX/Equinox hierarchical PQ-VQ-VAE codec for CIFAR-10, built as a faithful, generalized port of the `qcute_lagcodec` torch reference's `StackDecoder`. Unlike `image_gen_cifar` (a from-scratch hard-fork), this port deliberately tracks the reference's architecture and diverges only where explicitly decided (see below).

## Architecture

- **Encoder** (`HierEncoder`/`EncoderLevel`): N hierarchical levels (`strides` tuple, e.g. `(3,4,4)`), each a causal-LM level producing PQ categorical codes (`code_vocab` per chunk, `pq_chunks` chunks) via `quantize_hard` (softmax→argmax→one-hot STE), plus an auxiliary per-level NTP loss/accuracy.
- **`StackDecoder`** (`decoder_type="stack"`, the only actively-trained decoder): generalized to any depth/stride. Deliberate divergences from the torch reference, all explicitly approved:
  - **BOS token**: a separate trainable `(D,)` param (not part of the byte embedding table), prepended once — replaces the reference's dual h0/h0_shifted seed mechanism with plain shifted NTP.
  - **3 norms per layer** (`Level1Layer`: `norm1` self-attn, `norm_cross` cross-attn, `norm2` mlp) vs. the reference's 2 (shared `ln1` for self+cross).
  - **Every level self-attends** (self-attn → cross-attn-to-that-level's-code → mlp, per level per layer) vs. the reference's self-attn-once-at-track0-only.
  - **`lag` parameter** (invented this session, generalizes the reference's `own_code_min_lag`): `lag_bytes = (lag+1) * prod(strides[:-1])`; cross-attn mask is `code_pos <= query_pos + lag_bytes` — a pure forward-shift of the cross-attn mask, self-attention always fully causal/unbounded. `lag=0` = own-code-only visibility, higher = more codes visible earlier. `lag<0` (true causal NTP) is `assert`ed invalid — would need encoder-like shifted targets, judged too complex for now.
- **`StageLocalDecoder`** (`decoder_type="self_attn_local"`) and **`StageLagDecoder`** (`decoder_type="self_attn_lag"`): older/alternate decoders, kept but not used in active experiments. `StageLagDecoder` has a known divergence (code-as-token self-attention, no real cross-attention) — not fixed, out of scope.
- **Generation** (`StackDecoder` only): `reconstruct_full_recompute` (O(T²) diagnostic reference), `reconstruct_kv_cache` (incremental, un-jitted), `reconstruct_kv_cache_scan` (jit + `lax.scan`, ~12x faster than the un-jitted version, the one actually used in training's periodic qual-gen) — all three verified to agree byte-exact at toy scale.

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

- **2026-09-09**: all 4 full-CIFAR-10 runs (`cifar10_stack_lag{0,4,16,max}`) launched on `tpu1`/`tpu2`/`tpu3`/`tpu4` respectively (v4-8, `us-central2-b`), currently in epoch 1-2/100, ~1.6s/it, 195 steps/epoch, no crashes since the splash-attention tracer-leak fix (see above). First qual-gen samples land at epoch 10. Earlier `overfit1000_stack_lag0` sample already pulled to `image_lagcodec/logs/cifar_lagcodec_overfit1000_stack_lag0/samples_epoch3000_reconstruct.png`.
- Monitor: `tmux attach -t cifar10_stack_lag0` (tpu1) / `cifar10_stack_lag4` (tpu2) / `cifar10_stack_lag16` (tpu3) / `cifar10_stack_lagmax` (tpu4); `tmux capture-pane -t <name> -p -S -N` for a snapshot without attaching.
