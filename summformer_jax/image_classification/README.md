# image_classification

Byte-level ImageNet-1k classifier built on the same `summformer.py` backbone (trunk + FuseStage
cross-attention) as `image_gen/` and `lm/`, used here as a general vision backbone rather than an
autoregressive image/text model. Goal: exercise the FuseStage mechanism on a real downstream task
(classification) at a ViT-Tiny-scale param count, not just density estimation.

## Files

- `summformer.py` — frozen backbone, verbatim duplicate of `image_gen/summformer.py`. Not modified
  here; if the backbone needs a change, make it in `image_gen/summformer.py` and re-copy.
- `classifier.py` — `SummClassifier`: wraps the backbone's `_cascade` (trunk forward pass, already
  returns `ln_f`-normalized final hidden states, no generation/LM-head/MTP machinery involved) with
  a linear classification head. Two pooling variants:
  - **unidirectional** (`bidirectional=False`): single causal pass, last position's hidden state →
    linear head.
  - **"bidirectional"** (`bidirectional=True`): NOT true non-causal attention — this codebase's
    windowed attention (`chunked_windowed_attention`/`causal_mask`) is only defined for a causal
    direction, so genuine bidirectional windowing isn't supported. Instead: forward-causal pass +
    reverse-causal pass (sequence reversed), mean-pooled hidden states from each, averaged together
    → linear head.
  - Also provides `cross_entropy_logits`, `topk_accuracy` (top-1/top-5), and `loss_and_metrics`.
- `dataloader.py` — `ImageNetClassificationLoader`: reads `scripts/download_imagenet.py`'s labeled
  `.bin` shard format (`[4-byte label][8-byte length][raw JPEG bytes]`), decodes via PIL, and
  applies preprocessing ported from Flax's `examples/imagenet/input_pipeline.py`
  (`ref_impl_files/input_pipeline.py`): random-resized-crop + horizontal flip for training,
  center-crop for eval — **no float normalization** (unlike the Flax reference's `MEAN_RGB`/
  `STDDEV_RGB`), output stays raw `uint8` byte tokens, matching `image_gen`'s byte-level convention.
  Uses `mmap` for shard access (not eager `f.read()`) — see "Known issues found" below for why that
  matters.
- `train.py` — config-driven training loop, `jax.pmap` data-parallel across all local devices
  (mirrors `image_gen/train.py`'s pattern), AdamW + linear-warmup/cosine-decay LR schedule
  (DeiT-style, not the Flax ResNet reference's SGD+momentum), top-1/top-5 accuracy logging.
- `scripts/download_imagenet.py` — copy of `image_gen`'s script, label-aware (writes each image's
  int class label alongside its raw bytes — `image_gen`'s own copy doesn't need labels, so this
  folder keeps its own).
- `configs/tiny_vit_like.py` — `d_model=128, n_layers=16` (~4.66M params, ViT-Ti/16 reference is
  ~5.7M), `main_window=24`, `code_window=24`, cross-attention window set to each fuse-stage's own
  stride (minimum valid — see below).
- `ref_impl_files/` — reference files fetched from
  [google/flax's examples/imagenet](https://github.com/google/flax/tree/main/examples/imagenet)
  (`input_pipeline.py`, `train.py`, `models.py`, `main.py`, `configs_default.py`, `README.md`) —
  the ResNet50 baseline this project's data pipeline/recipe structure is ported from. Not run
  directly (no TensorFlow dependency here), kept for reference only.

## Usage

```bash
# 1. Download labeled raw data (once) -- writes [label][JPEG bytes] shards
uv run python summformer_jax/image_classification/scripts/download_imagenet.py \
    --split train --out_dir /dev/shm/imagenet_raw --num_workers 32
uv run python summformer_jax/image_classification/scripts/download_imagenet.py \
    --split validation --out_dir /dev/shm/imagenet_raw --num_workers 14   # cap workers at file count, see caveat below

# 2. Train
uv run python summformer_jax/image_classification/train.py \
    --config summformer_jax/image_classification/configs/tiny_vit_like.py \
    --shard-dir /dev/shm/imagenet_raw
```

## Training recipe

Approximates **DeiT** (Touvron et al. 2021, "Training data-efficient image transformers"), the
standard from-scratch-on-ImageNet1k recipe at this exact param scale (`ViT-Ti/16`) — not the Flax
ResNet50 reference's SGD+momentum recipe, since this is a transformer, not a CNN:
- AdamW, `weight_decay=0.05`, `base_lr=5e-4` scaled by `batch_size/512` (DeiT's own linear-scaling
  convention — note this is `/512`, not the Flax ResNet reference's `/256`).
- Linear warmup (5 epochs) + cosine decay, ~100-300 epoch budget.

**Caveat on the 75% top-1 target**: DeiT-Tiny's own paper reports **~72.2%** top-1, not 75%, and
that's *with* their full augmentation stack (RandAugment, Mixup, CutMix, stochastic depth, label
smoothing) — none of which this pipeline implements yet (only random-resized-crop + flip). Matching
ViT-Tiny's param count alone doesn't get you to 75%; DeiT needed that whole augmentation stack to
get ViT training to work well from scratch on ImageNet1k-only data. Recalibrate the accuracy target
or add augmentation before expecting to hit 75%.

## Known issues found (and fixed) while building this

- **Unbounded cross-attention OOMs at this sequence length.** `context_len` here is `224*224*3 =
  150,528` — ~12x `image_gen`'s `12,288`. Leaving a fuse-stage's cross-attention window at `-1`
  (unbounded, `image_gen/tiny_1.py`'s own convention) blew HBM directly (confirmed: 165GB needed
  vs 30.75GB available on a v4 chip) — the dense `O(T*S)` cost scales quadratically with sequence
  length and is not optional to bound here. Fixed by setting each fuse-stage's cross-attention
  window to its own stride (the minimum valid value — window must be `>= stride` or most queries
  see zero code context), routing through the real `windowed_cross_attention` kernel instead.
- **Eager full-shard loads duplicate `/dev/shm`'s own RAM.** The first `dataloader.py` draft read
  each shard file fully into a Python `bytes` object at startup (`f.read()`) to build the entry
  index — since shards already live on `/dev/shm` (tmpfs, i.e. already RAM), this doubled memory
  usage for no reason (confirmed: 136GB RSS and climbing on a 400GB node). Fixed by `mmap`-ing each
  shard file instead — zero-copy view into the same pages, only a single index-building scan
  touches the bytes, no image data is copied out until an entry is actually requested.
- **`--num_workers` shouldn't exceed the split's file count.** `download_imagenet.py`'s
  `.shard(num_shards, index)` fails (`IndexError` in `datasets/utils/sharding.py`) for any worker
  index beyond the number of underlying parquet files — validation only has 14 files, so
  `--num_workers 32` there throws for workers 14-31. Harmless in practice (the workers that *do*
  get valid shards still succeed and write before the crash surfaces — confirmed the full 50,000
  validation images were still captured correctly), but cap `--num_workers` at the split's file
  count to avoid the error message.

## Status

Pipeline confirmed working end-to-end on real TPU (tpu5, v4-8, all 4 chips) as of 2026-08-29:
dataloader → model forward → loss/accuracy → `pmap` train step, no OOM, no errors, 160/160 smoke-test
steps completed. Not yet run for a real multi-epoch training session (only a short smoke test so
far) — eval cadence needs a saner override for short test runs (currently derived from a full real
epoch, so it never fired during the 160-step smoke test).

**Planned next**: a multi-scan/multi-ordering classifier variant — parallel forward passes over the
same image under different flattening orders (row-major reverse, column-major/vertical scan,
Hilbert/Z-order curve, random permutation), pooled and combined before the final linear head — not
yet implemented.
