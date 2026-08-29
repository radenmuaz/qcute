"""Streams ILSVRC/imagenet-1k from HF (no raw/Arrow caching -- streaming=True avoids the
tmpfs-fill failure mode hit downloading the full dataset via plain load_dataset), resizes each
image to 64x64 using the exact method from openai/improved-diffusion's image_datasets.py
(canonical ImageNet64 preprocessing used across the density-estimation literature, incl. the
Fractal Generative Models paper's own baselines):

  1. Repeatedly PIL.Image.BOX-downsample by 2x while min(size) >= 2*resolution.
  2. BICUBIC-resize so min(size) == resolution.
  3. Center-crop to resolution x resolution.

Writes flat raster-order uint8 RGB bytes (resolution*resolution*3 per image) into fixed-size
np.memmap shards -- same shard-per-file streaming pattern as scripts/jax/generate_pathfinder.py,
chosen for the same reason: never hold a whole split in RAM.

Non-RGB images (CMYK/L/RGBA) are resized in their native mode then converted to RGB as the last
step, exactly matching the reference's order (see center_crop_resize's docstring). Images that
fail to decode are skipped and counted, not silently dropped without a trace.

    uv run python summformer_jax/image_gen/scripts/prep_imagenet64_stream.py --split train --out_dir /dev/shm/imagenet64
    uv run python summformer_jax/image_gen/scripts/prep_imagenet64_stream.py --split validation --out_dir /dev/shm/imagenet64
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def center_crop_resize(pil_image: Image.Image, resolution: int) -> np.ndarray:
    """1:1 port of openai/improved-diffusion's ImageDataset.__getitem__ resize logic
    (raw.githubusercontent.com/openai/improved-diffusion/main/improved_diffusion/image_datasets.py,
    lines 75-96) -- resizing happens in the image's native color mode, RGB conversion is the
    LAST step before np.array, exactly matching the reference's order (not converted upfront)."""
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    return arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="/dev/shm/imagenet64")
    p.add_argument("--shard_size", type=int, default=50_000, help="images per shard")
    p.add_argument("--limit", type=int, default=None, help="debug: stop after N images")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bytes_per_image = args.resolution * args.resolution * 3

    ds = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=True)

    shard_idx = 0
    n_in_shard = 0
    n_written = 0
    n_failed = 0
    mmap = None

    def new_shard():
        nonlocal mmap, n_in_shard
        path = os.path.join(
            args.out_dir, f"imagenet64_{args.split}_{shard_idx:05d}.npy"
        )
        mmap = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.uint8, shape=(args.shard_size, bytes_per_image)
        )
        n_in_shard = 0
        return path

    path = new_shard()
    pbar = tqdm(ds, desc=f"imagenet64 {args.split}")
    for i, example in enumerate(pbar):
        if args.limit is not None and n_written >= args.limit:
            break
        try:
            arr = center_crop_resize(example["image"], args.resolution)
            assert arr.shape == (args.resolution, args.resolution, 3)
        except Exception as e:
            n_failed += 1
            pbar.set_postfix(failed=n_failed)
            continue

        mmap[n_in_shard] = arr.reshape(-1)
        n_in_shard += 1
        n_written += 1

        if n_in_shard == args.shard_size:
            mmap.flush()
            shard_idx += 1
            path = new_shard()

    if n_in_shard > 0:
        # truncate the final partial shard down to what was actually written
        mmap.flush()
        del mmap
        full = np.load(path, mmap_mode="r")
        trimmed = np.array(full[:n_in_shard])
        del full
        np.save(path, trimmed)
    else:
        os.remove(path)

    print(f"done: {n_written} images written, {n_failed} failed/skipped, "
          f"{shard_idx + (1 if n_in_shard > 0 else 0)} shards, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
