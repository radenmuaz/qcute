#!/usr/bin/env python3
"""Benchmark ImageNet64 batch indexing using the loader path from _pretrain."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_imagenet64(data_root: Path, split: str, resolution: int = 64):
    """Copy of _pretrain's raw ImageNet mmap-load/concatenate path for one split."""
    shards = sorted(data_root.glob(f"imagenet{resolution}_{split}_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no imagenet{resolution}_{split}_*.npy shards under {data_root}")
    label_paths = [data_root / "labels" / shard.name for shard in shards]
    missing = [str(path) for path in label_paths if not path.is_file()]
    image_parts = [np.load(shard, mmap_mode="r") for shard in shards]
    # _pretrain concatenates the mmap shards, then reshapes the result for BatchIterator.
    images = np.concatenate(image_parts, axis=0).reshape(-1, resolution, resolution, 3)
    if not missing:
        labels = np.concatenate([np.load(path, mmap_mode="r") for path in label_paths]).astype(np.int32)
    else:
        labels = np.zeros(len(images), dtype=np.int32)
    if len(labels) != len(images):
        raise ValueError(f"{split} image/label count mismatch: {len(images)} images, {len(labels)} labels")
    return images, labels, len(shards)


def run_random_indices(images: np.ndarray, indices: np.ndarray, batch_size: int) -> float:
    start = time.perf_counter()
    with tqdm(total=len(indices), desc="random-index batches", unit="image") as progress:
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset:offset + batch_size]
            _ = images[batch_indices]  # mirrors BatchIterator: images[sel]
            progress.update(len(batch_indices))
    return time.perf_counter() - start


def run_random_slices(images: np.ndarray, num_images: int, batch_size: int, rng) -> float:
    batch_size = min(batch_size, len(images))
    starts = rng.integers(0, max(1, len(images) - batch_size + 1),
                          size=(num_images + batch_size - 1) // batch_size)
    done = 0
    start = time.perf_counter()
    with tqdm(total=num_images, desc="random contiguous slices", unit="image") as progress:
        for slice_start in starts:
            count = min(batch_size, num_images - done)
            # A bare ndarray slice is only a view. Copy it so this measures reading and
            # materializing the image batch, comparable to advanced-index batch selection.
            _ = images[int(slice_start):int(slice_start) + count].copy()
            done += count
            progress.update(count)
    return time.perf_counter() - start


def report(label: str, elapsed: float, count: int, resolution: int):
    rate = count / elapsed if elapsed else float("inf")
    decoded_mib = count * resolution * resolution * 3 / (1024 ** 2)
    print(f"{label}: {count} images in {elapsed:.2f}s, {rate:.2f} images/s, "
          f"{decoded_mib / elapsed:.1f} MiB/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/dev/shm/imagenet64"))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--images", type=int, default=1000,
                        help="number of images read per benchmark mode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "random_indices", "random_slices"), default="both")
    args = parser.parse_args()
    if args.images < 1 or args.batch_size < 1:
        parser.error("--images and --batch-size must be positive")

    load_start = time.perf_counter()
    images, labels, shard_count = load_imagenet64(args.data_root, args.split)
    load_elapsed = time.perf_counter() - load_start
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(images), size=args.images)
    print(f"dataset={args.data_root} split={args.split} images={len(images)} "
          f"resolution=64 shards={shard_count} load_and_concat={load_elapsed:.2f}s "
          f"array_backing={type(images).__name__}")
    print("_pretrain opens .npy shards with mmap_mode='r' then concatenates them into a RAM array. "
          "These modes therefore benchmark post-load array indexing/copying, not persistent mmap access.")

    if args.mode in ("both", "random_indices"):
        elapsed = run_random_indices(images, indices, args.batch_size)
        report("random indices", elapsed, args.images, 64)
    if args.mode in ("both", "random_slices"):
        elapsed = run_random_slices(images, args.images, args.batch_size, rng)
        report("random slices", elapsed, args.images, 64)


if __name__ == "__main__":
    main()
