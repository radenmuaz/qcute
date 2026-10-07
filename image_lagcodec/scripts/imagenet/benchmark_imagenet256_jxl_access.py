#!/usr/bin/env python3
"""Benchmark random integer and contiguous-slice access to the ImageNet JXL shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


class ImageNetJXLByteDataset:
    """Array-like random access to concatenated JPEG XL byte shards.

    Payload and offset files stay memory-mapped; requested images are decoded on access.
    """

    def __init__(self, data_root: Path, resolution: int, shard_records: list):
        self.data_root = Path(data_root)
        self.resolution = int(resolution)
        self.shards = []
        counts = []
        for record in shard_records:
            payload_path = self.data_root / record["payload"]
            offsets_path = self.data_root / record["offsets"]
            payload_size = payload_path.stat().st_size
            offsets = np.load(offsets_path, mmap_mode="r")
            if offsets.dtype != np.uint64 or offsets.ndim != 1 or len(offsets) != int(record["count"]) + 1:
                raise ValueError(f"invalid JXL offsets metadata: {offsets_path}")
            if int(offsets[0]) != 0 or int(offsets[-1]) != payload_size:
                raise ValueError(f"offsets do not span payload {payload_path}")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"JXL offsets are not monotonic: {offsets_path}")
            payload = np.memmap(payload_path, mode="r", dtype=np.uint8)
            labels_path = self.data_root / record["labels"]
            labels = np.load(labels_path, mmap_mode="r")
            if labels.ndim != 1 or len(labels) != int(record["count"]):
                raise ValueError(f"label count does not match JXL records: {labels_path}")
            self.shards.append((payload, offsets, labels))
            counts.append(int(record["count"]))
        self.counts = np.asarray(counts, dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self._length = int(self.ends[-1]) if len(self.ends) else 0
        self.labels = np.concatenate([np.asarray(shard[2]) for shard in self.shards]).astype(np.int32) \
            if self.shards else np.empty((0,), dtype=np.int32)

    def __len__(self):
        return self._length

    @property
    def shape(self):
        return (len(self), self.resolution, self.resolution, 3)

    def _decode_one(self, index: int) -> np.ndarray:
        import imagecodecs

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = int(np.searchsorted(self.ends, index, side="right"))
        shard_start = 0 if shard_idx == 0 else int(self.ends[shard_idx - 1])
        local_idx = index - shard_start
        payload, offsets, _ = self.shards[shard_idx]
        start, end = int(offsets[local_idx]), int(offsets[local_idx + 1])
        decoded = np.asarray(imagecodecs.jpegxl_decode(payload[start:end]))
        expected = (self.resolution, self.resolution, 3)
        if decoded.shape != expected or decoded.dtype != np.uint8:
            raise ValueError(f"decoded JXL image {index} has {decoded.shape}/{decoded.dtype}, expected "
                             f"{expected}/uint8")
        return decoded

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._decode_one(int(index))
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
        else:
            indices = np.asarray(index)
            if indices.dtype == np.bool_:
                if indices.ndim != 1 or len(indices) != len(self):
                    raise IndexError("boolean index must match dataset length")
                indices = np.flatnonzero(indices)
            elif indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise IndexError("JXL dataset indices must be integers, slices, or 1D integer arrays")
        images = [self._decode_one(int(i)) for i in indices]
        return np.stack(images, axis=0) if images else np.empty((0, *self.shape[1:]), dtype=np.uint8)


def load_dataset(root: Path, resolution: int, split: str) -> ImageNetJXLByteDataset:
    manifest_path = root / f"imagenet{resolution}_{split}_manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    if manifest.get("format") != "jpeg-xl-lossless-concatenated":
        raise ValueError(f"unsupported JXL format in {manifest_path}: {manifest.get('format')}")
    if manifest.get("resolution") != resolution or manifest.get("split") != split:
        raise ValueError(f"manifest resolution/split mismatch: {manifest_path}")
    dataset = ImageNetJXLByteDataset(root, resolution, manifest["shards"])
    if not len(dataset):
        raise ValueError(f"empty JXL dataset: {manifest_path}")
    return dataset


def run_random_ints(dataset, indices: np.ndarray, batch_size: int) -> tuple[float, int]:
    start = time.perf_counter()
    with tqdm(total=len(indices), desc="random integer batches", unit="image") as progress:
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset:offset + batch_size]
            _ = dataset[batch_indices]
            progress.update(len(batch_indices))
    return time.perf_counter() - start, len(indices)


def run_random_slices(dataset, num_images: int, batch_size: int, rng) -> tuple[float, int]:
    batch_size = min(batch_size, len(dataset))
    starts = rng.integers(0, max(1, len(dataset) - batch_size + 1),
                          size=(num_images + batch_size - 1) // batch_size)
    done = 0
    start = time.perf_counter()
    with tqdm(total=num_images, desc="random contiguous slices", unit="image") as progress:
        for slice_start in starts:
            count = min(batch_size, num_images - done)
            _ = dataset[int(slice_start):int(slice_start) + count]
            done += count
            progress.update(count)
    return time.perf_counter() - start, done


def report(name: str, elapsed: float, count: int, resolution: int):
    rate = count / elapsed if elapsed else float("inf")
    mib = count * resolution * resolution * 3 / (1024 ** 2)
    print(f"{name}: {count} images in {elapsed:.2f}s, {rate:.2f} images/s, "
          f"{mib / elapsed:.1f} decoded MiB/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/dev/shm/imagenet256_jxl"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--images", type=int, default=1000,
                        help="number of images decoded per benchmark mode")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="images decoded per random-index batch or contiguous slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "random_ints", "random_slices"), default="both")
    args = parser.parse_args()
    if args.images < 1 or args.batch_size < 1:
        parser.error("--images and --batch-size must be positive")

    dataset = load_dataset(args.data_root, args.resolution, args.split)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(dataset), size=args.images)
    print(f"dataset={args.data_root} split={args.split} images={len(dataset)} "
          f"resolution={args.resolution} benchmark_images={args.images}")
    print("Each mode decodes the same number of images. Separate runs avoid one mode warming "
          "the OS file cache for the next.")

    if args.mode in ("both", "random_ints"):
        elapsed, count = run_random_ints(dataset, indices, args.batch_size)
        report("random integers", elapsed, count, args.resolution)
    if args.mode in ("both", "random_slices"):
        elapsed, count = run_random_slices(dataset, args.images, args.batch_size, rng)
        report("random slices", elapsed, count, args.resolution)


if __name__ == "__main__":
    main()
