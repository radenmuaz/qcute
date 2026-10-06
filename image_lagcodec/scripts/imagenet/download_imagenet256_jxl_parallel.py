"""Parallel ImageNet resize/JXL encoder writing compressed byte shards.

The compressed payload is a regular append-only file per shard, so it can be opened as a
read-only mmap. ``offsets/<shard>.npy`` contains uint64 byte boundaries of length N+1:
image i is ``payload[offsets[i]:offsets[i + 1]]``. Labels remain aligned in
``labels/<shard>.npy``. The manifest records the actual image count in each shard.

This keeps random access cheap while using much less space than fixed-size uint8 image
arrays. Decoding is needed when batches are read; keep the byte files and offset arrays on
disk/tmpfs and decode only the selected examples.

Resizing and JPEG XL encoding run in a bounded thread pool. Results are written in source
order, so image indices, labels, and offset metadata remain aligned. ``--workers`` controls
the number of concurrent CPU workers; at most twice that many images are queued.

Run with imagecodecs supplied to uv without adding it to every environment:

    uv run --with imagecodecs python image_lagcodec/scripts/imagenet/download_imagenet256_jxl_parallel.py \
      --split train --out_dir /dev/shm/imagenet256_jxl --workers 16
    uv run --with imagecodecs python image_lagcodec/scripts/imagenet/download_imagenet256_jxl_parallel.py \
      --split validation --out_dir /dev/shm/imagenet256_jxl --workers 16
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

import imagecodecs
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def center_crop_resize(pil_image: Image.Image, resolution: int) -> np.ndarray:
    """Match improved-diffusion preprocessing: resize in native mode, then convert RGB."""
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.asarray(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    return arr[crop_y:crop_y + resolution, crop_x:crop_x + resolution]


def encode_image(pil_image: Image.Image, resolution: int, effort: int):
    """Worker: resize and losslessly encode one source image."""
    try:
        arr = center_crop_resize(pil_image, resolution)
        if arr.shape != (resolution, resolution, 3):
            raise ValueError(f"unexpected resized image shape: {arr.shape}")
        encoded = imagecodecs.jpegxl_encode(arr, lossless=True, effort=effort)
        return bytes(encoded), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out_dir", type=Path, default=Path("/dev/shm/imagenet256_jxl"))
    parser.add_argument("--shard_size", type=int, default=10_000, help="images per shard")
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N images")
    parser.add_argument("--effort", type=int, default=7, help="JPEG XL effort, 1 (fast) to 9 (small)")
    parser.add_argument("--workers", type=int, default=16, help="concurrent resize/JXL encode workers")
    args = parser.parse_args()
    if args.resolution < 1 or args.shard_size < 1:
        parser.error("resolution and shard_size must be positive")
    if not 1 <= args.effort <= 9:
        parser.error("effort must be in [1, 9]")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive when set")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = args.out_dir / "labels"
    offsets_dir = args.out_dir / "offsets"
    labels_dir.mkdir(exist_ok=True)
    offsets_dir.mkdir(exist_ok=True)

    ds = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=True)
    print(f"dataset columns: {list(ds.features.keys())}")
    if "label" not in ds.features:
        raise ValueError(f"expected a 'label' column, found {list(ds.features.keys())}")

    shard_idx = 0
    n_in_shard = n_written = n_failed = 0
    offsets = [0]
    labels = []
    payload = None
    shard_records = []

    def open_shard():
        nonlocal payload, n_in_shard, offsets, labels
        stem = f"imagenet256_{args.split}_{shard_idx:05d}"
        path = args.out_dir / f"{stem}.jxlbytes"
        payload = open(path, "wb")
        n_in_shard = 0
        offsets = [0]
        labels = []
        return stem, path

    def finish_shard(stem, path):
        nonlocal payload
        payload.flush()
        os.fsync(payload.fileno())
        payload.close()
        np.save(offsets_dir / f"{stem}.npy", np.asarray(offsets, dtype=np.uint64))
        np.save(labels_dir / f"{stem}.npy", np.asarray(labels, dtype=np.int32))
        shard_records.append({
            "payload": path.name,
            "offsets": f"offsets/{stem}.npy",
            "labels": f"labels/{stem}.npy",
            "count": n_in_shard,
            "bytes": offsets[-1],
        })
        payload = None

    stem, path = open_shard()
    pbar = tqdm(desc=f"imagenet256-jxl {args.split}", unit="image")
    pending = deque()
    dataset_iter = iter(ds)
    source_rows = 0
    exhausted = False
    max_pending = 2 * args.workers
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="jxl-encode") as pool:
        while pending or not exhausted:
            while (not exhausted and len(pending) < max_pending
                   and (args.limit is None or n_written + len(pending) < args.limit)):
                try:
                    example = next(dataset_iter)
                except StopIteration:
                    exhausted = True
                    break
                row_idx = source_rows
                source_rows += 1
                future = pool.submit(encode_image, example["image"], args.resolution, args.effort)
                pending.append((future, int(example["label"]), row_idx))

            if not pending:
                # If pending failures exhausted the optimistic limit, fetch replacements until
                # the requested number of successfully encoded images is reached or input ends.
                if args.limit is not None and n_written >= args.limit:
                    break
                continue

            future, label, row_idx = pending.popleft()
            encoded, error = future.result()
            pbar.update(1)
            if error is not None:
                n_failed += 1
                pbar.set_postfix(failed=n_failed)
                pbar.write(f"skip source row {row_idx}: {error}")
                continue

            payload.write(encoded)
            offsets.append(offsets[-1] + len(encoded))
            labels.append(label)
            n_in_shard += 1
            n_written += 1
            if n_in_shard == args.shard_size:
                finish_shard(stem, path)
                shard_idx += 1
                stem, path = open_shard()
            if args.limit is not None and n_written >= args.limit:
                # Do not wait for or write speculative work past the limit.
                for queued, _, _ in pending:
                    queued.cancel()
                pending.clear()
                exhausted = True
        pbar.close()

    if n_in_shard:
        finish_shard(stem, path)
    else:
        if payload is not None:
            payload.close()
        path.unlink(missing_ok=True)

    manifest = {
        "format": "jpeg-xl-lossless-concatenated",
        "resolution": args.resolution,
        "channels": 3,
        "dtype": "uint8",
        "split": args.split,
        "images_written": n_written,
        "images_failed": n_failed,
        "effort": args.effort,
        "workers": args.workers,
        "offset_semantics": "uint64 byte boundaries; image i is payload[offsets[i]:offsets[i+1]]",
        "shards": shard_records,
    }
    manifest_path = args.out_dir / f"imagenet256_{args.split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total_bytes = sum(row["bytes"] for row in shard_records)
    print(f"done: {n_written} images written, {n_failed} failed/skipped, "
          f"{len(shard_records)} shards, payload={total_bytes / (1024 ** 3):.2f} GiB, "
          f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
