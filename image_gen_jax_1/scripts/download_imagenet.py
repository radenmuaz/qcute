"""Streams ILSVRC/imagenet-1k (streaming=True -- no raw-parquet caching, no HF "Generating split"
Arrow-materialization step) and persists each image's ORIGINAL ENCODED BYTES as-is (no decode, no
resize) into shard files. Replaces an earlier non-streaming version of this script (streaming=False)
that filled /dev/shm on tpu5 to 98%: that path downloads ~166GB of raw parquet AND THEN builds a
second, separate Arrow cache on top of it (HF's own on-disk dataset materialization step) -- this
version avoids both, doing its own lightweight I/O instead. Decouples the slow network pass from
resize experimentation: run this once, then prep_imagenet64.py's resize logic can be pointed at
the persisted shards and re-run as many times as needed with zero network I/O.

Shard format: `.bin` files, each a sequence of [4-byte big-endian label][8-byte big-endian
length][raw encoded image bytes] entries -- packs many images into few files (unlike
one-file-per-image, which would be 1.28M tiny files) while staying trivially appendable/streamable,
no numpy/PIL shape constraints since nothing is decoded here. Label is the dataset's own int class
index (0-999 for train/validation; both splits have real labels, unlike ImageNet-1k's unlabeled
"test" split which this script doesn't touch).

Uses `Image(decode=False)` on the dataset's image column so the underlying bytes are read
straight off the network without ever constructing a PIL Image -- cheaper than the resize
scripts (no decode work at all), and immune to resize-logic bugs since nothing is transformed.

--num_workers>1 runs that many processes in parallel, each streaming a DISJOINT `.shard(num_workers,
worker_id)` slice (same mechanism as prep_imagenet64_parallel.py) and writing its own
worker-prefixed shard files -- no collisions, network I/O (the actual bottleneck here, no
decode/resize work) parallelizes roughly linearly with worker count.

    uv run python summformer_jax/image_gen/scripts/download_imagenet.py --split train --out_dir /dev/shm/imagenet_raw --num_workers 8
    uv run python summformer_jax/image_gen/scripts/download_imagenet.py --split validation --out_dir /dev/shm/imagenet_raw --num_workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os


def _run(worker_id: int, num_workers: int, split: str, out_dir: str, shard_images: int,
         limit: int | None, hf_token: str | None):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

    from datasets import load_dataset, Image
    from tqdm import tqdm

    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    ds = ds.cast_column("image", Image(decode=False))  # raw bytes, no PIL decode
    if num_workers > 1:
        ds = ds.shard(num_shards=num_workers, index=worker_id)

    shard_idx = 0
    n_in_shard = 0
    n_written = 0
    f = None

    def new_shard():
        nonlocal f, n_in_shard
        if f is not None:
            f.close()
        path = os.path.join(out_dir, f"imagenet_raw_{split}_w{worker_id:02d}_{shard_idx:05d}.bin")
        f = open(path, "wb")
        n_in_shard = 0
        return path

    new_shard()
    pbar = tqdm(ds, desc=f"download {split} w{worker_id}", position=worker_id)
    for example in pbar:
        if limit is not None and n_written >= limit:
            break
        raw = example["image"]["bytes"]
        label = example["label"]
        f.write(int(label).to_bytes(4, "big", signed=True))
        f.write(len(raw).to_bytes(8, "big"))
        f.write(raw)
        n_in_shard += 1
        n_written += 1

        if n_in_shard == shard_images:
            shard_idx += 1
            new_shard()

    if f is not None:
        f.close()
    if n_in_shard == 0 and shard_idx > 0:
        # last shard ended up empty (n_written was an exact multiple of shard_images)
        os.remove(os.path.join(out_dir, f"imagenet_raw_{split}_w{worker_id:02d}_{shard_idx:05d}.bin"))

    return n_written, shard_idx + (1 if n_in_shard > 0 else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--out_dir", type=str, default="/dev/shm/imagenet_raw")
    p.add_argument("--shard_images", type=int, default=20_000, help="images per shard file")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="debug: stop after ~N images PER WORKER")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    if args.num_workers == 1:
        n_written, n_shards = _run(0, 1, args.split, args.out_dir, args.shard_images, args.limit, hf_token)
        print(f"done: {n_written} images persisted (raw bytes, no resize), {n_shards} shards, out_dir={args.out_dir}")
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.num_workers) as pool:
        results = pool.starmap(_run, [
            (i, args.num_workers, args.split, args.out_dir, args.shard_images, args.limit, hf_token)
            for i in range(args.num_workers)
        ])
    n_written = sum(r[0] for r in results)
    n_shards = sum(r[1] for r in results)
    print(f"done: {n_written} images persisted across {args.num_workers} workers "
          f"(raw bytes, no resize), {n_shards} shards, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
