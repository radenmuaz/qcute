"""Worker-parallel version of prep_imagenet64.py -- same streaming+resize+save-to-disk approach
(NOT a full non-streaming download, which doesn't fit tpu8's tmpfs -- see docs/image_gen_design.md
for the feasibility math: 166.8GB raw parquet alone leaves ~30GB headroom, well under the ~20GB
threshold that reliably triggers the D-state disk-hang bug documented in CLAUDE.md).

Each worker gets a DISJOINT `.shard(num_workers, worker_id)` slice of the split and writes its own
numbered output shards (prefixed by worker id, so no filename collisions) -- same
center_crop_resize (verified byte-identical to openai/improved-diffusion's reference, unchanged
here) as the single-process script.

Earlier multi-worker attempt (OnlineImageByteLoader, an in-training-loop streaming loader) was
SLOWER than single-process (153 img/s vs ~230-250 img/s) -- traced to `.shard()`'s apparent
redundant traversal of the underlying parquet shard files per worker. This script uses the exact
same `.shard()` mechanism, so it may hit the same problem -- BENCHMARK ON A SHORT BOUNDED RUN
FIRST (--limit) before committing to the full split; don't assume workers help here without
checking.

    uv run python summformer_jax/image_gen/scripts/prep_imagenet64_parallel.py --split train --out_dir /dev/shm/imagenet64 --num_workers 4 --limit 2000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache" if os.path.isdir("/dev/shm") else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

import numpy as np
from tqdm import tqdm

from prep_imagenet64 import center_crop_resize


def _worker(worker_id: int, num_workers: int, split: str, resolution: int, out_dir: str,
            shard_size: int, limit: int | None, progress_q: mp.Queue, hf_token: str | None):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    from datasets import load_dataset

    bytes_per_image = resolution * resolution * 3
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    ds = ds.shard(num_shards=num_workers, index=worker_id)

    shard_idx = 0
    n_in_shard = 0
    n_written = 0
    n_failed = 0
    mmap = None

    def new_shard():
        nonlocal mmap, n_in_shard
        path = os.path.join(out_dir, f"imagenet64_{split}_w{worker_id:02d}_{shard_idx:05d}.npy")
        mmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(shard_size, bytes_per_image))
        n_in_shard = 0
        return path

    path = new_shard()
    for example in ds:
        if limit is not None and n_written >= limit:
            break
        try:
            arr = center_crop_resize(example["image"], resolution)
            assert arr.shape == (resolution, resolution, 3)
        except Exception:
            n_failed += 1
            continue
        mmap[n_in_shard] = arr.reshape(-1)
        n_in_shard += 1
        n_written += 1
        if n_written % 50 == 0:
            progress_q.put(1 * 50)
        if n_in_shard == shard_size:
            mmap.flush()
            shard_idx += 1
            path = new_shard()

    if n_in_shard > 0:
        mmap.flush()
        del mmap
        full = np.load(path, mmap_mode="r")
        trimmed = np.array(full[:n_in_shard])
        del full
        np.save(path, trimmed)
    else:
        os.remove(path)
    progress_q.put(("done", worker_id, n_written, n_failed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="/dev/shm/imagenet64")
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="per-worker image cap, for benchmarking before a full run")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()

    workers = [
        ctx.Process(target=_worker, args=(i, args.num_workers, args.split, args.resolution,
                                           args.out_dir, args.shard_size, args.limit, progress_q, hf_token))
        for i in range(args.num_workers)
    ]
    t0 = time.time()
    for w in workers:
        w.start()

    n_done_workers = 0
    total = 0
    results = []
    pbar = tqdm(desc=f"imagenet64 {args.split} ({args.num_workers} workers)")
    while n_done_workers < args.num_workers:
        item = progress_q.get()
        if isinstance(item, tuple):
            results.append(item)
            n_done_workers += 1
        else:
            total += item
            pbar.update(item)
    pbar.close()

    dt = time.time() - t0
    n_written = sum(r[2] for r in results)
    n_failed = sum(r[3] for r in results)
    print(f"done: {n_written} images written, {n_failed} failed, {dt:.1f}s = {n_written/dt:.1f} img/s "
          f"({args.num_workers} workers)")
    for w in workers:
        w.join(timeout=5)


if __name__ == "__main__":
    main()
