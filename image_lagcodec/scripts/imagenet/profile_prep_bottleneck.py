"""Isolates network-fetch cost from CPU-resize cost for the offline prep pipeline, to determine
which one is the actual bottleneck before trying to speed it up.

    uv run python summformer_jax/image_gen/scripts/profile_prep_bottleneck.py --n 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
from prep_imagenet64 import center_crop_resize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    it = iter(ds)

    t0 = time.time()
    first = next(it)["image"]
    t_first = time.time() - t0
    print(f"first fetch (connection warmup): {t_first:.2f}s", flush=True)

    t0 = time.time()
    imgs = [first]
    for i in range(args.n - 1):
        imgs.append(next(it)["image"])
    t_fetch = time.time() - t0
    print(f"fetch-only (post-warmup): {args.n - 1} images in {t_fetch:.2f}s = {(args.n-1)/t_fetch:.1f} img/s", flush=True)

    t0 = time.time()
    for img in imgs:
        center_crop_resize(img, 64)
    t_resize = time.time() - t0
    print(f"resize-only (in-memory, no network): {args.n} images in {t_resize:.2f}s = {args.n/t_resize:.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
