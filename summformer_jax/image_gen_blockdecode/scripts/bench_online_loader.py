"""Throughput bench for OnlineImageByteLoader -- steady-state img/s after warmup, since the first
HF streaming connection alone can take ~60s (confirmed earlier this session), which would
dominate/distort a short sample.

    uv run python summformer_jax/image_gen/scripts/bench_online_loader.py --n-images 2000 --num-workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imagenet_dataloader import OnlineImageByteLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-images", type=int, default=100)
    args = p.parse_args()

    loader = OnlineImageByteLoader(batch_size=args.batch_size, resolution=64, split="validation",
                                    num_workers=args.num_workers, queue_size=256)
    n = 0
    t_warmup_start = time.time()
    while n < args.warmup_images:
        loader.next_batch()
        n += args.batch_size
    t_warmup = time.time() - t_warmup_start
    print(f"warmup: {n} images in {t_warmup:.1f}s ({n/t_warmup:.1f} img/s, includes connection setup)", flush=True)

    n2 = 0
    t0 = time.time()
    while n2 < args.n_images:
        loader.next_batch()
        n2 += args.batch_size
    dt = time.time() - t0
    print(f"steady-state: {n2} images in {dt:.1f}s = {n2/dt:.1f} img/s, num_workers={args.num_workers}", flush=True)
    loader.close()


if __name__ == "__main__":
    main()
