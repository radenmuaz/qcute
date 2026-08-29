"""Experimental alternative to download_imagenet.py: reads ILSVRC/imagenet-1k via HF's Croissant
(mlcroissant) metadata API instead of `datasets.load_dataset(streaming=True)`. Persists each
image's original encoded bytes as-is (no decode, no resize) into the same shard format as
download_imagenet.py -- [8-byte length][raw bytes] entries per shard file -- so
prep_imagenet64.py's resize pass can consume either script's output identically.

Croissant records don't cleanly separate "train" vs "validation" by a single split argument the
way `datasets` does -- this script assumes the `record_set` name IS the split name ("default" is
Croissant's generic top-level record set, per HF's own generated JSON-LD for this dataset; if that
doesn't hold, --record-set lets you point at whatever record set name the dataset's actual
Croissant metadata exposes).

    uv run python summformer_jax/image_gen/scripts/download_imagenet_croissant.py --out_dir /dev/shm/imagenet_raw
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

import requests
from mlcroissant import Dataset
from tqdm import tqdm

CROISSANT_URL = "https://huggingface.co/api/datasets/ILSVRC/imagenet-1k/croissant"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record-set", type=str, default="default")
    p.add_argument("--out_dir", type=str, default="/dev/shm/imagenet_raw")
    p.add_argument("--shard_images", type=int, default=20_000, help="images per shard file")
    p.add_argument("--limit", type=int, default=None, help="debug: stop after N images")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    jsonld = requests.get(CROISSANT_URL, headers=headers).json()  # gated dataset -- needs HF_TOKEN
    ds = Dataset(jsonld=jsonld)
    records = ds.records(args.record_set)

    shard_idx = 0
    n_in_shard = 0
    n_written = 0
    n_failed = 0
    f = None

    def new_shard():
        nonlocal f, n_in_shard
        if f is not None:
            f.close()
        path = os.path.join(args.out_dir, f"imagenet_raw_croissant_{shard_idx:05d}.bin")
        f = open(path, "wb")
        n_in_shard = 0
        return path

    new_shard()
    pbar = tqdm(records, desc=f"croissant record_set={args.record_set}")
    for record in pbar:
        if args.limit is not None and n_written >= args.limit:
            break
        raw = None
        for key, value in record.items():
            if isinstance(value, (bytes, bytearray)):
                raw = bytes(value)
                break
        if raw is None:
            n_failed += 1
            pbar.set_postfix(failed=n_failed)
            continue

        f.write(len(raw).to_bytes(8, "big"))
        f.write(raw)
        n_in_shard += 1
        n_written += 1

        if n_in_shard == args.shard_images:
            shard_idx += 1
            new_shard()

    if f is not None:
        f.close()
    if n_in_shard == 0 and shard_idx > 0:
        os.remove(os.path.join(args.out_dir, f"imagenet_raw_croissant_{shard_idx:05d}.bin"))

    print(f"done: {n_written} images persisted, {n_failed} failed/skipped (no byte field found), "
          f"{shard_idx + (1 if n_in_shard > 0 else 0)} shards, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
