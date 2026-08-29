"""Tiny non-streaming download of ILSVRC/imagenet-1k (train + validation) into the HF cache --
just the download step, no resize/shard processing (see prep_imagenet64.py for that).

    uv run python summformer_jax/image_gen/scripts/download_imagenet.py
"""
import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

from datasets import load_dataset

for split in ("train", "validation"):
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=False)
    print(f"{split}: {len(ds)} examples")
