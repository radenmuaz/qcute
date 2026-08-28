"""Download ILSVRC/imagenet-1k via HF `datasets`. Requires a gated-dataset-approved HF_TOKEN
(read from .env in this repo's root, or already exported in the environment). Defaults the HF
cache to /dev/shm (tmpfs) rather than persistent disk -- see CLAUDE.md's TPU-node data-prep
convention.

    uv run python scripts/download_imagenet.py
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

if "HF_TOKEN" not in os.environ:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                break

from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("ILSVRC/imagenet-1k")
    print(ds)
