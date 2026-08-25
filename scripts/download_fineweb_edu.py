"""Download the FineWeb-Edu 10B-token sample subset from the Hub.

    uv run python scripts/download_fineweb_edu.py

Reads HF_TOKEN from the repo's .env file and downloads the `sample/10BT/`
parquet shards of HuggingFaceFW/fineweb-edu into datasets/fineweb_edu_10BT/.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = ["sample/10BT/*"]


def load_hf_token(env_path: Path) -> str:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"HF_TOKEN not found in {env_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest_dir", type=Path, default=Path("datasets/fineweb_edu_10BT"))
    p.add_argument("--env_path", type=Path, default=Path(".env"))
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or load_hf_token(args.env_path)
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {REPO_ID} ({ALLOW_PATTERNS}) -> {args.dest_dir}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=args.dest_dir,
        token=token,
    )
    print(f"done: {path}")


if __name__ == "__main__":
    main()
