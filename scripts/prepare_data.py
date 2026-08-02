"""Download enwik8 and cut a tiny gzip subset for quick local runs.

    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --tiny_bytes 200_000 --force

Produces:
  datasets/enwik8.gz       full ~35MB corpus (skipped if already present)
  datasets/enwik8_tiny.gz  a gzip'd prefix of --tiny_bytes raw bytes, for fast
                            train+val smoke runs (qcute.bytelm / qcute.qcutelm
                            both default to the full file via --data; pass
                            --data datasets/enwik8_tiny.gz to use the subset)
"""
from __future__ import annotations

import argparse
import gzip
import urllib.request
from pathlib import Path

ENWIK8_URL = (
    "https://github.com/lucidrains/memory-transformer-xl/raw/master/"
    "examples/enwik8_simple/data/enwik8.gz"
)


def download_full(dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"{dest} already exists, skipping download (--force to re-download)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ENWIK8_URL} -> {dest}")
    urllib.request.urlretrieve(ENWIK8_URL, dest)
    print(f"done: {dest.stat().st_size} bytes")


def cut_tiny(src: Path, dest: Path, n_bytes: int, force: bool) -> None:
    if dest.exists() and not force:
        print(f"{dest} already exists, skipping cut (--force to overwrite)")
        return
    with gzip.open(src, "rb") as f:
        data = f.read(n_bytes)
    with gzip.open(dest, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest_dir", type=Path, default=Path("datasets"))
    p.add_argument("--tiny_bytes", type=int, default=500_000, help="raw bytes to cut into the tiny subset")
    p.add_argument("--force", action="store_true", help="re-download / re-cut even if files exist")
    args = p.parse_args()

    full = args.dest_dir / "enwik8.gz"
    tiny = args.dest_dir / "enwik8_tiny.gz"
    download_full(full, args.force)
    cut_tiny(full, tiny, args.tiny_bytes, args.force)


if __name__ == "__main__":
    main()
