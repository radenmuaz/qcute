"""Download enwik8 and cut the standard 1M-byte gzip subset for local runs.

    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --subset_bytes 200_000 --force

Produces:
  datasets/enwik8.gz     full ~35MB corpus (skipped if already present)
  datasets/enwik8_1M.gz  a gzip'd prefix of --subset_bytes raw bytes (default
                          1,000,000). This is the standard corpus: all three
                          modules (qcute.bytelm / qcute.bpelm / qcute.qcutelm)
                          default --data to this file, so no --n_bytes cutoff
                          is needed for normal runs.
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


def cut_subset(src: Path, dest: Path, n_bytes: int, force: bool) -> None:
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
    p.add_argument("--subset_bytes", type=int, default=1_000_000, help="raw bytes to cut into the standard subset")
    p.add_argument("--force", action="store_true", help="re-download / re-cut even if files exist")
    args = p.parse_args()

    full = args.dest_dir / "enwik8.gz"
    subset = args.dest_dir / "enwik8_1M.gz"
    download_full(full, args.force)
    cut_subset(full, subset, args.subset_bytes, args.force)


if __name__ == "__main__":
    main()
