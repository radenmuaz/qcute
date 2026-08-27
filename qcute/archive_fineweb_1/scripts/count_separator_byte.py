"""Count occurrences of a candidate separator byte across FineWeb-Edu's sample-10BT text.

    uv run python scripts/count_separator_byte.py

Validates the assumption behind prep_fineweb_edu_bytes.py's --separator default (0x00): that it
does not occur naturally in this corpus, so it's an unambiguous document-boundary marker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src_dir", type=Path, default=Path("datasets/fineweb_edu_10BT/sample/10BT"))
    p.add_argument("--byte_value", type=lambda s: int(s, 0), default=0x00)
    args = p.parse_args()

    needle = bytes([args.byte_value])
    paths = sorted(args.src_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no .parquet files found under {args.src_dir}")

    total_count, total_bytes, total_docs = 0, 0, 0
    for path in tqdm(paths, desc="scanning shards"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"]):
            for text in batch.column("text"):
                b = str(text).encode("utf-8")
                total_count += b.count(needle)
                total_bytes += len(b)
                total_docs += 1

    print(f"byte 0x{args.byte_value:02x}: {total_count} occurrences across {total_docs} docs, "
          f"{total_bytes} total bytes ({total_count / max(total_bytes, 1) * 1e9:.2f} per billion bytes)")


if __name__ == "__main__":
    main()
