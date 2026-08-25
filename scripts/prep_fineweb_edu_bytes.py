"""Byte-encode FineWeb-Edu's sample-10BT parquet shards into flat mmap-able binaries.

    uv run python scripts/prep_fineweb_edu_bytes.py

No tokenizer: each document's `text` column is UTF-8-encoded straight to bytes and concatenated
(no separator by default -- matches qcute.bytelm_tpu's enwik8 loader, which also has no document
boundaries). Pass --use_separator to insert --separator between documents instead (default
"\\x00", chosen because it does not occur naturally in this corpus -- see
scripts/count_separator_byte.py).

Reads shards via pyarrow (streaming per row-group, never materializing a whole shard's decoded
text as one Python list), writes two pre-allocated np.memmap uint8 files so training can mmap
random windows straight off disk instead of loading the whole ~27GB corpus into RAM:

  datasets/fineweb_edu_10BT/train.bin
  datasets/fineweb_edu_10BT/val.bin   (held out: the last shard)
"""
from __future__ import annotations

import argparse
import codecs
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def shard_paths(src_dir: Path, limit_shards: int | None) -> list[Path]:
    paths = sorted(src_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no .parquet files found under {src_dir}")
    return paths[:limit_shards] if limit_shards else paths


def encode_shard(args: tuple[Path, bool, bytes]) -> tuple[Path, bytes]:
    """Runs in a worker process: read one shard's text column, UTF-8-encode + concatenate
    (optionally separator-joined), return the whole shard's byte blob. Row-group streaming keeps
    peak memory to one row-group's decoded text at a time, not the whole shard."""
    path, use_separator, separator = args
    pf = pq.ParquetFile(path)
    chunks: list[bytes] = []
    for batch in pf.iter_batches(columns=["text"]):
        for text in batch.column("text"):
            chunks.append(str(text).encode("utf-8"))
    blob = separator.join(chunks) if use_separator else b"".join(chunks)
    return path, blob


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", type=Path, default=Path("datasets/fineweb_edu_10BT/sample/10BT"))
    p.add_argument("--dest_dir", type=Path, default=Path("datasets/fineweb_edu_10BT"))
    p.add_argument("--limit_shards", type=int, default=None, help="only process the first N shards (for a quick smoke test)")
    p.add_argument("--val_shards", type=int, default=1, help="trailing N shards held out as val")
    p.add_argument("--use_separator", action="store_true", help="insert --separator between documents (default: plain concatenation, no separator)")
    p.add_argument("--separator", type=str, default="\\x00", help="Python-escape-decoded separator string, e.g. \\x00 or \\n\\n (only used with --use_separator)")
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    train_path = args.dest_dir / "train.bin"
    val_path = args.dest_dir / "val.bin"
    if train_path.exists() and val_path.exists() and not args.force:
        print(f"{train_path} and {val_path} already exist, skipping (--force to overwrite)")
        return

    separator = codecs.decode(args.separator, "unicode_escape").encode("latin1")
    paths = shard_paths(args.src_dir, args.limit_shards)
    val_paths, train_paths = paths[-args.val_shards :], paths[: -args.val_shards]
    print(f"{len(train_paths)} train shards, {len(val_paths)} val shards, "
          f"use_separator={args.use_separator} separator={separator!r}")

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_paths, out_path in [("train", train_paths, train_path), ("val", val_paths, val_path)]:
        jobs = [(sp, args.use_separator, separator) for sp in split_paths]
        blobs: dict[Path, bytes] = {}
        if jobs:
            with Pool(min(args.n_workers, len(jobs))) as pool:
                for path, blob in tqdm(pool.imap(encode_shard, jobs), total=len(jobs), desc=f"encoding {split_name}"):
                    blobs[path] = blob
        total_len = sum(len(b) for b in blobs.values())
        mm = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(total_len,))
        offset = 0
        for sp in split_paths:  # write back in original shard order
            blob = blobs[sp]
            mm[offset : offset + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
            offset += len(blob)
        mm.flush()
        print(f"wrote {total_len} bytes -> {out_path}")


if __name__ == "__main__":
    main()
