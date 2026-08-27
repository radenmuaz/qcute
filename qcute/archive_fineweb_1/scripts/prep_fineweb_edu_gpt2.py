"""GPT-2-BPE-encode FineWeb-Edu's sample-10BT parquet shards into flat mmap-able binaries.

    uv run python scripts/prep_fineweb_edu_gpt2.py

Same shard-streaming/held-out-val-shard/chunked-write design as prep_fineweb_edu_bytes.py (see
that script's docstring for the disk-hang lesson behind the chunked file.write() below), except
each document's `text` column is encoded via tiktoken's "gpt2" BPE (50257 tokens, includes GPT-2's
own `<|endoftext|>` id 50256 as a real document separator -- unlike the byte-level prep script,
there is no "no separator" option here: GPT-2's own tokenizer already reserves this id for exactly
this purpose) instead of raw UTF-8 bytes, and written as flat `uint16` (fits 50257) instead of
`uint8`.

  datasets/fineweb_edu_10BT_gpt2/train.bin
  datasets/fineweb_edu_10BT_gpt2/val.bin   (held out: the last shard)
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

_WRITE_CHUNK = 64 * 1024 * 1024  # bounded write() size -- see prep_fineweb_edu_bytes.py
_EOT = tiktoken.get_encoding("gpt2").eot_token  # 50256, document separator


def encode_shard(path: Path) -> tuple[Path, np.ndarray]:
    """Runs in a worker process: read one shard's text column, GPT-2-BPE-encode each document,
    append <|endoftext|> after each, concatenate. Row-group streaming keeps peak memory to one
    row-group's decoded text at a time."""
    enc = tiktoken.get_encoding("gpt2")
    pf = pq.ParquetFile(path)
    ids: list[int] = []
    for batch in pf.iter_batches(columns=["text"]):
        for text in batch.column("text"):
            ids.extend(enc.encode_ordinary(str(text)))
            ids.append(_EOT)
    return path, np.array(ids, dtype=np.uint16)


def shard_paths(src_dir: Path, limit_shards: int | None) -> list[Path]:
    paths = sorted(src_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no .parquet files found under {src_dir}")
    return paths[:limit_shards] if limit_shards else paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", type=Path, default=Path("datasets/fineweb_edu_10BT/sample/10BT"))
    p.add_argument("--dest_dir", type=Path, default=Path("datasets/fineweb_edu_10BT_gpt2"))
    p.add_argument("--limit_shards", type=int, default=None, help="only process the first N shards (for a quick smoke test)")
    p.add_argument("--val_shards", type=int, default=1, help="trailing N shards held out as val")
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--val_only", action="store_true", help="only (re)process the val split")
    args = p.parse_args()

    train_path = args.dest_dir / "train.bin"
    val_path = args.dest_dir / "val.bin"
    if train_path.exists() and val_path.exists() and not args.force:
        print(f"{train_path} and {val_path} already exist, skipping (--force to overwrite)")
        return

    paths = shard_paths(args.src_dir, args.limit_shards)
    val_paths, train_paths = paths[-args.val_shards :], paths[: -args.val_shards]
    print(f"{len(train_paths)} train shards, {len(val_paths)} val shards, eot_token={_EOT}")

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    splits = [("val", val_paths, val_path)] if args.val_only else [("train", train_paths, train_path), ("val", val_paths, val_path)]
    for split_name, split_paths, out_path in splits:
        arrs: dict[Path, np.ndarray] = {}
        if split_paths:
            with Pool(min(args.n_workers, len(split_paths))) as pool:
                for path, arr in tqdm(pool.imap(encode_shard, split_paths), total=len(split_paths), desc=f"encoding {split_name}"):
                    arrs[path] = arr
        total_n = sum(len(a) for a in arrs.values())
        with open(out_path, "wb") as f:
            for sp in split_paths:  # write back in original shard order
                buf = arrs[sp].tobytes()
                for i in range(0, len(buf), _WRITE_CHUNK):
                    f.write(buf[i : i + _WRITE_CHUNK])
        print(f"wrote {total_n} tokens ({total_n * 2} bytes) -> {out_path}")


if __name__ == "__main__":
    main()
