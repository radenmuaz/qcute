"""Byte-level FineWeb-Edu-10B prep for summformer_jax's byte-vocab mode (Config.vocab_size=None,
input_preset=8). Writes raw UTF-8 bytes (uint8) instead of gpt2_jax/dataset_preparation.py's
GPT2-BPE token ids (uint16) -- but the train/val BOUNDARY must land on the exact same document
as that script's, not an independently-computed byte-count threshold (a doc's byte length and
its GPT2-BPE token length differ, so re-deriving the split from a raw-byte shard_size would move
the boundary to a different document and silently break comparability between the two lineages).

To guarantee document-exact parity: this script runs the SAME tiktoken accumulation as
gpt2_jax/dataset_preparation.py (shard_size=1e8 tokens, first-shard-is-val) purely to find which
document index crosses that threshold -- the token ids themselves are discarded immediately after
computing their count; only the raw UTF-8 bytes of each document are ever written to disk. Streaming
order from `load_dataset(..., split="train")` is deterministic (no shuffle), so document N is the
same document in both scripts' runs, and the val/train split point matches exactly.
"""
import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, choices=["fineweb-10B", "fineweb-edu-10B"], required=True)
args = parser.parse_args()

if args.dataname == "fineweb-10B":
    local_dir = "../data/fineweb-10B-bytes"
    data_path = "HuggingFaceFW/fineweb"
    sample = "sample-10BT"
elif args.dataname == "fineweb-edu-10B":
    local_dir = "../data/fineweb-edu-10B-bytes"
    data_path = "HuggingFaceFW/fineweb-edu"
    sample = "sample-10BT"
else:
    raise ValueError(f"Unknown dataname {args.dataname}!")

TOKEN_SHARD_SIZE = int(1e8)  # must match gpt2_jax/dataset_preparation.py's shard_size exactly --
                              # this is the boundary-determination unit, not the byte output unit.
BYTE_SHARD_SIZE = int(1e8)   # output chunking granularity for the byte shards themselves (does
                              # NOT affect the train/val split point, only how many byte-shard
                              # files train ends up split across).

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset(data_path, sample, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def doc_to_token_count_and_bytes(doc):
    # Same accumulation unit as gpt2_jax/dataset_preparation.py's tokenize() (incl. the leading
    # eot token) so the cumulative-token boundary check below lands on the identical document --
    # the token ids themselves are discarded right after len() is taken.
    n_tokens = 1 + len(enc.encode_ordinary(doc["text"]))
    doc_bytes = np.frombuffer(doc["text"].encode("utf-8"), dtype=np.uint8)
    return n_tokens, doc_bytes


def write_datafile(filename, bytes_np):
    np.save(filename, bytes_np)


nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0             # byte-shard index (output chunking)
    token_shard_count = 0       # cumulative tokens in the CURRENT token-boundary shard (val=shard 0 only)
    crossed_val_boundary = False
    all_bytes_np = np.empty((BYTE_SHARD_SIZE,), dtype=np.uint8)
    byte_count = 0
    progress_bar = None
    for n_tokens, doc_bytes in pool.imap(doc_to_token_count_and_bytes, fw, chunksize=16):
        if not crossed_val_boundary:
            token_shard_count += n_tokens
            if token_shard_count >= TOKEN_SHARD_SIZE:
                # this document is the last one in gpt2_jax's val shard (shard_index==0) -- flush
                # whatever's accumulated so far as the (only) val byte-shard, then switch to train.
                split = "val"
                filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_{split}_{shard_index:06d}")
                if byte_count + len(doc_bytes) <= BYTE_SHARD_SIZE:
                    all_bytes_np[byte_count:byte_count + len(doc_bytes)] = doc_bytes
                    byte_count += len(doc_bytes)
                    write_datafile(filename, all_bytes_np[:byte_count])
                else:
                    # this doc's bytes overflow the buffer -- write what's there, doc itself
                    # becomes train's first bytes (still the same document-level boundary as
                    # gpt2_jax: everything up to and including this doc is val there too, but
                    # gpt2_jax's own per-token split can also fall mid-document -- see note below).
                    write_datafile(filename, all_bytes_np[:byte_count])
                shard_index += 1
                byte_count = 0
                crossed_val_boundary = True
                continue
            else:
                if progress_bar is None:
                    progress_bar = tqdm(total=TOKEN_SHARD_SIZE, unit="tokens", desc="val (shard 0)")
                progress_bar.update(n_tokens)
                if byte_count + len(doc_bytes) < BYTE_SHARD_SIZE:
                    all_bytes_np[byte_count:byte_count + len(doc_bytes)] = doc_bytes
                    byte_count += len(doc_bytes)
                    continue
                # val byte buffer full before the token boundary is crossed (rare, only for a
                # huge sample) -- flush and keep accumulating into the next val byte-shard.
                filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_val_{shard_index:06d}")
                write_datafile(filename, all_bytes_np[:byte_count])
                shard_index += 1
                byte_count = 0
                all_bytes_np[0:len(doc_bytes)] = doc_bytes
                byte_count = len(doc_bytes)
                continue

        # train phase: plain byte-count-chunked shards from here on (boundary already fixed above)
        if byte_count + len(doc_bytes) < BYTE_SHARD_SIZE:
            all_bytes_np[byte_count:byte_count + len(doc_bytes)] = doc_bytes
            byte_count += len(doc_bytes)
            if progress_bar is None or progress_bar.desc != f"train (shard {shard_index})":
                progress_bar = tqdm(total=BYTE_SHARD_SIZE, unit="bytes", desc=f"train (shard {shard_index})")
            progress_bar.update(len(doc_bytes))
        else:
            filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_train_{shard_index:06d}")
            remainder = BYTE_SHARD_SIZE - byte_count
            all_bytes_np[byte_count:byte_count + remainder] = doc_bytes[:remainder]
            write_datafile(filename, all_bytes_np)
            shard_index += 1
            progress_bar = None
            leftover = doc_bytes[remainder:]
            all_bytes_np[0:len(leftover)] = leftover
            byte_count = len(leftover)

    if byte_count != 0:
        split = "train" if crossed_val_boundary else "val"
        filename = os.path.join(DATA_CACHE_DIR, f"{args.dataname}_{split}_{shard_index:06d}")
        write_datafile(filename, all_bytes_np[:byte_count])
