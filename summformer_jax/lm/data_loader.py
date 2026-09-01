"""Verbatim port of gpt2_jax/data_loader.py's DataLoaderLite -- sequential (NOT random), per-process
deterministic access, one shard in memory at a time, wraps to the next shard on exhaustion. Copied
rather than imported (no cross-directory dependency, matches this codebase's other lineages'
convention) so the LM lineage's dataloader is bit-for-bit the same access pattern as the gpt2_jax
baseline it's compared against -- the earlier version of this file used random sampling with
replacement from a concatenated array, a real, unauthorized divergence from the baseline (see chat
2026-08-31). Do not reintroduce random sampling here without an explicit instruction to do so."""
from __future__ import annotations

import os

import numpy as np


def load_tokens(filename: str) -> np.ndarray:
    return np.load(filename).astype(np.int32)


class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str, path: str,
                 master_process: bool = True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        shards = sorted(s for s in os.listdir(path) if split in s)
        self.shards = [os.path.join(path, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split} in {path}"

        self.num_total_tokens = sum(len(load_tokens(s)) for s in self.shards)
        if master_process:
            print(f"found {len(self.shards)} shards for split {split}, {self.num_total_tokens} tokens")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
