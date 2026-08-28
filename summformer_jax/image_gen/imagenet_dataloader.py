"""Dataloader for the flat-byte image shard format scripts/prep_imagenet64.py writes (uint8
(N, H*W*3) .npy, one row per whole image -- NOT the sliding-window text-stream semantics
summformer_jax/lm/data_loader.py uses, since one image is one training example here, no arbitrary
re-windowing). Also supports a synthetic random-RGB mode (no shards needed) for testing.

Train/val split follows the standard downsampled-ImageNet literature convention (van den Oord et
al. 2016 PixelRNN/PixelCNN, the TFDS `downsampled_imagenet` set, through Fractal Generative
Models): ImageNet-1k's own train (~1.28M)/validation (~50K) split used directly, with validation
serving as the eval/bits-per-dim set (ImageNet has no publicly-labeled test split). Matches
scripts/prep_imagenet64.py's own file naming (`imagenet64_train_*.npy` / `imagenet64_validation_*.npy`,
itself pulled straight from ILSVRC/imagenet-1k's train/validation splits).
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

# center_crop_resize lives in scripts/prep_imagenet64.py -- imported, not duplicated, so the
# online path uses the EXACT SAME already-verified-byte-identical-to-openai/improved-diffusion
# resize function (scripts/jax/check_resize_consistency.py), not a second copy that could drift.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))


class ImageByteLoader:
    """Batches whole images as flat raster-order uint8 sequences. `shard_dir` holding real
    prep_imagenet64.py output, or pass shard_dir=None for synthetic random-RGB batches (useful
    for testing model/train code with zero real data, e.g. --resolution 64 gives seq_len=12288
    rows of random bytes, reshuffled fresh each call rather than a fixed dataset). `split`
    filters shard filenames by substring ("train" or "validation") -- required whenever
    shard_dir is given and holds both splits in one directory, as prep_imagenet64.py's default
    output does."""

    def __init__(self, batch_size: int, resolution: int, shard_dir: str | None = None,
                 split: str = "train", seed: int = 0):
        self.batch_size = batch_size
        self.resolution = resolution
        self.seq_len = resolution * resolution * 3
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.synthetic = shard_dir is None

        if not self.synthetic:
            all_shards = sorted(glob.glob(os.path.join(shard_dir, "*.npy")))
            self.shard_paths = [p for p in all_shards if split in os.path.basename(p)]
            if not self.shard_paths:
                raise FileNotFoundError(
                    f"no .npy shards matching split={split!r} found under {shard_dir} "
                    f"({len(all_shards)} shards total, none matched)"
                )
            self._shard_idx = 0
            self._row_idx = 0
            self._load_shard(0)

    def _load_shard(self, idx: int):
        self._shard_idx = idx % len(self.shard_paths)
        self._data = np.load(self.shard_paths[self._shard_idx])
        assert self._data.shape[1] == self.seq_len, (
            f"shard {self.shard_paths[self._shard_idx]} has row length {self._data.shape[1]}, "
            f"expected {self.seq_len} (resolution={self.resolution})"
        )
        self._perm = self.rng.permutation(self._data.shape[0])
        self._row_idx = 0

    def next_batch(self) -> np.ndarray:
        """Returns (batch_size, seq_len) uint8 array, one flat raster-RGB image per row.
        Reshuffled/wraps indefinitely -- for training, not a bounded eval sweep."""
        if self.synthetic:
            return self.rng.integers(0, 256, size=(self.batch_size, self.seq_len), dtype=np.uint8)

        rows = []
        while len(rows) < self.batch_size:
            if self._row_idx >= len(self._perm):
                self._load_shard(self._shard_idx + 1)
            take = min(self.batch_size - len(rows), len(self._perm) - self._row_idx)
            idxs = self._perm[self._row_idx:self._row_idx + take]
            rows.append(self._data[idxs])
            self._row_idx += take
        return np.concatenate(rows, axis=0)

    def full_sweep(self, batch_size: int | None = None):
        """Yields (batch_size, seq_len) uint8 batches covering every image in this split EXACTLY
        ONCE (sequential, no reshuffling, last batch per shard may be smaller) -- for a real
        full-val-set bpd measurement, not next_batch()'s indefinite reshuffled sampling. Synthetic
        mode has no bounded set, so this yields a single arbitrary batch instead (there is no
        "full" synthetic sweep)."""
        bs = batch_size or self.batch_size
        if self.synthetic:
            yield self.rng.integers(0, 256, size=(bs, self.seq_len), dtype=np.uint8)
            return
        for path in self.shard_paths:
            data = np.load(path)
            for start in range(0, data.shape[0], bs):
                yield data[start:start + bs]


def _stream_worker(worker_id: int, num_workers: int, split: str, resolution: int,
                    out_q: mp.Queue, stop_evt: mp.Event, hf_token: str | None):
    """Runs in a separate process: streams its OWN shard of ILSVRC/imagenet-1k (HF's own
    `.shard(num_shards, index)` on the streaming IterableDataset -- no duplicate/colliding images
    across workers, same disjoint-shard guarantee torch's DataLoader worker_id/num_workers gives),
    decodes+resizes each image via the verified center_crop_resize, and pushes flat uint8 rows
    into out_q. Loops forever over its shard (re-iterating once exhausted) since training wants an
    indefinite stream, not a bounded one -- full_sweep-style bounded iteration isn't implemented
    for the online loader (would need a "done" sentinel per worker; not needed yet)."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    # /dev/shm is Linux-only tmpfs (TPU nodes) -- falls back to the platform default cache dir
    # when unavailable (e.g. local macOS testing), rather than hard-failing on os.makedirs.
    if os.path.isdir("/dev/shm"):
        os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")
    from datasets import load_dataset
    from prep_imagenet64 import center_crop_resize

    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    ds = ds.shard(num_shards=num_workers, index=worker_id)

    while not stop_evt.is_set():
        for example in ds:
            if stop_evt.is_set():
                return
            try:
                arr = center_crop_resize(example["image"], resolution)
                assert arr.shape == (resolution, resolution, 3)
            except Exception:
                continue
            row = arr.reshape(-1).astype(np.uint8)
            try:
                out_q.put(row, timeout=1.0)
            except Exception:
                if stop_evt.is_set():
                    return


class OnlineImageByteLoader:
    """torch-DataLoader-style online loader: N worker processes stream+decode+resize directly
    from ILSVRC/imagenet-1k (no upfront prep-to-disk pass, no cached raw dataset -- streaming=True
    throughout, same as scripts/prep_imagenet64.py, avoiding the tmpfs-fill failure mode a full
    non-streaming download hit earlier). Correctness: reuses center_crop_resize UNCHANGED (see
    module-level import), so this is the exact same resize math as the offline shard-file path,
    just applied on demand instead of precomputed -- no new resize-correctness surface.

    next_batch() matches ImageByteLoader's contract so train.py needs minimal changes. No
    full_sweep() (see _stream_worker's docstring) -- eval on the online path uses a bounded number
    of next_batch() calls instead of a true full-set sweep, a real (documented) limitation vs. the
    offline loader's full_sweep()."""

    def __init__(self, batch_size: int, resolution: int, split: str = "train",
                 num_workers: int = 8, queue_size: int = 256, seed: int = 0):
        self.batch_size = batch_size
        self.resolution = resolution
        self.seq_len = resolution * resolution * 3
        self.synthetic = False
        self.split = split

        hf_token = os.environ.get("HF_TOKEN")
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=queue_size)
        self._stop_evt = self._ctx.Event()
        self._workers = [
            self._ctx.Process(
                target=_stream_worker,
                args=(i, num_workers, split, resolution, self._queue, self._stop_evt, hf_token),
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for w in self._workers:
            w.start()

    def next_batch(self) -> np.ndarray:
        rows = [self._queue.get() for _ in range(self.batch_size)]
        return np.stack(rows, axis=0)

    def close(self):
        self._stop_evt.set()
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
