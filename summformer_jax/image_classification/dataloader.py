"""Dataloader for image_classification -- reads the [4-byte label][8-byte length][raw JPEG bytes]
shard format scripts/download_imagenet.py writes (see that script's docstring), decodes each JPEG
via PIL, and applies the same resize/crop/flip logic as Flax's examples/imagenet/input_pipeline.py
(ref_impl_files/input_pipeline.py) -- ported from tf.image ops to PIL/numpy since this repo has no
TensorFlow dependency. Key difference from the Flax reference: NO float normalization (no
MEAN_RGB/STDDEV_RGB) -- output stays raw uint8 byte tokens, matching image_gen's byte-level
convention, since the model consumes a flat byte sequence (vocab_size=256), not normalized floats.

Preprocessing (mirrors _decode_and_random_crop / _decode_and_center_crop / normalize_image):
  - train: random-resized-crop (area 0.08-1.0 of original, aspect ratio 3/4-4/3, up to 10 attempts,
    matching distorted_bounding_box_crop's own defaults) + random horizontal flip.
  - eval: center-crop with CROP_PADDING=32 (image_size/(image_size+32) fraction), matching
    _decode_and_center_crop exactly.
  - both: BICUBIC resize to (image_size, image_size), flattened to (image_size*image_size*3,)
    raster-order uint8 bytes (RGB channel-last order flattened, matching image_gen's own
    resolution*resolution*3 raster convention).
"""
from __future__ import annotations

import glob
import io
import mmap
import os

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
CROP_PADDING = 32


def _random_resized_crop(img: Image.Image, image_size: int, rng: np.random.Generator) -> Image.Image:
    """Port of Flax's distorted_bounding_box_crop + _decode_and_random_crop -- area/aspect-ratio
    random crop with up to 10 attempts, falling back to a plain center-crop-and-resize if none of
    the attempts land a valid box (matches the reference's _at_least_x_are_equal fallback intent,
    simplified since we don't need the exact same bad-crop detection tf.image.sample_distorted_
    bounding_box does internally)."""
    width, height = img.size
    area = width * height
    for _ in range(10):
        target_area = rng.uniform(0.08, 1.0) * area
        log_ratio = (np.log(3 / 4), np.log(4 / 3))
        aspect_ratio = np.exp(rng.uniform(*log_ratio))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            x = rng.integers(0, width - w + 1)
            y = rng.integers(0, height - h + 1)
            crop = img.crop((x, y, x + w, y + h))
            return crop.resize((image_size, image_size), resample=Image.BICUBIC)
    return _center_crop(img, image_size)


def _center_crop(img: Image.Image, image_size: int) -> Image.Image:
    """Port of Flax's _decode_and_center_crop -- crops to a padded-center square then resizes."""
    width, height = img.size
    padded_center_crop_size = int((image_size / (image_size + CROP_PADDING)) * min(width, height))
    offset_x = (width - padded_center_crop_size + 1) // 2
    offset_y = (height - padded_center_crop_size + 1) // 2
    crop = img.crop((offset_x, offset_y, offset_x + padded_center_crop_size, offset_y + padded_center_crop_size))
    return crop.resize((image_size, image_size), resample=Image.BICUBIC)


def preprocess(raw_bytes: bytes, image_size: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """Returns a flat (image_size*image_size*3,) uint8 array, raster RGB order -- no float
    normalization (see module docstring)."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if train:
        img = _random_resized_crop(img, image_size, rng)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = _center_crop(img, image_size)
    arr = np.array(img, dtype=np.uint8)
    assert arr.shape == (image_size, image_size, 3), f"got {arr.shape}"
    return arr.reshape(-1)


class ImageNetClassificationLoader:
    """Reads shards written by scripts/download_imagenet.py: [4-byte signed label]
    [8-byte length][raw JPEG bytes] entries, one shard per worker/split. Decodes+preprocesses
    on the fly (no separate resize-to-disk pass, unlike image_gen's pipeline -- classification
    needs per-epoch random-crop augmentation, which can't be baked into a fixed-size shard the
    way image_gen's deterministic resize can)."""

    def __init__(self, batch_size: int, image_size: int, shard_dir: str, split: str, seed: int = 0):
        self.batch_size = batch_size
        self.image_size = image_size
        self.seq_len = image_size * image_size * 3
        self.split = split
        self.train = split == "train"
        self.rng = np.random.default_rng(seed)

        all_shards = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
        self.shard_paths = [p for p in all_shards if split in os.path.basename(p)]
        if not self.shard_paths:
            raise FileNotFoundError(f"no .bin shards matching split={split!r} found under {shard_dir}")

        # mmap each shard (zero-copy view into the file's own pages -- on /dev/shm those pages
        # ARE the tmpfs RAM already, so this doesn't duplicate memory the way `f.read()` would;
        # on real disk it's lazily paged in on access instead of eagerly loading everything
        # upfront). Only a single sequential scan per shard is done here, just to build the
        # (shard_idx, offset, entry_len) index -- no image bytes are copied out during indexing.
        self._index: list[tuple[int, int, int]] = []
        self._shard_files = []
        self._shard_mmaps = []
        for shard_path in self.shard_paths:
            f = open(shard_path, "rb")
            self._shard_files.append(f)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._shard_mmaps.append(mm)
            offset = 0
            shard_idx = len(self._shard_mmaps) - 1
            while offset < len(mm):
                entry_start = offset
                offset += 4  # label
                length = int.from_bytes(mm[offset:offset + 8], "big")
                offset += 8 + length
                self._index.append((shard_idx, entry_start, offset - entry_start))

        self._perm = self.rng.permutation(len(self._index))
        self._pos = 0

    @property
    def n_images(self) -> int:
        return len(self._index)

    def _read_entry(self, i: int) -> tuple[int, bytes]:
        shard_idx, start, entry_len = self._index[i]
        mm = self._shard_mmaps[shard_idx]
        offset = start
        label = int.from_bytes(mm[offset:offset + 4], "big", signed=True)
        offset += 4
        length = int.from_bytes(mm[offset:offset + 8], "big")
        offset += 8
        raw = mm[offset:offset + length]
        return label, raw

    def close(self):
        for mm in self._shard_mmaps:
            mm.close()
        for f in self._shard_files:
            f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (images (B, seq_len) uint8, labels (B,) int32). Reshuffles indefinitely for
        training; wraps for eval too (use full_sweep for a bounded one-pass eval)."""
        images = np.empty((self.batch_size, self.seq_len), dtype=np.uint8)
        labels = np.empty((self.batch_size,), dtype=np.int32)
        for b in range(self.batch_size):
            if self._pos >= len(self._perm):
                self._perm = self.rng.permutation(len(self._index))
                self._pos = 0
            idx = self._perm[self._pos]
            self._pos += 1
            label, raw = self._read_entry(idx)
            images[b] = preprocess(raw, self.image_size, self.train, self.rng)
            labels[b] = label
        return images, labels

    def full_sweep(self, batch_size: int | None = None):
        """Yields (images, labels) batches covering every image in this split exactly once,
        deterministic center-crop (train=False regardless of self.split) -- for eval."""
        bs = batch_size or self.batch_size
        n = len(self._index)
        for start in range(0, n, bs):
            idxs = range(start, min(start + bs, n))
            images = np.empty((len(idxs), self.seq_len), dtype=np.uint8)
            labels = np.empty((len(idxs),), dtype=np.int32)
            for j, idx in enumerate(idxs):
                label, raw = self._read_entry(idx)
                images[j] = preprocess(raw, self.image_size, False, self.rng)
                labels[j] = label
            yield images, labels
