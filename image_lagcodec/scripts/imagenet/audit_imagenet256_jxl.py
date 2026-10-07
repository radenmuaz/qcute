"""Audit lossless ImageNet256 JXL shards against raw shards and the HF source stream.

The JXL downloader stores images in successful-write order but does not store source-row IDs.
This audit therefore replays the HF split in order, applies the exact ImageNet256 resize/crop
preprocessing, and indexes only successfully preprocessed images. It compares a deterministic
sample against both the raw uint8 shards and the decoded JXL payloads.

Example:
    uv run python image_lagcodec/scripts/imagenet/audit_imagenet256_jxl.py \
      --split train --raw_dir /dev/shm/imagenet256 \
      --jxl_dir /dev/shm/imagenet256_jxl --seed 0 --num_samples 1000

The HF split is streamed from its beginning through the largest sampled index. No full source
split is downloaded or cached as an Arrow dataset. JXL comparisons are exact pixel equality.
Pass --full to check every common raw/JXL index instead of sampling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_cache/datasets")

import imagecodecs
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def center_crop_resize(pil_image: Image.Image, resolution: int) -> np.ndarray:
    """Match download_imagenet256.py: resize in native mode, then convert to RGB."""
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.asarray(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    return arr[crop_y:crop_y + resolution, crop_x:crop_x + resolution]


class RawShards:
    def __init__(self, root: Path, split: str, resolution: int):
        self.root = root
        self.resolution = resolution
        self.paths = sorted(root.glob(f"imagenet{resolution}_{split}_*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"no imagenet{resolution}_{split}_*.npy shards in {root}")
        self.images = []
        self.labels = []
        counts = []
        for path in self.paths:
            images = np.load(path, mmap_mode="r")
            if images.dtype != np.uint8 or images.shape[-1] != resolution * resolution * 3:
                raise ValueError(f"unexpected raw shard shape/dtype: {path}: {images.shape}/{images.dtype}")
            label_path = root / "labels" / path.name
            if not label_path.is_file():
                raise FileNotFoundError(f"missing raw label shard: {label_path}")
            labels = np.load(label_path, mmap_mode="r")
            if labels.ndim != 1 or len(labels) != len(images):
                raise ValueError(f"raw image/label count mismatch: {path} and {label_path}")
            self.images.append(images)
            self.labels.append(labels)
            counts.append(len(images))
        self.counts = np.asarray(counts, dtype=np.int64)
        self.ends = np.cumsum(self.counts)

    def get(self, index: int) -> tuple[np.ndarray, int]:
        shard = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        local = index - start
        image = np.asarray(self.images[shard][local]).reshape(
            self.resolution, self.resolution, 3
        )
        return image, int(self.labels[shard][local])


class JXLShards:
    def __init__(self, root: Path, split: str, resolution: int):
        manifest_path = root / f"imagenet{resolution}_{split}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"JXL manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if (self.manifest.get("format") != "jpeg-xl-lossless-concatenated"
                or self.manifest.get("resolution") != resolution
                or self.manifest.get("split") != split):
            raise ValueError(f"unexpected JXL manifest format/resolution/split: {manifest_path}")
        self.shards = []
        counts = []
        for record in self.manifest["shards"]:
            payload_path = root / record["payload"]
            offsets = np.load(root / record["offsets"], mmap_mode="r")
            labels = np.load(root / record["labels"], mmap_mode="r")
            count = int(record["count"])
            if len(offsets) != count + 1 or len(labels) != count:
                raise ValueError(f"JXL metadata count mismatch for {payload_path}")
            if offsets[0] != 0 or offsets[-1] != payload_path.stat().st_size:
                raise ValueError(f"JXL offsets do not cover payload: {payload_path}")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"JXL offsets are not monotonic: {payload_path}")
            payload = np.memmap(payload_path, mode="r", dtype=np.uint8)
            self.shards.append((payload, offsets, labels))
            counts.append(count)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self.count = int(self.ends[-1]) if len(self.ends) else 0
        if self.count != int(self.manifest.get("images_written", -1)):
            raise ValueError("JXL manifest images_written does not match shard counts")

    def get(self, index: int, resolution: int) -> tuple[np.ndarray, int]:
        shard = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        local = index - start
        payload, offsets, labels = self.shards[shard]
        lo, hi = int(offsets[local]), int(offsets[local + 1])
        image = np.asarray(imagecodecs.jpegxl_decode(payload[lo:hi]))
        expected = (resolution, resolution, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"decoded JXL {index} has {image.shape}/{image.dtype}, expected {expected}/uint8")
        return image, int(labels[local])


def compare_images(reference: np.ndarray, actual: np.ndarray) -> tuple[bool, str]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return False, f"shape/dtype {actual.shape}/{actual.dtype}, expected {reference.shape}/{reference.dtype}"
    diff = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    if not np.any(diff):
        return True, "exact"
    y, x, channel = np.argwhere(diff != 0)[0]
    return False, (f"{np.count_nonzero(np.any(diff != 0, axis=-1))} pixels differ; "
                   f"max_abs_diff={int(diff.max())}; first=({y},{x},{channel}) "
                   f"{int(reference[y, x, channel])}!={int(actual[y, x, channel])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--raw_dir", type=Path, default=Path("/dev/shm/imagenet256"))
    parser.add_argument("--jxl_dir", type=Path, default=Path("/dev/shm/imagenet256_jxl"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--full", action="store_true",
                        help="check every common raw/JXL sample; ignore --num_samples and --seed")
    parser.add_argument("--hf_dataset", type=str, default="ILSVRC/imagenet-1k")
    args = parser.parse_args()
    if args.resolution < 1 or args.num_samples < 1:
        parser.error("resolution and num_samples must be positive")

    raw = RawShards(args.raw_dir, args.split, args.resolution)
    jxl = JXLShards(args.jxl_dir, args.split, args.resolution)
    common_count = min(int(raw.ends[-1]), jxl.count)
    if common_count == 0:
        raise ValueError("raw and JXL datasets have no common samples")
    count_mismatch = int(raw.ends[-1]) != jxl.count
    if count_mismatch:
        print(f"WARNING: raw count={int(raw.ends[-1])}, JXL count={jxl.count}; "
              f"checking common prefix of {common_count}")

    n = common_count if args.full else min(args.num_samples, common_count)
    if args.full:
        selected = np.arange(common_count, dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(common_count, size=n, replace=False))
    errors = []
    error_count = 0

    def record_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 50:
            errors.append(message)

    if args.full and count_mismatch:
        record_error(f"full check requires equal counts: raw={int(raw.ends[-1])}, JXL={jxl.count}")

    # In full mode the HF pass below checks both local datasets directly, avoiding a second
    # full decode pass over all JXL images.
    if not args.full:
        for index in selected:
            index = int(index)
            raw_image, raw_label = raw.get(index)
            jxl_image, jxl_label = jxl.get(index, args.resolution)
            ok, detail = compare_images(raw_image, jxl_image)
            if not ok:
                record_error(f"index {index}: raw vs JXL: {detail}")
            if raw_label != jxl_label:
                record_error(f"index {index}: raw/JXL labels differ: {raw_label} != {jxl_label}")

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    if "label" not in ds.features:
        raise ValueError(f"HF stream has no label column: {list(ds.features.keys())}")
    target_pos = 0
    valid_index = 0
    failed_preprocess = 0
    pbar = tqdm(ds, desc=f"HF {args.split}: replaying rows", unit="row")
    for source_row, example in enumerate(pbar):
        if target_pos >= len(selected):
            break
        try:
            source_image = center_crop_resize(example["image"], args.resolution)
            if source_image.shape != (args.resolution, args.resolution, 3):
                raise ValueError(f"unexpected source shape {source_image.shape}")
            source_label = int(example["label"])
        except Exception as exc:
            failed_preprocess += 1
            continue

        if valid_index == int(selected[target_pos]):
            raw_image, raw_label = raw.get(valid_index)
            jxl_image, jxl_label = jxl.get(valid_index, args.resolution)
            ok_raw, detail_raw = compare_images(source_image, raw_image)
            ok_jxl, detail_jxl = compare_images(source_image, jxl_image)
            if not ok_raw:
                record_error(f"index {valid_index} (HF row {source_row}): HF vs raw: {detail_raw}")
            if not ok_jxl:
                record_error(f"index {valid_index} (HF row {source_row}): HF vs JXL: {detail_jxl}")
            if source_label != raw_label or source_label != jxl_label:
                record_error(f"index {valid_index} (HF row {source_row}): labels HF/raw/JXL="
                             f"{source_label}/{raw_label}/{jxl_label}")
            target_pos += 1
            if target_pos == n or target_pos % 1000 == 0:
                pbar.set_postfix(checked=target_pos, selected=n)
        valid_index += 1

    if target_pos < n:
        record_error(f"HF stream ended after matching {target_pos}/{n} selected samples; "
                     f"processed {valid_index} valid images and skipped {failed_preprocess} rows")

    mode = "full" if args.full else f"sampled seed={args.seed}"
    print(f"split={args.split} mode={mode} checked={target_pos}/{n} "
          f"raw_count={int(raw.ends[-1])} jxl_count={jxl.count} "
          f"hf_preprocess_failures={failed_preprocess}")
    if error_count:
        print(f"FAIL: {error_count} mismatch/error(s)")
        for error in errors[:50]:
            print(f"  {error}")
        if error_count > len(errors):
            print(f"  ... {error_count - len(errors)} more")
        return 1

    print("PASS: every checked decoded JXL image exactly matches both the raw shard and HF source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
