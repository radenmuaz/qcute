"""Procedurally generate a Pathfinder-style long-range-dependency dataset (two marked points on a
canvas, connected by a snake-like dashed curve or not) and write it as gpt2_jax-compatible uint16
.npy token shards -- same shard convention as gpt2_jax/dataset_preparation.py (filename contains
"train"/"val", flat 1D array, read whole by gpt2_jax/data_loader.py's DataLoaderLite).

NOT a port of the original LRA/drewlinsley renderer (that generator -- snakes2.py in
github.com/drewlinsley/pathfinder -- is a separate, unpublished-as-pip MATLAB-derived tool this
script does not depend on). This is an independent reimplementation of the same *task definition*
(random-walk "snake" contours built from short connected dashes, distractor snakes, two circular
end markers, binary same-path/different-path label), parameterized with the same per-resolution
constants LRA's TFDS builder documents (lra_benchmarks/data/pathfinder.py's Pathfinder32/
Pathfinder256 docstrings) where they map onto this simplified renderer. Pixel-level statistics will
not match the original benchmark's released images.

Each example is serialized as: [flattened grayscale pixels (0-255)] + [1 label token: 256 or 257
(connected / not)], concatenated end-to-end across all examples in a split -- matching
DataLoaderLite's flat next-token-prediction stream. vocab needed = 258, well under gpt2_jax's
default vocab_size=50304.

    uv run python scripts/generate_pathfinder.py --resolution 32 --n_train 4000 --n_val 400
    uv run python scripts/generate_pathfinder.py --resolution 256 --n_train 4000 --n_val 400 \
      --out_dir /dev/shm/pathfinder256
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL_CONNECTED = 256
LABEL_DISCONNECTED = 257
VOCAB_SIZE = 258

# per-resolution constants, transcribed from lra_benchmarks/data/pathfinder.py's Pathfinder32 /
# Pathfinder256 docstrings (drewlinsley/pathfinder args), adapted to this renderer's parameters.
RES_PARAMS = {
    32: dict(marker_radius=1.5, contour_length=14, paddle_thickness=0.5, antialias_scale=4,
             num_distractor_snakes=5, padding=1),
    256: dict(marker_radius=5.0, contour_length=14, paddle_thickness=2.0, antialias_scale=2,
              num_distractor_snakes=7, padding=3),
}


def _random_snake(rng: random.Random, size: int, n_segments: int, seg_len: float,
                   max_turn_deg: float, margin: float) -> list[tuple[float, float]]:
    """Random walk of n_segments connected dashes with bounded turning angle. Returns the polyline
    vertices (n_segments + 1 points), reflecting off the canvas edges to stay in bounds."""
    x = rng.uniform(margin, size - margin)
    y = rng.uniform(margin, size - margin)
    heading = rng.uniform(0, 2 * math.pi)
    pts = [(x, y)]
    for _ in range(n_segments):
        heading += math.radians(rng.uniform(-max_turn_deg, max_turn_deg))
        nx, ny = x + seg_len * math.cos(heading), y + seg_len * math.sin(heading)
        if nx < margin or nx > size - margin:
            heading = math.pi - heading
            nx = min(max(nx, margin), size - margin)
        if ny < margin or ny > size - margin:
            heading = -heading
            ny = min(max(ny, margin), size - margin)
        x, y = nx, ny
        pts.append((x, y))
    return pts


def _draw_snake(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], thickness: float,
                 fill: int = 255) -> None:
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=max(1, round(thickness)))


def generate_example(rng: random.Random, resolution: int) -> tuple[np.ndarray, int]:
    p = RES_PARAMS[resolution]
    scale = p["antialias_scale"]
    hi_res = resolution * scale
    seg_len = hi_res / (p["contour_length"] * 1.4)
    margin = p["padding"] * scale + p["marker_radius"] * scale

    img = Image.new("L", (hi_res, hi_res), color=0)
    draw = ImageDraw.Draw(img)

    main_pts = _random_snake(rng, hi_res, p["contour_length"], seg_len, max_turn_deg=35, margin=margin)
    _draw_snake(draw, main_pts, p["paddle_thickness"] * scale)

    distractor_len = max(2, p["contour_length"] // 3)
    distractors = [
        _random_snake(rng, hi_res, distractor_len, seg_len, max_turn_deg=45, margin=margin)
        for _ in range(p["num_distractor_snakes"])
    ]
    for d_pts in distractors:
        _draw_snake(draw, d_pts, p["paddle_thickness"] * scale)

    connected = rng.random() < 0.5
    if connected or not distractors:
        p0, p1 = main_pts[0], main_pts[-1]
        label = LABEL_CONNECTED
    else:
        p0 = main_pts[0]
        p1 = rng.choice(distractors)[-1]
        label = LABEL_DISCONNECTED

    r = p["marker_radius"] * scale
    for (mx, my) in (p0, p1):
        draw.ellipse([mx - r, my - r, mx + r, my + r], fill=255)

    img = img.resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8), label


def write_shard(examples: list[tuple[np.ndarray, int]], dest: Path) -> None:
    parts = []
    for pixels, label in examples:
        parts.append(pixels.reshape(-1).astype(np.uint16))
        parts.append(np.array([label], dtype=np.uint16))
    tokens = np.concatenate(parts)
    np.save(dest, tokens)
    print(f"wrote {dest} ({len(examples)} examples, {len(tokens)} tokens)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", type=int, required=True, choices=sorted(RES_PARAMS))
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None,
                    help="defaults to data/pathfinder<resolution>")
    args = p.parse_args()

    out_dir = args.out_dir or Path("data") / f"pathfinder{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        examples = [generate_example(rng, args.resolution) for _ in range(n)]
        n_pos = sum(1 for _, l in examples if l == LABEL_CONNECTED)
        print(f"{split}: {n} examples, {n_pos} connected ({100*n_pos/n:.1f}%)")
        write_shard(examples, out_dir / f"pathfinder{args.resolution}_{split}_0.npy")


if __name__ == "__main__":
    main()
