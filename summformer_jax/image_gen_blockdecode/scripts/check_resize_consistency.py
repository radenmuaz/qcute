"""Asserts prep_imagenet64.center_crop_resize is byte-identical to the vendored reference in
reference_openai_resize.py, across varied sizes/aspect-ratios/color modes -- the mode variation
specifically exercises the RGB-conversion-order bug found and fixed by the earlier manual diff
(converting to RGB before vs. after the BOX/BICUBIC resize steps can change pixel values).

    uv run python scripts/jax/check_resize_consistency.py
"""
import numpy as np
from PIL import Image

from prep_imagenet64 import center_crop_resize
from reference_openai_resize import reference_center_crop_resize

RNG = np.random.default_rng(0)

CASES = [
    ("RGB square, exact 2x", (128, 128), "RGB"),
    ("RGB wide, non-power-of-2", (200, 133), "RGB"),
    ("RGB tall", (133, 200), "RGB"),
    ("RGB just above resolution", (70, 90), "RGB"),
    ("RGBA (alpha channel)", (150, 150), "RGBA"),
    ("L (grayscale)", (150, 150), "L"),
    ("CMYK", (150, 150), "CMYK"),
    ("P (palette)", (150, 150), "P"),
]


def make_test_image(size, mode):
    if mode == "P":
        base = Image.fromarray(
            RNG.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        )
        return base.convert("P")
    n_channels = {"RGB": 3, "RGBA": 4, "L": 1, "CMYK": 4}[mode]
    shape = (size[1], size[0]) if n_channels == 1 else (size[1], size[0], n_channels)
    arr = RNG.integers(0, 255, size=shape, dtype=np.uint8)
    return Image.fromarray(arr, mode=mode if mode != "L" else "L")


def main():
    resolution = 64
    n_pass = 0
    n_fail = 0
    for name, size, mode in CASES:
        img = make_test_image(size, mode)
        mine = center_crop_resize(img, resolution)
        ref = reference_center_crop_resize(img, resolution)
        ok = mine.shape == ref.shape and np.array_equal(mine, ref)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: size={size} mode={mode} "
              f"mine.shape={mine.shape} ref.shape={ref.shape} "
              f"max_abs_diff={np.abs(mine.astype(int) - ref.astype(int)).max() if ok is False and mine.shape == ref.shape else (0 if ok else 'shape mismatch')}")
        n_pass += ok
        n_fail += not ok

    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
