"""Vendored reference for a consistency check only -- NOT used by prep_imagenet64.py itself.

The resize/crop math is copied verbatim from openai/improved-diffusion's
ImageDataset.__getitem__ (raw.githubusercontent.com/openai/improved-diffusion/main/
improved_diffusion/image_datasets.py, lines 75-96), with everything unrelated to that math
stripped out: no blobfile/mpi4py/torch imports, no file I/O, no Dataset class, no class-label
handling, no [-1,1] float normalization, no CHW transpose -- those are dataloader plumbing, not
part of the resize algorithm being checked.

    uv run python scripts/jax/check_resize_consistency.py
"""
import numpy as np
from PIL import Image


def reference_center_crop_resize(pil_image: Image.Image, resolution: int) -> np.ndarray:
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
    return arr
