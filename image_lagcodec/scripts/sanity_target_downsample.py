"""CPU-only, no model/checkpoint needed: sanity-checks rgb_label_fn_jax in isolation -- plots real
GT next to its target downsample (the same target label_reg_weight/label_mse train against) for a
few real CIFAR images. Pure function check: does the target look like a real color downsample, or
still degenerate (e.g. red-only, the default_label_fn_jax bug this was written to fix)?
Usage: python3 -m image_lagcodec.scripts.sanity_target_downsample [--n_img N] [--level 0]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from image_lagcodec.run_lagcodec import (
    Config, CONFIG_FIELDS, load_config_module, load_cifar10, images_to_positions, pixel_order_for,
    rgb_label_fn_jax, default_label_fn_jax,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_img", type=int, default=6)
    ap.add_argument("--level_n_blocks", type=int, default=256)  # matches cifar_probe_labelreg1's level0 (side=16)
    ap.add_argument("--out", default=str(REPO_ROOT / "image_lagcodec/scripts/sanity_target_downsample.png"))
    a = ap.parse_args()

    cv = load_config_module(REPO_ROOT / "image_lagcodec/configs/cifar_probe_labelreg1.py")
    cv.pop("label_fn", None)
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    (train_x, _), _ = load_cifar10(REPO_ROOT / "datasets")
    imgs = train_x[: a.n_img]
    pixel_order = pixel_order_for(cfg)
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))

    target_rgb = np.asarray(rgb_label_fn_jax(flat, cfg, pixel_order, a.level_n_blocks, 3, 256))
    target_old = np.asarray(default_label_fn_jax(flat, cfg, pixel_order, a.level_n_blocks, 3, 256))

    side = round(a.level_n_blocks ** 0.5)
    from image_lagcodec.run_lagcodec import zorder_pixel_order
    low_order = zorder_pixel_order(side)

    def to_grid(idx):
        M = idx.shape[0]
        raster = np.zeros((M, side * side, 3), dtype=np.uint8)
        raster[:, low_order, :] = idx.astype(np.uint8)
        return raster.reshape(M, side, side, 3)

    grid_new = to_grid(target_rgb)
    grid_old = to_grid(target_old)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = imgs.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    axes = axes.reshape(n, 3)
    for i in range(n):
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title("GT (32x32)" if i == 0 else "", fontsize=9)
        axes[i, 1].imshow(grid_old[i])
        axes[i, 1].set_title(f"OLD default_label_fn_jax ({side}x{side})" if i == 0 else "", fontsize=9)
        axes[i, 2].imshow(grid_new[i])
        axes[i, 2].set_title(f"NEW rgb_label_fn_jax ({side}x{side})" if i == 0 else "", fontsize=9)
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle("target-downsample sanity check: GT vs old (red-only bug) vs new (real RGB)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")
    print(f"OLD target per-channel means: {target_old.astype(float).mean(axis=(0,1))}  (expect ~[X,0,0])")
    print(f"NEW target per-channel means: {target_rgb.astype(float).mean(axis=(0,1))}  (expect all ~similar, nonzero)")


if __name__ == "__main__":
    main()
