"""Pixel-grid scan orderings for MultiScanClassifier -- each function returns a (H*W,) int array
of pixel indices (row*W+col) giving the traversal order a causal scan should visit them in.
`pixel_order_to_byte_order` expands a pixel order into the actual flat byte-sequence order (each
pixel contributes `channels` consecutive bytes in raster RGB layout).

Non-power-of-2 grids (e.g. 224x224): z_order/hilbert_order pad to the next power-of-2 square,
compute the curve over the full padded grid, then filter down to the real H*W cells, keeping their
relative order along the curve -- standard technique, not an approximation of the traversal itself
(the filtered sequence is an exact sub-sequence of the true curve order).

Custom orderings: subclass `ScanOrder` and implement `pixel_order(H, W)`, or just pass any
(H*W,)-shaped permutation array directly to MultiScanClassifier -- no subclassing required unless
you want it registered by name alongside the built-ins.
"""
from __future__ import annotations

import numpy as np


def pixel_order_to_byte_order(pixel_order: np.ndarray, channels: int = 3) -> np.ndarray:
    """(H*W,) pixel order -> (H*W*channels,) byte order, each pixel's `channels` bytes kept
    consecutive and in original channel order."""
    return (pixel_order[:, None] * channels + np.arange(channels)[None, :]).reshape(-1)


def row_major(H: int, W: int) -> np.ndarray:
    """Standard raster order -- the identity/default scan (top-left to bottom-right, row by row)."""
    return np.arange(H * W)


def row_major_reverse(H: int, W: int) -> np.ndarray:
    return row_major(H, W)[::-1].copy()


def col_major(H: int, W: int) -> np.ndarray:
    """Vertical scan -- column by column, top to bottom within each column."""
    return np.arange(H * W).reshape(H, W).T.reshape(-1).copy()


def col_major_reverse(H: int, W: int) -> np.ndarray:
    return col_major(H, W)[::-1].copy()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def z_order(H: int, W: int) -> np.ndarray:
    """Morton/Z-order curve. Pads to the next power-of-2 square, computes the curve over the full
    padded grid via bit-interleaving (vectorized), then filters to the real H x W cells."""
    P = _next_pow2(max(H, W))
    rows = np.arange(P, dtype=np.uint32)
    cols = np.arange(P, dtype=np.uint32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")  # (P, P)

    def _interleave_bits(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.uint64)
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
        x = (x | (x << 2)) & 0x3333333333333333
        x = (x | (x << 1)) & 0x5555555555555555
        return x

    morton = (_interleave_bits(rr) << 1) | _interleave_bits(cc)  # (P, P)
    order = np.argsort(morton, axis=None, kind="stable")  # flat index into (P,P), curve order
    order_rows = order // P
    order_cols = order % P
    valid = (order_rows < H) & (order_cols < W)
    real_rows = order_rows[valid]
    real_cols = order_cols[valid]
    return (real_rows * W + real_cols).astype(np.int64)


def hilbert_order(H: int, W: int) -> np.ndarray:
    """Hilbert curve. Pads to the next power-of-2 square, computes d2xy for every curve index
    0..P*P-1 (vectorizable per-bit-pair iteration, standard public-domain algorithm), then filters
    to the real H x W cells, same padding technique as z_order."""
    P = _next_pow2(max(H, W))
    n_bits = int(np.log2(P))
    d = np.arange(P * P, dtype=np.int64)
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    t = d.copy()
    s = 1
    for _ in range(n_bits):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # rotate
        swap = ry == 0
        flip = swap & (rx == 1)
        x_new = np.where(flip, s - 1 - x, x)
        y_new = np.where(flip, s - 1 - y, y)
        x, y = np.where(swap, y_new, x_new), np.where(swap, x_new, y_new)
        x = x + s * rx
        y = y + s * ry
        t = t // 4
        s *= 2

    order_rows = y  # curve index d's position, in traversal order (d already ascending)
    order_cols = x
    valid = (order_rows < H) & (order_cols < W)
    real_rows = order_rows[valid]
    real_cols = order_cols[valid]
    return (real_rows * W + real_cols).astype(np.int64)


SCAN_ORDERS = {
    "row_major": row_major,
    "row_major_reverse": row_major_reverse,
    "col_major": col_major,
    "col_major_reverse": col_major_reverse,
    "z_order": z_order,
    "hilbert": hilbert_order,
}


class ScanOrder:
    """Subclass this and implement pixel_order(H, W) for a custom ordering not covered above,
    then pass an instance's .pixel_order method (or just a precomputed (H*W,) array) directly to
    MultiScanClassifier -- registration in SCAN_ORDERS is only needed if you want it selectable
    by name from a config."""

    def pixel_order(self, H: int, W: int) -> np.ndarray:
        raise NotImplementedError
