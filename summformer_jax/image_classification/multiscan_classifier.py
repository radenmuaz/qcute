"""General multi-scan classifier: runs the summformer.py backbone over the SAME image under
several different pixel-traversal orders (see scan_orders.py), pools each pass, and combines them
into a final linear classification head.

Scan selection: `scan_groups` is a list of groups, each group a list of scan names (from
SCAN_ORDERS) or raw (H*W,) pixel-order arrays. Every scan within one group shares ONE backbone
instance (weights literally reused across those forward calls) -- e.g. group=["row_major",
"row_major_reverse"] is the "bidirectional, shared weights" case: one backbone, run forward and
over the reversed sequence, weights identical both times. Different groups each get their OWN
backbone instance (independent weights) -- e.g. groups=[["row_major"], ["col_major"]] trains two
separate backbones, one per direction, not sharing anything.

Per-scan pooling is always mean-pool over that scan's own final hidden sequence (uniform across
every ordering, matches the plain classifier's "bidirectional" convention already established).

Output combination (`output_mode`):
  - "mean_pool": average every scan's pooled vector together, one shared (d_model -> num_classes)
    linear head.
  - "concat_linear": concatenate every scan's pooled vector (dimension = n_scans * d_model), one
    larger (n_scans*d_model -> num_classes) linear head -- lets the model weight different scans
    differently instead of forcing an unweighted average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from functools import lru_cache

import numpy as np
import jax.numpy as jnp
from flax import nnx

from summformer import ConfigV2, SummTransformerV2
from scan_orders import SCAN_ORDERS, pixel_order_to_byte_order


@lru_cache(maxsize=None)
def _cached_byte_order(scan_name: str, image_size: int, channels: int) -> np.ndarray:
    """Recomputed (not stored as module/nnx state) so the permutation never ends up as a leaf in
    the differentiated param pytree -- jax.value_and_grad fails on non-float leaves. Cheap and
    deterministic given (scan_name, image_size, channels), so a process-level cache is enough;
    only accepts registered scan names (raw-array custom scans must be resolved by the caller)."""
    return _resolve_scan_byte_order(scan_name, image_size, channels)


@dataclass
class MultiScanConfig(ConfigV2):
    num_classes: int = 1000
    image_size: int = 224
    channels: int = 3
    # list of groups; each group is a list of scan names (str, looked up in SCAN_ORDERS) or raw
    # (H*W,) pixel-order arrays. Scans within a group share one backbone; different groups don't.
    scan_groups: tuple = (("row_major",),)
    output_mode: str = "mean_pool"  # "mean_pool" or "concat_linear"


def _resolve_scan_byte_order(scan, image_size: int, channels: int) -> np.ndarray:
    if isinstance(scan, str):
        pixel_order = SCAN_ORDERS[scan](image_size, image_size)
    else:
        pixel_order = np.asarray(scan)
    return pixel_order_to_byte_order(pixel_order, channels)


class MultiScanClassifier(nnx.Module):
    def __init__(self, cfg: MultiScanConfig, *, rngs: nnx.Rngs):
        assert cfg.output_mode in ("mean_pool", "concat_linear")
        self.cfg = cfg

        # one backbone per group (shared across every scan within that group)
        self.backbones = nnx.List([SummTransformerV2(cfg, rngs=rngs) for _ in cfg.scan_groups])
        self._n_scans = sum(len(g) for g in cfg.scan_groups)

        init = nnx.initializers.normal(stddev=0.02)
        head_in = cfg.d_model if cfg.output_mode == "mean_pool" else cfg.d_model * self._n_scans
        self.head = nnx.Linear(head_in, cfg.num_classes, use_bias=True, kernel_init=init,
                                dtype=cfg.compute_dtype, param_dtype=cfg.param_dtype, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """token_ids: (B, L) uint8-range int32 byte tokens, in the ORIGINAL raster order (each
        scan's own permutation is applied internally). Returns (B, num_classes) logits."""
        pooled_per_scan = []
        for backbone, group in zip(self.backbones, self.cfg.scan_groups):
            for scan in group:
                if isinstance(scan, str):
                    order = _cached_byte_order(scan, self.cfg.image_size, self.cfg.channels)
                else:
                    order = _resolve_scan_byte_order(scan, self.cfg.image_size, self.cfg.channels)
                permuted = token_ids[:, jnp.asarray(order)]
                h = backbone._cascade(permuted)  # (B, L, D)
                pooled_per_scan.append(h.mean(axis=1))

        if self.cfg.output_mode == "mean_pool":
            pooled = jnp.mean(jnp.stack(pooled_per_scan, axis=0), axis=0)
        else:
            pooled = jnp.concatenate(pooled_per_scan, axis=-1)
        return self.head(pooled)
