"""Image classifier built on the frozen summformer.py backbone (verbatim duplicate of
image_gen/summformer.py -- not modified here, see that file's own freeze note). Reuses
SummTransformerV2._cascade directly (trunk + fuse-stage forward pass, already returns
ln_f-normalized final hidden states of shape (B, L, D) -- exactly the hook point needed, no
generation/LM-head/MTP machinery involved).

Two pooling variants (both still causal internally -- this codebase's windowed attention
(chunked_windowed_attention/causal_mask) is only defined for a causal direction, so genuine
non-causal/bidirectional attention isn't supported; "bidirectional" here means running the causal
backbone twice, forward and over the reversed sequence, then combining):
  - unidirectional: single causal pass, last position's hidden state -> linear head.
  - bidirectional: forward-causal pass + reverse-causal pass (sequence reversed), mean-pooled
    hidden states from each averaged together, -> linear head.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from summformer import ConfigV2, SummTransformerV2


@dataclass
class ClassifierConfig(ConfigV2):
    num_classes: int = 1000
    bidirectional: bool = False


class SummClassifier(nnx.Module):
    def __init__(self, cfg: ClassifierConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.backbone = SummTransformerV2(cfg, rngs=rngs)
        init = nnx.initializers.normal(stddev=0.02)
        self.head = nnx.Linear(cfg.d_model, cfg.num_classes, use_bias=True, kernel_init=init,
                                dtype=cfg.compute_dtype, param_dtype=cfg.param_dtype, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """token_ids: (B, L) uint8-range int32 byte tokens (flattened raster RGB). Returns
        (B, num_classes) logits."""
        h_fwd = self.backbone._cascade(token_ids)  # (B, L, D)
        if not self.cfg.bidirectional:
            pooled = h_fwd[:, -1, :]
        else:
            h_bwd = self.backbone._cascade(token_ids[:, ::-1])
            pooled = (h_fwd.mean(axis=1) + h_bwd.mean(axis=1)) / 2
        return self.head(pooled)


def cross_entropy_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1).mean()
