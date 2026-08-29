"""Analytic parameter + KV-cache memory estimate for summformer_jax_v2 vs. gpt2_jax at matched
configs -- not a benchmark, a closed-form estimate from each model's own config fields (n_layers,
d_model, n_heads, context_len, main_window, fuse_stages/Ks). KV cache is the memory needed to hold
one full-context autoregressive generation's K/V state (bytes = 2 (K+V) * n_heads * head_dim *
seq_len * batch * dtype_bytes per layer, summed over layers -- standard transformer KV-cache
formula, applied per-mechanism for summformer's trunk + each fuse-stage's code-LM).

    uv run python scripts/estimate_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gpt2_jax"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "summformer_jax" / "lm"))

import jax.numpy as jnp
from flax import nnx


def _param_count(module) -> int:
    state = nnx.state(module, nnx.Param)
    return sum(x.size for x in jax_tree_leaves(state))


def jax_tree_leaves(tree):
    import jax
    return jax.tree_util.tree_leaves(tree)


def gpt2_kv_cache_bytes(n_layer: int, n_embd: int, n_head: int, seq_len: int, batch: int, dtype_bytes: int = 2) -> int:
    head_dim = n_embd // n_head
    return 2 * n_layer * n_head * head_dim * seq_len * batch * dtype_bytes


def summformer_kv_cache_bytes(n_layers: int, d_model: int, n_heads: int, context_len: int,
                               main_window, fuse_stages: tuple, batch: int, dtype_bytes: int = 2) -> tuple:
    """Returns (trunk_bytes, code_lm_bytes, total_bytes). fuse_stages entries:
    (insert_after, stride, window, code_n_layers, source_index) -- the v2 lm lineage's flat format."""
    head_dim = d_model // n_heads
    if main_window is None:
        windows = [context_len] * n_layers
    elif isinstance(main_window, (tuple, list)):
        windows = list(main_window)
    else:
        windows = [main_window] * n_layers
    trunk_bytes = sum(2 * n_heads * head_dim * min(w, context_len) * batch * dtype_bytes for w in windows)

    code_bytes = 0
    for (insert_after, stride, window, code_n_layers, source_index) in fuse_stages:
        code_len = context_len // stride
        code_bytes += 2 * code_n_layers * n_heads * head_dim * code_len * batch * dtype_bytes
    return trunk_bytes, code_bytes, trunk_bytes + code_bytes


def human(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def main():
    from model_gpt import Model, ModelConfig
    from model_summformer_v2 import SummTransformerV2, ConfigV2

    batch = 1
    seq_len = 1024
    dtype_bytes = 2  # bf16 KV cache, matches these models' compute_dtype

    # Real shapes used in the ablation (from MODEL_SHAPES / configs/summformer_jax_v2/*.py)
    gpt2_shapes = {
        "gpt2-small": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    }
    summformer_configs = {
        "summformer-small-parammatch": dict(n_layers=2, d_model=768, n_heads=12,
                                             fuse_stages=((1, 2, None, 1, -1), (2, 2, None, 1, -1))),
        "summformer-small-flopsmatch": dict(n_layers=9, d_model=768, n_heads=12,
                                             fuse_stages=((4, 2, None, 1, -1), (9, 2, None, 1, -1))),
        "summformer-medium-parammatch": dict(n_layers=10, d_model=1024, n_heads=16,
                                              fuse_stages=((2, 2, None, 1, -1), (5, 2, None, 1, -1),
                                                           (8, 2, None, 1, -1), (10, 2, None, 1, -1))),
        "summformer-medium-flopsmatch": dict(n_layers=18, d_model=1024, n_heads=16,
                                              fuse_stages=((4, 2, None, 1, -1), (9, 2, None, 1, -1),
                                                           (14, 2, None, 1, -1), (18, 2, None, 1, -1))),
    }

    print(f"KV-cache memory at context_len={seq_len}, batch={batch}, dtype=bf16(2 bytes)\n")
    print(f"{'model':32s} {'params':>12s} {'kv cache':>12s} {'kv vs gpt2':>12s}")
    gpt2_kv_ref = {}
    for name, shape in gpt2_shapes.items():
        cfg = ModelConfig(**shape)
        model = Model(cfg, rngs=nnx.Rngs(0))
        params = _param_count(model)
        kv = gpt2_kv_cache_bytes(shape["n_layer"], shape["n_embd"], shape["n_head"], seq_len, batch, dtype_bytes)
        gpt2_kv_ref[name] = kv
        print(f"{name:32s} {params:>12,d} {human(kv):>12s} {'--':>12s}")

    for name, sc in summformer_configs.items():
        cfg = ConfigV2(n_layers=sc["n_layers"], d_model=sc["d_model"], n_heads=sc["n_heads"],
                        context_len=seq_len, main_window=None, fuse_stages=sc["fuse_stages"])
        model = SummTransformerV2(cfg, rngs=nnx.Rngs(0))
        params = _param_count(model)
        trunk_b, code_b, total_b = summformer_kv_cache_bytes(
            sc["n_layers"], sc["d_model"], sc["n_heads"], seq_len, None, sc["fuse_stages"], batch, dtype_bytes)
        ref = gpt2_kv_ref["gpt2-small"] if "small" in name else gpt2_kv_ref["gpt2-medium"]
        print(f"{name:32s} {params:>12,d} {human(total_b):>12s} {total_b/ref*100:>11.1f}%")
        print(f"{'  (trunk / code-LM split)':32s} {'':>12s} {human(trunk_b)+'/'+human(code_b):>12s}")


if __name__ == "__main__":
    main()
