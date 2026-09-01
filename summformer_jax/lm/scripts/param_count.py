"""Param-count comparison: a summformer_jax/lm config vs. gpt2_jax small/medium baselines.

    uv run python summformer_jax/lm/scripts/param_count.py [config_path]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "gpt2_jax"))
sys.path.insert(0, str(_REPO_ROOT / "summformer_jax"))

from model_gpt import ModelConfig, Model  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "summformer_jax/lm/configs/sweep_window_1/thin512_win128_allfuse8.py"


def count_params(module) -> int:
    state = nnx.state(module, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(state))


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    summ_model = cfg.build_summformer(nnx.Rngs(0))
    results = [(f"summformer {config_path.stem}", count_params(summ_model))]

    for name, kw in [("small", dict(n_layer=12, n_head=12, n_embd=768)),
                      ("medium", dict(n_layer=24, n_head=16, n_embd=1024))]:
        mc = ModelConfig(pos_method="rope", block_size=1024, vocab_size=50304, **kw)
        gpt_model = Model(mc, rngs=nnx.Rngs(0))
        results.append((f"gpt2-{name}", count_params(gpt_model)))

    for name, n in results:
        print(f"{name}: {n:,} params ({n / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
