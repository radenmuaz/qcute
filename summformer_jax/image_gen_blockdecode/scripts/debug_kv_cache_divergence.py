"""Finds the FIRST token position where generate_kv_cache diverges from generate_no_cache's
reference trajectory, for a known-failing case, plus logit diffs at that position -- pinpoints
which fuse-stage/mechanism is at fault instead of just re-confirming match_rate<1.0.

    uv run python summformer_jax/image_gen/scripts/debug_kv_cache_divergence.py --config summformer_jax/image_gen/configs/image64_mixdim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax
import jax.numpy as jnp
from flax import nnx

from summformer import SummTransformerV2
from train import load_config_module, build_config


def main():
    cfg_vars = load_config_module(Path("summformer_jax/image_gen/configs/image64_mixdim.py"))
    cfg = build_config(cfg_vars)
    cfg = cfg.__class__(**{**cfg.__dict__, "compute_dtype": jnp.float32})
    model = SummTransformerV2(cfg, rngs=nnx.Rngs(0))

    key = jax.random.PRNGKey(0)
    n_checks, prompt_len_base, n_new = 4, 17, 18
    for i in range(n_checks):
        pl = max(1, prompt_len_base - i * (prompt_len_base // n_checks))
        key, subkey = jax.random.split(key)
        prompt = jax.random.randint(subkey, (pl,), 0, model.vocab)

        out_full = model.generate_no_cache(prompt, n_new)
        out_cache = model.generate_kv_cache(prompt, n_new)
        match = jnp.array_equal(out_full, out_cache)
        print(f"check {i}: prompt_len={pl} match={bool(match)}", flush=True)
        if not match:
            diff_positions = jnp.where(out_full != out_cache)[0]
            print(f"  diverges at absolute positions: {diff_positions.tolist()}")
            print(f"  out_full:  {out_full.tolist()}")
            print(f"  out_cache: {out_cache.tolist()}")
            first_diff = int(diff_positions[0])
            print(f"  first divergence at position {first_diff} (prompt_len={pl}, "
                  f"so this is generated-token index {first_diff - pl})")


if __name__ == "__main__":
    main()
