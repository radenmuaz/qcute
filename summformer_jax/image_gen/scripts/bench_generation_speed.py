"""Real per-token generation speed for a given image_gen config, via generate_kv_cache_fully_static
(not yet jax.jit-wrapped end-to-end for fuse-stage-having configs -- see docs/image_gen_design.md).

    uv run python summformer_jax/image_gen/scripts/bench_generation_speed.py --config summformer_jax/image_gen/configs/image64_mixdim.py --n-tokens 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax
from flax import nnx

from summformer import SummTransformerV2
from train import load_config_module, build_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--n-tokens", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg_vars = load_config_module(args.config)
    cfg = build_config(cfg_vars)
    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    prompt = jax.random.randint(jax.random.PRNGKey(args.seed), (args.prompt_len,), 0, model.vocab)

    t0 = time.time()
    out = model.generate_kv_cache_fully_static(prompt, args.n_tokens, key=jax.random.PRNGKey(args.seed + 1))
    jax.block_until_ready(out)
    dt = time.time() - t0
    per_token = dt / args.n_tokens
    print(f"{args.n_tokens} tokens in {dt:.2f}s = {per_token*1000:.1f}ms/token", flush=True)

    full_new = cfg.context_len - args.prompt_len
    est = per_token * full_new
    print(f"estimated full image ({full_new} new tokens): {est:.1f}s = {est/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
