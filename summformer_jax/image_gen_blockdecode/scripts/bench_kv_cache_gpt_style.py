"""Prefill + 1 jitted decode step, gpt2_jax-style (plain growing/self-truncating KV cache, pure
functions, nnx.jit on the decode step only) -- apples-to-apples comparison against
gpt2_jax/scripts/bench_generation_speed.py's identical methodology, using image64_nofuse.py
(trunk only, no fuse_stages) so both models are self-attn-only.

    uv run python summformer_jax/image_gen/scripts/bench_kv_cache_gpt_style.py --config summformer_jax/image_gen/configs/image64_nofuse.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax
import jax.numpy as jnp
from flax import nnx

from summformer import SummTransformerV2, _jitted_decode_step_pure_growing
from train import load_config_module, build_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg_vars = load_config_module(args.config)
    cfg = build_config(cfg_vars)
    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    prompt = jax.random.randint(jax.random.PRNGKey(args.seed), (1, args.prompt_len), 0, model.vocab)

    t_pf0 = time.time()
    logits, caches = model._prime_pure_growing(prompt)
    jax.block_until_ready(logits)
    t_prefill = time.time() - t_pf0

    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    pos_scalar = jnp.array(args.prompt_len, dtype=jnp.int32)

    t_d0 = time.time()
    logits2, caches2 = _jitted_decode_step_pure_growing(model, next_token, pos_scalar, caches)
    jax.block_until_ready(logits2)
    t_decode1 = time.time() - t_d0  # includes one-time compile

    t_d1 = time.time()
    logits3, caches3 = _jitted_decode_step_pure_growing(model, next_token, pos_scalar, caches)
    jax.block_until_ready(logits3)
    t_decode2 = time.time() - t_d1  # steady-state, same cache shape (repeat call, not advanced)

    print(f"prefill ({args.prompt_len} tokens): {t_prefill*1000:.1f}ms", flush=True)
    print(f"decode step 1 (incl. compile): {t_decode1*1000:.1f}ms", flush=True)
    print(f"decode step 2 (steady-state): {t_decode2*1000:.1f}ms "
          f"(n_layers={cfg.n_layers}, d_model={cfg.d_model}, n_fuse_stages={len(cfg.fuse_stages)})", flush=True)


if __name__ == "__main__":
    main()
