"""Real, full end-to-end single-image generation timing (not extrapolated from a short burst) --
runs the ACTUAL production generate call for the full context_len, for one of two paths:
"eager" (generate_kv_cache, eager growing cache, now fixed-shape code-LM recompute) or "static"
(generate_kv_cache_fully_static, jitted fully-static circular-buffer path).

    uv run python summformer_jax/image_gen/scripts/bench_full_image.py --config summformer_jax/image_gen/configs/image64_mixdim.py --path eager
    uv run python summformer_jax/image_gen/scripts/bench_full_image.py --config summformer_jax/image_gen/configs/image64_mixdim.py --path static
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
from tqdm import tqdm

from summformer import SummTransformerV2, _jitted_decode_step_pure
from train import load_config_module, build_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--path", choices=["eager", "static"], required=True)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg_vars = load_config_module(args.config)
    cfg = build_config(cfg_vars)
    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    prompt = jax.random.randint(jax.random.PRNGKey(args.seed), (args.prompt_len,), 0, model.vocab)
    n_new = cfg.context_len - args.prompt_len

    def run_eager_pass(prompt):
        if prompt.ndim == 1:
            prompt = prompt[None]
        step = model._make_incremental_stepper(prompt.shape[0])
        all_tokens = prompt
        logits_all = step(all_tokens, 0)
        next_logits = logits_all[:, -1, :]
        for _ in tqdm(range(n_new), smoothing=0.05):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits_all = step(next_token, all_tokens.shape[1] - 1)
            next_logits = logits_all[:, -1, :]
        return all_tokens[0]

    def run_static_pass(prompt):
        if prompt.ndim == 1:
            prompt = prompt[None]
        state = model._init_decode_state(prompt.shape[0])
        logits, state = model._prime_pure(prompt, state)
        all_tokens = prompt
        next_logits = logits[:, -1, :]
        for _ in tqdm(range(n_new), smoothing=0.05):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits, state = _jitted_decode_step_pure(model, state, next_token, all_tokens.shape[1] - 1)
            next_logits = logits[:, -1, :]
        return all_tokens[0]

    run_pass = run_eager_pass if args.path == "eager" else run_static_pass

    # Pass 1: pays every shape's first-seen compile cost (cold). Pass 2: same live process,
    # every shape already jit-cached -- isolates the "repeated generation" steady-state cost
    # from the one-time warmup, per the "jit caches per shape" discussion.
    for pass_i in (1, 2):
        print(f"=== pass {pass_i} ({'cold' if pass_i == 1 else 'warm, all shapes cached'}) ===", flush=True)
        t0 = time.time()
        out = run_pass(prompt)
        jax.block_until_ready(out)
        dt = time.time() - t0
        print(f"DONE path={args.path} pass={pass_i}: {n_new} tokens in {dt:.1f}s = {dt/60:.2f} min "
              f"({dt/n_new*1000:.1f}ms/token avg)", flush=True)


if __name__ == "__main__":
    main()
