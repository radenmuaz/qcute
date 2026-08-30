"""Prefill + 1-token decode timing for the EAGER, naive-growing-cache incremental stepper
(_make_incremental_stepper / generate_kv_cache, NOT the jitted fully-static path) -- isolates
whether the fully-static architecture's own complexity (circular-buffer indexing, lax.cond
branches, code-LM cache bookkeeping) is adding cost beyond plain op-count, by comparing against
this simpler eager baseline. Companion to gpt2_jax/scripts/bench_generation_speed.py's identical
prefill+1-decode-step methodology.

    uv run python summformer_jax/image_gen/scripts/bench_prefill_decode1.py --config summformer_jax/image_gen/configs/image64_mixdim.py
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
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg_vars = load_config_module(args.config)
    cfg = build_config(cfg_vars)
    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    prompt = jax.random.randint(jax.random.PRNGKey(args.seed), (1, args.prompt_len), 0, model.vocab)

    import jax.numpy as jnp
    import statistics

    step = model._make_incremental_stepper(1)

    t0 = time.time()
    logits = step(prompt, 0)
    jax.block_until_ready(logits)
    t_prefill = time.time() - t0

    all_tokens = prompt
    next_logits = logits[:, -1, :]
    times = []
    for i in range(args.n_steps):
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
        t = time.time()
        logits2 = step(next_token, all_tokens.shape[1] - 1)
        jax.block_until_ready(logits2)
        times.append(time.time() - t)
        next_logits = logits2[:, -1, :]

    print(f"prefill ({args.prompt_len} tokens): {t_prefill*1000:.1f}ms", flush=True)
    print(f"decode steps 1..{args.n_steps} (ms, eager, growing cache): "
          f"{[round(t*1000, 1) for t in times]}", flush=True)
    print(f"median (excl. step 1): {statistics.median(times[1:])*1000:.1f}ms/token, "
          f"mean (excl. step 1): {statistics.mean(times[1:])*1000:.1f}ms/token", flush=True)


if __name__ == "__main__":
    main()
