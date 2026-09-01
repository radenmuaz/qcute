"""Sanity check + per-token generation speed for gpt2_jax's jitted KV-cache decode step
(Model.generate_kv_cache_jit / _decode_step_pure in model_gpt.py) -- baseline comparison point
for summformer_jax/image_gen's own generation-speed numbers (674ms/token on tpu8 for a much
smaller model with 2 fuse-stages; this isolates plain self-attn-only KV-cache decode cost with a
known-good architecture).

    uv run python gpt2_jax/scripts/bench_generation_speed.py --model tiny --n-tokens 40
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

from model_gpt import Model, ModelConfig
from train_gpt import MODEL_SHAPES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_SHAPES), default="tiny")
    p.add_argument("--pos-method", default="rope")
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = ModelConfig(pos_method=args.pos_method, **MODEL_SHAPES[args.model])
    model = Model(cfg, rngs=nnx.Rngs(args.seed))
    prompt = jax.random.randint(jax.random.PRNGKey(args.seed), (1, args.prompt_len), 0, cfg.vocab_size)

    # sanity check: prefill + exactly 1 decode step must match full-recompute reference.
    t_pf0 = time.time()
    logits, caches = model._prime_pure(prompt)
    jax.block_until_ready(logits)
    t_prefill = time.time() - t_pf0

    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    from model_gpt import _jitted_decode_step_pure

    t_d0 = time.time()
    logits2, caches2 = _jitted_decode_step_pure(model, next_token, prompt.shape[1], caches)
    jax.block_until_ready(logits2)
    t_decode1 = time.time() - t_d0  # includes one-time compile

    t_d1 = time.time()
    logits3, caches3 = _jitted_decode_step_pure(model, next_token, prompt.shape[1], caches)
    jax.block_until_ready(logits3)
    t_decode2 = time.time() - t_d1  # steady-state, same cache shape (S=prompt_len+1 both times)

    ref = model.generate_no_cache(prompt[0], 1)
    got = jnp.concatenate([prompt[0], jnp.argmax(logits2[:, -1, :], axis=-1)])
    match = bool(jnp.array_equal(ref, got))
    print(f"sanity check (prefill + 1 decode step vs. full recompute): match={match}", flush=True)
    assert match, "generate_kv_cache_jit diverged from generate_no_cache reference"

    print(f"prefill ({args.prompt_len} tokens): {t_prefill*1000:.1f}ms", flush=True)
    print(f"decode step 1 (incl. compile): {t_decode1*1000:.1f}ms", flush=True)
    print(f"decode step 2 (same shape, steady-state): {t_decode2*1000:.1f}ms "
          f"(model={args.model}, n_layer={cfg.n_layer}, n_embd={cfg.n_embd})", flush=True)


if __name__ == "__main__":
    main()
