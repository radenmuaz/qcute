"""CPU-mode launcher for run_lagcodec_stack.py: monkeypatches eqx_common.splash_attention with a
plain dense causal-softmax reference (splash's Pallas kernel has no CPU interpret path wired up
here), same pattern as pardec_v2_consistency_check.py/kv_consistency_check.py. For CPU smoke/dev
runs only -- production TPU runs use run_lagcodec_stack.py directly (real splash kernel).

Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.run_stack_cpu -- --config <path> ...
(all args after -- are forwarded to run_lagcodec_stack.main())
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

import jax
import jax.numpy as jnp

import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            q_idx, kv_idx = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
            if window is None and lookahead <= 0:
                mask = q_idx >= kv_idx
            else:
                mask = jnp.ones((T, T), dtype=bool)
                if window is not None:
                    mask = mask & (q_idx - window <= kv_idx)
                mask = mask & (q_idx + max(0, lookahead) >= kv_idx)
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)
    eqx_common.splash_attention = _cpu_dense_attention

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    sys.argv = [sys.argv[0]] + args
    from image_lagcodec.run_lagcodec_stack import main
    main()
