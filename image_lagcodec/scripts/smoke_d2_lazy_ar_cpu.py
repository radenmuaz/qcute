"""CPU-only smoke test for d2_lazy_ar.py's generation path, using RANDOM synthetic input (no real dataset
needed) -- times decode_generate_interleave's compile at real config scale (n_blocks=256/64, decoder_d_model=
512, decoder_ncodes=n_blocks single-group). Meant to catch compile-time regressions before a real TPU launch.

Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.smoke_d2_lazy_ar_cpu \
    --config image_lagcodec/configs/d2_lazy_ar.py
"""
import argparse
import dataclasses
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_config_module, images_to_positions, pixel_order_for, code_embed_proj,
)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--n_images", type=int, default=1)
    args = p.parse_args()

    cv = load_config_module(Path(args.config))
    valid = {f.name for f in dataclasses.fields(Config)}
    cfg_kwargs = {k: v for k, v in cv.items() if k in valid}
    cfg_kwargs["precision"] = "fp32"  # CPU: avoid bf16 slowness/precision issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = Config(**cfg_kwargs)

    print(f"building model (level={args.level}, decoder_ncodes={cfg.decoder_ncodes[args.level]}, "
          f"interleave_decode={cfg.interleave_decode[args.level]}, "
          f"decoder_d_model={cfg.decoder_d_model[args.level]})...")
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    lev = model.levels[args.level]

    print("generating RANDOM synthetic input images (no real dataset)...")
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (args.n_images, cfg.img_size, cfg.img_size, 3)).astype(np.uint8)
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
    x = code_embed_proj(flat, lev.own_input_embed, lev.own_input_proj)
    enc = lev.encode(x, flat, rng=None)
    G = cfg.decoder_ncodes[args.level]
    print(f"n_blocks={enc['code_idx'].shape[1]}, G={G} -- running decode_generate (timed)...")

    t0 = time.time()
    gen = lev.decode_generate_pardec(enc["code_idx"], G, greedy=True, seed=0)
    t1 = time.time()
    print(f"\ngeneration (incl. compile) took {t1 - t0:.2f}s")
    print(f"output shape: {gen.shape}, range: [{int(gen.min())}, {int(gen.max())}]")


if __name__ == "__main__":
    main()
