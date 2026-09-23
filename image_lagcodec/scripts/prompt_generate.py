"""Prompted generation through an encoder's own NTP head, CPU only (sets JAX_PLATFORMS=cpu; never touches a TPU).
Given the first `prompt_frac` of an image's bytes (a long real-byte warmup avoids the free-run collapsing), free-run
the encoder of a top-two level (greedy and/or sampled), encode the completed sequence back to the top, and decode the
emitted codes down the cascade. Writes samples_prompt_L{level}_p{P}_{mode}_{decode_from}.png into the run's log dir.
Usage: python3 -m image_lagcodec.scripts.prompt_generate <run_name> [--levels 1,0] [--prompt_fracs 0.5,0.75]
       [--modes greedy,sample] [--temperature 0.9] [--top_k 40] [--decode_sample] [--n_img N] [--tag prompt]
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import dataclasses
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common
from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, dataset_from_config, images_to_positions, pixel_order_for, positions_to_image, load_config_module,
    CONFIG_FIELDS, save_compare_grid, pixel_mse, generate_from_prompt,
)


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    # CPU has no splash kernel: dense causal attention (sink is not modelled by this stand-in, as in the other CPU checks)
    n = q.shape[1] // k.shape[1]
    if n > 1:
        k, v = jnp.repeat(k, n, 1), jnp.repeat(v, n, 1)
    lg = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
    T = q.shape[2]
    m = jnp.arange(T)[:, None] >= jnp.arange(T)[None, :]
    if window is not None:
        m = m & (jnp.arange(T)[:, None] - window <= jnp.arange(T)[None, :])
    if sink is not None:
        s = jnp.broadcast_to(sink[None, :, None, None], lg.shape[:3] + (1,))
        w = jax.nn.softmax(jnp.concatenate([jnp.where(m[None, None], lg, -1e9), s.astype(lg.dtype)], -1), -1)[..., :-1]
    else:
        w = jax.nn.softmax(jnp.where(m[None, None], lg, -1e9), -1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense


def scalar(v, i=-1):
    return v[i] if isinstance(v, (list, tuple)) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--levels", default="1,0")
    ap.add_argument("--prompt_fracs", default="0.5,0.75")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--decode_sample", action="store_true", help="sample (not argmax) in the decoder cascade too")
    ap.add_argument("--decode_temperature", type=float, default=0.8)
    ap.add_argument("--decode_top_k", type=int, default=0, help="top-k of the decoder cascade when --decode_sample")
    ap.add_argument("--n_img", type=int, default=None)
    ap.add_argument("--tag", default="prompt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run_dir = REPO_ROOT / f"image_lagcodec/logs/{a.run}"
    ck = sorted((run_dir / "checkpoints").iterdir())[-1]
    cv = load_config_module(run_dir / f"config_{a.run}.py")
    cv.pop("label_fn", None)
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    cfg = dataclasses.replace(cfg, gen_top_k=a.decode_top_k)
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    n_img = a.n_img or int(scalar(cv.get("val_batch_size", 8)))
    enc_t = float(scalar(cv.get("encode_temperature", 1.0)))
    (_, _), (val_np, _) = dataset_from_config(cv, REPO_ROOT)
    pixel_order = pixel_order_for(cfg)
    imgs = val_np[:n_img]
    fb = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    T0 = fb.shape[1]
    K0 = model.levels[0].K
    print(f"run={a.run} ckpt={ck.name} backend={jax.default_backend()} n_img={n_img} T0={T0}", flush=True)

    for L in [int(x) for x in a.levels.split(",")]:
        for frac in [float(x) for x in a.prompt_fracs.split(",")]:
            P = max(K0, int(frac * T0) // K0 * K0)
            for mode in a.modes.split(","):
                greedy = mode == "greedy"
                t0 = time.monotonic()
                res = generate_from_prompt(
                    model, cfg, fb[:, :P], T0, L, jax.random.PRNGKey(a.seed), greedy=greedy,
                    temperature=a.temperature, top_k=a.top_k, encode_temperature=enc_t,
                    decode_greedy=not a.decode_sample, decode_temperature=a.decode_temperature, decode_seed=a.seed)
                for src in ("emitted", "sampled"):
                    rec = np.asarray(res[src])
                    img = positions_to_image(rec, cfg, pixel_order)
                    gt = imgs.astype(np.uint8)
                    name = f"samples_{a.tag}_L{L}_p{P}_{mode}_{src}.png"
                    save_compare_grid(img, gt, run_dir / name)
                    gtb = np.asarray(fb)
                    cont = rec[:, P:]
                    top_freq = max(float((cont == v).mean()) for v in np.unique(cont)[:256]) if cont.size else 0.0
                    print(f"[L{L} p={P} {mode:6s} {src:7s}] prompt_mse={pixel_mse(rec[:, :P].astype(np.float64), gtb[:, :P]):8.2f} "
                          f"cont_mse={pixel_mse(rec[:, P:].astype(np.float64), gtb[:, P:]):8.2f} "
                          f"most_common_byte_frac={top_freq:.3f} -> {name}", flush=True)
                print(f"   ({time.monotonic() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
