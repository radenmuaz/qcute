"""CPU-only: encodes real images with a level's own encoder and plots GT | target downsample
(from label_fn) | pred downsample (code_idx read directly as RGB bytes) side by side. Thin wrapper
around run_lagcodec.plot_encoder_outs -- the same function the training loop calls automatically at
every gen_eval_every_step (see run_lagcodec.py's periodic eval block), so this script is only needed
for ad-hoc/offline checks against an arbitrary checkpoint.
Usage: python3 -m image_lagcodec.scripts.plot_encoder_outs <run_name> [--level 0] [--n_img N]
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import dataclasses
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common
from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, dataset_from_config, pixel_order_for, load_config_module,
    CONFIG_FIELDS, plot_encoder_outs, default_label_fn_jax,
)


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
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


def scalar(v, i=0):
    return v[i] if isinstance(v, (list, tuple)) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n_img", type=int, default=None)
    a = ap.parse_args()

    run_dir = REPO_ROOT / f"image_lagcodec/logs/{a.run}"
    ck = sorted((run_dir / "checkpoints").iterdir())[-1]
    cv = load_config_module(run_dir / f"config_{a.run}.py")
    label_fn = cv.pop("label_fn", default_label_fn_jax)
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)

    n_img = a.n_img or int(scalar(cv.get("val_batch_size", 8)))
    (_, _), (val_np, _) = dataset_from_config(cv, REPO_ROOT)
    pixel_order = pixel_order_for(cfg)
    imgs = val_np[:n_img]
    print(f"run={a.run} ckpt={ck.name} backend={jax.default_backend()} n_img={n_img} "
          f"level={a.level} img_size={cfg.img_size}", flush=True)

    out_path = run_dir / f"samples_offline_level{a.level}_codegrid.png"
    util = plot_encoder_outs(model, cfg, imgs, pixel_order, out_path, level=a.level, label_fn=label_fn)
    print(f"util={util:.3f}" if util is not None else "skipped (pq_chunks/code_vocab != 3/256)", flush=True)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
