"""CPU-only (safe next to a live TPU job): load a run_lagcodec_res_denoise checkpoint and compare cascade generation
with different numbers of generation cycles (gen_level_cycles) on the same val/train prompts -- separates what the
cycles add at inference from the rest of the model. Greedy, generate-mode encode (same as gen-eval).
Usage: python3 -m image_lagcodec.scripts.denoise_cycle_eval <run_name> [--cycles 1,2,3] [--n 8] [--top -1] [--ckpt NAME]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import dataclasses
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = jnp.repeat(k, rep, 1), jnp.repeat(v, rep, 1)
    lg = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)
    T = q.shape[2]
    t, s = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
    m = (s <= t + lookahead) & ((s >= t - window) if window is not None else True)
    lg = jnp.where(m[None, None], lg, -jnp.inf)
    if sink is not None:
        sk = jnp.broadcast_to(sink[None, :, None, None].astype(lg.dtype), lg.shape[:3] + (1,))
        w = jax.nn.softmax(jnp.concatenate([lg, sk], -1), -1)[..., :-1]
    else:
        w = jax.nn.softmax(lg, -1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense
import image_lagcodec.run_lagcodec_res_denoise as R
R.splash_attention = _dense
jax.config.update("jax_default_matmul_precision", "highest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cycles", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--top", type=int, default=-1, help="-1 = top level")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint dir name (default: latest)")
    a = ap.parse_args()
    run_dir = REPO_ROOT / "image_lagcodec/logs" / a.run
    ck = run_dir / "checkpoints" / a.ckpt if a.ckpt else R.find_latest_checkpoint(run_dir)
    cv = R.load_config_module(run_dir / f"config_{a.run}.py")
    cv.pop("label_fn", None)
    base = R.Config(**{k: cv[k] for k in R.CONFIG_FIELDS if k in cv})
    n_levels = len(base.strides)
    top = n_levels - 1 if a.top < 0 else a.top
    (train_np, _), (val_np, _) = R.load_dataset(cv.get("dataset", "cifar"), Path(cv.get("data_root") or REPO_ROOT / "datasets"),
                                                base.img_size if base.modality == "image" else None, cfg=base)
    po = R.pixel_order_for(base)
    print(f"run={a.run} ckpt={ck.name} top={top} n={a.n} cycles trained={base.level_cycles} mode={base.level_cycle_mode}")
    for cyc in a.cycles:
        cfg = dataclasses.replace(base, gen_level_cycles=(cyc,) * n_levels)
        model = eqx.tree_deserialise_leaves(ck / "model.eqx", R.LagCodecModel(jax.random.PRNGKey(0), cfg))
        model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model)
        for name, imgs in (("val", val_np[:a.n]), ("train", train_np[:a.n])):
            flat = jnp.array(R.images_to_positions(imgs, cfg, po))
            raw = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
            for i in range(top + 1):
                raw = R.encode_pardec_downsampler_generate(model.codelm_for(i), model.downsampler_for(i), raw, model.K(i),
                                                           cfg, rate_id=model.bos_rate_id(i), rng=jax.random.PRNGKey(i),
                                                           greedy=True, codelm_rate_id=model.codelm_bos_rate_id(i),
                                                           downsampler_ncodes=cfg.downsampler_ncodes[i])["code_idx"]
            cur = raw
            for i in range(top, -1, -1):
                cur = R.decode_generate_cycles(model, i, cur, cfg.upsampler_ncodes[i], greedy=True)
            img = R.positions_to_image(np.asarray(cur), cfg, po)
            print(f"  gen_level_cycles={cyc} {name}: gen_cascade_mse={R.pixel_mse(img, imgs.astype(np.uint8)):.1f} "
                  f"byte_acc={float(jnp.mean(cur == flat)):.4f}", flush=True)


if __name__ == "__main__":
    main()
