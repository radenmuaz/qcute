"""CPU-only (safe next to a live TPU job): shape of the upsampler's first-digit distribution P(x | code) for the
first token of each row (it sees only the context code), aligned on the code value c. Compares the average predicted
profile P(c+d) with the empirical histogram of (true x - c), how spiky single-row distributions are, and where the
argmax / mean land, for hard-argmax codes (generation) and sampled codes (training).
Usage: python3 -m image_lagcodec.scripts.audit_output_profile <run> [--module run_lagcodec_res] [--ckpt DIR] [--n 64] [--level 0]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import importlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import image_lagcodec.scripts.audit_tf_vs_gen_timesteps as A
from image_lagcodec.scripts.audit_decode_rule import digit_logits

REPO_ROOT = A.REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--level", type=int, default=0)
    a = ap.parse_args()
    R = A.R = importlib.import_module(f"image_lagcodec.{a.module}")
    if hasattr(R, "splash_attention"):
        R.splash_attention = A._dense
    run_dir = REPO_ROOT / "image_lagcodec/logs" / a.run
    ck = Path(a.ckpt) if a.ckpt else R.find_latest_checkpoint(run_dir)
    cv = R.load_config_module(run_dir / f"config_{a.run}.py")
    label_fn = getattr(R, cv.pop("label_fn", "default_label_fn_jax"))
    cfg = R.Config(**{k: cv[k] for k in R.CONFIG_FIELDS if k in cv})
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", R.LagCodecModel(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model)
    (train_np, _), _ = R.load_cifar10(Path(cv.get("data_root") or REPO_ROOT / "datasets"))
    po = R.pixel_order_for(cfg)
    imgs = train_np[:a.n]
    flat = jnp.array(R.images_to_positions(imgs, cfg, po))
    tok0 = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
    raw, tgt = tok0, tok0
    lvl = a.level
    for i in range(lvl + 1):
        enc = lambda rng: R.encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, tgt, flat, cfg, po, label_fn,
                                                      model.K(i), rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
                                                      rng=rng, downsampler_ncodes=cfg.downsampler_ncodes[i])
        ev = enc(None)
        if i < lvl:
            raw, tgt = ev["code_soft"], ev["code_idx"]
    T = np.asarray(tok0 if lvl == 0 else tgt)
    tr = enc(jax.random.PRNGKey(100))
    K = model.K(lvl)
    n_pass, Pp, fill, p1_len = A.pass_plan(cfg, lvl)
    V = cfg.code_vocab[lvl]
    vals = np.arange(V)
    x = T.reshape(T.shape[0], -1, K, T.shape[-1])[:, :, 0, 0].reshape(-1)  # true first digit of each row's first token
    print(f"run={a.run} level={lvl} rows={x.shape[0]} (first token of each row, digit 0, pass 1: sees only its context code)")
    D = np.arange(-16, 17)
    for nm, code in (("hard argmax code (generation)", np.asarray(ev["code_idx"])), ("hard sampled code (training)", np.asarray(tr["code_idx"]))):
        hid = A.hidden(model, lvl, jnp.asarray(T), jnp.asarray(code), None, p1_len, "mask")
        lg = np.asarray(digit_logits(model, lvl, hid, jnp.asarray(T)))
        lg = lg.reshape(lg.shape[0], -1, K, lg.shape[-2], V)[:, :, 0, 0].reshape(-1, V).astype(np.float64)
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        c = code[..., 0].reshape(-1)
        am, mean = p.argmax(-1), (p * vals).sum(-1)
        sd = np.sqrt((p * (vals - mean[:, None]) ** 2).sum(-1))
        ker = np.exp(-0.5 * ((vals[:, None] - vals[None, :]) / 2.0) ** 2)
        ker /= ker.sum(-1, keepdims=True)
        rough = np.abs(p - p @ ker.T).sum(-1)
        idx = np.clip(c[:, None] + D[None, :], 0, V - 1)
        prof = np.take_along_axis(p, idx, 1).mean(0)
        emp = np.array([((x - c) == d).mean() for d in D])
        print(f"\n[{nm}]")
        print(f"  true x - c: std={np.std(x - c):.2f} | argmax - c: std={np.std(am - c):.2f} mean={np.mean(am - c):+.2f} | mean - c: std={np.std(mean - c):.2f} mean={np.mean(mean - c):+.2f} | "
              f"predictive std={sd.mean():.2f} | argmax == c exactly: {float((am == c).mean()):.3f} | true x == c: {float((x == c).mean()):.3f}")
        print(f"  mse vs true: argmax={np.mean((am - x) ** 2):.1f} mean={np.mean((mean - x) ** 2):.1f} code itself={np.mean((c - x) ** 2):.1f} | "
              f"corr(argmax - c, x - c)={np.corrcoef(am - c, x - c)[0, 1]:+.3f} corr(mean - c, x - c)={np.corrcoef(mean - c, x - c)[0, 1]:+.3f}")
        print(f"  roughness (mass moved by a sigma=2 smoothing; 0 = smooth): mean={rough.mean():.3f} | max_prob mean={p.max(-1).mean():.3f} | "
              f"p(argmax) / smoothed p at the argmax: {float((p.max(-1) / np.take_along_axis(p @ ker.T, am[:, None], 1)[:, 0]).mean()):.2f}")
        print(f"  d = -16..16 | predicted profile mean P(c+d): {A.fmt(prof)}")
        print(f"              | empirical freq of x-c=d     : {A.fmt(emp)}")
        print(f"  mass within |d|<=16: predicted={prof.sum():.3f} empirical={emp.sum():.3f}")
        for r in (0, 777, 5000):
            if r < p.shape[0]:
                top = np.argsort(-p[r])[:10]
                print(f"  row {r}: code c={int(c[r])} true x={int(x[r])} top-10 value:prob = {' '.join(f'{int(v)}:{p[r, v]:.3f}' for v in top)}", flush=True)


if __name__ == "__main__":
    main()
