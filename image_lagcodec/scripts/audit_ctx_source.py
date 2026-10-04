"""CPU-only (safe next to a live TPU job): does the upsampler decode differently when its context code is given the
way TRAINING gives it (quantize_dispatch with rng: a sampled digit, or the soft probability vector for a
quantize_drop fraction of digits) vs the way GENERATION gives it (hard argmax index)? Per level, same downsampler
logits, only the context representation changes; reports teacher-forced and free-generation mse.
Usage: python3 -m image_lagcodec.scripts.audit_ctx_source <run> [--module run_lagcodec_res] [--ckpt DIR] [--n 64]
       [--levels 0,1,2] [--splits train,val]
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

REPO_ROOT = A.REPO_ROOT


def _tf_soft(model, lvl, target, ctx_soft):
    outs = A.R.decode_logits_and_target_multipass(model, lvl, target, ctx_soft, model.cfg.upsampler_ncodes[lvl], rng=None,
                                                  force_teacher_forced=True, return_passes=True)
    return outs[-1][0]


tf_soft = eqx.filter_jit(_tf_soft)


def gen_soft(model, lvl, ctx):
    # free generation, every refine pass, context given as-is (int index or soft vector)
    R, cfg = A.R, model.cfg
    n_pass, Pp, fill, p1_len = A.pass_plan(cfg, lvl)
    nc = cfg.upsampler_ncodes[lvl]
    out = R._decode_generate_pardec_jit(model, lvl, ctx, nc, True, 1.0, 0, None, Pp, "mask") if p1_len else \
        R._decode_generate_pardec_jit(model, lvl, ctx, nc, True, 1.0, 0)
    for _ in range(n_pass - 1):
        out = R._decode_generate_pardec_jit(model, lvl, ctx, nc, True, 1.0, 0, out, Pp, fill)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--splits", default="train,val")
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
    (train_np, _), (val_np, _) = R.load_cifar10(Path(cv.get("data_root") or REPO_ROOT / "datasets"))
    po = R.pixel_order_for(cfg)
    print(f"run={a.run} module={a.module} ckpt={ck} n={a.n} quantize_mode={cfg.quantize_mode} quantize_drop={cfg.quantize_drop} "
          f"gumbel_at_inference={cfg.gumbel_at_inference} encode_temperature={cv.get('encode_temperature', 1.0)}", flush=True)
    for name in a.splits.split(","):
        imgs = (train_np if name == "train" else val_np)[:a.n]
        B = imgs.shape[0]
        flat = jnp.array(R.images_to_positions(imgs, cfg, po))
        tok0 = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
        print(f"\n================ split={name} n={B} ================", flush=True)
        raw, tgt = tok0, tok0
        for lvl in range(max(a.levels) + 1):
            cl, ds = model.codelm_for(lvl), model.downsampler_for(lvl)
            enc = lambda rng: R.encode_pardec_downsampler(cl, ds, raw, tgt, flat, cfg, po, label_fn, model.K(lvl),
                                                          rate_id=model.bos_rate_id(lvl), codelm_rate_id=model.codelm_bos_rate_id(lvl),
                                                          rng=rng, downsampler_ncodes=cfg.downsampler_ncodes[lvl])
            ev = enc(None)  # eval / generation: greedy digits, hard one-hot
            tr = [enc(jax.random.PRNGKey(100 + s)) for s in range(2)]  # training: sampled digits, soft for a quantize_drop fraction
            V = cfg.code_vocab[lvl]
            idx_ev = np.asarray(ev["code_idx"])
            p_ev = np.asarray(jax.nn.softmax(ev["logits"].astype(jnp.float32), -1)).astype(np.float64)
            vals = np.arange(V)
            mean_ev = (p_ev * vals).sum(-1)
            sd_ev = np.sqrt((p_ev * (vals - mean_ev[..., None]) ** 2).sum(-1))
            ent = -(p_ev * np.log(np.maximum(p_ev, 1e-12))).sum(-1)
            cs_tr = np.asarray(tr[0]["code_soft"]).astype(np.float64)
            soft_frac = float((cs_tr.max(-1) < 0.999).mean())
            lab = np.asarray(label_fn(flat, cfg, po, idx_ev.shape[1], cfg.pq_chunks[lvl], V))
            print(f"\n[level {lvl}] downsampler digit distribution: entropy={ent.mean():.3f} nats std={sd_ev.mean():.2f} max_prob={p_ev.max(-1).mean():.3f} | "
                  f"mse vs label: argmax={A.vmse(idx_ev, lab):.2f} mean={A.vmse(mean_ev, lab):.2f} train sample={A.vmse(np.asarray(tr[0]['code_idx']), lab):.2f} | "
                  f"train-mode code_soft: fraction of digits that are SOFT vectors={soft_frac:.3f} | "
                  f"argmax vs train sample digits equal={float((idx_ev == np.asarray(tr[0]['code_idx'])).mean()):.3f}", flush=True)
            T = tok0 if lvl == 0 else tgt
            Tn = np.asarray(T)
            rs = cfg.upsampler_ncodes[lvl] * model.K(lvl)
            oh = lambda i: jax.nn.one_hot(jnp.asarray(i), V, dtype=jnp.float32)
            rng_np = np.random.default_rng(0)
            mixmask = rng_np.random(idx_ev.shape) < cfg.quantize_drop
            variants = [
                ("generation: hard argmax", oh(idx_ev)),
                ("hard train sample (seed a)", oh(np.asarray(tr[0]["code_idx"]))),
                ("training code_soft (seed a)", jnp.asarray(tr[0]["code_soft"]).astype(jnp.float32)),
                ("training code_soft (seed b)", jnp.asarray(tr[1]["code_soft"]).astype(jnp.float32)),
                ("all digits soft p (eval logits)", jnp.asarray(p_ev).astype(jnp.float32)),
                ("argmax, quantize_drop frac soft p", jnp.where(jnp.asarray(mixmask)[..., None], jnp.asarray(p_ev).astype(jnp.float32), oh(idx_ev))),
                ("hard label_fn target", oh(lab)),
                ("hard round(mean of p)", oh(np.clip(np.rint(mean_ev), 0, V - 1).astype(np.int32))),
            ]
            print(f"  decode level {lvl} (target mse; A = teacher-forced digits+row, D = free generation); row pos x digit mse for D")
            for nm, ctx in variants:
                a_pred = np.asarray(jnp.argmax(tf_soft(model, lvl, T, ctx), -1))
                d_pred = gen_soft(model, lvl, ctx)
                print(f"    ctx = {nm:34s}: A mse={A.vmse(a_pred, Tn):8.2f}  D mse={A.vmse(d_pred, Tn):8.2f}  D by AR step {A.fmt(A.mse_steps(d_pred, Tn, rs))}", flush=True)
            raw, tgt = ev["code_soft"], ev["code_idx"]


if __name__ == "__main__":
    main()
