"""CPU-only (safe next to a live TPU job): free generation re-done with dense passes under a different per-digit
decode rule (argmax | mean | median | smoothed argmax), per level with real codes and as full cascades. The argmax
rule must reproduce the real generation code exactly. No retraining; shows how much of the generation error is the
argmax of a wide digit distribution.
Usage: python3 -m image_lagcodec.scripts.audit_decode_rule <run> [--module run_lagcodec_res] [--ckpt DIR] [--n 64]
       [--tops 0,1,2] [--rules argmax,mean,median,smooth4] [--splits train,val]
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


def _digit_logits(model, lvl, hid, digits):
    up = model.upsampler_for(lvl)
    R = A.R
    return R.token_ar_teacher_forced(up.token_in_proj, up.token_member_embed, up.token_norm1, up.token_attn,
                                     up.token_ln_f, up.token_out_head, up.token_dim, up.output_vocab, hid, digits)


digit_logits = eqx.filter_jit(_digit_logits)


def pick(logits, rule):
    # logits (B, L, V) for one digit -> chosen value per position
    lg = np.asarray(logits).astype(np.float64)
    if rule == "argmax":
        return lg.argmax(-1)
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    v = np.arange(p.shape[-1])
    if rule == "mean":
        return np.clip(np.rint((p * v).sum(-1)), 0, p.shape[-1] - 1).astype(np.int64)
    if rule == "median":
        return (np.cumsum(p, -1) < 0.5).sum(-1).clip(0, p.shape[-1] - 1)
    if rule.startswith("smooth"):
        s = float(rule[len("smooth"):])
        k = np.exp(-0.5 * ((v[:, None] - v[None, :]) / s) ** 2)
        return (p @ k).argmax(-1)
    raise ValueError(rule)


def decode_rule(model, lvl, ctx_idx, rule):
    # same schedule as decode_generate_multipass (refine passes, draft = previous pass), dense passes, digit rule = `rule`
    cfg = model.cfg
    n_pass, Pp, fill, p1_len = A.pass_plan(cfg, lvl)
    K = model.K(lvl)
    rs = cfg.upsampler_ncodes[lvl] * K
    B, n_ctx = ctx_idx.shape[0], ctx_idx.shape[1]
    L = n_ctx * K
    chunks = model.upsampler_for(lvl).output_chunks
    out = None
    for p in range(n_pass):
        kw = (None, p1_len, "mask") if p == 0 else (jnp.asarray(out).astype(jnp.int32), Pp, fill)
        cur = np.zeros((B, L, chunks), np.int32)
        for j in range(rs):
            hid = A.hidden(model, lvl, jnp.asarray(cur), ctx_idx, *kw)
            for m in range(chunks):
                lg = digit_logits(model, lvl, hid, jnp.asarray(cur))
                cur[:, j::rs, m] = pick(lg[:, j::rs, m, :], rule)
        out = cur
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tops", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--rules", default="argmax,mean,median,smooth4")
    ap.add_argument("--splits", default="train,val")
    a = ap.parse_args()
    R = A.R = importlib.import_module(f"image_lagcodec.{a.module}")
    if hasattr(R, "splash_attention"):
        R.splash_attention = A._dense
    run_dir = REPO_ROOT / "image_lagcodec/logs" / a.run
    ck = Path(a.ckpt) if a.ckpt else R.find_latest_checkpoint(run_dir)
    cv = R.load_config_module(run_dir / f"config_{a.run}.py")
    cv.pop("label_fn", None)
    cfg = R.Config(**{k: cv[k] for k in R.CONFIG_FIELDS if k in cv})
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", R.LagCodecModel(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model)
    (train_np, _), (val_np, _) = R.load_cifar10(Path(cv.get("data_root") or REPO_ROOT / "datasets"))
    po = R.pixel_order_for(cfg)
    rules = a.rules.split(",")
    print(f"run={a.run} module={a.module} ckpt={ck} n={a.n} tops={a.tops} rules={rules}", flush=True)
    for name in a.splits.split(","):
        imgs = (train_np if name == "train" else val_np)[:a.n]
        flat = jnp.array(R.images_to_positions(imgs, cfg, po))
        tok0 = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
        codes, raw = [], tok0
        for i in range(max(a.tops) + 1):
            o = R.encode_pardec_downsampler_generate(model.codelm_for(i), model.downsampler_for(i), raw, model.K(i), cfg,
                                                     rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
                                                     rng=jax.random.PRNGKey(i), greedy=True, temperature=cfg.gen_temperature,
                                                     top_k=cfg.gen_top_k, downsampler_ncodes=cfg.downsampler_ncodes[i])
            codes.append(o["code_idx"])
            raw = o["code_soft"]
        targets = [np.asarray(tok0)] + [np.asarray(c) for c in codes[:-1]]
        print(f"\n================ split={name} n={imgs.shape[0]} ================", flush=True)
        for lvl in range(max(a.tops) + 1):
            real = np.asarray(R.decode_generate_multipass(model, lvl, codes[lvl], cfg.upsampler_ncodes[lvl], greedy=True))
            rs = cfg.upsampler_ncodes[lvl] * model.K(lvl)
            rep = np.repeat(np.asarray(codes[lvl]), rs // cfg.upsampler_ncodes[lvl], axis=1)
            print(f"[level {lvl}] real codes as context: real generation mse={A.vmse(real, targets[lvl]):.2f} | "
                  f"repeat-the-code baseline mse={A.vmse(rep, targets[lvl]):.2f}")
            for rule in rules:
                out = decode_rule(model, lvl, codes[lvl], rule)
                extra = f" (== real generation: {float((out == real).mean()):.6f})" if rule == "argmax" else ""
                print(f"  rule={rule:8s} mse={A.vmse(out, targets[lvl]):8.2f} by AR step {A.fmt(A.mse_steps(out, targets[lvl], rs))}{extra}", flush=True)
        for top in a.tops:
            print(f"[cascade top={top}] pixel mse (and per-level mse vs the encoder's codes)")
            for rule in rules:
                cur, parts = codes[top], []
                for i in range(top, -1, -1):
                    out = decode_rule(model, i, cur, rule)
                    parts.append(f"L{i}={A.vmse(out, targets[i]):.1f}")
                    cur = jnp.asarray(out)
                img = R.positions_to_image(np.asarray(cur), cfg, po)
                print(f"  rule={rule:8s} pixel_mse={R.pixel_mse(img, imgs.astype(np.uint8)):8.2f}  {' '.join(parts)}", flush=True)
            rep = np.asarray(codes[top])
            for i in range(top, -1, -1):
                rep = np.repeat(rep, model.K(i), axis=1)
            print(f"  baseline: repeat the top code down to pixels: mse={A.vmse(rep, targets[0]):.2f}", flush=True)


if __name__ == "__main__":
    main()
