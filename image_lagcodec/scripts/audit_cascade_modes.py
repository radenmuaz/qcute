"""CPU-only (safe next to a live TPU job): cascade generation re-done with dense passes under different choices of
(a) how the top code is given (hard argmax as generation does, or the soft probability vector as training does for a
quantize_drop fraction of digits), (b) the per-digit decode rule (argmax | mean), and (c) what a level hands down as
the next level's context (its chosen tokens, or its soft digit distributions). No retraining. Pixel mse per top.
Usage: python3 -m image_lagcodec.scripts.audit_cascade_modes <run> [--module run_lagcodec_res] [--ckpt DIR] [--n 64]
       [--tops 0,1,2] [--splits train,val]
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
from image_lagcodec.scripts.audit_decode_rule import digit_logits, pick

REPO_ROOT = A.REPO_ROOT


def decode(model, lvl, ctx, rule):
    # dense re-implementation of decode_generate_multipass; ctx = int codes or soft vectors. Returns tokens and the
    # per-digit distributions they were chosen from (last refine pass).
    cfg = model.cfg
    n_pass, Pp, fill, p1_len = A.pass_plan(cfg, lvl)
    K = model.K(lvl)
    rs = cfg.upsampler_ncodes[lvl] * K
    B, L = ctx.shape[0], ctx.shape[1] * K
    chunks, V = model.upsampler_for(lvl).output_chunks, cfg.code_vocab[lvl - 1] if lvl > 0 else cfg.code_vocab[0]
    out = None
    for p in range(n_pass):
        kw = (None, p1_len, "mask") if p == 0 else (jnp.asarray(out).astype(jnp.int32), Pp, fill)
        cur = np.zeros((B, L, chunks), np.int32)
        probs = np.zeros((B, L, chunks, V), np.float32)
        for j in range(rs):
            hid = A.hidden(model, lvl, jnp.asarray(cur), ctx, *kw)
            for m in range(chunks):
                lg = np.asarray(digit_logits(model, lvl, hid, jnp.asarray(cur))[:, j::rs, m, :]).astype(np.float64)
                cur[:, j::rs, m] = pick(lg, rule)
                pr = np.exp(lg - lg.max(-1, keepdims=True))
                probs[:, j::rs, m] = pr / pr.sum(-1, keepdims=True)
        out = cur
    return out, probs


def encode_rule(model, lvl, raw, rule):
    # downsampler code with digit rule `rule` (argmax = the real greedy encode); needs downsampler_ncodes == 1
    R, cfg = A.R, model.cfg
    assert cfg.downsampler_ncodes[lvl] == 1
    cl, ds, K = model.codelm_for(lvl), model.downsampler_for(lvl), model.K(lvl)
    h = R.pardec_context_hidden(cl, ds, raw, cfg, model.codelm_bos_rate_id(lvl), None, group_size=K)
    n = h.shape[1] // K
    cur = np.zeros((h.shape[0], n, ds.output_chunks), np.int32)
    hid = R.pardec_score(ds, jnp.asarray(cur), h, context_group_size=K, output_group_size=1, rate_id=model.bos_rate_id(lvl),
                         return_hidden=True)
    for m in range(ds.output_chunks):
        lg = R.token_ar_teacher_forced(ds.token_in_proj, ds.token_member_embed, ds.token_norm1, ds.token_attn, ds.token_ln_f,
                                       ds.token_out_head, ds.token_dim, ds.output_vocab, hid, jnp.asarray(cur))
        cur[:, :, m] = pick(np.asarray(lg[:, :, m, :]), rule)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=None, help="comma list of enc:rule:handoff, e.g. meanchain:mean:tokens")
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tops", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--save_dir", default=None, help="write GT|recon compare grids (first 8 images) per top and mode here")
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
    print(f"run={a.run} module={a.module} ckpt={ck} n={a.n} tops={a.tops}", flush=True)
    modes = [("argmax", "argmax", "tokens"), ("argmax", "mean", "tokens"), ("soft", "argmax", "tokens"), ("soft", "mean", "tokens"),
             ("soft", "argmax", "soft"), ("soft", "mean", "soft"), ("argmax", "mean", "soft")]
    if a.modes:
        modes = [tuple(m.split(":")) for m in a.modes.split(",")]
    for name in a.splits.split(","):
        imgs = (train_np if name == "train" else val_np)[:a.n]
        flat = jnp.array(R.images_to_positions(imgs, cfg, po))
        tok0 = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
        T0 = np.asarray(tok0)
        hard, soft, raw, tgt = [], [], tok0, tok0
        for i in range(max(a.tops) + 1):
            ev = R.encode_pardec_downsampler(model.codelm_for(i), model.downsampler_for(i), raw, tgt, flat, cfg, po, label_fn, model.K(i),
                                             rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i), rng=None,
                                             downsampler_ncodes=cfg.downsampler_ncodes[i])
            hard.append(ev["code_idx"])
            soft.append(jax.nn.softmax(ev["logits"].astype(jnp.float32), -1))
            raw, tgt = ev["code_soft"], ev["code_idx"]
        # same chain but every code digit = rounded mean of its distribution, fed upward as a hard code
        meanc, rawm = [], tok0
        for i in range(max(a.tops) + 1):
            c = encode_rule(model, i, rawm, "mean")
            meanc.append(jnp.asarray(c))
            rawm = jax.nn.one_hot(jnp.asarray(c), cfg.code_vocab[i], dtype=jnp.float32)
        chk = encode_rule(model, 0, tok0, "argmax")
        print(f"\n================ split={name} n={imgs.shape[0]} ================", flush=True)
        print(f"encode_rule(argmax) == real greedy encode (level 0): {float((chk == np.asarray(hard[0])).mean()):.6f} | "
              f"mean-rule code vs argmax code digits equal: {float((np.asarray(meanc[0]) == np.asarray(hard[0])).mean()):.3f}")
        for top in a.tops:
            # floors: the true image averaged over the top level's blocks, and the top code repeated
            span = 1
            for i in range(top + 1):
                span *= model.K(i)
            blk = T0.reshape(T0.shape[0], -1, span, T0.shape[-1]).astype(np.float64)
            rep = np.repeat(np.asarray(hard[top]), span, axis=1)
            print(f"[cascade top={top}] one code per {span} pixels | floors: true block mean repeated={float(((blk - blk.mean(2, keepdims=True)) ** 2).mean()):.1f} "
                  f"top code repeated={A.vmse(rep, T0):.1f}", flush=True)
            if a.save_dir:
                Path(a.save_dir).mkdir(parents=True, exist_ok=True)
                floor = np.clip(np.rint(np.repeat(blk.mean(2), span, axis=1)), 0, 255).astype(T0.dtype)
                R.save_compare_grid(R.positions_to_image(floor[:8], cfg, po), imgs[:8].astype(np.uint8),
                                    Path(a.save_dir) / f"{name}_top{top}_floor_block_mean.png")
            for enc_mode, rule, handoff in modes:
                if top == 0 and handoff == "soft":
                    continue
                ctx = {"argmax": hard, "soft": soft, "meanchain": meanc}[enc_mode][top]
                for i in range(top, -1, -1):
                    out, probs = decode(model, i, ctx, rule)
                    ctx = jnp.asarray(out) if handoff == "tokens" else jnp.asarray(probs)
                img = R.positions_to_image(out, cfg, po)
                if a.save_dir:
                    R.save_compare_grid(img[:8], imgs[:8].astype(np.uint8),
                                        Path(a.save_dir) / f"{name}_top{top}_code-{enc_mode}_digit-{rule}_down-{handoff}.png")
                tag = " (= real generation)" if (enc_mode, rule, handoff) == ("argmax", "argmax", "tokens") else ""
                print(f"  top code={enc_mode:6s} digit rule={rule:6s} handed down={handoff:6s}: pixel_mse={R.pixel_mse(img, imgs.astype(np.uint8)):8.2f}{tag}", flush=True)


if __name__ == "__main__":
    main()
