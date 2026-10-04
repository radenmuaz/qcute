"""CPU-only (safe next to a live TPU job): transfer curves of each level's downsampler and upsampler on flat inputs.
A flat gray 2x2 block of value c should encode to a code ~c (label_fn = block mean) and a code c should decode back to
four pixels ~c. Sweeps c = 0..255 in one batch (window 1: every row only sees its own group) and reports how far
and how jagged the learned value->value maps are, for argmax and for the mean of the digit distribution.
Usage: python3 -m image_lagcodec.scripts.audit_transfer_curve <run> [--module run_lagcodec_res] [--ckpt DIR] [--levels 0,1,2]
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


def curve_stats(name, out, c):
    # out, c: (256,) predicted value for input value c
    d = out.astype(np.float64) - c
    step = np.diff(out.astype(np.float64))
    print(f"    {name:34s}: mean|out-c|={np.abs(d).mean():6.2f} rms={np.sqrt((d ** 2).mean()):6.2f} bias={d.mean():+6.2f} max={np.abs(d).max():5.0f} | "
          f"step std={step.std():5.2f} (1 = smooth identity has 0) decreasing steps={int((step < 0).sum())}/255 | "
          f"out at c=0,32,..,224,255: {[int(round(float(out[k]))) for k in list(range(0, 256, 32)) + [255]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
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
    print(f"run={a.run} module={a.module} ckpt={ck}", flush=True)
    c = np.arange(256)
    vals = np.arange(256)
    for lvl in a.levels:
        K = model.K(lvl)
        V = cfg.code_vocab[lvl]
        print(f"\n[level {lvl}] flat gray sweep c=0..255 (all 3 digits = c)")
        # ---- downsampler: K identical input tokens (c,c,c) per group -> one code
        tok = np.repeat(np.stack([c, c, c], -1)[:, None, :], K, axis=1).reshape(1, 256 * K, 3).astype(np.int32)
        cl, ds = model.codelm_for(lvl), model.downsampler_for(lvl)
        raw = jnp.asarray(tok) if lvl == 0 else jax.nn.one_hot(jnp.asarray(tok), cfg.code_vocab[lvl - 1], dtype=jnp.float32)
        h = R.pardec_context_hidden(cl, ds, raw, cfg, model.codelm_bos_rate_id(lvl), None, group_size=K * cfg.downsampler_ncodes[lvl])
        hid = R.pardec_score(ds, jnp.zeros((1, 256, 3), jnp.int32), h, context_group_size=K * cfg.downsampler_ncodes[lvl],
                             output_group_size=cfg.downsampler_ncodes[lvl], rate_id=model.bos_rate_id(lvl), return_hidden=True)
        code = np.asarray(R.token_ar_generate(ds.token_in_proj, ds.token_member_embed, ds.token_norm1, ds.token_attn, ds.token_ln_f,
                                              ds.token_out_head, ds.output_chunks, hid, jax.random.PRNGKey(0), True, 1.0, 0)[0])[0]
        lg0 = np.asarray(R.token_ar_teacher_forced(ds.token_in_proj, ds.token_member_embed, ds.token_norm1, ds.token_attn, ds.token_ln_f,
                                                   ds.token_out_head, ds.token_dim, ds.output_vocab, hid, jnp.asarray(code)[None])[0]).astype(np.float64)
        p = np.exp(lg0 - lg0.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        print("  downsampler (4 identical inputs c -> code digit):")
        for m in range(3):
            curve_stats(f"code digit {m} argmax", code[:, m], c)
        curve_stats("code digit 0 mean of distribution", (p[:, 0] * vals).sum(-1), c)
        print(f"    digit-0 distribution: entropy={float(-(p[:, 0] * np.log(np.maximum(p[:, 0], 1e-12))).sum(-1).mean()):.3f} nats "
              f"std={float(np.sqrt((p[:, 0] * (vals - (p[:, 0] * vals).sum(-1, keepdims=True)) ** 2).sum(-1)).mean()):.2f}")

        # ---- upsampler: code (c,c,c) -> K tokens; pass 1 only (no draft), free generation and distribution mean
        ctx = jnp.asarray(np.stack([c, c, c], -1)[None].astype(np.int32))
        n_pass, Pp, fill, p1_len = A.pass_plan(cfg, lvl)
        nc = cfg.upsampler_ncodes[lvl]
        if nc != 1:
            print("  upsampler sweep skipped (needs upsampler_ncodes == 1)")
            continue
        g1 = np.asarray(R._decode_generate_pardec_jit(model, lvl, ctx, nc, True, 1.0, 0, None, Pp, "mask") if p1_len else
                        R._decode_generate_pardec_jit(model, lvl, ctx, nc, True, 1.0, 0))[0].reshape(256, K, 3)
        gl = np.asarray(R.decode_generate_multipass(model, lvl, ctx, nc, greedy=True))[0].reshape(256, K, 3)
        print("  upsampler (code c -> 4 tokens), free generation:")
        for j in range(K):
            curve_stats(f"pass 1 token {j} digit 0", g1[:, j, 0], c)
        curve_stats("pass 1 token 0 digit 1", g1[:, 0, 1], c)
        curve_stats("pass 1 token 0 digit 2", g1[:, 0, 2], c)
        curve_stats("last pass token 0 digit 0", gl[:, 0, 0], c)
        curve_stats("pass 1 row mean (all tokens, digits)", g1.reshape(256, -1).mean(-1), c)
        hid = A.hidden(model, lvl, jnp.zeros((1, 256 * K, 3), jnp.int32), ctx, None, p1_len, "mask")
        lg = np.asarray(digit_logits(model, lvl, hid, jnp.zeros((1, 256 * K, 3), jnp.int32)))[0].reshape(256, K, 3, -1)[:, 0, 0].astype(np.float64)
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        curve_stats("pass 1 token 0 digit 0 MEAN of dist", (p * vals).sum(-1), c)
        print(f"    token-0 digit-0 distribution: entropy={float(-(p * np.log(np.maximum(p, 1e-12))).sum(-1).mean()):.3f} nats "
              f"std={float(np.sqrt((p * (vals - (p * vals).sum(-1, keepdims=True)) ** 2).sum(-1)).mean()):.2f} max_prob={float(p.max(-1).mean()):.3f}")
        # round trip: flat c -> code -> decode
        rt = np.asarray(R.decode_generate_multipass(model, lvl, jnp.asarray(code)[None], nc, greedy=True))[0].reshape(256, K, 3)
        curve_stats("round trip flat c -> code -> tokens (mean)", rt.reshape(256, -1).mean(-1), c)
        print(f"    round trip: mean within-row std of the decoded tokens (0 for a flat block)={float(rt.std(1).mean()):.2f} "
              f"| generated-from-(c,c,c) within-row std={float(gl.std(1).mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
