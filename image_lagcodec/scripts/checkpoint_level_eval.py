"""CPU-only (safe to run alongside a live TPU training job on the same node, since it never touches
the TPU chips): loads a run_lagcodec_res.py checkpoint and, for each level 0..top, compares TWO
reconstruction modes:
  - "generate": self-generated at every level AND every AR digit timestep (encode_pardec_downsampler_
    generate + decode_generate_multipass, greedy/argmax) -- what real inference actually does, errors
    compound across levels.
  - "teacher_force": real label_fn-targeted encode (encode_pardec_downsampler) AND dense teacher-forced
    decode fed the REAL downstream target at every AR digit timestep (decode_logits_and_target_multipass
    + argmax, never self-fed) -- isolates "how good is the decoder's per-step prediction when it's never
    shown its own past mistakes", matching what run_val_eval's logged dec_acc/mse already measure.
Saves a GT|recon compare grid per level per mode. Standalone for ad-hoc checkpoint checks.
Usage: python3 -m image_lagcodec.scripts.checkpoint_level_eval <run_name> [--n_train 4] [--n_val 4]
       [--levels 0,1,2,3] [--tag offline] [--mode both|generate|teacher_force]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
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
from image_lagcodec.run_lagcodec_res import (
    Config, LagCodecModel, dataset_from_config, images_to_positions, pixel_order_for, positions_to_image,
    rgb_byte_pq_fn, load_config_module, CONFIG_FIELDS, encode_pardec_downsampler_generate,
    encode_pardec_downsampler, decode_generate_multipass, decode_logits_and_target_multipass,
    save_compare_grid, pixel_mse, default_label_fn_jax, rgb_label_fn_jax, default_label_fn_pil,
)

jax.config.update("jax_default_matmul_precision", "highest")


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    # CodeLM's own self-attention is the only place run_lagcodec_res.py calls the Pallas TPU
    # splash_attention kernel (downsampler/upsampler/digit heads already use plain dense JAX) --
    # this replaces it so the module runs on CPU. Native GQA (Hkv < Hq) handled via repeat.
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


def eval_level_generate(model, cfg, top, imgs, name, pixel_order, run_dir, tag):
    # self-generated at EVERY level and EVERY AR digit timestep -- real inference's actual path.
    t0 = time.monotonic()
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    codelm0 = model.codelm_for(0)
    raw = rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)
    codes = []
    for i in range(top + 1):
        codelm_i = model.codelm_for(i)
        out = encode_pardec_downsampler_generate(
            codelm_i, model.downsampler_for(i), raw, model.K(i), cfg,
            rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
            rng=jax.random.fold_in(jax.random.PRNGKey(0), i), greedy=True,
            temperature=cfg.gen_temperature, top_k=cfg.gen_top_k,
            downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes.append(out["code_idx"])
        if i < top:
            raw = out["code_soft"]

    cur = codes[top]
    for i in range(top, 0, -1):
        cur = decode_generate_multipass(model, i, cur, cfg.upsampler_ncodes[i], greedy=True,
                                         temperature=cfg.gen_temperature, seed=0)
    recon = np.asarray(decode_generate_multipass(model, 0, cur, cfg.upsampler_ncodes[0], greedy=True,
                                                  temperature=cfg.gen_temperature, seed=0))
    img = positions_to_image(recon, cfg, pixel_order)
    gt = imgs.astype(np.uint8)
    out_path = run_dir / f"samples_{tag}_generate_top{top}_{name}.png"
    save_compare_grid(img, gt, out_path)
    acc = float((recon == np.asarray(flat)).mean())
    mse = pixel_mse(img, gt)
    print(f"[generate {name} top={top}] byte_acc={acc:.4f} mse={mse:.2f} saved={out_path} "
          f"time={time.monotonic() - t0:.0f}s", flush=True)
    return mse


def eval_level_teacher_forced(model, cfg, top, imgs, name, pixel_order, run_dir, tag, label_fn):
    # Mirrors level_forward's OWN decode-cascade behavior under level_gt_drop=1.0 (what training
    # actually does): ctx starts at codes_soft[top] (the real, teacher-forced-encoded top-level code
    # -- nothing above it to chain from), then at each level i the digit-level AR steps are
    # teacher-forced against the REAL target (dec_target=codes[i], via decode_logits_and_target_
    # multipass's pardec_score), but the context fed to the NEXT (lower) level is THIS level's own
    # PREDICTION (greedy argmax, standing in for training's temperature-1 ZGR sample) -- never the
    # real codes_soft[i-1]. So digit-level mistakes never compound (always corrected), but
    # level-to-level mistakes DO compound, same as training. Prints a per-level agreement-with-GT
    # trace to localize exactly where the cascade starts diverging.
    t0 = time.monotonic()
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    codelm0 = model.codelm_for(0)
    tok0 = rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    codes, codes_soft = [tok0], []
    for i in range(top + 1):
        codelm_i = model.codelm_for(i)
        out = encode_pardec_downsampler(codelm_i, model.downsampler_for(i), raw, target, flat, cfg,
                                         pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                         codelm_rate_id=model.codelm_bos_rate_id(i), rng=None,
                                         downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        if i < top:
            raw, target = out["code_soft"], out["code_idx"]
    # codes[i] is level (i-1)'s real code (codes[0]==tok0), codes_soft[i] is level i's real code

    ctx = codes_soft[top]
    pred = None
    for i in range(top, -1, -1):
        dec_target = codes[i]
        logits, _, _, _, _ = decode_logits_and_target_multipass(model, i, dec_target, ctx,
                                                                  cfg.upsampler_ncodes[i])
        pred = jnp.argmax(logits, axis=-1)
        agree = float((pred == dec_target).mean())
        print(f"  [teacher_force {name} top={top}] level={i} agree_with_gt={agree:.4f}", flush=True)
        if i > 0:
            ctx = jax.nn.one_hot(pred, model.codelm_for(i).code_vocab, dtype=ctx.dtype)
    recon = np.asarray(pred)
    img = positions_to_image(recon, cfg, pixel_order)
    gt = imgs.astype(np.uint8)
    out_path = run_dir / f"samples_{tag}_teacherforce_top{top}_{name}.png"
    save_compare_grid(img, gt, out_path)
    acc = float((recon == np.asarray(flat)).mean())
    mse = pixel_mse(img, gt)
    print(f"[teacher_force {name} top={top}] byte_acc={acc:.4f} mse={mse:.2f} saved={out_path} "
          f"time={time.monotonic() - t0:.0f}s", flush=True)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_train", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--levels", type=str, default=None, help="comma-separated tops, default all 0..n_levels-1")
    ap.add_argument("--tag", type=str, default="offline")
    ap.add_argument("--mode", type=str, default="both", choices=["both", "generate", "teacher_force"])
    a = ap.parse_args()

    run_dir = REPO_ROOT / f"image_lagcodec/logs/{a.run}"
    ck = sorted((run_dir / "checkpoints").iterdir())[-1]
    cv = load_config_module(run_dir / f"config_{a.run}.py")
    label_fn_registry = {"default_label_fn_jax": default_label_fn_jax, "rgb_label_fn_jax": rgb_label_fn_jax,
                          "default_label_fn_pil": default_label_fn_pil}
    label_fn_raw = cv.pop("label_fn", "default_label_fn_jax")
    label_fn = label_fn_registry[label_fn_raw] if isinstance(label_fn_raw, str) else label_fn_raw
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    n_levels = len(cfg.strides)
    levels = [int(x) for x in a.levels.split(",")] if a.levels else list(range(n_levels))

    model = LagCodecModel(jax.random.PRNGKey(0), cfg)
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", model)
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)

    (train_np, _), (val_np, _) = dataset_from_config(cv, REPO_ROOT)
    pixel_order = pixel_order_for(cfg)
    print(f"run={a.run} ckpt={ck.name} backend={jax.default_backend()} n_levels={n_levels} "
          f"levels={levels} n_train={a.n_train} n_val={a.n_val} mode={a.mode}", flush=True)

    rows = []
    for top in levels:
        for name, imgs in (("val", val_np[:a.n_val]), ("train", train_np[:a.n_train])):
            mse_g = mse_tf = None
            if a.mode in ("both", "generate"):
                mse_g = eval_level_generate(model, cfg, top, imgs, name, pixel_order, run_dir, a.tag)
            if a.mode in ("both", "teacher_force"):
                mse_tf = eval_level_teacher_forced(model, cfg, top, imgs, name, pixel_order, run_dir, a.tag, label_fn)
            rows.append((top, name, mse_g, mse_tf))

    if a.mode == "both":
        print("\n=== generate vs teacher_force MSE ===", flush=True)
        for top, name, mse_g, mse_tf in rows:
            ratio = (mse_g / mse_tf) if mse_tf else float("nan")
            print(f"top={top} {name}: generate_mse={mse_g:.2f} teacher_force_mse={mse_tf:.2f} "
                  f"ratio={ratio:.2f}x", flush=True)


if __name__ == "__main__":
    main()
