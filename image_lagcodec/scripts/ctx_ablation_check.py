"""CPU-only (safe alongside a live TPU/CPU training job on the same node -- never touches TPU chips):
loads a run_lagcodec_res.py checkpoint and runs level_forward's teacher-forced-digit-AR decode
(digit_teacher_force=True, same cross-level ctx cascade as training) three ways:
  - ctx_ablation=None     ("real"): normal decode, real top-level ctx.
  - ctx_ablation="zero"   : top-level ctx replaced with zeros before any decode.
  - ctx_ablation="shuffle": top-level ctx permuted across the batch (sample i gets sample i-1's ctx).
If byte_mse barely changes between "real" and "zero"/"shuffle", the decode cascade is not actually
using the encoded content -- it's reconstructing via teacher-forced AR self-attention over real
previous tokens alone (exposure-bias/posterior-collapse-style failure). Also prints the TOP level's
raw code_idx values per sample (exact digits, not just a util/entropy summary) to directly inspect
whether/how much they vary across the batch.
Usage: python3 -m image_lagcodec.scripts.ctx_ablation_check <run_name> [--n_train 8] [--n_val 0]
       [--top -1] [--tag ablation]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import sys
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
    load_config_module, CONFIG_FIELDS, level_forward, save_compare_grid, pixel_mse,
    default_label_fn_jax, rgb_label_fn_jax, default_label_fn_pil,
)

jax.config.update("jax_default_matmul_precision", "highest")


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


def run_ablation(model, cfg, phase, imgs, name, pixel_order, run_dir, tag, label_fn, level_gt_drop):
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    results = {}
    for mode in (None, "zero", "shuffle"):
        loss, aux = level_forward(model, flat, phase, rng=None, level_gt_drop=level_gt_drop,
                                   label_reg_weight=0.0, label_fn=label_fn, pixel_order=pixel_order,
                                   digit_teacher_force=True, return_recon=True, ctx_ablation=mode)
        pred_bytes = np.asarray(aux[-1])
        recon_img = positions_to_image(pred_bytes, cfg, pixel_order)
        gt_img = imgs.astype(np.uint8)
        mse = pixel_mse(recon_img, gt_img)
        tag_mode = mode or "real"
        save_compare_grid(recon_img, gt_img, run_dir / f"samples_{tag}_{name}_ctxablate_{tag_mode}.png")
        results[tag_mode] = mse
        print(f"[{tag}_{name}] ctx_ablation={tag_mode!r:10s} byte_mse={mse:.2f}", flush=True)
    real, zero, shuf = results["real"], results["zero"], results["shuffle"]
    print(f"[{tag}_{name}] zero/real ratio={zero / real:.3f}  shuffle/real ratio={shuf / real:.3f}  "
          f"(near 1.0 => decode ignores ctx; near/above typical generate-mse gap => ctx matters)",
          flush=True)
    return results


def print_top_level_codes(model, cfg, phase, imgs, name, pixel_order, label_fn):
    # Re-run just the encode side (phase levels) to get the TOP level's raw code_idx, printed in
    # full so variation (or collapse) across the batch can be inspected directly, not just summarized
    # by util/entropy. Reuses level_forward's own encode loop via a trimmed local copy would
    # duplicate logic -- instead call level_forward with return_recon=False and read nothing from it
    # (loss-only call is wasteful); simplest is to just use the real encode helper directly.
    from image_lagcodec.run_lagcodec_res import encode_pardec_downsampler, rgb_byte_pq_fn
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    codelm0 = model.codelm_for(0)
    tok0 = rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)
    raw, target = tok0, tok0
    code_idx = None
    for i in range(phase):
        codelm_i = model.codelm_for(i)
        out = encode_pardec_downsampler(codelm_i, model.downsampler_for(i), raw, target, flat, cfg,
                                         pixel_order, label_fn, model.K(i), rate_id=model.bos_rate_id(i),
                                         codelm_rate_id=model.codelm_bos_rate_id(i), rng=None,
                                         downsampler_ncodes=cfg.downsampler_ncodes[i])
        code_idx = out["code_idx"]
        if i < phase - 1:
            raw, target = out["code_soft"], out["code_idx"]
    code_idx = np.asarray(code_idx)  # (B, n_blocks, pq_chunks)
    print(f"\n[{name}] TOP level (phase={phase}) code_idx exact values, shape={code_idx.shape}:")
    B = code_idx.shape[0]
    for b in range(B):
        print(f"  sample{b}: {code_idx[b].reshape(-1).tolist()}")
    flat_codes = code_idx.reshape(B, -1)
    n_unique_rows = len(set(tuple(r) for r in flat_codes.tolist()))
    per_digit_unique = [len(set(flat_codes[:, d].tolist())) for d in range(flat_codes.shape[1])]
    print(f"[{name}] n_unique_full_code_rows={n_unique_rows}/{B}  "
          f"per_digit_unique_count={per_digit_unique} (vocab={cfg.code_vocab[phase - 1]} each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_val", type=int, default=0)
    ap.add_argument("--top", type=int, default=-1, help="-1 = top level (phase=n_levels)")
    ap.add_argument("--tag", type=str, default="ablation")
    a = ap.parse_args()

    run_dir = REPO_ROOT / f"image_lagcodec/logs/{a.run}"
    ck = sorted((run_dir / "checkpoints").iterdir())[-1]
    cv = load_config_module(run_dir / f"config_{a.run}.py")
    label_fn_registry = {"default_label_fn_jax": default_label_fn_jax, "rgb_label_fn_jax": rgb_label_fn_jax,
                          "default_label_fn_pil": default_label_fn_pil}
    label_fn_raw = cv.pop("label_fn", "default_label_fn_jax")
    label_fn = label_fn_registry[label_fn_raw] if isinstance(label_fn_raw, str) else label_fn_raw
    level_gt_drop = cv.get("level_gt_drop", 0.5)  # level_gt_drop is argparse-only, not a Config field
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    n_levels = len(cfg.strides)
    phase = n_levels if a.top == -1 else a.top + 1

    model = LagCodecModel(jax.random.PRNGKey(0), cfg)
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", model)
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)

    (train_np, _), (val_np, _) = dataset_from_config(cv, REPO_ROOT)
    pixel_order = pixel_order_for(cfg)
    print(f"run={a.run} ckpt={ck.name} backend={jax.default_backend()} n_levels={n_levels} "
          f"phase={phase} (top level={phase - 1}) n_train={a.n_train} n_val={a.n_val}", flush=True)

    sets = []
    if a.n_train > 0:
        sets.append(("train", train_np[:a.n_train]))
    if a.n_val > 0:
        sets.append(("val", val_np[:a.n_val]))

    for name, imgs in sets:
        print_top_level_codes(model, cfg, phase, imgs, name, pixel_order, label_fn)
        run_ablation(model, cfg, phase, imgs, name, pixel_order, run_dir, a.tag, label_fn, level_gt_drop)


if __name__ == "__main__":
    main()
