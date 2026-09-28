"""Audit the periodic "dot"/grid artifact in cascade reconstruction using a real checkpoint.
Computes, for the SAME real images: (a) real teacher-forced cascade decode (every level's decode
sees REAL ground-truth context/target, no self-generated dependency at all) and (b) the real
autoregressive rollout cascade (decode_generate_multipass, what run_gen_eval/save_compare_grid
already produce) -- if the periodic artifact appears in (a) too, it's a decode-logic issue, not
exposure bias from (b)'s self-generated history. Run once with JAX_PLATFORMS=cpu (dense attention
fallback, real Pallas splash_attention kernel doesn't run on CPU) and once on the real TPU backend,
diff the two, to rule out TPU-JIT-specific miscompilation (see safe_argmax's own history of exactly
this class of bug).

Usage (on the node holding the checkpoint -- never pull checkpoints off a TPU node, run here):
  python3 -m image_lagcodec.scripts.audit_gen_dots --run_dir image_lagcodec/logs/cifar_res_full1 \
      --checkpoint phase_2_step30000
  JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.audit_gen_dots --run_dir ... --checkpoint ...
"""
import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_ON_CPU = jax.default_backend() == "cpu"
if _ON_CPU:
    import image_lagcodec.eqx_common as eqx_common

    def _dense_splash_equivalent(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
        B, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        rep = Hq // Hkv
        if rep > 1:
            k = jnp.repeat(k, rep, axis=1)
            v = jnp.repeat(v, rep, axis=1)
        scores = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)
        idx = jnp.arange(T)
        if window is not None or lookahead > 0:
            left = window if window is not None else T
            mask = (idx[:, None] - idx[None, :] <= left) & (idx[None, :] - idx[:, None] <= lookahead)
        elif causal:
            mask = idx[:, None] >= idx[None, :]
        else:
            mask = jnp.ones((T, T), dtype=bool)
        scores = jnp.where(mask[None, None], scores, -1e9)
        if sink is not None:
            sink_col = jnp.broadcast_to(sink[None, :, None, None], (B, Hq, T, 1)).astype(scores.dtype)
            scores_ext = jnp.concatenate([scores, sink_col], axis=-1)
            w_ext = jax.nn.softmax(scores_ext, axis=-1)
            w = w_ext[..., :-1]
        else:
            w = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", w, v)

    eqx_common.splash_attention = _dense_splash_equivalent

import image_lagcodec.run_lagcodec_res as R

if _ON_CPU:
    R.splash_attention = _dense_splash_equivalent


def save_triple_grid(gt: np.ndarray, tf: np.ndarray, gen: np.ndarray, path: Path, pad: int = 2) -> None:
    from PIL import Image
    n, h, w, c = gt.shape
    grid = np.full((n * (h + pad) + pad, 3 * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        y = pad + i * (h + pad)
        grid[y:y + h, pad:pad + w] = gt[i]
        grid[y:y + h, 2 * pad + w:2 * pad + 2 * w] = tf[i]
        grid[y:y + h, 3 * pad + 2 * w:3 * pad + 3 * w] = gen[i]
    Image.fromarray(grid).save(path)


def periodic_stats(gen: np.ndarray, gt: np.ndarray, period: int = 8) -> dict:
    diff = np.abs(gen.astype(int) - gt.astype(int)).sum(axis=-1)  # (B,H,W)
    W = diff.shape[2]
    col_mean = diff.mean(axis=(0, 1))
    row_mean = diff.mean(axis=(0, 2))
    edge_cols = [c for c in range(W) if (c + 1) % period == 0]
    other_cols = [c for c in range(W) if (c + 1) % period != 0]
    return dict(mse=float(np.mean(diff.astype(np.float64) ** 2)),
                edge_col_mean=float(col_mean[edge_cols].mean()) if edge_cols else float("nan"),
                other_col_mean=float(col_mean[other_cols].mean()) if other_cols else float("nan"),
                col_mean=col_mean.tolist(), row_mean=row_mean.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / args.checkpoint

    cfg_vars = {}
    exec((run_dir / "resolved_config.py").read_text(), cfg_vars)
    data_root = Path(cfg_vars.get("data_root", str(REPO_ROOT / "datasets")))
    dataset = cfg_vars.get("dataset", "cifar")
    cfg_vars = {k: v for k, v in cfg_vars.items() if not k.startswith("_") and k in R.Config.__dataclass_fields__}
    cfg = R.Config(**cfg_vars)
    print(f"backend={jax.default_backend()} strides={cfg.strides} share_across_levels={cfg.share_across_levels} "
          f"bos_rate_mode={cfg.bos_rate_mode} upsampler_ncodes={cfg.upsampler_ncodes}")

    model_skeleton = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    model = eqx.tree_deserialise_leaves(ckpt_dir / "model.eqx", model_skeleton)
    print(f"checkpoint loaded from {ckpt_dir}")

    pixel_order = R.pixel_order_for(cfg)
    (train_np, _), (val_np, _) = R.load_dataset(dataset, data_root, cfg.img_size)
    imgs = val_np[:args.batch]
    flat_raw = jnp.array(R.images_to_positions(imgs, cfg, pixel_order))
    n = len(cfg.strides)

    codelm0 = model.codelm_for(0)
    tok0 = R.rgb_byte_pq_fn(flat_raw, codelm0.pq_chunks, codelm0.code_vocab)

    # --- REAL teacher-forced encode through all levels (ground-truth codes/code_soft per level) ---
    codes, codes_soft = [], []
    x, target = R.code_embed_proj(tok0, codelm0.own_input_embed, codelm0.own_input_proj), tok0
    for i in range(n):
        codelm_i = model.codelm_for(i)
        out = R.encode_pardec_downsampler(codelm_i, model.downsampler_for(i), x, target, flat_raw, cfg,
                                           pixel_order, R.rgb_label_fn_jax, model.K(i), rate_id=model.bos_rate_id(i),
                                           codelm_rate_id=model.codelm_bos_rate_id(i),
                                           downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes.append(out["code_idx"])
        codes_soft.append(out["code_soft"])
        if i < n - 1:
            codelm_next = model.codelm_for(i + 1)
            x = R.code_embed_proj(out["code_soft"], codelm_next.own_input_embed, codelm_next.own_input_proj)
            target = out["code_idx"]

    # --- (a) FULLY TEACHER-FORCED cascade decode: every level sees REAL ctx/target, argmax logits ---
    cur_tf = codes[n - 1]
    for i in range(n - 1, -1, -1):
        dec_target = tok0 if i == 0 else codes[i - 1]
        logits, target_out, _, _, _ = R.decode_logits_and_target_multipass(
            model, i, dec_target, codes_soft[i], cfg.upsampler_ncodes[i])
        cur_tf = jnp.argmax(logits, axis=-1).astype(dec_target.dtype)
    bytes_tf = np.asarray(cur_tf)
    img_tf = R.positions_to_image(bytes_tf, cfg, pixel_order)

    # --- (b) REAL autoregressive rollout cascade (what run_gen_eval/save_compare_grid produce) ---
    cur_gen = codes[n - 1]
    for i in range(n - 1, -1, -1):
        cur_gen = R.decode_generate_multipass(model, i, cur_gen, cfg.upsampler_ncodes[i], greedy=True)
    bytes_gen = np.asarray(cur_gen)
    img_gen = R.positions_to_image(bytes_gen, cfg, pixel_order)

    gt = imgs.astype(np.uint8)
    stats_tf = periodic_stats(img_tf, gt)
    stats_gen = periodic_stats(img_gen, gt)
    print(f"\n[teacher-forced cascade] mse={stats_tf['mse']:.2f} "
          f"edge_col(every 8th)={stats_tf['edge_col_mean']:.2f} other_col={stats_tf['other_col_mean']:.2f}")
    print(f"[rollout cascade]        mse={stats_gen['mse']:.2f} "
          f"edge_col(every 8th)={stats_gen['edge_col_mean']:.2f} other_col={stats_gen['other_col_mean']:.2f}")

    backend = jax.default_backend()
    png_path = run_dir / f"audit_gen_dots_{backend}.png"
    save_triple_grid(gt, img_tf, img_gen, png_path)
    print(f"saved GT | teacher-forced | rollout comparison to {png_path}")

    out_path = run_dir / f"audit_gen_dots_{backend}.npz"
    np.savez(out_path, img_tf=img_tf, img_gen=img_gen, gt=gt,
             col_mean_tf=stats_tf["col_mean"], col_mean_gen=stats_gen["col_mean"])
    print(f"saved raw arrays to {out_path}")


if __name__ == "__main__":
    main()
