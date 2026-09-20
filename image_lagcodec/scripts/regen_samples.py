"""Regenerate the gen-eval sample grids for a finished run from its latest checkpoint, using the current (fixed)
generation code and full matmul precision. Mirrors run_gen_eval: first val_batch_size test + train images, encode
to `top`, cascade-decode down to bytes. Writes samples_<tag>_top{t}_{val,train}.png into the run's log dir.
Usage: python3 -m image_lagcodec.scripts.regen_samples <run_name> [tag=fixed]
"""
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
from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_cifar10, images_to_positions, pixel_order_for, positions_to_image, code_embed_proj,
    load_config_module, CONFIG_FIELDS, decode_generate_multipass, save_compare_grid, pixel_mse,
)

jax.config.update("jax_default_matmul_precision", "highest")
if jax.default_backend() == "cpu":
    def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
        n = q.shape[1] // k.shape[1]
        if n > 1:
            k, v = jnp.repeat(k, n, 1), jnp.repeat(v, n, 1)
        lg = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        T = q.shape[2]
        m = jnp.arange(T)[:, None] >= jnp.arange(T)[None, :]
        return jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(jnp.where(m[None, None], lg, -1e9), -1), v)
    eqx_common.splash_attention = _dense


def scalar(v, i=-1):
    return v[i] if isinstance(v, (list, tuple)) else v


def main():
    run = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "fixed"
    run_dir = REPO_ROOT / f"image_lagcodec/logs/{run}"
    ck = sorted((run_dir / "checkpoints").iterdir())[-1]
    cv = load_config_module(run_dir / f"config_{run}.py")
    cv.pop("label_fn", None)
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    assert cfg.cycle_refine_passes == 1, "regen_samples does not implement the cyclic-refine generation path"
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    n_levels = len(model.levels)
    n_img = int(scalar(cv.get("val_batch_size", 8)))
    enc_t = float(scalar(cv.get("encode_temperature", 1.0)))
    (train_np, _), (val_np, _) = load_cifar10(REPO_ROOT / "datasets")
    pixel_order = pixel_order_for(cfg)
    print(f"run={run} ckpt={ck.name} backend={jax.default_backend()} n_img={n_img} levels={n_levels}", flush=True)

    def gen(top, imgs, name):
        t0 = time.monotonic()
        flat = jnp.array(images_to_positions(imgs, cfg, pixel_order))
        x = code_embed_proj(flat, model.levels[0].own_input_embed, model.levels[0].own_input_proj)
        target = flat
        codes = []
        for i in range(top + 1):
            out = model.levels[i].encode(x, target, rng=None, encode_temperature=enc_t)
            codes.append(out["code_idx"])
            if i < top:
                x = code_embed_proj(out["code_soft"], model.levels[i + 1].own_input_embed,
                                    model.levels[i + 1].own_input_proj)
                target = out["code_idx"]
        cur = codes[top]
        for i in range(top, 0, -1):
            lv = model.levels[i]
            extra = [codes[j] if j <= top else None for j in range(i + 1, i + lv.cond_depth)] \
                if lv.cond_depth > 1 else None
            cur = decode_generate_multipass(lv, cur, cfg.decoder_ncodes[i], greedy=True, seed=0, extra_ctx_idx=extra)
        lv0 = model.levels[0]
        extra0 = [codes[j] if j <= top else None for j in range(1, lv0.cond_depth)] if lv0.cond_depth > 1 else None
        recon = np.asarray(decode_generate_multipass(lv0, cur, cfg.decoder_ncodes[0], greedy=True, seed=0,
                                                     extra_ctx_idx=extra0))
        bad = float(((recon < 0) | (recon > 255)).mean())
        img = positions_to_image(recon, cfg, pixel_order)
        gt = imgs.astype(np.uint8)
        save_compare_grid(img, gt, run_dir / f"samples_{tag}_top{top}_{name}.png")
        print(f"[{name} top={top}] byte_acc={float((recon == np.asarray(flat)).mean()):.4f} "
              f"mse={pixel_mse(img, gt):.2f} invalid_byte_frac={bad:.4f} time={time.monotonic() - t0:.0f}s", flush=True)

    for top in (1, 0) if n_levels >= 2 else (0,):
        gen(top, val_np[:n_img], "val")
        gen(top, train_np[:n_img], "train")


if __name__ == "__main__":
    main()
