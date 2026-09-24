"""CPU-only: audits a real run_lagcodec_stack.py checkpoint's self-consistency -- greedy
incremental-KV generation (decode_generate_dispatch) vs teacher-forced re-scoring of that SAME
generated output (decode_logits_and_target_dispatch), argmax(logits) must equal the generated
tokens exactly. Same methodology as image_lagcodec/scripts/audit_ckpt_tf_vs_kv.py (pardec's own
version) -- NOT a check against real ground truth (exposure bias there is expected, not a bug).

Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.audit_stack_ckpt \
    --config image_lagcodec/configs/cifar_stack_ar1.py \
    --ckpt image_lagcodec/logs/cifar_stack_ar1/checkpoints/phase_2_step5000 \
    --n_images 2 --level 0
"""
import argparse
import dataclasses
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from image_lagcodec.run_lagcodec_stack import (
    Config, HierEncDec, load_config_module, dataset_from_config, images_to_positions, pixel_order_for,
    code_embed_proj,
)
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            q_idx, kv_idx = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
            if window is None and lookahead <= 0:
                mask = q_idx >= kv_idx
            else:
                mask = jnp.ones((T, T), dtype=bool)
                if window is not None:
                    mask = mask & (q_idx - window <= kv_idx)
                mask = mask & (q_idx + max(0, lookahead) >= kv_idx)
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)
    eqx_common.splash_attention = _cpu_dense_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_images", type=int, default=2)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cv = load_config_module(Path(args.config))
    valid = {f.name for f in dataclasses.fields(Config)}
    cfg_kwargs = {k: v for k, v in cv.items() if k in valid}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = Config(**cfg_kwargs)

    print(f"building skeleton model (level={args.level}, decoder_ncodes={cfg.decoder_ncodes[args.level]}, "
          f"cond_depth={cfg.cond_depth[args.level]}, weight_sharing={cfg.weight_sharing[args.level]})...")
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    ckpt_path = Path(args.ckpt) / "model.eqx"
    print(f"loading checkpoint: {ckpt_path}")
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    levels = model.levels

    print("loading a few real images...")
    (train_x, _), _ = dataset_from_config(cv, Path("."), train_shards=1)
    imgs = np.asarray(train_x[: args.n_images])
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))

    print("encoding real ctx (own codes + any coarser levels needed for cond_depth)...")
    codes_idx, codes_soft = [], []
    x = code_embed_proj(flat, levels[0].own_input_embed, levels[0].own_input_proj)
    tgt = flat
    n_needed = args.level + max(1, cfg.cond_depth[args.level])
    for i in range(min(n_needed, len(levels))):
        o = levels[i].encode(x, tgt, rng=None)
        codes_idx.append(o["code_idx"])
        codes_soft.append(o["code_soft"])
        if i + 1 < len(levels):
            x = code_embed_proj(o["code_soft"], levels[i + 1].own_input_embed, levels[i + 1].own_input_proj)
            tgt = o["code_idx"]

    lev = levels[args.level]
    cd = cfg.cond_depth[args.level]
    extra_soft = codes_soft[args.level + 1: args.level + cd] if cd > 1 else None
    extra_idx = codes_idx[args.level + 1: args.level + cd] if cd > 1 else None
    target_seq = flat if args.level == 0 else codes_idx[args.level - 1]
    ctx_code_soft = codes_soft[args.level]
    ctx_idx = codes_idx[args.level]
    G = cfg.decoder_ncodes[args.level]

    print(f"n_blocks={ctx_idx.shape[1]}, G={G} -- running greedy incremental-KV generation...")
    gen = lev.decode_generate_dispatch(ctx_idx, G, greedy=True, seed=args.seed, extra_ctx_idx=extra_idx)

    print("re-scoring the model's OWN generated output, teacher-forced (the real self-consistency check)...")
    logits_rescore, *_ = lev.decode_logits_and_target_dispatch(
        gen, ctx_code_soft, G, extra_ctx_code_soft=extra_soft)
    argmax_rescore = jnp.argmax(logits_rescore, -1)

    n = min(argmax_rescore.shape[1], gen.shape[1])
    a, g = argmax_rescore[:, :n], gen[:, :n]
    same = bool(jnp.array_equal(a, g))
    n_mismatch = int((a != g).sum())
    total = int(a.size)
    print(f"\nSELF-CONSISTENCY (teacher-forced argmax(logits) == incremental-KV greedy generation): {same}")
    print(f"mismatches: {n_mismatch} / {total} ({100.0*n_mismatch/max(total,1):.3f}%)")
    if not same:
        b_idx = np.argwhere(np.asarray(a != g).reshape(a.shape[0], -1).any(-1))
        print("mismatching batch rows:", b_idx[:10].reshape(-1).tolist())
    print("PASS (self-consistent)" if same else "FAIL (see mismatches above -- this IS a real bug signal)")

    print("\n--- informational only, NOT a correctness check (exposure bias is expected mid-training) ---")
    n_gt = min(target_seq.shape[1], gen.shape[1]) if target_seq.ndim == gen.ndim else None
    if n_gt is not None:
        gt = target_seq[:, :n_gt]
        gen_c = gen[:, :n_gt]
        gt_acc = float((gt == gen_c).mean())
        print(f"free-running generation byte-accuracy vs real ground truth: {gt_acc*100:.2f}% "
              f"(low is normal/expected mid-training, this is exposure bias not a bug)")


if __name__ == "__main__":
    main()
