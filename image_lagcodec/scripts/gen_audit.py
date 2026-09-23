"""Per-level generation audit on the current backend (run on the TPU node, no checkpoint pulling): for each level i
generate level i-1 tokens greedily from the REAL level-i codes and compare with (a) the real tokens, (b) the
teacher-forced argmax on the real tokens, (c) the dense re-scoring argmax of the generated tokens themselves.
Usage: python3 -m image_lagcodec.scripts.gen_audit <run_name> [n_img=8]"""
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.run_lagcodec as rl
from image_lagcodec.run_lagcodec import (Config, HierEncDec, images_to_positions, pixel_order_for, code_embed_proj,
                                         load_config_module, CONFIG_FIELDS, decode_generate_multipass,
                                         decode_logits_and_target_multipass, cast_pytree)

run = sys.argv[1]
n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 8
force_fp32 = len(sys.argv) > 3 and sys.argv[3] == "fp32"
run_dir = REPO_ROOT / f"image_lagcodec/logs/{run}"
ck = sorted(p for p in (run_dir / "checkpoints").iterdir() if (p / "model.eqx").exists())[-1]
cv = load_config_module(run_dir / f"config_{run}.py")
cv.pop("label_fn", None)
cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
if cfg.precision == "bf16" and not force_fp32:
    model = cast_pytree(model, jnp.bfloat16)
else:
    jax.config.update("jax_default_matmul_precision", "highest")
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
(_, _), (val_np, _) = rl.load_cifar10(REPO_ROOT / "datasets")
flat = jnp.array(images_to_positions(val_np[:n_img], cfg, pixel_order_for(cfg)))
print(f"run={run} ckpt={ck.name} backend={jax.default_backend()} n_img={n_img} fp32={force_fp32 or cfg.precision != 'bf16'}", flush=True)

levels = model.levels
n_lv = len(levels)
codes, soft, tgt = [], [], flat
x = code_embed_proj(flat, levels[0].own_input_embed, levels[0].own_input_proj)
for i in range(n_lv):
    out = levels[i].encode(x, tgt, rng=None)
    codes.append(out["code_idx"]); soft.append(out["code_soft"])
    if i < n_lv - 1:
        x = code_embed_proj(out["code_soft"], levels[i + 1].own_input_embed, levels[i + 1].own_input_proj)
        tgt = out["code_idx"]

print(f"{'lvl':>3} {'G':>3} {'tf_acc':>7} {'gen_acc':>8} {'gen==dense(own)':>16} {'invalid':>8} {'gen_time':>9}")
for i in range(n_lv - 1, -1, -1):
    lv = levels[i]
    G = cfg.decoder_ncodes[i]
    target = flat if i == 0 else codes[i - 1]
    extras_idx = [codes[j] for j in range(i + 1, i + lv.cond_depth)] if lv.cond_depth > 1 else None
    extras_soft = [soft[j] for j in range(i + 1, i + lv.cond_depth)] if lv.cond_depth > 1 else None
    logits = decode_logits_and_target_multipass(lv, target, soft[i], G, rng=None, extra_ctx_code_soft=extras_soft)[0]
    tf_acc = float(jnp.mean(jnp.argmax(logits, -1) == target))
    t0 = time.monotonic()
    gen = decode_generate_multipass(lv, codes[i], G, greedy=True, seed=0, extra_ctx_idx=extras_soft and extras_idx)
    gen.block_until_ready()
    gt = time.monotonic() - t0
    gen_acc = float(jnp.mean(gen == target))
    vmax = 255 if i == 0 else cfg.code_vocab[i - 1] - 1
    invalid = float(((gen < 0) | (gen > vmax)).mean())
    lg2 = decode_logits_and_target_multipass(lv, gen, soft[i], G, rng=None, extra_ctx_code_soft=extras_soft)[0]
    dense_same = float(jnp.mean(jnp.argmax(lg2, -1) == gen))
    print(f"{i:>3} {G:>3} {tf_acc:7.3f} {gen_acc:8.3f} {dense_same:16.3f} {invalid:8.4f} {gt:8.0f}s", flush=True)
    print(f"      distinct generated values={len(np.unique(np.asarray(gen)))}  top-value frac="
          f"{max(float((gen == v).mean()) for v in np.unique(np.asarray(gen))[:300]):.3f}  "
          f"real distinct={len(np.unique(np.asarray(target)))}", flush=True)

print("\nfirst-token exactness: generation fed REAL previous tokens as drafts must equal the teacher-forced argmax")
print(f"{'lvl':>3} {'Pp':>4} {'first_tok_match(real draft)':>28} {'first_tok_match(private redecode)':>34}")
for i in range(n_lv - 1, -1, -1):
    lv = levels[i]
    G = cfg.decoder_ncodes[i]
    Pp = lv.decode_past
    if Pp == 0:
        continue
    target = flat if i == 0 else codes[i - 1]
    extras_idx = [codes[j] for j in range(i + 1, i + lv.cond_depth)] if lv.cond_depth > 1 else None
    extras_soft = [soft[j] for j in range(i + 1, i + lv.cond_depth)] if lv.cond_depth > 1 else None
    logits = lv.decode_logits_and_target_pardec(target, soft[i], G, rng=None, extra_ctx_code_soft=extras_soft)[0]
    tf_first = jnp.argmax(logits, -1)
    B = target.shape[0]
    n_blocks = codes[i].shape[1]
    n_groups = -(-n_blocks // G)
    Kspan = G * lv.K
    tp = jnp.pad(target, ((0, 0), (0, n_groups * Kspan - target.shape[1])) + ((0, 0),) * (target.ndim - 2))
    dp = jnp.pad(tp, ((0, 0), (Pp, 0)) + ((0, 0),) * (tp.ndim - 2))
    dw = jnp.stack([dp[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1).reshape(B * n_groups, Pp, *target.shape[2:])
    dv = jnp.broadcast_to(jnp.asarray(rl._draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * lv.K))[None],
                          (B, n_groups, Pp)).reshape(B * n_groups, Pp)
    g_real = lv.decode_generate_pardec(codes[i], G, greedy=True, seed=0, decode_past_override=Pp, draft_override_flat=dw,
                                       draft_valid_flat=dv, extra_ctx_idx=extras_idx)
    g_priv = lv.decode_generate_pardec(codes[i], G, greedy=True, seed=0, extra_ctx_idx=extras_idx)
    first = jnp.arange(n_groups) * Kspan
    first = first[first < tf_first.shape[1]]
    m_real = float(jnp.mean(g_real[:, first] == tf_first[:, first]))
    m_priv = float(jnp.mean(g_priv[:, first] == tf_first[:, first]))
    print(f"{i:>3} {Pp:>4} {m_real:28.4f} {m_priv:34.4f}", flush=True)
