"""Bisect the TPU jit-only corrupt generation, part 2: jit the whole pass-1 free-run stepping loop (transformer KV
steps + token head) with group state closed over (like decode_generate_pardec) vs passed as jit args.
Usage: python3 -m image_lagcodec.scripts.jit_loop_bisect <run_name>"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from image_lagcodec.scripts import stepwise_generation_check as sg
from image_lagcodec.run_lagcodec import (Config, HierEncDec, dataset_from_config, images_to_positions, pixel_order_for,
                                         code_embed_proj, load_config_module, CONFIG_FIELDS, pardec_block_step,
                                         pardec_block_chunk_step)


def free_loop(level, ctx, rope, min_valid, rope_bos, Wg, extra_len, plen, Kspan, B2, D):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    hd = D // level.n_heads
    ck = jnp.zeros((len(blocks), B2, level.n_kv_heads, plen, hd))
    cv = jnp.zeros_like(ck)
    bos = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    x = jnp.concatenate([ctx, bos], 1)
    crope = jnp.concatenate([rope, rope_bos[:, None]], 1)
    nk, nv = [], []
    for i, blk in enumerate(blocks):
        x, a, b = pardec_block_chunk_step(blk, x, ck[i], cv[i], jnp.array(0), crope, min_valid, plen)
        nk.append(a); nv.append(b)
    ck, cv = jnp.stack(nk), jnp.stack(nv)
    h = ln_f(x)[:, -1]
    pos, rp = Wg + extra_len + 1, rope_bos + 1
    vals, absmax, nans = [], [], []
    for t in range(Kspan):
        if t > 0:
            x = prev
            nk, nv = [], []
            for i, blk in enumerate(blocks):
                x, a, b = pardec_block_step(blk, x, ck[i], cv[i], pos, rp, min_valid, plen)
                nk.append(a); nv.append(b)
            ck, cv = jnp.stack(nk), jnp.stack(nv)
            h = ln_f(x)
            pos += 1
            rp = rp + 1
        v = level._token_generate_ar(h, None, True, 1.0)[0]
        vals.append(v); absmax.append(jnp.abs(h).max()); nans.append(jnp.isnan(h).sum())
        prev = level._dec_embed_target(v)
    return jnp.stack(vals, 1), jnp.stack(absmax), jnp.stack(nans)


run = sys.argv[1]
ck = sorted((sg.REPO_ROOT / f"image_lagcodec/logs/{run}/checkpoints").iterdir())[-1]
cv = load_config_module(sg.REPO_ROOT / f"image_lagcodec/logs/{run}/config_{run}.py")
cv.pop("label_fn", None)
cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
(_, _), (val, _) = dataset_from_config(cv, sg.REPO_ROOT)
NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 2
fb = jnp.array(images_to_positions(val[:NIMG], cfg, pixel_order_for(cfg)))
L0, L1 = model.levels
e0 = L0.encode(code_embed_proj(fb, L0.own_input_embed, L0.own_input_proj), fb, rng=None)
e1 = L1.encode(code_embed_proj(e0["code_soft"], L1.own_input_embed, L1.own_input_proj), e0["code_idx"], rng=None)
G = cfg.decoder_ncodes[0]
Kspan = G * L0.K
extra = [e1["code_idx"]] if L0.cond_depth > 1 else None
st = sg.group_state(L0, e0["code_idx"], extra, G)
static = (st["Wg"], st["extra_len"], st["per_group_len"], Kspan, st["B2"], st["D"])


def report(tag, out):
    v, am, nn = [np.asarray(o) for o in out]
    bad = (v < 0) | (v > 255)
    first = int(np.argmax(bad.any((0, 2)))) if bad.any() else -1
    print(f"RESULT {tag}: bad={bad.mean():.3f} zero={(v == 0).mean():.3f} first_bad_t={first} "
          f"h_nan_total={int(nn.sum())} h_absmax_by_t={np.round(am, 1).tolist()}", flush=True)


if NIMG <= 4:
    report("eager", free_loop(L0, st["ctx"], st["rope"], st["key_valid"], st["rope_bos"], *static))
report(f"jit_args_B{NIMG}", jax.jit(lambda lvl, c, r, m, rb: free_loop(lvl, c, r, m, rb, *static))(
    L0, st["ctx"], st["rope"], st["key_valid"], st["rope_bos"]))
report(f"jit_closure_B{NIMG}", jax.jit(lambda: free_loop(L0, st["ctx"], st["rope"], st["key_valid"], st["rope_bos"], *static))())
