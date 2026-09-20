"""Bisect the TPU jit-only corrupt generation: feed real per-step hidden states (from eager stepping) into the AR
token head eager vs jitted, and compare. Usage: python3 -m image_lagcodec.scripts.jit_head_bisect <run_name>"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from image_lagcodec.scripts import stepwise_generation_check as sg
from image_lagcodec.run_lagcodec import (Config, HierEncDec, load_cifar10, images_to_positions, pixel_order_for,
                                         code_embed_proj, load_config_module, CONFIG_FIELDS)

run = sys.argv[1]
ck = sorted((sg.REPO_ROOT / f"image_lagcodec/logs/{run}/checkpoints").iterdir())[-1]
cv = load_config_module(sg.REPO_ROOT / f"image_lagcodec/logs/{run}/config_{run}.py")
cv.pop("label_fn", None)
cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
(_, _), (val, _) = load_cifar10(sg.REPO_ROOT / "datasets")
NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 2
fb = jnp.array(images_to_positions(val[:NIMG], cfg, pixel_order_for(cfg)))
L0, L1 = model.levels
e0 = L0.encode(code_embed_proj(fb, L0.own_input_embed, L0.own_input_proj), fb, rng=None)
e1 = L1.encode(code_embed_proj(e0["code_soft"], L1.own_input_embed, L1.own_input_proj), e0["code_idx"], rng=None)
G = cfg.decoder_ncodes[0]
Kspan = G * L0.K
extra = [e1["code_idx"]] if L0.cond_depth > 1 else None
st = sg.group_state(L0, e0["code_idx"], extra, G)
gt = jnp.stack([fb[:, g * Kspan:(g + 1) * Kspan] for g in range(st["n_groups"])], 1).reshape(st["B2"], Kspan, -1)
hs = sg.run_mode(L0, st, gt, "tf", return_h=True)
jit_gen = jax.jit(lambda lvl, h: lvl._token_generate_ar(h, None, True, 1.0)[0])
jit_ref = jax.jit(sg.greedy_ar_reference)
import image_lagcodec.run_lagcodec as rl
h0 = hs[0]
N = h0.shape[0]
ctx = (h0 @ L0.token_in_proj)[:, None, :]
h1 = ctx + rl.dense_self_attention(L0.token_attn, L0.token_norm1(ctx), causal=True)
logit0 = rl.RMSNorm.__call__(L0.token_ln_f, h1)[:, -1, :] @ L0.token_out_head if False else L0.token_ln_f(h1)[:, -1, :] @ L0.token_out_head
am_e = np.asarray(jnp.argmax(logit0, -1))
am_j = np.asarray(jax.jit(lambda x: jnp.argmax(x, -1))(logit0))
lg_j = np.asarray(jax.jit(lambda lvl, h: (lambda c: lvl.token_ln_f(c + rl.dense_self_attention(lvl.token_attn, lvl.token_norm1(c), causal=True))[:, -1, :] @ lvl.token_out_head)((h @ lvl.token_in_proj)[:, None, :]))(L0, h0))
print(f"RESULT N={N} chunk0: jit-argmax-only bad={((am_j<0)|(am_j>255)).mean():.3f} eq_eager={(am_e==am_j).mean():.3f} | "
      f"jit-logits maxdiff_vs_eager={np.abs(lg_j-np.asarray(logit0)).max():.4f} argmax(jit logits) eq eager={(lg_j.argmax(-1)==am_e).mean():.3f}", flush=True)
for t, h in enumerate(hs):
    e = np.asarray(L0._token_generate_ar(h, None, True, 1.0)[0])
    j = np.asarray(jit_gen(L0, h))
    r = np.asarray(jit_ref(L0, h))
    print(f"RESULT t={t:2d} eager_range=({e.min()},{e.max()}) jit_gen_range=({j.min()},{j.max()}) "
          f"jit_gen_bad={((j < 0) | (j > 255)).mean():.3f} jit_gen_zero={(j == 0).mean():.3f} "
          f"eager==jit={(e == j).mean():.3f} jit_ref_range=({r.min()},{r.max()})", flush=True)
