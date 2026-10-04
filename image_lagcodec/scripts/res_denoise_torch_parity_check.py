"""CPU parity check: run_lagcodec_res_denoise_torch.py vs run_lagcodec_res_denoise.py (JAX). Builds the same
Config in both, copies the JAX weights into the torch model by parameter path, then compares (deterministic
paths, rng=None): level_forward loss + every aux metric, every parameter gradient, label targets, and greedy
generation (encode + cycle-aware decode cascade). Must end PASS all.
Usage: python3 -m image_lagcodec.scripts.res_denoise_torch_parity_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    # reference for the TPU splash kernel: LocalMask(window, lookahead) / causal / full, optional sink logit
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = jnp.repeat(k, rep, 1), jnp.repeat(v, rep, 1)
    lg = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)
    T = q.shape[2]
    t, s = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
    if window is not None or lookahead > 0:
        m = s <= t + lookahead
        if window is not None:
            m = m & (s >= t - window)
    else:
        m = (s <= t) if causal else jnp.ones((T, T), bool)
    lg = jnp.where(m[None, None], lg, -jnp.inf)
    if sink is not None:
        sk = jnp.broadcast_to(sink[None, :, None, None].astype(lg.dtype), lg.shape[:3] + (1,))
        w = jax.nn.softmax(jnp.concatenate([lg, sk], -1), -1)[..., :-1]
    else:
        w = jax.nn.softmax(lg, -1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense
import image_lagcodec.run_lagcodec_res_denoise as J
J.splash_attention = _dense
import image_lagcodec.run_lagcodec_res_denoise_torch as T

jax.config.update("jax_default_matmul_precision", "highest")
torch.set_default_dtype(torch.float32)


def jax_to_torch_state(jmodel) -> dict:
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(jmodel, eqx.is_array))[0]
    out = {}
    for path, x in leaves:
        name = jax.tree_util.keystr(path).replace("[", ".").replace("]", "").lstrip(".")
        out[name] = torch.from_numpy(np.array(x, dtype=np.float32))
    return out


def build_pair(**over):
    base = dict(img_size=32, strides=(4, 4), share_across_levels=False, codelm_d_model=32, codelm_n_layers=2,
                codelm_n_heads=2, codelm_n_kv_heads=1, downsampler_d_model=32, downsampler_n_layers=1,
                downsampler_n_heads=2, downsampler_n_kv_heads=1, upsampler_d_model=32, upsampler_n_layers=2,
                upsampler_n_heads=2, upsampler_n_kv_heads=2, upsampler_window=1, downsampler_window=1,
                code_vocab=256, pq_chunks=3, pq_dim=16, byte_group=3, upsampler_ncodes=1, precision="fp32",
                quantize_mode="reinmax_limit", ctx_stop_gradient="pseudo", label_reg_weight=1.0,
                pardec_token_head="ar", codelm_token_head="ar", token_head_type="ar", token_dim=16, token_n_heads=2,
                traversal="zorder", mse_weight=0.1, entropy_weight=0.1, label_mse_weight=0.01)
    base.update(over)
    jcfg, tcfg = J.Config(**base), T.Config(**base)
    jm = J.LagCodecModel(jax.random.PRNGKey(0), jcfg)
    tm = T.LagCodecModel(tcfg)
    sd = jax_to_torch_state(jm)
    missing, unexpected = tm.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"state mismatch missing={missing} unexpected={unexpected}"
    return jm, tm, jcfg, tcfg


def load_flat(cfg, n=2):
    (train_np, _), _ = J.load_cifar10(REPO_ROOT / "datasets")
    po = J.pixel_order_for(cfg)
    flat = J.images_to_positions(train_np[:n], cfg, po)
    return flat, po


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-8))


def check(name, gen_levels=True, phase=2, **over):
    jm, tm, jcfg, tcfg = build_pair(**over)
    flat_np, po = load_flat(jcfg)
    jflat, tflat = jnp.array(flat_np), torch.as_tensor(flat_np).long()
    jlab, tlab = J.rgb_label_fn_jax, T.rgb_label_fn
    # identical label targets in both (label fn parity is measured separately below)
    tlab_same = lambda fb, c, po_, nb, pc, cv: torch.as_tensor(
        np.asarray(jlab(jnp.array(fb.numpy()), jcfg, po_, nb, pc, cv))).long()
    kw_j = dict(level_gt_drop=1.0, label_reg_weight=1.0, label_fn=jlab, pixel_order=po)
    kw_t = dict(level_gt_drop=1.0, label_reg_weight=1.0, label_fn=tlab_same, pixel_order=po)
    f = lambda m: J.level_forward(m, jflat, phase, rng=None, **kw_j)
    (jl, jaux), jg = eqx.filter_jit(eqx.filter_value_and_grad(f, has_aux=True))(jm)
    tm.zero_grad()
    tl, taux = T.level_forward(tm, tflat, phase, rng=None, **kw_t)
    tl.backward()
    errs = {"loss": rel(float(tl), float(jl))}
    for k, (a, b) in enumerate(zip(taux, jaux)):
        errs[f"aux{k}"] = rel(float(a), float(b))
    jgrads = jax_to_torch_state(J.tie_shared_grads(jg, jcfg))  # torch shares one Parameter: compare summed grads
    gerr, worst = 0.0, ""
    for n, prm in tm.named_parameters():
        g = prm.grad.numpy() if prm.grad is not None else np.zeros(prm.shape, np.float32)
        e = float(np.abs(g - jgrads[n].numpy()).max() / max(np.abs(jgrads[n].numpy()).max(), 1e-6))
        if e > gerr:
            gerr, worst = e, n
    errs["grad"] = gerr
    n_blocks = J.n_blocks_for_level(jcfg, 0)
    lab_j = np.asarray(jlab(jflat, jcfg, po, n_blocks, 3, 256))
    lab_t = tlab(tflat, tcfg, po, n_blocks, 3, 256).numpy()
    errs["label_mismatch"] = float((lab_j != lab_t).mean())
    errs["label_maxdiff"] = int(np.abs(lab_j.astype(np.int64) - lab_t.astype(np.int64)).max())
    gen_mis = 0
    if gen_levels:
        with torch.no_grad():
            jtok, ttok = jflat, tflat
            for i in range(phase):
                jo = J.encode_pardec_downsampler_generate(jm.codelm_for(i), jm.downsampler_for(i), jtok, jm.K(i), jcfg,
                                                          rate_id=jm.bos_rate_id(i), rng=jax.random.PRNGKey(0),
                                                          greedy=True, codelm_rate_id=jm.codelm_bos_rate_id(i))
                to = T.encode_pardec_downsampler_generate(tm.codelm_for(i), tm.downsampler_for(i), ttok, tm.K(i), tcfg,
                                                          rate_id=tm.bos_rate_id(i), rng=T.Key(0), greedy=True,
                                                          codelm_rate_id=tm.codelm_bos_rate_id(i))
                gen_mis += int((np.asarray(jo["code_idx"]) != to["code_idx"].numpy()).sum())
                jtok, ttok = jo["code_idx"], torch.as_tensor(np.asarray(jo["code_idx"])).long()
            jcur, tcur = jtok, ttok
            for i in range(phase - 1, -1, -1):
                jcur = J.decode_generate_cycles(jm, i, jcur, jcfg.upsampler_ncodes[i], greedy=True)
                tnext = T.decode_generate_cycles(tm, i, tcur, tcfg.upsampler_ncodes[i], greedy=True)
                gen_mis += int((np.asarray(jcur) != tnext.numpy()).sum())
                tcur = torch.as_tensor(np.asarray(jcur)).long()  # same input to the next level
    errs["gen_mismatch"] = gen_mis
    ok = (errs["loss"] < 1e-4 and all(errs[f"aux{k}"] < 1e-3 for k in range(len(taux))) and errs["grad"] < 2e-3
          and errs["label_mismatch"] < 0.005 and errs["label_maxdiff"] <= 1 and gen_mis == 0)
    print(f"{name}: loss j={float(jl):.6f} t={float(tl):.6f} rel={errs['loss']:.1e} | max aux rel="
          f"{max(errs[f'aux{k}'] for k in range(len(taux))):.1e} | max grad rel={gerr:.1e} ({worst}) | "
          f"label mismatch={errs['label_mismatch']:.4f} (max |diff| {errs['label_maxdiff']}, exact-.5 rounding) | gen mismatch={gen_mis}  {'OK' if ok else 'DIFF'}")
    return ok


def check_reinmax():
    # torch quantize_reinmax_limit (O(K)) vs its dense slow reference, and vs JAX's: forward exact, grads ~1e-6.
    # rng=None so both frameworks draw idx by argmax (their random streams differ)
    rs = np.random.RandomState(0)
    ok = True
    for scale, drop in ((1.0, 0.0), (30.0, 0.0), (1.0, 0.3)):
        lg = rs.randn(2, 7, 3, 256).astype(np.float32) * scale
        w = rs.randn(*lg.shape).astype(np.float32)
        res = []
        for fn in (T.quantize_reinmax_limit, T.quantize_reinmax_limit_dense):
            x = torch.tensor(lg, requires_grad=True)
            cs, idx = fn(x, None if drop == 0 else T.Key(1), 1.0, drop)
            (cs * torch.tensor(w)).sum().backward()
            res.append((cs.detach().numpy(), idx.numpy(), x.grad.numpy()))
        (c1, i1, g1), (c2, i2, g2) = res
        jg = np.asarray(jax.grad(lambda l: jnp.sum(J.quantize_reinmax_limit(l, None, 1.0, 0.0)[0] * jnp.array(w)))(jnp.array(lg)))
        e_dense = float(np.abs(g1 - g2).max() / np.abs(g2).max())
        e_jax = float(np.abs(g1 - jg).max() / np.abs(jg).max()) if drop == 0 else 0.0
        this = np.array_equal(c1, c2) and np.array_equal(i1, i2) and e_dense < 1e-5 and e_jax < 1e-5
        ok &= this
        print(f"reinmax torch scale={scale} drop={drop}: forward exact={np.array_equal(c1, c2)}, grad vs dense {e_dense:.1e}, "
              f"vs JAX {e_jax:.1e}  {'OK' if this else 'DIFF'}")
    return ok


if __name__ == "__main__":
    cases = [
        ("transformer ar", {}),
        ("transformer linear", dict(pardec_token_head="linear", codelm_token_head="linear", token_head_type="linears")),
        ("refine fixed + decode_future", dict(level_refine_passes=2, level_refine_window=1, upsampler_decode_future=2)),
        ("refine variable + sink/xsa/window", dict(level_refine_passes=2, level_refine_window=1,
                                                   level_refine_layout="variable", use_sink=True, use_xsa=True,
                                                   attn_window=16)),
        ("shared levels + ncodes 2", dict(share_across_levels=True, upsampler_ncodes=2, bos_rate_mode="relative")),
        ("cycles memoryless rollout", dict(level_cycles=2)),
        ("cycles memoryless pss", dict(level_cycles=2, level_cycle_input="pss")),
        ("cycles stack gt + refine", dict(level_cycles=3, level_cycle_mode="stack", level_cycle_input="gt",
                                          level_refine_passes=2, level_refine_window=1)),
        ("cycles stack, 1 gen cycle", dict(level_cycles=3, level_cycle_mode="stack", gen_level_cycles=1)),
        ("cycles stack rollout linear", dict(level_cycles=3, level_cycle_mode="stack", pardec_token_head="linear",
                                             codelm_token_head="linear", token_head_type="linears")),
        ("gru backbone", dict(codelm_backbone="gru", downsampler_backbone="gru", upsampler_backbone="gru")),
        ("linear_gru backbone", dict(codelm_backbone="linear_gru", downsampler_backbone="linear_gru",
                                     upsampler_backbone="linear_gru")),
        ("ssm backbone + stack cycles", dict(codelm_backbone="ssm", downsampler_backbone="ssm", upsampler_backbone="ssm",
                                             level_cycles=2, level_cycle_mode="stack")),
        ("mixed backbones", dict(codelm_backbone=("transformer", "ssm"), downsampler_backbone="linear_gru",
                                 upsampler_backbone=("gru", "transformer"))),
        ("context own_embed", dict(context_source="own_embed")),
        ("remat chunks + codelm_remat", dict(downsampler_remat_chunks=3, upsampler_remat_chunks=(5, 2), remat=True,
                                             codelm_remat=False, level_refine_passes=2, level_refine_window=1)),
        ("remat chunks, ssm + stack", dict(downsampler_remat_chunks=2, upsampler_remat_chunks=3, codelm_backbone="ssm",
                                           upsampler_backbone="ssm", level_cycles=2, level_cycle_mode="stack")),
        ("shared ds/up LM + shared_embed", dict(share_downsampler_upsampler_lm=True, upsampler_d_model=32,
                                                upsampler_n_layers=1, upsampler_n_heads=2, upsampler_n_kv_heads=1,
                                                context_source="shared_embed")),
        ("codelm_upper unshared + bos + freeze", dict(context_source="codelm_upper", use_codelm_bos=True,
                                                      codelm_bos_rates=2, curriculum_mode="freeze")),
        ("codelm_upper shared + stack cycles", dict(context_source="codelm_upper", share_across_levels=True,
                                                    level_cycles=2, level_cycle_mode="stack", use_codelm_bos=True)),
        ("pss upsampler", dict(upsampler_pss_passes=(3, -1))),
        ("pss both sides + ncodes 2 + refine", dict(upsampler_pss_passes=2, downsampler_pss_passes=-1, downsampler_ncodes=2,
                                                    upsampler_ncodes=2, level_refine_passes=2, level_refine_window=1)),
        ("pss + upsampler_rollout + stack cycles", dict(upsampler_pss_passes=-1, upsampler_rollout=True, level_cycles=2,
                                                        level_cycle_mode="stack")),
    ]
    only = sys.argv[1:]
    results = [(n, check(n, **kw)) for n, kw in cases if not only or n in only]
    if not only or "reinmax" in only:
        results.append(("reinmax fast vs dense (torch) + vs JAX", check_reinmax()))
    print()
    for n, ok in results:
        print(f"{n}: {'PASS' if ok else 'FAIL'}")
    print("PASS all" if all(ok for _, ok in results) else "FAIL")
    sys.exit(0 if all(ok for _, ok in results) else 1)
