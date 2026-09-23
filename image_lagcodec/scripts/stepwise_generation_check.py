"""Timestep-by-timestep decode check on a trained checkpoint (pass-1, level 0): at every step, hidden state
+ token head under (a) teacher-forced input and (b) free-run (own previous argmax) input; also checks the
AR token head's generate vs a greedy reference built from its teacher-forced head.
Usage (TPU or JAX_PLATFORMS=cpu): python3 -m image_lagcodec.scripts.stepwise_generation_check <run_name> [n_images]
"""
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common
from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, dataset_from_config, images_to_positions, pixel_order_for, code_embed_proj,
    load_config_module, CONFIG_FIELDS, pardec_block_step, pardec_block_chunk_step,
    token_ar_teacher_forced,
)

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


def group_state(level, ctx_idx, extra_ctx_idx, G):
    Bc, n_blocks, _ = ctx_idx.shape
    D = level.bos_embed.shape[-1]
    n_groups = -(-n_blocks // G)
    B2 = Bc * n_groups
    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    ctx_flat, rope, valid, Wg, extra_len = level._pardec_ctx_rows(ctx_tok, extra_ctx_idx, G, n_groups, n_blocks)
    rope_flat = jnp.broadcast_to(rope[None], (Bc, n_groups, rope.shape[1])).reshape(B2, -1)
    key_valid = jnp.concatenate([valid, jnp.ones((B2, 1 + G * level.K), dtype=bool)], axis=1)
    rope_bos = jnp.broadcast_to(jnp.array([(g + 1) * G for g in range(n_groups)])[None], (Bc, n_groups)).reshape(B2)
    return dict(ctx=ctx_flat, rope=rope_flat, key_valid=key_valid, rope_bos=rope_bos, Wg=Wg, extra_len=extra_len,
                n_groups=n_groups, B2=B2, D=D, per_group_len=Wg + extra_len + 1 + G * level.K)


def greedy_ar_reference(level, h):
    """greedy AR token head built purely from the teacher-forced head (feeding own argmax back)."""
    chunks = level.in_pq_chunks
    tgt = jnp.zeros(h.shape[:-1] + (chunks,), dtype=jnp.int32)
    for m in range(chunks):
        lg = token_ar_teacher_forced(level.token_in_proj, level.token_member_embed, level.token_norm1, level.token_attn,
                                     level.token_ln_f, level.token_out_head, level.token_dim, level.in_code_vocab, h, tgt)
        tgt = tgt.at[..., m].set(jnp.argmax(lg[..., m, :], -1))
    return tgt


def run_mode(level, st, gt_flat, mode, return_h=False):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    hd = st["D"] // level.n_heads
    K = level.K
    Kspan = st["per_group_len"] - st["Wg"] - st["extra_len"] - 1
    plen = st["per_group_len"]
    ck = jnp.zeros((len(blocks), st["B2"], level.n_kv_heads, plen, hd))
    cv = jnp.zeros_like(ck)

    def step(x, ck, cv, pos, rp):
        nk, nv = [], []
        for i, blk in enumerate(blocks):
            x, a, b = pardec_block_step(blk, x, ck[i], cv[i], pos, rp, st["key_valid"], plen)
            nk.append(a)
            nv.append(b)
        return ln_f(x), jnp.stack(nk), jnp.stack(nv)

    bos = jnp.broadcast_to(level.bos_embed, (st["B2"], 1, st["D"]))
    chunk = jnp.concatenate([st["ctx"], bos], 1)
    crope = jnp.concatenate([st["rope"], st["rope_bos"][:, None]], 1)
    x = chunk
    nk, nv = [], []
    for i, blk in enumerate(blocks):
        x, a, b = pardec_block_chunk_step(blk, x, ck[i], cv[i], jnp.array(0), crope, st["key_valid"], plen)
        nk.append(a)
        nv.append(b)
    h_all = [ln_f(x)[:, -1]]
    ck, cv = jnp.stack(nk), jnp.stack(nv)
    pos = st["Wg"] + st["extra_len"] + 1
    rp = st["rope_bos"] + 1
    rows, logit_list, val_list, h_list = [], [], [], []
    prev_in = None
    for t in range(Kspan):
        if t > 0:
            h, ck, cv = step(prev_in, ck, cv, pos, rp)
            pos += 1
            rp = rp + 1
        else:
            h = h_all[0]
        gen_val, _ = level._token_generate_ar(h, None, True, 1.0) if level.token_head_type == "ar" \
            else (jnp.argmax(level._token_logits_linears(h), -1), None)
        ref_val = greedy_ar_reference(level, h) if level.token_head_type == "ar" else gen_val
        tf_lg = level._token_teacher_forced_ar(h, gt_flat[:, t]) if level.token_head_type == "ar" \
            else level._token_logits_linears(h)
        tf_val = jnp.argmax(tf_lg, -1)
        gt = gt_flat[:, t]
        logit_list.append(np.asarray(tf_lg))
        h_list.append(h)
        val_list.append(np.asarray(gen_val))
        rows.append(dict(
            t=t, h_nan=int(jnp.isnan(h).sum()), h_absmax=float(jnp.nanmax(jnp.abs(h))),
            gen_invalid=float(((gen_val < 0) | (gen_val > 255)).mean()), gen_zero=float((gen_val == 0).mean()),
            gen_vs_ref=float((gen_val == ref_val).mean()), gen_acc=float((gen_val == gt).mean()),
            tfhead_acc=float((tf_val == gt).mean())))
        prev_in = level._dec_embed_target(gt if mode == "tf" else gen_val)
    if return_h:
        return h_list
    return rows, np.stack(logit_list, 1), np.stack(val_list, 1)


def main():
    run = sys.argv[1]
    n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ck = sorted((REPO_ROOT / f"image_lagcodec/logs/{run}/checkpoints").iterdir())[-1]
    cv = load_config_module(REPO_ROOT / f"image_lagcodec/logs/{run}/config_{run}.py")
    cv.pop("label_fn", None)
    cfg = Config(**{k: cv[k] for k in CONFIG_FIELDS if k in cv})
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", HierEncDec(jax.random.PRNGKey(0), cfg))
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    (_, _), (val, _) = dataset_from_config(cv, REPO_ROOT)
    fb = jnp.array(images_to_positions(val[:n_img], cfg, pixel_order_for(cfg)))
    L0, L1 = model.levels
    e0 = L0.encode(code_embed_proj(fb, L0.own_input_embed, L0.own_input_proj), fb, rng=None)
    e1 = L1.encode(code_embed_proj(e0["code_soft"], L1.own_input_embed, L1.own_input_proj), e0["code_idx"], rng=None)
    G = cfg.decoder_ncodes[0]
    Kspan = G * L0.K
    extra = [e1["code_idx"]] if L0.cond_depth > 1 else None
    st = group_state(L0, e0["code_idx"], extra, G)
    n_groups = st["n_groups"]
    gt_flat = jnp.stack([fb[:, g * Kspan:(g + 1) * Kspan] for g in range(n_groups)], 1).reshape(st["B2"], Kspan, -1)
    print(f"backend={jax.default_backend()} run={run} G={G} Kspan={Kspan} rows={st['B2']} head={L0.token_head_type}")
    import json
    dense_lg, _, _, _, _, _ = L0.decode_logits_and_target_pardec(fb, e0["code_soft"], G, rng=None,
                                                                 extra_ctx_code_soft=[e1["code_soft"]] if extra else None)
    dense_lg = np.asarray(dense_lg).reshape(st["B2"], Kspan, *dense_lg.shape[2:])
    saved = {"dense_tf_logits": dense_lg, "gt": np.asarray(gt_flat)}
    report = {}
    for mode in ("tf", "free"):
        rows, inc_lg, gen_vals = run_mode(L0, st, gt_flat, mode)
        saved[f"incremental_{mode}_headlogits"] = inc_lg
        saved[f"incremental_{mode}_generated"] = gen_vals
        if mode == "tf":
            diff = np.abs(dense_lg - inc_lg).max(axis=(0, 2, 3)) if inc_lg.ndim == 4 else None
            argeq = (dense_lg.argmax(-1) == inc_lg.argmax(-1)).all(axis=(0, 2))
            report["tf_dense_vs_incremental"] = dict(
                max_abs_logit_diff_per_t=diff.tolist(),
                argmax_equal_per_t=[bool(a) for a in argeq],
                within_tol_strict_1em3_per_t=[bool(d <= 1e-3) for d in diff],
                within_tol_lax_1em1_per_t=[bool(d <= 1e-1) for d in diff],
                frac_argmax_equal_per_t=(dense_lg.argmax(-1) == inc_lg.argmax(-1)).mean(axis=(0, 2)).tolist())
            print("dense-vs-incremental (teacher-forced input), per t:")
            print("t argmax_all_equal within1e-3 within1e-1 frac_argmax_eq maxdiff")
            for t in range(Kspan):
                print(f"{t:2d} {argeq[t]!s:5} {diff[t] <= 1e-3!s:5} {diff[t] <= 1e-1!s:5} "
                      f"{report['tf_dense_vs_incremental']['frac_argmax_equal_per_t'][t]:.3f} {diff[t]:.4f}")
        print(f"--- input mode: {mode} (tf=real previous token, free=own previous argmax) ---")
        print("t  h_nan h_absmax gen_invalid gen_zero gen==ref gen_acc tfhead_acc")
        for r in rows:
            print(f"{r['t']:2d} {r['h_nan']:5d} {r['h_absmax']:8.2f} {r['gen_invalid']:11.3f} {r['gen_zero']:8.3f} "
                  f"{r['gen_vs_ref']:7.3f} {r['gen_acc']:7.3f} {r['tfhead_acc']:10.3f}")
        report[f"steps_{mode}"] = rows
    out = REPO_ROOT / f"image_lagcodec/logs/{run}"
    tag = jax.default_backend()
    np.savez_compressed(out / f"stepwise_check_{tag}.npz", **saved)
    (out / f"stepwise_check_{tag}.json").write_text(json.dumps(report, indent=1))
    print("saved", out / f"stepwise_check_{tag}.npz", "and .json")


if __name__ == "__main__":
    main()
