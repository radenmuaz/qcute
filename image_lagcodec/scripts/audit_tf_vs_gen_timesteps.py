"""CPU-only (safe next to a live TPU job): per-timestep audit of teacher-forced decode vs free generation on a
run_lagcodec_res / run_lagcodec_res_denoise checkpoint. Per level: module TF paths, generation with real codes,
gen-vs-TF self-consistency, cascade error attribution, input-source patching, noise sensitivity, and the exact
level-0 timesteps (of 1024) where the argmax differs from the target.
Usage: python3 -m image_lagcodec.scripts.audit_tf_vs_gen_timesteps <run> [--module run_lagcodec_res] [--ckpt DIR]
       [--n 32] [--tops 0,1,2] [--dtype f32|bf16] [--list 3] [--out FILE.npz]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import importlib
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import image_lagcodec.eqx_common as eqx_common


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = jnp.repeat(k, rep, 1), jnp.repeat(v, rep, 1)
    lg = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)
    T = q.shape[2]
    t, s = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
    m = (s <= t + lookahead) & ((s >= t - window) if window is not None else True)
    lg = jnp.where(m[None, None], lg, -jnp.inf)
    if sink is not None:
        sk = jnp.broadcast_to(sink[None, :, None, None].astype(lg.dtype), lg.shape[:3] + (1,))
        w = jax.nn.softmax(jnp.concatenate([lg, sk], -1), -1)[..., :-1]
    else:
        w = jax.nn.softmax(lg, -1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense
jax.config.update("jax_default_matmul_precision", "highest")
R = None


def fmt(a):
    return "[" + " ".join(f"{float(x):.3f}" for x in np.asarray(a).ravel()) + "]"


def tok_ok(p, t):
    return (np.asarray(p) == np.asarray(t)).all(-1)


def by_row(ok, rs):
    return ok.reshape(ok.shape[0], -1, rs).mean((0, 1))


def up_ctx(model, lvl, ctx_idx):
    cl = getattr(model, "context_codelm_for", model.codelm_for)(lvl)
    rid = getattr(model, "context_codelm_rate_id", model.codelm_bos_rate_id)(lvl)
    return R.pardec_context_hidden(cl, model.upsampler_for(lvl), ctx_idx, model.cfg, rid, None,
                                   group_size=model.cfg.upsampler_ncodes[lvl])


def _hidden(model, lvl, row_inputs, ctx_idx, draft_seq, draft_len, draft_fill):
    nc = model.cfg.upsampler_ncodes[lvl]
    kw = dict(draft_seq=draft_seq, draft_len=draft_len, draft_fill=draft_fill) if draft_len > 0 else {}
    return R.pardec_score(model.upsampler_for(lvl), row_inputs, up_ctx(model, lvl, ctx_idx), context_group_size=nc,
                          output_group_size=nc, rate_id=model.bos_rate_id(lvl), output_expansion=model.K(lvl),
                          return_hidden=True, **kw)


def _predict(model, lvl, row_inputs, ctx_idx, draft_seq, draft_len, draft_fill):
    # greedy tokens at every position given `row_inputs` as the row's token inputs; digits self-fed (as generation)
    up = model.upsampler_for(lvl)
    hid = _hidden(model, lvl, row_inputs, ctx_idx, draft_seq, draft_len, draft_fill)
    if up.token_head == "linear":
        return jnp.argmax(R.reshape_pq(hid @ up.output_head_linear, up.output_chunks, up.output_vocab), -1)
    return R.token_ar_generate(up.token_in_proj, up.token_member_embed, up.token_norm1, up.token_attn, up.token_ln_f,
                               up.token_out_head, up.output_chunks, hid, jax.random.PRNGKey(0), True, 1.0, 0)[0]


def _tf_logits(model, lvl, target, ctx_idx, draft_seq, draft_len, draft_fill):
    # digit-teacher-forced logits on `target` (row inputs and digit inputs both from target)
    nc = model.cfg.upsampler_ncodes[lvl]
    kw = dict(draft_seq=draft_seq, draft_len=draft_len, draft_fill=draft_fill) if draft_len > 0 else {}
    return R.pardec_score(model.upsampler_for(lvl), target, up_ctx(model, lvl, ctx_idx), context_group_size=nc,
                          output_group_size=nc, rate_id=model.bos_rate_id(lvl), output_expansion=model.K(lvl), **kw)[0]


def _module_passes(model, lvl, target, ctx_idx, force):
    V = model.cfg.code_vocab[lvl]
    ctx_soft = jax.nn.one_hot(ctx_idx, V, dtype=model.upsampler_for(lvl).bos_embed.dtype)
    outs = R.decode_logits_and_target_multipass(model, lvl, target, ctx_soft, model.cfg.upsampler_ncodes[lvl], rng=None,
                                                force_teacher_forced=force, return_passes=True)
    return [o[0] for o in outs]


predict = eqx.filter_jit(_predict)
tf_logits = eqx.filter_jit(_tf_logits)
module_passes = eqx.filter_jit(_module_passes)


def pass_plan(cfg, lvl):
    n_pass = cfg.level_refine_passes[lvl]
    Pp = cfg.level_refine_window[lvl] * cfg.upsampler_ncodes[lvl] * (cfg.strides[lvl] if cfg.strides[lvl] != -1 else 1)
    fixed = n_pass > 1 and cfg.level_refine_layout == "fixed"
    return n_pass, Pp, ("mask" if fixed else "zero"), (Pp if fixed else 0)


def gen_passes(model, lvl, ctx_idx):
    # free generation, every refine pass (same calls as decode_generate_multipass)
    cfg = model.cfg
    n_pass, Pp, fill, p1_len = pass_plan(cfg, lvl)
    nc = cfg.upsampler_ncodes[lvl]
    if p1_len:
        out = [R._decode_generate_pardec_jit(model, lvl, ctx_idx, nc, True, 1.0, 0, None, Pp, "mask")]
    else:
        out = [R._decode_generate_pardec_jit(model, lvl, ctx_idx, nc, True, 1.0, 0)]
    for _ in range(n_pass - 1):
        out.append(R._decode_generate_pardec_jit(model, lvl, ctx_idx, nc, True, 1.0, 0, out[-1], Pp, fill))
    return out


def predict_passes(model, lvl, row_inputs_per_pass, ctx_idx, drafts=None):
    # predict() for every refine pass; pass p's draft = drafts[p-1] if given else the previous pass's own prediction
    cfg = model.cfg
    n_pass, Pp, fill, p1_len = pass_plan(cfg, lvl)
    out = [predict(model, lvl, row_inputs_per_pass[0], ctx_idx, None, p1_len, "mask")]
    for p in range(1, n_pass):
        d = out[-1] if drafts is None else drafts[p - 1]
        out.append(predict(model, lvl, row_inputs_per_pass[p], ctx_idx, d.astype(jnp.int32), Pp, fill))
    return out


def timeline(err_1d, width=64):
    # one char per timestep: '.' max-channel |err|<=4, 'o' <=16, 'x' >16; groups of 4 = one decode row (2x2 block)
    s = "".join("." if e <= 4 else ("o" if e <= 16 else "x") for e in err_1d)
    return "\n".join("      " + " ".join(s[i + k:i + k + 4] for k in range(0, width, 4)) for i in range(0, len(s), width))


def vmse(p, t):
    return float(((np.asarray(p).astype(np.float64) - np.asarray(t)) ** 2).mean())


def mse_steps(p, t, rs):
    # mse per AR step inside a row: (row pos, digit) flattened in generation order
    e = (np.asarray(p).astype(np.float64) - np.asarray(t)) ** 2
    return e.reshape(e.shape[0], -1, rs, e.shape[-1]).mean((0, 1))


def mse_pos(p, t, rs):
    return mse_steps(p, t, rs).mean(-1)


def _digit_tf(model, lvl, hid, digit_target):
    up = model.upsampler_for(lvl)
    return jnp.argmax(R.token_ar_teacher_forced(up.token_in_proj, up.token_member_embed, up.token_norm1, up.token_attn,
                                                up.token_ln_f, up.token_out_head, up.token_dim, up.output_vocab,
                                                hid, digit_target), -1)


hidden = eqx.filter_jit(_hidden)
digit_tf = eqx.filter_jit(_digit_tf)


def audit_split(name, imgs, model, cfg, po, label_fn, tops, n_list, dump):
    B = imgs.shape[0]
    flat = jnp.array(R.images_to_positions(imgs, cfg, po))
    tok0 = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
    top_max = max(tops)
    print(f"\n================ split={name} n={B} ================", flush=True)

    # ---- encoder chain: generation path vs train-eval path, and code vs label_fn target
    codes, raw_g, raw_t, tgt_t = [], tok0, tok0, tok0
    print("[encoder] per level: gen-path code == train-eval-path code | code vs label_fn target")
    for i in range(top_max + 1):
        cl, ds = model.codelm_for(i), model.downsampler_for(i)
        og = R.encode_pardec_downsampler_generate(cl, ds, raw_g, model.K(i), cfg, rate_id=model.bos_rate_id(i),
                                                  codelm_rate_id=model.codelm_bos_rate_id(i), rng=jax.random.PRNGKey(i),
                                                  greedy=True, temperature=cfg.gen_temperature, top_k=cfg.gen_top_k,
                                                  downsampler_ncodes=cfg.downsampler_ncodes[i])
        ot = R.encode_pardec_downsampler(cl, ds, raw_t, tgt_t, flat, cfg, po, label_fn, model.K(i),
                                         rate_id=model.bos_rate_id(i), codelm_rate_id=model.codelm_bos_rate_id(i),
                                         rng=None, downsampler_ncodes=cfg.downsampler_ncodes[i])
        lab = np.asarray(label_fn(flat, cfg, po, og["code_idx"].shape[1], cfg.pq_chunks[i], cfg.code_vocab[i]))
        cg, ct = np.asarray(og["code_idx"]), np.asarray(ot["code_idx"])
        print(f"  level {i}: n_codes={cg.shape[1]} same_code={float((cg == ct).mean()):.4f} "
              f"digit==label {float((cg == lab).mean()):.4f} mse(code,label)={vmse(cg, lab):.2f} per digit {fmt(((cg - lab.astype(np.float64)) ** 2).mean((0, 1)))} "
              f"unique_codes={len(np.unique(cg.reshape(-1, cg.shape[-1]), axis=0))}/{cg.shape[0] * cg.shape[1]}", flush=True)
        codes.append(og["code_idx"])
        raw_g = og["code_soft"]
        raw_t, tgt_t = ot["code_soft"], ot["code_idx"]
    targets = [tok0] + codes[:-1]  # targets[i] = what level i's upsampler decodes

    res = {}
    for lvl in range(top_max + 1):
        T, C = targets[lvl], codes[lvl]
        Tn, Cn = np.asarray(T), np.asarray(C)
        rs = cfg.upsampler_ncodes[lvl] * model.K(lvl)
        n_pass, Pp, fill, p1_len = pass_plan(cfg, lvl)
        print(f"\n[level {lvl}] REAL codes as context: seq_len={Tn.shape[1]} row={rs} tokens x {Tn.shape[-1]} digits, "
              f"refine_passes={n_pass} draft_len={Pp} fill={fill}", flush=True)
        blk = Tn.reshape(B, -1, rs, Tn.shape[-1]).astype(np.float64)
        rep = lambda x: np.repeat(np.asarray(x), rs // cfg.upsampler_ncodes[lvl], axis=1)
        print(f"  baselines (mse vs target): repeat the context code x{rs}={vmse(rep(Cn), Tn):.2f} | "
              f"oracle row mean={float(((blk - blk.mean(2, keepdims=True)) ** 2).mean()):.2f} | "
              f"global mean={float(((Tn - Tn.mean((0, 1), keepdims=True)) ** 2).mean()):.2f}")
        m_tf = [np.asarray(jnp.argmax(l, -1)) for l in module_passes(model, lvl, T, C, True)]
        m_ev = [np.asarray(jnp.argmax(l, -1)) for l in module_passes(model, lvl, T, C, False)]
        g = [np.asarray(x) for x in gen_passes(model, lvl, C)]
        g_full = np.asarray(R.decode_generate_multipass(model, lvl, C, cfg.upsampler_ncodes[lvl], greedy=True))
        p_gt = [np.asarray(x) for x in predict_passes(model, lvl, [T] * n_pass, C)]
        print("  A = module TF path (GT digits + GT row tokens; = TF_SANITY/dec_loss);  B = GT row tokens, own digits;")
        print("  C = module eval path (rng=None);  D = free generation (own digits + own row tokens)")
        for p in range(n_pass):
            rows = (("A", m_tf[p]), ("B", p_gt[p]), ("C", m_ev[p]), ("D", g[p]))
            for nm, pr in rows:
                print(f"  pass {p + 1} {nm}: mse={vmse(pr, Tn):8.2f} exact digit={float((pr == Tn).mean()):.4f} token={tok_ok(pr, Tn).mean():.4f} | "
                      f"mse by AR step (row pos x digit) {fmt(mse_steps(pr, Tn, rs))}")
            print(f"  pass {p + 1}: C==D digits {float((m_ev[p] == g[p]).mean()):.4f} | B==C digits {float((p_gt[p] == m_ev[p]).mean()):.4f}", flush=True)
        print(f"  decode_generate_multipass == last gen pass: {bool((g_full == g[-1]).all())}")

        # ---- self-consistency: score generation's own tokens with the dense teacher-forced code
        q = [np.asarray(x) for x in predict_passes(model, lvl, [jnp.asarray(x) for x in g], C, drafts=[jnp.asarray(x) for x in g[:-1]])]
        for p in range(n_pass):
            same = (q[p] == g[p])
            print(f"  self-consistency pass {p + 1}: dense-TF argmax on generated tokens == generated tokens: "
                  f"digits {float(same.mean()):.6f} (n_diff_tokens={int((~same.all(-1)).sum())} of {same.shape[0] * same.shape[1]})")

        # ---- error correlation along the AR chain (signed error, last pass)
        def chain_corr(pr):
            e = (pr.astype(np.float64) - Tn).reshape(B, -1, rs * Tn.shape[-1])
            e = e.reshape(-1, e.shape[-1])
            return [float(np.corrcoef(e[:, k - 1], e[:, k])[0, 1]) for k in range(1, e.shape[-1])]
        print(f"  signed-error correlation between consecutive AR steps (last pass): A {fmt(chain_corr(m_tf[-1]))}")
        print(f"                                                                     D {fmt(chain_corr(g[-1]))}")
        e_g = (g[-1].astype(np.float64) - Tn).reshape(B, -1, rs, Tn.shape[-1])
        print(f"  D: mean signed err by row pos {fmt(e_g.mean((0, 1, 3)))} | within-row std of the error {float(e_g.std(2).mean()):.2f} "
              f"vs across-rows std of the row-mean error {float(e_g.mean(2).std()):.2f}", flush=True)

        # ---- last pass helper on arbitrary inputs
        d_tf = p_gt[0] if n_pass > 1 else None

        def last(row, ctx, draft):
            if n_pass == 1:
                return np.asarray(predict(model, lvl, row, ctx, None, p1_len, "mask"))
            return np.asarray(predict(model, lvl, row, ctx, jnp.asarray(draft).astype(jnp.int32), Pp, fill))
        base = p_gt[-1]

        # ---- input-source patching: one source comes from another image
        roll = lambda x: jnp.roll(jnp.asarray(x), 1, axis=0)
        print("  source patching (B, last pass; one source taken from another image): mean |prediction change| by row pos | mse vs target by row pos")
        variants = [("ctx code", last(T, roll(C), d_tf)), ("row tokens", last(roll(T), C, d_tf))]
        if n_pass > 1:
            variants.append(("draft", last(T, C, roll(d_tf))))
        print(f"    unpatched            :                                 | mse {fmt(mse_pos(base, Tn, rs))}")
        for nm, pr in variants:
            ch = np.abs(pr.astype(np.float64) - base).reshape(B, -1, rs, Tn.shape[-1]).mean((0, 1, 3))
            print(f"    patch {nm:10s}     : |change| {fmt(ch)} | mse {fmt(mse_pos(pr, Tn, rs))}")

        # ---- gain: shift every digit of one source by +d, how far does the prediction move (per unit)
        Vt, Vc = (cfg.code_vocab[lvl - 1] if lvl > 0 else cfg.code_vocab[0]), cfg.code_vocab[lvl]
        shift = lambda x, d, V: jnp.asarray(np.clip(np.asarray(x) + d, 0, V - 1).astype(np.int32))
        print("  gain (B, last pass): (prediction after +d on every digit of a source - prediction) / d, by row pos")
        for d in (4, 16):
            gn = lambda pr: fmt(((pr.astype(np.float64) - base) / d).reshape(B, -1, rs, Tn.shape[-1]).mean((0, 1, 3)))
            line = f"    d={d:2d}: row tokens {gn(last(shift(T, d, Vt), C, d_tf))} | ctx code {gn(last(T, shift(C, d, Vc), d_tf))}"
            if n_pass > 1:
                line += f" | draft {gn(last(T, C, shift(d_tf, d, Vt)))}"
            print(line, flush=True)
        # digit-level gain: same hidden (GT row tokens), digit inputs shifted
        kw = (jnp.asarray(d_tf).astype(jnp.int32), Pp, fill) if n_pass > 1 else (None, p1_len, "mask")
        hid = hidden(model, lvl, T, C, *kw)
        a0 = np.asarray(digit_tf(model, lvl, hid, T)).astype(np.float64)
        for d in (4, 16):
            a1 = np.asarray(digit_tf(model, lvl, hid, shift(T, d, Vt))).astype(np.float64)
            print(f"    d={d:2d}: digit inputs (previous digits of the same token +d): gain per digit {fmt(((a1 - a0) / d).mean((0, 1)))}")

        # ---- random +-d noise on one source: mse by row pos (B, last pass)
        rng_np = np.random.default_rng(0)
        noisy = lambda x, d, V: jnp.asarray(np.clip(np.asarray(x) + rng_np.choice([-d, d], size=np.asarray(x).shape), 0, V - 1).astype(np.int32))
        print("  noise (B, last pass): mse by row pos after random +-d on every digit of one source")
        for d in (2, 8, 16):
            line = f"    d={d:2d}: row tokens {fmt(mse_pos(last(noisy(T, d, Vt), C, d_tf), Tn, rs))} | ctx code {fmt(mse_pos(last(T, noisy(C, d, Vc), d_tf), Tn, rs))}"
            if n_pass > 1:
                line += f" | draft {fmt(mse_pos(last(T, C, noisy(d_tf, d, Vt)), Tn, rs))}"
            print(line, flush=True)

        # ---- confidence of the digit-TF logits (last pass)
        lg = np.asarray(module_passes(model, lvl, T, C, True)[-1]).astype(np.float64)
        lg = lg - lg.max(-1, keepdims=True)
        pr = np.exp(lg) / np.exp(lg).sum(-1, keepdims=True)
        p_gt_tok = np.take_along_axis(pr, Tn[..., None], -1)[..., 0]
        rank = (pr > p_gt_tok[..., None]).sum(-1)
        ent = -(pr * np.log(np.maximum(pr, 1e-12))).sum(-1)
        vals = np.arange(pr.shape[-1])
        mean_pred = (pr * vals).sum(-1)
        sd = np.sqrt((pr * (vals - mean_pred[..., None]) ** 2).sum(-1))
        print(f"  A distribution: max_prob={pr.max(-1).mean():.3f} p(GT)={p_gt_tok.mean():.3f} nll={-np.log(np.maximum(p_gt_tok, 1e-12)).mean():.3f} "
              f"entropy={ent.mean():.3f} nats predictive std={sd.mean():.2f} | GT rank median={np.median(rank):.0f} | "
              f"mse of the distribution MEAN={vmse(mean_pred, Tn):.2f} vs argmax={vmse(m_tf[-1], Tn):.2f}")
        print(f"    entropy by AR step (row pos x digit): {fmt(ent.reshape(B, -1, rs, ent.shape[-1]).mean((0, 1)))}", flush=True)
        res[lvl] = dict(T=Tn, tf=m_tf, ev=m_ev, gen=g, C=Cn)
        for k, v in (("target", Tn), ("ctx", Cn), ("tf", m_tf[-1]), ("eval", m_ev[-1]), ("gen", g[-1]), ("gen_p1", g[0]), ("ownd", p_gt[-1])):
            dump[f"{name}_l{lvl}_{k}"] = v

    # ---- cascades from each top: generation fed its own codes downward
    casc0 = {}
    for top in tops:
        cur = codes[top]
        print(f"\n[cascade top={top}] generation fed its own codes downward", flush=True)
        for i in range(top, -1, -1):
            Tn = res[i]["T"]
            rs = cfg.upsampler_ncodes[i] * model.K(i)
            cerr = vmse(cur, res[i]["C"])
            out = np.asarray(R.decode_generate_multipass(model, i, cur, cfg.upsampler_ncodes[i], greedy=True))
            # how much of this level's error is explained by the context error: split rows by ctx |err|
            ce = np.abs(np.asarray(cur).astype(np.float64) - res[i]["C"]).max(-1).reshape(B, -1, cfg.upsampler_ncodes[i]).max(-1)
            re = ((out.astype(np.float64) - Tn) ** 2).reshape(B, -1, rs, Tn.shape[-1]).mean((2, 3))
            bins = [(0, 0), (1, 4), (5, 16), (17, 255)]
            parts = " ".join(f"|ctx err| {lo}-{hi}: {re[(ce >= lo) & (ce <= hi)].mean():.1f} (n={int(((ce >= lo) & (ce <= hi)).sum())})"
                             for lo, hi in bins if ((ce >= lo) & (ce <= hi)).any())
            print(f"  level {i}: ctx mse={cerr:.2f} -> output mse={vmse(out, Tn):.2f} exact digit={float((out == Tn).mean()):.4f} "
                  f"by row pos {fmt(mse_pos(out, Tn, rs))} | row mse by ctx error: {parts}", flush=True)
            cur = jnp.asarray(out)
            if i == 0:
                casc0[top] = out
                img = R.positions_to_image(out, cfg, po)
                print(f"    pixel_mse={R.pixel_mse(img, imgs.astype(np.uint8)):.2f} byte_acc={float((out == np.asarray(flat)).mean()):.4f}")
                dump[f"{name}_cascade_top{top}"] = out

    # ---- level-0 exact timesteps
    T0 = res[0]["T"]
    L = T0.shape[1]
    series = [("A TF (GT digits+row)", res[0]["tf"][-1]), ("C module eval path", res[0]["ev"][-1]),
              ("D generation, real ctx (top=0)", res[0]["gen"][-1])]
    series += [(f"cascade top={t}", casc0[t]) for t in tops if t > 0]
    print(f"\n[level-0 timesteps] seq_len={L}; 'exact' = all {T0.shape[-1]} digits of the token equal the target; err = max-channel |pred-gt|")
    blk = T0.reshape(B, -1, 4, T0.shape[-1]).astype(np.float64)
    contrast = (blk.max(2) - blk.min(2)).max(-1)  # (B, n_rows)
    qs = np.quantile(contrast, [0.25, 0.5, 0.75])
    cbin = np.digitize(contrast, qs)
    for nm, pred in series:
        ok = tok_ok(pred, T0)
        ae = np.abs(pred.astype(np.float64) - T0).max(-1)  # (B, L)
        e_t = ae.mean(0)
        print(f"  {nm}: timesteps not exact={1 - ok.mean():.4f} | err>4: {(ae > 4).mean():.4f} err>16: {(ae > 16).mean():.4f} err>32: {(ae > 32).mean():.4f} mean err={ae.mean():.2f}")
        g4 = lambda v, n: fmt([v.reshape(-1, 4, n)[:, j].mean() for j in range(4)])
        print(f"    mean err by t%4 {g4(e_t, 1)} | (t//4)%4 {g4(e_t, 4)} | (t//16)%4 {g4(e_t, 16)} | (t//64)%4 {g4(e_t, 64)} | t//256 {fmt(e_t.reshape(4, 256).mean(1))}")
        nx = 1.0 - ok.mean(0)
        print(f"    not-exact rate by t%4 {g4(nx, 1)} | (t//4)%4 {g4(nx, 4)} | (t//16)%4 {g4(nx, 16)}")
        order = np.argsort(-e_t)
        print(f"    worst timesteps (t:mean err): {' '.join(f'{int(t)}:{e_t[t]:.0f}' for t in order[:12])} | best: {' '.join(f'{int(t)}:{e_t[t]:.0f}' for t in order[::-1][:12])}")
        row_e = ae.reshape(B, -1, 4).mean(-1)
        print(f"    mean err by 2x2-block contrast quartile (cuts {fmt(qs)}): {fmt([row_e[cbin == k].mean() for k in range(4)])}")
        se = pred.astype(np.float64) - T0
        print(f"    signed err mean per channel {fmt(se.mean((0, 1)))} | corr(err R,G)={np.corrcoef(se[..., 0].ravel(), se[..., 1].ravel())[0, 1]:.3f} "
              f"corr(err G,B)={np.corrcoef(se[..., 1].ravel(), se[..., 2].ravel())[0, 1]:.3f} | mse per channel {fmt((se ** 2).mean((0, 1)))}", flush=True)
    for b in range(min(n_list, B)):
        for nm, pred in series:
            ok = tok_ok(pred, T0)[b]
            ae = np.abs(pred[b].astype(np.float64) - T0[b]).max(-1)
            bad = np.nonzero(~ok)[0]
            print(f"  {name} image {b} | {nm}: {len(bad)}/{L} timesteps not exact; {int((ae > 16).sum())} with err>16")
            print(f"    exact timesteps t = {np.nonzero(ok)[0].tolist()}")
            print(f"    err>16 timesteps t = {np.nonzero(ae > 16)[0].tolist()}")
            print(timeline(ae))
        print(flush=True)


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="run_lagcodec_res")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (default: latest in the run dir)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tops", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    ap.add_argument("--list", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default="train,val")
    a = ap.parse_args()
    R = importlib.import_module(f"image_lagcodec.{a.module}")
    if hasattr(R, "splash_attention"):
        R.splash_attention = _dense
    run_dir = REPO_ROOT / "image_lagcodec/logs" / a.run
    ck = Path(a.ckpt) if a.ckpt else R.find_latest_checkpoint(run_dir)
    cv = R.load_config_module(run_dir / f"config_{a.run}.py")
    label_fn = getattr(R, cv.pop("label_fn", "default_label_fn_jax"))
    cfg = R.Config(**{k: cv[k] for k in R.CONFIG_FIELDS if k in cv})
    model = eqx.tree_deserialise_leaves(ck / "model.eqx", R.LagCodecModel(jax.random.PRNGKey(0), cfg))
    dt = jnp.float32 if a.dtype == "f32" else jnp.bfloat16
    model = jax.tree_util.tree_map(lambda x: x.astype(dt) if eqx.is_inexact_array(x) else x, model)
    (train_np, _), (val_np, _) = R.load_cifar10(Path(cv.get("data_root") or REPO_ROOT / "datasets"))
    po = R.pixel_order_for(cfg)
    n_levels = len(cfg.strides)
    tops = a.tops if a.tops is not None else list(range(n_levels))
    print(f"run={a.run} module={a.module} ckpt={ck} backend={jax.default_backend()} dtype={a.dtype} n={a.n} tops={tops}")
    print(f"cfg: strides={cfg.strides} upsampler_ncodes={cfg.upsampler_ncodes} upsampler_window={cfg.upsampler_window} "
          f"downsampler_window={cfg.downsampler_window} refine_passes={cfg.level_refine_passes} refine_window={cfg.level_refine_window} "
          f"layout={cfg.level_refine_layout} upsampler_rollout={cfg.upsampler_rollout}/{cfg.upsampler_rollout_prob} "
          f"downsampler_rollout={cfg.downsampler_rollout}/{cfg.downsampler_rollout_prob} level_gt_drop={cfg.level_gt_drop if hasattr(cfg, 'level_gt_drop') else cv.get('level_gt_drop')} "
          f"pss={getattr(cfg, 'upsampler_pss_passes', None)} context_source={cfg.context_source} head={cfg.pardec_token_head}", flush=True)
    dump = {}
    for name in a.splits.split(","):
        imgs = (train_np if name == "train" else val_np)[:a.n]
        audit_split(name, imgs, model, cfg, po, label_fn, tops, a.list, dump)
    if a.out:
        np.savez_compressed(a.out, **dump)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
