"""CPU-only correctness check for image_lagcodec/run_lagcodec_res_denoise.py: dense teacher-forced
reference vs incremental generation (KV cache, or recurrent state for gru/linear_gru/ssm backbones), for
CodeLM's free-run and the PardecLM decode, plus same-level cycle checks (stack-slot dense-vs-KV,
slot causality, rollout no-leak, train-vs-gen re-encode, end-to-end finite loss/grads). Real CIFAR data.
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.res_denoise_consistency_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import image_lagcodec.eqx_common as eqx_common


def _dense(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    B, Hq, T, hd = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    if rep > 1:
        k = jnp.repeat(k, rep, axis=1)
        v = jnp.repeat(v, rep, axis=1)
    scores = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)
    if causal:
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(mask[None, None], scores, -1e9)
    w = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense
import image_lagcodec.run_lagcodec_res_denoise as R
R.splash_attention = _dense

B = 2
G = 4


def build_cfg(**over):
    kw = dict(
        codelm_d_model=(64, 64), codelm_n_layers=(2, 2), codelm_n_heads=(2, 2), codelm_n_kv_heads=(None, None),
        strides=(4, -1), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), byte_group=3,
        token_head_type="linears", upsampler_ncodes=(G, G),
        curriculum_mode="no_freeze",
        downsampler_d_model=(64, 64), downsampler_n_layers=(2, 2), downsampler_n_heads=(2, 2),
        downsampler_n_kv_heads=(2, 2), downsampler_window=(4, 4),
        upsampler_d_model=(64, 64), upsampler_n_layers=(2, 2), upsampler_n_heads=(2, 2),
        upsampler_n_kv_heads=(2, 2), upsampler_window=(4, 4),
        label_reg_weight=1.0,
    )
    kw.update(over)
    return R.Config(**kw)


def load_data(cfg):
    (train_np, _), _ = R.load_cifar10(REPO_ROOT / "datasets")
    pixel_order = R.pixel_order_for(cfg)
    imgs = train_np[:B]
    flat = jnp.array(R.images_to_positions(imgs, cfg, pixel_order))
    return flat, pixel_order


def run_encoder_free_run_kv_check(head="linear", **over):
    """New KV-cache _encoder_free_run must exactly match a slow reference that recomputes the
    entire forward pass at every step (the OLD, correct-by-construction implementation)."""
    cfg = build_cfg(codelm_token_head=head, **over)
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    codelm = model.codelm_for(0)
    ok = True
    print(f"[codelm_token_head={head} {over}]")
    for (C, V, T, P) in [(3, 256, 24, 8), (3, 256, 16, 5)]:
        toks = jnp.array(np.random.RandomState(0).randint(0, V, (B, T, C)))
        prompt = toks[:, :P]
        got = R.encoder_free_run(codelm, prompt, T, model.K(0), jax.random.PRNGKey(0), greedy=True)

        # slow reference: full-buffer recompute every step (mirrors the pre-KV-cache implementation)
        ref = prompt
        for t in range(P, T):
            h = R.encoder_hidden(codelm, R.code_embed_proj(ref, codelm.own_input_embed, codelm.own_input_proj))
            nxt = R.codelm_sample_next(codelm, h[:, -1], jax.random.PRNGKey(0), True, 1.0)[:, None].astype(ref.dtype)
            ref = jnp.concatenate([ref, nxt], axis=1)
        same = bool(jnp.array_equal(got, ref))
        kept = bool(jnp.array_equal(got[:, :P], prompt))
        this = same and kept
        ok &= this
        print(f"ENCODER free-run KV-cache T={T} P={P}: == slow full-recompute reference={same} "
              f"prompt kept={kept} {'OK' if this else 'WRONG'}")
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_pardec_dense_vs_kv_check(head="ar", **over):
    """pardec_score (dense, teacher-forced with the real target) vs pardec_generate (incremental
    KV-cache): greedy generation's argmax at each position must match dense scoring's argmax when
    dense is fed that SAME generated sequence as its target (self-consistency, the standard way to
    check a KV-cache decoder matches its own teacher-forced scoring path)."""
    cfg = build_cfg(pardec_token_head=head, **over)
    print(f"[pardec_token_head={head} {over}]")
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, pixel_order = load_data(cfg)
    codelm = model.codelm_for(0)

    tok0 = R.rgb_byte_pq_fn(flat, codelm.pq_chunks, codelm.code_vocab)
    x = R.code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
    out = codelm.encode(x, tok0, model.K(0), rng=None)
    ctx_code_soft = out["code_soft"]

    # 1) downsampler: pardec_generate (greedy) vs re-scoring that exact output with pardec_score
    gen_ctx = R.encoder_hidden(codelm, x)  # same h the downsampler would see (encode_pardec_downsampler's h)
    n_blocks = gen_ctx.shape[1] // model.K(0)
    label_tgt = R.rgb_label_fn_jax(flat, cfg, pixel_order, n_blocks, codelm.pq_chunks, codelm.code_vocab)
    logits_dense, target_dense, _, _ = R.pardec_score(model.downsampler_for(0), label_tgt, gen_ctx,
                                                        context_group_size=model.K(0), output_group_size=1)
    dense_argmax = jnp.argmax(logits_dense, axis=-1)
    ok1 = bool(jnp.array_equal(dense_argmax, label_tgt[:, :dense_argmax.shape[1]]))
    print(f"DOWNSAMPLER pardec_score self-check: argmax-of-dense-scoring vs real label target match "
          f"(sanity, not full gen check)={ok1}")

    # 2) upsampler: KV-cache pardec_generate's greedy output, re-scored with pardec_score (dense,
    # teacher-forced with the GENERATED sequence as target) must reproduce the SAME argmax --
    # this is the real dense-vs-KV-cache consistency check.
    x_ctx = R.code_embed_proj(ctx_code_soft, codelm.own_input_embed, codelm.own_input_proj)
    h_ctx = R.encoder_hidden(codelm, x_ctx)
    gen_out = R.pardec_generate(model.upsampler_for(0), h_ctx, context_group_size=G, output_group_size=G,
                                 rng=jax.random.PRNGKey(0), greedy=True, output_expansion=model.K(0))
    dense_logits, dense_target, _, _ = R.pardec_score(model.upsampler_for(0), gen_out, h_ctx,
                                                        context_group_size=G, output_group_size=G,
                                                        output_expansion=model.K(0))
    dense_argmax2 = jnp.argmax(dense_logits, axis=-1)
    n = min(dense_argmax2.shape[1], gen_out.shape[1])
    ok2 = bool(jnp.array_equal(dense_argmax2[:, :n], gen_out[:, :n]))
    mism = int((dense_argmax2[:, :n] != gen_out[:, :n]).sum())
    print(f"UPSAMPLER dense-vs-KV-cache self-consistency: greedy KV-cache generation reproduces "
          f"itself under dense re-scoring={ok2} mismatches={mism}/{n * gen_out.shape[0] * gen_out.shape[-1] if gen_out.ndim > 2 else n}")
    print(f"  {'CONSISTENT' if ok2 else 'DIVERGES'}")
    return ok2


def run_pardec_refine_check(head="ar", window=1, **over):
    """Level-refine draft slot: (a) pardec_generate with a draft must reproduce itself under dense
    pardec_score with the same draft; (b) a group's own span in the draft source must not affect that
    group's logits (draft = preceding groups only, no target leak)."""
    cfg = build_cfg(pardec_token_head=head, **over)
    print(f"[refine pardec_token_head={head} window={window} {over}]")
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, _ = load_data(cfg)
    codelm = model.codelm_for(0)
    up = model.upsampler_for(0)
    tok0 = R.rgb_byte_pq_fn(flat, codelm.pq_chunks, codelm.code_vocab)
    x = R.code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
    ctx_code_soft = codelm.encode(x, tok0, model.K(0), rng=None)["code_soft"]
    h_ctx = R.encoder_hidden(codelm, R.code_embed_proj(ctx_code_soft, codelm.own_input_embed, codelm.own_input_proj))
    kw = dict(context_group_size=G, output_group_size=G, output_expansion=model.K(0))
    Kspan = G * model.K(0)
    Pp = window * Kspan
    draft = R.pardec_generate(up, h_ctx, rng=jax.random.PRNGKey(0), greedy=True, **kw)

    gen2 = R.pardec_generate(up, h_ctx, rng=jax.random.PRNGKey(0), greedy=True, draft_seq=draft, draft_len=Pp, **kw)
    dl, _, _, _ = R.pardec_score(up, gen2, h_ctx, draft_seq=draft, draft_len=Pp, **kw)
    da = jnp.argmax(dl, axis=-1)
    n = min(da.shape[1], gen2.shape[1])
    ok_a = bool(jnp.array_equal(da[:, :n], gen2[:, :n]))
    print(f"REFINE dense-vs-KV-cache with draft: consistent={ok_a} mismatches={int((da[:, :n] != gen2[:, :n]).sum())} "
          f"(draft changed {int((gen2 != draft).any(-1).sum())}/{gen2.shape[0] * gen2.shape[1]} positions vs pass 1)")

    g = 2
    pert = draft.at[:, g * Kspan:(g + 1) * Kspan].set((draft[:, g * Kspan:(g + 1) * Kspan] + 1) % 256)
    l0, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_seq=draft, draft_len=Pp, **kw)
    l1, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_seq=pert, draft_len=Pp, **kw)
    diff = jnp.abs(l0 - l1).max(axis=tuple(range(2, l0.ndim)))  # (B, L)
    own = float(diff[:, g * Kspan:(g + 1) * Kspan].max())
    before = float(diff[:, :g * Kspan].max())
    after = float(diff[:, (g + 1) * Kspan:(g + 1 + window) * Kspan].max())
    ok_b = own == 0.0 and before == 0.0 and after > 0.0
    print(f"REFINE no-leak: perturbing group {g}'s own draft span -> own-group logit change={own:.2e} "
          f"earlier={before:.2e} next {window} group(s)={after:.2e}  {'OK' if ok_b else 'LEAK/BROKEN'}")
    return ok_a and ok_b


def run_pardec_refine_fixed_check(head="ar", window=1, **over):
    """'fixed' level-refine layout (draft_fill='mask'): (a) dense-vs-KV-cache for pass 1 (all-mask slot)
    and pass 2 (draft slot); (b) no leak of a group's own span; (c) position check: group 0's window is
    entirely before the image, so its pass-2 logits must equal its pass-1 (all-mask) logits; groups whose
    window is fully inside the image must match the 'zero' (variable-layout) refine pass exactly."""
    cfg = build_cfg(pardec_token_head=head, **over)
    print(f"[refine FIXED pardec_token_head={head} window={window} {over}]")
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, _ = load_data(cfg)
    codelm = model.codelm_for(0)
    up = model.upsampler_for(0)
    tok0 = R.rgb_byte_pq_fn(flat, codelm.pq_chunks, codelm.code_vocab)
    x = R.code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
    ctx_code_soft = codelm.encode(x, tok0, model.K(0), rng=None)["code_soft"]
    h_ctx = R.encoder_hidden(codelm, R.code_embed_proj(ctx_code_soft, codelm.own_input_embed, codelm.own_input_proj))
    kw = dict(context_group_size=G, output_group_size=G, output_expansion=model.K(0))
    Kspan = G * model.K(0)
    Pp = window * Kspan
    oks = []

    def kv_vs_dense(label, draft):
        gen = R.pardec_generate(up, h_ctx, rng=jax.random.PRNGKey(0), greedy=True, draft_seq=draft, draft_len=Pp,
                                draft_fill="mask", **kw)
        dl, _, _, _ = R.pardec_score(up, gen, h_ctx, draft_seq=draft, draft_len=Pp, draft_fill="mask", **kw)
        da = jnp.argmax(dl, axis=-1)
        n = min(da.shape[1], gen.shape[1])
        ok = bool(jnp.array_equal(da[:, :n], gen[:, :n]))
        print(f"FIXED {label} dense-vs-KV-cache: consistent={ok} mismatches={int((da[:, :n] != gen[:, :n]).sum())}")
        oks.append(ok)
        return gen

    pass1 = kv_vs_dense("pass 1 (all-mask slot)", None)
    kv_vs_dense("pass 2 (draft slot)", pass1)

    g = 2
    pert = pass1.at[:, g * Kspan:(g + 1) * Kspan].set((pass1[:, g * Kspan:(g + 1) * Kspan] + 1) % 256)
    l0, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_seq=pass1, draft_len=Pp, draft_fill="mask", **kw)
    l1, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_seq=pert, draft_len=Pp, draft_fill="mask", **kw)
    diff = jnp.abs(l0 - l1).max(axis=tuple(range(2, l0.ndim)))
    own, before = float(diff[:, g * Kspan:(g + 1) * Kspan].max()), float(diff[:, :g * Kspan].max())
    after = float(diff[:, (g + 1) * Kspan:(g + 1 + window) * Kspan].max())
    ok_b = own == 0.0 and before == 0.0 and after > 0.0
    print(f"FIXED no-leak: own-group change={own:.2e} earlier={before:.2e} next={after:.2e}  {'OK' if ok_b else 'LEAK/BROKEN'}")
    oks.append(ok_b)

    p1, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_len=Pp, draft_fill="mask", **kw)
    zero_l, _, _, _ = R.pardec_score(up, tok0, h_ctx, draft_seq=pass1, draft_len=Pp, draft_fill="zero", **kw)
    g0 = float(jnp.abs(l0[:, :Kspan] - p1[:, :Kspan]).max())
    inside = float(jnp.abs(l0[:, window * Kspan:] - zero_l[:, window * Kspan:]).max())
    ok_c = g0 == 0.0 and inside < 1e-5
    print(f"FIXED positions: group0 pass2-vs-pass1 diff={g0:.2e} (expect 0); groups>={window} fixed-vs-variable "
          f"refine diff={inside:.2e} (expect ~0)  {'OK' if ok_c else 'OFF-BY-ONE/BROKEN'}")
    oks.append(ok_c)
    return all(oks)


def run_mixer_step_check(kind):
    """RecurrentMixer dense (parallel scan / scan) vs one-step-at-a-time state updates, with a random
    validity mask (invalid positions must leave the state untouched in both paths)."""
    mixer = R.RecurrentMixer(jax.random.PRNGKey(0), 32, kind, state_dim=8, n_layers=2)
    rs = np.random.RandomState(0)
    x = jnp.array(rs.randn(3, 11, 32).astype(np.float32))
    valid = jnp.array(rs.rand(3, 11) > 0.3)
    dense = mixer(x, valid)
    st = mixer.init_state(3)
    outs = []
    for t in range(x.shape[1]):
        y, st = mixer.step(x[:, t], st, valid[:, t])
        outs.append(y)
    step = jnp.stack(outs, axis=1)
    err = float(jnp.abs(dense - step).max())
    st2 = mixer.init_state(1)
    for t in range(x.shape[1]):
        if bool(valid[0, t]):
            _, st2 = mixer.step(x[:1, t], st2)
    gate_err = float(jnp.abs(st[:1] - st2).max())
    ok = err < 1e-4 and gate_err < 1e-4
    print(f"MIXER {kind}: dense-vs-step max|diff|={err:.2e}, gated-vs-skipped state diff={gate_err:.2e}  {'OK' if ok else 'BROKEN'}")
    return ok


def _ctx_and_hidden(model, flat):
    codelm = model.codelm_for(0)
    tok0 = R.rgb_byte_pq_fn(flat, codelm.pq_chunks, codelm.code_vocab)
    x = R.code_embed_proj(tok0, codelm.own_input_embed, codelm.own_input_proj)
    ctx_code_soft = codelm.encode(x, tok0, model.K(0), rng=None)["code_soft"]
    h_ctx = R.encoder_hidden(codelm, R.code_embed_proj(ctx_code_soft, codelm.own_input_embed, codelm.own_input_proj))
    return tok0, ctx_code_soft, h_ctx


def run_cycle_slot_check(head="ar", n_cycles=3, **over):
    """Stack slots: (a) dense-vs-KV/state with filled + mask slots (alone and with a fixed refine draft);
    (b) causality: a revision code at position j is seen only by groups whose window covers j (same as
    the main context), never by earlier groups; (c) a slot's content actually changes the output."""
    cfg = build_cfg(pardec_token_head=head, level_cycles=(n_cycles, 1), level_cycle_mode="stack", **over)
    print(f"[cycle STACK slots head={head} cycles={n_cycles} {over}]")
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, _ = load_data(cfg)
    codelm, up = model.codelm_for(0), model.upsampler_for(0)
    tok0, ctx_code_soft, h_ctx = _ctx_and_hidden(model, flat)
    rev_idx = jnp.array(np.random.RandomState(1).randint(0, 256, ctx_code_soft.shape[:-1]))
    h_rev = R.encoder_hidden(codelm, R.code_embed_proj(rev_idx, codelm.own_input_embed, codelm.own_input_proj))
    slots = [h_rev] + [None] * (n_cycles - 2)
    kw = dict(context_group_size=G, output_group_size=G, output_expansion=model.K(0))
    oks = []

    def kv_vs_dense(label, **dkw):
        gen = R.pardec_generate(up, h_ctx, rng=jax.random.PRNGKey(0), greedy=True, cycle_ctx=slots, **kw, **dkw)
        dkw2 = dict(dkw)
        dl, _, _, _ = R.pardec_score(up, gen, h_ctx, cycle_ctx=slots, **kw, **dkw2)
        da = jnp.argmax(dl, axis=-1)
        ok = bool(jnp.array_equal(da, gen))
        print(f"STACK {label} dense-vs-KV: consistent={ok} mismatches={int((da != gen).sum())}")
        oks.append(ok)
        return gen

    gen1 = kv_vs_dense("slots only")
    Pp = G * model.K(0)
    kv_vs_dense("slots + fixed refine draft", draft_seq=gen1, draft_len=Pp, draft_fill="mask")

    n_codes = ctx_code_soft.shape[1]
    j = n_codes // 2
    pert = h_rev.at[:, j].add(1.0)
    l0, _, _, _ = R.pardec_score(up, tok0, h_ctx, cycle_ctx=[h_rev] + slots[1:], **kw)
    l1, _, _, _ = R.pardec_score(up, tok0, h_ctx, cycle_ctx=[pert] + slots[1:], **kw)
    diff = jnp.abs(l0 - l1).max(axis=tuple(range(2, l0.ndim)))  # (B, L)
    Kspan = G * model.K(0)
    g_j = j // G
    before = float(diff[:, :g_j * Kspan].max()) if g_j > 0 else 0.0
    own = float(diff[:, g_j * Kspan:(g_j + 1) * Kspan].max())
    ok_b = before == 0.0 and own > 0.0
    print(f"STACK causality: perturb revision code {j} (group {g_j}) -> earlier groups change={before:.2e} "
          f"(expect 0), own group={own:.2e} (expect >0)  {'OK' if ok_b else 'LEAK/OFF-BY-ONE'}")
    oks.append(ok_b)
    lm, _, _, _ = R.pardec_score(up, tok0, h_ctx, cycle_ctx=[None] * (n_cycles - 1), **kw)
    ok_c = float(jnp.abs(lm - l0).max()) > 0.0
    print(f"STACK slot used: filled-vs-mask logit change={float(jnp.abs(lm - l0).max()):.2e}  {'OK' if ok_c else 'UNUSED'}")
    oks.append(ok_c)
    return all(oks)


def run_cycle_leak_check(mode="memoryless", head="ar", **over):
    """Revision leak: with the same context and rng, swapping the level's GT target tokens must not change
    cycle-1 digit-0 logits at each group's first position (they see only context/slots/bos, never targets).
    rollout must give 0; pss/gt are expected to differ (they re-encode GT-derived tokens by design)."""
    print(f"[cycle LEAK mode={mode} head={head} {over}]")
    res = {}
    for inp in ("rollout", "pss", "gt"):
        cfg = build_cfg(pardec_token_head=head, level_cycles=(2, 1), level_cycle_mode=mode, level_cycle_input=inp,
                        **over)
        model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
        flat, _ = load_data(cfg)
        tok0, ctx_code_soft, _ = _ctx_and_hidden(model, flat)
        other = jnp.roll(tok0, 1, axis=0)  # another image's bytes as the "GT"
        z0 = []
        for tgt in (tok0, other):
            cyc = R.decode_logits_and_target_cycles(model, 0, tgt, ctx_code_soft, G, rng=jax.random.PRNGKey(3))
            lg = cyc[1][-1][0]
            Kspan = G * model.K(0)
            z0.append(lg[:, ::Kspan, 0])  # digit 0: later AR digits are teacher-forced on the token's own GT digits
        res[inp] = float(jnp.abs(z0[0] - z0[1]).max())
    ok = res["rollout"] == 0.0 and res["pss"] > 0.0 and res["gt"] > 0.0
    print(f"LEAK cycle-1 group-start logit change when GT swapped: rollout={res['rollout']:.2e} (expect 0) "
          f"pss={res['pss']:.2e} gt={res['gt']:.2e} (expect >0, by design)  {'OK' if ok else 'LEAK'}")
    return ok


def run_cycle_reencode_check(head="ar", **over):
    """Training re-encode (cycle_reencode, dense, rng=None -> argmax) must give the same code as generation's
    re-encode (encode_pardec_downsampler_generate, greedy KV/state)."""
    cfg = build_cfg(pardec_token_head=head, level_cycles=(2, 1), **over)
    print(f"[cycle REENCODE head={head} {over}]")
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, _ = load_data(cfg)
    tok0, _, _ = _ctx_and_hidden(model, flat)
    _, ci_train = R.cycle_reencode(model, 0, tok0, None)
    ci_gen, _ = R._cycle_reencode_generate(model, 0, tok0, G, True, 1.0, 0, False)
    mism = int((ci_train != ci_gen).sum())
    ok = mism == 0
    print(f"REENCODE train(dense) vs gen(KV) code mismatches={mism}/{ci_gen.size}  {'OK' if ok else 'DIVERGES'}")
    return ok


def run_cycle_e2e_check():
    """level_forward + generation over cycle settings: finite loss/grads, stack slot params trained,
    detach=False reaches the re-encoder, eval deterministic, generation shapes."""
    import equinox as eqx
    print("[cycle E2E]")
    ok = True
    base = dict(strides=(4, 4), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), upsampler_ncodes=(1, 1),
                downsampler_window=(1, 1), upsampler_window=(1, 1), share_across_levels=False,
                quantize_mode="reinmax_limit", ctx_stop_gradient="pseudo")
    for head in ("linear", "ar"):
        for mode in ("memoryless", "stack"):
            for inp in ("rollout", "pss", "gt"):
                for detach in (True, False):
                    if inp != "rollout" and not detach:
                        continue
                    cfg = build_cfg(pardec_token_head=head, codelm_token_head=head,
                                    token_head_type="linears" if head == "linear" else "ar",
                                    level_cycles=(3, 2), level_cycle_mode=mode, level_cycle_input=inp,
                                    level_cycle_detach=detach, level_refine_passes=(2, 1), level_refine_window=(1, 0),
                                    **base)
                    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
                    flat, po = load_data(cfg)
                    f = lambda m: R.level_forward(m, flat, 2, rng=jax.random.PRNGKey(3), level_gt_drop=1.0,
                                                  cascade_rng=jax.random.PRNGKey(4), label_reg_weight=1.0,
                                                  label_fn=R.rgb_label_fn_jax, pixel_order=po)
                    (loss, aux), g = eqx.filter_jit(eqx.filter_value_and_grad(f, has_aux=True))(model)
                    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
                    finite = bool(np.isfinite(float(loss))) and all(bool(jnp.isfinite(x).all()) for x in leaves)
                    slot_g = (float(jnp.abs(g.upsamplers[0].cycle_slot_embed).sum()) if mode == "stack" else 1.0)
                    ev = eqx.filter_jit(lambda m: R.level_forward(m, flat, 2, rng=None, label_reg_weight=1.0,
                                                                  label_fn=R.rgb_label_fn_jax, pixel_order=po)[0])
                    e1, e2 = ev(model), ev(model)
                    codes = R.encode_pardec_downsampler(model.codelm_for(0), model.downsampler_for(0), flat, flat, flat,
                                                        cfg, po, R.rgb_label_fn_jax, 4)["code_idx"]
                    gen = R.decode_generate_cycles(model, 0, codes, 1, greedy=False, seed=1)
                    this = finite and slot_g > 0 and float(e1) == float(e2) and gen.shape == flat.shape
                    ok &= this
                    print(f"  head={head} mode={mode} input={inp} detach={detach}: loss={float(loss):.4f} "
                          f"finite={finite} slot_grad={slot_g:.2e} eval_det={float(e1) == float(e2)} "
                          f"gen={tuple(gen.shape)}  {'OK' if this else 'BROKEN'}")
    lin = dict(pardec_token_head="linear", codelm_token_head="linear", token_head_type="linears")
    cfg_t = build_cfg(level_cycles=(2, 1), level_cycle_input="rollout", level_cycle_detach=True, **lin, **base)
    cfg_f = build_cfg(level_cycles=(2, 1), level_cycle_input="rollout", level_cycle_detach=False, **lin, **base)
    grads = []
    for cfg in (cfg_t, cfg_f):
        model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
        flat, po = load_data(cfg)
        f = lambda m: R.level_forward(m, flat, 1, rng=jax.random.PRNGKey(3), label_reg_weight=1.0,
                                      label_fn=R.rgb_label_fn_jax, pixel_order=po)[0]
        grads.append(eqx.filter_jit(eqx.filter_grad(f))(model).downsamplers[0].output_head_linear)
    reach = float(jnp.abs(grads[0] - grads[1]).max())
    print(f"  detach=False reaches re-encoder: downsampler grad change={reach:.2e} (expect >0)  {'OK' if reach > 0 else 'BROKEN'}")
    return ok and reach > 0


if __name__ == "__main__":
    results = []
    for kind in ("gru", "linear_gru", "ssm"):
        results.append((f"mixer dense-vs-step ({kind})", run_mixer_step_check(kind)))
    results.append(("pardec refine FIXED (ar)", run_pardec_refine_fixed_check("ar")))
    results.append(("pardec refine FIXED (linear, window 2)", run_pardec_refine_fixed_check("linear", window=2)))
    results.append(("pardec refine draft (ar)", run_pardec_refine_check("ar")))
    results.append(("pardec refine draft (linear, window 2)", run_pardec_refine_check("linear", window=2)))
    results.append(("encoder_free_run KV-cache (linear)", run_encoder_free_run_kv_check("linear")))
    results.append(("encoder_free_run KV-cache (ar)", run_encoder_free_run_kv_check("ar")))
    results.append(("pardec dense-vs-KV-cache (ar)", run_pardec_dense_vs_kv_check("ar")))
    results.append(("pardec dense-vs-KV-cache (linear)", run_pardec_dense_vs_kv_check("linear")))
    for bb in ("gru", "linear_gru", "ssm"):
        bkw = dict(codelm_backbone=bb, downsampler_backbone=bb, upsampler_backbone=bb)
        results.append((f"encoder_free_run state ({bb})", run_encoder_free_run_kv_check("linear", **bkw)))
        results.append((f"pardec dense-vs-state ({bb}, ar)", run_pardec_dense_vs_kv_check("ar", **bkw)))
        results.append((f"pardec refine FIXED ({bb}, linear)", run_pardec_refine_fixed_check("linear", **bkw)))
        results.append((f"cycle stack slots ({bb})", run_cycle_slot_check("linear", **bkw)))
    results.append(("cycle stack slots (ar)", run_cycle_slot_check("ar")))
    results.append(("cycle stack slots (linear, 4 cycles)", run_cycle_slot_check("linear", n_cycles=4)))
    for mode in ("memoryless", "stack"):
        results.append((f"cycle leak ({mode}, ar)", run_cycle_leak_check(mode, "ar")))
        results.append((f"cycle leak ({mode}, linear)", run_cycle_leak_check(mode, "linear")))
    results.append(("cycle re-encode train-vs-gen (ar)", run_cycle_reencode_check("ar")))
    results.append(("cycle re-encode train-vs-gen (linear)", run_cycle_reencode_check("linear")))
    results.append(("cycle re-encode train-vs-gen (ssm)", run_cycle_reencode_check(
        "linear", codelm_backbone="ssm", downsampler_backbone="ssm", upsampler_backbone="ssm")))
    results.append(("cycle end-to-end", run_cycle_e2e_check()))
    print()
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if all(ok for _, ok in results):
        print("PASS all")
    else:
        print("FAIL")
        sys.exit(1)
