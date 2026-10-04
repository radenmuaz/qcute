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
    # gen_level_cycles=1 on a stack-trained model must still decode with the (all-mask) slots, like cycle 0 in training
    import dataclasses
    model1 = R.LagCodecModel(jax.random.PRNGKey(0), dataclasses.replace(cfg, gen_level_cycles=(1, 1)))
    code_idx = jnp.argmax(ctx_code_soft, -1)
    g1 = R.decode_generate_cycles(model1, 0, code_idx, G, greedy=True)
    ref = R.decode_generate_multipass(model1, 0, code_idx, G, greedy=True, cycle_ctx=[None] * (n_cycles - 1))
    ok_d = bool(jnp.array_equal(g1, ref))
    print(f"STACK gen_level_cycles=1 keeps the masked slots: {'OK' if ok_d else 'LAYOUT MISMATCH'}")
    oks.append(ok_d)
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


def run_codelm_upper_check():
    """context_source="codelm_upper": structure (extra CodeLM-only level unless shared, bos rows), upsampler i
    reads CodeLM i+1 and not CodeLM i, dense-vs-KV decode consistency, finite loss/grads that reach the
    extra CodeLM, and the freeze filter's per-phase trainable CodeLMs."""
    import equinox as eqx
    print("[codelm_upper]")
    oks = []
    base = dict(strides=(4, 4), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), upsampler_ncodes=(1, 1),
                downsampler_window=(1, 1), upsampler_window=(1, 1), context_source="codelm_upper",
                use_codelm_bos=True, codelm_bos_rates=(2, 2), codelm_d_model=(64, 48))
    cfg = build_cfg(share_across_levels=False, **base)
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    n_ok = len(model.codelms) == 3 and model.codelms[2].bos_embed.shape == (2, 48) \
        and model.upsamplers[0].context_proj.shape[0] == 48 and model.upsamplers[1].context_proj.shape[0] == 48
    cfg_s = build_cfg(share_across_levels=True, **dict(base, codelm_d_model=(64, 64)))
    model_s = R.LagCodecModel(jax.random.PRNGKey(0), cfg_s)
    s_ok = len(model_s.codelms) == 1 and model_s.codelms[0].bos_embed.shape[0] == 3
    print(f"  structure: unshared codelms={len(model.codelms)} (expect 3, extra has bos {model.codelms[2].bos_embed.shape}),"
          f" shared bos rows={model_s.codelms[0].bos_embed.shape[0]} (expect 3)  {'OK' if n_ok and s_ok else 'BROKEN'}")
    oks.append(n_ok and s_ok)

    flat, po = load_data(cfg)
    codelm0 = model.codelm_for(0)
    tok0 = R.rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)
    code0 = R.encode_pardec_downsampler(codelm0, model.downsampler_for(0), tok0, tok0, flat, cfg, po, R.rgb_label_fn_jax,
                                        4)["code_soft"]
    lg = lambda m: R.decode_logits_and_target_multipass(m, 0, tok0, code0, 1)[0]
    base_lg = lg(model)
    bump = lambda m, j: eqx.tree_at(lambda t: t.codelms[j].own_input_proj, m, m.codelms[j].own_input_proj + 0.5)
    d_up, d_own = float(jnp.abs(lg(bump(model, 1)) - base_lg).max()), float(jnp.abs(lg(bump(model, 0)) - base_lg).max())
    ok = d_up > 0 and d_own == 0.0
    print(f"  upsampler 0 reads CodeLM 1: change from CodeLM 1={d_up:.2e} (>0), from CodeLM 0={d_own:.2e} (0)  "
          f"{'OK' if ok else 'BROKEN'}")
    oks.append(ok)

    code_idx = jnp.argmax(code0, -1)
    gen = R.decode_generate_multipass(model, 0, code_idx, 1, greedy=True)
    dl = R.decode_logits_and_target_multipass(model, 0, gen, code_idx, 1)[0]
    mism = int((jnp.argmax(dl, -1) != gen).sum())
    print(f"  dense-vs-KV decode: mismatches={mism}  {'OK' if mism == 0 else 'DIVERGES'}")
    oks.append(mism == 0)

    f = lambda m: R.level_forward(m, flat, 2, rng=jax.random.PRNGKey(3), level_gt_drop=1.0,
                                  cascade_rng=jax.random.PRNGKey(4), label_reg_weight=1.0,
                                  label_fn=R.rgb_label_fn_jax, pixel_order=po)[0]
    loss, g = eqx.filter_jit(eqx.filter_value_and_grad(f))(model)
    g_extra = float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(eqx.filter(g.codelms[2], eqx.is_array))))
    ok = bool(np.isfinite(float(loss))) and g_extra > 0
    print(f"  e2e: loss={float(loss):.4f} extra CodeLM grad={g_extra:.2e}  {'OK' if ok else 'BROKEN'}")
    oks.append(ok)

    cfg_f = build_cfg(share_across_levels=False, curriculum_mode="freeze", **base)
    model_f = R.LagCodecModel(jax.random.PRNGKey(0), cfg_f)
    trains = []
    for ph in (1, 2):
        spec = R.phase_trainable_filter(model_f, ph)
        on = lambda sub: any(jax.tree_util.tree_leaves(sub))
        trains.append((tuple(j for j in range(3) if on(spec.codelms[j])),
                       tuple(j for j in range(2) if on(spec.downsamplers[j]) and on(spec.upsamplers[j]))))
    ok = trains == [((0, 1), (0,)), ((2,), (1,))]
    print(f"  freeze trainable (codelms, levels) per phase={trains} (expect [((0, 1), (0,)), ((2,), (1,))])  "
          f"{'OK' if ok else 'BROKEN'}")
    oks.append(ok)
    return all(oks)


def run_remat_chunks_check():
    """downsampler/upsampler_remat_chunks: chunked rows give the same loss and grads as unchunked (chunk counts that
    don't divide the row count exercise padding; stack slots + refine drafts + recurrent backbone covered), and the
    compiled step's temp memory drops."""
    import dataclasses
    import equinox as eqx
    print("[remat chunks]")
    oks = []
    base = dict(strides=(4, 4), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), upsampler_ncodes=(1, 1),
                downsampler_window=(1, 1), upsampler_window=(1, 1), share_across_levels=False,
                quantize_mode="reinmax_limit", ctx_stop_gradient="pseudo")
    cases = [("transformer, remat off", {}),
             ("transformer, remat on, stack cycles + refine", dict(remat=True, level_cycles=(2, 2), level_cycle_mode="stack",
                                                                  level_cycle_input="gt", level_refine_passes=(2, 1),
                                                                  level_refine_window=(1, 0))),
             ("ssm backbone, remat on", dict(remat=True, codelm_backbone="ssm", downsampler_backbone="ssm",
                                             upsampler_backbone="ssm"))]
    for name, over in cases:
        cfg1 = build_cfg(**base, **over)
        cfgc = dataclasses.replace(cfg1, downsampler_remat_chunks=(3, 3), upsampler_remat_chunks=(5, 2))
        flat, po = load_data(cfg1)
        res = []
        for cfg in (cfg1, cfgc):
            model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
            f = lambda m: R.level_forward(m, flat, 2, rng=None, level_gt_drop=1.0, label_reg_weight=1.0,
                                          label_fn=R.rgb_label_fn_jax, pixel_order=po)[0]
            step = eqx.filter_jit(eqx.filter_value_and_grad(f))
            loss, g = step(model)
            try:
                mem = step.lower(model).compile().memory_analysis().temp_size_in_bytes
            except Exception:
                mem = -1
            res.append((float(loss), jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array)), mem))
        (l1, g1, m1), (lc, gc, mc) = res
        gerr = max(float(jnp.abs(a - b).max() / jnp.maximum(jnp.abs(a).max(), 1e-6)) for a, b in zip(g1, gc))
        ok = abs(lc - l1) <= 1e-5 * abs(l1) and gerr < 1e-4
        print(f"  {name}: loss {l1:.6f} vs chunked {lc:.6f}, max grad rel diff={gerr:.1e}, compiled temp "
              f"{m1 / 2**20:.1f} -> {mc / 2**20:.1f} MiB  {'OK' if ok else 'DIFF'}")
        oks.append(ok)
    return all(oks)


def run_reinmax_check():
    """quantize_reinmax_limit (O(K)) vs quantize_reinmax_limit_dense (slow (K,K) reference): forward code_soft and idx
    bit-identical, gradient equal up to rounding (with/without rng, quantize_drop, extreme logits); saved residuals
    shrink from O(K^2) to O(K) per position."""
    from jax._src.ad_checkpoint import saved_residuals
    print("[reinmax fast vs dense]")
    oks = []
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    for name, scale, rng, drop in (("normal", 1.0, k[1], 0.0), ("no rng", 1.0, None, 0.0), ("extreme logits", 30.0, k[2], 0.0),
                                   ("quantize_drop 0.3", 1.0, k[3], 0.3)):
        lg = jax.random.normal(k[0], (2, 7, 3, 256)) * scale
        w = jax.random.normal(jax.random.PRNGKey(9), lg.shape)
        outs = []
        for fn in (R.quantize_reinmax_limit, R.quantize_reinmax_limit_dense):
            (cs, idx) = fn(lg, rng, 1.0, drop)
            g = jax.grad(lambda l: jnp.sum(fn(l, rng, 1.0, drop)[0] * w))(lg)
            outs.append((cs, idx, g))
        (c1, i1, g1), (c2, i2, g2) = outs
        fwd = bool(jnp.array_equal(c1, c2)) and bool(jnp.array_equal(i1, i2))
        gerr = float(jnp.abs(g1 - g2).max() / jnp.maximum(jnp.abs(g2).max(), 1e-12))
        ok = fwd and gerr < 1e-5
        print(f"  {name}: forward bit-identical={fwd}, grad max rel diff={gerr:.1e}  {'OK' if ok else 'DIFF'}")
        oks.append(ok)
    lg = jax.random.normal(k[0], (4, 64, 3, 256))
    mb = lambda fn: sum(int(np.prod(a.shape)) * a.dtype.itemsize for a, _ in
                        saved_residuals(lambda l: jnp.sum(fn(l, k[1], 1.0, 0.0)[0] * l), lg)) / 2 ** 20
    fast, dense = mb(R.quantize_reinmax_limit), mb(R.quantize_reinmax_limit_dense)
    print(f"  saved for backward (4x64x3 positions): fast {fast:.2f} MiB vs dense {dense:.2f} MiB")
    oks.append(fast < dense / 10)
    return all(oks)


def run_pss_check():
    """upsampler/downsampler_pss_passes: n passes make the first n tokens of every row equal a real greedy rollout
    (-1 = whole row, linear head and ar head + upsampler_rollout); prob=0 equals teacher forcing; passes=1 is the
    old path; training grads finite with both sides on."""
    import dataclasses
    import equinox as eqx
    print("[parallel scheduled sampling]")
    oks = []
    base = dict(strides=(4, 4), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), upsampler_ncodes=(2, 1),
                downsampler_window=(1, 1), upsampler_window=(1, 1), share_across_levels=False)
    for name, over in (("linear", dict(pardec_token_head="linear", codelm_token_head="linear")),
                       ("ar + upsampler_rollout", dict(pardec_token_head="ar", codelm_token_head="ar", token_head_type="ar",
                                                       upsampler_rollout=True, upsampler_ncodes=(1, 1)))):
        cfg1 = build_cfg(**{**base, **over})
        model = R.LagCodecModel(jax.random.PRNGKey(0), cfg1)
        flat, po = load_data(cfg1)
        tgt = R.rgb_byte_pq_fn(flat, cfg1.pq_chunks[0], cfg1.code_vocab[0])
        nc = cfg1.upsampler_ncodes[0]
        T = nc * 4
        ctx = jax.random.randint(jax.random.PRNGKey(5), (B, tgt.shape[1] // 4, 3), 0, 256)
        gen = np.asarray(R._decode_generate_pardec_call(model, 0, ctx, nc, True, 1.0, 0)).reshape(B, -1, T, 3)
        with_cfg = lambda **kw: R.LagCodecModel(jax.random.PRNGKey(0), dataclasses.replace(cfg1, **kw))  # same weights
        score = lambda m, rng=None: R.decode_logits_and_target_multipass(m, 0, tgt, ctx, nc, rng=rng)[0]
        for n in (1, 2, -1):
            pred = np.asarray(jnp.argmax(score(with_cfg(upsampler_pss_passes=(n, 1))), -1)).reshape(B, -1, T, 3)
            k = T if n == -1 else n
            ok = bool((pred[:, :, :k] == gen[:, :, :k]).all())
            print(f"  {name}: passes={n}: first {k}/{T} row tokens equal greedy rollout={ok}, whole-row match="
                  f"{float((pred == gen).mean()):.3f}  {'OK' if ok else 'DIFF'}")
            oks.append(ok)
        rng = jax.random.PRNGKey(7)
        tf = score(model, rng)
        p0 = score(with_cfg(upsampler_pss_passes=(3, 1), upsampler_pss_prob=0.0), rng)
        samp = score(with_cfg(upsampler_pss_passes=(3, 1), upsampler_pss_prob=0.5, pss_input_mode="sample"), rng)
        ok = bool(jnp.array_equal(tf, p0)) and bool(jnp.isfinite(samp).all()) and not bool(jnp.array_equal(tf, samp))
        print(f"  {name}: prob=0 equals teacher forcing={bool(jnp.array_equal(tf, p0))}, sample/prob=0.5 differs  "
              f"{'OK' if ok else 'DIFF'}")
        oks.append(ok)
    cfg = build_cfg(**{**base, "downsampler_ncodes": (2, 2), "upsampler_pss_passes": (3, -1), "downsampler_pss_passes": (2, -1),
                       "upsampler_pss_prob": 0.7, "downsampler_pss_prob": 0.7, "quantize_mode": "reinmax_limit"})
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, po = load_data(cfg)
    f = lambda m: R.level_forward(m, flat, 2, rng=jax.random.PRNGKey(3), level_gt_drop=1.0, cascade_rng=jax.random.PRNGKey(4),
                                  label_reg_weight=1.0, label_fn=R.rgb_label_fn_jax, pixel_order=po)[0]
    loss, g = eqx.filter_jit(eqx.filter_value_and_grad(f))(model)
    ok = bool(np.isfinite(float(loss))) and all(bool(jnp.isfinite(x).all()) for x in
                                                jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array)))
    print(f"  train step, both sides on (downsampler_ncodes=2): loss={float(loss):.4f} finite grads={ok}  {'OK' if ok else 'DIFF'}")
    oks.append(ok)
    return all(oks)


def run_rollout_ncodes_check():
    """downsampler_rollout / upsampler_rollout with ncodes>1: eval rollout equals greedy generation (downsampler: all
    row codes, capped passes: the first ones; upsampler: with pss -1), and a train step has finite grads."""
    import dataclasses
    import equinox as eqx
    print("[rollout with ncodes>1]")
    oks = []
    base = dict(strides=(4, 4), code_vocab=(256, 256), pq_chunks=(3, 3), pq_dim=(16, 16), downsampler_window=(1, 1),
                upsampler_window=(1, 1), share_across_levels=False, pardec_token_head="ar", codelm_token_head="ar",
                token_head_type="ar", downsampler_rollout=True, upsampler_rollout=True)
    for nc in (2, 4):
        cfg = build_cfg(**{**base, "downsampler_ncodes": (nc, nc), "upsampler_ncodes": (nc, 1)})
        model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
        flat, po = load_data(cfg)
        tok = R.rgb_byte_pq_fn(flat, cfg.pq_chunks[0], cfg.code_vocab[0])
        cl, ds = model.codelm_for(0), model.downsampler_for(0)
        gen = np.asarray(R.encode_pardec_downsampler_generate(cl, ds, tok, 4, cfg, rng=jax.random.PRNGKey(1), greedy=True,
                                                              downsampler_ncodes=nc)["code_idx"])
        for passes in (1, 2):
            roll = np.asarray(R.encode_pardec_downsampler(cl, ds, tok, tok, flat, cfg, po, R.rgb_label_fn_jax, 4, rng=None,
                                                          downsampler_ncodes=nc, pss_passes=passes)["code_idx"])
            k = nc if passes == 1 else min(passes, nc)
            a, b = roll.reshape(B, -1, nc, 3)[:, :, :k], gen.reshape(B, -1, nc, 3)[:, :, :k]
            ok = bool((a == b).all())
            print(f"  downsampler ncodes={nc} passes={passes}: first {k}/{nc} row codes equal greedy generation={ok}, "
                  f"whole-row match={float((roll == gen).mean()):.3f}  {'OK' if ok else 'DIFF'}")
            oks.append(ok)
        ctx = jax.random.randint(jax.random.PRNGKey(5), (B, tok.shape[1] // 4, 3), 0, 256)
        g_up = np.asarray(R._decode_generate_pardec_call(model, 0, ctx, nc, True, 1.0, 0))
        m_pss = R.LagCodecModel(jax.random.PRNGKey(0), dataclasses.replace(cfg, upsampler_pss_passes=(-1, 1)))
        pred = np.asarray(jnp.argmax(R.decode_logits_and_target_multipass(m_pss, 0, tok, ctx, nc, rng=None)[0], -1))
        ok = bool((pred == g_up).all())
        print(f"  upsampler ncodes={nc} rollout + pss -1 equals greedy generation={ok}  {'OK' if ok else 'DIFF'}")
        oks.append(ok)
    cfg = build_cfg(**{**base, "downsampler_ncodes": (2, 2), "upsampler_ncodes": (2, 1), "downsampler_rollout_prob": 0.5,
                       "upsampler_rollout_prob": 0.5, "upsampler_pss_passes": (2, 1), "quantize_mode": "zgr"})
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    flat, po = load_data(cfg)
    f = lambda m: R.level_forward(m, flat, 2, rng=jax.random.PRNGKey(3), level_gt_drop=1.0, cascade_rng=jax.random.PRNGKey(4),
                                  label_reg_weight=1.0, label_fn=R.rgb_label_fn_jax, pixel_order=po)[0]
    loss, g = eqx.filter_jit(eqx.filter_value_and_grad(f))(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
    ds_g = float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(eqx.filter(g.downsamplers, eqx.is_array))))
    ok = bool(np.isfinite(float(loss))) and all(bool(jnp.isfinite(x).all()) for x in leaves) and ds_g > 0
    print(f"  train step, both rollouts prob 0.5, ncodes=2: loss={float(loss):.4f} finite grads, downsampler |g|={ds_g:.3g}  "
          f"{'OK' if ok else 'DIFF'}")
    oks.append(ok)
    return all(oks)


if __name__ == "__main__":
    checks = []
    for kind in ("gru", "linear_gru", "ssm"):
        checks.append((f"mixer dense-vs-step ({kind})", lambda k=kind: run_mixer_step_check(k)))
    checks += [
        ("pardec refine FIXED (ar)", lambda: run_pardec_refine_fixed_check("ar")),
        ("pardec refine FIXED (linear, window 2)", lambda: run_pardec_refine_fixed_check("linear", window=2)),
        ("pardec refine draft (ar)", lambda: run_pardec_refine_check("ar")),
        ("pardec refine draft (linear, window 2)", lambda: run_pardec_refine_check("linear", window=2)),
        ("encoder_free_run KV-cache (linear)", lambda: run_encoder_free_run_kv_check("linear")),
        ("encoder_free_run KV-cache (ar)", lambda: run_encoder_free_run_kv_check("ar")),
        ("pardec dense-vs-KV-cache (ar)", lambda: run_pardec_dense_vs_kv_check("ar")),
        ("pardec dense-vs-KV-cache (linear)", lambda: run_pardec_dense_vs_kv_check("linear")),
    ]
    for bb in ("gru", "linear_gru", "ssm"):
        bkw = dict(codelm_backbone=bb, downsampler_backbone=bb, upsampler_backbone=bb)
        checks += [
            (f"encoder_free_run state ({bb})", lambda b=bkw: run_encoder_free_run_kv_check("linear", **b)),
            (f"pardec dense-vs-state ({bb}, ar)", lambda b=bkw: run_pardec_dense_vs_kv_check("ar", **b)),
            (f"pardec refine FIXED ({bb}, linear)", lambda b=bkw: run_pardec_refine_fixed_check("linear", **b)),
            (f"cycle stack slots ({bb})", lambda b=bkw: run_cycle_slot_check("linear", **b)),
        ]
    checks += [
        ("cycle stack slots (ar)", lambda: run_cycle_slot_check("ar")),
        ("cycle stack slots (linear, 4 cycles)", lambda: run_cycle_slot_check("linear", n_cycles=4)),
    ]
    for mode in ("memoryless", "stack"):
        checks += [(f"cycle leak ({mode}, ar)", lambda m=mode: run_cycle_leak_check(m, "ar")),
                   (f"cycle leak ({mode}, linear)", lambda m=mode: run_cycle_leak_check(m, "linear"))]
    checks += [
        ("cycle re-encode train-vs-gen (ar)", lambda: run_cycle_reencode_check("ar")),
        ("cycle re-encode train-vs-gen (linear)", lambda: run_cycle_reencode_check("linear")),
        ("cycle re-encode train-vs-gen (ssm)", lambda: run_cycle_reencode_check(
            "linear", codelm_backbone="ssm", downsampler_backbone="ssm", upsampler_backbone="ssm")),
        ("cycle end-to-end", run_cycle_e2e_check),
        ("codelm_upper", run_codelm_upper_check),
        ("remat chunks", run_remat_chunks_check),
        ("reinmax fast vs dense", run_reinmax_check),
        ("parallel scheduled sampling", run_pss_check),
        ("rollout with ncodes>1", run_rollout_ncodes_check),
    ]
    only = sys.argv[1:]  # optional: run only the named checks
    results = [(name, fn()) for name, fn in checks if not only or name in only]
    print()
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if all(ok for _, ok in results):
        print("PASS all")
    else:
        print("FAIL")
        sys.exit(1)
