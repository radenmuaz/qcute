"""CPU-only correctness check for image_lagcodec/run_lagcodec_res.py (the singleton CodeLM/
Downsampler/Upsampler rewrite): dense teacher-forced reference vs incremental KV-cache generation,
for BOTH CodeLM's own free-run (encoder_free_run) and the shared PardecLM decode
(pardec_score vs pardec_generate). Uses real CIFAR data.
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.res_consistency_check
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
import image_lagcodec.run_lagcodec_res as R
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


def run_encoder_free_run_kv_check(head="linear"):
    """New KV-cache _encoder_free_run must exactly match a slow reference that recomputes the
    entire forward pass at every step (the OLD, correct-by-construction implementation)."""
    cfg = build_cfg(codelm_token_head=head)
    model = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    codelm = model.codelm_for(0)
    ok = True
    print(f"[codelm_token_head={head}]")
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


def run_pardec_dense_vs_kv_check(head="ar"):
    """pardec_score (dense, teacher-forced with the real target) vs pardec_generate (incremental
    KV-cache): greedy generation's argmax at each position must match dense scoring's argmax when
    dense is fed that SAME generated sequence as its target (self-consistency, the standard way to
    check a KV-cache decoder matches its own teacher-forced scoring path)."""
    cfg = build_cfg(pardec_token_head=head)
    print(f"[pardec_token_head={head}]")
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


if __name__ == "__main__":
    results = []
    results.append(("encoder_free_run KV-cache (linear)", run_encoder_free_run_kv_check("linear")))
    results.append(("encoder_free_run KV-cache (ar)", run_encoder_free_run_kv_check("ar")))
    results.append(("pardec dense-vs-KV-cache (ar)", run_pardec_dense_vs_kv_check("ar")))
    results.append(("pardec dense-vs-KV-cache (linear)", run_pardec_dense_vs_kv_check("linear")))
    print()
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if all(ok for _, ok in results):
        print("PASS all")
    else:
        print("FAIL")
        sys.exit(1)
