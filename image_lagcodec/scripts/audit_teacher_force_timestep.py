"""Audit teacher-forced (dense pardec_score) argmax accuracy PER POSITION within the sequence, using
a real trained checkpoint. Reveals whether divergence is a training-quality issue (argmax already
wrong even with REAL teacher-forced previous context) vs an exposure-bias issue (only shows up in
real autoregressive generation, not here) -- and which specific positions/timesteps it starts at.

Usage (run with JAX_PLATFORMS=cpu, does not touch the TPU):
  JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.audit_teacher_force_timestep \
      --run_dir image_lagcodec/logs/cifar_res_full2 --checkpoint phase_1_step5000 --level 0
"""
import argparse
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import image_lagcodec.eqx_common as eqx_common


def _dense_splash_equivalent(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
    # Faithful dense CPU reference for eqx_common.splash_attention's real semantics (real Pallas
    # kernel only runs on TPU) -- needed here (unlike other CPU smoke tests in this codebase) because
    # this script computes REAL accuracy numbers against a REAL trained checkpoint, not just a
    # finite/shape check, so an inaccurate window/sink approximation would give misleading numbers.
    # window: None=unbounded causal. int=causal sliding window, query i sees keys [i-window, i].
    # lookahead: int>0, query i additionally sees keys up to i+lookahead (right-side window).
    # sink: (Hq,) per-head bias logit appended to the softmax denominator only (StreamingLLM-style
    # attention sink) -- contributes no value (no corresponding V row), matches splash_attention's
    # native `sinks` kernel arg semantics (see eqx_common.py Attention's own docstring/comment).
    B, Hq, T, hd = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    if rep > 1:
        k = jnp.repeat(k, rep, axis=1)
        v = jnp.repeat(v, rep, axis=1)
    scores = jnp.einsum("bhtd,bhsd->bhts", q * sm_scale, k)  # q already pre-scaled by caller? no -- sm_scale applied here
    idx = jnp.arange(T)
    if window is not None or lookahead > 0:
        left = window if window is not None else T
        mask = (idx[:, None] - idx[None, :] <= left) & (idx[None, :] - idx[:, None] <= lookahead)
    elif causal:
        mask = idx[:, None] >= idx[None, :]
    else:
        mask = jnp.ones((T, T), dtype=bool)
    scores = jnp.where(mask[None, None], scores, -1e9)
    if sink is not None:
        sink_col = jnp.broadcast_to(sink[None, :, None, None], (B, Hq, T, 1)).astype(scores.dtype)
        scores_ext = jnp.concatenate([scores, sink_col], axis=-1)
        w_ext = jax.nn.softmax(scores_ext, axis=-1)
        w = w_ext[..., :-1]  # sink's own weight absorbs probability mass but contributes no V
    else:
        w = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", w, v)


eqx_common.splash_attention = _dense_splash_equivalent
import image_lagcodec.run_lagcodec_res as R
R.splash_attention = _dense_splash_equivalent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="checkpoint dir name under run_dir/checkpoints/")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / args.checkpoint

    cfg_vars = {}
    exec((run_dir / "resolved_config.py").read_text(), cfg_vars)
    cfg_vars = {k: v for k, v in cfg_vars.items() if not k.startswith("_") and k in R.Config.__dataclass_fields__}
    cfg = R.Config(**cfg_vars)
    print(f"config OK: strides={cfg.strides} "
          f"upsampler_decode_future={cfg.upsampler_decode_future} use_codelm_bos={cfg.use_codelm_bos} "
          f"codelm_bos_prob={cfg.codelm_bos_prob}")

    model_skeleton = R.LagCodecModel(jax.random.PRNGKey(0), cfg)
    model = eqx.tree_deserialise_leaves(ckpt_dir / "model.eqx", model_skeleton)
    print(f"checkpoint loaded from {ckpt_dir}")

    codelm0 = model.codelm_for(0)
    pixel_order = R.pixel_order_for(cfg)
    data_root = Path(cfg_vars_data_root(run_dir))
    (train_np, _), (val_np, _) = R.load_dataset(cfg_vars.get("dataset", "cifar"), data_root, cfg.img_size)
    imgs = val_np[:args.batch]
    flat = jnp.array(R.images_to_positions(imgs, cfg, pixel_order))
    tok0 = R.rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)

    level = args.level
    # Build REAL encode-side codes up through `level` teacher-forced (matches level_forward's own
    # encode loop exactly), so the decode side's context (ctx_code_soft) is genuine, not synthetic.
    x = R.code_embed_proj(tok0, codelm0.own_input_embed, codelm0.own_input_proj)
    target = tok0
    codes_soft = []
    for i in range(level + 1):
        codelm = model.codelm_for(i)
        out = R.encode_pardec_downsampler(codelm, model.downsampler_for(i), x, target, flat, cfg,
                                           pixel_order, R.rgb_label_fn_jax, model.K(i), rate_id=model.bos_rate_id(i),
                                           codelm_rate_id=model.codelm_bos_rate_id(i),
                                           downsampler_ncodes=cfg.downsampler_ncodes[i])
        codes_soft.append(out["code_soft"])
        if i < level:
            codelm_next = model.codelm_for(i + 1)
            x = R.code_embed_proj(out["code_soft"], codelm_next.own_input_embed, codelm_next.own_input_proj)
            target = out["code_idx"]

    dec_target = tok0 if level == 0 else out["code_idx"]  # placeholder; real dec_target below
    # decode_logits_and_target_multipass needs THIS level's own real target_seq (the ground-truth
    # tokens at level `level`'s own input granularity) and ctx = level `level`'s own code (what it
    # produced), matching level_forward's start_i loop exactly: dec_target = tok0 if i==0 else codes[i-1]
    dec_target = tok0 if level == 0 else codes_prev_idx(model, cfg, pixel_order, flat, level)
    logits, target_out, _, aux_loss, aux_acc = R.decode_logits_and_target_multipass(
        model, level, dec_target, codes_soft[level], cfg.upsampler_ncodes[level])

    argmax = jnp.argmax(logits, axis=-1)  # (batch, n_positions, chunks)
    match = (argmax == target_out).astype(jnp.float32)  # (batch, n_positions, chunks)
    per_position_acc = jnp.mean(match, axis=(0, 2))  # (n_positions,)
    per_position_acc = np.asarray(per_position_acc)

    print(f"\nlevel={level} n_positions={per_position_acc.shape[0]} aux_loss={float(aux_loss):.4f} aux_acc={float(aux_acc):.4f}")
    print("overall teacher-forced argmax accuracy:", float(np.mean(per_position_acc)))
    print("\nper-timestep accuracy (position index: accuracy), flagging drops >0.15 vs previous:")
    prev = None
    first_bad = None
    for t, acc in enumerate(per_position_acc):
        flag = ""
        if prev is not None and prev - acc > 0.15:
            flag = "  <-- DROP"
            if first_bad is None:
                first_bad = t
        if t < 20 or t % 32 == 0 or flag:
            print(f"  t={t:4d}: acc={acc:.3f}{flag}")
        prev = acc
    if first_bad is not None:
        print(f"\nFIRST significant divergence at timestep t={first_bad} (acc dropped >0.15 vs t-1)")
    else:
        print("\nNo single sharp divergence point found (drops, if any, are gradual)")


def cfg_vars_data_root(run_dir):
    cfg_vars = {}
    exec((run_dir / "resolved_config.py").read_text(), cfg_vars)
    return cfg_vars.get("data_root", str(REPO_ROOT / "datasets"))


def codes_prev_idx(model, cfg, pixel_order, flat, level):
    # Re-derive codes[level-1] (the previous level's own code_idx) via the same encode loop,
    # needed as decode_logits_and_target_multipass's target_seq when level>0.
    codelm0 = model.codelm_for(0)
    tok0 = R.rgb_byte_pq_fn(flat, codelm0.pq_chunks, codelm0.code_vocab)
    x2 = R.code_embed_proj(tok0, codelm0.own_input_embed, codelm0.own_input_proj)
    target2 = tok0
    code_idx = None
    for i in range(level):
        codelm = model.codelm_for(i)
        out = R.encode_pardec_downsampler(codelm, model.downsampler_for(i), x2, target2, flat, cfg,
                                           pixel_order, R.rgb_label_fn_jax, model.K(i), rate_id=model.bos_rate_id(i),
                                           codelm_rate_id=model.codelm_bos_rate_id(i),
                                           downsampler_ncodes=cfg.downsampler_ncodes[i])
        code_idx = out["code_idx"]
        codelm_next = model.codelm_for(i + 1)
        x2 = R.code_embed_proj(out["code_soft"], codelm_next.own_input_embed, codelm_next.own_input_proj)
        target2 = out["code_idx"]
    return code_idx


if __name__ == "__main__":
    main()
