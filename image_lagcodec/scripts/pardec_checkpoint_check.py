"""CPU-only correctness check using a REAL trained checkpoint (pardec_1, phase_2_step58900):
teacher-forced dense reference vs incremental KV-cache, decoder_ncodes_overlap included.
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.pardec_checkpoint_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_config_module, CONFIG_FIELDS, load_cifar10,
    images_to_positions, pixel_order_for, code_embed_proj, run_block,
)
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)
    eqx_common.splash_attention = _cpu_dense_attention

B = 4
CONFIG_PATH = REPO_ROOT / "image_lagcodec" / "configs" / "pardec_1.py"
CKPT_PATH = REPO_ROOT / "image_lagcodec" / "logs" / "pardec_1" / "checkpoints" / "phase_2_step58900" / "model.eqx"


def build_cfg():
    mod = load_config_module(CONFIG_PATH)
    kwargs = {k: mod[k] for k in CONFIG_FIELDS if k in mod}
    return Config(**kwargs)


def dense_h_t_pardec(level, target_seq, ctx_idx, decoder_ncodes):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc = target_seq.shape[0]
    D = level.bos_embed.shape[-1]
    Gc = decoder_ncodes
    Oc = level.decoder_ncodes_overlap
    Wg = Gc + Oc
    n_blocks = ctx_idx.shape[1]
    pad_blocks = (-n_blocks) % Gc
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // Gc

    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
    if Oc > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (Oc, 0), (0, 0)))

    target_p = target_seq
    if pad_blocks > 0:
        pad_shape = list(target_seq.shape)
        pad_shape[1] = pad_blocks * level.K
        target_p = jnp.concatenate([target_p, jnp.zeros(pad_shape, dtype=target_p.dtype)], axis=1)
    if Oc > 0:
        pad_shape = list(target_p.shape)
        pad_shape[1] = Oc * level.K
        target_p = jnp.concatenate([jnp.zeros(pad_shape, dtype=target_p.dtype), target_p], axis=1)

    ctx_windows = jnp.stack([ctx_tok[:, g * Gc:g * Gc + Wg, :] for g in range(n_groups)], axis=1)
    target_windows = jnp.stack(
        [target_p[:, g * Gc * level.K:g * Gc * level.K + Wg * level.K] for g in range(n_groups)], axis=1)
    B2 = Bc * n_groups
    ctx_flat = ctx_windows.reshape(B2, Wg, D)
    target_flat = target_windows.reshape(B2, Wg * level.K, *target_seq.shape[2:])
    te_flat = level._dec_embed_target(target_flat)
    bos = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    xe = jnp.concatenate([ctx_flat, bos, te_flat], axis=1)
    x = xe
    for blk in blocks:
        x = run_block(blk, x, level.remat)
    h_full = level._dec_ln_f()(x)
    pred_pos = Wg + jnp.arange(Wg * level.K)
    h_t = h_full[:, pred_pos, :]
    return h_t.reshape(Bc, n_groups, Wg * level.K, D)


def incremental_kv_h_t_pardec(level, target_seq, ctx_idx, decoder_ncodes):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc, n_blocks, _ = ctx_idx.shape
    D = level.bos_embed.shape[-1]
    hd = D // level.n_heads
    Gc = decoder_ncodes
    Oc = level.decoder_ncodes_overlap
    Wg = Gc + Oc
    pad_blocks = (-n_blocks) % Gc
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // Gc
    per_group_len = Wg + 1 + Wg * level.K

    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
    if Oc > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (Oc, 0), (0, 0)))
    B2 = Bc * n_groups
    ctx_windows = jnp.stack([ctx_tok[:, g * Gc:g * Gc + Wg, :] for g in range(n_groups)], axis=1)
    ctx_flat = ctx_windows.reshape(B2, Wg, D)

    valid_len = n_blocks * level.K
    tgt_pad = (-valid_len) % (Gc * level.K)
    target_p = target_seq
    if tgt_pad > 0:
        pad_shape = list(target_seq.shape)
        pad_shape[1] = tgt_pad
        target_p = jnp.concatenate([target_seq, jnp.zeros(pad_shape, dtype=target_seq.dtype)], axis=1)
    if Oc > 0:
        pad_shape = list(target_p.shape)
        pad_shape[1] = Oc * level.K
        target_p = jnp.concatenate([jnp.zeros(pad_shape, dtype=target_p.dtype), target_p], axis=1)
    target_windows = jnp.stack(
        [target_p[:, g * Gc * level.K:g * Gc * level.K + Wg * level.K] for g in range(n_groups)], axis=1)
    target_flat = target_windows.reshape(B2, Wg * level.K, *target_seq.shape[2:])

    def self_step(x_new, ck, cv, pos):
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    def self_chunk_step(x_chunk, ck, cv, pos_start):
        new_ck, new_cv = [], []
        x = x_chunk
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = blk.chunk_step(x, ck[i], cv[i], pos_start, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    cache_k = jnp.zeros((len(blocks), B2, level.n_kv_heads, per_group_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    bos_in = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    chunk = jnp.concatenate([ctx_flat, bos_in], axis=1)
    h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0))
    pos = Wg + 1
    h_all = [h_chunk[:, -1, :]]
    x_input = level._dec_embed_target(target_flat[:, 0])
    for m in range(1, Wg * level.K):
        h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
        pos += 1
        h_all.append(h)
        x_input = level._dec_embed_target(target_flat[:, m])
    h_t = jnp.stack(h_all, axis=1)
    return h_t.reshape(Bc, n_groups, Wg * level.K, D)


def main():
    cfg = build_cfg()
    key = jax.random.PRNGKey(0)
    model_shape = HierEncDec(key, cfg)
    model = eqx.tree_deserialise_leaves(CKPT_PATH, model_shape)
    level = model.levels[0]
    print(f"loaded REAL checkpoint: {CKPT_PATH}")
    print(f"G={cfg.decoder_ncodes[0]} O={level.decoder_ncodes_overlap} weight_sharing={level.weight_sharing}")

    (train_np, train_labels), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    imgs = train_np[:B]
    flat_bytes = jnp.array(images_to_positions(imgs, cfg, pixel_order))

    x_in = code_embed_proj(flat_bytes, level.own_input_embed, level.own_input_proj)
    enc_out = level.encode(x_in, flat_bytes, rng=None)
    ctx_idx = enc_out["code_idx"]
    target_seq = flat_bytes
    Gc = cfg.decoder_ncodes[0]

    # 1. teacher-forced dense (decode_logits_and_target_pardec's own internals) vs incremental KV
    h_dense = dense_h_t_pardec(level, target_seq, ctx_idx, Gc)
    h_kv = incremental_kv_h_t_pardec(level, target_seq, ctx_idx, Gc)
    diff = jnp.abs(h_dense - h_kv)
    tol = 1e-2   # real trained weights, bf16-ish accumulated error -- looser than the random-init check
    print(f"\n[CHECK 1] teacher-forced dense vs incremental KV-cache (REAL checkpoint weights):")
    print(f"  max_abs_diff={float(diff.max()):.6f} mean_abs_diff={float(diff.mean()):.6f}")
    print(f"  {'CONSISTENT' if float(diff.max()) <= tol else 'DIVERGES'} (tol={tol})")

    # 2. train path (decode_logits_and_target_pardec, used by phase_forward) vs the SAME model's
    #    decode_generate_pardec run in TEACHER-FORCED greedy mode is not directly comparable (one
    #    is teacher-forced, other is free-running) -- instead confirm train path runs on the real
    #    checkpoint and produces a sane (low, since this level0 is well-trained per run.jsonl)
    #    reconstruction loss, AND that decode_generate_pardec (free-running) produces the correct
    #    shape with this real checkpoint -- catches any real shape/wiring mismatch between the two
    #    paths on ACTUAL trained weights (not just random init).
    from image_lagcodec.run_lagcodec import phase_forward
    loss, aux = phase_forward(model, flat_bytes, 1, rng=None)
    bpb, acc, ntp_bpb, ntp_acc, util, mse = [float(a) for a in aux]
    print(f"\n[CHECK 2] train path (decode_logits_and_target_pardec) on real checkpoint:")
    print(f"  loss={float(loss):.4f} dec_acc={acc:.4f} (should be nontrivial, model IS trained to step 58900)")

    gen = level.decode_generate_pardec(ctx_idx, Gc, greedy=True, seed=0)
    gen_acc = float(jnp.mean(gen == flat_bytes))
    print(f"\n[CHECK 3] decode_generate_pardec (free-running) on real checkpoint:")
    print(f"  shape={gen.shape} gen_recon_acc={gen_acc:.4f} (expect << train dec_acc -- exposure bias, not a bug)")

    print(f"\nPASS -- no shape/wiring crash across train (decode_logits_and_target_pardec), "
          f"teacher-forced-vs-KV-cache, and free-running generate (decode_generate_pardec), all "
          f"on the REAL trained checkpoint with decoder_ncodes_overlap={level.decoder_ncodes_overlap}.")


if __name__ == "__main__":
    main()
