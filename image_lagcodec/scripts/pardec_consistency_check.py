"""CPU-only correctness check for decode_generate_pardec (run_lagcodec.py), INCLUDING the
decoder_ncodes_overlap lookback/prune logic: teacher-forced dense reference vs incremental
KV-cache, on a real train sample. No checkpoint needed (random-init weights) -- this tests
implementation correctness, not trained accuracy. Mirrors kv_consistency_check.py's methodology
(same-ground-truth-prefix comparison), extended to the pardec group-window construction.
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.pardec_consistency_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_cifar10, images_to_positions, pixel_order_for, code_embed_proj,
    run_block,
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
G = 4       # decoder_ncodes
O = 2       # decoder_ncodes_overlap -- the new feature under test (0 < O <= G)


def build_cfg():
    return Config(
        d_model=(64, 64, 64, 64), n_layers=(2, 2, 2, 2), n_heads=(2, 2, 2, 2),
        strides=(4, 4, 4, -1), code_vocab=(16, 16, 16, 16), pq_chunks=(4, 4, 4, 4),
        pq_dim=(32, 16, 16, 16), byte_group=3, token_head_type="linears", mtp_horizon=1,
        decoder_ncodes=(G, G, G, G), decoder_ncodes_overlap=(O, 0, 0, 0),
        weight_sharing=True, curriculum_mode="no_freeze",
    )


def dense_h_t_pardec(level, target_seq, ctx_idx, decoder_ncodes):
    """Dense (batched, no cache) teacher-forced reference: builds the SAME windowed groups as
    decode_generate_pardec (Wg=G+O ctx blocks per group, O-zero-padded for group 0), but using
    dense self-attention over each group's full window, fed with REAL target values."""
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

    # pad RAW target indices (not the embedded tensor) so the lookback region's "ground truth" is
    # embed(index=0), matching incremental_kv_h_t_pardec's convention exactly -- padding the
    # already-embedded tensor with true zero vectors would NOT equal embed(0) (test-harness-only
    # subtlety; decode_generate_pardec itself never needs this since real generation has no
    # ground truth to simulate for the lookback region).
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
    h_full = ln_f(x)
    pred_pos = Wg + jnp.arange(Wg * level.K)
    h_t = h_full[:, pred_pos, :]   # (B2, Wg*K, D)
    return h_t.reshape(Bc, n_groups, Wg * level.K, D)


def incremental_kv_h_t_pardec(level, target_seq, ctx_idx, decoder_ncodes):
    """Teacher-forced mirror of decode_generate_pardec's run_pardec -- feeds REAL target_seq
    values (windowed the same way, zero-padded lookback for group 0) instead of sampled ones."""
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
    h_t = jnp.stack(h_all, axis=1)   # (B2, Wg*K, D)
    return h_t.reshape(Bc, n_groups, Wg * level.K, D)


def main():
    cfg = build_cfg()
    key = jax.random.PRNGKey(0)
    model = HierEncDec(key, cfg)
    level = model.levels[0]

    (train_np, train_labels), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    imgs = train_np[:B]
    flat_bytes = jnp.array(images_to_positions(imgs, cfg, pixel_order))

    x_in = code_embed_proj(flat_bytes, level.own_input_embed, level.own_input_proj)
    enc_out = level.encode(x_in, flat_bytes, rng=None)
    ctx_idx = enc_out["code_idx"]
    target_seq = flat_bytes
    Gc = cfg.decoder_ncodes[0]

    h_dense = dense_h_t_pardec(level, target_seq, ctx_idx, Gc)
    h_kv = incremental_kv_h_t_pardec(level, target_seq, ctx_idx, Gc)

    diff = jnp.abs(h_dense - h_kv)
    tol = 1e-3
    print(f"G={Gc} O={level.decoder_ncodes_overlap} Wg={Gc + level.decoder_ncodes_overlap} "
          f"n_groups={h_dense.shape[1]} shape={h_dense.shape}")
    print(f"dense-vs-kv-cache (pardec, full Wg*K window incl. lookback): "
          f"max_abs_diff={float(diff.max()):.6f} mean_abs_diff={float(diff.mean()):.6f}")
    if float(diff.max()) <= tol:
        print(f"CONSISTENT (all positions within tol={tol})")
    else:
        per_pos_max = diff.max(axis=(0, 1, 3))
        first = int(jnp.where(per_pos_max > tol)[0][0])
        print(f"DIVERGES starting at within-group position {first} (tol={tol})")

    # sanity: also run the actual generation entrypoint end to end, greedy, and check shapes/dtype
    out = level.decode_generate_pardec(ctx_idx, Gc, greedy=True, seed=0)
    print(f"decode_generate_pardec output shape={out.shape} dtype={out.dtype} "
          f"expected=({B}, {flat_bytes.shape[1]}, {cfg.byte_group})")
    assert out.shape == (B, flat_bytes.shape[1], cfg.byte_group)
    print("PASS -- decode_generate_pardec (with overlap) runs, correct shape" +
          (", and hidden states are numerically consistent with the dense teacher-forced reference"
           if float(diff.max()) <= tol else " (see divergence above)"))


if __name__ == "__main__":
    main()
