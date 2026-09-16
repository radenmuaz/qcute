"""CPU-only correctness check for the v2 (ncodes_window) scheme: teacher-forced dense reference
vs incremental KV-cache, on a real train sample, no target lookback/redecode, global RoPE
positions, masked padding. Tests bounded N and -1 (unbounded, naive full-width).
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.pardec_v2_consistency_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_cifar10, images_to_positions, pixel_order_for, code_embed_proj,
    run_block_pardec, pardec_block_step, pardec_block_chunk_step,
)
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale, window=None, sink=None):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            if window is not None:
                mask = mask & (jnp.arange(T)[None, :] > jnp.arange(T)[:, None] - window)
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)   # sink not modeled here (fullctx-path test only)
    eqx_common.splash_attention = _cpu_dense_attention

B = 4
G = 4


def build_cfg(ncodes_window, streaming=True):
    return Config(
        d_model=(64, 64, 64, 64), n_layers=(2, 2, 2, 2), n_heads=(2, 2, 2, 2),
        strides=(4, 4, 4, -1), code_vocab=(16, 16, 16, 16), pq_chunks=(4, 4, 4, 4),
        pq_dim=(32, 16, 16, 16), byte_group=3, token_head_type="linears", mtp_horizon=1,
        decoder_ncodes=(G, G, G, G), ncodes_window=(ncodes_window, 0, 0, 0),
        streaming=(streaming, True, True, True),
        weight_sharing=True, curriculum_mode="no_freeze",
    )


def dense_h_t(level, target_seq, ctx_idx, decoder_ncodes):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc = target_seq.shape[0]
    D = level.bos_embed.shape[-1]
    Gc = decoder_ncodes
    n_blocks = ctx_idx.shape[1]
    pad_blocks = (-n_blocks) % Gc
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // Gc
    fullctx = (not level.streaming) and (level.ncodes_window == -1)
    N = level.ncodes_window if level.ncodes_window >= 0 else (n_groups - 1)
    Wg = n_blocks_p if fullctx else (N + 1) * Gc
    per_group_len = Wg + 1 + Gc * level.K

    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    target_p = target_seq
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        target_p = jnp.pad(target_p, ((0, 0), (0, pad_blocks * level.K), (0, 0)))
    if not fullctx and N > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (N * Gc, 0), (0, 0)))

    B2 = Bc * n_groups
    if fullctx:
        ctx_flat = jnp.broadcast_to(ctx_tok[:, None, :, :], (Bc, n_groups, Wg, D)).reshape(B2, Wg, D)
        min_valid_pos = jnp.zeros((B2,), dtype=jnp.int32)
        rope_ctx_g = jnp.broadcast_to(jnp.arange(Wg)[None, :], (n_groups, Wg))
    else:
        ctx_windows = jnp.stack([ctx_tok[:, g * Gc:g * Gc + Wg, :] for g in range(n_groups)], axis=1)
        ctx_flat = ctx_windows.reshape(B2, Wg, D)
        fake_counts = jnp.array([max(0, N - g) * Gc for g in range(n_groups)])
        min_valid_pos = jnp.broadcast_to(fake_counts[None, :], (Bc, n_groups)).reshape(B2)
        rope_ctx_g = jnp.stack([jnp.clip(jnp.arange(Wg) - N * Gc + g * Gc, 0, None) for g in range(n_groups)], axis=0)

    target_windows = jnp.stack(
        [target_p[:, g * Gc * level.K:(g + 1) * Gc * level.K] for g in range(n_groups)], axis=1)
    target_flat = target_windows.reshape(B2, Gc * level.K, *target_seq.shape[2:])
    te_flat = level._dec_embed_target(target_flat)
    bos = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    xe = jnp.concatenate([ctx_flat, bos, te_flat], axis=1)

    rope_bos = jnp.array([(g + 1) * Gc for g in range(n_groups)])[:, None]
    rope_target = jnp.stack([(g + 1) * Gc + 1 + jnp.arange(Gc * level.K) for g in range(n_groups)], axis=0)
    rope_pos_ids_g = jnp.concatenate([rope_ctx_g, rope_bos, rope_target], axis=1)
    rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (Bc, n_groups, per_group_len)).reshape(B2, per_group_len)

    x = xe
    for blk in blocks:
        x = run_block_pardec(blk, x, rope_pos_ids, min_valid_pos, level.remat)
    h = ln_f(x)
    pred_pos = Wg + jnp.arange(Gc * level.K)
    h_t = h[:, pred_pos, :]
    return h_t.reshape(Bc, n_groups, Gc * level.K, D), n_groups, N, Wg, per_group_len


def incremental_kv_h_t(level, target_seq, ctx_idx, decoder_ncodes):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc, n_blocks, _ = ctx_idx.shape
    D = level.bos_embed.shape[-1]
    hd = D // level.n_heads
    Gc = decoder_ncodes
    pad_blocks = (-n_blocks) % Gc
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // Gc
    fullctx = (not level.streaming) and (level.ncodes_window == -1)
    N = level.ncodes_window if level.ncodes_window >= 0 else (n_groups - 1)
    Wg = n_blocks_p if fullctx else (N + 1) * Gc
    per_group_len = Wg + 1 + Gc * level.K

    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
    if not fullctx and N > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (N * Gc, 0), (0, 0)))
    B2 = Bc * n_groups
    if fullctx:
        ctx_flat = jnp.broadcast_to(ctx_tok[:, None, :, :], (Bc, n_groups, Wg, D)).reshape(B2, Wg, D)
        min_valid_pos = jnp.zeros((B2,), dtype=jnp.int32)
        rope_ctx_flat = jnp.broadcast_to(jnp.arange(Wg)[None, :], (B2, Wg))
    else:
        ctx_windows = jnp.stack([ctx_tok[:, g * Gc:g * Gc + Wg, :] for g in range(n_groups)], axis=1)
        ctx_flat = ctx_windows.reshape(B2, Wg, D)
        fake_counts = jnp.array([max(0, N - g) * Gc for g in range(n_groups)])
        min_valid_pos = jnp.broadcast_to(fake_counts[None, :], (Bc, n_groups)).reshape(B2)
        rope_ctx = jnp.stack([jnp.clip(jnp.arange(Wg) - N * Gc + g * Gc, 0, None) for g in range(n_groups)], axis=0)
        rope_ctx_flat = jnp.broadcast_to(rope_ctx[None], (Bc, n_groups, Wg)).reshape(B2, Wg)
    rope_bos = jnp.array([(g + 1) * Gc for g in range(n_groups)])
    rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (Bc, n_groups)).reshape(B2)

    valid_len = n_blocks * level.K
    tgt_pad = (-valid_len) % (Gc * level.K)
    target_p = target_seq
    if tgt_pad > 0:
        pad_shape = list(target_seq.shape)
        pad_shape[1] = tgt_pad
        target_p = jnp.concatenate([target_seq, jnp.zeros(pad_shape, dtype=target_seq.dtype)], axis=1)
    target_windows = jnp.stack(
        [target_p[:, g * Gc * level.K:(g + 1) * Gc * level.K] for g in range(n_groups)], axis=1)
    target_flat = target_windows.reshape(B2, Gc * level.K, *target_seq.shape[2:])

    def self_step(x_new, ck, cv, pos, rope_pos_row):
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = pardec_block_step(blk, x, ck[i], cv[i], pos, rope_pos_row, min_valid_pos, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    def self_chunk_step(x_chunk, ck, cv, pos_start, rope_pos_ids_chunk):
        new_ck, new_cv = [], []
        x = x_chunk
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = pardec_block_chunk_step(blk, x, ck[i], cv[i], pos_start, rope_pos_ids_chunk,
                                                      min_valid_pos, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    cache_k = jnp.zeros((len(blocks), B2, level.n_kv_heads, per_group_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    bos_in = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    chunk = jnp.concatenate([ctx_flat, bos_in], axis=1)
    chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
    h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
    pos = Wg + 1
    rope_pos_row = rope_bos_flat + 1
    h_all = [h_chunk[:, -1, :]]
    x_input = level._dec_embed_target(target_flat[:, 0])
    for m in range(1, Gc * level.K):
        h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rope_pos_row)
        pos += 1
        rope_pos_row = rope_pos_row + 1
        h_all.append(h)
        x_input = level._dec_embed_target(target_flat[:, m])
    h_t = jnp.stack(h_all, axis=1)
    return h_t.reshape(Bc, n_groups, Gc * level.K, D)


def run_one(ncodes_window, streaming=True):
    cfg = build_cfg(ncodes_window, streaming=streaming)
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

    h_dense, n_groups, N, Wg, per_group_len = dense_h_t(level, target_seq, ctx_idx, G)
    h_kv = incremental_kv_h_t(level, target_seq, ctx_idx, G)
    diff = jnp.abs(h_dense - h_kv)
    tol = 1e-3
    n_nan_dense = int(jnp.isnan(h_dense).sum())
    n_nan_kv = int(jnp.isnan(h_kv).sum())
    print(f"ncodes_window={ncodes_window} streaming={streaming} (effective N={N}) G={G} Wg={Wg} "
          f"n_groups={n_groups} per_group_len={per_group_len}")
    print(f"  NaN check: h_dense={n_nan_dense} h_kv={n_nan_kv}")
    print(f"  dense-vs-kv-cache: max_abs_diff={float(diff.max()):.6f} mean_abs_diff={float(diff.mean()):.6f}")
    ok = (float(diff.max()) <= tol) and n_nan_dense == 0 and n_nan_kv == 0
    print(f"  {'CONSISTENT' if ok else 'DIVERGES/NAN'} (tol={tol})")

    out = level.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0)
    print(f"  decode_generate_pardec shape={out.shape} expected=({B}, {flat_bytes.shape[1]}, {cfg.byte_group})")
    assert out.shape == (B, flat_bytes.shape[1], cfg.byte_group)
    return ok


if __name__ == "__main__":
    ok0 = run_one(0)                        # disjoint (sanity baseline)
    ok2 = run_one(2)                        # bounded N=2, causal streaming
    okm1 = run_one(-1)                      # all, causal streaming (unbounded)
    okfc = run_one(-1, streaming=False)     # all, non-causal (true fullctx)
    print(f"\nPASS all" if (ok0 and ok2 and okm1 and okfc) else "\nFAIL -- see divergence above")
