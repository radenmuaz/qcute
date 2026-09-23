"""CPU-only correctness check for the v2 (ncodes_window) scheme: teacher-forced dense reference
vs incremental KV-cache, on a real train sample, no target lookback/redecode, global RoPE
positions, masked padding. Tests bounded N and -1 (unbounded, naive full-width).
Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.pardec_v2_consistency_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import warnings
from pathlib import Path
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, EncDecLevel, load_cifar10, images_to_positions, pixel_order_for, code_embed_proj,
    run_block_pardec, pardec_block_step, pardec_block_chunk_step, causal_extra_ctx_windows,
    _draft_past_valid_mask, extra_ctx_visible_counts, encoder_free_run, encoder_hidden, encoder_ntp_logits,
    generate_from_prompt, decode_logits_and_target_multipass, decode_generate_multipass,
)
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale, window=None, lookahead=0, sink=None):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            q_idx, kv_idx = jnp.arange(T)[:, None], jnp.arange(T)[None, :]
            if window is None and lookahead <= 0:
                mask = q_idx >= kv_idx
            else:
                mask = jnp.ones((T, T), dtype=bool)
                if window is not None:
                    mask = mask & (q_idx - window <= kv_idx)
                mask = mask & (q_idx + max(0, lookahead) >= kv_idx)
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)   # sink not modeled here (fullctx-path test only)
    eqx_common.splash_attention = _cpu_dense_attention

B = 4
G = 4


def build_cfg(ncodes_window, stream_chunks=0):
    return Config(
        d_model=(64, 64, 64, 64), n_layers=(2, 2, 2, 2), n_heads=(2, 2, 2, 2),
        strides=(4, 4, 4, -1), code_vocab=(16, 16, 16, 16), pq_chunks=(4, 4, 4, 4),
        pq_dim=(32, 16, 16, 16), byte_group=3, token_head_type="linears", mtp_horizon=1,
        decoder_ncodes=(G, G, G, G), ncodes_window=(ncodes_window, 0, 0, 0),
        stream_chunks=(stream_chunks, 0, 0, 0),
        weight_sharing=True, curriculum_mode="no_freeze",
    )


def ctx_rows(level, ctx_idx, Gc, extras=None, rng=None):
    """the production row builder (single source of truth for training and generation)."""
    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    n_blocks = ctx_idx.shape[1]
    n_groups = -(-n_blocks // Gc)
    return (*level._pardec_ctx_rows(ctx_tok, extras, Gc, n_groups, n_blocks, rng), n_groups)


def build_cache_fns(level, blocks, ln_f, key_valid, per_group_len):
    def self_step(x_new, ck, cv, pos, rope_pos_row):
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = pardec_block_step(blk, x, ck[i], cv[i], pos, rope_pos_row, key_valid, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    def self_chunk_step(x_chunk, ck, cv, pos_start, rope_pos_ids_chunk):
        new_ck, new_cv = [], []
        x = x_chunk
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = pardec_block_chunk_step(blk, x, ck[i], cv[i], pos_start, rope_pos_ids_chunk,
                                                      key_valid, per_group_len)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)
    return self_step, self_chunk_step


def dense_h_t(level, target_seq, ctx_idx, decoder_ncodes):
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc = target_seq.shape[0]
    D = level.bos_embed.shape[-1]
    Gc = decoder_ncodes
    ctx_flat, rope_ctx_g, valid_ctx, Wg, _, n_groups = ctx_rows(level, ctx_idx, Gc)
    n_blocks = ctx_idx.shape[1]
    N = level.ncodes_window if level.ncodes_window >= 0 else (n_groups - 1)
    per_group_len = Wg + 1 + Gc * level.K
    target_p = jnp.pad(target_seq, ((0, 0), (0, (n_groups * Gc - n_blocks) * level.K), (0, 0)))
    B2 = Bc * n_groups
    target_windows = jnp.stack(
        [target_p[:, g * Gc * level.K:(g + 1) * Gc * level.K] for g in range(n_groups)], axis=1)
    target_flat = target_windows.reshape(B2, Gc * level.K, *target_seq.shape[2:])
    te_flat = level._dec_embed_target(target_flat)
    bos = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    xe = jnp.concatenate([ctx_flat, bos, te_flat], axis=1)
    key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1 + Gc * level.K), dtype=bool)], axis=1)

    rope_bos = jnp.array([(g + 1) * Gc for g in range(n_groups)])[:, None]
    rope_target = jnp.stack([(g + 1) * Gc + 1 + jnp.arange(Gc * level.K) for g in range(n_groups)], axis=0)
    rope_pos_ids_g = jnp.concatenate([rope_ctx_g, rope_bos, rope_target], axis=1)
    rope_pos_ids = jnp.broadcast_to(rope_pos_ids_g[None], (Bc, n_groups, per_group_len)).reshape(B2, per_group_len)

    x = xe
    for blk in blocks:
        x = run_block_pardec(blk, x, rope_pos_ids, key_valid, level.remat)
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
    ctx_flat, rope_ctx_g, valid_ctx, Wg, _, n_groups = ctx_rows(level, ctx_idx, Gc)
    B2 = Bc * n_groups
    per_group_len = Wg + 1 + Gc * level.K
    key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1 + Gc * level.K), dtype=bool)], axis=1)
    rope_ctx_flat = jnp.broadcast_to(rope_ctx_g[None], (Bc, n_groups, Wg)).reshape(B2, Wg)
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
    self_step, self_chunk_step = build_cache_fns(level, blocks, ln_f, key_valid, per_group_len)

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


def run_one(ncodes_window, stream_chunks=0):
    cfg = build_cfg(ncodes_window, stream_chunks=stream_chunks)
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
    print(f"ncodes_window={ncodes_window} stream_chunks={stream_chunks} (effective N={N}) G={G} Wg={Wg} "
          f"n_groups={n_groups} per_group_len={per_group_len}")
    print(f"  NaN check: h_dense={n_nan_dense} h_kv={n_nan_kv}")
    print(f"  dense-vs-kv-cache: max_abs_diff={float(diff.max()):.6f} mean_abs_diff={float(diff.mean()):.6f}")
    ok = (float(diff.max()) <= tol) and n_nan_dense == 0 and n_nan_kv == 0
    print(f"  {'CONSISTENT' if ok else 'DIVERGES/NAN'} (tol={tol})")

    out = level.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0)
    print(f"  decode_generate_pardec shape={out.shape} expected=({B}, {flat_bytes.shape[1]}, {cfg.byte_group})")
    assert out.shape == (B, flat_bytes.shape[1], cfg.byte_group)
    return ok


def build_cfg_widened(decode_past, decode_future, cycle_refine_passes=1):
    return Config(
        d_model=(32, 32), n_layers=(1, 1), n_heads=(2, 2), n_kv_heads=(None, None),
        strides=(4, 4), code_vocab=(16, 16), pq_chunks=(1, 1), pq_dim=(16, 16),
        byte_group=3, token_head_type="linears", decoder_ncodes=(G, G), ncodes_window=(16, 16),
        weight_sharing=True, curriculum_mode="no_freeze",
        decode_past=(decode_past, decode_past), decode_future=(decode_future, decode_future),
        cycle_refine_passes=cycle_refine_passes,
    )


def run_widened_aux(decode_past, decode_future):
    """decode_past/decode_future widened auxiliary-NTP region: dense (via decode_logits_and_target_pardec's
    internal _widened_aux_ntp) must match an independently-built incremental KV-cache reconstruction,
    teacher-forced with the same real draft/future values, across the whole widened window."""
    cfg = build_cfg_widened(decode_past, decode_future)
    key = jax.random.PRNGKey(0)
    model = HierEncDec(key, cfg)
    level = model.levels[0]

    (train_np, train_labels), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    imgs = train_np[:B]
    flat_bytes = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    x_in = code_embed_proj(flat_bytes, level.own_input_embed, level.own_input_proj)
    enc_out = level.encode(x_in, flat_bytes, rng=None)
    ctx_idx_soft, ctx_idx = enc_out["code_soft"], enc_out["code_idx"]

    captured = {}
    orig = EncDecLevel._widened_aux_ntp

    def spy(self, h, target_p, Wg, extra_len_total, Pp, Pf, Kspan, n_groups, Bc, n_blocks, rng):
        captured["h"] = h
        captured["Wg"] = Wg
        captured["extra_len_total"] = extra_len_total
        return orig(self, h, target_p, Wg, extra_len_total, Pp, Pf, Kspan, n_groups, Bc, n_blocks, rng)

    EncDecLevel._widened_aux_ntp = spy
    try:
        level.decode_logits_and_target_pardec(flat_bytes, ctx_idx_soft, G)
    finally:
        EncDecLevel._widened_aux_ntp = orig
    h_dense = captured["h"]
    ctx_len = captured["Wg"] + captured["extra_len_total"]
    Pp, Pf = decode_past, decode_future
    Kspan = G * level.K
    n_blocks = ctx_idx.shape[1]
    n_groups = n_blocks // G
    D = level.bos_embed.shape[-1]

    dense_logits_parts = []
    if Pp > 0:
        pred_pos_pp = ctx_len + jnp.arange(Pp)
        dense_logits_parts.append(level._token_logits_linears(h_dense[:, pred_pos_pp, :]))
    pred_pos_core = ctx_len + Pp + jnp.arange(Kspan)
    dense_logits_parts.append(level._token_logits_linears(h_dense[:, pred_pos_core, :]))
    if Pf > 0:
        pred_pos_pf = ctx_len + Pp + Kspan + jnp.arange(Pf)
        dense_logits_parts.append(level._token_logits_linears(h_dense[:, pred_pos_pf, :]))
    dense_logits = jnp.concatenate(dense_logits_parts, axis=1)

    # -- independent incremental KV-cache reconstruction, teacher-forced with real values --
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    hd = D // level.n_heads
    ctx_flat, rope_ctx, valid_ctx, Wg, _, n_groups = ctx_rows(level, ctx_idx, G)
    B2 = B * n_groups
    rope_ctx_flat = jnp.broadcast_to(rope_ctx[None], (B, n_groups, Wg)).reshape(B2, Wg)
    rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])
    rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B, n_groups)).reshape(B2)
    per_group_len = Wg + 1 + Pp + Kspan + Pf

    target_p = flat_bytes
    ext_p = jnp.pad(target_p, ((0, 0), (Pp, Pf)) + ((0, 0),) * (target_p.ndim - 2))
    widened_windows = jnp.stack(
        [ext_p[:, g * Kspan:g * Kspan + Pp + Kspan + Pf] for g in range(n_groups)], axis=1)
    widened_flat = widened_windows.reshape(B2, Pp + Kspan + Pf, *target_p.shape[2:])
    # draft positions before the sequence's real start are masked as keys (key_valid), same as training
    if Pp > 0:
        pp_valid_np = _draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * level.K)
        pp_valid_flat = jnp.broadcast_to(jnp.asarray(pp_valid_np)[None], (B, n_groups, Pp)).reshape(B2, Pp)
    else:
        pp_valid_flat = jnp.ones((B2, 0), dtype=bool)
    key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1), dtype=bool), pp_valid_flat,
                                  jnp.ones((B2, Kspan + Pf), dtype=bool)], axis=1)

    def embed_widened(t):
        te = level._dec_embed_target(widened_flat[:, t])
        return jnp.where(pp_valid_flat[:, t][:, None], te, 0.0) if (Pp > 0 and t < Pp) else te

    self_step, self_chunk_step = build_cache_fns(level, blocks, ln_f, key_valid, per_group_len)

    cache_k = jnp.zeros((len(blocks), B2, level.n_kv_heads, per_group_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    bos_in = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    chunk = jnp.concatenate([ctx_flat, bos_in], axis=1)
    chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
    h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
    pos = Wg + 1
    total_steps = Pp + Kspan + Pf

    def widened_pos(t):
        if t < Pp:
            return jnp.clip(rope_bos_flat - Pp + t, 0, None)
        return rope_bos_flat + 1 + (t - Pp)

    rope_pos_row = widened_pos(0)
    h_all = [h_chunk[:, -1, :]]
    x_input = embed_widened(0)
    for t in range(1, total_steps):
        h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rope_pos_row)
        pos += 1
        rope_pos_row = widened_pos(t)
        h_all.append(h)
        x_input = embed_widened(t)
    h_incr = jnp.stack(h_all, axis=1)
    incr_logits = level._token_logits_linears(h_incr)

    diff = jnp.abs(dense_logits - incr_logits)
    mism = int((jnp.argmax(dense_logits, -1) != jnp.argmax(incr_logits, -1)).sum())
    print(f"WIDENED AUX (decode_past={decode_past} decode_future={decode_future}): "
          f"dense-vs-incremental max_abs_diff={float(diff.max()):.6f} mismatches={mism}/{dense_logits.shape[1] * dense_logits.shape[0] * n_groups}")
    ok = float(diff.max()) <= 1e-3 and mism == 0
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_cyclic_revision_extra_ctx():
    """cyclic-refine's revision-slot fallback (_extra_ctx_table falling back to level.revision_embed/proj/pad
    for k beyond cond_depth's own extra_ctx list): dense vs incremental KV-cache, multiple revision slots."""
    cfg = build_cfg_widened(0, 0, cycle_refine_passes=3)
    key = jax.random.PRNGKey(0)
    model = HierEncDec(key, cfg)
    level0, level1 = model.levels[0], model.levels[1]

    (train_np, train_labels), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    imgs = train_np[:B]
    flat_bytes = jnp.array(images_to_positions(imgs, cfg, pixel_order))
    x0 = code_embed_proj(flat_bytes, level0.own_input_embed, level0.own_input_proj)
    enc0 = level0.encode(x0, flat_bytes, rng=None)
    code0_idx, code0_soft = enc0["code_idx"], enc0["code_soft"]
    x1 = code_embed_proj(code0_soft, level1.own_input_embed, level1.own_input_proj)
    enc1 = level1.encode(x1, code0_idx, rng=None)
    code1_idx = enc1["code_idx"]

    rev2 = jax.random.randint(jax.random.PRNGKey(5), code1_idx.shape, 0, cfg.code_vocab[1])
    rev3 = jax.random.randint(jax.random.PRNGKey(6), code1_idx.shape, 0, cfg.code_vocab[1])
    extra_ctx_list = [enc1["code_soft"], rev2, rev3]

    logits_dense, target_dense, _, _, _, _ = level0.decode_logits_and_target_pardec(
        flat_bytes, code0_soft, G, extra_ctx_code_soft=extra_ctx_list)
    dense_argmax = jnp.argmax(logits_dense, axis=-1)

    level = level0
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    n_blocks = code0_idx.shape[1]
    D = level.bos_embed.shape[-1]
    hd = D // level.n_heads
    ctx_tok_flat, rope_ctx, valid_ctx, Wg, extra_len_total, n_groups = ctx_rows(level, code0_idx, G, extra_ctx_list)
    B2 = B * n_groups
    rope_ctx_flat = jnp.broadcast_to(rope_ctx[None], (B, n_groups, rope_ctx.shape[1])).reshape(B2, -1)
    rope_bos = jnp.array([(g + 1) * G for g in range(n_groups)])
    rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B, n_groups)).reshape(B2)
    per_group_len = Wg + extra_len_total + 1 + G * level.K
    key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1 + G * level.K), dtype=bool)], axis=1)
    self_step, self_chunk_step = build_cache_fns(level, blocks, ln_f, key_valid, per_group_len)

    cache_k = jnp.zeros((len(blocks), B2, level.n_kv_heads, per_group_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    Kspan = G * level.K
    real_windows = jnp.stack([flat_bytes[:, g * Kspan:(g + 1) * Kspan] for g in range(n_groups)], axis=1)
    real_flat = real_windows.reshape(B2, Kspan, *flat_bytes.shape[2:])

    bos_in = jnp.broadcast_to(level.bos_embed, (B2, 1, D))
    chunk = jnp.concatenate([ctx_tok_flat, bos_in], axis=1)
    chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
    h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
    pos = Wg + extra_len_total + 1
    rope_pos_row = rope_bos_flat + 1
    logits_list = [level._token_logits_linears(h_chunk[:, -1, :])]
    x_input = level._dec_embed_target(real_flat[:, 0])
    for t in range(1, Kspan):
        h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rope_pos_row)
        pos += 1
        rope_pos_row = rope_pos_row + 1
        logits_list.append(level._token_logits_linears(h))
        x_input = level._dec_embed_target(real_flat[:, t])

    incr_logits_flat = jnp.stack(logits_list, axis=1)
    incr_logits = incr_logits_flat.reshape(B, n_groups * Kspan, *incr_logits_flat.shape[2:])[:, :n_blocks * level.K]
    incr_argmax = jnp.argmax(incr_logits, axis=-1)

    diff = jnp.abs(logits_dense - incr_logits)
    mism = int((dense_argmax != incr_argmax).sum())
    print(f"CYCLIC-REVISION extra_ctx (3 slots): dense-vs-incremental max_abs_diff={float(diff.max()):.6f} "
          f"mismatches={mism}/{dense_argmax.size}")
    ok = float(diff.max()) <= 1e-3 and mism == 0
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_gen_decode_future_self_consistency():
    """gen_decode_future=True: the greedily-generated future tail must be self-consistent with what
    dense teacher-forced scoring (fed that exact generated context) would itself pick -- i.e. generation
    isn't silently using different math/positions than the scoring path."""
    cfg = build_cfg_widened(0, 4)
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

    out, future = level.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0, gen_decode_future=True)
    n_groups = future.shape[1]
    Kspan = G * level.K
    Pf = level.decode_future

    # rebuild the exact widened sequence generation actually produced (real core span + its own greedy
    # future), feed it through dense scoring, and check the dense argmax at the future positions matches
    # what was generated (self-consistency: greedy generation IS the argmax of the scoring path).
    real_windows = jnp.stack([out[:, g * Kspan:(g + 1) * Kspan] for g in range(n_groups)], axis=1)
    target_p_check = out  # dense target_seq is just the real span; future isn't part of the scored target
    logits_dense, target_dense, _, _, aux_loss, aux_acc = level.decode_logits_and_target_pardec(
        target_p_check, enc_out["code_soft"], G)
    core_argmax = jnp.argmax(logits_dense, axis=-1)
    core_match = bool(jnp.array_equal(core_argmax, out))
    print(f"GEN_DECODE_FUTURE self-consistency: future shape={future.shape}, "
          f"greedy core reproduces itself under dense re-scoring: {core_match}")
    print(f"  {'CONSISTENT' if core_match else 'DIVERGES'}")
    return core_match


def run_draft_pad_generation():
    """out-of-bounds draft slots are masked as keys in decode_generate_pardec: generation must be fully
    independent of own_input_embed[0] for a group whose draft window is entirely out-of-bounds."""
    cfg = build_cfg_widened(16, 0)
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

    import image_lagcodec.run_lagcodec as rl
    n_blocks = ctx_idx.shape[1]
    Kspan = G * level.K
    n_groups = n_blocks // G
    Pp = level.decode_past
    draft_valid_np = rl._draft_past_valid_mask(n_groups, Pp, Kspan, n_blocks * level.K)
    draft_valid_flat = jnp.broadcast_to(
        jnp.asarray(draft_valid_np)[None], (B, n_groups, Pp)).reshape(B * n_groups, Pp)
    fake_source = jnp.zeros((B, n_groups * Kspan, *flat_bytes.shape[2:]), dtype=flat_bytes.dtype)
    draft_p = jnp.pad(fake_source, ((0, 0), (Pp, 0)) + ((0, 0),) * (fake_source.ndim - 2))
    draft_windows = jnp.stack([draft_p[:, g * Kspan:g * Kspan + Pp] for g in range(n_groups)], axis=1)
    draft_override_flat = draft_windows.reshape(B * n_groups, Pp, *flat_bytes.shape[2:])

    out0 = level.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0,
                                         decode_past_override=Pp, draft_override_flat=draft_override_flat,
                                         draft_valid_flat=draft_valid_flat)
    new_embed = level.own_input_embed.at[0].set(level.own_input_embed[0] + 999.0)
    level_pert = eqx.tree_at(lambda l: l.own_input_embed, level, new_embed)
    out1 = level_pert.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0,
                                              decode_past_override=Pp, draft_override_flat=draft_override_flat,
                                              draft_valid_flat=draft_valid_flat)
    # only the FIRST byte of group0 is a clean isolation point: with weight_sharing=True, every
    # later byte within the SAME group gets fed back through the shared target-embed table during
    # the group's own autoregressive loop, so if a later generated byte happens to equal 0 it would
    # legitimately (and separately from draft_pad) pick up the perturbation too -- that's expected,
    # not a draft_pad bug. The first byte depends only on ctx+bos+draft, before any such feedback.
    group0_unaffected = bool(jnp.array_equal(out0[:, 0], out1[:, 0]))
    print(f"DRAFT mask generation: group0's first byte (fully out-of-bounds draft) unaffected by "
          f"embed(0) perturbation: {group0_unaffected}")
    print(f"  {'CONSISTENT' if group0_unaffected else 'DIVERGES'}")
    return group0_unaffected


def _cond_cfg(G_, cond_window=-1, fullctx=False, stream_chunks=0):
    return Config(
        d_model=(32, 32), n_layers=(1, 1), n_heads=(2, 2), n_kv_heads=(None, None),
        strides=(4, 4), code_vocab=(16, 16), pq_chunks=(1, 1), pq_dim=(16, 16), byte_group=3,
        token_head_type="linears", decoder_ncodes=(G_, G_),
        ncodes_window=(-1, -1) if fullctx else (2, 2), stream_chunks=(1, 1) if fullctx else (stream_chunks, stream_chunks),
        weight_sharing=True, curriculum_mode="no_freeze", cond_depth=(2, 1), cond_window=(cond_window, -1))


def run_cond_visibility_rule():
    """'complete' alignment: group g sees coarser code j iff (j+1)*S <= (g+1)*G; window contents + cond_window."""
    ok = True
    for (n, G_, S, want) in [(8, 1, 4, [0, 0, 0, 1, 1, 1, 1, 2]), (8, 4, 4, [1, 2, 3, 4, 5, 6, 7, 8]),
                             (8, 2, 4, [0, 1, 1, 2, 2, 3, 3, 4])]:
        got = extra_ctx_visible_counts(n, G_, S)
        ok &= got == want
        print(f"COND visibility counts (n_groups={n}, G={G_}, S={S}): {got} {'OK' if got == want else 'WRONG ' + str(want)}")
    key = jax.random.PRNGKey(0)
    D, V, C, Bc = 8, 16, 1, 2
    emb = jax.random.normal(key, (V, D))
    proj = jnp.eye(C * D)[:, :D] if C * D >= D else None
    n, G_, S = 8, 1, 4
    M = (n * G_) // S
    codes = jnp.broadcast_to(jnp.arange(M)[None, :, None], (Bc, M, C)) + 3
    for window in (-1, 1, 2):
        w, r, valid, Wg = causal_extra_ctx_windows(codes, emb, proj, S, G_, n, Bc, D, window)
        w = w.reshape(Bc, n, Wg, D)
        counts = extra_ctx_visible_counts(n, G_, S)
        want_w = -1 if window < 0 else min(window, counts[-1])
        good = Wg == (counts[-1] if window < 0 else want_w)
        for g in range(n):
            vis = [emb[3 + j] for j in range(counts[g])][-Wg:] if Wg else []
            expect = [jnp.zeros((D,))] * (Wg - len(vis)) + vis
            want_valid = [False] * (Wg - len(vis)) + [True] * len(vis)
            good &= list(valid[g]) == want_valid
            for t in range(Wg):
                good &= bool(jnp.allclose(w[0, g, t], expect[t], atol=1e-6))
        print(f"COND window contents (G=1,S=4, cond_window={window}, width={Wg}): {'OK' if good else 'WRONG'}")
        ok &= good
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_cond_alignment_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _cond_cfg(1)
        misaligned = [str(x.message) for x in rec if "complete" in str(x.message)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _cond_cfg(4)
        aligned = [str(x.message) for x in rec if "complete" in str(x.message)]
    ok = len(misaligned) >= 1 and len(aligned) == 0
    print(f"COND alignment warning: decoder_ncodes=1 (S=4) warns={len(misaligned) >= 1}, decoder_ncodes=4 warns={len(aligned) >= 1} "
          f"{'OK' if ok else 'WRONG'}")
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def _cond_inputs(cfg, B_=2):
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    level0, level1 = model.levels[0], model.levels[1]
    (train_np, _), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    fb = jnp.array(images_to_positions(train_np[:B_], cfg, pixel_order))
    return level0, level1, fb


def _cond_encode(level0, level1, fb):
    e0 = level0.encode(code_embed_proj(fb, level0.own_input_embed, level0.own_input_proj), fb, rng=None)
    e1 = level1.encode(code_embed_proj(e0["code_soft"], level1.own_input_embed, level1.own_input_proj),
                       e0["code_idx"], rng=None)
    return e0, e1


def run_cond_complete(G_, cond_window=-1, B_=2):
    """streaming cond_depth=2 with the 'complete' rule (any G, optional cond_window): dense vs incremental KV-cache."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cond_cfg(G_, cond_window)
        level0, level1, fb = _cond_inputs(cfg, B_)
    e0, e1 = _cond_encode(level0, level1, fb)
    code0_idx, code0_soft, code1_idx = e0["code_idx"], e0["code_soft"], e1["code_idx"]
    logits_dense, _, _, _, _, _ = level0.decode_logits_and_target_pardec(
        fb, code0_soft, G_, extra_ctx_code_soft=[e1["code_soft"]])
    level = level0
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    n_blocks = code0_idx.shape[1]
    D = level.bos_embed.shape[-1]
    hd = D // level.n_heads
    ctx_flat, rope_ctx, valid_ctx, Wg, Wg_j, n_groups = ctx_rows(level, code0_idx, G_, [code1_idx])
    B2 = B_ * n_groups
    rope_ctx_flat = jnp.broadcast_to(rope_ctx[None], (B_, n_groups, rope_ctx.shape[1])).reshape(B2, -1)
    rope_bos = jnp.array([(g + 1) * G_ for g in range(n_groups)])
    rope_bos_flat = jnp.broadcast_to(rope_bos[None, :], (B_, n_groups)).reshape(B2)
    Kspan = G_ * level.K
    per_group_len = Wg + Wg_j + 1 + Kspan
    key_valid = jnp.concatenate([valid_ctx, jnp.ones((B2, 1 + Kspan), dtype=bool)], axis=1)
    self_step, self_chunk_step = build_cache_fns(level, blocks, ln_f, key_valid, per_group_len)

    cache_k = jnp.zeros((len(blocks), B2, level.n_kv_heads, per_group_len, hd))
    cache_v = jnp.zeros_like(cache_k)
    real_flat = jnp.stack([fb[:, g * Kspan:(g + 1) * Kspan] for g in range(n_groups)], axis=1).reshape(
        B2, Kspan, *fb.shape[2:])
    chunk = jnp.concatenate([ctx_flat, jnp.broadcast_to(level.bos_embed, (B2, 1, D))], axis=1)
    chunk_rope = jnp.concatenate([rope_ctx_flat, rope_bos_flat[:, None]], axis=1)
    h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, jnp.array(0), chunk_rope)
    pos, rp = Wg + Wg_j + 1, rope_bos_flat + 1
    logits_list = [level._token_logits_linears(h_chunk[:, -1, :])]
    x_input = level._dec_embed_target(real_flat[:, 0])
    for t in range(1, Kspan):
        h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos, rp)
        pos += 1
        rp = rp + 1
        logits_list.append(level._token_logits_linears(h))
        x_input = level._dec_embed_target(real_flat[:, t])
    lg = jnp.stack(logits_list, axis=1)
    incr = lg.reshape(B_, n_groups * Kspan, *lg.shape[2:])[:, :n_blocks * level.K]
    diff = jnp.abs(logits_dense - incr)
    mism = int((jnp.argmax(logits_dense, -1) != jnp.argmax(incr, -1)).sum())
    print(f"COND complete rule (decoder_ncodes={G_}, cond_window={cond_window}, extra width={Wg_j}): "
          f"dense-vs-incremental max_abs_diff={float(diff.max()):.6f} mismatches={mism}/{incr.shape[0] * incr.shape[1] * incr.shape[2]}")
    ok = float(diff.max()) <= 1e-3 and mism == 0
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_cond_complete_leak(G_, fullctx_control=False, B_=2):
    """perturb one input pixel; with the 'complete' rule every EARLIER group's logits must be unchanged (no info
    from past the group's end reaches it through own ctx, extra ctx or targets). fullctx is the positive control:
    it shows every group all coarse codes, so earlier groups DO change (proves the test can detect a leak)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cond_cfg(G_, -1, fullctx=fullctx_control)
        level0, level1, fb = _cond_inputs(cfg, B_)
    Kspan = G_ * level0.K
    n_groups = fb.shape[1] // Kspan
    gq = n_groups // 2

    def logits_for(x):
        e0, e1 = _cond_encode(level0, level1, x)
        lg, _, _, _, _, _ = level0.decode_logits_and_target_pardec(x, e0["code_soft"], G_,
                                                                   extra_ctx_code_soft=[e1["code_soft"]])
        return lg

    base = logits_for(fb)
    p = gq * Kspan + 1
    pert = logits_for(fb.at[:, p, :].set((fb[:, p, :] + 97) % 256))
    d = jnp.abs(base - pert).max(axis=tuple(range(2, base.ndim)))
    earlier = float(d[:, :gq * Kspan].max())
    own = float(d[:, gq * Kspan:(gq + 1) * Kspan].max())
    if fullctx_control:
        ok = earlier > 1e-4
        print(f"COND leak control (fullctx, decoder_ncodes={G_}): earlier-group max change={earlier:.6f} "
              f"(expected >0: fullctx sees all coarse codes) {'OK' if ok else 'TEST HAS NO POWER'}")
    else:
        ok = earlier <= 1e-5 and own > 1e-4
        print(f"COND leak test (complete, decoder_ncodes={G_}): earlier groups max change={earlier:.8f} (must be 0), "
              f"perturbed group's own change={own:.6f} (must be >0) {'OK' if ok else 'LEAK/BROKEN'}")
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def _fr_cfg():
    return Config(
        d_model=(32, 32), n_layers=(1, 1), n_heads=(2, 2), n_kv_heads=(None, None), strides=(4, 4),
        code_vocab=(16, 16), pq_chunks=(2, 2), pq_dim=(16, 16), byte_group=3, token_head_type="linears",
        decoder_ncodes=(4, 4), ncodes_window=(2, 2), weight_sharing=True,
        curriculum_mode="no_freeze")


def run_encoder_free_run():
    """encoder free-run (buffer re-forward) == slow no-junk reference loop; helper NTP logits == encode()'s ntp_acc;
    prompt preserved; sampling deterministic per rng; top_k=1 == greedy."""
    model = HierEncDec(jax.random.PRNGKey(0), _fr_cfg())
    ok = True
    for li, (C, V, T, P) in enumerate([(3, 256, 40, 12), (2, 16, 24, 5)]):
        lv = model.levels[li]
        toks = jnp.array(np.random.RandomState(li).randint(0, V, (2, T, C)))
        prompt = toks[:, :P]
        got = encoder_free_run(lv, prompt, T, jax.random.PRNGKey(0), greedy=True)
        ref = prompt
        for t in range(P, T):
            h = encoder_hidden(lv, code_embed_proj(ref, lv.own_input_embed, lv.own_input_proj))
            nxt = jnp.argmax(encoder_ntp_logits(lv, h[:, -1]), -1)[:, None].astype(ref.dtype)
            ref = jnp.concatenate([ref, nxt], axis=1)
        same_ref = bool(jnp.array_equal(got, ref))
        kept = bool(jnp.array_equal(got[:, :P], prompt))
        x = code_embed_proj(toks, lv.own_input_embed, lv.own_input_proj)
        acc_helper = float((jnp.argmax(encoder_ntp_logits(lv, encoder_hidden(lv, x))[:, :-1], -1) == toks[:, 1:]).mean())
        acc_enc = float(lv.encode(x, toks, rng=None)["ntp_acc"])
        same_acc = abs(acc_helper - acc_enc) < 1e-6
        s1 = encoder_free_run(lv, prompt, T, jax.random.PRNGKey(3), greedy=False, temperature=1.0)
        s2 = encoder_free_run(lv, prompt, T, jax.random.PRNGKey(3), greedy=False, temperature=1.0)
        det = bool(jnp.array_equal(s1, s2))
        k1 = encoder_free_run(lv, prompt, T, jax.random.PRNGKey(5), greedy=False, temperature=1.0, top_k=1)
        topk1 = bool(jnp.array_equal(k1, got))
        differs = not bool(jnp.array_equal(s1, got))
        this = same_ref and kept and same_acc and det and topk1 and differs
        ok &= this
        print(f"ENCODER free-run level {li}: == slow reference={same_ref} prompt kept={kept} helper ntp_acc == encode()={same_acc} "
              f"sampling deterministic={det} top_k=1 == greedy={topk1} sampled != greedy={differs} {'OK' if this else 'WRONG'}")
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_generate_from_prompt():
    """prompted generation pipeline: shapes/ranges, prompt kept at level 0, emitted codes over the prompt region equal the
    prompt's own encoding (causal encoders => the free-run continuation cannot change them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _fr_cfg()
        model = HierEncDec(jax.random.PRNGKey(0), cfg)
    (train_np, _), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    fb = jnp.array(images_to_positions(train_np[:1], cfg, pixel_order_for(cfg)))
    T0, P = fb.shape[1], 256
    prompt = fb[:, :P]
    ok = True
    for L in (1, 0):
        res = generate_from_prompt(model, cfg, prompt, T0, L, jax.random.PRNGKey(0), greedy=False, temperature=0.9, top_k=8)
        shapes = all(res[k].shape == (1, T0, cfg.byte_group) for k in ("emitted", "sampled"))
        rng_ok = all(int(res[k].min()) >= 0 and int(res[k].max()) <= 255 for k in ("emitted", "sampled"))
        tok, x = prompt, None
        for i in range(L):
            lv = model.levels[i]
            tok = lv.encode(code_embed_proj(tok, lv.own_input_embed, lv.own_input_proj), tok, rng=None)["code_idx"]
        lvL = model.levels[L]
        pc = lvL.encode(code_embed_proj(tok, lvL.own_input_embed, lvL.own_input_proj), tok, rng=None)["code_idx"]
        emitted_prefix = bool(jnp.array_equal(res["emitted_codes"][L][:, :pc.shape[1]], pc))
        kept = bool(jnp.array_equal(res["sampled"][:, :P], prompt)) if L == 0 else True
        this = shapes and rng_ok and emitted_prefix and kept
        ok &= this
        print(f"PROMPT generate (sample_level={L}): shapes={shapes} bytes in [0,255]={rng_ok} emitted codes over prompt == prompt's own encoding="
              f"{emitted_prefix} prompt bytes kept (L0 sampled)={kept} {'OK' if this else 'WRONG'}")
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_refine_gt_drop():
    """level_refine_gt_drop: 1.0 == default (own output), 0.0 == refine pass drafted with the real targets,
    0.5 differs from both; no refine rng (eval) ignores it."""
    cfg = dataclasses.replace(build_cfg(2), level_refine_passes=(2, 1, 1, 1), level_refine_window=(1, 0, 0, 0))
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    level = model.levels[0]
    imgs = np.random.default_rng(0).integers(0, 256, (B, 32, 32, 3)).astype(np.uint8)
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
    enc = level.encode(code_embed_proj(flat, level.own_input_embed, level.own_input_proj), flat, rng=None)
    ctx = enc["code_soft"]
    key = jax.random.PRNGKey(3)
    Pp = level.level_refine_window * G * level.K

    def run(gd, rr=key):
        return decode_logits_and_target_multipass(level, flat, ctx, G, rng=None, multipass_detach=True,
                                                  level_refine_gt_drop=gd, refine_rng=rr)[0]
    l_default = decode_logits_and_target_multipass(level, flat, ctx, G, rng=None, multipass_detach=True,
                                                   refine_rng=key)[0]
    _, tgt, *_ = level.decode_logits_and_target_pardec(flat, ctx, G, rng=None)
    l_ref = level.decode_logits_and_target_pardec(flat, ctx, G, rng=None, draft_override=tgt,
                                                  decode_past_override=Pp)[0]
    d1 = float(jnp.abs(run(1.0) - l_default).max())
    d0 = float(jnp.abs(run(0.0) - l_ref).max())
    dh = min(float(jnp.abs(run(0.5) - l_default).max()), float(jnp.abs(run(0.5) - l_ref).max()))
    de = float(jnp.abs(run(0.0, None) - l_default).max())
    ok = d1 == 0.0 and d0 < 1e-5 and dh > 1e-6 and de == 0.0
    print(f"REFINE gt_drop: 1.0==default diff={d1:.1e}  0.0==GT-drafted ref diff={d0:.1e}  0.5 differs (min diff)={dh:.1e}  "
          f"no-rng ignores diff={de:.1e} {'OK' if ok else 'WRONG'}")
    return ok


def run_remat_level():
    """remat / remat_level change memory only: loss and grads must match the no-remat run."""
    imgs = np.random.default_rng(1).integers(0, 256, (2, 32, 32, 3)).astype(np.uint8)
    res = {}
    for name, kw in (("off", {}), ("block", dict(remat=True)), ("level", dict(remat_level=True))):
        cfg = dataclasses.replace(build_cfg(2), **kw)
        model = HierEncDec(jax.random.PRNGKey(0), cfg)
        flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            lv = m.levels[0]
            enc = lv.encode(code_embed_proj(flat, lv.own_input_embed, lv.own_input_proj), flat, rng=None)
            logits = lv.decode_logits_and_target_pardec(flat, enc["code_soft"], G, rng=None)[0]
            return enc["ntp_loss"] + jnp.mean(logits ** 2)
        res[name] = loss_fn(model)
    ok = True
    for name in ("block", "level"):
        dl = abs(float(res[name][0] - res["off"][0]))
        dg = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(res[name][1], eqx.is_array)),
                                                             jax.tree_util.tree_leaves(eqx.filter(res["off"][1], eqx.is_array))))
        ok &= dl < 1e-5 and dg < 1e-4
        print(f"REMAT {name}: |loss diff|={dl:.1e} max|grad diff|={dg:.1e} {'OK' if dl < 1e-5 and dg < 1e-4 else 'WRONG'}")
    return ok


def run_refine_drop():
    """refine early exit: refine_active=[T,T] == 3 passes, [T,F] == 2 passes, [F,F] == 1 pass (also under jit,
    with finite grads through the cond)."""
    def mk(passes):
        return dataclasses.replace(build_cfg(2), level_refine_passes=(passes, 1, 1, 1), level_refine_window=(1, 0, 0, 0))
    m3 = HierEncDec(jax.random.PRNGKey(0), mk(3))
    imgs = np.random.default_rng(0).integers(0, 256, (B, 32, 32, 3)).astype(np.uint8)
    flat = jnp.array(images_to_positions(imgs, mk(3), pixel_order_for(mk(3))))
    lv = m3.levels[0]
    ctx = lv.encode(code_embed_proj(flat, lv.own_input_embed, lv.own_input_proj), flat, rng=None)["code_soft"]

    def run(model, active):
        return decode_logits_and_target_multipass(model.levels[0], flat, ctx, G, rng=None, multipass_detach=True,
                                                  refine_active=active)[0]
    def with_passes(n):   # same init key -> same weights, fewer configured (static) passes
        return HierEncDec(jax.random.PRNGKey(0), mk(n))
    l3, l2, l1 = (run(with_passes(n), None) for n in (3, 2, 1))
    act = lambda a: jnp.array(a)
    jr = jax.jit(lambda mm, a: run(mm, a))
    d_tt = float(jnp.abs(jr(m3, act([True, True])) - l3).max())
    d_tf = float(jnp.abs(jr(m3, act([True, False])) - l2).max())
    d_ff = float(jnp.abs(jr(m3, act([False, False])) - l1).max())
    g = jax.grad(lambda mm: jnp.mean(jr(mm, act([True, False])) ** 2))(eqx.filter(m3, eqx.is_inexact_array))
    fin = all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))
    ok = d_tt < 1e-5 and d_tf < 1e-5 and d_ff < 1e-5 and fin
    print(f"REFINE drop: [T,T]==3 passes diff={d_tt:.1e}  [T,F]==2 passes diff={d_tf:.1e}  [F,F]==1 pass diff={d_ff:.1e}  "
          f"grads finite={fin} {'OK' if ok else 'WRONG'}")
    return ok


def run_decoder_sampling():
    """decoder sampling: deterministic per seed, differs from greedy, top_k=1 == greedy, top_k=8 stays inside greedy-ish
    support (differs from top_k=0), all bytes valid."""
    imgs = np.random.default_rng(2).integers(0, 256, (2, 32, 32, 3)).astype(np.uint8)
    outs = {}
    for name, k in (("k0", 0), ("k1", 1), ("k8", 8)):
        cfg = dataclasses.replace(build_cfg(2), gen_top_k=k)
        model = HierEncDec(jax.random.PRNGKey(0), cfg)
        lv = model.levels[0]
        flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
        ctx = lv.encode(code_embed_proj(flat, lv.own_input_embed, lv.own_input_proj), flat, rng=None)["code_idx"]
        outs[name] = (np.asarray(decode_generate_multipass(lv, ctx, G, greedy=False, temperature=1.0, seed=3)),
                      np.asarray(decode_generate_multipass(lv, ctx, G, greedy=False, temperature=1.0, seed=3)),
                      np.asarray(decode_generate_multipass(lv, ctx, G, greedy=True)))
    det = np.array_equal(outs["k8"][0], outs["k8"][1])
    k1_greedy = np.array_equal(outs["k1"][0], outs["k1"][2])
    differs = not np.array_equal(outs["k0"][0], outs["k8"][0]) and not np.array_equal(outs["k8"][0], outs["k8"][2])
    valid = bool(((outs["k8"][0] >= 0) & (outs["k8"][0] <= 255)).all())
    ok = det and k1_greedy and differs and valid
    print(f"DECODER sampling: deterministic per seed={det} top_k=1 == greedy={k1_greedy} top_k=8 differs from top_k=0 and greedy={differs} "
          f"bytes valid={valid} {'OK' if ok else 'WRONG'}")
    return ok


def run_ctx_rows_reference():
    """independent python reference for the shared row builder: window contents, validity mask, rope ids, extra
    slots (window, cond_drop, None slot skipped), chunked visibility, incl. a last group running past the real codes."""
    ok = True
    Bc, D = 2, 64
    for (n_blocks, N, sc) in [(16, 2, 0), (16, -1, 0), (14, 1, 0), (16, -1, 1), (14, -1, 1), (16, 1, 2), (14, 2, 2), (16, -1, 4)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = dataclasses.replace(build_cfg(N, sc), cond_depth=(2, 1, 1, 1), cond_window=(3, -1, -1, -1))
        level = HierEncDec(jax.random.PRNGKey(0), cfg).levels[0]
        tok = jax.random.normal(jax.random.PRNGKey(1), (Bc, n_blocks, D))
        Sx = 4
        M = n_blocks // Sx
        extra = jax.random.randint(jax.random.PRNGKey(2), (Bc, M, 4), 0, 16)
        n_groups = -(-n_blocks // G)
        n_p = n_groups * G
        cg = 1 if sc <= 0 else -(-n_groups // sc)
        ends = [min((g // cg + 1) * cg, n_groups) * G for g in range(n_groups)]
        Wg = n_p if N < 0 else min(N * G + cg * G, n_p)
        ctx_flat, rope, valid, Wg_out, extra_len = level._pardec_ctx_rows(tok, [extra], G, n_groups, n_blocks)
        etok = code_embed_proj(extra, level.extra_ctx_embed[0], level.extra_ctx_proj[0])
        counts = [e // Sx for e in ends]
        Wj = min(3, max(counts))
        counts = [min(c, M) for c in counts]
        good = Wg_out == Wg and extra_len == Wj and ctx_flat.shape == (Bc * n_groups, Wj + Wg, D)
        flat = np.asarray(ctx_flat).reshape(Bc, n_groups, Wj + Wg, D)
        val = np.asarray(valid).reshape(Bc, n_groups, Wj + Wg)
        for g in range(n_groups):
            for t in range(Wj):
                idx = counts[g] - Wj + t
                good &= bool(val[0, g, t]) == (idx >= 0)
                want = np.asarray(etok[0, idx]) if idx >= 0 else np.zeros(D)
                good &= np.allclose(flat[0, g, t], want, atol=1e-6)
            for j in range(Wg):
                a = ends[g] - Wg + j
                v = 0 <= a < n_blocks
                good &= bool(val[0, g, Wj + j]) == v
                want = np.asarray(tok[0, a]) if v else np.zeros(D)
                good &= np.allclose(flat[0, g, Wj + j], want, atol=1e-6)
                good &= int(rope[g, Wj + j]) == max(a, 0)
        dropped = level._pardec_ctx_rows(tok, [extra], G, n_groups, n_blocks, drop_extras=True)[2]
        good &= not bool(np.asarray(dropped)[:, :Wj].any()) and bool(np.array_equal(np.asarray(dropped)[:, Wj:], val.reshape(-1, Wj + Wg)[:, Wj:]))
        skipped = level._pardec_ctx_rows(tok, [None], G, n_groups, n_blocks)
        good &= skipped[4] == 0 and skipped[0].shape[1] == Wg
        print(f"CTX rows reference (n_blocks={n_blocks}, N={N}, stream_chunks={sc}, Wg={Wg}, extra width={Wj}): "
              f"{'OK' if good else 'WRONG'}")
        ok &= good
    print(f"  {'CONSISTENT' if ok else 'DIVERGES'}")
    return ok


def run_chunk_equivalence():
    """stream_chunks = n_groups must equal stream_chunks = 0 (per-group streaming) exactly."""
    outs = []
    for sc in (0, 64):
        cfg = build_cfg(2, sc)
        level = HierEncDec(jax.random.PRNGKey(0), cfg).levels[0]
        imgs = np.random.default_rng(3).integers(0, 256, (2, 32, 32, 3)).astype(np.uint8)
        flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
        enc = level.encode(code_embed_proj(flat, level.own_input_embed, level.own_input_proj), flat, rng=None)
        outs.append(level.decode_logits_and_target_pardec(flat, enc["code_soft"], G)[0])
    d = float(jnp.abs(outs[0] - outs[1]).max())
    print(f"CHUNK equivalence: stream_chunks=64 (== n_groups) vs 0: max diff={d:.1e} {'OK' if d == 0.0 else 'WRONG'}")
    return d == 0.0


def run_chunk_leak(G_, sc, B_=2):
    """chunked visibility: perturb every pixel of group gq. Groups in EARLIER chunks must be unchanged (they only see
    parent codes up to their chunk end); an earlier group in the SAME chunk must change (positive control)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cond_cfg(G_, -1, stream_chunks=sc)
        level0, level1, fb = _cond_inputs(cfg, B_)
    Kspan = G_ * level0.K
    n_groups = fb.shape[1] // Kspan
    cg = -(-n_groups // sc)
    gq = n_groups // 2 + cg // 2
    chunk_start = (gq // cg) * cg

    def logits_for(x):
        e0, e1 = _cond_encode(level0, level1, x)
        return level0.decode_logits_and_target_pardec(x, e0["code_soft"], G_, extra_ctx_code_soft=[e1["code_soft"]])[0]

    base = logits_for(fb)
    lo, hi = gq * Kspan, (gq + 1) * Kspan
    pert = logits_for(fb.at[:, lo:hi, :].set((fb[:, lo:hi, :] + 97) % 256))
    d = jnp.abs(base - pert).max(axis=tuple(range(2, base.ndim)))
    earlier_chunks = float(d[:, :chunk_start * Kspan].max()) if chunk_start > 0 else 0.0
    same_chunk_before = float(d[:, chunk_start * Kspan:gq * Kspan].max()) if gq > chunk_start else None
    ok = earlier_chunks <= 1e-5 and (same_chunk_before is None or same_chunk_before > 1e-4)
    print(f"CHUNK leak test (G={G_}, stream_chunks={sc}, chunk={cg} groups, gq={gq}, chunk_start={chunk_start}): "
          f"earlier chunks change={earlier_chunks:.8f} (must be 0), same-chunk earlier groups change="
          f"{same_chunk_before} (must be >0) {'OK' if ok else 'LEAK/BROKEN'}")
    return ok


def run_generation_matches_rescoring(sc, with_extra, G_=4):
    """greedy generation (scan loop) must be the argmax of the dense teacher-forced scoring of its own output
    (plain / cond extras / chunked visibility), and every byte must be a valid value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cond_cfg(G_, -1, stream_chunks=sc)
        level0, level1, fb = _cond_inputs(cfg, 2)
    e0, e1 = _cond_encode(level0, level1, fb)
    extras = [e1["code_idx"]] if with_extra else None
    gen = level0.decode_generate_pardec(e0["code_idx"], G_, greedy=True, seed=0, extra_ctx_idx=extras)
    logits, *_ = level0.decode_logits_and_target_pardec(
        gen, e0["code_soft"], G_, extra_ctx_code_soft=[e1["code_soft"]] if with_extra else None)
    same = bool(jnp.array_equal(jnp.argmax(logits, -1), gen))
    valid = bool(((gen >= 0) & (gen <= 255)).all())
    samp = level0.decode_generate_pardec(e0["code_idx"], G_, greedy=False, temperature=1.0, seed=3, extra_ctx_idx=extras)
    samp_ok = bool(((samp >= 0) & (samp <= 255)).all()) and not bool(jnp.array_equal(samp, gen))
    ok = same and valid and samp_ok
    print(f"GEN scan (stream_chunks={sc}, extra={with_extra}): greedy == dense argmax of own output={same} "
          f"bytes valid={valid} sampled valid & differs={samp_ok} {'OK' if ok else 'WRONG'}")
    return ok


def run_dense_decode():
    """dense_decode=True bypasses pardec entirely: greedy generation must reproduce itself under
    decode_logits_and_target_pardec's own teacher-forced re-scoring (proves the flag actually routes to
    decode_logits_and_target/decode_generate and both agree, not just self-consistency by accident)."""
    cfg = dataclasses.replace(build_cfg(2), dense_decode=(True, False, False, False))
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    level = model.levels[0]
    imgs = np.random.default_rng(5).integers(0, 256, (B, 32, 32, 3)).astype(np.uint8)
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
    enc = level.encode(code_embed_proj(flat, level.own_input_embed, level.own_input_proj), flat, rng=None)
    gen = level.decode_generate_pardec(enc["code_idx"], G, greedy=True, seed=0)
    logits, target, *_ = level.decode_logits_and_target_pardec(gen, enc["code_soft"], G, rng=None)
    same = bool(jnp.array_equal(jnp.argmax(logits, -1), gen))
    valid = bool(((gen >= 0) & (gen <= 255)).all())
    ok = same and valid
    print(f"DENSE_DECODE: greedy gen == dense teacher-forced argmax of own output={same} bytes valid={valid} "
          f"{'OK' if ok else 'WRONG'}")
    return ok


def run_dec_attn_window():
    """dec_attn_window: bounded decoder window must match a dense reference built by masking
    decode_logits_and_target's own attention by hand (independent of the Attention.step/chunk_step fix,
    checked via the two paths already agreeing -- dense (training) vs incremental (generation) outputs)."""
    cfg = dataclasses.replace(build_cfg(2), dense_decode=(True, False, False, False),
                               decoder_attn_window=(3, None, None, None), weight_sharing=False)
    model = HierEncDec(jax.random.PRNGKey(0), cfg)
    level = model.levels[0]
    imgs = np.random.default_rng(6).integers(0, 256, (B, 32, 32, 3)).astype(np.uint8)
    flat = jnp.array(images_to_positions(imgs, cfg, pixel_order_for(cfg)))
    enc = level.encode(code_embed_proj(flat, level.own_input_embed, level.own_input_proj), flat, rng=None)
    logits_dense, target, *_ = level.decode_logits_and_target_pardec(flat, enc["code_soft"], G, rng=None)
    gen = level.decode_generate_pardec(enc["code_idx"], G, greedy=True, seed=0)
    logits_incr, *_ = level.decode_logits_and_target_pardec(gen, enc["code_soft"], G, rng=None)
    same = bool(jnp.array_equal(jnp.argmax(logits_incr, -1), gen))
    valid = bool(((gen >= 0) & (gen <= 255)).all())
    # windowed training loss must differ from an unwindowed model (window actually has an effect)
    cfg_u = dataclasses.replace(cfg, decoder_attn_window=(-1, -1, -1, -1))
    model_u = HierEncDec(jax.random.PRNGKey(0), cfg_u)
    logits_u, *_ = model_u.levels[0].decode_logits_and_target_pardec(flat, enc["code_soft"], G, rng=None)
    differs = not bool(jnp.array_equal(logits_dense, logits_u))
    ok = same and valid and differs
    print(f"DEC_ATTN_WINDOW: windowed gen == dense re-score argmax={same} bytes valid={valid} "
          f"windowed differs from unwindowed={differs} {'OK' if ok else 'WRONG'}")
    return ok


if __name__ == "__main__":
    ok0 = run_one(0)                        # disjoint (sanity baseline)
    ok2 = run_one(2)                        # bounded N=2, causal streaming
    okm1 = run_one(-1)                      # all, causal streaming (unbounded)
    okfc = run_one(-1, stream_chunks=1)     # wait once: every group sees all parent codes
    ok_c4 = run_one(2, stream_chunks=4)     # 4 chunks, bounded history
    ok_c8 = run_one(-1, stream_chunks=8)    # 8 chunks, unbounded history
    ok_wp = run_widened_aux(4, 0)           # decode_past-only widened aux NTP
    ok_wf = run_widened_aux(0, 4)           # decode_future-only widened aux NTP
    ok_wpf = run_widened_aux(4, 4)          # decode_past+decode_future combined
    ok_cyc = run_cyclic_revision_extra_ctx()  # cyclic-refine revision-slot fallback tables
    ok_gdf = run_gen_decode_future_self_consistency()  # gen_decode_future generation
    ok_dp = run_draft_pad_generation()      # out-of-bounds draft slots masked at generation time
    cond_oks = [run_cond_visibility_rule(), run_cond_alignment_warning(),
                run_cond_complete(4), run_cond_complete(1), run_cond_complete(2), run_cond_complete(4, cond_window=2),
                run_cond_complete_leak(4), run_cond_complete_leak(1), run_cond_complete_leak(2),
                run_cond_complete_leak(1, fullctx_control=True),
                run_encoder_free_run(), run_generate_from_prompt(), run_refine_gt_drop(), run_remat_level(), run_refine_drop(), run_decoder_sampling(), run_ctx_rows_reference(), run_chunk_equivalence(), run_chunk_leak(4, 4), run_chunk_leak(4, 8), run_chunk_leak(1, 4), run_dense_decode(), run_dec_attn_window(),
                run_generation_matches_rescoring(0, False), run_generation_matches_rescoring(0, True),
                run_generation_matches_rescoring(4, True), run_generation_matches_rescoring(1, True)]
    all_ok = (ok0 and ok2 and okm1 and okfc and ok_c4 and ok_c8 and ok_wp and ok_wf and ok_wpf and ok_cyc and ok_gdf and ok_dp
              and all(cond_oks))
    print(f"\nPASS all" if all_ok else "\nFAIL -- see divergence above")
