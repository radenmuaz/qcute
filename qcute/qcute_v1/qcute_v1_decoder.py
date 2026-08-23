import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qcute.qcute_v1.qcute_v1_common import (
    LM, Config, apply_rope, apply_track_dropout, chunked_windowed_attention, make_dict, pack_words,
    rope_cos_sin_for_positions, self_code_active, warn_degenerate_self_code, warn_thin_window,
)

_SELF_CONST = object()  # sentinel marking a track as "use bb.self_code_const", not a real code tensor


def merged_layout(bb: LM, L: int, tracks_meta: tuple, device: torch.device) -> dict:
    key = (L, tracks_meta, str(device))
    cached = bb.merged_cache.get(key)
    if cached is not None:
        return cached
    T = len(tracks_meta)
    byte_true_pos = torch.arange(L, device=device)
    byte_category = torch.zeros(L, dtype=torch.long, device=device)

    code_true_pos_parts, code_category_parts, code_window_parts, n_blocks_list = [], [], [], []
    for j, (K, window) in enumerate(tracks_meta):
        n_blocks = L // K
        n_blocks_list += [n_blocks]
        tp = (torch.arange(n_blocks, device=device) + 1) * K - 1
        code_true_pos_parts += [tp]
        code_category_parts += [torch.full((n_blocks,), j + 1, dtype=torch.long, device=device)]
        wv = float(window) if window is not None else float(L)
        code_window_parts += [torch.full((n_blocks,), wv, device=device)]
    code_true_pos = torch.cat(code_true_pos_parts) if T > 0 else torch.empty(0, dtype=torch.long, device=device)
    code_category = torch.cat(code_category_parts) if T > 0 else torch.empty(0, dtype=torch.long, device=device)
    code_window = torch.cat(code_window_parts) if T > 0 else torch.empty(0, device=device)

    total_true_pos = torch.cat([byte_true_pos, code_true_pos])
    total_category = torch.cat([byte_category, code_category])
    sort_key = (total_true_pos + 1) * (T + 2) + total_category
    perm = torch.argsort(sort_key, stable=True)

    w0 = tracks_meta[0][1] if tracks_meta[0][1] is not None else L
    byte_window = torch.full((L,), float(w0), device=device)
    total_window = torch.cat([byte_window, code_window])

    true_pos_sorted = total_true_pos[perm]
    window_of_slot = total_window[perm]
    category_sorted = total_category[perm]
    K0 = tracks_meta[0][0] if T > 0 else None
    Le = L + code_true_pos.shape[0]
    extract_pos = torch.searchsorted(true_pos_sorted, torch.arange(L, device=device), right=True) - 1
    struct = dict(perm=perm, extract_pos=extract_pos, true_pos_sorted=true_pos_sorted,
                  window_of_slot=window_of_slot, category_sorted=category_sorted, K0=K0,
                  Le=Le, n_blocks_list=n_blocks_list)

    finite_windows = [w for _, w in tracks_meta if w is not None]
    if finite_windows:
        sc = max(1, min(min(finite_windows), Le))
        n_chunks = -(-Le // sc)
        Lp = n_chunks * sc
        pad_len = Lp - Le
        W_max = max((w if w is not None else Le) for _, w in tracks_meta)
        density = len(tracks_meta) + 1
        n_prev_chunks = min(max(1, -(-(W_max * density) // sc)), max(0, n_chunks - 1))

        true_pos_p, window_p, cat_p = true_pos_sorted, window_of_slot, category_sorted
        if pad_len > 0:
            true_pos_p = F.pad(true_pos_p, (0, pad_len), value=-10 ** 9)
            window_p = F.pad(window_p, (0, pad_len), value=0.0)
            cat_p = F.pad(cat_p, (0, pad_len), value=0)
        pos_b = true_pos_p.view(n_chunks, sc)
        win_b = window_p.view(n_chunks, sc)
        cat_b = cat_p.view(n_chunks, sc)
        pad_pos = torch.full((n_prev_chunks, sc), -10 ** 9, device=device, dtype=pos_b.dtype)
        pad_win = torch.zeros((n_prev_chunks, sc), device=device, dtype=win_b.dtype)
        pad_cat = torch.zeros((n_prev_chunks, sc), device=device, dtype=cat_b.dtype)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        win_ext = torch.cat([pad_win, win_b], dim=0)
        cat_ext = torch.cat([pad_cat, cat_b], dim=0)
        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
        Kc = (n_prev_chunks + 1) * sc
        pos_win = pos_ext[idx].reshape(n_chunks, Kc)
        win_win = win_ext[idx].reshape(n_chunks, Kc)
        cat_win = cat_ext[idx].reshape(n_chunks, Kc)

        ti = pos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        local_row = torch.arange(sc, device=device).view(1, sc, 1)
        local_col = torch.arange(Kc, device=device).view(1, 1, Kc) - n_prev_chunks * sc
        causal = local_col <= local_row
        windowed = (ti - tj) < win_win.unsqueeze(1)
        is_byte_key = (cat_win == 0).unsqueeze(1)
        same_block = (ti // K0) == (tj // K0)
        byte_ok = (~is_byte_key) | same_block
        allow = causal & windowed & byte_ok
        struct.update(sc=sc, n_chunks=n_chunks, Lp=Lp, pad_len=pad_len,
                      n_prev_chunks=n_prev_chunks, idx=idx, Kc=Kc,
                      chunk_mask=allow.view(1, n_chunks, 1, sc, Kc))
    bb.merged_cache[key] = struct
    return struct


def merged_decode_forward(bb: LM, x0: torch.Tensor, tracks: list, extra_query: bool = False) -> tuple:
    cfg = bb.cfg
    B, L, D = x0.shape
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = x0.device
    tracks_meta = tuple((K, window) for _, K, window in tracks)
    struct = merged_layout(bb, L, tracks_meta, device)
    perm, extract_pos, Le = struct["perm"], struct["extract_pos"], struct["Le"]

    code_parts = []
    for j, (code_kv, K, _window) in enumerate(tracks):
        n_blocks = struct["n_blocks_list"][j]
        code_parts += [code_kv[:, :n_blocks, :]]
    all_code = torch.cat(code_parts, dim=1) if code_parts else x0.new_zeros(B, 0, D)
    unordered = torch.cat([x0, all_code], dim=1)
    combined = unordered[:, perm, :]

    finite_windows = [w for _, w in tracks_meta if w is not None]
    use_chunked = bool(finite_windows) and "sc" in struct and Le > struct["sc"]

    if not use_chunked:
        cos, sin = rope_cos_sin_for_positions(struct["true_pos_sorted"].clamp(min=0), hd, cfg.rope_base, device)
        T = len(tracks_meta)
        fully_causal = not finite_windows and T == 0
        attn_mask = None
        if not fully_causal:
            ti = struct["true_pos_sorted"].unsqueeze(1)
            tj = struct["true_pos_sorted"].unsqueeze(0)
            buf_i = torch.arange(Le, device=device).unsqueeze(1)
            buf_j = torch.arange(Le, device=device).unsqueeze(0)
            causal = buf_j <= buf_i
            allow = causal & ((ti - tj) < struct["window_of_slot"].unsqueeze(0)) if finite_windows else causal
            if T > 0:
                K0 = struct["K0"]
                is_byte_key = (struct["category_sorted"] == 0).unsqueeze(0)
                same_block = (ti // K0) == (tj // K0)
                allow = allow & ((~is_byte_key) | same_block)
            attn_mask = allow.view(1, 1, Le, Le)

        xe = combined
        for block in bb.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = (F.scaled_dot_product_attention(q, k, v, is_causal=True) if fully_causal
                 else F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask))
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))
        he = bb.ln_f(xe)
    else:
        sc, n_chunks, Lp, pad_len = struct["sc"], struct["n_chunks"], struct["Lp"], struct["pad_len"]
        n_prev_chunks, idx, Kc = struct["n_prev_chunks"], struct["idx"], struct["Kc"]
        warn_thin_window(tracks, sc)
        xe = combined
        true_pos_p = struct["true_pos_sorted"]
        if pad_len > 0:
            xe = F.pad(xe, (0, 0, 0, pad_len))
            true_pos_p = F.pad(true_pos_p, (0, pad_len), value=0)
        cos, sin = rope_cos_sin_for_positions(true_pos_p.clamp(min=0), hd, cfg.rope_base, device)
        for block in bb.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Lp, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

            qb = q.view(B, H, n_chunks, sc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, sc, hd)
            kb_flat = k.view(B, H, n_chunks, sc, hd)
            vb_flat = v.view(B, H, n_chunks, sc, hd)
            pad_k = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=k.dtype)
            pad_v = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=v.dtype)
            k_ext = torch.cat([pad_k, kb_flat], dim=2)
            v_ext = torch.cat([pad_v, vb_flat], dim=2)
            k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)
            v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)

            mask_batched = struct["chunk_mask"].expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
            yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
            y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))
        he = bb.ln_f(xe)[:, :Le, :]

    h_out = he[:, extract_pos, :]
    query_last = he[:, -1, :] if extra_query else None
    return h_out, query_last


def cross_attn_stage(bb: LM, x_in: torch.Tensor, code_kv: torch.Tensor, seq_repr: torch.Tensor,
                      level: int, track_K: int, window: int | None, compute_ntp: bool, want_code: bool) -> dict:
    cfg = bb.cfg
    K = cfg.Ks[level]
    D = bb.d_model
    is_byte_level = level == 0
    B, L, _ = x_in.shape
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = x_in.device

    n_blocks = code_kv.shape[1]
    code_pos = (torch.arange(n_blocks, device=device) + 1) * track_K - 1
    query_pos = torch.arange(L, device=device)

    use_chunked = window is not None and L % track_K == 0 and (L // track_K) == n_blocks and L > window

    if not use_chunked:
        cos_q, sin_q = rope_cos_sin_for_positions(query_pos, hd, cfg.rope_base, device)
        cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)
        causal = code_pos.view(1, -1) <= query_pos.view(-1, 1)
        allow = (causal & ((query_pos.view(-1, 1) - code_pos.view(1, -1)) < window)) if window is not None else causal
        attn_mask = allow.view(1, 1, L, n_blocks)

        x = x_in
        for block in bb.blocks:
            x = block.forward_cross(x, code_kv, cos_q, sin_q, cos_k, sin_k, attn_mask)
        h = bb.ln_f(x)
    else:
        warn_thin_window([(code_kv, track_K, window)], window)
        codes_per_chunk = max(1, window // track_K)
        qbucket = codes_per_chunk * track_K
        n_chunks = -(-L // qbucket)
        pad_len = n_chunks * qbucket - L
        Lp = n_chunks * qbucket
        n_prev_chunks = min(max(1, -(-window // qbucket) + 1), n_chunks)

        query_pos_p = query_pos if pad_len == 0 else F.pad(query_pos, (0, pad_len), value=-10 ** 9)
        cos_q, sin_q = rope_cos_sin_for_positions(query_pos_p.clamp(min=0), hd, cfg.rope_base, device)

        n_code_slots = n_chunks * codes_per_chunk
        code_pad_len = n_code_slots - n_blocks
        code_pos_p = code_pos if code_pad_len <= 0 else F.pad(code_pos, (0, code_pad_len), value=-10 ** 9)
        code_pos_p = code_pos_p[:n_code_slots]
        cos_k, sin_k = rope_cos_sin_for_positions(code_pos_p.clamp(min=0), hd, cfg.rope_base, device)

        pos_b = code_pos_p.view(n_chunks, codes_per_chunk)
        pad_pos = torch.full((n_prev_chunks - 1, codes_per_chunk), -10 ** 9, device=device, dtype=pos_b.dtype)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks, device=device).view(1, n_prev_chunks))
        Kc = n_prev_chunks * codes_per_chunk
        pos_win = pos_ext[idx].reshape(n_chunks, Kc)

        qpos_b = query_pos_p.view(n_chunks, qbucket)
        ti = qpos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        allow = (tj <= ti) & (ti - tj < window)
        mask_flat = allow.view(1, n_chunks, 1, qbucket, Kc).expand(
            B, n_chunks, 1, qbucket, Kc).reshape(B * n_chunks, 1, qbucket, Kc)

        x = x_in if pad_len == 0 else F.pad(x_in, (0, 0, 0, pad_len))
        coden_pad_len = n_code_slots - n_blocks
        for block in bb.blocks:
            xn = block.ln1(x)
            coden = block.ln1(code_kv)
            if coden_pad_len > 0:
                coden = F.pad(coden, (0, 0, 0, coden_pad_len))
            Wq = block.attn.qkv.weight[:D]
            Wk = block.attn.qkv.weight[D:2 * D]
            Wv = block.attn.qkv.weight[2 * D:3 * D]

            q = F.linear(xn, Wq).view(B, Lp, H, hd).transpose(1, 2)
            k = F.linear(coden, Wk).view(B, n_code_slots, H, hd).transpose(1, 2)
            v = F.linear(coden, Wv).view(B, n_code_slots, H, hd).transpose(1, 2)
            q = apply_rope(q, cos_q, sin_q)
            k = apply_rope(k, cos_k, sin_k)

            k_b = k.view(B, H, n_chunks, codes_per_chunk, hd)
            v_b = v.view(B, H, n_chunks, codes_per_chunk, hd)
            pad_k = torch.zeros(B, H, n_prev_chunks - 1, codes_per_chunk, hd, device=device, dtype=k.dtype)
            pad_v = torch.zeros(B, H, n_prev_chunks - 1, codes_per_chunk, hd, device=device, dtype=v.dtype)
            k_ext = torch.cat([pad_k, k_b], dim=2)
            v_ext = torch.cat([pad_v, v_b], dim=2)
            k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd)
            v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd)

            qb = q.view(B, H, n_chunks, qbucket, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, qbucket, hd)
            kb = k_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)
            vb = v_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)

            yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_flat)
            y = yb.view(B, n_chunks, H, qbucket, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
            x = x + a
            x = x + block.mlp(block.ln2(x))
        h = bb.ln_f(x)[:, :L, :]

    query_seq = h[:, :-1, :]
    if compute_ntp:
        ntp_loss, ntp_acc = bb.ntp_loss_acc(query_seq[:, K - 1:, :].reshape(-1, D), seq_repr[:, K:], is_byte_level)
    else:
        ntp_loss, ntp_acc = h.new_zeros(()), h.new_zeros(())

    code = None
    if want_code:
        x0 = bb.embed_input(seq_repr, is_byte_level)
        code = bb.extract_code(h, x0, K, window)["code"]

    query_last = h[:, -1, :]
    return make_dict(hidden=h, query_last=query_last, loss=ntp_loss, acc=ntp_acc, code=code)


def bos_interleaved_self_attn(bb: LM, x0: torch.Tensor, K: int, window: int | None,
                               strip_bos: bool = True) -> torch.Tensor:
    """qcute_v1 non-top-level decode self-attention (see docs/qcute_v1_plan.md): prepends
    bb.self_code_const (a per-block SEED TOKEN -- not "BOS", a single sequence-start marker, and
    not a "sink" either, a passive fallback key others attend to but that itself predicts nothing;
    it recurs every block AND is a full token, going through self-attn/MLP/cross-attn like any real
    byte and genuinely predicting its block's own first byte, see own_block_cross_attn_decode below
    and chat 2026-08-20 for the naming corrections) before every K-block -- [seed,x0,x1]
    [seed,x2,x3] ... -- then runs plain causal self-attention over the augmented sequence, RoPE'd by
    augmented-sequence position. window=None: sync (unbounded, one continuous causal chain across
    every block, today's default). window=K+1: async ablation (this block's seed+K only,
    independently schedulable per block, no cross-block visibility) -- any other finite value is
    interpreted directly in augmented-sequence units.

    strip_bos=True (default): returns h stripped back to the original K*n_blocks raw positions
    (seed-token positions dropped from the output, not from the attention computation) -- the seed
    then serves only as an extra key, never itself predicting anything (genuinely sink-like in this
    mode). strip_bos=False: returns the full augmented (n_blocks*(K+1)) sequence, seed-token
    positions included -- for own_block_cross_attn_decode below, where the seed's own hidden state
    genuinely predicts each block's own first byte using that block's own code (the mode
    decode_level actually uses as of 2026-08-20; strip_bos=True is kept for any caller that only
    wants the extra-key effect, none currently exist)."""
    cfg = bb.cfg
    B, L, D = x0.shape
    n_blocks = L // K
    x0 = x0[:, :n_blocks * K, :]
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = x0.device
    xb = x0.view(B, n_blocks, K, D)
    bos = bb.self_code_const.view(1, 1, 1, D).expand(B, n_blocks, 1, D)
    xe = torch.cat([bos, xb], dim=2).view(B, n_blocks * (K + 1), D)
    Le = n_blocks * (K + 1)
    pos = torch.arange(Le, device=device)
    cos, sin = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device)
    for block in bb.blocks:
        xn = block.ln1(xe)
        qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = chunked_windowed_attention(q, k, v, window)
        a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
        xe = xe + a
        xe = xe + block.mlp(block.ln2(xe))
    he = bb.ln_f(xe).view(B, n_blocks, K + 1, D)
    if not strip_bos:
        return he.reshape(B, Le, D)
    return he[:, :, 1:, :].reshape(B, n_blocks * K, D)


def own_block_cross_attn_decode(bb: LM, h_self_full: torch.Tensor, code_kv: torch.Tensor,
                                 K: int, window: int | None) -> torch.Tensor:
    """Cross-attention counterpart to bos_interleaved_self_attn(strip_bos=False): code_pos is each
    block's own seed-token position (b*(K+1) in the augmented sequence), not its last real byte --
    so code_b is visible to EVERY position in block b, including the seed token itself, letting the
    whole block genuinely reconstruct from its own code (seed_b -> block b's own first byte, using
    code_b -- see chat 2026-08-20: this is standard VQ-VAE/discrete-autoencoder teacher-forcing,
    reconstructing a block from its own real code during training is not a leak, since at
    generation time the SAME slot is filled by a level-above PREDICTION available the instant
    block b starts, before any of block b's own bytes exist -- causal at the point that matters,
    per docs/qcute_v1_plan.md's "causality is enforced by where the conditioning code comes from"
    rule. Returns h at the full augmented length (seed-token positions included, not stripped) --
    see own_block_decode_loss for how targets align to this."""
    cfg = bb.cfg
    B, Le, D = h_self_full.shape
    n_blocks = Le // (K + 1)
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = h_self_full.device

    code_pos = torch.arange(n_blocks, device=device) * (K + 1)
    query_pos = torch.arange(Le, device=device)
    causal = code_pos.view(1, -1) <= query_pos.view(-1, 1)
    allow = (causal & ((query_pos.view(-1, 1) - code_pos.view(1, -1)) < window)) if window is not None else causal
    attn_mask = allow.view(1, 1, Le, n_blocks)

    cos_q, sin_q = rope_cos_sin_for_positions(query_pos, hd, cfg.rope_base, device)
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)
    x = h_self_full
    for block in bb.blocks:
        x = block.forward_cross(x, code_kv, cos_q, sin_q, cos_k, sin_k, attn_mask)
    return bb.ln_f(x)


def own_block_decode_loss(bb: LM, h_full: torch.Tensor, x_real: torch.Tensor, K: int,
                           is_byte_level: bool) -> tuple:
    """Target alignment for own_block_cross_attn_decode's output. Each block contributes K queries
    (seed_b + its own first K-1 real bytes -- the block's LAST real byte is dropped, since its
    'next' augmented slot is the next block's seed token, not a real target) predicting, in order,
    that SAME block's own K real bytes UNSHIFTED -- so concatenated across blocks the target is
    exactly x_real itself, not x_real shifted by 1 (unlike the reverted mechanism's shift-by-1
    convention). This is deliberate: seed_b's query already stands in for 'the position before
    block b', so no additional shift is needed here."""
    B, Le, D = h_full.shape
    n_blocks = Le // (K + 1)
    h_q = h_full.view(B, n_blocks, K + 1, D)[:, :, :K, :].reshape(B, n_blocks * K, D)
    L = x_real.shape[1]
    h_q = h_q[:, :L, :]
    return bb.ntp_loss_acc(h_q.reshape(-1, D), x_real, is_byte_level)


def split_track0_window(track0_window) -> tuple:
    """track0_window (model.decode_windows[i][0]) is either a scalar/None (broadcasts to both
    self_window and own_code_window, the historical shared-knob behavior) or a
    (self_window, own_code_window) 2-tuple (2026-08-23, decouples track0's own byte/code
    self-attention window from its cross-attention window into level (i+1)'s code -- see
    qcute_v1.py's _norm_track0). Every call site that used to write `track0_window, track0_window`
    should use this instead, so training (_track0) and generation (_stack_generate_blockwise) never
    diverge on which window applies where."""
    if isinstance(track0_window, tuple):
        return track0_window
    return track0_window, track0_window


def encode_like_self_attn_decode(bb: LM, x0: torch.Tensor, code_kv: torch.Tensor, K: int,
                                  self_window: int | None, own_code_window: int | None) -> dict:
    """StackDecoder pass 1 (2026-08-20, "don't interleave seed tokens" design -- see
    docs/qcute_v1_plan.md and chat that day): plain causal self-attention over the real byte
    sequence, literally the encode pass (no seed token in the sequence at all, contrast
    bos_interleaved_self_attn above), with cross-attention to that byte's own-block code spliced in
    per layer between self-attn and MLP -- masked so a byte in block b sees code_b (or, if
    cfg.own_code_min_lag > 0, only strictly-earlier codes starting at lag min_lag -- 2026-08-23,
    the causal/exact retargeting of docs/maths.md's Part 8), within own_code_window codes of that
    lag. Also saves each layer's post-RoPE self-attention K/V (real bytes
    only, computed before this layer's cross-attn mutates the residual stream) for
    seed_query_decode's second pass to reuse directly -- these are exactly what an incremental
    KV-cache over the real byte stream would already hold, so the seed token never occupies a cache
    slot or shifts any downstream position, unlike bos_interleaved_self_attn's augmented sequence.

    Returns dict(hidden=h, saved_k=[...], saved_v=[...]), one (k, v) per layer, each (B, H, L, hd)."""
    cfg = bb.cfg
    B, L, D = x0.shape
    n_blocks = L // K
    x0 = x0[:, :n_blocks * K, :]
    L = n_blocks * K
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = x0.device

    pos = torch.arange(L, device=device)
    cos, sin = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device)
    code_pos = torch.arange(n_blocks, device=device) * K
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)

    block_lag = (pos.view(-1, 1) // K) - code_pos.view(1, -1) // K
    min_lag = cfg.own_code_min_lag
    win = own_code_window if own_code_window is not None else n_blocks
    cross_mask = ((block_lag >= min_lag) & (block_lag < min_lag + win)).view(1, 1, L, n_blocks)

    x = x0
    saved_k, saved_v = [], []
    for block in bb.blocks:
        xn = block.ln1(x)
        qkv = block.attn.qkv(xn).reshape(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = (chunked_windowed_attention(q, k, v, self_window) if self_window is not None
             else F.scaled_dot_product_attention(q, k, v, is_causal=True))
        a = block.attn.out(y.transpose(1, 2).reshape(B, L, D))
        x = x + a
        saved_k += [k]
        saved_v += [v]
        x = block.forward_cross(x, code_kv, cos, sin, cos_k, sin_k, cross_mask)
    he = bb.ln_f(x)
    return make_dict(hidden=he, saved_k=saved_k, saved_v=saved_v)


def seed_query_decode(bb: LM, saved_k: list, saved_v: list, code_kv: torch.Tensor, n_blocks: int,
                       K: int, self_window: int | None, own_code_window: int | None) -> torch.Tensor:
    """StackDecoder pass 2, counterpart to encode_like_self_attn_decode: this level's per-block
    trainable seed token (bb.self_code_const) used ONLY as a query -- never spliced into pass 1's
    sequence as a key, unlike bos_interleaved_self_attn -- cross-attending directly to pass 1's
    saved_k/saved_v (never recomputed). Masked causal at block granularity: seed_b sees only bytes
    strictly before block b starts (block b's own bytes don't exist yet at generation time -- same
    rule as own_block_cross_attn_decode's causality note), then cross-attends to block b's own code
    (or, if cfg.own_code_min_lag > 0, only strictly-earlier codes -- same knob as
    encode_like_self_attn_decode). Static shape (exactly n_blocks queries, deterministic from L, K)
    and KV-cache-able (pass 2 reads
    the existing cache, writes nothing new). Returns h_seed (B, n_blocks, D), predicting each
    block's own first byte."""
    cfg = bb.cfg
    D = bb.d_model
    B = code_kv.shape[0]
    device = code_kv.device
    H, hd = cfg.n_heads, D // cfg.n_heads
    L = n_blocks * K

    seed_pos = torch.arange(n_blocks, device=device) * K
    cos_q, sin_q = rope_cos_sin_for_positions(seed_pos, hd, cfg.rope_base, device)
    byte_pos = torch.arange(L, device=device)
    causal = byte_pos.view(1, -1) < seed_pos.view(-1, 1)
    if self_window is not None:
        causal = causal & ((seed_pos.view(-1, 1) - byte_pos.view(1, -1)) < self_window)
    self_mask = causal.view(1, 1, n_blocks, L)

    code_pos = torch.arange(n_blocks, device=device) * K
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)
    block_lag = torch.arange(n_blocks, device=device).view(-1, 1) - torch.arange(n_blocks, device=device).view(1, -1)
    min_lag = cfg.own_code_min_lag
    win = own_code_window if own_code_window is not None else n_blocks
    cross_mask = ((block_lag >= min_lag) & (block_lag < min_lag + win)).view(1, 1, n_blocks, n_blocks)

    x = bb.self_code_const.view(1, 1, D).expand(B, n_blocks, D)
    for i, block in enumerate(bb.blocks):
        xn = block.ln1(x)
        Wq = block.attn.qkv.weight[:D]
        q = apply_rope(F.linear(xn, Wq).view(B, n_blocks, H, hd).transpose(1, 2), cos_q, sin_q)
        k, v = saved_k[i], saved_v[i]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self_mask)
        a = block.attn.out(y.transpose(1, 2).reshape(B, n_blocks, D))
        x = x + a
        x = block.forward_cross(x, code_kv, cos_q, sin_q, cos_k, sin_k, cross_mask)
    return bb.ln_f(x)


def encode_like_step(bb: LM, x_embed: torch.Tensor, cache_k: list, cache_v: list, pos: int,
                      code_kv_b: torch.Tensor, code_pos_b: int) -> torch.Tensor:
    """Single-token incremental version of encode_like_self_attn_decode's pass1, for real
    KV-cache generation (StackDecoder.generate_kv_cache). Appends this byte's own self-attention
    K/V to cache_k/cache_v IN PLACE (one entry per layer, growing) -- exactly what an incremental
    cache over the real byte stream would already hold, per encode_like_self_attn_decode's own
    docstring -- and applies the SAME per-layer own-code cross-attention splice as the batched
    version, using code_kv_b (this byte's block's own code: real during prompt warm-up, predicted
    one block ahead during generation, see generate_kv_cache). Caller is responsible for calling
    this for EVERY byte position, including a block's own last byte, even when its own output is
    unused -- skipping it would leave that position missing from the cache for future queries."""
    cfg = bb.cfg
    B, _, D = x_embed.shape
    H, hd = cfg.n_heads, D // cfg.n_heads
    device = x_embed.device
    pos_t = torch.tensor([pos], device=device)
    cos, sin = rope_cos_sin_for_positions(pos_t, hd, cfg.rope_base, device)
    code_pos_t = torch.tensor([code_pos_b], device=device)
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos_t, hd, cfg.rope_base, device)

    x = x_embed
    for li, block in enumerate(bb.blocks):
        xn = block.ln1(x)
        qkv = block.attn.qkv(xn).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache_k[li] = k if cache_k[li] is None else torch.cat([cache_k[li], k], dim=2)
        cache_v[li] = v if cache_v[li] is None else torch.cat([cache_v[li], v], dim=2)
        y = F.scaled_dot_product_attention(q, cache_k[li], cache_v[li])
        a = block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
        x = x + a
        x = block.forward_cross(x, code_kv_b, cos, sin, cos_k, sin_k, attn_mask=None)
    return bb.ln_f(x)


def seed_step(bb: LM, cache_k: list, cache_v: list, seed_pos: int, code_kv_b: torch.Tensor,
              code_pos_b: int) -> torch.Tensor:
    """Single-token incremental version of seed_query_decode, for StackDecoder.generate_kv_cache:
    the per-block seed token as a pure query against the EXISTING cache built by encode_like_step --
    NEVER appends anything to it, matching the whole design (the seed token is a key nowhere,
    ever). The very first block's cache is empty (nothing precedes it, correctly causal) so its
    self-attention contribution is exactly zero rather than a masked special case."""
    cfg = bb.cfg
    D = bb.d_model
    device = code_kv_b.device
    B = code_kv_b.shape[0]
    H, hd = cfg.n_heads, D // cfg.n_heads

    pos_t = torch.tensor([seed_pos], device=device)
    cos_q, sin_q = rope_cos_sin_for_positions(pos_t, hd, cfg.rope_base, device)
    code_pos_t = torch.tensor([code_pos_b], device=device)
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos_t, hd, cfg.rope_base, device)

    x = bb.self_code_const.view(1, 1, D).expand(B, 1, D)
    for li, block in enumerate(bb.blocks):
        if cache_k[li] is None:
            a = x.new_zeros(B, 1, D)
        else:
            xn = block.ln1(x)
            Wq = block.attn.qkv.weight[:D]
            q = apply_rope(F.linear(xn, Wq).view(B, 1, H, hd).transpose(1, 2), cos_q, sin_q)
            y = F.scaled_dot_product_attention(q, cache_k[li], cache_v[li])
            a = block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
        x = x + a
        x = block.forward_cross(x, code_kv_b, cos_q, sin_q, cos_k, sin_k, attn_mask=None)
    return bb.ln_f(x)


def upper_track_step(bb: LM, x_last: torch.Tensor, pos: int, code_kv: torch.Tensor, track_K: int) -> torch.Tensor:
    """Single-step cross-attention-only stage for StackDecoder.generate_kv_cache's upper tracks
    (cross_attn_stage's counterpart for one query at one absolute position): code_kv already holds
    every code produced so far for this track, no cache needed since code sequences are O(L/track_K)
    -- far shorter than the byte stream encode_like_step/seed_step cache. Visibility mirrors
    cross_attn_stage's own (non-chunked) causal rule (code_pos <= pos), unbounded window only --
    matching generate_kv_cache's own documented scope."""
    cfg = bb.cfg
    D = bb.d_model
    device = x_last.device
    H, hd = cfg.n_heads, D // cfg.n_heads
    n_blocks = code_kv.shape[1]
    code_pos = (torch.arange(n_blocks, device=device) + 1) * track_K - 1
    visible = code_pos <= pos
    code_kv_vis = code_kv[:, visible, :]
    code_pos_vis = code_pos[visible]
    pos_t = torch.tensor([pos], device=device)
    cos_q, sin_q = rope_cos_sin_for_positions(pos_t, hd, cfg.rope_base, device)
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos_vis, hd, cfg.rope_base, device)
    x = x_last
    for block in bb.blocks:
        x = block.forward_cross(x, code_kv_vis, cos_q, sin_q, cos_k, sin_k, attn_mask=None)
    return bb.ln_f(x)


def code_context_pass(bb: LM, code_embeds: torch.Tensor) -> torch.Tensor:
    """Causal self-attention + MLP over a code-embedding sequence (own sequential index, not
    absolute byte position -- a separate RoPE application from whatever cross-attention consumes
    the result downstream), used by KVContextLM's "fresh"/"shared" kv_lm modes to contextualize
    each code position from earlier codes in the same track, before it becomes cross-attention K/V.
    Input stays purely the discrete code's own embedding -- never the producing encoder's raw
    pre-quantization hidden state -- so this doesn't reopen the discrete-bottleneck gap
    StackEncAttnDecoder would have (see chat 2026-08-22)."""
    cfg = bb.cfg
    hd = bb.d_model // cfg.n_heads
    n = code_embeds.shape[1]
    pos = torch.arange(n, device=code_embeds.device)
    cos, sin = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, code_embeds.device)
    x = code_embeds
    for block in bb.blocks:
        x = block(x, cos, sin, window=None)
    return bb.ln_f(x)


class KVContextLM(nn.Module):
    """kv_lm_mode="copy"/"shared": wraps an LM's transformer blocks so cross-attention K/V comes
    from a causally-contextualized code representation (code_context_pass) instead of an isolated
    per-position code embedding. Same calling convention as nn.Identity() (kv_lm_mode="identity",
    the pre-2026-08-23 default): embeds in, embeds out."""
    def __init__(self, bb: LM):
        super().__init__()
        self.bb = bb

    def forward(self, code_embeds: torch.Tensor) -> torch.Tensor:
        return code_context_pass(self.bb, code_embeds)


def bos_query_only_parallel_sync_decode(bb: LM, x0: torch.Tensor, K: int) -> torch.Tensor:
    """STUB, not implemented -- design note for a future parallel-decode generation strategy,
    distinct from both branches of bos_interleaved_self_attn above.

    Idea: instead of decoding block-by-block sequentially, decode ACROSS blocks in lockstep by
    within-block offset. At synchronized step t=0, every block's own per-block seed-token query
    (never a key here -- genuinely sink-like in this stub, unlike the query+key seed token
    own_block_cross_attn_decode above actually uses -- see the "double forward" mechanism below)
    attends to its own code + past codes and produces every block's first byte in parallel. At t=1,
    every block's first byte (now real, just produced) is used as a key so every block's second
    byte can be produced in parallel, one synchronized wave per within-block offset -- K waves total
    instead of K per-block sequential steps times n_blocks blocks. Cross-block ordering stays causal
    (block b's own code, and only past blocks' codes, are visible throughout, matching the lag
    already established as intended -- see docs/symmetric_hierarchy_generalizations.md #3's
    downward-lag discussion for the same causality constraint in a different context).

    Mechanism ("double forward", the trick this stub is named for): two attention calls per layer,
    not two sequential passes. (1) The real byte stream self-attends among itself only, via the
    fast is_causal=True path -- unaffected by the seed token. (2) Each block's seed-token query
    attends, via a rectangular (Q != KV shape) causal cross-attention, to that SAME layer's
    just-updated byte hidden states, using bb.self_code_const as its query and Block.forward_cross
    as the op (already used elsewhere for code cross-attention; here the byte stream stands in for
    the code_kv argument). The seed token never contributes a key/value itself in this stub, so it's
    causal, static-shape (n_blocks extra query rows, deterministic from L//K), and KV-cache
    compatible (byte cache still grows 1:1 with real tokens; the seed query is transient,
    cache-free, computed fresh per synchronized step). Cost: the byte pass stays O(L^2) at full
    speed; the seed-query pass adds O(L^2/K) on the slower custom-mask backend, sub-dominant for
    K>1.

    Not the sync default (bos_interleaved_self_attn's window=None branch) because that branch
    optimizes for the already-fully-sequential case (real content already provides left-context, no
    per-block seed needed) rather than genuine cross-block parallelism, which is what this stub is
    for. Not the async ablation either -- async is still block-by-block, just with a bounded window;
    this is block-*parallel*, synchronized on offset within block. See docs/qcute_v1_plan.md's
    "Generation feasibility" section for the surrounding staged-plan context this would extend."""
    raise NotImplementedError("parallel sync local AR decode -- design note only, see docstring")


def sample_next_byte(embed_weight: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h_last, embed_weight)
    return logits.argmax(-1)


def encode_up_to(model, seq_repr: torch.Tensor, level: int) -> torch.Tensor:
    for j in range(level):
        seq_repr = model.encoders[j](seq_repr, level=j, window=model.windows[j], compute_ntp=False)["code"]
    return seq_repr


class Decoder(nn.Module):
    def __init__(self, cfg: Config, n_levels: int):
        super().__init__()
        self.cfg = cfg
        self.n_levels = n_levels

    def decode_level(self, model, i: int, x_list: list, c_list: list, decode_derived_c: dict,
                      compute_ntp: bool, max_srcs, want_next_query: bool):
        raise NotImplementedError

    @torch.no_grad()
    def generate_encode_only(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        enc0 = model.encoders[0]
        for _ in range(n_new_bytes):
            out = enc0(all_bytes, level=0, window=model.windows[0], compute_ntp=False)
            next_byte = sample_next_byte(enc0.byte_output_weight, out["hidden"][:, -1, :])
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
        if was_training:
            model.train()
        return all_bytes[0]

    @torch.no_grad()
    def _generate_blockwise(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                             code_source: str = "pred", gt_full_bytes: torch.Tensor | None = None) -> dict:
        """Real fix (chat 2026-08-20) for the base class's generate_no_cache/generate_kv_cache,
        which for this decoder always fell back to result["h_list"][0][:, -1, :] (StackDecoderV1's
        non-top decode_level always returns query_last=None) -- that fallback is causally BLIND to
        whichever byte was JUST appended whenever it doesn't complete a new block (decode's own
        hidden output is truncated to n_blocks*K, whole blocks only), so every other byte reused the
        IDENTICAL hidden state as the previous step and argmax-sampled the identical byte again --
        confirmed empirically as exact character-doubling in generated text (e.g. "username" ->
        "uusseerrnnaammee"), not just a theoretical gap.

        Fix: decode one whole NEW block at a time using the SAME primitives
        check_roundtrip_consistency/check_decode_modes already validate (bos_interleaved_self_attn +
        own_block_cross_attn_decode), teacher-forcing each new byte back in before predicting the
        next one WITHIN that block. code_source picks where each new block's own code comes from:
        "pred" (default, the actual generation-time path) samples it from level1's genuine NTP over
        whatever's been generated so far (see Encoder.forward's docstring); "gt" (diagnostic only,
        needs gt_full_bytes to already contain the true continuation) uses the real encoded code
        instead, isolating loop-mechanics correctness from level1's forecast quality -- see
        check_blockwise_gen_consistency, which reuses THIS call's own `code_used` return value for
        its batched comparison, so the two passes can never disagree about which code was used.

        Returns dict(bytes=(B, prompt_len+n_new_bytes), code_used=(B, n_blocks_total, code_dim))."""
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        K = model.cfg.Ks[0]
        all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]
        bb_self, bb_cross = self.stage_lms[0][0], self.stage_lms[0][1]
        self_window = model.decode_windows[0][0]
        own_code_window = model.decode_windows[0][1] if len(model.decode_windows[0]) > 1 else None

        n_blocks_prompt = all_bytes.shape[1] // K
        code_parts = [model.encoders[0](all_bytes, level=0, window=model.windows[0],
                                         compute_ntp=False)["code"][:, :n_blocks_prompt, :]] if n_blocks_prompt > 0 else []

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            n_blocks_prev = all_bytes.shape[1] // K
            if code_source == "gt":
                gt = gt_full_bytes.to(device)
                if gt.dim() == 1:
                    gt = gt.unsqueeze(0)
                real_code = model.encoders[0](gt[:, :(n_blocks_prev + 1) * K], level=0,
                                               window=model.windows[0], compute_ntp=False)["code"]
                next_code = real_code[:, n_blocks_prev:n_blocks_prev + 1, :]
            elif code_source == "pred":
                codes = encode_up_to(model, all_bytes, level=1)
                enc1 = model.encoders[1]
                out1 = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
                sampled = enc1.quant.sample_next(enc1.lm, out1["hidden"][:, -1, :], model.cfg.vocab)
                next_code = sampled.unsqueeze(1)
            else:
                raise ValueError(f"code_source must be 'gt' or 'pred', got {code_source!r}")
            code_parts += [next_code]
            code = torch.cat(code_parts, dim=1)
            code_embeds = bb_cross.quant.embed_for_decode(bb_cross, code)
            n_blocks = n_blocks_prev + 1

            buf = torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], K)], dim=1)
            for t in range(K):
                x0 = bb_self.embed_input(buf, True)
                h_self_full = bos_interleaved_self_attn(bb_self, x0, K, self_window, strip_bos=False)
                h_full = own_block_cross_attn_decode(bb_cross, h_self_full, code_embeds, K, own_code_window)
                h = h_full.view(buf.shape[0], n_blocks, K + 1, -1)[:, :, :K, :]
                next_byte = sample_next_byte(bb_cross.byte_output_weight, h[:, -1, t, :])
                buf = buf.clone()
                buf[:, n_blocks_prev * K + t] = next_byte
            all_bytes = buf

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            model.train()
        return make_dict(bytes=all_bytes, code_used=torch.cat(code_parts, dim=1))

    @torch.no_grad()
    def generate_no_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_srcs: int | None = None) -> torch.Tensor:
        if self.n_levels < 2:
            return super().generate_no_cache(model, prompt_bytes, n_new_bytes, device, max_srcs)
        return self._generate_blockwise(model, prompt_bytes, n_new_bytes, device, code_source="pred")["bytes"][0]

    @torch.no_grad()
    def generate_kv_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_srcs: int | None = None) -> torch.Tensor:
        if self.n_levels < 2:
            return super().generate_kv_cache(model, prompt_bytes, n_new_bytes, device, max_srcs)
        return self._generate_blockwise(model, prompt_bytes, n_new_bytes, device, code_source="pred")["bytes"][0]

    @torch.no_grad()
    def check_blockwise_gen_consistency(self, model, full_bytes: torch.Tensor, device: str,
                                         prompt_len: int, code_source: str, log=print, label: str = "") -> int:
        """Mechanics-only correctness check for _generate_blockwise (chat 2026-08-20), NOT an
        accuracy check: given a FIXED code sequence, decode is deterministic, so the incremental
        per-block loop MUST exactly match a single batched bos_interleaved_self_attn/
        own_block_cross_attn_decode call over the same span -- reuses _generate_blockwise's OWN
        `code_used` return value for that batched pass (never re-derives it independently), so the
        two computations can only disagree if the incremental LOOP itself is wrong, not because of
        two different code assemblies drifting apart. Expect 0/N mismatches in EITHER mode: "gt"
        isolates loop correctness from level1's forecast quality (diagnostic only, needs the true
        continuation already in full_bytes, never valid for real generation -- own code is
        autoencoder-circular); "pred" additionally exercises the real generation-time code source,
        still expecting 0/N since this checks the LOOP, not level1's prediction accuracy (that's
        check_decode_modes's job)."""
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        K = model.cfg.Ks[0]
        prompt_len = prompt_len // K * K
        n_new_bytes = (full_bytes.shape[1] - prompt_len) // K * K
        prefix = f"blockwise_gen_consistency_{code_source}_{label}" if label else f"blockwise_gen_consistency_{code_source}"
        if n_new_bytes < K:
            log(f"{prefix}: skipped (not enough trailing bytes for a full new block)")
            if was_training:
                model.train()
            return 0

        out = self._generate_blockwise(model, full_bytes[:, :prompt_len], n_new_bytes, device,
                                        code_source=code_source, gt_full_bytes=full_bytes)
        incremental, code = out["bytes"], out["code_used"]
        n_blocks = (prompt_len + n_new_bytes) // K

        bb_self, bb_cross = self.stage_lms[0][0], self.stage_lms[0][1]
        self_window = model.decode_windows[0][0]
        own_code_window = model.decode_windows[0][1] if len(model.decode_windows[0]) > 1 else None
        code_embeds = bb_cross.quant.embed_for_decode(bb_cross, code)

        predicted = incremental[:, :n_blocks * K].clone()
        predicted[:, :prompt_len] = full_bytes[:, :prompt_len]
        for t in range(K):
            x0 = bb_self.embed_input(predicted, True)
            h_self_full = bos_interleaved_self_attn(bb_self, x0, K, self_window, strip_bos=False)
            h_full = own_block_cross_attn_decode(bb_cross, h_self_full, code_embeds, K, own_code_window)
            h = h_full.view(predicted.shape[0], n_blocks, K + 1, -1)[:, :, :K, :]
            next_byte = sample_next_byte(bb_cross.byte_output_weight, h[:, :, t, :])
            predicted = predicted.view(predicted.shape[0], n_blocks, K).clone()
            predicted[:, prompt_len // K:, t] = next_byte[:, prompt_len // K:]
            predicted = predicted.view(predicted.shape[0], n_blocks * K)

        batched_new = predicted[0, prompt_len:prompt_len + n_new_bytes]
        n_mismatch = (batched_new != incremental[0, prompt_len:prompt_len + n_new_bytes]).sum().item()
        if was_training:
            model.train()
        log(f"{prefix}: {n_mismatch}/{n_new_bytes} bytes mismatched (incremental blockwise loop vs "
            f"single batched decode call, same code -- mechanics check, not an accuracy check)")
        return n_mismatch

    def validate_generation(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
        out_a = self.generate_no_cache(model, prompt_bytes, n_new_bytes, device)
        out_b = self.generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
        assert torch.equal(out_a, out_b), "generate_no_cache and generate_kv_cache diverged"
        return True

    @torch.no_grad()
    def check_gen_consistency(self, model, full_bytes: torch.Tensor, device: str, prompt_len: int = 32,
                               tol: float = 1e-3, log=print, label: str = "") -> int:
        """n_levels>=2 only: level0's non-top decode (bos_interleaved_self_attn) reshapes its input
        into (n_blocks, K) blocks, silently dropping any trailing bytes that don't complete a block
        -- so a truncated context length that isn't a multiple of K0 shifts which absolute position
        the returned last-position query actually corresponds to (see docs/status.md's 2026-08-20
        session log). Restricting t to K0-aligned values keeps this check meaningful; n_levels==1
        (level0 is top, no such reshape) is unaffected and checks every t as before."""
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        L_total = full_bytes.shape[1]
        K0 = model.cfg.Ks[0]
        block_aligned_only = model.n_levels >= 2

        result_tf = model._run(full_bytes, compute_ntp=False, max_srcs=None, want_next_query=False)
        embed0 = result_tf["embed_weights"][0] if result_tf["embed_weights"][0] is not None else model.encoders[0].byte_output_weight
        logits_tf_all = F.linear(result_tf["h_list"][0][0], embed0)

        n_mismatch, n_checked = 0, 0
        for t in range(prompt_len, L_total - 1):
            if block_aligned_only and t % K0 != 0:
                continue
            ref_idx = t - 1
            if ref_idx < 0 or ref_idx >= logits_tf_all.shape[0]:
                continue
            padded = full_bytes[:, :t]
            result_gen = model._run(padded, compute_ntp=False, max_srcs=None, want_next_query=True)
            embed_gen = result_gen["embed_weights"][0] if result_gen["embed_weights"][0] is not None else model.encoders[0].byte_output_weight
            query_gen = result_gen["next_query"][0] if result_gen["next_query"][0] is not None else result_gen["h_list"][0][:, -1, :]
            logits_gen = F.linear(query_gen[0], embed_gen)
            n_checked += 1
            if (logits_gen - logits_tf_all[ref_idx]).abs().max().item() >= tol:
                n_mismatch += 1
        if was_training:
            model.train()
        prefix = f"gen_consistency_{label}" if label else "gen_consistency"
        log(f"{prefix}: {n_mismatch}/{n_checked} timesteps mismatched "
            f"(generation vs teacher-forced logits on ground-truth input"
            f"{', K0-aligned t only' if block_aligned_only else ''})")
        return n_mismatch

    @torch.no_grad()
    def check_roundtrip_consistency(self, model, full_bytes: torch.Tensor, device: str, log=print,
                                     label: str = "") -> int:
        """Baseline metric 1 from docs/qcute_v1_plan.md's generation-feasibility section: decode
        level0 using ONLY the real own-code (teacher-forced, no upper-level-LM sampling at all),
        generating each block's K bytes autoregressively from the model's own predictions, then
        re-encode the result and compare against the real code that produced it -- isolates
        encode/decode round-trip noise from the (separately flagged, deferred) context-asymmetry
        question. Diagnostic only: prints a mismatch count, never halts or gates anything.
        StackDecoderV1-specific (assumes the seed-token-interleaved self-attn + cross-attn-to-own-code
        non-top-level structure); no-op (prints and returns 0) for n_levels==1, where level0 is
        the (structurally unrelated, unchanged-from-v5) top level."""
        if model.n_levels < 2:
            log(f"roundtrip{'_' + label if label else ''}: skipped (n_levels==1, level0 is top, "
                f"not the seed-token-interleaved decode this check targets)")
            return 0
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        cfg = model.cfg
        K = cfg.Ks[0]
        L = full_bytes.shape[1]
        n_blocks = L // K
        prefix = f"roundtrip_{label}" if label else "roundtrip"
        if n_blocks < 1:
            log(f"{prefix}: skipped (sequence shorter than one block)")
            if was_training:
                model.train()
            return 0

        enc0 = model.encoders[0]
        enc_out = enc0(full_bytes, level=0, window=model.windows[0], compute_ntp=False)
        real_code = enc_out["code"][:, :n_blocks, :]

        decoder = model.decoder
        bb_self, bb_cross = decoder.stage_lms[0][0], decoder.stage_lms[0][1]
        self_window = model.decode_windows[0][0]
        own_code_window = model.decode_windows[0][1] if len(model.decode_windows[0]) > 1 else None
        code_embeds = bb_cross.quant.embed_for_decode(bb_cross, real_code)

        predicted = full_bytes[:, :n_blocks * K].clone()
        for t in range(K):
            x0 = bb_self.embed_input(predicted, True)
            h_self_full = bos_interleaved_self_attn(bb_self, x0, K, self_window, strip_bos=False)
            h_full = own_block_cross_attn_decode(bb_cross, h_self_full, code_embeds, K, own_code_window)
            h = h_full.view(predicted.shape[0], n_blocks, K + 1, -1)[:, :, :K, :]
            next_byte = sample_next_byte(bb_cross.byte_output_weight, h[:, :, t, :])
            predicted = predicted.view(predicted.shape[0], n_blocks, K)
            predicted = predicted.clone()
            predicted[:, :, t] = next_byte
            predicted = predicted.view(predicted.shape[0], n_blocks * K)

        reenc_out = enc0(predicted, level=0, window=model.windows[0], compute_ntp=False)
        reenc_code = reenc_out["code"][:, :n_blocks, :]
        real_ids = enc0.quant.to_ids(real_code)
        reenc_ids = enc0.quant.to_ids(reenc_code)
        n_mismatch = (real_ids != reenc_ids).sum().item()
        n_total = real_ids.numel()
        acc = (n_total - n_mismatch) / n_total if n_total > 0 else 1.0
        if was_training:
            model.train()
        acc_str = "1.0" if acc >= 1.0 else f"{acc:.2f}".lstrip("0")
        log(f"{prefix}_acc: {acc_str} ({n_total - n_mismatch}/{n_total} blocks round-tripped to "
            f"the same code after decode->re-encode -- real ground-truth codes, no upper-level "
            f"sampling, baseline noise floor, diagnostic only)")
        return n_mismatch

    @torch.no_grad()
    def check_decode_modes(self, model, full_bytes: torch.Tensor, device: str, log=print,
                            label: str = "") -> dict:
        """Dual-mode generation-quality proxy from docs/qcute_v1_plan.md's generation-feasibility
        section: decode level0 twice for the same span -- once from the REAL ground-truth own-code
        (gt mode, upper bound on decode quality) and once from level1's OWN sampled prediction of
        that code (pred mode, the actual generation-time signal, "always lead one block ahead") --
        report byte accuracy against ground truth for each. A large gt-vs-pred gap means level1's
        code forecast isn't informative enough to support real generation on its own (see the
        "two paths" discussion -- path (a), draft+encode+refine, would be the fallback). Position
        0's block has no level1 prediction (nothing precedes it), so both modes are compared over
        blocks 1..n_blocks-1 only, the same span either way. Diagnostic only. n_levels>=2 required
        (no-op otherwise); n_levels>2 not yet generalized (only level0/level1 handled)."""
        if model.n_levels < 2:
            log(f"decode_modes{'_' + label if label else ''}: skipped (n_levels==1)")
            return {}
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        cfg = model.cfg
        K = cfg.Ks[0]
        L = full_bytes.shape[1]
        n_blocks = L // K
        prefix = f"decode_modes_{label}" if label else "decode_modes"
        if n_blocks < 2:
            log(f"{prefix}: skipped (need >=2 blocks for level1 to have a real prediction)")
            if was_training:
                model.train()
            return {}

        enc0, enc1 = model.encoders[0], model.encoders[1]
        enc0_out = enc0(full_bytes, level=0, window=model.windows[0], compute_ntp=False)
        real_code = enc0_out["code"][:, :n_blocks, :]
        enc1_out = enc1(real_code, level=1, window=model.windows[1], compute_ntp=False)
        h1 = enc1_out["hidden"]
        predicted_code = enc1.quant.sample_next(enc1.lm, h1[:, :-1, :], cfg.vocab)

        gt_code = real_code[:, 1:, :]
        n_cmp = gt_code.shape[1]
        B = full_bytes.shape[0]
        target_bytes = full_bytes[:, K:n_blocks * K].reshape(B, n_cmp, K)

        decoder = model.decoder
        bb_self, bb_cross = decoder.stage_lms[0][0], decoder.stage_lms[0][1]
        self_window = model.decode_windows[0][0]
        own_code_window = model.decode_windows[0][1] if len(model.decode_windows[0]) > 1 else None

        def decode_from(code):
            code_embeds = bb_cross.quant.embed_for_decode(bb_cross, code)
            buf = target_bytes.reshape(B, n_cmp * K).clone()  # causal, so initial values here never leak (see check_roundtrip_consistency)
            for t in range(K):
                x0 = bb_self.embed_input(buf, True)
                h_self_full = bos_interleaved_self_attn(bb_self, x0, K, self_window, strip_bos=False)
                h_full = own_block_cross_attn_decode(bb_cross, h_self_full, code_embeds, K, own_code_window)
                h = h_full.view(B, n_cmp, K + 1, -1)[:, :, :K, :]
                next_byte = sample_next_byte(bb_cross.byte_output_weight, h[:, :, t, :])
                buf = buf.view(B, n_cmp, K).clone()
                buf[:, :, t] = next_byte
                buf = buf.view(B, n_cmp * K)
            return buf.view(B, n_cmp, K)

        gt_acc = (decode_from(gt_code) == target_bytes).float().mean().item()
        pred_acc = (decode_from(predicted_code) == target_bytes).float().mean().item()
        if was_training:
            model.train()

        def fmt(a):
            return "1.0" if a >= 1.0 else f"{a:.2f}".lstrip("0")
        log(f"{prefix}: gt_byte_acc={fmt(gt_acc)}  pred_byte_acc={fmt(pred_acc)}  "
            f"(gt=decode from real ground-truth code, upper bound; pred=decode from level1's own "
            f"sampled code prediction, the real generation-time signal)")
        return {"gt_byte_acc": gt_acc, "pred_byte_acc": pred_acc}

    @torch.no_grad()
    def generate_level_codes(self, model, prompt_bytes: torch.Tensor, level: int, n_new_codes: int,
                              device: str) -> torch.Tensor:
        """THE mechanism for getting a genuinely NEW (not-yet-existing) code at any level, per
        Encoder.forward's own docstring (chat 2026-08-20): `model.encoders[level]` samples from ITS
        OWN NTP hidden state to produce the next value in level `level`'s INPUT alphabet, i.e. the
        next code produced by level `level-1` (or the next raw byte for level=0). Feed the result
        into that lower level's decode_level as an ordinary real code to actually decode new bytes
        -- decode_level itself has no way to answer "what comes next" for its OWN autoencoder-style
        code (see StackDecoder.decode_level's query_last=None comment), this is the answer."""
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        codes = encode_up_to(model, prompt_bytes, level)
        n_prompt_codes = codes.shape[1]
        enc_level = model.encoders[level]
        for _ in range(n_new_codes):
            out = enc_level(codes, level=level, window=model.windows[level], compute_ntp=False)
            next_code = enc_level.quant.sample_next(enc_level.lm, out["hidden"][:, -1, :], model.cfg.vocab)
            codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
        if was_training:
            model.train()
        return enc_level.quant.to_ids(codes[0, n_prompt_codes:])

    @torch.no_grad()
    def level_ground_truth_codes(self, model, full_bytes: torch.Tensor, level: int, prompt_len: int,
                                  device: str) -> torch.Tensor:
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        codes = encode_up_to(model, full_bytes, level)
        cum_K = 1
        for j in range(level):
            cum_K *= model.cfg.Ks[j]
        ids = model.encoders[level - 1].quant.to_ids(codes[0])
        n_prompt_codes = prompt_len // cum_K
        return ids[n_prompt_codes:]

    def qualitative_generate(self, model, prompt_bytes: torch.Tensor, gen_len: int,
                              ground_truth, device: str, log=print, label: str = "") -> None:
        prefix = f"qual_{label}_" if label else "qual_"
        bits = model.cfg.input_preset
        out_uncond = self.generate_encode_only(model, prompt_bytes, gen_len, device)
        gen_bytes_uncond = pack_words(out_uncond[prompt_bytes.numel():].tolist(), bits)
        log(f"{prefix}prompt:              {pack_words(prompt_bytes.tolist(), bits)!r}")
        if ground_truth is not None:
            log(f"{prefix}ground_truth:        {pack_words(ground_truth.tolist(), bits)!r}")
        log(f"{prefix}level0_uncond:       {gen_bytes_uncond!r}")
        align_width = len("level0_uncond:") + 7
        # A single scalar max_srcs can't isolate "as if the top level didn't exist" once
        # n_levels>=3 -- a level's OWN nearest upper track survives any cap>=2 regardless (e.g.
        # level1 in a Ks=(2,2,1) model always sees level2 under a global cap of 2, since level2 is
        # its only upper track). Use the genuine per-level truncation instead: drop the real top
        # level from every non-top level's conditioning entirely (own code + at most the next
        # coarser level that ISN'T the real top), the actual submodel a shallower Ks would train --
        # see qcute_v1_common.py's Config.active_srcs_mode docstring (renamed from
        # curriculum_max_srcs 2026-08-23), chat 2026-08-21.
        n = model.n_levels
        ks21_equiv = tuple((1 + max(0, (n - 2) - i)) if i < n - 1 else None for i in range(n))
        for tag, max_srcs in (("ks21", ks21_equiv), ("full", None)):
            out_m = self.generate_no_cache(model, prompt_bytes, gen_len, device, max_srcs=max_srcs)
            gen_bytes_m = pack_words(out_m[prompt_bytes.numel():].tolist(), bits)
            label = f"level0_mode{tag}:"
            pad = " " * max(1, align_width - len(label))
            log(f"{prefix}{label}{pad}{gen_bytes_m!r}")
        for level in range(1, model.n_levels):
            cum_K = 1
            for k in model.cfg.Ks[:level]:
                cum_K *= k
            n_new_codes = gen_len // cum_K
            if n_new_codes <= 0:
                continue
            gen = self.generate_level_codes(model, prompt_bytes, level, n_new_codes, device)
            log(f"{prefix}level{level}_gen:          {gen.tolist()}")
            if ground_truth is not None:
                full_bytes = torch.cat([prompt_bytes.reshape(-1), ground_truth.reshape(-1)])
                gt = self.level_ground_truth_codes(model, full_bytes, level, prompt_bytes.numel(), device)
                log(f"{prefix}level{level}_gt:           {gt.tolist()}")


class ConcatDecoder(Decoder):
    def __init__(self, cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs):
        super().__init__(cfg, n_levels)
        self.stage_lms = nn.ModuleList([LM(cfg, d_models[i], n_layers_list[i], vocabs[i]) for i in range(n_levels)])
        for bb in self.stage_lms:
            bb.merged_cache = {}

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_srcs, want_next_query):
        cfg = self.cfg
        L_i = x_list[i].shape[1]
        self_active = self_code_active(i, self.n_levels, cfg.use_self_code)
        if not self_active and self.n_levels == 1:
            warn_degenerate_self_code(i)
            return None
        tracks = []
        cum_K = 1
        bb = self.stage_lms[i]
        for j in range(i, self.n_levels):
            cum_K *= cfg.Ks[j]
            window = model.decode_windows[i][j - i]
            if window == 0:
                continue
            if L_i // cum_K < 1:
                break
            if j == i and not self_active:
                n_blocks = L_i // cum_K
                code_embeds = bb.self_code_const.view(1, 1, -1).expand(x_list[i].shape[0], n_blocks, -1)
            else:
                source_c = decode_derived_c[j] if (j > i and j in decode_derived_c) else c_list[j]
                code_embeds = bb.quant.embed_for_decode(bb, source_c)
            tracks += [(code_embeds, cum_K, window)]
        if not tracks:
            return None
        if torch.is_grad_enabled():
            # ConcatDecoder shares one LM per level across every track (no per-track weights),
            # so original-index bookkeeping doesn't matter here -- just drop the indices.
            tracks = [t for _, t in apply_track_dropout(tracks, getattr(model, "track_dropout_p", 0.0))]
        full_tracks = tracks[:max_srcs] if max_srcs is not None else tracks

        is_byte_level = i == 0
        K = cfg.Ks[i]
        D = bb.d_model
        x0 = bb.embed_input(x_list[i], is_byte_level)
        h, query_last = merged_decode_forward(bb, x0, full_tracks, extra_query=(want_next_query and i == 0))

        if compute_ntp:
            h_flat = h[:, K - 1:-1, :].reshape(-1, D)
            loss, acc = bb.ntp_loss_acc(h_flat, x_list[i][:, K:], is_byte_level)
        else:
            loss, acc = h.new_zeros(()), h.new_zeros(())

        code = bb.extract_code(h, x0, K, model.windows[i])["code"]
        return make_dict(hidden=h, query_last=query_last, loss=loss, acc=acc, code=code,
                          extra_losses=[], embed_weight=bb.byte_output_weight)


class StackDecoderV1(Decoder):
    """qcute_v1: only the top level is a genuine NTP/AR decoder (self-code recurrence,
    unconditional -- self_code_active/use_self_code no longer gate this, superseded). Every level
    below top decodes as: seed-token-interleaved causal self-attention over its own actual sequence
    (bos_interleaved_self_attn with strip_bos=False, self_code_const repurposed as the per-block
    seed token -- a full token, not a passive sink: it's a real query AND key like any real byte)
    THEN cross-attention to that SAME block's own-level code (own_block_cross_attn_decode, own
    code -- not a coarser level's, and NOT cross_attn_stage, which is reserved for the top level's
    coarser-code case), predicting each block's own K real bytes UNSHIFTED from that block's own
    code (own_block_decode_loss -- the seed token's own hidden state predicts the block's own first
    byte). See docs/qcute_v1_plan.md's worked example (c1 reconstructs 'ab', c2 reconstructs 'cd',
    directly, no lag). `decode_windows[i][0]` is the self-attention (seed-token) window
    (None=sync/default, K+1=async ablation); `decode_windows[i][1]` is the own-code cross-attention
    window (how many codes back are visible -- 1 = only this block's own code, as in the base
    worked example). n_levels>=3 (additional coarser-than-own-code cross-attention) not yet
    generalized here."""

    def __init__(self, cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs):
        super().__init__(cfg, n_levels)

        def make_self_stage(i):
            # decoder_own_stage_mode="shared" (default, 2026-08-23): reuse encode's own LM (embed,
            # blocks, self_code_const -- everything except decode's separate cross-attention stage)
            # for decode's self-attn stage, for BOTH top (unchanged v5 behavior) and non-top levels
            # (bb_self only -- bb_cross always stays a separate LM regardless of this setting).
            # "copy" (was share_encode_decode_self=False, the old default) trains an independent LM.
            bb = encoders[i].lm if cfg.decoder_own_stage_mode == "shared" else LM(cfg, d_models[i], n_layers_list[i], vocabs[i])
            if not hasattr(bb, "merged_cache"):
                bb.merged_cache = {}
            return bb

        def make_level(i):
            is_top = i == n_levels - 1
            if is_top:
                return nn.ModuleList([make_self_stage(i)])
            cross_layers = cfg.decode_cross_stage_layers if cfg.decode_cross_stage_layers is not None else n_layers_list[i]
            bb_self = make_self_stage(i)
            bb_cross = LM(cfg, d_models[i], cross_layers, vocabs[i])
            if not hasattr(bb_cross, "merged_cache"):
                bb_cross.merged_cache = {}
            return nn.ModuleList([bb_self, bb_cross])

        self.stage_lms = nn.ModuleList([make_level(i) for i in range(n_levels)])

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_srcs, want_next_query):
        cfg = self.cfg
        L_i = x_list[i].shape[1]
        K = cfg.Ks[i]
        is_top = i == self.n_levels - 1
        is_byte_level = i == 0

        if is_top:
            bb = self.stage_lms[i][0]
            D = bb.d_model
            window = model.decode_windows[i][0]
            n_blocks = L_i // K
            if window == 0 or n_blocks < 1:
                return None
            code_embeds = bb.quant.embed_for_decode(bb, c_list[i])
            x0 = bb.embed_input(x_list[i], is_byte_level)
            h, query_last = merged_decode_forward(bb, x0, [(code_embeds, K, window)],
                                                    extra_query=(want_next_query and i == 0))
            if compute_ntp:
                h_flat = h[:, K - 1:-1, :].reshape(-1, D)
                loss, acc = bb.ntp_loss_acc(h_flat, x_list[i][:, K:], is_byte_level)
            else:
                loss, acc = h.new_zeros(()), h.new_zeros(())
            code = bb.extract_code(h, x0, K, window)["code"]
            valid_next_query = want_next_query and i == 0 and L_i % K == 0
            return make_dict(hidden=h, query_last=(query_last if valid_next_query else None),
                              loss=loss, acc=acc, code=code, extra_losses=[], embed_weight=bb.byte_output_weight)

        n_blocks = L_i // K
        if n_blocks < 1:
            return None
        bb_self, bb_cross = self.stage_lms[i][0], self.stage_lms[i][1]
        D = bb_self.d_model
        self_window = model.decode_windows[i][0]
        own_code_window = model.decode_windows[i][1] if len(model.decode_windows[i]) > 1 else None

        x0 = bb_self.embed_input(x_list[i], is_byte_level)
        h_self_full = bos_interleaved_self_attn(bb_self, x0, K, self_window, strip_bos=False)

        source_c = c_list[i][:, :n_blocks, :]
        code_embeds = bb_cross.quant.embed_for_decode(bb_cross, source_c)
        h_full = own_block_cross_attn_decode(bb_cross, h_self_full, code_embeds, K, own_code_window)

        L_used = n_blocks * K
        h = h_full.view(h_full.shape[0], n_blocks, K + 1, D)[:, :, :K, :].reshape(h_full.shape[0], L_used, D)
        if compute_ntp:
            loss, acc = own_block_decode_loss(bb_cross, h_full, x_list[i][:, :L_used], K, is_byte_level)
        else:
            loss, acc = h.new_zeros(()), h.new_zeros(())
        return make_dict(hidden=h, query_last=None, loss=loss, acc=acc, code=None,
                          extra_losses=[], embed_weight=bb_cross.byte_output_weight)


def encode_like_decode_loss(bb: LM, h_real: torch.Tensor, h_seed: torch.Tensor, x_real: torch.Tensor,
                             K: int, is_byte_level: bool) -> tuple:
    """Target alignment for StackDecoder, same UNSHIFTED-per-block convention as
    own_block_decode_loss: block b's K queries are [h_seed_b, h_real[b*K], ..., h_real[b*K+K-2]]
    (seed_b stands in for 'before block b'; h_real's own within-block causal chain -- already
    conditioned on code_b via encode_like_self_attn_decode's spliced cross-attn -- covers the
    rest), predicting that SAME block's own K real bytes in order."""
    B, _, D = h_real.shape
    n_blocks = h_seed.shape[1]
    L_used = n_blocks * K
    h_real_b = h_real[:, :L_used, :].view(B, n_blocks, K, D)
    h_q = torch.cat([h_seed.unsqueeze(2), h_real_b[:, :, :K - 1, :]], dim=2).reshape(B, L_used, D)
    return bb.ntp_loss_acc(h_q.reshape(-1, D), x_real[:, :L_used], is_byte_level)


class StackDecoder(Decoder):
    """qcute_v1 (2026-08-20, "every upper code conditioning"): "like v5's [old] StackDecoder except
    no self code" -- chains v5-style staged cross-attention (cross_attn_stage: cross-attn + MLP per
    layer, one stage per additional coarser code) on top of a non-interleaved track0 mechanism.
    Supersedes an earlier, single-track-only StackDecoder draft (track0 only, no upper conditioning
    at all) -- that shape is now just this class's cfg.cond_depth=1 special case, not a separate
    class; see cond_depth below.

    NAMING (fixed 2026-08-23, was backwards throughout this file/CLAUDE.md/docs/status.md before
    that date): a code is named by the level that owns it AS INPUT, not the level that produced it
    -- Encoder.forward's own docstring says this of itself ("level j's own INPUT = level (j-1)'s
    code stream"). So `c_list[N]` is always level (N+1)'s code, never level N's -- level i's
    decoder never has anything of its own to condition on beyond the bytes/values it's
    reconstructing; every code it cross-attends to, including "track0", belongs to the level above.
    Calling track0 "own code" throughout this file means "this SAME BLOCK's code" (temporal/spatial
    locality within the code stream, as in own_block_cross_attn_decode) -- a different, still-valid
    "own" than the level-ownership one that was wrong; left as-is, only level-ownership claims were
    fixed.

    Track 0 (cum_K=Ks[i], conditions on level (i+1)'s code) uses encode_like_self_attn_decode +
    seed_query_decode: a single per-level LM (no separate bb_self/bb_cross split -- 'feed back the
    kv' means reusing the SAME weights and SAME K/V, not recomputing them in a second LM) runs two
    passes -- plain causal self-attn over real bytes, literally the encode pass, with cross-attention
    to level (i+1)'s code spliced in per layer (encode_like_self_attn_decode), saving each layer's
    self-attention K/V; then a per-block trainable seed token reuses those saved K/V directly as its
    cross-attention source (seed_query_decode), never recomputing or re-interleaving them, and never
    falling back to a self-code substitute the way v5's old StackDecoder/_SELF_CONST could -- track0's
    code is always real (self_code_active no longer gates this, same as StackDecoderV1). Causal,
    static-shape, and KV-cache-able -- see chat 2026-08-20 for the validity argument. Top level
    unchanged (the genuine self-code-recurrent NTP path, no seed token involved either way).

    Tracks 1..T-1 (level i+t+1's code, t=1..T-1) each run a SEPARATE cross_attn_stage LM, staged
    sequentially -- track t's query is track (t-1)'s output, so gradient depth grows with the
    number of levels above (same n_layers*(1+n_tracks) depth tradeoff the old v5 StackDecoder had,
    see CLAUDE.md's Architecture section).

    `decode_windows[i][0]` doubles as BOTH track0's self-attention window (pass1's causal chain)
    AND its cross-attention window into level (i+1)'s code -- a deliberate collapse of the two
    separate knobs a track0-only decode would need into v5/ConcatDecoder's one-window-per-track
    convention, so decode_windows keeps its existing n_sources-length shape unchanged (no plumbing
    changes needed in qcute_v1.py). `decode_windows[i][t]` for t>=1 is track t's (level i+t+1's
    code) cross-attention window, same meaning as v5's StackDecoder. Track0 is never dropped/pruned
    (every later stage is built on it); tracks 1..T-1 are subject to apply_track_dropout +
    max_srcs, same policy as v5, counted so max_srcs==1 means "track0 (level i+1's code) only,
    no upper conditioning".

    `cfg.cond_depth` caps how many levels above level (i+1) each level actually conditions on --
    -1 (default) = pervasive, every level above EXCEPT one past the current top (n_levels-2-i
    tracks, see below); 1 = track0 only, the minimal case (just level i+1's code, no further
    tracks). Static, unlike max_srcs (a per-forward-call ablation knob) -- it also caps how many
    cross-attn-stage LMs __init__ allocates, so a shallow cond_depth means genuinely fewer
    parameters, not just unused ones.

    The code one level above the topmost real level (`c_list[n_levels-1]`, i.e. "level n_levels" --
    a domain nothing actually owns as input, since no `encoders[n_levels]` exists) is
    HARD-EXCLUDED from every level's conditioning, unconditionally (not gated by
    cond_depth/max_srcs/a curriculum) -- no LM ever forecasts that code during real generation
    (it's produced solely by the topmost level's own encoder's self-NTP, self-extracted, never
    consumed as anyone's input), so conditioning on it trains every level against a signal that
    free-rollout generation can't actually supply reliably. Confirmed empirically (2026-08-23,
    ks21/ks221 "notoplevel" curriculum_max_srcs ablations, docs/status.md) that both hierarchies
    still overfit cleanly -- and ks221 generates measurably better -- with this code excluded, so
    it's now baked into __init__'s allocation (n_upper caps at n_levels-2-i, never n_levels-1-i)
    rather than left as an opt-in curriculum knob."""

    def __init__(self, cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs):
        super().__init__(cfg, n_levels)

        if (cfg.decoder_own_stage_mode == "shared" or cfg.kv_lm_mode == "shared") and cfg.byte_head_tied:
            print("WARNING: decoder_own_stage_mode/kv_lm_mode='shared' reuses an encoder's weights "
                  "for track0's self-attention/code-contextualization -- combined with "
                  "byte_head_tied=True, track0's conditional (code-informed) and the encoder's "
                  "unconditional byte-output projections collapse to the exact same tensor, "
                  "leaving no parameter specific to 'predict a byte given this cross-attended "
                  "hidden state'. Allowed, but byte_head_tied=False (the default) is recommended "
                  "whenever kv_lm_mode/decoder_own_stage_mode reuse an encoder, so the output head "
                  "stays independent even when the backbone doesn't.")

        def make_own_stage(i):
            bb = encoders[i].lm if cfg.decoder_own_stage_mode == "shared" else LM(cfg, d_models[i], n_layers_list[i], vocabs[i])
            if not hasattr(bb, "merged_cache"):
                bb.merged_cache = {}
            return bb

        def make_cross_stage(i):
            cross_layers = cfg.decode_cross_stage_layers if cfg.decode_cross_stage_layers is not None else n_layers_list[i]
            bb = LM(cfg, d_models[i], cross_layers, vocabs[i])
            if not hasattr(bb, "merged_cache"):
                bb.merged_cache = {}
            return bb

        def make_level(i):
            if i == n_levels - 1:
                return nn.ModuleList([make_own_stage(i)])
            n_upper = max(0, n_levels - 2 - i)  # excludes the topmost level (n_levels-1), see class docstring
            if cfg.cond_depth != -1:
                n_upper = min(n_upper, cfg.cond_depth)
            return nn.ModuleList([make_own_stage(i)] + [make_cross_stage(i) for _ in range(n_upper)])

        self.stage_lms = nn.ModuleList([make_level(i) for i in range(n_levels)])

        def make_kv_lm(i, j):
            # j = the code array index (c_list[j]) this kv_lm contextualizes for level i's decode
            # -- under the corrected naming (2026-08-23, see StackDecoder's docstring), c_list[j]
            # is "level (j+1)"'s code: the domain encoders[j+1] treats as its own input. "shared"
            # therefore reuses encoders[j+1].lm (FIXED 2026-08-23 -- was encoders[j].lm, a bug
            # inherited from the pre-fix naming convention where c_list[j] was wrongly thought of
            # as "level j's own code", making encoders[j].lm look like the natural choice when it
            # actually self-attends over c_list[j-1], not c_list[j]) -- requires
            # d_models[i]==d_models[j+1], since the result is used directly as level i's stage K/V.
            if cfg.kv_lm_mode == "identity":
                return nn.Identity()
            if cfg.kv_lm_mode == "shared":
                return KVContextLM(encoders[j + 1].lm)
            assert cfg.kv_lm_mode == "copy", f"kv_lm_mode must be identity|copy|shared, got {cfg.kv_lm_mode!r}"
            layers = cfg.kv_lm_layers if cfg.kv_lm_layers is not None else n_layers_list[i]
            return KVContextLM(LM(cfg, d_models[i], layers, vocabs[i]))

        def make_level_kv(i):
            if i == n_levels - 1:
                return nn.ModuleList([])
            n_upper = max(0, n_levels - 2 - i)  # excludes the topmost level (n_levels-1), see class docstring
            if cfg.cond_depth != -1:
                n_upper = min(n_upper, cfg.cond_depth)
            return nn.ModuleList([make_kv_lm(i, i + 1 + t) for t in range(n_upper)])

        self.kv_lms = nn.ModuleList([make_level_kv(i) for i in range(n_levels)])

        # track0's own kv_lm (2026-08-23, chat: "shouldn't kvlm if enabled exist to process level 1
        # code for level 0 decoder" -- it should, and this was a real gap: track0's cross-attention
        # target (c_list[i], "level (i+1)"'s code) never went through ANY kv_lm, unlike every upper
        # track, regardless of kv_lm_mode. Applies to every non-top level i (not just level0), so it
        # generalizes to decode_scope="pervasive" automatically -- module allocation here doesn't
        # depend on decode_scope, only on which levels HAVE a track0 to begin with.
        def make_track0_kv_lm(i):
            return make_kv_lm(i, i)

        self.track0_kv_lms = nn.ModuleList([make_track0_kv_lm(i) if i < n_levels - 1 else nn.Identity()
                                             for i in range(n_levels)])

    def _track0(self, bb0: LM, x_list_i: torch.Tensor, code_embeds0: torch.Tensor, K: int, n_blocks: int,
                track0_window: int | None, is_byte_level: bool, compute_ntp: bool) -> tuple:
        """Track0 (own-code) computation for one non-top level -- factored out of decode_level so
        subclasses can swap in a different same-level conditioning scheme without duplicating the
        upper-track chaining/cond_depth/h0_shifted-handoff logic below (see StackDecoderLocal for
        the block-diagonal variant this enables). Default here: the full design
        (encode_like_self_attn_decode + seed_query_decode), same-level self-attention reaching
        back `track0_window` bytes (None = unbounded) across ALL prior blocks, not just this one.

        Returns (h0, h0_shifted, loss0, acc0): h0 is UNSHIFTED (own-code reconstruction alignment,
        h0[p] reconstructs x_real[p]) -- correct for loss0/acc0 (track0's own reconstruction loss)
        but NOT valid input for cross_attn_stage (used by upper tracks), which assumes standard
        shifted alignment (h[p] predicts x_real[p+1]) -- h0_shifted is the re-derived view for that
        handoff (see the comment inline below for why/how)."""
        D = bb0.d_model
        self_w, cross_w = split_track0_window(track0_window)
        pass1 = encode_like_self_attn_decode(bb0, bb0.embed_input(x_list_i, is_byte_level), code_embeds0,
                                              K, self_w, cross_w)
        h_seed = seed_query_decode(bb0, pass1["saved_k"], pass1["saved_v"], code_embeds0, n_blocks, K,
                                    self_w, cross_w)

        B = x_list_i.shape[0]
        L_used = n_blocks * K
        h_real_b = pass1["hidden"][:, :L_used, :].view(B, n_blocks, K, D)
        h0 = torch.cat([h_seed.unsqueeze(2), h_real_b[:, :, :K - 1, :]], dim=2).reshape(B, L_used, D)
        if compute_ntp:
            loss0, acc0 = encode_like_decode_loss(bb0, pass1["hidden"], h_seed, x_list_i[:, :L_used], K, is_byte_level)
        else:
            loss0, acc0 = h0.new_zeros(()), h0.new_zeros(())

        # h0 is UNSHIFTED (own-code reconstruction alignment: h0[p] reconstructs x_real[p], correct
        # for loss0 above) but cross_attn_stage (used by every upper track below) assumes the
        # STANDARD shifted alignment (h[p] conditions on bytes 0..p, predicts x_real[p+1]) --
        # matching merged_decode_forward's own convention, which cross_attn_stage was written
        # against. h0 must NOT be handed to it directly (chat 2026-08-20: caught via a
        # generate_kv_cache validation mismatch -- corrupts every upper-track's training signal,
        # not just generation). Re-derive a shifted-aligned view instead: within a block, offsets
        # 0..K-2 are h_real itself (already valid causal states, alignment-agnostic); offset K-1
        # (last byte of block b) needs "conditions on all of block b, ready to predict block b+1's
        # first byte" -- exactly h_seed of block b+1. The very last block has no b+1 within L_used
        # (same boundary every standard NTP window has); its slot is a placeholder, provably unused
        # by cross_attn_stage's own loss (query_seq = h[:, :-1, :] excludes exactly this position).
        if n_blocks >= 2:
            next_block_seed = torch.cat([h_seed[:, 1:, :], h_seed[:, -1:, :]], dim=1)
        else:
            next_block_seed = h_seed
        h0_shifted = torch.cat([h_real_b[:, :, :K - 1, :], next_block_seed.unsqueeze(2)],
                                dim=2).reshape(B, L_used, D)
        return h0, h0_shifted, loss0, acc0

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_srcs, want_next_query):
        cfg = self.cfg
        L_i = x_list[i].shape[1]
        K = cfg.Ks[i]
        is_top = i == self.n_levels - 1
        is_byte_level = i == 0
        stage_bbs = self.stage_lms[i]
        bb0 = stage_bbs[0]

        if is_top:
            D = bb0.d_model
            window, _ = split_track0_window(model.decode_windows[i][0])  # top has no cross target
            n_blocks = L_i // K
            if window == 0 or n_blocks < 1:
                return None
            code_embeds = bb0.quant.embed_for_decode(bb0, c_list[i])
            x0 = bb0.embed_input(x_list[i], is_byte_level)
            h, query_last = merged_decode_forward(bb0, x0, [(code_embeds, K, window)],
                                                    extra_query=(want_next_query and i == 0))
            if compute_ntp:
                h_flat = h[:, K - 1:-1, :].reshape(-1, D)
                loss, acc = bb0.ntp_loss_acc(h_flat, x_list[i][:, K:], is_byte_level)
            else:
                loss, acc = h.new_zeros(()), h.new_zeros(())
            code = bb0.extract_code(h, x0, K, window)["code"]
            valid_next_query = want_next_query and i == 0 and L_i % K == 0
            return make_dict(hidden=h, query_last=(query_last if valid_next_query else None),
                              loss=loss, acc=acc, code=code, extra_losses=[], embed_weight=bb0.byte_output_weight)

        n_blocks = L_i // K
        if n_blocks < 1:
            return None
        D = bb0.d_model
        track0_window = model.decode_windows[i][0]

        source_c0 = c_list[i][:, :n_blocks, :]
        code_embeds0 = bb0.quant.embed_for_decode(bb0, source_c0)
        code_embeds0 = self.track0_kv_lms[i](code_embeds0)
        h0, h0_shifted, loss0, acc0 = self._track0(bb0, x_list[i], code_embeds0, K, n_blocks,
                                                     track0_window, is_byte_level, compute_ntp)

        upper_specs = []
        cum_K = K
        # capped at n_levels-1, not n_levels: the topmost level's own code is hard-excluded from
        # every other level's conditioning (see class docstring) -- must match __init__'s allocation.
        j_max = (self.n_levels - 1) if cfg.cond_depth == -1 else min(self.n_levels - 1, i + 1 + cfg.cond_depth)
        for j in range(i + 1, j_max):
            cum_K *= cfg.Ks[j]
            window = model.decode_windows[i][j - i]
            if window == 0:
                continue
            if L_i // cum_K < 1:
                break
            source_c = decode_derived_c[j] if j in decode_derived_c else c_list[j]
            upper_specs += [(source_c, cum_K, window)]

        if torch.is_grad_enabled():
            indexed = apply_track_dropout(upper_specs, getattr(model, "track_dropout_p", 0.0))
        else:
            indexed = list(enumerate(upper_specs))
        if max_srcs is not None:
            indexed = indexed[:max(0, max_srcs - 1)]

        x = h0_shifted
        loss_final, acc_final, code_final = loss0, acc0, None
        embed_weight_final = bb0.byte_output_weight
        extra_losses = []
        for t, (orig_idx, (source_c, cum_K, window)) in enumerate(indexed):
            bb = stage_bbs[orig_idx + 1]
            code_embeds = bb.quant.embed_for_decode(bb, source_c)
            code_embeds = self.kv_lms[i][orig_idx](code_embeds)
            is_last = t == len(indexed) - 1
            stage_result = cross_attn_stage(bb, x, code_embeds, x_list[i], i, cum_K, window, compute_ntp, is_last)
            x = stage_result["hidden"]
            if is_last:
                loss_final, acc_final, code_final = stage_result["loss"], stage_result["acc"], stage_result["code"]
                embed_weight_final = bb.byte_output_weight
            else:
                extra_losses += [stage_result["loss"]]

        if indexed:
            extra_losses = [loss0] + extra_losses

        # query_last (predicting a genuinely NEW, not-yet-existing block's own first byte) is left
        # unset here, same as StackDecoderV1's non-top branch -- track0's own code is autoencoder-
        # style (block b's code reconstructs block b's OWN bytes), so decode_level has no way to
        # answer "what predicts a brand-new block" FROM ITSELF (generate_no_cache/generate_kv_cache's
        # generic query_last fallback would otherwise silently resolve to a stale, already-known
        # position instead of erroring). This is NOT a gap decode_level needs to fill: the actual
        # answer (chat 2026-08-20, see Encoder.forward's and generate_level_codes's docstrings) is
        # that model.encoders[level] does genuine uncircular NTP over its own input and can sample
        # the next code directly -- decode_level only ever consumes a code someone else produced,
        # never invents one. check_decode_modes's 'pred mode' already exercises this end to end.
        return make_dict(hidden=x, query_last=None,
                          loss=loss_final, acc=acc_final, code=code_final,
                          extra_losses=extra_losses, embed_weight=embed_weight_final)

    @torch.no_grad()
    def _stack_generate_blockwise(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                             code_source: str = "pred", gt_full_bytes: torch.Tensor | None = None,
                             max_srcs: int | None = None) -> dict:
        """Generation fix (chat 2026-08-20) for this decoder, same root cause and same fix strategy
        as StackDecoderV1._generate_blockwise: query_last=None means the base class's
        generate_no_cache/generate_kv_cache fall back to a stale hidden state (see decode_level's
        query_last=None comment). Fix: decode one whole NEW block at a time using THIS decoder's own
        validated primitives (encode_like_self_attn_decode + seed_query_decode for track0, chained
        through upper_track_step for each active upper track), teacher-forcing each new byte back
        in before predicting the next one within the block. code_source picks where the new block's
        own (track0) code comes from -- "pred" (default, real generation) samples it from level1's
        genuine NTP; "gt" (diagnostic only, needs gt_full_bytes) uses the real encoded code, per
        check_blockwise_gen_consistency. Upper tracks (levels above the immediate one) are never
        circular -- each is a plain function of the level below's already-available code (real or
        just-predicted), computed fresh via encoders[j], not "predicted" the way track0's own code
        must be.

        Generalized 2026-08-21 (chat) to n_levels>2: chains through however many upper-track stages
        `self.stage_lms[0]` actually holds (capped by cond_depth at __init__ time already), in order
        level1, level2, .... `max_srcs` (int|None, own-level0 semantics only -- this method only
        ever produces level0's bytes) caps how many of those tracks are actually used, same
        own-code-always-kept convention as decode_level's max_srcs (max_srcs==1 means own code
        only). Unlike decode_level's max_srcs, there is no per-level tuple support here since this
        method only ever handles i=0's own generation -- pass an int/None, not a tuple.

        Returns dict(bytes=(B, prompt_len+n_new_bytes), code_used=(B, n_blocks_total, code_dim)) --
        code_used is track0's own code only, matching StackDecoderV1's return contract."""
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        cfg = self.cfg
        K = cfg.Ks[0]
        stage_bbs = self.stage_lms[0]
        bb0 = stage_bbs[0]
        n_upper_avail = len(stage_bbs) - 1
        n_upper = n_upper_avail if max_srcs is None else max(0, min(n_upper_avail, max_srcs - 1))
        track0_self_w, track0_cross_w = split_track0_window(model.decode_windows[0][0])
        D = bb0.d_model
        all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]

        n_blocks_prompt = all_bytes.shape[1] // K
        code_parts = [model.encoders[0](all_bytes, level=0, window=model.windows[0],
                                         compute_ntp=False)["code"][:, :n_blocks_prompt, :]] if n_blocks_prompt > 0 else []

        cum_Ks = []
        cum = K
        for j in range(1, n_upper + 1):
            cum *= cfg.Ks[j]
            cum_Ks += [cum]

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            n_blocks_prev = all_bytes.shape[1] // K
            if code_source == "gt":
                gt = gt_full_bytes.to(device)
                if gt.dim() == 1:
                    gt = gt.unsqueeze(0)
                real_code = model.encoders[0](gt[:, :(n_blocks_prev + 1) * K], level=0,
                                               window=model.windows[0], compute_ntp=False)["code"]
                next_code = real_code[:, n_blocks_prev:n_blocks_prev + 1, :]
            elif code_source == "pred":
                codes = encode_up_to(model, all_bytes, level=1)
                enc1 = model.encoders[1]
                out1 = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
                sampled = enc1.quant.sample_next(enc1.lm, out1["hidden"][:, -1, :], cfg.vocab)
                next_code = sampled.unsqueeze(1)
            else:
                raise ValueError(f"code_source must be 'gt' or 'pred', got {code_source!r}")
            code_parts += [next_code]
            code0 = torch.cat(code_parts, dim=1)
            code_embeds0 = bb0.quant.embed_for_decode(bb0, code0)
            code_embeds0 = self.track0_kv_lms[0](code_embeds0)
            n_blocks = n_blocks_prev + 1

            # Each active upper track j's OWN code over the level-(j-1) code stream -- never
            # circular, a plain function of the level below, unlike track0's own code.
            upper_code_embeds = []
            seq_repr = code0
            for t, j in enumerate(range(1, n_upper + 1)):
                seq_repr = model.encoders[j](seq_repr, level=j, window=model.windows[j], compute_ntp=False)["code"]
                bb_j = stage_bbs[t + 1]
                code_embeds_j = bb_j.quant.embed_for_decode(bb_j, seq_repr)
                upper_code_embeds += [self.kv_lms[0][t](code_embeds_j)]

            buf = torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], K)], dim=1)
            for t in range(K):
                x0 = bb0.embed_input(buf, True)
                pass1 = encode_like_self_attn_decode(bb0, x0, code_embeds0, K, track0_self_w, track0_cross_w)
                h_seed = seed_query_decode(bb0, pass1["saved_k"], pass1["saved_v"], code_embeds0, n_blocks, K,
                                            track0_self_w, track0_cross_w)
                h_real_b = pass1["hidden"].view(buf.shape[0], n_blocks, K, D)
                h_query = h_seed[:, -1, :] if t == 0 else h_real_b[:, -1, t - 1, :]

                pos = n_blocks_prev * K + t
                h_final = h_query.unsqueeze(1)
                for bb_j, code_embeds_j, cum_Kj in zip(stage_bbs[1:n_upper + 1], upper_code_embeds, cum_Ks):
                    h_final = upper_track_step(bb_j, h_final, pos, code_embeds_j, cum_Kj)
                embed_weight_final = stage_bbs[n_upper].byte_output_weight
                next_byte = sample_next_byte(embed_weight_final, h_final[:, 0, :])
                buf = buf.clone()
                buf[:, pos] = next_byte
            all_bytes = buf

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            model.train()
        return make_dict(bytes=all_bytes, code_used=torch.cat(code_parts, dim=1))

    @torch.no_grad()
    def generate_no_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_srcs: int | None | tuple = None) -> torch.Tensor:
        srcs0 = max_srcs[0] if isinstance(max_srcs, (list, tuple)) else max_srcs
        return self._stack_generate_blockwise(model, prompt_bytes, n_new_bytes, device,
                                               code_source="pred", max_srcs=srcs0)["bytes"][0]

    @torch.no_grad()
    def generate_kv_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_srcs: int | None | tuple = None) -> torch.Tensor:
        srcs0 = max_srcs[0] if isinstance(max_srcs, (list, tuple)) else max_srcs
        return self._stack_generate_blockwise(model, prompt_bytes, n_new_bytes, device,
                                               code_source="pred", max_srcs=srcs0)["bytes"][0]

    @torch.no_grad()
    def check_blockwise_gen_consistency(self, model, full_bytes: torch.Tensor, device: str,
                                         prompt_len: int, code_source: str, log=print, label: str = "") -> int:
        """Mechanics-only correctness check for _stack_generate_blockwise, same contract as
        StackDecoderV1's version: given a FIXED track0 code sequence, decode is deterministic, so
        the incremental per-block loop must exactly match a single batched
        encode_like_self_attn_decode + seed_query_decode + upper_track_step pass over the same
        span. Reuses _stack_generate_blockwise's OWN code_used return value (never re-derives it), so a
        mismatch can only mean the LOOP is wrong, not that two code assemblies drifted apart."""
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        cfg = self.cfg
        K = cfg.Ks[0]
        prompt_len = prompt_len // K * K
        n_new_bytes = (full_bytes.shape[1] - prompt_len) // K * K
        prefix = f"blockwise_gen_consistency_{code_source}_{label}" if label else f"blockwise_gen_consistency_{code_source}"
        if n_new_bytes < K:
            log(f"{prefix}: skipped (not enough trailing bytes for a full new block)")
            if was_training:
                model.train()
            return 0

        out = self._stack_generate_blockwise(model, full_bytes[:, :prompt_len], n_new_bytes, device,
                                        code_source=code_source, gt_full_bytes=full_bytes)
        incremental, code0 = out["bytes"], out["code_used"]
        n_blocks = (prompt_len + n_new_bytes) // K

        bb0, bb1 = self.stage_lms[0][0], self.stage_lms[0][1]
        cum_K1 = K * cfg.Ks[1]
        track0_self_w, track0_cross_w = split_track0_window(model.decode_windows[0][0])
        D = bb0.d_model
        code_embeds0 = bb0.quant.embed_for_decode(bb0, code0)
        code_embeds0 = self.track0_kv_lms[0](code_embeds0)
        code1 = model.encoders[1](code0, level=1, window=model.windows[1], compute_ntp=False)["code"]
        code_embeds1 = bb1.quant.embed_for_decode(bb1, code1)

        predicted = incremental[:, :n_blocks * K].clone()
        predicted[:, :prompt_len] = full_bytes[:, :prompt_len]
        pos_all = torch.arange(n_blocks * K, device=device)
        for t in range(K):
            x0 = bb0.embed_input(predicted, True)
            pass1 = encode_like_self_attn_decode(bb0, x0, code_embeds0, K, track0_self_w, track0_cross_w)
            h_seed = seed_query_decode(bb0, pass1["saved_k"], pass1["saved_v"], code_embeds0, n_blocks, K,
                                        track0_self_w, track0_cross_w)
            h_real_b = pass1["hidden"].view(predicted.shape[0], n_blocks, K, D)
            h_query = h_seed if t == 0 else h_real_b[:, :, t - 1, :]  # (B, n_blocks, D)

            new_bytes = predicted.new_zeros(predicted.shape[0], n_blocks)
            for b in range(prompt_len // K, n_blocks):
                pos = b * K + t
                h_final = upper_track_step(bb1, h_query[:, b:b + 1, :], pos, code_embeds1, cum_K1)
                new_bytes[:, b] = sample_next_byte(bb1.byte_output_weight, h_final[:, 0, :])
            predicted = predicted.view(predicted.shape[0], n_blocks, K).clone()
            predicted[:, prompt_len // K:, t] = new_bytes[:, prompt_len // K:]
            predicted = predicted.view(predicted.shape[0], n_blocks * K)

        batched_new = predicted[0, prompt_len:prompt_len + n_new_bytes]
        n_mismatch = (batched_new != incremental[0, prompt_len:prompt_len + n_new_bytes]).sum().item()
        if was_training:
            model.train()
        log(f"{prefix}: {n_mismatch}/{n_new_bytes} bytes mismatched (incremental blockwise loop vs "
            f"single batched decode call, same code -- mechanics check, not an accuracy check)")
        return n_mismatch

    @torch.no_grad()
    def check_gen_consistency(self, model, full_bytes: torch.Tensor, device: str, prompt_len: int = 32,
                               tol: float = 1e-3, log=print, label: str = "") -> int:
        """StackDecoder override (chat 2026-08-20): the base Decoder.check_gen_consistency
        re-derives its own mini generation step via model._run's query_last/h_list[0][:,-1,:]
        fallback, bypassing _stack_generate_blockwise entirely -- it would hit the exact same stale-
        hidden-state bug _stack_generate_blockwise was built to fix (see decode_level's query_last=None
        comment), just inside a diagnostic instead of real generation. Delegates to
        check_blockwise_gen_consistency (code_source='pred', the real generation-time signal)
        instead -- tol is unused (byte-exact comparison, not a logit tolerance). n_levels==2 only
        (matching _stack_generate_blockwise's scope) -- skips rather than crashing otherwise.
        Also needs level0 to actually have an upper track (`stage_lms[0][1]`, i.e. level 2's code)
        -- since the hard-exclusion (2026-08-23) that track no longer exists for a bare 2-level
        model, so this specifically-two-track diagnostic has nothing left to check there either."""
        prefix = f"blockwise_gen_consistency_pred_{label}" if label else "blockwise_gen_consistency_pred"
        if self.n_levels != 2:
            log(f"{prefix}: skipped (StackDecoder's check only covers n_levels==2 so far)")
            return 0
        if len(self.stage_lms[0]) < 2:
            log(f"{prefix}: skipped (level0 has no upper track since the hard-exclusion -- "
                f"nothing this two-track check can compare)")
            return 0
        return self.check_blockwise_gen_consistency(model, full_bytes, device, prompt_len=prompt_len,
                                                      code_source="pred", log=log, label=label)

    @torch.no_grad()
    def check_roundtrip_consistency(self, model, full_bytes: torch.Tensor, device: str, log=print,
                                     label: str = "") -> int:
        """StackDecoder counterpart to StackDecoderV1's version (same diagnostic intent: decode
        purely from real own-code, teacher-forced K bytes autoregressively per block, re-encode,
        compare against the real code that produced it) -- built on _stack_generate_blockwise(code_source
        ='gt') with an EMPTY prompt (predict every block from scratch, real code only) instead of
        StackDecoderV1's own bos_interleaved_self_attn/own_block_cross_attn_decode calls directly.
        n_levels==2 only, matching _stack_generate_blockwise's own scope."""
        prefix = f"roundtrip_{label}" if label else "roundtrip"
        if self.n_levels != 2:
            log(f"{prefix}: skipped (StackDecoder's check only covers n_levels==2 so far)")
            return 0
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        K = model.cfg.Ks[0]
        n_blocks = full_bytes.shape[1] // K
        if n_blocks < 1:
            log(f"{prefix}: skipped (sequence shorter than one block)")
            if was_training:
                model.train()
            return 0

        out = self._stack_generate_blockwise(model, full_bytes[:, :0], n_blocks * K, device,
                                        code_source="gt", gt_full_bytes=full_bytes)
        predicted = out["bytes"]
        enc0 = model.encoders[0]
        real_code = enc0(full_bytes[:, :n_blocks * K], level=0, window=model.windows[0],
                          compute_ntp=False)["code"][:, :n_blocks, :]
        reenc_code = enc0(predicted, level=0, window=model.windows[0],
                           compute_ntp=False)["code"][:, :n_blocks, :]
        real_ids = enc0.quant.to_ids(real_code)
        reenc_ids = enc0.quant.to_ids(reenc_code)
        n_mismatch = (real_ids != reenc_ids).sum().item()
        n_total = real_ids.numel()
        acc = (n_total - n_mismatch) / n_total if n_total > 0 else 1.0
        if was_training:
            model.train()
        acc_str = "1.0" if acc >= 1.0 else f"{acc:.2f}".lstrip("0")
        log(f"{prefix}_acc: {acc_str} ({n_total - n_mismatch}/{n_total} blocks round-tripped to "
            f"the same code after decode->re-encode -- real ground-truth codes, no upper-level "
            f"sampling, baseline noise floor, diagnostic only)")
        return n_mismatch

    def _decode_gt_context(self, model, full_bytes: torch.Tensor, prompt_len: int, n_new_bytes: int,
                            device: str) -> torch.Tensor:
        """True code-quality upper bound (chat 2026-08-20, fixes check_decode_modes's gt mode):
        decode blocks [prompt_len//K, n_blocks) from REAL ground-truth own-code AND real
        ground-truth self-attention context throughout (context is never overwritten with the
        model's own predictions), same pattern as StackDecoderV1's decode_from -- causal, so real
        values sitting at not-yet-predicted positions never leak (see check_roundtrip_consistency).
        Unlike _stack_generate_blockwise(code_source='gt'), errors at one position can't compound into
        the next, isolating decode quality from autoregressive context drift."""
        cfg = self.cfg
        K = cfg.Ks[0]
        bb0, bb1 = self.stage_lms[0][0], self.stage_lms[0][1]
        cum_K1 = K * cfg.Ks[1]
        track0_self_w, track0_cross_w = split_track0_window(model.decode_windows[0][0])
        D = bb0.d_model
        n_blocks = (prompt_len + n_new_bytes) // K
        context = full_bytes[:, :n_blocks * K]
        real_code = model.encoders[0](context, level=0, window=model.windows[0], compute_ntp=False)["code"]
        code_embeds0 = bb0.quant.embed_for_decode(bb0, real_code)
        code_embeds0 = self.track0_kv_lms[0](code_embeds0)
        code1 = model.encoders[1](real_code, level=1, window=model.windows[1], compute_ntp=False)["code"]
        code_embeds1 = bb1.quant.embed_for_decode(bb1, code1)

        pos_start_block = prompt_len // K
        scored = context.clone()
        for t in range(K):
            x0 = bb0.embed_input(context, True)
            pass1 = encode_like_self_attn_decode(bb0, x0, code_embeds0, K, track0_self_w, track0_cross_w)
            h_seed = seed_query_decode(bb0, pass1["saved_k"], pass1["saved_v"], code_embeds0, n_blocks, K,
                                        track0_self_w, track0_cross_w)
            h_real_b = pass1["hidden"].view(context.shape[0], n_blocks, K, D)
            h_query = h_seed if t == 0 else h_real_b[:, :, t - 1, :]

            new_bytes = context.new_zeros(context.shape[0], n_blocks)
            for b in range(pos_start_block, n_blocks):
                pos = b * K + t
                h_final = upper_track_step(bb1, h_query[:, b:b + 1, :], pos, code_embeds1, cum_K1)
                new_bytes[:, b] = sample_next_byte(bb1.byte_output_weight, h_final[:, 0, :])
            scored = scored.view(scored.shape[0], n_blocks, K).clone()
            scored[:, pos_start_block:, t] = new_bytes[:, pos_start_block:]
            scored = scored.view(scored.shape[0], n_blocks * K)
        return scored

    @torch.no_grad()
    def check_decode_modes(self, model, full_bytes: torch.Tensor, device: str, log=print,
                            label: str = "") -> dict:
        """StackDecoder counterpart to StackDecoderV1's version: decode the same span twice --
        once from real ground-truth own-code with real ground-truth context (gt mode, true
        code-quality upper bound, via _decode_gt_context) and once via real autoregressive
        generation from level1's own sampled code predictions (pred mode, the real
        generation-time signal, via _stack_generate_blockwise) -- report byte accuracy against ground
        truth for each. n_levels==2 only, and needs level0 to have an upper track (`stage_lms[0][1]`,
        level 2's code) -- gone for a bare 2-level model since the hard-exclusion (2026-08-23)."""
        prefix = f"decode_modes_{label}" if label else "decode_modes"
        if self.n_levels != 2:
            log(f"{prefix}: skipped (StackDecoder's check only covers n_levels==2 so far)")
            return {}
        if len(self.stage_lms[0]) < 2:
            log(f"{prefix}: skipped (level0 has no upper track since the hard-exclusion -- "
                f"nothing this two-track check can compare)")
            return {}
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        K = model.cfg.Ks[0]
        n_blocks = full_bytes.shape[1] // K
        if n_blocks < 2:
            log(f"{prefix}: skipped (need >=2 blocks for level1 to have a real prediction)")
            if was_training:
                model.train()
            return {}

        prompt_len = K
        n_new = (n_blocks - 1) * K
        gt_bytes = self._decode_gt_context(model, full_bytes, prompt_len, n_new, device)
        pred_out = self._stack_generate_blockwise(model, full_bytes[:, :prompt_len], n_new, device,
                                             code_source="pred")
        target = full_bytes[0, prompt_len:prompt_len + n_new]
        gt_acc = (gt_bytes[0, prompt_len:prompt_len + n_new] == target).float().mean().item()
        pred_acc = (pred_out["bytes"][0, prompt_len:prompt_len + n_new] == target).float().mean().item()
        if was_training:
            model.train()

        def fmt(a):
            return "1.0" if a >= 1.0 else f"{a:.2f}".lstrip("0")
        log(f"{prefix}: gt_byte_acc={fmt(gt_acc)}  pred_byte_acc={fmt(pred_acc)}  "
            f"(gt=decode from real ground-truth code with real ground-truth context, true upper "
            f"bound; pred=real autoregressive decode from level1's own sampled code prediction, "
            f"the real generation-time signal)")
        return {"gt_byte_acc": gt_acc, "pred_byte_acc": pred_acc}


def block_local_track0_decode(bb: LM, x_list_i: torch.Tensor, code_embeds0: torch.Tensor, K: int,
                               n_blocks: int, is_byte_level: bool) -> dict:
    """Variant B ("cross-attention-only conditioning", chat 2026-08-20): block-diagonal same-level
    self-attention -- every block decodes independently of every OTHER block at the same level
    (still genuinely autoregressive WITHIN a block: byte t sees bytes 0..t-1 of its OWN block), so
    all n_blocks blocks are one parallel batched call, not n_blocks sequential steps.

    This can't be expressed as a smaller value of the existing SWA window (decode_windows[i][0]):
    SWA masks by raw position difference, and a block-boundary query's window always counts
    backward across the boundary by construction (window=w>0 at seed position b*K always includes
    position b*K-1, the previous block's last byte -- verified directly, see chat). Block-diagonal
    is a different mask shape entirely, implemented here the cheap way: fold n_blocks into the
    batch dimension and run ordinary causal self-attention over each K-length block independently,
    rather than masking a shared L-length sequence -- correct AND avoids ever materializing an L x L
    (or even chunked) mask for something that's actually block-local.

    Consequence: the seed token's self-attention contribution is PROVABLY always zero under
    same-block + causal (its query position is the block's own start; a same-block causal key would
    need position < block start while sharing the block's own index -- impossible for any K >= 1),
    so unlike seed_query_decode there is no K/V cache/self-attn call for the seed at all here --
    it's pure cross-attention to its own code, chained through block.forward_cross per layer with
    no separate self-attention step (the residual it would add is exactly zero).

    Positions are entirely LOCAL (0..K-1 within a block, code at a fixed reference position) since
    blocks never interact -- no global/cross-block RoPE position is needed or meaningful anymore.

    Returns the same shape contract as _track0: dict(h0=..., h0_shifted=..., h_seed=...) (loss/acc
    computed by the caller via encode_like_decode_loss, same as the default track0)."""
    cfg = bb.cfg
    D = bb.d_model
    B = x_list_i.shape[0]
    device = x_list_i.device
    H, hd = cfg.n_heads, D // cfg.n_heads
    L_used = n_blocks * K

    x0 = bb.embed_input(x_list_i[:, :L_used], is_byte_level)
    xb = x0.view(B * n_blocks, K, D)
    code_b = code_embeds0.reshape(B * n_blocks, 1, D)

    pos = torch.arange(K, device=device)
    cos, sin = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device)
    code_pos = torch.zeros(1, device=device)
    cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)

    x = xb
    for block in bb.blocks:
        xn = block.ln1(x)
        qkv = block.attn.qkv(xn).reshape(B * n_blocks, K, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = block.attn.out(y.transpose(1, 2).reshape(B * n_blocks, K, D))
        x = x + a
        x = block.forward_cross(x, code_b, cos, sin, cos_k, sin_k, attn_mask=None)
    h_real = bb.ln_f(x).view(B, n_blocks, K, D)

    x_seed = bb.self_code_const.view(1, 1, D).expand(B * n_blocks, 1, D)
    for block in bb.blocks:
        x_seed = block.forward_cross(x_seed, code_b, cos_k, sin_k, cos_k, sin_k, attn_mask=None)
    h_seed = bb.ln_f(x_seed).view(B, n_blocks, D)

    return make_dict(h_real=h_real, h_seed=h_seed)


class StackDecoderLocal(StackDecoder):
    """Variant B, plugged into StackDecoder's existing decode_level/__init__/upper-track chain
    via the _track0 override point (see StackDecoder._track0's docstring). Same module structure
    as StackDecoder (own-code stage + per-track cross-attn-stage LMs, cond_depth etc. all
    unchanged) -- only track0's same-level conditioning differs: block_local_track0_decode instead
    of encode_like_self_attn_decode/seed_query_decode. decode_windows[i][0] (the byte-level window)
    is ignored for this variant -- there is no tunable window, visibility is always exactly "this
    block only," by construction (see chat 2026-08-20's SWA-can't-express-this argument)."""

    def _track0(self, bb0, x_list_i, code_embeds0, K, n_blocks, track0_window, is_byte_level, compute_ntp):
        D = bb0.d_model
        out = block_local_track0_decode(bb0, x_list_i, code_embeds0, K, n_blocks, is_byte_level)
        h_real_b, h_seed = out["h_real"], out["h_seed"]

        B = x_list_i.shape[0]
        L_used = n_blocks * K
        h0 = torch.cat([h_seed.unsqueeze(2), h_real_b[:, :, :K - 1, :]], dim=2).reshape(B, L_used, D)
        if compute_ntp:
            # h_real_b plays the same role pass1["hidden"] does in the default _track0 -- same
            # UNSHIFTED-within-block semantics, just block-batched instead of one L-length sequence.
            pass1_hidden = h_real_b.reshape(B, L_used, D)
            loss0, acc0 = encode_like_decode_loss(bb0, pass1_hidden, h_seed, x_list_i[:, :L_used], K, is_byte_level)
        else:
            loss0, acc0 = h0.new_zeros(()), h0.new_zeros(())

        if n_blocks >= 2:
            next_block_seed = torch.cat([h_seed[:, 1:, :], h_seed[:, -1:, :]], dim=1)
        else:
            next_block_seed = h_seed
        h0_shifted = torch.cat([h_real_b[:, :, :K - 1, :], next_block_seed.unsqueeze(2)],
                                dim=2).reshape(B, L_used, D)
        return h0, h0_shifted, loss0, acc0

    @torch.no_grad()
    def _stack_generate_blockwise(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                             code_source: str = "pred", gt_full_bytes: torch.Tensor | None = None,
                             max_srcs: int | None = None) -> dict:
        """Override of StackDecoder._stack_generate_blockwise (2026-08-23 bugfix): the base version
        hardcodes encode_like_self_attn_decode/seed_query_decode for track0's byte-by-byte
        generation, which lets the seed token's self-attention reach K/V from EVERY prior block --
        but this decoder was trained with block_local_track0_decode, whose own docstring proves
        the seed's same-block self-attention contribution is ALWAYS EXACTLY ZERO (own-block causal
        self-attn can never see a same-block key from before the block's own start). Generation was
        therefore feeding the model real cross-block self-attention signal it never learned to
        produce or consume -- a genuine train/inference mismatch (found via qual sample generation
        collapsing to repetitive garbage at ~99.5% teacher-forced byte_acc, 2026-08-23), not a
        capacity/architecture limitation. Fix: every new block's track0 hidden state is computed by
        calling block_local_track0_decode on JUST that one block (n_blocks=1) -- identical
        computation to training, zero visibility into any other block, matching _track0 above
        exactly. Everything else (code sourcing, upper-track chaining) is unchanged from the base
        class; only the inner byte loop's track0 call differs."""
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        cfg = self.cfg
        K = cfg.Ks[0]
        stage_bbs = self.stage_lms[0]
        bb0 = stage_bbs[0]
        n_upper_avail = len(stage_bbs) - 1
        n_upper = n_upper_avail if max_srcs is None else max(0, min(n_upper_avail, max_srcs - 1))
        all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]

        n_blocks_prompt = all_bytes.shape[1] // K
        code_parts = [model.encoders[0](all_bytes, level=0, window=model.windows[0],
                                         compute_ntp=False)["code"][:, :n_blocks_prompt, :]] if n_blocks_prompt > 0 else []

        cum_Ks = []
        cum = K
        for j in range(1, n_upper + 1):
            cum *= cfg.Ks[j]
            cum_Ks += [cum]

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            n_blocks_prev = all_bytes.shape[1] // K
            if code_source == "gt":
                gt = gt_full_bytes.to(device)
                if gt.dim() == 1:
                    gt = gt.unsqueeze(0)
                real_code = model.encoders[0](gt[:, :(n_blocks_prev + 1) * K], level=0,
                                               window=model.windows[0], compute_ntp=False)["code"]
                next_code = real_code[:, n_blocks_prev:n_blocks_prev + 1, :]
            elif code_source == "pred":
                codes = encode_up_to(model, all_bytes, level=1)
                enc1 = model.encoders[1]
                out1 = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
                sampled = enc1.quant.sample_next(enc1.lm, out1["hidden"][:, -1, :], cfg.vocab)
                next_code = sampled.unsqueeze(1)
            else:
                raise ValueError(f"code_source must be 'gt' or 'pred', got {code_source!r}")
            code_parts += [next_code]
            code0 = torch.cat(code_parts, dim=1)
            code_embeds0 = bb0.quant.embed_for_decode(bb0, code0)
            code_embeds0 = self.track0_kv_lms[0](code_embeds0)
            own_code_embed = code_embeds0[:, -1:, :]  # just this new block's own code

            # Each active upper track j's OWN code over the level-(j-1) code stream -- never
            # circular, a plain function of the level below, unlike track0's own code.
            upper_code_embeds = []
            seq_repr = code0
            for t, j in enumerate(range(1, n_upper + 1)):
                seq_repr = model.encoders[j](seq_repr, level=j, window=model.windows[j], compute_ntp=False)["code"]
                bb_j = stage_bbs[t + 1]
                code_embeds_j = bb_j.quant.embed_for_decode(bb_j, seq_repr)
                upper_code_embeds += [self.kv_lms[0][t](code_embeds_j)]

            buf = torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], K)], dim=1)
            for t in range(K):
                cur_block_bytes = buf[:, -K:]  # this new block only -- never any other block's bytes
                out = block_local_track0_decode(bb0, cur_block_bytes, own_code_embed, K, 1, True)
                h_query = out["h_seed"][:, 0, :] if t == 0 else out["h_real"][:, 0, t - 1, :]

                pos = n_blocks_prev * K + t
                h_final = h_query.unsqueeze(1)
                for bb_j, code_embeds_j, cum_Kj in zip(stage_bbs[1:n_upper + 1], upper_code_embeds, cum_Ks):
                    h_final = upper_track_step(bb_j, h_final, pos, code_embeds_j, cum_Kj)
                embed_weight_final = stage_bbs[n_upper].byte_output_weight
                next_byte = sample_next_byte(embed_weight_final, h_final[:, 0, :])
                buf = buf.clone()
                buf[:, pos] = next_byte
            all_bytes = buf

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            model.train()
        return make_dict(bytes=all_bytes, code_used=torch.cat(code_parts, dim=1))


class StackDecoderSync(StackDecoder):
    """Variant C, PLANNED / NOT IMPLEMENTED (chat 2026-08-20) -- descriptive enough to resume from.
    Instantiable (inherits StackDecoder's __init__/module structure unchanged) but decode_level
    raises NotImplementedError; do not select --decoder_type stack_sync for real training yet.

    Idea (synchronized wavefront, generalizes bos_query_only_parallel_sync_decode's stub above from
    a generation-time design note into an actual trainable decode_level): decode ACROSS blocks in
    lockstep by within-block offset t, instead of block-by-block (StackDecoder default) or fully
    block-parallel with zero cross-block visibility (StackDecoderLocal / Variant B). At wave
    t=0, every block's seed query (cross-attending to its own code only, same as Variant B) produces
    every block's first byte IN PARALLEL. At wave t=1, every block's SECOND-byte query can now
    additionally see every OTHER block's byte at offset 0 (just produced, same wave-boundary as
    itself) -- and generally, wave t's queries see every block's bytes at offsets < t, own block's
    included, growing monotonically with t. K sequential waves total (not n_blocks, and not the
    fully-sequential n_blocks*K of the default) -- parallelism scales with n_blocks, not K, unlike
    Variant A's tunable-but-bounded window.

    Masking is the hard, NOT-YET-BUILT part, and needs new logic distinct from BOTH the plain SWA
    window (position-difference based) AND Variant B's block-diagonal mask (block-index based):
    visibility here depends on comparing WITHIN-BLOCK OFFSET (query's t vs key's t'), not raw
    position or block index -- reshape the (n_blocks, K) grid and mask query offset t against key
    offset t' < t, for every block pair (own block included, since own-block bytes at offset < t are
    exactly what the current sequential design already grants via normal causal self-attention).
    Chat's conditioning-expressivity analysis (2026-08-20) is directly relevant before building
    this: at t=0 (predicting every block's first, boundary-crossing byte -- arguably the single
    hardest position for any parallel variant), C's visibility is IDENTICAL to Variant B's (zero
    cross-block info, nothing has been produced in any wave yet) -- C's advantage over B is entirely
    concentrated in t>=1 (later within-block bytes), where the block's own causal chain is already
    doing a lot of the work. Measure Variant B (and ideally Variant A at a few window settings)
    FIRST; only build this if the data shows late-in-block bytes, not boundary bytes, are the
    dominant remaining error source -- otherwise this is a lot of new masking machinery for a gain
    concentrated where the model already does comparatively well."""

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_srcs, want_next_query):
        raise NotImplementedError(
            "StackDecoderSync (Variant C, synchronized wavefront) -- design note only, see class docstring")


def make_decoder(cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs) -> Decoder:
    if cfg.decoder_type == "stack_v1":
        return StackDecoderV1(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    if cfg.decoder_type == "stack":
        return StackDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    if cfg.decoder_type == "stack_local":
        return StackDecoderLocal(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    if cfg.decoder_type == "stack_sync":
        return StackDecoderSync(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    return ConcatDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
