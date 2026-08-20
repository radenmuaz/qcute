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


def bos_interleaved_self_attn(bb: LM, x0: torch.Tensor, K: int, window: int | None) -> torch.Tensor:
    """qcute_v1 non-top-level decode self-attention (see docs/qcute_v1_plan.md): prepends
    bb.self_code_const (BOS) before every K-block -- [BOS,x0,x1] [BOS,x2,x3] ... -- then runs
    plain causal self-attention over the augmented sequence, RoPE'd by augmented-sequence
    position. window=None: sync (unbounded, one continuous causal chain across every block,
    today's default). window=K+1: async ablation (this block's BOS+K only, independently
    schedulable per block, no cross-block visibility) -- any other finite value is interpreted
    directly in augmented-sequence units. Returns h stripped back to the original K*n_blocks raw
    positions (BOS positions dropped from the output, not from the attention computation)."""
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
    return he[:, :, 1:, :].reshape(B, n_blocks * K, D)


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
                      compute_ntp: bool, max_decode_sources, want_next_query: bool):
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
            next_byte = sample_next_byte(enc0.embed.weight, out["hidden"][:, -1, :])
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
        if was_training:
            model.train()
        return all_bytes[0]

    @torch.no_grad()
    def generate_no_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_decode_sources: int | None = None) -> torch.Tensor:
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        K0 = model.cfg.Ks[0]
        all_bytes = prompt_bytes
        for _ in range(n_new_bytes):
            L = all_bytes.shape[1]
            block_aligned = L % K0 == 0
            result = model._run(all_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
                                 want_next_query=block_aligned)
            embed_w = result["embed_weights"][0] if result["embed_weights"][0] is not None else model.encoders[0].embed.weight
            query = result["next_query"][0] if result["next_query"][0] is not None else result["h_list"][0][:, -1, :]
            next_byte = sample_next_byte(embed_w, query)
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
        if was_training:
            model.train()
        return all_bytes[0]

    @torch.no_grad()
    def generate_kv_cache(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                           max_decode_sources: int | None = None) -> torch.Tensor:
        was_training = model.training
        model.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        K0 = model.cfg.Ks[0]
        context_len = model.cfg.context_len
        all_bytes = prompt_bytes
        for _ in range(n_new_bytes):
            window_bytes = all_bytes[:, -context_len:]
            L = window_bytes.shape[1]
            block_aligned = L % K0 == 0
            result = model._run(window_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
                                 want_next_query=block_aligned)
            embed_w = result["embed_weights"][0] if result["embed_weights"][0] is not None else model.encoders[0].embed.weight
            query = result["next_query"][0] if result["next_query"][0] is not None else result["h_list"][0][:, -1, :]
            next_byte = sample_next_byte(embed_w, query)
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
        if was_training:
            model.train()
        return all_bytes[0]

    def validate_generation(self, model, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
        out_a = self.generate_no_cache(model, prompt_bytes, n_new_bytes, device)
        out_b = self.generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
        assert torch.equal(out_a, out_b), "generate_no_cache and generate_kv_cache diverged"
        return True

    @torch.no_grad()
    def check_gen_consistency(self, model, full_bytes: torch.Tensor, device: str, prompt_len: int = 32,
                               tol: float = 1e-3, log=print, label: str = "") -> int:
        was_training = model.training
        model.eval()
        full_bytes = full_bytes.to(device)
        if full_bytes.dim() == 1:
            full_bytes = full_bytes.unsqueeze(0)
        L_total = full_bytes.shape[1]

        result_tf = model._run(full_bytes, compute_ntp=False, max_decode_sources=None, want_next_query=False)
        embed0 = result_tf["embed_weights"][0] if result_tf["embed_weights"][0] is not None else model.encoders[0].embed.weight
        logits_tf_all = F.linear(result_tf["h_list"][0][0], embed0)

        n_mismatch = 0
        for t in range(prompt_len, L_total - 1):
            ref_idx = t - 1
            if ref_idx < 0 or ref_idx >= logits_tf_all.shape[0]:
                continue
            padded = full_bytes[:, :t]
            result_gen = model._run(padded, compute_ntp=False, max_decode_sources=None, want_next_query=True)
            embed_gen = result_gen["embed_weights"][0] if result_gen["embed_weights"][0] is not None else model.encoders[0].embed.weight
            query_gen = result_gen["next_query"][0] if result_gen["next_query"][0] is not None else result_gen["h_list"][0][:, -1, :]
            logits_gen = F.linear(query_gen[0], embed_gen)
            if (logits_gen - logits_tf_all[ref_idx]).abs().max().item() >= tol:
                n_mismatch += 1
        if was_training:
            model.train()
        prefix = f"gen_consistency_{label}" if label else "gen_consistency"
        log(f"{prefix}: {n_mismatch}/{L_total - 1 - prompt_len} timesteps mismatched "
            f"(generation vs teacher-forced logits on ground-truth input)")
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
        StackDecoder-specific (assumes the BOS-interleaved self-attn + cross-attn-to-own-code
        non-top-level structure); no-op (prints and returns 0) for n_levels==1, where level0 is
        the (structurally unrelated, unchanged-from-v5) top level."""
        if model.n_levels < 2:
            log(f"roundtrip{'_' + label if label else ''}: skipped (n_levels==1, level0 is top, "
                f"not the BOS-interleaved decode this check targets)")
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
            h_self = bos_interleaved_self_attn(bb_self, x0, K, self_window)
            stage_result = cross_attn_stage(bb_cross, h_self, code_embeds, predicted, 0, K,
                                              own_code_window, compute_ntp=False, want_code=False)
            h = stage_result["hidden"].view(predicted.shape[0], n_blocks, K, -1)
            next_byte = sample_next_byte(bb_cross.embed.weight, h[:, :, t, :])
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
                h_self = bos_interleaved_self_attn(bb_self, x0, K, self_window)
                stage_result = cross_attn_stage(bb_cross, h_self, code_embeds, buf, 0, K,
                                                  own_code_window, compute_ntp=False, want_code=False)
                h = stage_result["hidden"].view(B, n_cmp, K, -1)
                next_byte = sample_next_byte(bb_cross.embed.weight, h[:, :, t, :])
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
        for m in range(1, model.n_levels + 1):
            out_m = self.generate_no_cache(model, prompt_bytes, gen_len, device, max_decode_sources=m)
            gen_bytes_m = pack_words(out_m[prompt_bytes.numel():].tolist(), bits)
            tag = "full" if m == model.n_levels else str(m)
            pad = " " * 5 if tag == "full" else " " * 6
            log(f"{prefix}level0_mode{tag}:{pad}{gen_bytes_m!r}")
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

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_decode_sources, want_next_query):
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
        full_tracks = tracks[:max_decode_sources] if max_decode_sources is not None else tracks

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
                          extra_losses=[], embed_weight=bb.embed.weight)


class StackDecoder(Decoder):
    """qcute_v1: only the top level is a genuine NTP/AR decoder (self-code recurrence,
    unconditional -- self_code_active/use_self_code no longer gate this, superseded). Every level
    below top decodes as: BOS-interleaved causal self-attention over its own actual sequence
    (bos_interleaved_self_attn, self_code_const repurposed as the per-block BOS) THEN
    cross-attention to that SAME block's own-level code (cross_attn_stage, own code -- not a
    coarser level's), predicting shift-by-1 NTP over every position. See docs/qcute_v1_plan.md's
    worked example. `decode_windows[i][0]` is the self-attention (BOS) window (None=sync/default,
    K+1=async ablation); `decode_windows[i][1]` is the own-code cross-attention window (how many
    codes back are visible -- 1 = only this block's own code, as in the base worked example).
    n_levels>=3 (additional coarser-than-own-code cross-attention) not yet generalized here."""

    def __init__(self, cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs):
        super().__init__(cfg, n_levels)

        def make_self_stage(i):
            # share_encode_decode_self: reuse encode's own LM (embed, blocks, self_code_const --
            # everything except decode's separate cross-attention stage) for decode's self-attn
            # stage, for BOTH top (unchanged v5 behavior) and non-top levels (bb_self only --
            # bb_cross always stays a separate LM regardless of this flag).
            bb = encoders[i].lm if cfg.share_encode_decode_self else LM(cfg, d_models[i], n_layers_list[i], vocabs[i])
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

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_decode_sources, want_next_query):
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
                              loss=loss, acc=acc, code=code, extra_losses=[], embed_weight=bb.embed.weight)

        n_blocks = L_i // K
        if n_blocks < 1:
            return None
        bb_self, bb_cross = self.stage_lms[i][0], self.stage_lms[i][1]
        D = bb_self.d_model
        self_window = model.decode_windows[i][0]
        own_code_window = model.decode_windows[i][1] if len(model.decode_windows[i]) > 1 else None

        x0 = bb_self.embed_input(x_list[i], is_byte_level)
        h_self = bos_interleaved_self_attn(bb_self, x0, K, self_window)

        source_c = c_list[i][:, :n_blocks, :]
        own_code_embeds = bb_cross.quant.embed_for_decode(bb_cross, source_c)
        stage_result = cross_attn_stage(bb_cross, h_self, own_code_embeds, x_list[i], i, K,
                                          own_code_window, compute_ntp=False, want_code=False)
        h = stage_result["hidden"]

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            loss, acc = bb_cross.ntp_loss_acc(h_flat, x_list[i][:, 1:], is_byte_level)
        else:
            loss, acc = h.new_zeros(()), h.new_zeros(())
        return make_dict(hidden=h, query_last=None, loss=loss, acc=acc, code=None,
                          extra_losses=[], embed_weight=bb_cross.embed.weight)


def make_decoder(cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs) -> Decoder:
    if cfg.decoder_type == "stack":
        return StackDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    return ConcatDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
