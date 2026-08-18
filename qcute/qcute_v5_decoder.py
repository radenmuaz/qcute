import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qcute.qcute_v5_common import (
    LM, Config, apply_rope, make_dict, pack_words, rope_cos_sin_for_positions, warn_thin_window,
)


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
            log(f"{prefix}level0_mode{tag}:      {gen_bytes_m!r}")
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
            source_c = decode_derived_c[j] if (j > i and j in decode_derived_c) else c_list[j]
            code_embeds = bb.quant.embed_for_decode(bb, source_c)
            tracks += [(code_embeds, cum_K, window)]
        if not tracks:
            return None
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
    def __init__(self, cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs):
        super().__init__(cfg, n_levels)

        def make_stage(i, t):
            if t == 0:
                bb = encoders[i].lm if cfg.share_encode_decode_self else LM(cfg, d_models[i], n_layers_list[i], vocabs[i])
            else:
                cross_layers = cfg.decode_cross_stage_layers if cfg.decode_cross_stage_layers is not None else n_layers_list[i]
                bb = LM(cfg, d_models[i], cross_layers, vocabs[i])
            if not hasattr(bb, "merged_cache"):
                bb.merged_cache = {}
            return bb

        self.stage_lms = nn.ModuleList([
            nn.ModuleList([make_stage(i, t) for t in range(n_levels - i)])
            for i in range(n_levels)
        ])

    def decode_level(self, model, i, x_list, c_list, decode_derived_c, compute_ntp, max_decode_sources, want_next_query):
        cfg = self.cfg
        L_i = x_list[i].shape[1]
        track_specs = []
        cum_K = 1
        for j in range(i, self.n_levels):
            cum_K *= cfg.Ks[j]
            window = model.decode_windows[i][j - i]
            if window == 0:
                continue
            if L_i // cum_K < 1:
                break
            source_c = decode_derived_c[j] if (j > i and j in decode_derived_c) else c_list[j]
            track_specs += [(source_c, cum_K, window)]
        if not track_specs:
            return None
        full_track_specs = track_specs[:max_decode_sources] if max_decode_sources is not None else track_specs

        is_byte_level = i == 0
        K = cfg.Ks[i]
        D = self.stage_lms[i][0].d_model
        stage_bbs = self.stage_lms[i]
        x = None
        extra_losses = []
        loss_final = acc_final = code_final = query_last_final = embed_weight_final = None
        for t, (source_c, track_K, window) in enumerate(full_track_specs):
            bb = stage_bbs[t]
            code_embeds = bb.quant.embed_for_decode(bb, source_c)
            is_last = t == len(full_track_specs) - 1
            if t == 0:
                x0 = bb.embed_input(x_list[i], is_byte_level)
                h, query_last = merged_decode_forward(bb, x0, [(code_embeds, track_K, window)], extra_query=True)
                if compute_ntp:
                    h_flat = h[:, K - 1:-1, :].reshape(-1, D)
                    loss_stage, acc_stage = bb.ntp_loss_acc(h_flat, x_list[i][:, K:], is_byte_level)
                else:
                    loss_stage, acc_stage = h.new_zeros(()), h.new_zeros(())
                code_stage = bb.extract_code(h, x0, K, window)["code"] if is_last else None
            else:
                stage_result = cross_attn_stage(bb, x, code_embeds, x_list[i], i, track_K, window, compute_ntp, is_last)
                h, query_last = stage_result["hidden"], stage_result["query_last"]
                loss_stage, acc_stage, code_stage = stage_result["loss"], stage_result["acc"], stage_result["code"]
            x = h
            if is_last:
                loss_final, acc_final, code_final, query_last_final = loss_stage, acc_stage, code_stage, query_last
                embed_weight_final = bb.embed.weight
            else:
                extra_losses += [loss_stage]

        valid_next_query = want_next_query and i == 0 and L_i % full_track_specs[-1][1] == 0
        return make_dict(hidden=x, query_last=(query_last_final if valid_next_query else None),
                          loss=loss_final, acc=acc_final, code=code_final,
                          extra_losses=extra_losses, embed_weight=embed_weight_final)


def make_decoder(cfg: Config, n_levels: int, encoders, d_models, n_layers_list, vocabs) -> Decoder:
    if cfg.decoder_type == "stack":
        return StackDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
    return ConcatDecoder(cfg, n_levels, encoders, d_models, n_layers_list, vocabs)
