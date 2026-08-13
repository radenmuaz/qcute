import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.text_path = run_dir / "run.log"
        self.json_path = run_dir / "run.jsonl"
        self.text_f = open(self.text_path, "a")
        self.json_f = open(self.json_path, "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        elapsed_hms = format_hms(elapsed_s)
        line = f"[{elapsed_hms}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        json_record = {"elapsed_s": elapsed_s, "elapsed_hms": elapsed_hms, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(json_record) + "\n")
        self.json_f.flush()


class Checkpointer:
    def __init__(self, run_dir: Path, save_every_n_evals: int = 1, minimize: bool = True):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = run_dir / "best.pt"
        self.last_path = run_dir / "last.pt"
        self.save_every_n_evals = max(1, save_every_n_evals)
        self.minimize = minimize
        self.best_metric = float("inf") if minimize else float("-inf")
        self._eval_count = 0

    def is_better(self, metric: float) -> bool:
        return metric < self.best_metric if self.minimize else metric > self.best_metric

    def step(self, state: dict, metric: float) -> None:
        self._eval_count += 1
        if self.is_better(metric):
            self.best_metric = metric
            torch.save(state, self.best_path)
        if self._eval_count % self.save_every_n_evals == 0:
            torch.save(state, self.last_path)


@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32)
    d_model: int = 256
    n_layers: int = 2
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int | tuple[int, ...] = 32
    rope_base: float = 10000.0
    byte_ntp_weight: float = 1.0
    code_ntp_weight: float = 1.0
    decode_ntp_weight: float = 1.0
    gumbel_tau: float = 1.0
    use_gumbel_noise: bool = False
    vocab: int = 256
    code_extract_mode: str = "last_h"  # "last_h" | "softmax_pool" | "light_query_attn" | "query_embed"
    code_head_tied: bool = False


def gumbel_quantize(logits: torch.Tensor, tau: float, use_gumbel_noise: bool = False) -> torch.Tensor:
    if use_gumbel_noise:
        eps = torch.finfo(logits.dtype).tiny
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(u))
        soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    else:
        soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(soft.argmax(-1), num_classes=logits.shape[-1]).to(soft.dtype)
    return soft + (hard - soft).detach()


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def jagged_causal_mask_and_positions(L: int, n_blocks: int, K: int, device: torch.device, kv_window: int | None = None):
    t_idx = torch.arange(L, device=device).unsqueeze(1)
    b_idx = torch.arange(n_blocks, device=device).unsqueeze(0)
    n_complete = t_idx // K
    visible = b_idx < n_complete
    if kv_window is not None:
        visible = visible & (b_idx >= n_complete - kv_window)
    block_pos = (torch.arange(n_blocks, device=device) + 1) * K - 1
    disallow = ~visible
    return disallow, block_pos


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._warned_dense_fallback = False
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _decode_kv_proj(self, decode_kv: torch.Tensor, decode_rope_k) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nf, D = decode_kv.shape
        H, hd = self.n_heads, self.head_dim
        qkv_f = self.qkv(decode_kv).reshape(B, Nf, 3, H, hd).permute(2, 0, 3, 1, 4)
        fk, fv = qkv_f[1], qkv_f[2]
        if decode_rope_k is not None:
            fk = apply_rope(fk, *decode_rope_k)
        return fk, fv

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None,
                decode_kv: torch.Tensor | None = None, decode_disallow: torch.Tensor | None = None,
                decode_rope_k=None, decode_K: int | None = None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None and T % window == 0 and T > window:
            if decode_kv is not None and decode_K == 1 and decode_kv.size(1) == T:
                decode_kv_shifted = torch.cat([torch.zeros_like(decode_kv[:, :1, :]), decode_kv[:, :-1, :]], dim=1)
                fk, fv = self._decode_kv_proj(decode_kv_shifted, None)
                fk = apply_rope(fk, cos, sin)
                y = self._forward_chunked_aligned_decode(q, k, v, fk, fv, window)
            else:
                y = self._forward_chunked(q, k, v, window, decode_kv, decode_disallow, decode_rope_k)
        else:
            if window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window -- falling back to DENSE attention for this layer.")
                self._warned_dense_fallback = True
            if decode_kv is not None:
                fk, fv = self._decode_kv_proj(decode_kv, decode_rope_k)
                k_full = torch.cat([k, fk], dim=2)
                v_full = torch.cat([v, fv], dim=2)
                local_allow = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                attn_mask = torch.cat([local_allow, ~decode_disallow], dim=1)
                y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D)), k, v

    def _chunk_local_window(self, t: torch.Tensor, B: int, H: int, n_chunks: int, W: int, hd: int) -> torch.Tensor:
        tc = t.view(B, H, n_chunks, W, hd)
        zero_chunk = torch.zeros(B, H, 1, W, hd, device=t.device, dtype=t.dtype)
        t_prev = torch.cat([zero_chunk, tc[:, :, :-1]], dim=2)
        t_local = torch.cat([t_prev, tc], dim=3)
        return t_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)

    def _causal_window_mask(self, n_chunks: int, W: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(W, device=device).view(W, 1)
        j_prev = torch.arange(W, device=device).view(1, W) - W
        j_cur = torch.arange(W, device=device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)
        mask_per_chunk = causal_window.unsqueeze(0).expand(n_chunks, W, 2 * W).clone()
        mask_per_chunk[0, :, 0:W] = False
        return mask_per_chunk

    def _forward_chunked_aligned_decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                         fk: torch.Tensor, fv: torch.Tensor, window: int) -> torch.Tensor:
        """decode_K==1 fast path: decode_kv is already raw-position-aligned (one row per
        position, same as k/v), so it gets the SAME rolling local window (previous+current
        chunk, 2*window reach -- the "32+32" rule) via the SAME chunking mechanism as local
        self-attention, instead of _forward_chunked's broadcast-to-every-chunk over the full
        (much larger, block-count-sized) decode_kv tensor."""
        B, H, T, hd = q.shape
        W = window
        n_chunks = T // W

        qb = q.view(B, H, n_chunks, W, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = self._chunk_local_window(k, B, H, n_chunks, W, hd)
        vb = self._chunk_local_window(v, B, H, n_chunks, W, hd)
        fkb = self._chunk_local_window(fk, B, H, n_chunks, W, hd)
        fvb = self._chunk_local_window(fv, B, H, n_chunks, W, hd)

        mask_per_chunk = self._causal_window_mask(n_chunks, W, q.device)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)

        kb_all = torch.cat([kb, fkb], dim=2)
        vb_all = torch.cat([vb, fvb], dim=2)
        mask_all = torch.cat([mask_batched, mask_batched], dim=-1)

        yb = F.scaled_dot_product_attention(qb, kb_all, vb_all, attn_mask=mask_all)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                          decode_kv: torch.Tensor | None, decode_disallow: torch.Tensor | None,
                          decode_rope_k=None) -> torch.Tensor:
        B, H, T, hd = q.shape
        W = window
        n_chunks = T // W

        qb = q.view(B, H, n_chunks, W, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = self._chunk_local_window(k, B, H, n_chunks, W, hd)
        vb = self._chunk_local_window(v, B, H, n_chunks, W, hd)
        mask_per_chunk = self._causal_window_mask(n_chunks, W, q.device)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)

        if decode_kv is not None:
            fk, fv = self._decode_kv_proj(decode_kv, decode_rope_k)
            Nf = fk.size(2)
            fk_b = fk.unsqueeze(2).expand(B, H, n_chunks, Nf, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Nf, hd)
            fv_b = fv.unsqueeze(2).expand(B, H, n_chunks, Nf, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Nf, hd)
            kb = torch.cat([kb, fk_b], dim=2)
            vb = torch.cat([vb, fv_b], dim=2)
            decode_allow_chunked = (~decode_disallow).view(n_chunks, W, Nf)
            decode_mask_batched = decode_allow_chunked.unsqueeze(0).expand(B, n_chunks, W, Nf).reshape(B * n_chunks, 1, W, Nf)
            mask_batched = torch.cat([mask_batched, decode_mask_batched], dim=-1)

        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None,
                decode_kv: torch.Tensor | None = None, decode_disallow: torch.Tensor | None = None,
                decode_rope_k=None, decode_K: int | None = None):
        a, k, v = self.attn(self.ln1(x), cos, sin, window, decode_kv, decode_disallow, decode_rope_k, decode_K)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class LevelLM(nn.Module):
    def __init__(self, cfg: Config, level: int, window: int | None,
                 shared: "LevelLM | None" = None):
        super().__init__()
        self.cfg = cfg
        self.level = level
        self.window = window
        self.is_byte_level = level == 0
        D = cfg.d_model
        V = cfg.vocab
        if shared is not None:
            self.embed = shared.embed
            self.blocks = shared.blocks
            self.ln_f = shared.ln_f
            self.code_head = shared.code_head
            self.code_query = shared.code_query
            self.code_out = shared.code_out
            self.query_embed = shared.query_embed
        else:
            self.embed = nn.Embedding(V, D)
            self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(D)
            self.code_head = None if cfg.code_head_tied else nn.Linear(D, V, bias=False)
            self.code_query = self.code_out = self.query_embed = None
            if cfg.code_extract_mode == "light_query_attn":
                self.code_query = nn.Parameter(torch.zeros(D))
                nn.init.normal_(self.code_query, std=0.02)
                self.code_out = nn.Linear(D, D, bias=False)
            elif cfg.code_extract_mode == "query_embed":
                self.query_embed = nn.Parameter(torch.zeros(D))
                nn.init.normal_(self.query_embed, std=0.02)

    def _classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.code_head(pooled) if self.code_head is not None else F.linear(pooled, self.embed.weight)

    def _query_embed_pool(self, x0: torch.Tensor, K: int, n_blocks: int) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device

        x0_blocks = x0.view(B, n_blocks, K, D)
        q_tok = self.query_embed.view(1, 1, 1, D).expand(B, n_blocks, 1, D)
        xe = torch.cat([x0_blocks, q_tok], dim=2).view(B, n_blocks * (K + 1), D)
        Le = n_blocks * (K + 1)

        slot = torch.arange(K + 1, device=device).repeat(n_blocks)
        is_query = slot == K
        block_of = torch.arange(n_blocks, device=device).repeat_interleave(K + 1)
        within_block_pos = torch.where(is_query, torch.full_like(slot, K - 1), slot)
        real_pos = block_of * K + within_block_pos

        cos, sin = rope_cos_sin_for_positions(real_pos, hd, cfg.rope_base, device)

        win = self.window if self.window is not None else L
        ti = real_pos.unsqueeze(1)
        tj = real_pos.unsqueeze(0)
        causal = tj <= ti
        windowed = (ti - tj) < win
        query_only_sees_own_block = (~is_query).unsqueeze(0) | (block_of.unsqueeze(1) == block_of.unsqueeze(0))
        real_never_sees_query = ~(is_query.unsqueeze(0) & (~is_query).unsqueeze(1))
        allow = causal & windowed & query_only_sees_own_block & real_never_sees_query
        attn_mask = allow.view(1, 1, Le, Le)

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_blocks = he.view(B, n_blocks, K + 1, D)
        return he_blocks[:, :, K, :]

    def forward(self, seq_repr: torch.Tensor, compute_ntp: bool = True,
                decode_kv: torch.Tensor | None = None, decode_K: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        K = cfg.Ks[self.level]
        D = cfg.d_model

        if self.is_byte_level:
            x = self.embed(seq_repr)
            B, L = seq_repr.shape
        else:
            x = seq_repr @ self.embed.weight
            B, L, _ = seq_repr.shape

        decode_disallow = decode_rope_k = None
        if decode_kv is not None:
            Nf = decode_kv.size(1)
            Kd = decode_K if decode_K is not None else K
            kv_window = math.ceil(2 * self.window / Kd) if self.window is not None else None
            decode_disallow, k_pos = jagged_causal_mask_and_positions(L, Nf, Kd, x.device, kv_window=kv_window)
            head_dim = D // cfg.n_heads
            decode_rope_k = rope_cos_sin_for_positions(k_pos, head_dim, cfg.rope_base, x.device)

        x0 = x
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        Kd_for_blocks = decode_K if decode_K is not None else K
        for block in self.blocks:
            x, _, _ = block(x, cos, sin, self.window, decode_kv, decode_disallow, decode_rope_k, Kd_for_blocks)

        h = self.ln_f(x)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            target = (seq_repr[:, 1:].reshape(-1) if self.is_byte_level
                      else seq_repr[:, 1:, :].argmax(-1).reshape(-1))
            logits = F.linear(h_flat, self.embed.weight)
            ntp_loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                ntp_acc = (logits.argmax(-1) == target).float().mean()
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        n_blocks = L // K
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        if cfg.code_extract_mode == "last_h":
            pooled = h_blocks[:, :, K - 1, :]
        elif cfg.code_extract_mode == "softmax_pool":
            q_implicit = h_blocks[:, :, K - 1, :]
            scores = (h_blocks * q_implicit.unsqueeze(2)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
        elif cfg.code_extract_mode == "light_query_attn":
            scores = (h_blocks * self.code_query.view(1, 1, 1, D)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
            pooled = self.code_out(pooled)
        elif cfg.code_extract_mode == "query_embed":
            pooled = self._query_embed_pool(x0, K, n_blocks)
        else:
            raise ValueError(f"unknown code_extract_mode {cfg.code_extract_mode!r}")
        pre_q = self._classify(pooled)
        c_i = gumbel_quantize(pre_q, cfg.gumbel_tau, cfg.use_gumbel_noise)

        return c_i, ntp_loss, ntp_acc, h


class RefineLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert cfg.d_model % cfg.n_heads == 0, f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens

        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels, f"attn_window tuple must have length n_levels={self.n_levels}, got {len(raw_windows)}"
        windows = [None if w == -1 else w for w in raw_windows]
        self.windows = windows
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] ({window}) must divide level {i}'s sequence length ({L}), or be >= it"

        encoders: list[LevelLM] = []
        for i in range(self.n_levels):
            lvl = LevelLM(cfg, i, windows[i], shared=(encoders[0] if i > 0 else None))
            encoders.append(lvl)
        self.encoders = nn.ModuleList(encoders)

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True):
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list = [], [], [], [], []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, loss_i, acc_i, h_i = self.encoders[i](seq_repr, compute_ntp=want_ntp)
            encode_losses.append(loss_i)
            encode_accs.append(acc_i)
            h_list.append(h_i)
            c_list.append(c_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        h_out = list(h_list)
        decode_levels = list(range(self.n_levels - 1)) if self.n_levels > 1 else [0]
        for i in decode_levels:
            if i + 1 < self.n_levels:
                source_c = c_list[i + 1]
                decode_K = cfg.Ks[i] * cfg.Ks[i + 1]
            else:
                source_c = c_list[i]
                decode_K = cfg.Ks[i]
            code_ids = source_c.argmax(-1)
            code_embeds = self.encoders[i].embed(code_ids)
            _, loss_i2, acc_i2, h_i2 = self.encoders[i](x_list[i], compute_ntp=compute_ntp,
                                                          decode_kv=code_embeds, decode_K=decode_K)
            decode_losses[i] = loss_i2
            decode_accs[i] = acc_i2
            h_out[i] = h_i2

        return encode_losses, encode_accs, decode_losses, decode_accs, h_out

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        encode_losses, encode_accs, decode_losses, decode_accs, h_list = self._run(byte_ids)

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                              else byte_loss.new_zeros(()))
        encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

        decode_terms = [l for l in decode_losses if l is not None]
        decode_total = (cfg.decode_ntp_weight * torch.stack(decode_terms).sum() if decode_terms
                         else byte_loss.new_zeros(()))

        loss = encode_total + decode_total
        ntp_total = torch.stack(encode_losses + decode_terms).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total, "ntp_loss_total": ntp_total,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
        }
        return loss, metrics


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor) -> torch.Tensor:
    enc0 = model.encoders[0]
    logits = F.linear(h_last, enc0.embed.weight)
    return logits.argmax(-1)


def _step_block(block: "Block", x_new: torch.Tensor, pos: int, cos_new, sin_new,
                 cache_k: torch.Tensor | None, cache_v: torch.Tensor | None, window: int | None,
                 decode_embed_new: torch.Tensor | None = None,
                 decode_cache_k: torch.Tensor | None = None, decode_cache_v: torch.Tensor | None = None):
    H, hd = block.attn.n_heads, block.attn.head_dim
    B, _, D = x_new.shape
    xn = block.ln1(x_new)
    qkv = block.attn.qkv(xn).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
    new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
    new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)

    new_dk = new_dv = None
    if decode_embed_new is not None:
        dqkv = block.attn.qkv(decode_embed_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        dk, dv = dqkv[1], dqkv[2]
        dk = apply_rope(dk, cos_new, sin_new)
        new_dk = dk if decode_cache_k is None else torch.cat([decode_cache_k, dk], dim=2)
        new_dv = dv if decode_cache_v is None else torch.cat([decode_cache_v, dv], dim=2)

    lo = max(0, (pos // window - 1) * window) if window is not None else 0
    k_attn, v_attn = new_k[:, :, lo:, :], new_v[:, :, lo:, :]
    if new_dk is not None:
        k_attn = torch.cat([k_attn, new_dk[:, :, lo:, :]], dim=2)
        v_attn = torch.cat([v_attn, new_dv[:, :, lo:, :]], dim=2)

    y = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=False)
    a = block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
    x_new = x_new + a
    x_new = x_new + block.mlp(block.ln2(x_new))
    return x_new, new_k, new_v, new_dk, new_dv


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Incremental generation. Only implemented for code_extract_mode=='last_h' and decode_K==1
    (Ks[0]==1 for n_levels==1, or Ks[0]*Ks[1]==1 for n_levels==2) -- the only cases the queued
    l1_k1/l2_k1 configs actually exercise. See _forward_chunked_aligned_decode's own docstring for
    why decode_K==1 lets decode_kv share the exact same chunk-aligned rolling window as local
    self-attention (kc_prev+kc) -- this function replicates that same window via a plain cache
    slice (`lo = max(0, (pos//window - 1) * window)`) instead of chunk-batched matmuls, since
    generation is one token at a time."""
    cfg = model.cfg
    assert cfg.code_extract_mode == "last_h", "generate_kv_cache only implemented for code_extract_mode='last_h'"
    n_levels = model.n_levels
    decode_K = cfg.Ks[0] * cfg.Ks[1] if n_levels >= 2 else cfg.Ks[0]
    assert decode_K == 1, "generate_kv_cache only implemented for decode_K==1"

    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    D = cfg.d_model
    head_dim = D // cfg.n_heads
    enc0 = model.encoders[0]
    n_layers = len(enc0.blocks)
    window0 = model.windows[0]

    clean0_k = [None] * n_layers
    clean0_v = [None] * n_layers
    fused0_k = [None] * n_layers
    fused0_v = [None] * n_layers
    decode0_k = [None] * n_layers
    decode0_v = [None] * n_layers

    if n_levels >= 2:
        enc1 = model.encoders[1]
        window1 = model.windows[1]
        clean1_k = [None] * n_layers
        clean1_v = [None] * n_layers

    def rope_at(pos):
        return rope_cos_sin_for_positions(torch.tensor([pos], device=device), head_dim, cfg.rope_base, device)

    def step_level0_clean(byte_id, pos):
        x = enc0.embed(byte_id).unsqueeze(1)
        cos_new, sin_new = rope_at(pos)
        for li, block in enumerate(enc0.blocks):
            x, clean0_k[li], clean0_v[li], _, _ = _step_block(
                block, x, pos, cos_new, sin_new, clean0_k[li], clean0_v[li], window0)
        return enc0.ln_f(x).squeeze(1)

    def step_level1_clean(code_id, pos):
        onehot = F.one_hot(code_id, num_classes=cfg.vocab).to(enc1.embed.weight.dtype)
        x = (onehot @ enc1.embed.weight).unsqueeze(1)
        cos_new, sin_new = rope_at(pos)
        for li, block in enumerate(enc1.blocks):
            x, clean1_k[li], clean1_v[li], _, _ = _step_block(
                block, x, pos, cos_new, sin_new, clean1_k[li], clean1_v[li], window1)
        return enc1.ln_f(x).squeeze(1)

    def step_level0_fused(byte_id, pos, decode_embed):
        x = enc0.embed(byte_id).unsqueeze(1)
        cos_new, sin_new = rope_at(pos)
        de = decode_embed.unsqueeze(1)
        for li, block in enumerate(enc0.blocks):
            x, fused0_k[li], fused0_v[li], decode0_k[li], decode0_v[li] = _step_block(
                block, x, pos, cos_new, sin_new, fused0_k[li], fused0_v[li], window0,
                de, decode0_k[li], decode0_v[li])
        return enc0.ln_f(x).squeeze(1)

    def advance(byte_id, pos, pending_decode_embed):
        h_fused = step_level0_fused(byte_id, pos, pending_decode_embed)
        h_clean0 = step_level0_clean(byte_id, pos)
        pre_q0 = enc0._classify(h_clean0)
        c0_id = gumbel_quantize(pre_q0, cfg.gumbel_tau, cfg.use_gumbel_noise).argmax(-1)
        if n_levels == 1:
            new_decode_embed = enc0.embed(c0_id)
        else:
            h_clean1 = step_level1_clean(c0_id, pos)
            pre_q1 = enc1._classify(h_clean1)
            c1_id = gumbel_quantize(pre_q1, cfg.gumbel_tau, cfg.use_gumbel_noise).argmax(-1)
            new_decode_embed = enc0.embed(c1_id)
        return h_fused, new_decode_embed

    L0 = prompt_bytes.size(1)
    B = prompt_bytes.size(0)
    pending_decode_embed = torch.zeros(B, D, device=device)
    last_h = None
    for pos in range(L0):
        last_h, pending_decode_embed = advance(prompt_bytes[:, pos], pos, pending_decode_embed)

    out_bytes = [prompt_bytes]
    for i in range(n_new_bytes):
        next_byte = _sample_next_byte(model, last_h)
        out_bytes.append(next_byte.unsqueeze(1))
        last_h, pending_decode_embed = advance(next_byte, L0 + i, pending_decode_embed)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def validate_generation(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    for _ in range(n_new_bytes):
        _, _, _, _, h_list = model._run(all_bytes, compute_ntp=False)
        next_byte = _sample_next_byte(model, h_list[0][:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_encode_only(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encoders[0]
    for _ in range(n_new_bytes):
        _, _, _, h = enc0(all_bytes, compute_ntp=False)
        next_byte = _sample_next_byte(model, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_level1_codes(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_codes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    enc0, enc1 = model.encoders[0], model.encoders[1]
    codes, _, _, _ = enc0(prompt_bytes, compute_ntp=False)
    n_prompt_codes = codes.shape[1]
    for _ in range(n_new_codes):
        _, _, _, h1 = enc1(codes, compute_ntp=False)
        logits = F.linear(h1[:, -1, :], enc1.embed.weight)
        next_id = logits.argmax(-1)
        next_code = F.one_hot(next_id, num_classes=model.cfg.vocab).to(codes.dtype)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return codes[0, n_prompt_codes:].argmax(-1)


@torch.no_grad()
def level1_ground_truth_codes(model: "RefineLM", full_bytes: torch.Tensor, prompt_len: int, device: str) -> torch.Tensor:
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    enc0 = model.encoders[0]
    K0 = model.cfg.Ks[0]
    c0, _, _, _ = enc0(full_bytes, compute_ntp=False)
    ids = c0[0].argmax(-1)
    n_prompt_codes = prompt_len // K0
    return ids[n_prompt_codes:]


def qualitative_generate(model: "RefineLM", prompt_bytes: torch.Tensor, gen_len: int,
                          ground_truth: torch.Tensor | None, device: str, log=print, label: str = "") -> None:
    prefix = f"qual_{label}_" if label else "qual_"
    out_cond = generate_no_cache(model, prompt_bytes, gen_len, device)
    gen_bytes_cond = bytes(out_cond[prompt_bytes.numel():].tolist())
    out_uncond = generate_encode_only(model, prompt_bytes, gen_len, device)
    gen_bytes_uncond = bytes(out_uncond[prompt_bytes.numel():].tolist())
    log(f"{prefix}prompt:              {bytes(prompt_bytes.tolist())!r}")
    log(f"{prefix}level0_uncond:       {gen_bytes_uncond!r}")
    log(f"{prefix}level0_cond:         {gen_bytes_cond!r}")
    if ground_truth is not None:
        log(f"{prefix}ground_truth:        {bytes(ground_truth.tolist())!r}")
    if model.n_levels > 1:
        K0 = model.cfg.Ks[0]
        n_new_codes = gen_len // K0
        if n_new_codes > 0:
            level1_gen = generate_level1_codes(model, prompt_bytes, n_new_codes, device)
            log(f"{prefix}level1_gen:          {level1_gen.tolist()}")
            if ground_truth is not None:
                full_bytes = torch.cat([prompt_bytes.reshape(-1), ground_truth.reshape(-1)])
                level1_gt = level1_ground_truth_codes(model, full_bytes, prompt_bytes.numel(), device)
                log(f"{prefix}level1_gt:           {level1_gt.tolist()}")


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    n = max(1, len(data) - context_len)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + context_len] for s in starts]).to(device)


def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def lr_at_warmup_constant_cosine(
    step: int, warmup: int, constant_steps: int, peak: float, total_steps: int, min_lr_frac: float = 0.1,
) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_start = warmup + constant_steps
    if step < decay_start:
        return peak
    min_lr = peak * min_lr_frac
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


@torch.no_grad()
def _add_per_level_bpb(result: dict) -> dict:
    for k in list(result.keys()):
        if k.endswith("_ntp_loss_encode") or k.endswith("_ntp_loss_decode"):
            result[k.replace("_ntp_loss_", "_bpb_")] = result[k] / math.log(2)
    return result


def eval_model(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    return _add_per_level_bpb(result)


def build_param_groups(model: RefineLM) -> list[dict]:
    seen: set[int] = set()
    params = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return [{"params": params}]


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v4_3", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
        )

        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = _add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.qual_gen_bytes > 0:
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                          window[args.qual_prompt_bytes:], device, log=log, label=label)


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Full weight-sharing encode/decode tower, dq=8 gumbel simplex, concat-only (qcute_refine_v4_3)", parents=[pre])
    p.add_argument("--Ks", default=(32, 32))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", default=32)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--decode_ntp_weight", type=float, default=1.0)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--use_gumbel_noise", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--code_extract_mode", type=str, default="last_h",
                    choices=["last_h", "softmax_pool", "light_query_attn", "query_embed"])
    p.add_argument("--code_head_tied", type=lambda x: x.lower() != "false", default=False)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--qual_gen_bytes", type=int, default=0)
    p.add_argument("--qual_prompt_bytes", type=int, default=64)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--compile", type=lambda x: x.lower() != "false", default=False)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.Ks = _parse_int_tuple(args.Ks)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers, context_len=args.context_len,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window, rope_base=args.rope_base,
        byte_ntp_weight=args.byte_ntp_weight, code_ntp_weight=args.code_ntp_weight,
        decode_ntp_weight=args.decode_ntp_weight, gumbel_tau=args.gumbel_tau, use_gumbel_noise=args.use_gumbel_noise,
        code_extract_mode=args.code_extract_mode, code_head_tied=args.code_head_tied,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_3_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} d_model={cfg.d_model} n_layers={cfg.n_layers} seq_lens={model.seq_lens} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
