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
    decode_pack_mode: str = "interleave"  # "interleave" | "prepend"
    decode_chunked: bool = False  # use windowed chunked decode attention instead of dense O((2L)^2); interleave only
    decode_code_ste: bool = True  # straight-through: decode's code_kv is source_c @ embed.weight (gradient flows into
    # the code producer). False = source_c.detach() @ embed.weight (hard argmax-equivalent forward value, no gradient
    # into the code producer -- the original behavior before this flag existed). Forward VALUE is identical either
    # way (source_c's forward value is already the hard one-hot); only the backward path differs.


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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None and T % window == 0 and T > window:
            y = self._forward_chunked(q, k, v, window)
        else:
            if window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window -- falling back to DENSE attention for this layer.")
                self._warned_dense_fallback = True
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

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

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        B, H, T, hd = q.shape
        W = window
        n_chunks = T // W

        qb = q.view(B, H, n_chunks, W, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = self._chunk_local_window(k, B, H, n_chunks, W, hd)
        vb = self._chunk_local_window(v, B, H, n_chunks, W, hd)
        mask_per_chunk = self._causal_window_mask(n_chunks, W, q.device)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)

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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        a = self.attn(self.ln1(x), cos, sin, window)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class LevelLM(nn.Module):
    def __init__(self, cfg: Config, level: int, window: int | None, decode_window: int | None,
                 shared: "LevelLM | None" = None):
        super().__init__()
        self.cfg = cfg
        self.level = level
        self.window = window
        self.decode_window = decode_window
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
            self.decode_bos = shared.decode_bos
        else:
            self.embed = nn.Embedding(V, D)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(D)
            self.code_head = None if cfg.code_head_tied else nn.Linear(D, V, bias=False)
            if self.code_head is not None:
                nn.init.normal_(self.code_head.weight, std=0.02)
            self.code_query = self.code_out = self.query_embed = None
            if cfg.code_extract_mode == "light_query_attn":
                self.code_query = nn.Parameter(torch.zeros(D))
                nn.init.normal_(self.code_query, std=0.02)
                self.code_out = nn.Linear(D, D, bias=False)
            elif cfg.code_extract_mode == "query_embed":
                self.query_embed = nn.Parameter(torch.zeros(D))
                nn.init.normal_(self.query_embed, std=0.02)
            self.decode_bos = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.decode_bos, std=0.02)

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

    def _packed_decode_forward(self, x0: torch.Tensor, code_kv: torch.Tensor, decode_K: int) -> torch.Tensor:
        """One prefix token (BOS or a code) per K-byte block, generalizing the decode_K==1 case
        (one prefix per byte) to arbitrary decode_K == Ks[i]*Ks[i+1]: code_kv has n_blocks=L//K
        entries, one per block; prefix b is decode_bos for b==0 else code_kv[b-1] (the PREVIOUS
        block's own code -- the last block's code is never consumed, same as the K==1 case,
        since there's no block after it to condition). A prefix's true_pos is set to the last
        raw byte position of the block it summarizes (b*K - 1 for prefix b, i.e. one before the
        block it precedes) -- combined with the existing same-position exclusion, this reproduces
        exactly the "code b visible only strictly after its last covered byte" rule at any K,
        including K==1 (where it reduces to the original code_shifted/code_true_pos formulas)."""
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        W = self.decode_window if self.decode_window is not None else L
        K = decode_K
        assert L % K == 0
        n_blocks = L // K

        bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
        prefixes = torch.cat([bos, code_kv[:, :-1, :]], dim=1)  # [B, n_blocks, D]
        prefix_true_pos = torch.arange(n_blocks, device=device) * K - 1  # [n_blocks]

        byte_pos = torch.arange(L, device=device)

        if cfg.decode_pack_mode == "interleave":
            x0_blocks = x0.view(B, n_blocks, K, D)
            combined = torch.cat([prefixes.unsqueeze(2), x0_blocks], dim=2).view(B, n_blocks * (K + 1), D)
            true_pos = torch.cat([prefix_true_pos.view(n_blocks, 1), byte_pos.view(n_blocks, K)], dim=1).reshape(-1)
            is_code = torch.cat([torch.ones(n_blocks, 1, dtype=torch.bool, device=device),
                                  torch.zeros(n_blocks, K, dtype=torch.bool, device=device)], dim=1).reshape(-1)
            Le = n_blocks * (K + 1)
        elif cfg.decode_pack_mode == "prepend":
            combined = torch.cat([prefixes, x0], dim=1)
            true_pos = torch.cat([prefix_true_pos, byte_pos], dim=0)
            is_code = torch.cat([torch.ones(n_blocks, dtype=torch.bool, device=device),
                                  torch.zeros(L, dtype=torch.bool, device=device)])
            Le = n_blocks + L
        else:
            raise ValueError(f"unknown decode_pack_mode {cfg.decode_pack_mode!r}")

        cos, sin = rope_cos_sin_for_positions(true_pos.clamp(min=0), hd, cfg.rope_base, device)

        ti = true_pos.unsqueeze(1)
        tj = true_pos.unsqueeze(0)
        key_is_code = is_code.unsqueeze(0)
        causal = tj <= ti
        same_pos_code_excluded = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < (2 * W)
        allow = causal & same_pos_code_excluded & windowed
        attn_mask = allow.view(1, 1, Le, Le)

        xe = combined
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
        if cfg.decode_pack_mode == "interleave":
            he_blocks = he.view(B, n_blocks, K + 1, D)
            return he_blocks[:, :, 1:, :].reshape(B, L, D)
        else:
            return he[:, n_blocks:, :]

    def _packed_decode_forward_chunked(self, x0: torch.Tensor, code_kv: torch.Tensor, n_prev_chunks: int = 2) -> torch.Tensor:
        """Chunked equivalent of _packed_decode_forward for decode_pack_mode == "interleave" only.

        Interleave's combined sequence (code_0,byte_0,code_1,byte_1,...) has true_pos
        non-decreasing in sequence order, so it can be chunked contiguously like plain windowed
        self-attention. Each byte occupies 2 slots (itself + its paired code), so a query chunk of
        sc=2*W slots is used, with n_prev_chunks previous chunks kept as extra key context to
        cover the required true-position reach R=2*W (margin empirically verified, see
        scripts/test_v4_4_chunked_decode.py).
        """
        cfg = self.cfg
        assert cfg.decode_pack_mode == "interleave", "chunked decode only implemented for interleave"
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        W = self.decode_window
        assert W is not None and L % W == 0

        bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
        code_shifted = torch.cat([bos, code_kv[:, :-1, :]], dim=1)

        byte_pos = torch.arange(L, device=device)
        code_true_pos = byte_pos - 1

        combined = torch.stack([code_shifted, x0], dim=2).view(B, 2 * L, D)
        true_pos = torch.stack([code_true_pos, byte_pos], dim=1).reshape(-1)
        is_code = torch.tensor([True, False], device=device).repeat(L)

        Le = 2 * L
        cos, sin = rope_cos_sin_for_positions(true_pos.clamp(min=0), hd, cfg.rope_base, device)

        sc = 2 * W
        n_chunks = Le // sc
        R = 2 * W

        pos_b = true_pos.view(n_chunks, sc)
        code_b = is_code.view(n_chunks, sc)
        pad_pos = torch.full((n_prev_chunks, sc), -10 ** 9, device=device, dtype=pos_b.dtype)
        pad_code = torch.zeros(n_prev_chunks, sc, dtype=torch.bool, device=device)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        code_ext = torch.cat([pad_code, code_b], dim=0)

        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
        pos_win = pos_ext[idx].reshape(n_chunks, (n_prev_chunks + 1) * sc)
        code_win = code_ext[idx].reshape(n_chunks, (n_prev_chunks + 1) * sc)

        ti = pos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        key_is_code = code_win.unsqueeze(1)
        causal = tj <= ti
        same_pos_excl = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < R
        allow = causal & same_pos_excl & windowed  # [n_chunks, sc, Kc]
        Kc = (n_prev_chunks + 1) * sc
        attn_mask = allow.view(1, n_chunks, 1, sc, Kc)

        xe = combined
        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
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

            mask_batched = attn_mask.expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
            yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
            y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Le, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_pairs = he.view(B, L, 2, D)
        return he_pairs[:, :, 1, :]

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

        x0 = x
        head_dim = D // cfg.n_heads

        if decode_kv is not None:
            assert decode_K is not None
            can_chunk = (decode_K == 1 and cfg.decode_chunked and cfg.decode_pack_mode == "interleave"
                         and self.decode_window is not None and L % self.decode_window == 0)
            if can_chunk:
                h = self._packed_decode_forward_chunked(x0, decode_kv)
            else:
                h = self._packed_decode_forward(x0, decode_kv, decode_K)
        else:
            cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
            for block in self.blocks:
                x = block(x, cos, sin, self.window)
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
        windows: list[int | None] = []
        decode_windows: list[int | None] = []
        for w in raw_windows:
            if isinstance(w, (tuple, list)):
                assert len(w) == 2, f"attn_window per-level entry must be a scalar or an (encode_window, decode_window) 2-tuple, got {w!r}"
                ew, dw = w
            else:
                ew = dw = w
            windows.append(None if ew == -1 else ew)
            decode_windows.append(None if dw == -1 else dw)
        self.windows = windows
        self.decode_windows = decode_windows
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] encode window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
        for i, (L, dwindow) in enumerate(zip(seq_lens, decode_windows)):
            if dwindow is not None:
                assert L % dwindow == 0 or L <= dwindow, f"attn_window[{i}] decode window ({dwindow}) must divide level {i}'s sequence length ({L}), or be >= it"

        encoders: list[LevelLM] = []
        for i in range(self.n_levels):
            lvl = LevelLM(cfg, i, windows[i], decode_windows[i], shared=(encoders[0] if i > 0 else None))
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
            if x_list[i].shape[1] % decode_K != 0:
                # Ragged length (only happens during generation, where the sequence grows one
                # byte at a time and won't generally be a multiple of decode_K -- training always
                # uses context_len, guaranteed divisible per RefineLM.__init__'s own asserts).
                # Decode conditioning isn't well-defined here; fall back to the encode-only h_i
                # already sitting in h_out[i], same graceful-degradation pattern as the dense/
                # chunked attention fallbacks elsewhere in this file.
                continue
            src = source_c if cfg.decode_code_ste else source_c.detach()
            code_embeds = src @ self.encoders[i].embed.weight
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


def _annotate_bytes_with_codes(byte_ids: torch.Tensor, code_ids: torch.Tensor, K: int) -> str:
    """Group byte_ids into K-byte blocks, tagging each block with its own code_i in {}
    (one code covers K raw bytes, decode_K in RefineLM._run terms)."""
    bytes_list = byte_ids.tolist()
    codes_list = code_ids.tolist()
    parts = []
    for b in range(0, len(bytes_list), K):
        block = "".join(repr(bytes([x]))[2:-1] for x in bytes_list[b:b + K])
        code_idx = b // K
        c = codes_list[code_idx] if code_idx < len(codes_list) else "?"
        parts.append(f"{block}{{{c}}}")
    return "".join(parts)


@torch.no_grad()
def _decode_source_codes(model: "RefineLM", full_bytes: torch.Tensor, device: str) -> torch.Tensor:
    """Re-derive, per-position, the code_i (level i's own self-code, see the stream_i/code_i
    terminology note in docs/status.md) that decode conditions on (source_c in RefineLM._run)
    for the given full (prompt+generated) byte sequence. Only valid for the
    decode_K==1, decode-at-level-0-only configurations this file currently trains
    (n_levels in {1, 2})."""
    was_training = model.training
    model.eval()
    seq_repr = full_bytes.to(device)
    if seq_repr.dim() == 1:
        seq_repr = seq_repr.unsqueeze(0)
    c_list = []
    for i in range(model.n_levels):
        c_i, _, _, _ = model.encoders[i](seq_repr, compute_ntp=False)
        c_list.append(c_i)
        seq_repr = c_i
    source_c = c_list[1] if model.n_levels > 1 else c_list[0]
    if was_training:
        model.train()
    return source_c.argmax(-1)[0]


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
    decode_K = model.cfg.Ks[0] * model.cfg.Ks[1] if model.n_levels > 1 else model.cfg.Ks[0]
    code_ids_full = _decode_source_codes(model, out_cond, device)
    n_prompt_codes = prompt_bytes.numel() // decode_K
    gen_code_ids = code_ids_full[n_prompt_codes:]
    annotated = _annotate_bytes_with_codes(out_cond[prompt_bytes.numel():], gen_code_ids, decode_K)
    log(f"{prefix}level0_cond_codes:   {annotated}  <{gen_code_ids.tolist()}>")
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
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v4_4", dynamic_ncols=True)
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

    p = argparse.ArgumentParser(description="Packed-sequence decode (prepend/interleave), trainable BOS, correct RoPE timing (qcute_refine_v4_4)", parents=[pre])
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
    p.add_argument("--decode_pack_mode", type=str, default="interleave", choices=["interleave", "prepend"])
    p.add_argument("--decode_chunked", type=lambda x: x.lower() != "false", default=False)

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
        decode_pack_mode=args.decode_pack_mode, decode_chunked=args.decode_chunked,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_4_{int(time.time())}")
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
