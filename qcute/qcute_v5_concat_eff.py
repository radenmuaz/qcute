"""v4.4.1 fork ("concat" naming: self track's code is packed/concatenated inline into one
self-attention sequence, same mechanism as v4.4.1). Adds Config.quant_type: "softmax" (default,
unchanged categorical code_head + gumbel/argmax) or "bsq" (binary spherical quantization,
Config.bsq_bits-wide sign code, straight-through). See docs/qcute_refine_v4_4_1_v4_5_1_math.md for
the base mechanism (self-code LM continuation, unchanged here).

    uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_overfit10k_k4single.py
"""
import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
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
    # scalar, or per-level tuple; each level entry is scalar or (encode_window, decode_window),
    # where decode_window is scalar or a per-source tuple [self, +1, ..., top]; -1 = unbounded, 0 = drop that source
    attn_window: int | tuple[int, ...] = 32
    rope_base: float = 10000.0
    byte_ntp_weight: float = 1.0
    code_ntp_weight: float = 1.0
    decode_ntp_weight: float = 1.0
    gumbel_tau: float = 1.0
    use_gumbel_noise: bool = False
    vocab: int = 256
    code_extract_mode: str = "last_h"
    code_head_tied: bool = False
    decode_pack_mode: str = "interleave"
    decode_chunked: bool = False
    decode_banded: bool = False
    cross_track_source: str = "encode"
    decode_self_only_aux: bool = False
    decode_self_only_weight: float = 1.0
    decode_self_only_dropout: bool = False
    decode_self_only_dropout_p: float = 0.5
    decode_code_ste: bool = True
    share_level_weights: bool = False
    quant_type: str = "softmax"   # "softmax" (categorical code_head + gumbel/argmax) or "bsq"
    bsq_bits: int = 4             # code width in bits when quant_type="bsq"


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


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def quantize_code(pre_q: torch.Tensor, cfg: "Config") -> torch.Tensor:
    if cfg.quant_type == "bsq":
        return bsq_quantize(pre_q, cfg.bsq_bits)
    return gumbel_quantize(pre_q, cfg.gumbel_tau, cfg.use_gumbel_noise)


def code_to_ids(source_c: torch.Tensor, cfg: "Config") -> torch.Tensor:
    """Display-only: packs a bsq bit-vector into an int; argmax for softmax's one-hot."""
    if cfg.quant_type == "bsq":
        bits = (source_c > 0).long()
        weights = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        return (bits * weights).sum(-1)
    return source_c.argmax(-1)


MAX_PQ_TABLE_DQ = 16   # 2**16 = 65536 rows -- ceiling bsq_bits this table lookup allows


class CodeEmbed(nn.Module):
    """Maps a bsq code (one of 2**dq discrete hypersphere corners) to a D-dim vector via an exact
    table lookup, not a linear combination of its +-1 components (default/only mode for bsq's code
    consumption -- ported from qcute_refine_v2's pq_table, see docs/archive2/kv_contribution.md).
    A naive lookup is non-differentiable in the code, severing the only gradient path back to the
    code's own producer; fixed with the same straight-through idiom bsq_quantize uses: a continuous
    `proxy` linear carries the backward gradient, the table row is what's used forward."""

    def __init__(self, dq: int, D: int):
        super().__init__()
        assert dq <= MAX_PQ_TABLE_DQ, f"CodeEmbed: dq={dq} would need a 2**{dq}-row table -- keep dq<={MAX_PQ_TABLE_DQ}."
        self.table = nn.Embedding(2 ** dq, D)
        self.proxy = nn.Linear(dq, D)
        self.register_buffer("_powers", (2 ** torch.arange(dq)).long(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = ((x > 0).long() * self._powers).sum(-1)
        hard = self.table(idx)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


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


def chunked_windowed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    """Causal attention restricted to (i - j < window) for a PLAIN (non-interleaved) sequence,
    computed via chunking (gather current + previous chunk, O(T*window)) instead of materializing
    a dense T x T mask (O(T^2)). Produces bit-identical results to the dense
    causal & (i-j<window) mask -- see docs/status.md's windowed-attention efficiency review.

    Fast path requires T % window == 0 (always true during training, where context_len is built
    to divide evenly -- see RefineLM.__init__); falls back to the dense mask (correct for any T,
    same as the original qcute_v5_concat.py) otherwise, which no-cache generation's ragged, growing
    T can hit.
    """
    B, H, T, hd = q.shape
    w = window
    device = q.device
    if T <= w:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if T % w != 0:
        pos = torch.arange(T, device=device)
        ti, tj = pos.unsqueeze(1), pos.unsqueeze(0)
        attn_mask = ((tj <= ti) & (ti - tj < w)).view(1, 1, T, T)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    n_chunks = T // w
    qb = q.view(B, H, n_chunks, w, hd)
    kb = k.view(B, H, n_chunks, w, hd)
    vb = v.view(B, H, n_chunks, w, hd)
    pad_k = torch.zeros(B, H, 1, w, hd, device=device, dtype=k.dtype)
    pad_v = torch.zeros(B, H, 1, w, hd, device=device, dtype=v.dtype)
    k_ext = torch.cat([pad_k, kb], dim=2)
    v_ext = torch.cat([pad_v, vb], dim=2)

    idx = torch.arange(n_chunks, device=device).view(n_chunks, 1) + torch.arange(2, device=device).view(1, 2)
    k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)
    v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)

    pos = torch.arange(T, device=device)
    pos_b = pos.view(n_chunks, w)
    pad_pos = torch.full((1, w), -10 ** 9, device=device, dtype=pos.dtype)
    pos_ext = torch.cat([pad_pos, pos_b], dim=0)
    pos_win = pos_ext[idx].reshape(n_chunks, 2 * w)

    ti = pos_b.unsqueeze(-1)
    tj = pos_win.unsqueeze(1)
    allow = (tj <= ti) & (ti - tj < w)
    mask_flat = allow.view(1, n_chunks, 1, w, 2 * w).expand(B, n_chunks, 1, w, 2 * w).reshape(B * n_chunks, 1, w, 2 * w)

    qb_flat = qb.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, w, hd)
    k_win_flat = k_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)
    v_win_flat = v_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)

    y = F.scaled_dot_product_attention(qb_flat, k_win_flat, v_win_flat, attn_mask=mask_flat)
    return y.view(B, n_chunks, H, w, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None:
            y = chunked_windowed_attention(q, k, v, window)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


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


_warned_thin_window: set[tuple[int, ...]] = set()


def _warn_thin_window(tracks: list[tuple[torch.Tensor, int, int | None]], window: int, min_codes: int = 2) -> None:
    """Cumulative code coverage a banded decode window actually buys, across all tracks (levels):
    codes_in_window[j] ~= window // K_j, summed. Below min_codes the window is thinner than the
    minimum useful conditioning (2 codes = the self-code LM-continuation floor used everywhere
    else in this codebase) -- almost certainly starving the model of the coarser-level context the
    multi-track decode exists to provide. Not a correctness check (attention is exact for whatever
    window is set); this is due diligence the caller owns, same spirit as sample_context's warning."""
    key = tuple(K for _, K, _ in tracks) + (window,)
    if key in _warned_thin_window:
        return
    total_codes = sum(max(0, window // K) for _, K, _ in tracks)
    if total_codes < min_codes:
        _warned_thin_window.add(key)
        Ks_str = ",".join(str(K) for _, K, _ in tracks)
        print(f"WARNING: banded decode window={window} covers only ~{total_codes} cumulative code(s) "
              f"across tracks Ks=({Ks_str}) -- below min_codes={min_codes}. Coarser-level context is "
              f"likely starved; consider a larger attn_window decode_window for these levels.")


class LevelLM(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        V = cfg.vocab
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(D)
        if cfg.quant_type == "bsq":
            self.code_head = nn.Linear(D, cfg.bsq_bits, bias=False)
            nn.init.normal_(self.code_head.weight, std=0.02)
            self.code_embed = CodeEmbed(cfg.bsq_bits, D)
            self.code_predict = nn.Linear(D, cfg.bsq_bits, bias=False)
            nn.init.normal_(self.code_predict.weight, std=0.02)
        else:
            self.code_head = None if cfg.code_head_tied else nn.Linear(D, V, bias=False)
            if self.code_head is not None:
                nn.init.normal_(self.code_head.weight, std=0.02)
            self.code_embed = self.code_predict = None
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
        self._banded_cache: dict = {}   # (L, tracks_sig, device) -> structural tensors, see _banded_structure

    def _classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.code_head(pooled) if self.code_head is not None else F.linear(pooled, self.embed.weight)

    def _query_embed_pool(self, x0: torch.Tensor, K: int, n_blocks: int, window: int | None) -> torch.Tensor:
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

        win = window if window is not None else L
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

    def _packed_decode_forward_selfcode(self, x0: torch.Tensor, code_kv: torch.Tensor, K: int,
                                         window: int | None, extra_query: bool = False
                                         ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        """code_kv[b] predicts block b+1, not block b -- see docs/qcute_refine_v4_4_1_v4_5_1_math.md.

        extra_query=True appends code_kv's last (otherwise-unused) code as a bare trailing query
        token with no following byte block, and returns its post-attention state as query_last --
        see qcute_v5_stack.py's selfcode_decode docstring for why generate_no_cache needs this
        instead of sampling from the byte-slot `h`.

        Chunked fast path (O(Le*window), gather-current-plus-previous-unit-chunks, same pattern as
        _packed_decode_forward_banded) applies when extra_query=False (the training hot path) and
        window is set and smaller than Le -- bit-identical to the dense causal & (i-j<window) mask,
        see docs/status.md's windowed-attention efficiency review. extra_query=True is
        generation-only (see generate_no_cache/check_gen_consistency), called far less often than
        the training forward pass, and falls back to the dense mask unchanged from the original.
        """
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        n_blocks = L // K
        n_units = n_blocks - 1
        assert n_units >= 1, f"self-code LM-continuation decode needs at least 2 blocks (n_blocks={n_blocks})"
        # Floor, not exact-divide: L is a clean multiple of K during training (see RefineLM.__init__),
        # but generation's growing prefix rarely is. Drop the ragged tail here -- it belongs to a
        # block that isn't complete yet and can't get a self-code-conditioned representation; the
        # caller (RefineLM._run) splices the corresponding tail of the plain encode-only h back in.
        x0 = x0[:, :n_blocks * K, :]
        x0_blocks = x0.view(B, n_blocks, K, D)
        codes = code_kv[:, :n_units, :]
        blocks_ = x0_blocks[:, 1:, :, :]
        combined = torch.cat([codes.unsqueeze(2), blocks_], dim=2).view(B, n_units * (K + 1), D)
        Le = n_units * (K + 1)
        if extra_query:
            last_code = code_kv[:, n_units:n_units + 1, :]
            combined = torch.cat([combined, last_code], dim=1)
            Le += 1

        use_chunked = (not extra_query) and window is not None and Le > window
        cos, sin = rope_cos_sin(Le, hd, cfg.rope_base, device)

        if not use_chunked:
            # window is None, or window >= Le (checked by use_chunked's `Le > window` gate above):
            # the windowed comparison (ti-tj)<window is then a no-op (positions are contiguous
            # 0..Le-1 with no ties), so this is exactly regular full causal attention -- use SDPA's
            # fused is_causal path instead of materializing an equivalent explicit mask.
            fully_causal = window is None or window >= Le
            attn_mask = None
            if not fully_causal:
                true_pos = torch.arange(Le, device=device)
                ti, tj = true_pos.unsqueeze(1), true_pos.unsqueeze(0)
                attn_mask = ((tj <= ti) & ((ti - tj) < window)).view(1, 1, Le, Le)

            xe = combined
            for block in self.blocks:
                xn = block.ln1(xe)
                qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                if fully_causal:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
                xe = xe + a
                xe = xe + block.mlp(block.ln2(xe))
            he_all = self.ln_f(xe)
        else:
            # Chunk size = the configured window itself, not K+1 (a single code+block unit is
            # often tiny -- K=1..4 -- which floods SDPA with thousands of tiny batched calls, same
            # issue as the multi-track banded path; see _warn_thin_window). Regular windowed
            # attention: due diligence that `window` covers enough of this track's own code
            # history is on the caller.
            _warn_thin_window([(code_kv, K, window)], window)
            sc = max(1, min(window, Le))
            n_chunks = -(-Le // sc)
            pad_len = n_chunks * sc - Le
            Lp = n_chunks * sc
            n_prev_chunks = min(max(1, -(-window // sc)), n_chunks - 1)   # never look back past real chunks
            if pad_len > 0:
                cos, sin = rope_cos_sin(Lp, hd, cfg.rope_base, device)   # padded rows unused (masked out), value irrelevant

            true_pos = torch.arange(Le, device=device)
            if pad_len > 0:
                true_pos = F.pad(true_pos, (0, pad_len), value=-10 ** 9)
            pos_b = true_pos.view(n_chunks, sc)
            pad_pos = torch.full((n_prev_chunks, sc), -10 ** 9, device=device, dtype=pos_b.dtype)
            pos_ext = torch.cat([pad_pos, pos_b], dim=0)
            idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
                   + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
            Kc = (n_prev_chunks + 1) * sc
            pos_win = pos_ext[idx].reshape(n_chunks, Kc)

            ti = pos_b.unsqueeze(-1)
            tj = pos_win.unsqueeze(1)
            allow = (tj <= ti) & (ti - tj < window)
            attn_mask = allow.view(1, n_chunks, 1, sc, Kc)

            xe = combined
            if pad_len > 0:
                xe = F.pad(xe, (0, 0, 0, pad_len))
            for block in self.blocks:
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

                mask_batched = attn_mask.expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
                yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
                y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

                a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
                xe = xe + a
                xe = xe + block.mlp(block.ln2(xe))
            he_all = self.ln_f(xe)[:, :Le, :]

        query_last = None
        if extra_query:
            query_last = he_all[:, -1, :]
            he_all = he_all[:, :-1, :]
        return he_all, n_units, query_last

    def _packed_decode_forward(self, x0: torch.Tensor, tracks: list[tuple[torch.Tensor, int, int | None]],
                                return_debug: bool = False):
        # Always called with len(tracks) >= 2 (LevelLM.forward routes the single-track case to
        # _packed_decode_forward_selfcode instead), so there is exactly one code path here -- the
        # flat "prepend" layout, used regardless of cfg.decode_pack_mode's string value.
        assert len(tracks) >= 2, "single-track decode must go through _packed_decode_forward_selfcode"
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        byte_pos = torch.arange(L, device=device)

        prefix_parts, true_pos_parts, window_parts = [], [], []
        for code_kv, K, window in tracks:
            # Floor, not exact-divide: during generation L grows byte-by-byte and is rarely a
            # multiple of K. n_blocks = L // K is the number of complete blocks currently available
            # -- a trailing partial block simply isn't given a prefix slot yet, never
            # padded/fabricated. Training always has L a clean multiple of every track's K (see
            # RefineLM.__init__ / _run's track-building loop), so the floor is a no-op there.
            #
            # n_avail = min(n_blocks, code_kv.shape[1]) prefixes get built, at true_pos
            # [-1, K-1, 2K-1, ..., n_avail*K-1] with values [bos, code_kv[0], ..., code_kv[n_avail-1]]
            # -- i.e. one prefix slot PER code, including code_kv[n_avail-1] itself (the freshest
            # complete block), not code_kv[:-1]. That last slot sits at true_pos = n_avail*K - 1,
            # right before the next (currently incomplete) block's bytes start -- exactly the
            # conditioning a mid-block generation query needs and, without it, silently lacked
            # (invisible in training/one-shot teacher-forcing, where n_blocks is always large enough
            # that this slot gets built anyway as part of a later, still-unused prefix; only a short,
            # growing generation prefix exposes the gap). code_kv can be shorter than n_blocks for a
            # decode-derived coarser track (its own LM-continuation shifts it by one) -- n_avail
            # simply clamps to what's actually there.
            n_blocks = L // K
            n_avail = min(n_blocks, code_kv.shape[1])
            W = window if window is not None else L
            bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
            prefixes = torch.cat([bos, code_kv[:, :n_avail, :]], dim=1)
            prefix_parts.append(prefixes)
            true_pos_parts.append(torch.arange(n_avail + 1, device=device) * K - 1)
            window_parts.append(torch.full((n_avail + 1,), float(W), device=device))
        prefix_parts, true_pos_parts, window_parts = prefix_parts[::-1], true_pos_parts[::-1], window_parts[::-1]

        all_prefixes = torch.cat(prefix_parts, dim=1)
        n_prefix = all_prefixes.shape[1]
        prefix_true_pos = torch.cat(true_pos_parts, dim=0)
        prefix_window = torch.cat(window_parts, dim=0)
        byte_window = tracks[0][2] if tracks[0][2] is not None else L

        combined = torch.cat([all_prefixes, x0], dim=1)
        true_pos = torch.cat([prefix_true_pos, byte_pos], dim=0)
        is_code = torch.cat([torch.ones(n_prefix, dtype=torch.bool, device=device),
                              torch.zeros(L, dtype=torch.bool, device=device)])
        window_of_key = torch.cat([prefix_window, torch.full((L,), float(byte_window), device=device)])
        Le = n_prefix + L
        extract = lambda he: he[:, n_prefix:, :]

        cos, sin = rope_cos_sin_for_positions(true_pos.clamp(min=0), hd, cfg.rope_base, device)

        ti = true_pos.unsqueeze(1)
        tj = true_pos.unsqueeze(0)
        key_is_code = is_code.unsqueeze(0)
        causal = tj <= ti
        same_pos_code_excluded = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < (2 * window_of_key.unsqueeze(0))
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
        out = extract(he)
        if return_debug:
            debug = {"true_pos": true_pos, "is_code": is_code, "window_of_key": window_of_key,
                     "attn_mask": allow, "n_prefix": Le - L, "Le": Le, "L": L}
            return out, debug
        return out

    def _banded_structure(self, L: int, tracks_sig: tuple[tuple[int, int | None], ...],
                           device: torch.device, margin_extra_chunks: int = 1) -> dict:
        """Everything in _packed_decode_forward_banded that depends only on shapes/config (L, each
        track's (K, window)) and never on data -- the permutation/chunk/mask machinery, including
        the argsort that used to run fresh every forward call. Training reuses the same (L,
        tracks_sig) every single step (context_len/Ks/windows are fixed for a run), so this cache
        turns the sort into a one-time cost instead of a per-step one -- no runtime resorting, per
        the user's request. Keyed also by device since index tensors aren't portable across
        devices."""
        key = (L, tracks_sig, str(device))
        cached = self._banded_cache.get(key)
        if cached is not None:
            return cached
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        byte_pos = torch.arange(L, device=device)

        true_pos_parts, window_parts, n_prefix_parts = [], [], []
        for K, window in tracks_sig:
            assert L % K == 0
            n_blocks = L // K
            W = window if window is not None else L
            true_pos_parts.append(torch.arange(n_blocks, device=device) * K - 1)
            window_parts.append(torch.full((n_blocks,), float(W), device=device))
            n_prefix_parts.append(n_blocks)
        true_pos_parts, window_parts = true_pos_parts[::-1], window_parts[::-1]
        n_prefix_parts = n_prefix_parts[::-1]
        n_prefix = sum(n_prefix_parts)

        prefix_true_pos = torch.cat(true_pos_parts, dim=0)
        prefix_window = torch.cat(window_parts, dim=0)
        byte_window = tracks_sig[0][1] if tracks_sig[0][1] is not None else L

        true_pos = torch.cat([prefix_true_pos, byte_pos], dim=0)
        is_code = torch.cat([torch.ones(n_prefix, dtype=torch.bool, device=device),
                              torch.zeros(L, dtype=torch.bool, device=device)])
        window_of_key = torch.cat([prefix_window, torch.full((L,), float(byte_window), device=device)])
        Le = n_prefix + L

        sort_key = true_pos * 2 + is_code.long()
        perm = torch.argsort(sort_key)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(Le, device=device)

        true_pos_s = true_pos[perm]
        is_code_s = is_code[perm]
        window_of_key_s = window_of_key[perm]

        W_max = float(window_of_key.max().item())
        R = 2.0 * W_max

        # Chunk size = the configured window itself, NOT the finest track's K: correctness comes
        # entirely from the true_pos-based mask below, not from sc matching any track's block
        # size -- a small sc (old: finest K, often 2-5) blows up n_prev_chunks (R/sc) and floods
        # SDPA with thousands of tiny batched calls. Using the window directly keeps n_chunks/
        # n_prev_chunks small (few, large batched matmuls instead), at the cost of gathering more
        # (masked-out, wasted) keys per chunk -- a bandwidth-for-launch-count trade that wins when
        # launch overhead dominates. Regular windowed attention: the window is whatever the user
        # configured via attn_window's decode side -- due diligence that it covers enough
        # cumulative code context across levels is on the caller (see _warn_thin_window above).
        sc = max(1, min(int(W_max) if W_max > 0 else Le, Le))
        n_chunks = -(-Le // sc)
        pad_len = n_chunks * sc - Le
        if pad_len > 0:
            true_pos_s = F.pad(true_pos_s, (0, pad_len), value=1e9)
            is_code_s = F.pad(is_code_s, (0, pad_len), value=False)
            window_of_key_s = F.pad(window_of_key_s, (0, pad_len), value=0.0)
        Lp = n_chunks * sc

        max_ties_per_pos = len(tracks_sig) + 1
        # Clamped to n_chunks-1: when window >= Le, sc==Le collapses to a single chunk (n_chunks==1)
        # -- the formula below would otherwise still ask for many "previous" chunks that don't
        # exist, gathering pure padding for nothing. This makes that case degenerate cleanly to one
        # dense chunk, i.e. regular causal attention (see docs/status.md).
        n_prev_needed = max(1, int(math.ceil(max_ties_per_pos * R / sc)) + margin_extra_chunks)
        n_prev_chunks = min(n_prev_needed, max(0, n_chunks - 1))

        cos, sin = rope_cos_sin_for_positions(true_pos_s.clamp(min=0), hd, cfg.rope_base, device)

        pos_b = true_pos_s.view(n_chunks, sc)
        code_b = is_code_s.view(n_chunks, sc)
        win_b = window_of_key_s.view(n_chunks, sc)
        pad_pos = torch.full((n_prev_chunks, sc), -1e9, device=device, dtype=pos_b.dtype)
        pad_code = torch.zeros(n_prev_chunks, sc, dtype=torch.bool, device=device)
        pad_win = torch.zeros(n_prev_chunks, sc, device=device, dtype=win_b.dtype)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        code_ext = torch.cat([pad_code, code_b], dim=0)
        win_ext = torch.cat([pad_win, win_b], dim=0)

        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
        Kc = (n_prev_chunks + 1) * sc
        pos_win = pos_ext[idx].reshape(n_chunks, Kc)
        code_win = code_ext[idx].reshape(n_chunks, Kc)
        win_win = win_ext[idx].reshape(n_chunks, Kc)

        ti = pos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        key_is_code = code_win.unsqueeze(1)
        key_window = win_win.unsqueeze(1)
        causal = tj <= ti
        same_pos_excl = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < (2.0 * key_window)
        allow = causal & same_pos_excl & windowed
        struct = dict(perm=perm, inv_perm=inv_perm, n_prefix=n_prefix, Le=Le, Lp=Lp, pad_len=pad_len,
                       sc=sc, n_chunks=n_chunks, n_prev_chunks=n_prev_chunks, Kc=Kc, idx=idx,
                       cos=cos, sin=sin, attn_mask=allow.view(1, n_chunks, 1, sc, Kc), W_max=W_max)
        self._banded_cache[key] = struct
        return struct

    def _packed_decode_forward_banded(self, x0: torch.Tensor, tracks: list[tuple[torch.Tensor, int, int | None]],
                                       margin_extra_chunks: int = 1) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device

        tracks_sig = tuple((K, window) for _, K, window in tracks)
        st = self._banded_structure(L, tracks_sig, device, margin_extra_chunks)
        perm, inv_perm = st["perm"], st["inv_perm"]
        n_prefix, Le, Lp, pad_len = st["n_prefix"], st["Le"], st["Lp"], st["pad_len"]
        sc, n_chunks, n_prev_chunks, Kc, idx = st["sc"], st["n_chunks"], st["n_prev_chunks"], st["Kc"], st["idx"]
        cos, sin, attn_mask = st["cos"], st["sin"], st["attn_mask"]
        _warn_thin_window(tracks, int(st["W_max"]))

        prefix_parts = []
        for code_kv, K, window in tracks:
            bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
            prefix_parts.append(torch.cat([bos, code_kv[:, :-1, :]], dim=1))
        prefix_parts = prefix_parts[::-1]
        all_prefixes = torch.cat(prefix_parts, dim=1)
        combined = torch.cat([all_prefixes, x0], dim=1)
        combined_s = combined[:, perm, :]
        if pad_len > 0:
            combined_s = F.pad(combined_s, (0, 0, 0, pad_len))

        xe = combined_s
        for block in self.blocks:
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

            mask_batched = attn_mask.expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
            yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
            y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he = he[:, :Le, :]
        he = he[:, inv_perm, :]
        return he[:, n_prefix:, :]

    def _ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor, is_byte_level: bool) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if is_byte_level:
            target = target_repr.reshape(-1)
            logits = F.linear(h_query, self.embed.weight)
            loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
            return loss, acc
        if cfg.quant_type == "bsq":
            target_bits = (target_repr.reshape(-1, cfg.bsq_bits) > 0).float()
            pred = self.code_predict(h_query)
            loss = F.binary_cross_entropy_with_logits(pred, target_bits)
            with torch.no_grad():
                acc = ((pred > 0).float() == target_bits).float().mean()
            return loss, acc
        target = target_repr.argmax(-1).reshape(-1)
        logits = F.linear(h_query, self.embed.weight)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def forward(self, seq_repr: torch.Tensor, level: int, window: int | None, compute_ntp: bool = True,
                decode_tracks: list[tuple[torch.Tensor, int, int | None]] | None = None,
                extra_query: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cfg = self.cfg
        K = cfg.Ks[level]
        D = cfg.d_model
        is_byte_level = level == 0

        if is_byte_level:
            x = self.embed(seq_repr)
            B, L = seq_repr.shape
        else:
            x = self.code_embed(seq_repr) if cfg.quant_type == "bsq" else seq_repr @ self.embed.weight
            B, L, _ = seq_repr.shape

        x0 = x
        head_dim = D // cfg.n_heads

        query_last = None
        query_seq = None
        selfcode_ntp_done = False
        if decode_tracks is not None:
            assert len(decode_tracks) >= 1
            _, _, w0 = decode_tracks[0]
            if len(decode_tracks) == 1:
                code_kv0 = decode_tracks[0][0]
                n_blocks_sc = L // K
                he, n_units, query_last = self._packed_decode_forward_selfcode(
                    x0, code_kv0, K, w0, extra_query=extra_query)
                he_blocks = he.view(B, n_units, K + 1, D)
                query_h = he_blocks[:, :, :-1, :].reshape(-1, D)
                # query_seq keeps the batch dim (unlike query_h, flattened for the loss) -- this is
                # the code-slot representation actually used for NTP at every position, needed by
                # check_gen_consistency's teacher-forced reference. Plain `h` below is byte-slot
                # (post byte-attendance) and is the WRONG tensor to compare against for this
                # single-track path -- see docs/qcute_refine_v4_4_1_v4_5_1_math.md /
                # docs/status.md's byte-slot-vs-code-slot bug.
                query_seq = he_blocks[:, :, :-1, :].reshape(B, n_units * K, D)
                h = torch.cat([x0[:, :K, :], he_blocks[:, :, 1:, :].reshape(B, n_units * K, D)], dim=1)
                if compute_ntp:
                    # query_h has n_units*K rows (floor-based -- see _packed_decode_forward_selfcode);
                    # seq_repr[:, K:] has L-K, which only matches when L is an exact multiple of K
                    # (always true in training, where L==cfg.context_len by construction). Eval can
                    # see a shorter/ragged L when data is smaller than context_len -- slice the target
                    # to the same length query_h actually has instead of assuming exact alignment.
                    target = seq_repr[:, K:K + n_units * K]
                    ntp_loss, ntp_acc = self._ntp_loss_acc(query_h, target, is_byte_level)
                else:
                    ntp_loss = h.new_zeros(())
                    ntp_acc = h.new_zeros(())
                selfcode_ntp_done = True
            else:
                # _packed_decode_forward_banded is the O(Le*window) chunked path (see
                # docs/status.md's windowed-attention efficiency review) but requires every track's
                # K to divide L exactly -- always true during training (context_len is built to
                # divide evenly at every level, see RefineLM.__init__), not necessarily true during
                # generation's growing, ragged L. Use it whenever safe and at least one track has a
                # real (finite) window (an all-unbounded set of tracks gains nothing from banding --
                # its n_prev_chunks would grow to cover the whole sequence anyway); fall back to the
                # dense, floor-tolerant _packed_decode_forward (unchanged from the original) for
                # ragged L or fully-unbounded windows. Unlike the original file's decode_chunked/
                # decode_banded, this dispatch isn't gated behind a config flag that defaults off --
                # the fast path is used whenever it's numerically safe. banded's argsort+gather is
                # cached per (L, tracks_sig) (see _banded_structure), so its fixed cost is paid once
                # per shape, not once per step.
                finite_windows = [w for _, _, w in decode_tracks if w is not None]
                can_band = all(L % K == 0 for _, K, _ in decode_tracks) and bool(finite_windows)
                if can_band:
                    h = self._packed_decode_forward_banded(x0, decode_tracks)
                else:
                    h = self._packed_decode_forward(x0, decode_tracks)
        else:
            cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
            for block in self.blocks:
                x = block(x, cos, sin, window)
            h = self.ln_f(x)

        if not selfcode_ntp_done:
            if compute_ntp:
                h_flat = h[:, :-1, :].reshape(-1, D)
                ntp_loss, ntp_acc = self._ntp_loss_acc(h_flat, seq_repr[:, 1:], is_byte_level)
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
            pooled = self._query_embed_pool(x0, K, n_blocks, window)
        else:
            raise ValueError(f"unknown code_extract_mode {cfg.code_extract_mode!r}")
        pre_q = self._classify(pooled)
        c_i = quantize_code(pre_q, cfg)

        return c_i, ntp_loss, ntp_acc, h, query_last, query_seq


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
        decode_windows: list[list[int | None]] = []
        for i, w in enumerate(raw_windows):
            n_sources = self.n_levels - i
            if isinstance(w, (tuple, list)):
                assert len(w) == 2, f"attn_window[{i}] must be a scalar or an (encode_window, decode_window) 2-tuple, got {w!r}"
                ew, dw = w
            else:
                ew = dw = w
            windows.append(None if ew == -1 else ew)
            if isinstance(dw, (tuple, list)):
                assert len(dw) == n_sources, (
                    f"attn_window[{i}]'s decode_window must be a scalar (broadcast) or a tuple of "
                    f"length n_levels-{i}={n_sources} (one per decode source: self, +1, ..., top), got {dw!r}")
                decode_windows.append([None if x == -1 else x for x in dw])
            else:
                decode_windows.append([None if dw == -1 else dw] * n_sources)
        self.windows = windows
        self.decode_windows = decode_windows
        # Effective-codes summary (see _warn_thin_window's formula): for each level's decode, how
        # much cumulative coarser-level history its configured window(s) actually buy, in code
        # units. Printed once at construction so this is visible before a step is ever run, not
        # just as a late warning if it turns out to be thin.
        for i, dwlist in enumerate(decode_windows):
            cum_K, per_track, invisible_srcs = 1, [], []
            for src_offset, dwindow in enumerate(dwlist):
                cum_K *= cfg.Ks[i + src_offset]
                if dwindow is None:
                    per_track.append(f"K={cum_K}:full")
                else:
                    n_codes = dwindow // cum_K
                    per_track.append(f"K={cum_K}:{n_codes}codes")
                    if dwindow != 0 and n_codes == 0:
                        invisible_srcs.append(cum_K)
            print(f"decode effective codes level{i}: " + ", ".join(per_track))
            if invisible_srcs:
                print(f"WARNING: level{i} decode_window is too small for cumulative K in "
                      f"{invisible_srcs} -- 0 codes fit, that source is completely invisible to "
                      f"this level's decode (not just thin). Increase attn_window[{i}]'s "
                      f"decode_window or drop that source.")
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] encode window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
        for i, dwlist in enumerate(decode_windows):
            L = seq_lens[i]
            for src_offset, dwindow in enumerate(dwlist):
                if dwindow is not None and dwindow != 0:
                    assert L % dwindow == 0 or L <= dwindow, (
                        f"attn_window[{i}]'s decode_window[{src_offset}] ({dwindow}) must divide "
                        f"level {i}'s sequence length ({L}), or be >= it")

        if cfg.share_level_weights:
            shared_lm = LevelLM(cfg)
            encode_lms = [shared_lm for _ in range(self.n_levels)]
            decode_lms = [shared_lm for _ in range(self.n_levels)]
        else:
            encode_lms = [LevelLM(cfg) for _ in range(self.n_levels)]
            decode_lms = [LevelLM(cfg) for _ in range(self.n_levels)]
        self.encode_lms = nn.ModuleList(encode_lms)
        self.decode_lms = nn.ModuleList(decode_lms)

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True, max_decode_sources: int | None = None,
             want_next_query: bool = False):
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list = [], [], [], [], []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, loss_i, acc_i, h_i, _, _ = self.encode_lms[i](seq_repr, level=i, window=self.windows[i], compute_ntp=want_ntp)
            encode_losses.append(loss_i)
            encode_accs.append(acc_i)
            h_list.append(h_i)
            c_list.append(c_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        # keyed by k (1..len(tracks)-1): the curriculum of partial-track combos below full
        # conditioning -- {self}, {self, +1}, ..., {self, +1, ..., top-1} (full itself is
        # decode_losses[i], not repeated here). For n_levels<=2 there's only one possible k (self
        # alone), so this is identical to the old single self-only aux in that case.
        decode_self_only_losses: list[dict[int, torch.Tensor]] = [{} for _ in range(self.n_levels)]
        decode_self_only_accs: list[dict[int, torch.Tensor]] = [{} for _ in range(self.n_levels)]
        decode_derived_c: dict[int, torch.Tensor] = {}
        h_out = list(h_list)
        next_query: list[torch.Tensor | None] = [None] * self.n_levels
        query_seq_out: list[torch.Tensor | None] = [None] * self.n_levels
        for i in reversed(range(self.n_levels)):
            L_i = x_list[i].shape[1]
            tracks: list[tuple[torch.Tensor, int, int | None]] = []
            cum_K = 1
            for j in range(i, self.n_levels):
                cum_K *= cfg.Ks[j]
                window = self.decode_windows[i][j - i]
                if window == 0:
                    continue
                # Training always calls with L_i an exact multiple of cum_K (context_len is built
                # to divide evenly at every level, see __init__), so this floor check is a no-op
                # there. It only bites during generation, where L_i grows one byte at a time and is
                # rarely block-aligned -- stop adding tracks (keep whichever finer ones already
                # collected) rather than discarding everything, so e.g. a self track can still be
                # used even when a coarser track isn't affordable yet. _packed_decode_forward's own
                # prefix construction is floor-based too, so a not-yet-complete trailing block is
                # simply excluded from the code prefix, never fabricated.
                if L_i // cum_K < 1:
                    break
                if j > i and cfg.cross_track_source == "decode" and j in decode_derived_c:
                    source_c = decode_derived_c[j]
                else:
                    source_c = c_list[j]
                if cfg.quant_type == "bsq":
                    code_embeds = self.decode_lms[i].code_embed(
                        source_c if cfg.decode_code_ste else source_c.detach())
                elif cfg.decode_code_ste:
                    code_embeds = source_c @ self.decode_lms[i].embed.weight
                else:
                    code_embeds = self.decode_lms[i].embed(source_c.argmax(-1))
                tracks.append((code_embeds, cum_K, window))
            if not tracks:
                continue
            if len(tracks) == 1 and (L_i // tracks[0][1]) < 2:
                continue  # self-code decode needs 2+ blocks; treat as ragged
            full_tracks = tracks
            if max_decode_sources is not None:
                full_tracks = full_tracks[:max_decode_sources]
            c_i2, loss_i2, acc_i2, h_i2, query_last_i, query_seq_i = self.decode_lms[i](
                x_list[i], level=i, window=self.windows[i], compute_ntp=compute_ntp,
                decode_tracks=full_tracks, extra_query=(want_next_query and i == 0))
            decode_losses[i] = loss_i2
            decode_accs[i] = acc_i2
            if h_i2.shape[1] < L_i:
                # Single-track selfcode decode floor-truncates a ragged tail internally (see
                # _packed_decode_forward_selfcode) -- splice the plain encode-only h back in for
                # those trailing positions rather than leaving them missing. Never triggered during
                # training (L_i is always block-aligned there).
                h_i2 = torch.cat([h_i2, h_list[i][:, h_i2.shape[1]:, :]], dim=1)
            h_out[i] = h_i2
            next_query[i] = query_last_i
            query_seq_out[i] = query_seq_i
            if max_decode_sources is None:
                decode_derived_c[i] = c_i2
            if cfg.decode_self_only_aux and self.training and max_decode_sources is None and len(tracks) > 1:
                for k in range(1, len(tracks)):
                    _, loss_k, acc_k, _, _, _ = self.decode_lms[i](x_list[i], level=i, window=self.windows[i],
                                                                    compute_ntp=compute_ntp, decode_tracks=tracks[:k])
                    decode_self_only_losses[i][k] = loss_k
                    decode_self_only_accs[i][k] = acc_k

        return (encode_losses, encode_accs, decode_losses, decode_accs, h_out, c_list,
                decode_self_only_losses, decode_self_only_accs, next_query, decode_derived_c, query_seq_out)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        (encode_losses, encode_accs, decode_losses, decode_accs, h_list, c_list,
         decode_self_only_losses, decode_self_only_accs, _next_query, _decode_derived_c,
         _query_seq) = self._run(byte_ids)

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                              else byte_loss.new_zeros(()))
        encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

        decode_terms = [l for l in decode_losses if l is not None]
        decode_total = (cfg.decode_ntp_weight * torch.stack(decode_terms).sum() if decode_terms
                         else byte_loss.new_zeros(()))

        # mean, not sum: the curriculum can have a variable number of partial-track combos per
        # level (n_levels-2 of them at the byte level for a deep stack) -- averaging keeps
        # decode_self_only_weight's meaning stable across different n_levels instead of the loss
        # term silently growing with depth. For n_levels<=2 there's exactly one term, so mean==sum
        # and this is identical to the old behavior.
        self_only_terms = [l for d in decode_self_only_losses for l in d.values()]
        decode_self_only_total = (cfg.decode_self_only_weight * torch.stack(self_only_terms).mean()
                                   if self_only_terms else byte_loss.new_zeros(()))

        # decode_self_only_dropout: both terms above are always computed (so metrics/logging stay
        # complete), but only one of {decode_total, decode_self_only_total} enters the backward
        # loss this step, chosen by a coin flip -- rather than summing both into every step's
        # gradient (the decode_self_only_aux default). Dropped term is still in `metrics`, just
        # not in `loss`, so it gets no gradient this step.
        if cfg.decode_self_only_aux and cfg.decode_self_only_dropout and self.training and self_only_terms:
            if torch.rand(()).item() < cfg.decode_self_only_dropout_p:
                loss = encode_total + decode_self_only_total
            else:
                loss = encode_total + decode_total
        else:
            loss = encode_total + decode_total + decode_self_only_total
        ntp_total = torch.stack(encode_losses + decode_terms + self_only_terms).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_self_only_total": decode_self_only_total, "ntp_loss_total": ntp_total,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
            **{f"level{i}_ntp_loss_decode_self_k{k}": l for i, d in enumerate(decode_self_only_losses) for k, l in d.items()},
            **{f"level{i}_ntp_acc_decode_self_k{k}": a for i, d in enumerate(decode_self_only_accs) for k, a in d.items()},
        }

        return loss, metrics


def _sample_next_byte(embed_weight: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h_last, embed_weight)
    return logits.argmax(-1)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
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
        # No padding: _run/_packed_decode_forward are floor-tolerant now, so feeding the true,
        # growing byte sequence gives exactly the same decode-conditioned (or, below a level's
        # minimum block count, encode-only) representation training would compute at this position
        # -- never a fabricated trailing byte. want_next_query only matters (and is only honored)
        # on a K0-aligned prefix, where the single-track selfcode path can append a genuine
        # bare-code extra query; elsewhere it's a no-op and next_query[0] stays None.
        block_aligned = L % K0 == 0
        _, _, _, _, h_list, _, _, _, next_query, _decode_derived_c, _query_seq = model._run(
            all_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        # next_query[0]: the single-track selfcode path's genuine code-slot query (see
        # _packed_decode_forward_selfcode's extra_query docstring). h_list[0][:, -1, :]: the
        # standard byte-slot next-token representation used everywhere else (multi-track decode, or
        # the encode-only fallback when no track has a complete block yet) -- both are real,
        # trained-for slots, matching what check_gen_consistency compares against.
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, query)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                       max_decode_sources: int | None = None) -> torch.Tensor:
    """FIFO-windowed generation: truncate to the trailing cfg.context_len bytes before every step,
    exactly mirroring training's own windowing (sample_context truncates to context_len the same
    way, and RoPE positions are always relative to whatever window is fed in, never absolute from
    generation's start -- so this isn't a new mechanism, it's training's existing window applied at
    generation time). Not a per-layer K/V tensor cache -- still recomputes the windowed forward each
    step -- but bounds that recompute to O(context_len) instead of O(current total length), which is
    what actually matters once generation runs long. New byte pushed on, oldest byte falls off once
    the window is full, same push-and-drop as any fixed-size FIFO.

    Only guaranteed to match generate_no_cache while prompt_len + n_new_bytes <= context_len (same
    caveat as qcute.bytelm's own generate_kv_cache/validate_generation pair) -- beyond that point
    generate_no_cache keeps growing its unbounded context while this one keeps sliding, so the two
    are expected to diverge by design, not by bug. See validate_generation.
    """
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    K0 = model.cfg.Ks[0]
    context_len = model.cfg.context_len
    all_bytes = prompt_bytes
    for _ in range(n_new_bytes):
        window_bytes = all_bytes[:, -context_len:]   # FIFO: only the trailing context_len bytes ever matter
        L = window_bytes.shape[1]
        block_aligned = L % K0 == 0
        _, _, _, _, h_list, _, _, _, next_query, _decode_derived_c, _query_seq = model._run(
            window_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, query)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


def validate_generation(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
    """Only meaningful while prompt_len + n_new_bytes <= model.cfg.context_len -- see
    generate_kv_cache's docstring for why the two are only guaranteed to agree inside that bound."""
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


@torch.no_grad()
def generate_self_only_cond(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    return generate_no_cache(model, prompt_bytes, n_new_bytes, device, max_decode_sources=1)


@torch.no_grad()
def generate_encode_only(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encode_lms[0]
    for _ in range(n_new_bytes):
        _, _, _, h, _, _ = enc0(all_bytes, level=0, window=model.windows[0], compute_ntp=False)
        next_byte = _sample_next_byte(enc0.embed.weight, h[:, -1, :])
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
    enc0, enc1 = model.encode_lms[0], model.encode_lms[1]
    codes, _, _, _, _, _ = enc0(prompt_bytes, level=0, window=model.windows[0], compute_ntp=False)
    n_prompt_codes = codes.shape[1]
    for _ in range(n_new_codes):
        _, _, _, h1, _, _ = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
        if model.cfg.quant_type == "bsq":
            pred = enc1.code_predict(h1[:, -1, :])
            next_code = bsq_quantize(pred, model.cfg.bsq_bits)
        else:
            logits = F.linear(h1[:, -1, :], enc1.embed.weight)
            next_id = logits.argmax(-1)
            next_code = F.one_hot(next_id, num_classes=model.cfg.vocab).to(codes.dtype)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return code_to_ids(codes[0, n_prompt_codes:], model.cfg)


@torch.no_grad()
def level1_ground_truth_codes(model: "RefineLM", full_bytes: torch.Tensor, prompt_len: int, device: str) -> torch.Tensor:
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    enc0 = model.encode_lms[0]
    K0 = model.cfg.Ks[0]
    c0, _, _, _, _, _ = enc0(full_bytes, level=0, window=model.windows[0], compute_ntp=False)
    ids = code_to_ids(c0[0], model.cfg)
    n_prompt_codes = prompt_len // K0
    return ids[n_prompt_codes:]


def _annotate_bytes_with_codes(byte_ids: torch.Tensor, code_ids: torch.Tensor, K: int) -> str:
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
def _decode_source_codes(model: "RefineLM", full_bytes: torch.Tensor, device: str, level: int = -1) -> torch.Tensor:
    was_training = model.training
    model.eval()
    seq_repr = full_bytes.to(device)
    if seq_repr.dim() == 1:
        seq_repr = seq_repr.unsqueeze(0)
    c_list = []
    for i in range(model.n_levels):
        c_i, _, _, _, _, _ = model.encode_lms[i](seq_repr, level=i, window=model.windows[i], compute_ntp=False)
        c_list.append(c_i)
        seq_repr = c_i
    source_c = c_list[level]
    if was_training:
        model.train()
    return code_to_ids(source_c, model.cfg)[0]


@torch.no_grad()
def check_gen_consistency(model: "RefineLM", full_bytes: torch.Tensor, device: str,
                           prompt_len: int = 32, tol: float = 1e-3, log=print, label: str = "") -> int:
    """Cheap, reusable correctness check: the incremental no-cache generation code path and the
    one-shot teacher-forced training forward pass MUST produce identical logits when both are fed
    the same ground-truth bytes -- any gap is a genuine generation-vs-training bug (this caught
    two real ones: the windowed-attention dense-fallback and the can_chunk track-dropping bug, see
    docs/status.md), not exposure bias or model quality. Returns the mismatch count (0 = pass).
    """
    was_training = model.training
    model.eval()
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    L_total = full_bytes.shape[1]
    embed0 = model.decode_lms[0].embed.weight
    K0 = model.cfg.Ks[0]

    _, _, _, _, h_list_tf, _, _, _, _, _, query_seq_tf = model._run(
        full_bytes, compute_ntp=False, max_decode_sources=None, want_next_query=False)
    # query_seq_tf[0] is the tensor training's own NTP loss actually reads for the single-track
    # selfcode-decode path -- h_list_tf[0] (byte-slot) is WRONG there (see docs/status.md); for the
    # multi-track path query_seq_tf[0] is None and h IS correct, so fall back to it. query_seq[j]
    # predicts byte (K0+j) (it's aligned to seq_repr[:, K0:], not seq_repr[:, 1:]) -- only equal to
    # "byte j+1" when K0==1, so byte t's reference index is (t-K0) via query_seq but (t-1) via h.
    using_query_seq = query_seq_tf[0] is not None
    query_ref_tf = query_seq_tf[0] if using_query_seq else h_list_tf[0]
    logits_tf_all = F.linear(query_ref_tf[0], embed0)

    n_mismatch = 0
    for t in range(prompt_len, L_total - 1):
        ref_idx = t - K0 if using_query_seq else t - 1
        if ref_idx < 0 or ref_idx >= logits_tf_all.shape[0]:
            continue
        padded = full_bytes[:, :t]
        _, _, _, _, h_list_gen, _, _, _, next_query_gen, _, _ = model._run(
            padded, compute_ntp=False, max_decode_sources=None, want_next_query=True)
        query_gen = next_query_gen[0] if next_query_gen[0] is not None else h_list_gen[0][:, t - 1, :]
        logits_gen = F.linear(query_gen[0], embed0)
        if (logits_gen - logits_tf_all[ref_idx]).abs().max().item() >= tol:
            n_mismatch += 1
    if was_training:
        model.train()
    prefix = f"gen_consistency_{label}" if label else "gen_consistency"
    log(f"{prefix}: {n_mismatch}/{L_total - 1 - prompt_len} timesteps mismatched "
        f"(generation vs teacher-forced logits on ground-truth input)")
    return n_mismatch


def qualitative_generate(model: "RefineLM", prompt_bytes: torch.Tensor, gen_len: int,
                          ground_truth: torch.Tensor | None, device: str, log=print, label: str = "") -> None:
    prefix = f"qual_{label}_" if label else "qual_"
    out_cond_full = generate_no_cache(model, prompt_bytes, gen_len, device)
    gen_bytes_cond_full = bytes(out_cond_full[prompt_bytes.numel():].tolist())
    out_uncond = generate_encode_only(model, prompt_bytes, gen_len, device)
    gen_bytes_uncond = bytes(out_uncond[prompt_bytes.numel():].tolist())
    log(f"{prefix}prompt:              {bytes(prompt_bytes.tolist())!r}")
    if ground_truth is not None:
        log(f"{prefix}ground_truth:        {bytes(ground_truth.tolist())!r}")
    log(f"{prefix}level0_uncond:       {gen_bytes_uncond!r}")
    log(f"{prefix}level0_cond_full:    {gen_bytes_cond_full!r}")
    decode_K = 1
    for k in model.cfg.Ks:
        decode_K *= k
    # code_ids_full = _decode_source_codes(model, out_cond_full, device, level=-1)
    # n_prompt_codes = prompt_bytes.numel() // decode_K
    # gen_code_ids = code_ids_full[n_prompt_codes:]
    # annotated = _annotate_bytes_with_codes(out_cond_full[prompt_bytes.numel():], gen_code_ids, decode_K)
    # log(f"{prefix}level0_cond_full_codes: {annotated}  <{gen_code_ids.tolist()}>")
    if model.n_levels > 1:
        out_cond_self = generate_self_only_cond(model, prompt_bytes, gen_len, device)
        gen_bytes_cond_self = bytes(out_cond_self[prompt_bytes.numel():].tolist())
        log(f"{prefix}level0_cond_self:    {gen_bytes_cond_self!r}")
        # self_K = model.cfg.Ks[0]
        # code_ids_self = _decode_source_codes(model, out_cond_self, device, level=0)
        # n_prompt_codes_self = prompt_bytes.numel() // self_K
        # gen_code_ids_self = code_ids_self[n_prompt_codes_self:]
        # annotated_self = _annotate_bytes_with_codes(out_cond_self[prompt_bytes.numel():], gen_code_ids_self, self_K)
        # log(f"{prefix}level0_cond_self_codes: {annotated_self}  <{gen_code_ids_self.tolist()}>")
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


_warned_short_data: set[int] = set()


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    if len(data) < context_len and id(data) not in _warned_short_data:
        _warned_short_data.add(id(data))
        print(f"WARNING: sample_context data ({len(data)} bytes) is shorter than context_len "
              f"({context_len}) -- every batch from this split is silently truncated to {len(data)} "
              f"bytes (e.g. a val split under a large context_len). Not an error by itself (LevelLM's "
              f"single-track NTP target now matches whatever length actually comes out), but the "
              f"resulting bpb/loss numbers reflect a shorter context than configured.")
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
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
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
                    check_gen_consistency(model, window, device, prompt_len=args.qual_prompt_bytes,
                                           log=log, label=label)


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
    p.add_argument("--decode_banded", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decode_self_only_aux", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--cross_track_source", type=str, default="encode", choices=["encode", "decode"])
    p.add_argument("--decode_self_only_weight", type=float, default=1.0)
    p.add_argument("--decode_self_only_dropout", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decode_self_only_dropout_p", type=float, default=0.5)
    p.add_argument("--decode_code_ste", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--share_level_weights", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--quant_type", type=str, default="softmax", choices=["softmax", "bsq"])
    p.add_argument("--bsq_bits", type=int, default=4)

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

    _cli_excluded_config_fields: set[str] = set()
    _missing_from_cli = ({f.name for f in dataclass_fields(Config)} - {a.dest for a in p._actions}
                          - _cli_excluded_config_fields)
    assert not _missing_from_cli, (
        f"Config field(s) {_missing_from_cli} have no matching --arg registered in main()'s argparse "
        f"setup -- a config .py file setting them would be silently ignored. Add a p.add_argument(...) "
        f"for each, and pass it through to Config(...) below.")

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
        decode_banded=args.decode_banded, decode_self_only_aux=args.decode_self_only_aux,
        cross_track_source=args.cross_track_source,
        decode_self_only_weight=args.decode_self_only_weight,
        decode_self_only_dropout=args.decode_self_only_dropout,
        decode_self_only_dropout_p=args.decode_self_only_dropout_p, decode_code_ste=args.decode_code_ste,
        vocab=args.vocab, share_level_weights=args.share_level_weights,
        decode_pack_mode=args.decode_pack_mode, decode_chunked=args.decode_chunked,
        quant_type=args.quant_type, bsq_bits=args.bsq_bits,
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
