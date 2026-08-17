"""qcute_v5_bos: ARCHIVED, superseded by qcute_v5_stack.py (which dropped the qfb mechanism this
file uses in favor of a decode_bos stand-in, then fixblock/skip refinements on top -- see
qcute_v5_fixblock.py/qcute_v5_stack.py's own docstrings). Kept for historical reference only, not a
comparison target anymore. Formerly qcute_v5.py, the default v5 stack prototype: staged
cross-attention decode, genuinely sub-quadratic
windowed/banded attention, with a "query first byte" (qfb) fix folded into cross_attn_stage's
strict causal mask (code_pos < query_pos): the row that would need to predict a block's FIRST
element from that block's own just-completed code is the same row the code was derived from
(code_pos == query_pos there), so it's excluded by construction -- a code only ever gets to
condition predictions starting from a block's SECOND element onward. This is a property of the
mask itself, so it affects EVERY track at EVERY level identically (self track or any coarser
track). qcute_v5_slow.py (the prior default, kept as a reference) and qcute_v5_ws_slow.py (its
weight-sharing variant, also kept as a reference) both have this gap: their selfcode_decode
accidentally avoided it for the topmost/single-track level only, by packing the code in as its own
self-attended buffer token -- every other track had this gap silently, uncorrected.

Fix, folded directly into cross_attn_stage itself (no separate method): every LevelLM gets one
trainable D-dim decode_boundary_query vector (shared across every block boundary that stage
handles, not per-block/per-position). One extra query row per block -- fixed content, not derived
from that block's own elements -- is cross-attended against that stage's own code_kv (identical
mask/window convention already used for the main queries), sidestepping the chicken-and-egg
problem entirely (nothing computed the placeholder's content FROM the code it's attending to). Its
output replaces the broken boundary-row prediction, UNCONDITIONALLY for every block (not just
"every block except whichever happens to be last in this call") -- he_bq[b] depends only on
code_kv[0..b], never on whether block b+1 exists, so patching is always well-defined and, crucially,
LENGTH-INVARIANT: a row's patched content depends only on its own causal past, never on how much
more sequence follows within the current call -- the same append-only property plain causal
attention already has, needed to stay incremental/KV-cache-compatible (an earlier version excluded
"the last block in this call" from the patch, which made a row's content depend on total window
length and could require retroactively revising an already-computed row once a later block
completed). Every other (non-boundary) transition is untouched, already correctly handled by the
staircase mask itself.

selfcode_decode is removed entirely. RefineLM._run's decode loop is now ONE uniform per-level loop
over every track (self and every coarser track alike) calling this same cross_attn_stage -- no
track-index or level special-casing anywhere. Each stage's boundary-patched output threads forward
as the next stage's x_in, so the fix propagates through the whole chain, not just any one stage's
own loss term -- structurally the same recursive relation a U-Net decoder uses (level i's decode =
self-attend, cross with its own code, then fold in the next-coarser level's OWN decode result),
just written as this iterative loop (easier to bound/reason about) rather than literal recursion.
The old level1-only diagnostics (generate_level1_codes, generate_level1_codes_via_decode,
level1_ground_truth_codes) are generalized the same way, to generate_level_codes(level=...) etc.

    uv run python -m qcute.archive3.qcute_v5_bos --config configs/qcute_v5_stack_overfit10k_k4single.py
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
    decode_ntp_weight: float | tuple[float, ...] = 1.0   # scalar (broadcast) or one weight per level
    gumbel_tau: float = 1.0
    use_gumbel_noise: bool = False
    vocab: int = 256
    code_extract_mode: str = "last_h"
    code_head_tied: bool = False
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


class QuantScheme:
    """Uniform interface for code quantization/embedding/prediction, so LevelLM and generation code
    dispatch through one polymorphic instance (self.quant / stage_lm.quant) instead of branching on
    cfg.quant_type at every call site. SoftmaxQuant/BSQQuant below are the only two implementations;
    Config.quant_type selects which one gets built, once, in make_quant."""

    def init_modules(self, D: int, V: int, code_head_tied: bool) -> tuple:
        """-> (code_head, code_embed, code_predict) nn.Module|None, assigned onto a fresh LevelLM."""
        raise NotImplementedError

    def quantize(self, pre_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_ids(self, source_c: torch.Tensor) -> torch.Tensor:
        """Display-only: packs a bsq bit-vector into an int; argmax for softmax's one-hot."""
        raise NotImplementedError

    def embed_for_decode(self, stage_lm: "LevelLM", source_c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ntp_loss_acc(self, stage_lm: "LevelLM", h_query: torch.Tensor,
                      target_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed_input(self, stage_lm: "LevelLM", seq_repr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_next(self, stage_lm: "LevelLM", h_query: torch.Tensor, vocab: int) -> torch.Tensor:
        raise NotImplementedError


class SoftmaxQuant(QuantScheme):
    def __init__(self, tau: float, use_gumbel_noise: bool):
        self.tau = tau
        self.use_gumbel_noise = use_gumbel_noise

    def init_modules(self, D, V, code_head_tied):
        code_head = None if code_head_tied else nn.Linear(D, V, bias=False)
        if code_head is not None:
            nn.init.normal_(code_head.weight, std=0.02)
        return code_head, None, None

    def quantize(self, pre_q):
        return gumbel_quantize(pre_q, self.tau, self.use_gumbel_noise)

    def to_ids(self, source_c):
        return source_c.argmax(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return source_c @ stage_lm.embed.weight

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target = target_repr.argmax(-1).reshape(-1)
        logits = F.linear(h_query, stage_lm.embed.weight)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return seq_repr @ stage_lm.embed.weight

    def sample_next(self, stage_lm, h_query, vocab):
        logits = F.linear(h_query, stage_lm.embed.weight)
        next_id = logits.argmax(-1)
        return F.one_hot(next_id, num_classes=vocab).to(h_query.dtype)


class BSQQuant(QuantScheme):
    def __init__(self, bsq_bits: int):
        self.bsq_bits = bsq_bits

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.bsq_bits, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = CodeEmbed(self.bsq_bits, D)
        code_predict = nn.Linear(D, self.bsq_bits, bias=False)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        return bsq_quantize(pre_q, self.bsq_bits)

    def to_ids(self, source_c):
        bits = (source_c > 0).long()
        weights = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        return (bits * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target_bits = (target_repr.reshape(-1, self.bsq_bits) > 0).float()
        pred = stage_lm.code_predict(h_query)
        loss = F.binary_cross_entropy_with_logits(pred, target_bits)
        with torch.no_grad():
            acc = ((pred > 0).float() == target_bits).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        pred = stage_lm.code_predict(h_query)
        return bsq_quantize(pred, self.bsq_bits)


def make_quant(cfg: "Config") -> QuantScheme:
    if cfg.quant_type == "bsq":
        return BSQQuant(cfg.bsq_bits)
    return SoftmaxQuant(cfg.gumbel_tau, cfg.use_gumbel_noise)


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
    causal & (i-j<window) mask -- see docs/status.md's windowed-attention efficiency review
    (ported unchanged from qcute_v5_concat_eff.py).

    Fast path requires T % window == 0 (always true during training, where context_len is built
    to divide evenly -- see RefineLM.__init__); falls back to the dense mask (correct for any T,
    same as the original qcute_v5_stack.py) otherwise, which no-cache generation's ragged, growing
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
        self.d_model = d_model
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

    def forward_cross(self, x_q: torch.Tensor, x_kv: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor,
                       cos_k: torch.Tensor, sin_k: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, hd = self.n_heads, self.head_dim
        Wq, Wk, Wv = self.qkv.weight[:D], self.qkv.weight[D:2 * D], self.qkv.weight[2 * D:3 * D]
        q = F.linear(x_q, Wq).view(B, T, H, hd).transpose(1, 2)
        k = F.linear(x_kv, Wk).view(B, S, H, hd).transpose(1, 2)
        v = F.linear(x_kv, Wv).view(B, S, H, hd).transpose(1, 2)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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

    def forward_cross(self, x: torch.Tensor, code_kv: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor,
                       cos_k: torch.Tensor, sin_k: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        xn = self.ln1(x)
        coden = self.ln1(code_kv)
        a = self.attn.forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


_warned_thin_window: set[tuple[int, ...]] = set()


def _warn_thin_window(tracks: list[tuple[torch.Tensor, int, int | None]], window: int, min_codes: int = 2) -> None:
    """Cumulative code coverage a windowed decode actually buys, across all tracks (levels):
    codes_in_window[j] ~= window // K_j, summed. Below min_codes the window is thinner than the
    minimum useful conditioning (2 codes = the self-code LM-continuation floor used everywhere
    else). Not a correctness check; due diligence is on the caller, ported from qcute_v5_concat_eff.py."""
    key = tuple(K for _, K, _ in tracks) + (window,)
    if key in _warned_thin_window:
        return
    total_codes = sum(max(0, window // K) for _, K, _ in tracks)
    if total_codes < min_codes:
        _warned_thin_window.add(key)
        Ks_str = ",".join(str(K) for _, K, _ in tracks)
        print(f"WARNING: windowed decode window={window} covers only ~{total_codes} cumulative code(s) "
              f"across tracks Ks=({Ks_str}) -- below min_codes={min_codes}. Coarser-level context is "
              f"likely starved; consider a larger attn_window decode_window for these levels.")


class LevelLM(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.quant = make_quant(cfg)
        D = cfg.d_model
        V = cfg.vocab
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.code_head, self.code_embed, self.code_predict = self.quant.init_modules(D, V, cfg.code_head_tied)
        self.code_query = self.code_out = self.query_embed = None
        if cfg.code_extract_mode == "light_query_attn":
            self.code_query = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.code_query, std=0.02)
            self.code_out = nn.Linear(D, D, bias=False)
        elif cfg.code_extract_mode == "query_embed":
            self.query_embed = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.query_embed, std=0.02)
        # qfb: one shared placeholder query, cross-attended against whichever code_kv this
        # instance's own cross_attn_stage call uses, to predict that block's FIRST element.
        self.decode_boundary_query = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.decode_boundary_query, std=0.02)

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

    def _ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor, is_byte_level: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if is_byte_level:
            target = target_repr.reshape(-1)
            logits = F.linear(h_query, self.embed.weight)
            loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
            return loss, acc
        return self.quant.ntp_loss_acc(self, h_query, target_repr)

    def _ntp(self, h: torch.Tensor, seq_repr: torch.Tensor, is_byte_level: bool, D: int,
             compute_ntp: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            ntp_loss, ntp_acc = self._ntp_loss_acc(h_flat, seq_repr[:, 1:], is_byte_level)
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())
        return ntp_loss, ntp_acc

    def _extract_code(self, h: torch.Tensor, x0: torch.Tensor, K: int, window: int | None) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = h.shape
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
        return self.quant.quantize(pre_q)

    def _embed_input(self, seq_repr: torch.Tensor, is_byte_level: bool) -> torch.Tensor:
        if is_byte_level:
            return self.embed(seq_repr)
        return self.quant.embed_input(self, seq_repr)

    def encode(self, seq_repr: torch.Tensor, level: int, window: int | None,
               compute_ntp: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        K = cfg.Ks[level]
        D = cfg.d_model
        is_byte_level = level == 0
        x = self._embed_input(seq_repr, is_byte_level)
        B, L = seq_repr.shape if is_byte_level else seq_repr.shape[:2]
        x0 = x
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, window)
        h = self.ln_f(x)
        ntp_loss, ntp_acc = self._ntp(h, seq_repr, is_byte_level, D, compute_ntp)
        c_i = self._extract_code(h, x0, K, window)
        return c_i, ntp_loss, ntp_acc, h

    def cross_attn_stage(self, x_in: torch.Tensor, code_kv: torch.Tensor, seq_repr: torch.Tensor, level: int,
                          track_K: int, window: int | None, compute_ntp: bool = True, want_code: bool = False
                          ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Byte-granularity queries cross-attend to coarser, track_K-strided code keys (STRICT
        causal: code_pos < query_pos, since a code at position track_K-1 is only available once
        that whole block has been seen, so it can't condition queries inside its own block).
        Dense mask is a staircase, not a triangle: code keys advance one per track_K query steps,
        so every block of track_K consecutive queries sees the identical key set.

        Chunked fast path (O(L*window/track_K), new -- no dense-mask original to port from, unlike
        the self-attention paths above; see docs/status.md's windowed-attention efficiency review)
        applies when window is set, L % track_K == 0, and L // track_K == n_blocks (all true
        during training; generation's growing/ragged L falls back to the dense mask, unchanged
        from the original qcute_v5_stack.py). Buckets queries into blocks of track_K (one bucket
        per code-key stride) and gathers, per bucket, only the bounded run of preceding code keys
        within `window` -- bit-identical to the dense causal & (reach<window) mask.

        qfb (query first byte) patch: the strict `code_pos < query_pos` mask above means the row
        that would need to predict a block's FIRST element from that block's own just-completed
        code is the SAME row the code was derived from (code_pos == query_pos there) -- excluded by
        construction, so a code only ever conditions predictions from a block's SECOND element
        onward. Applies identically to every track at every level (self or coarser), not just the
        self track, since it's a property of this mask alone. Fixed with one extra cross-attention
        query PER BLOCK, unconditionally (all n_blocks of them, not "every block except whichever
        happens to be last in this call"): self.decode_boundary_query, a SHARED fixed vector (not
        derived from that block's own content, so no chicken-and-egg problem), cross-attended
        against code_kv with the same causal/window convention as above -- he_bq[b] depends only on
        code_kv[0..b] (strictly backward-looking), never on whether block b+1 exists, so patching
        every block's row is always well-defined and, crucially, LENGTH-INVARIANT: a given row's
        patched content depends only on its own causal past, never on how much more sequence
        follows it in this particular call -- the same append-only property plain causal attention
        already has, needed for this to stay incremental/KV-cache-compatible (an earlier version
        patched only "internal" blocks, excluding whichever was last-in-this-call, which made a
        row's content depend on total window length and could require retroactively revising an
        already-computed row once a later block completed -- not just suboptimal, incompatible with
        incremental generation). Its output replaces the broken boundary row in `h`; every other
        (non-boundary) transition is untouched, already correctly handled by the staircase mask
        itself. Whether a patched row has a real loss target is a separate, already-handled
        concern -- the caller's query_seq[:, :-1, :] drops only the sequence's own trailing row.
        """
        cfg = self.cfg
        K = cfg.Ks[level]
        D = cfg.d_model
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
            causal = code_pos.view(1, -1) < query_pos.view(-1, 1)
            if window is not None:
                reach = query_pos.view(-1, 1) - code_pos.view(1, -1)
                allow = causal & (reach < window)
            else:
                allow = causal
            attn_mask = allow.view(1, 1, L, n_blocks)

            x = x_in
            for block in self.blocks:
                x = block.forward_cross(x, code_kv, cos_q, sin_q, cos_k, sin_k, attn_mask)
            h = self.ln_f(x)
        else:
            # Query bucket size = codes_per_chunk*track_K (<=window), not track_K alone (often tiny
            # -- K=1..4 -- which floods SDPA with thousands of tiny batched calls, same issue as
            # qcute_v5_concat_eff.py's banded path). Codes advance in lockstep with queries at rate
            # 1 code per track_K bytes, so a query chunk of qbucket=codes_per_chunk*track_K bytes
            # consumes exactly codes_per_chunk new codes -- keeping qbucket a clean multiple of
            # track_K lets the code side use the SAME chunked-gather trick (shift by 1 chunk index
            # per step) as the query side, at CODE-CHUNK granularity instead of single-code
            # granularity. Correctness comes entirely from the code_pos/query_pos-based mask below,
            # not from the chunk size matching track_K.
            _warn_thin_window([(code_kv, track_K, window)], window)
            codes_per_chunk = max(1, window // track_K)
            qbucket = codes_per_chunk * track_K
            n_chunks = -(-L // qbucket)
            pad_len = n_chunks * qbucket - L
            Lp = n_chunks * qbucket
            n_prev_chunks = min(max(1, -(-window // qbucket) + 1), n_chunks)   # code-chunks of lookback

            query_pos_p = query_pos if pad_len == 0 else F.pad(query_pos, (0, pad_len), value=-10 ** 9)
            cos_q, sin_q = rope_cos_sin_for_positions(query_pos_p.clamp(min=0), hd, cfg.rope_base, device)

            # Code side padded to n_chunks*codes_per_chunk (a code-chunk per query-chunk, 1:1 rate)
            n_code_slots = n_chunks * codes_per_chunk
            code_pad_len = n_code_slots - n_blocks
            code_pos_p = code_pos if code_pad_len <= 0 else F.pad(code_pos, (0, code_pad_len), value=-10 ** 9)
            code_pos_p = code_pos_p[:n_code_slots]
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_p.clamp(min=0), hd, cfg.rope_base, device)

            pos_b = code_pos_p.view(n_chunks, codes_per_chunk)
            # n_prev_chunks-1 pad chunks, not n_prev_chunks: the gather must include chunk c's OWN
            # code-chunk, not just strictly-previous ones -- codes produced mid-chunk are still
            # causally visible to later queries within that same chunk (only the mask, not chunk
            # boundaries, does the fine-grained within-chunk filtering).
            pad_pos = torch.full((n_prev_chunks - 1, codes_per_chunk), -10 ** 9, device=device, dtype=pos_b.dtype)
            pos_ext = torch.cat([pad_pos, pos_b], dim=0)
            idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
                   + torch.arange(n_prev_chunks, device=device).view(1, n_prev_chunks))   # code-CHUNK indices
            Kc = n_prev_chunks * codes_per_chunk
            pos_win = pos_ext[idx].reshape(n_chunks, Kc)

            qpos_b = query_pos_p.view(n_chunks, qbucket)
            ti = qpos_b.unsqueeze(-1)      # [n_chunks, qbucket, 1]
            tj = pos_win.unsqueeze(1)      # [n_chunks, 1, Kc]
            allow = (tj < ti) & (ti - tj < window)
            mask_flat = allow.view(1, n_chunks, 1, qbucket, Kc).expand(
                B, n_chunks, 1, qbucket, Kc).reshape(B * n_chunks, 1, qbucket, Kc)

            x = x_in if pad_len == 0 else F.pad(x_in, (0, 0, 0, pad_len))
            coden_pad_len = n_code_slots - n_blocks
            for block in self.blocks:
                xn = block.ln1(x)
                coden = block.ln1(code_kv)
                if coden_pad_len > 0:
                    coden = F.pad(coden, (0, 0, 0, coden_pad_len))
                Wq = block.attn.qkv.weight[:D]
                Wk = block.attn.qkv.weight[D:2 * D]
                Wv = block.attn.qkv.weight[2 * D:3 * D]

                q = F.linear(xn, Wq).view(B, Lp, H, hd).transpose(1, 2)                  # [B,H,Lp,hd]
                k = F.linear(coden, Wk).view(B, n_code_slots, H, hd).transpose(1, 2)     # [B,H,n_code_slots,hd]
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
            h = self.ln_f(x)[:, :L, :]

        # qfb boundary patch (see docstring): one extra cross-attention query per block, fixed
        # content, cross-attended against code_kv with the identical causal/window convention used
        # above. code_pos/bq_pos reuse the SAME n_blocks/track_K this stage already computed.
        bq_pos = code_pos + 1   # position of the block's first element -- the transition this query stands in for
        cos_qb, sin_qb = rope_cos_sin_for_positions(bq_pos, hd, cfg.rope_base, device)
        cos_kb, sin_kb = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device)
        bq_causal = code_pos.view(1, -1) < bq_pos.view(-1, 1)
        if window is not None:
            bq_reach = bq_pos.view(-1, 1) - code_pos.view(1, -1)
            bq_allow = bq_causal & (bq_reach < window)
        else:
            bq_allow = bq_causal
        bq_mask = bq_allow.view(1, 1, n_blocks, n_blocks)

        bq = self.decode_boundary_query.view(1, 1, D).expand(B, n_blocks, D)
        for block in self.blocks:
            bq = block.forward_cross(bq, code_kv, cos_qb, sin_qb, cos_kb, sin_kb, bq_mask)
        he_bq = self.ln_f(bq)   # (B, n_blocks, D): he_bq[:, b, :] predicts block b+1's first element

        # EVERY block's boundary row (code_pos[b], for ALL b, not just "every block except
        # whichever happens to be last") gets patched unconditionally -- he_bq[b] only ever
        # depends on code_kv[0..b] (see the mask above: strictly backward-looking), never on
        # whether block b+1 exists, so there's no reason to special-case "the last currently-
        # visible block" here. Excluding it was a real bug, not just a missed case: it made this
        # row's content a function of the TOTAL window length L (patched when more blocks follow
        # within L, unpatched when block b happens to be the tail of a SHORTER L) -- length-
        # dependent in a way plain causal attention never is, and specifically incompatible with
        # incremental/KV-cache generation (a row could need retroactive revision once a later
        # block completes). Whether a row's transition has a real LOSS target is a separate,
        # already-handled concern: query_seq drops the sequence's own last row before the loss sum
        # (below), and even a non-block-aligned trailing patch is *correct* when it happens to fall
        # on a real ragged-tail byte (predicting an actual next byte, not a "virtual" one).
        patched_h = h.clone()
        patched_h[:, code_pos, :] = he_bq

        query_seq = patched_h[:, :-1, :]
        if compute_ntp:
            ntp_loss, ntp_acc = self._ntp_loss_acc(query_seq.reshape(-1, D), seq_repr[:, 1:], is_byte_level)
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        c_i = None
        if want_code:
            # Deliberately extract from the UNPATCHED h, not patched_h: code_extract_mode="last_h"
            # (the default) pools each block's code from the row at that block's OWN last element --
            # exactly the row patched_h overwrites with he_bq (a query answering a different
            # question, "what predicts the NEXT block's first element"). Reading patched_h here
            # would make block b's own extracted code a function of block b's own code
            # (self-referential, corrupts decode_derived_c for every downstream consumer). Also
            # deliberately cfg.Ks[level] (this level's own granularity), not track_K -- unchanged
            # from the original, since c_i is this LEVEL's own code regardless of which track's
            # code_kv this particular stage cross-attended against.
            x0 = self._embed_input(seq_repr, is_byte_level)
            c_i = self._extract_code(h, x0, K, window)

        query_last = he_bq[:, -1, :]   # the boundary query for the position right after seq_repr's
        # last element -- only meaningful to a caller when L is itself an exact multiple of track_K
        # (block_aligned); always computed since it's a cheap byproduct of the patch above.
        return c_i, ntp_loss, ntp_acc, patched_h, query_last, query_seq

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
        # units. Printed once at construction, ported from qcute_v5_concat_eff.py.
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

        self.encode_lms = nn.ModuleList([LevelLM(cfg) for _ in range(self.n_levels)])
        self.decode_stage_lms = nn.ModuleList([
            nn.ModuleList([LevelLM(cfg) for _ in range(self.n_levels - i)])
            for i in range(self.n_levels)
        ])

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True, max_decode_sources: int | None = None,
             want_next_query: bool = False):
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list = [], [], [], [], []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, loss_i, acc_i, h_i = self.encode_lms[i].encode(seq_repr, level=i, window=self.windows[i], compute_ntp=want_ntp)
            encode_losses.append(loss_i)
            encode_accs.append(acc_i)
            h_list.append(h_i)
            c_list.append(c_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        decode_stage_extra_losses: list = []
        decode_derived_c: dict[int, torch.Tensor] = {}
        h_out = list(h_list)
        next_query: list[torch.Tensor | None] = [None] * self.n_levels
        query_seq_out: list[torch.Tensor | None] = [None] * self.n_levels
        final_embed_weight: list[torch.Tensor | None] = [None] * self.n_levels
        for i in reversed(range(self.n_levels)):
            L_i = x_list[i].shape[1]
            track_specs: list[tuple[torch.Tensor, int, int | None]] = []
            cum_K = 1
            for j in range(i, self.n_levels):
                cum_K *= cfg.Ks[j]
                window = self.decode_windows[i][j - i]
                if window == 0:
                    continue
                if L_i // cum_K < 1:
                    break
                if j > i and j in decode_derived_c:
                    source_c = decode_derived_c[j]
                else:
                    source_c = c_list[j]
                track_specs.append((source_c, cum_K, window))
            if not track_specs:
                continue
            full_track_specs = track_specs
            if max_decode_sources is not None:
                full_track_specs = full_track_specs[:max_decode_sources]

            # Every stage -- self track (t=0) or any coarser track -- goes through the SAME
            # cross_attn_stage call: no track/level special-casing anywhere in this loop. Each
            # stage's boundary-patched output feeds forward as the next stage's x_in, so the qfb fix
            # propagates through the whole staged chain, not just any one stage's own loss term.
            stage_lms_i = self.decode_stage_lms[i]
            x = h_list[i]
            loss_final = acc_final = c_final = query_last_final = query_seq_final = None
            for t, (source_c, track_K, window) in enumerate(full_track_specs):
                stage_lm = stage_lms_i[t]
                code_embeds = stage_lm.quant.embed_for_decode(stage_lm, source_c)
                is_last = t == len(full_track_specs) - 1
                c_stage, loss_stage, acc_stage, h_stage, query_last, query_seq = stage_lm.cross_attn_stage(
                    x, code_embeds, x_list[i], i, track_K, window, compute_ntp=compute_ntp, want_code=is_last)
                x = h_stage
                if is_last:
                    loss_final, acc_final, c_final = loss_stage, acc_stage, c_stage
                    query_last_final, query_seq_final = query_last, query_seq
                    final_embed_weight[i] = stage_lm.embed.weight
                else:
                    decode_stage_extra_losses.append(loss_stage)
            decode_losses[i] = loss_final
            decode_accs[i] = acc_final
            h_out[i] = x
            if i == 0:
                # query_seq_final is always valid (every block's boundary row is unconditionally
                # patched now, see cross_attn_stage's docstring); needed by check_gen_consistency's
                # teacher-forced reference even when want_next_query=False, so this is NOT gated on
                # want_next_query.
                query_seq_out[i] = query_seq_final
                if want_next_query and L_i % full_track_specs[-1][1] == 0:
                    # query_last_final (=he_bq[:, -1, :]) predicts "byte code_pos[-1]+1" -- only
                    # equal to "byte L_i" (the actual next unknown byte) when the LAST stage's own
                    # track_K divides L_i evenly. A coarser last track (track_K > K0) rarely
                    # satisfies this even when the byte-level sequence happens to be K0-aligned;
                    # when it doesn't, leave next_query[0] as None so callers fall back to
                    # h_out[0][:, -1, :] instead of trusting a query_last that predicts an earlier,
                    # already-known position.
                    next_query[i] = query_last_final
            if max_decode_sources is None:
                decode_derived_c[i] = c_final

        return (encode_losses, encode_accs, decode_losses, decode_accs, h_out, c_list,
                decode_stage_extra_losses, final_embed_weight,
                next_query, decode_derived_c, query_seq_out)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        (encode_losses, encode_accs, decode_losses, decode_accs, h_list, c_list,
         decode_stage_extra_losses,
         _final_embed_weight, _next_query, _decode_derived_c, _query_seq) = self._run(byte_ids)

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                              else byte_loss.new_zeros(()))
        encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

        decode_ntp_weight = (cfg.decode_ntp_weight if isinstance(cfg.decode_ntp_weight, (tuple, list))
                              else (cfg.decode_ntp_weight,) * self.n_levels)
        assert len(decode_ntp_weight) == self.n_levels, (
            f"decode_ntp_weight tuple must have length n_levels={self.n_levels}, got {len(decode_ntp_weight)}")
        decode_terms = [decode_ntp_weight[i] * l for i, l in enumerate(decode_losses) if l is not None]
        decode_total = torch.stack(decode_terms).sum() if decode_terms else byte_loss.new_zeros(())

        # decode_stage_extra_losses aren't tied to one specific level index (non-final tracks in a
        # multi-track chain), so a per-level decode_ntp_weight tuple doesn't map onto them cleanly
        # -- use the mean weight.
        decode_stage_extra_weight = sum(decode_ntp_weight) / len(decode_ntp_weight)
        decode_stage_extra_total = (decode_stage_extra_weight * torch.stack(decode_stage_extra_losses).sum()
                                     if decode_stage_extra_losses else byte_loss.new_zeros(()))

        loss = encode_total + decode_total + decode_stage_extra_total
        ntp_total = torch.stack(encode_losses + [l for l in decode_losses if l is not None]
                                 + decode_stage_extra_losses).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_stage_extra_total": decode_stage_extra_total, "ntp_loss_total": ntp_total,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
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
        # No padding: _run/cross_attn_stage are floor-tolerant now, so feeding the true, growing
        # byte sequence gives exactly the same decode-conditioned (or, below a level's minimum
        # block count, encode-only) representation training would compute at this position --
        # never a fabricated trailing byte. want_next_query only matters (and is only honored) when
        # L is a multiple of the FINAL decode stage's own track_K, where cross_attn_stage's
        # boundary-query patch reaches the sequence's trailing row too (see its docstring);
        # elsewhere it's a no-op and next_query[0] stays None.
        block_aligned = L % K0 == 0
        _, _, _, _, h_list, _, _, final_embed_weight, next_query, _decode_derived_c, _query_seq = model._run(
            all_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        embed_w = final_embed_weight[0] if final_embed_weight[0] is not None else model.encode_lms[0].embed.weight
        # next_query[0]: the final decode stage's genuine boundary-patched query (see
        # cross_attn_stage's docstring). h_list[0][:, -1, :]: the standard byte-slot next-token
        # representation used everywhere else (whenever next_query[0] wasn't valid, or the
        # encode-only fallback when no track has a complete block yet) -- both are real,
        # trained-for slots, matching what check_gen_consistency compares against.
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(embed_w, query)
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
    generation's start). Not a per-layer K/V tensor cache -- still recomputes the windowed forward
    each step -- but bounds that recompute to O(context_len) instead of O(current total length).
    New byte pushed on, oldest byte falls off once the window is full. Ported from
    qcute_v5_concat_eff.py; only guaranteed to match generate_no_cache while
    prompt_len + n_new_bytes <= context_len, see validate_generation.
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
        _, _, _, _, h_list, _, _, final_embed_weight, next_query, _decode_derived_c, _query_seq = model._run(
            window_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        embed_w = final_embed_weight[0] if final_embed_weight[0] is not None else model.encode_lms[0].embed.weight
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(embed_w, query)
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
def generate_true_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Genuine incremental per-layer K/V cache -- unlike generate_kv_cache (FIFO window
    re-truncation: recomputes the full windowed forward from scratch every step), this persists
    attention K/V tensors across steps and only does O(1) new-token/new-code work per step.
    Exploits RoPE's shift-invariance (attention depends only on relative position offset, see
    docs/kv_cache_design.md) to use absolute, never-reset positions instead of window-relative
    positions -- correctness is otherwise identical to the dense/windowed formulas in `encode` and
    `cross_attn_stage`, just replayed one token at a time with cached K/V instead of recomputed.

    Scoped to n_levels==1 (single level, one self track, no cross-level hierarchy) and
    code_extract_mode='last_h' -- the general multi-track/multi-level case needs cache bookkeeping
    at each level's own rate plus the qfb boundary-query stream at each level, deferred as future
    work (see docs/kv_cache_design.md). Ordering per position `pos` matters: the self track's main
    cross-attention path must see only codes with code_pos < pos (STRICT, matching
    cross_attn_stage's causal mask) so it's computed BEFORE this position's own code (if any) is
    appended; the qfb boundary-query path (bq_pos = pos+1) needs that same code visible, so it runs
    AFTER the append. Masking is done explicitly by exact position comparison (not implicit via
    cache eviction order) so correctness never depends on eviction timing -- eviction here is a
    pure performance optimization, safe any time after a code becomes permanently unreachable.
    """
    assert model.n_levels == 1, "generate_true_kv_cache: only n_levels==1 (single-level) supported so far"
    cfg = model.cfg
    assert cfg.code_extract_mode == "last_h", "generate_true_kv_cache: only code_extract_mode='last_h' supported so far"
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.shape[0]
    K = cfg.Ks[0]
    D = cfg.d_model
    H = cfg.n_heads
    hd = D // H
    enc = model.encode_lms[0]
    dec = model.decode_stage_lms[0][0]
    enc_window = model.windows[0]
    dec_window = model.decode_windows[0][0]
    n_layers = len(enc.blocks)

    enc_k: list = [None] * n_layers
    enc_v: list = [None] * n_layers
    dec_k: list = [None] * n_layers
    dec_v: list = [None] * n_layers
    code_positions = torch.zeros(0, dtype=torch.long, device=device)

    def encode_step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = enc.embed(byte_id).unsqueeze(1)
        q_pos = torch.tensor([pos], device=device)
        cos, sin = rope_cos_sin_for_positions(q_pos, hd, cfg.rope_base, device)
        for li, block in enumerate(enc.blocks):
            xn = block.ln1(x)
            qkv = block.attn.qkv(xn).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
            qn, kn, vn = qkv[0], qkv[1], qkv[2]
            qn = apply_rope(qn, cos, sin)
            kn = apply_rope(kn, cos, sin)
            enc_k[li] = kn if enc_k[li] is None else torch.cat([enc_k[li], kn], dim=2)
            enc_v[li] = vn if enc_v[li] is None else torch.cat([enc_v[li], vn], dim=2)
            if enc_window is not None and enc_k[li].shape[2] > enc_window:
                enc_k[li] = enc_k[li][:, :, -enc_window:]
                enc_v[li] = enc_v[li][:, :, -enc_window:]
            y = F.scaled_dot_product_attention(qn, enc_k[li], enc_v[li])
            x = x + block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
            x = x + block.mlp(block.ln2(x))
        return enc.ln_f(x)

    def code_mask(query_pos: int, strict_less: bool) -> torch.Tensor | None:
        if code_positions.numel() == 0:
            return None
        causal = code_positions < query_pos if strict_less else code_positions <= query_pos
        if dec_window is not None:
            reach = query_pos - code_positions
            causal = causal & (reach < dec_window)
        return causal.view(1, 1, 1, -1)

    def decode_cross(x_query: torch.Tensor, query_pos: int, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        x = x_query
        q_pos_t = torch.tensor([query_pos], device=device)
        cos_q, sin_q = rope_cos_sin_for_positions(q_pos_t, hd, cfg.rope_base, device)
        for li, block in enumerate(dec.blocks):
            xn = block.ln1(x)
            qn = F.linear(xn, block.attn.qkv.weight[:D]).view(B, 1, H, hd).transpose(1, 2)
            qn = apply_rope(qn, cos_q, sin_q)
            y = F.scaled_dot_product_attention(qn, dec_k[li], dec_v[li], attn_mask=mask)
            x = x + block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
            x = x + block.mlp(block.ln2(x))
        return dec.ln_f(x)

    def append_code(h_new: torch.Tensor, code_pos: int) -> None:
        nonlocal code_positions
        pooled = h_new.squeeze(1)
        pre_q = enc._classify(pooled)
        c_new = enc.quant.quantize(pre_q).unsqueeze(1)
        code_embed = dec.quant.embed_for_decode(dec, c_new)
        k_pos_t = torch.tensor([code_pos], device=device)
        cos_k, sin_k = rope_cos_sin_for_positions(k_pos_t, hd, cfg.rope_base, device)
        for li, block in enumerate(dec.blocks):
            coden = block.ln1(code_embed)
            kn = F.linear(coden, block.attn.qkv.weight[D:2 * D]).view(B, 1, H, hd).transpose(1, 2)
            vn = F.linear(coden, block.attn.qkv.weight[2 * D:3 * D]).view(B, 1, H, hd).transpose(1, 2)
            kn = apply_rope(kn, cos_k, sin_k)
            dec_k[li] = kn if dec_k[li] is None else torch.cat([dec_k[li], kn], dim=2)
            dec_v[li] = vn if dec_v[li] is None else torch.cat([dec_v[li], vn], dim=2)
        code_positions = torch.cat([code_positions, k_pos_t])
        if dec_window is not None:
            keep = (code_pos + 1 - code_positions) < dec_window
            if not keep.all():
                for li in range(n_layers):
                    dec_k[li] = dec_k[li][:, :, keep]
                    dec_v[li] = dec_v[li][:, :, keep]
                code_positions = code_positions[keep]

    all_bytes = prompt_bytes
    L0 = all_bytes.shape[1]
    embed_w = dec.embed.weight
    for pos in range(L0 + n_new_bytes - 1):
        byte_id = all_bytes[:, pos]
        h_new = encode_step(byte_id, pos)
        main_out = decode_cross(h_new, pos, code_mask(pos, strict_less=True))
        boundary_out = None
        if (pos + 1) % K == 0:
            append_code(h_new, pos)
            bq = dec.decode_boundary_query.view(1, 1, D).expand(B, 1, D)
            boundary_out = decode_cross(bq, pos + 1, code_mask(pos + 1, strict_less=True))
        if pos >= L0 - 1:
            query = boundary_out if boundary_out is not None else (main_out if main_out is not None else h_new)
            next_byte = _sample_next_byte(embed_w, query.squeeze(1))
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


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
        _, _, _, h = enc0.encode(all_bytes, level=0, window=model.windows[0], compute_ntp=False)
        next_byte = _sample_next_byte(enc0.embed.weight, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


def _encode_up_to(model: "RefineLM", seq_repr: torch.Tensor, level: int) -> torch.Tensor:
    """Walks the encode chain 0..level-1 (bytes -> level0 codes -> level1 codes -> ...), returning
    level (level-1)'s own code sequence -- the input sequence level `level`'s own LM consumes.
    level=0 returns seq_repr unchanged (bytes themselves)."""
    for j in range(level):
        seq_repr, _, _, _ = model.encode_lms[j].encode(seq_repr, level=j, window=model.windows[j], compute_ntp=False)
    return seq_repr


@torch.no_grad()
def generate_level_codes(model: "RefineLM", prompt_bytes: torch.Tensor, level: int, n_new_codes: int,
                          device: str) -> torch.Tensor:
    """Generalizes the old level1-only generate_level1_codes to any level >= 1: uses level
    `level`'s own LM (encode_lms[level]) to autoregressively extend level (level-1)'s code
    sequence -- predicting NEW level-(level-1) code symbols one at a time via level `level`'s
    ENCODE pass, as opposed to generate_level_codes_via_decode's DECODE readout. level=0 has no
    coarser LM to generate its own (byte) input sequence -- use generate_no_cache/kv_cache there."""
    assert level >= 1, "generate_level_codes needs level >= 1 -- level 0 predicts bytes, see generate_no_cache"
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    codes = _encode_up_to(model, prompt_bytes, level)
    n_prompt_codes = codes.shape[1]
    enc_level = model.encode_lms[level]
    for _ in range(n_new_codes):
        _, _, _, h = enc_level.encode(codes, level=level, window=model.windows[level], compute_ntp=False)
        next_code = enc_level.quant.sample_next(enc_level, h[:, -1, :], model.cfg.vocab)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return enc_level.quant.to_ids(codes[0, n_prompt_codes:])


@torch.no_grad()
def generate_level_codes_via_decode(model: "RefineLM", prompt_bytes: torch.Tensor, level: int, n_new_codes: int,
                                     device: str) -> torch.Tensor:
    """Generalizes the old level1-only generate_level1_codes_via_decode to any level >= 1: level
    `level`'s DECODE-based (self-track cross_attn_stage) code generation, as opposed to
    generate_level_codes's ENCODE-based generation -- tests whether level `level`'s own decode
    readout (decode_losses[level], teacher-forced acc = val_level{level}_ntp_acc_decode) is a
    coherent free-running generator, not just a good teacher-forced predictor."""
    assert level >= 1, "generate_level_codes_via_decode needs level >= 1"
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    cfg = model.cfg
    codes = _encode_up_to(model, prompt_bytes, level)
    n_prompt_codes = codes.shape[1]
    stage_lm = model.decode_stage_lms[level][0]
    enc_level = model.encode_lms[level]
    K = cfg.Ks[level]
    window = model.decode_windows[level][0]
    for _ in range(n_new_codes):
        L = codes.shape[1]
        if L // K < 1:
            # cross_attn_stage needs at least 1 complete block at this level -- too little
            # coarser-code context yet (short qual_prompt_bytes relative to this Ks's depth). Stop
            # rather than pad/crash; whatever's accumulated so far (possibly nothing) is returned
            # as-is, same "insufficient" semantics as _run's track-building.
            break
        pad_len = (-L) % K
        padded = (torch.cat([codes, codes.new_zeros(codes.shape[0], pad_len, codes.shape[2])], dim=1)
                  if pad_len > 0 else codes)
        c_lvl, _, _, h_lvl = enc_level.encode(padded, level=level, window=model.windows[level], compute_ntp=False)
        code_embeds = stage_lm.quant.embed_for_decode(stage_lm, c_lvl)
        _, _, _, _, query_last, _ = stage_lm.cross_attn_stage(
            h_lvl, code_embeds, padded, level, K, window, compute_ntp=False, want_code=False)
        if pad_len != 0:
            # query_last is only a valid "predict the symbol right after `padded`" query when
            # `padded`'s length is itself an exact multiple of K -- same block_aligned gate
            # cross_attn_stage checks internally before patching its trailing row (see its
            # docstring) and RefineLM._run applies for next_query[0].
            break
        next_code = stage_lm.quant.sample_next(stage_lm, query_last, cfg.vocab)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return stage_lm.quant.to_ids(codes[0, n_prompt_codes:])


@torch.no_grad()
def level_ground_truth_codes(model: "RefineLM", full_bytes: torch.Tensor, level: int, prompt_len: int,
                              device: str) -> torch.Tensor:
    """Generalizes the old level1-only level1_ground_truth_codes to any level >= 1: the TRUE
    level-(level-1) code sequence (from a full encode pass, not generation) -- the ground truth
    generate_level_codes/generate_level_codes_via_decode are trying to predict."""
    assert level >= 1, "level_ground_truth_codes needs level >= 1"
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    codes = _encode_up_to(model, full_bytes, level)
    cum_K = 1
    for j in range(level):
        cum_K *= model.cfg.Ks[j]
    ids = model.encode_lms[level - 1].quant.to_ids(codes[0])
    n_prompt_codes = prompt_len // cum_K
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
        c_i, _, _, _ = model.encode_lms[i].encode(seq_repr, level=i, window=model.windows[i], compute_ntp=False)
        c_list.append(c_i)
        seq_repr = c_i
    source_c = c_list[level]
    if was_training:
        model.train()
    return model.encode_lms[level].quant.to_ids(source_c)[0]


@torch.no_grad()
def check_gen_consistency(model: "RefineLM", full_bytes: torch.Tensor, device: str,
                           prompt_len: int = 32, tol: float = 1e-3, log=print, label: str = "") -> int:
    """Cheap, reusable correctness check: the incremental no-cache generation code path and the
    one-shot teacher-forced training forward pass MUST produce identical logits when both are fed
    the same ground-truth bytes -- any gap is a genuine generation-vs-training bug (this caught
    the windowed-attention dense-fallback bug, see docs/status.md), not exposure bias or model
    quality. Returns the mismatch count (0 = pass).
    """
    was_training = model.training
    model.eval()
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    L_total = full_bytes.shape[1]

    _, _, _, _, h_list_tf, _, _, final_embed_weight_tf, _, _, query_seq_tf = model._run(
        full_bytes, compute_ntp=False, max_decode_sources=None, want_next_query=False)
    embed_tf = (final_embed_weight_tf[0] if final_embed_weight_tf[0] is not None
                else model.encode_lms[0].embed.weight)
    # query_seq_tf[0] is the boundary-patched tensor training's own NTP loss actually reads for
    # level0's final decode stage -- h_list_tf[0] (unpatched) is wrong at block-boundary positions
    # there (see module docstring). query_seq is always aligned like h (query_seq[t] predicts byte
    # t+1) -- no K0-based offset needed; falls back to h_list_tf[0] only when level0 had no decode
    # stage at all (query_seq_tf[0] stays None then, e.g. too little context for any track).
    query_ref_tf = query_seq_tf[0] if query_seq_tf[0] is not None else h_list_tf[0]
    logits_tf_all = F.linear(query_ref_tf[0], embed_tf)

    n_mismatch = 0
    for t in range(prompt_len, L_total - 1):
        ref_idx = t - 1
        if ref_idx < 0 or ref_idx >= logits_tf_all.shape[0]:
            continue
        prefix = full_bytes[:, :t]
        _, _, _, _, h_list_gen, _, _, final_embed_weight_gen, next_query_gen, _, _ = model._run(
            prefix, compute_ntp=False, max_decode_sources=None, want_next_query=True)
        embed_gen = (final_embed_weight_gen[0] if final_embed_weight_gen[0] is not None
                     else model.encode_lms[0].embed.weight)
        query_gen = next_query_gen[0] if next_query_gen[0] is not None else h_list_gen[0][:, -1, :]
        logits_gen = F.linear(query_gen[0], embed_gen)
        if (logits_gen - logits_tf_all[ref_idx]).abs().max().item() >= tol:
            n_mismatch += 1
    if was_training:
        model.train()
    prefix_log = f"gen_consistency_{label}" if label else "gen_consistency"
    log(f"{prefix_log}: {n_mismatch}/{L_total - 1 - prompt_len} timesteps mismatched "
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
    
    for level in range(1, model.n_levels):
        cum_K = 1
        for k in model.cfg.Ks[:level]:
            cum_K *= k
        n_new_codes = gen_len // cum_K
        if n_new_codes <= 0:
            continue
        gen = generate_level_codes(model, prompt_bytes, level, n_new_codes, device)
        log(f"{prefix}level{level}_gen:          {gen.tolist()}")
        gen_decode = generate_level_codes_via_decode(model, prompt_bytes, level, n_new_codes, device)
        log(f"{prefix}level{level}_gen_decode:   {gen_decode.tolist()}")
        if ground_truth is not None:
            full_bytes = torch.cat([prompt_bytes.reshape(-1), ground_truth.reshape(-1)])
            gt = level_ground_truth_codes(model, full_bytes, level, prompt_bytes.numel(), device)
            log(f"{prefix}level{level}_gt:           {gt.tolist()}")


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
              f"bytes (e.g. a val split under a large context_len). Not an error by itself (selfcode_decode's "
              f"NTP target now matches whatever length actually comes out), but the resulting "
              f"bpb/loss numbers reflect a shorter context than configured.")
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

    p = argparse.ArgumentParser(description="Staged cross-attention decode, qfb boundary-query fix, independent per-level/per-track weights", parents=[pre])
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
    p.add_argument("--vocab", type=int, default=256)
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
        vocab=args.vocab, quant_type=args.quant_type, bsq_bits=args.bsq_bits,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_5_{int(time.time())}")
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
