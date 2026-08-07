"""qcute.qcutelm_mergetoken_v1 — qcutelm_vlt11 with level 0 restructured for
multi-token prediction, cutting level 0's FLOP cost (the dominant term in
v11's own compute breakdown) while still covering the full raw-byte
context directly.

Session motivation: "i dont think level 0 need to train on 1024 context
length, the point of coarser layer is to increase receptive field, hence
find combination level 0 1 2 context length and byte window k that can
save flop yet effective field 1024 matching bytelm." Worked through why
naive Ks/window retuning alone can't deliver this: per the 6*N*tokens
FLOPs estimate used throughout this session, the DOMINANT cost at these
widths is the O(tokens*d^2) MLP/projection term, not the O(tokens*window*d)
attention term — and a byte-level LM must emit a prediction for every one
of the 1024 bytes, so D_0's token count (and therefore its dominant FLOP
cost) can't shrink just by shrinking attn_window; something has to reduce
the actual TOKEN COUNT while still covering every byte. That something is
multi-token prediction (matching bytelm's own mtp_heads and qcute_fifo's
FetchHead) — this file's actual change from v11.

## What changed vs qcutelm_vlt11.py

Level 0 (E_0/D_0) now operates on Ks[0]-byte BLOCKS, not individual
bytes: raw bytes are merged into block embeddings first (a cheap, non-
attentional linear merge — same formula qcutelm_pyramid.py's local merge
uses, WITHOUT the quantize() step, since this is an INPUT embedding, not
a code — quantization still happens separately at code_pre's c_0
readout). E_0/D_0 then attend over context_len/Ks[0] block-positions
instead of context_len raw positions — e.g. 256 instead of 1024 for the
default Ks[0]=4, a genuine ~4x cut to level 0's dominant per-token cost
(level 0 was ~58 of v11's total ~76 GFLOP/step pre-e_ntp_weight, so this
is a real, not marginal, saving).

Each block position predicts its FOLLOWING Ks[0] bytes via a byte-chain
MTP head (FetchHead, ported from qcutelm_pyramid.py — chains over BYTES,
contrast BitPredictHead which chains over a code's BITS) — recovering all
1024 byte predictions from only 256 "expensive" attended positions. This
also simplifies code_pre's c_0 readout: since every E_0 position already
IS a block (not a raw byte), code_pre reads out at EVERY position
directly — no more "slice out every Ks[0]-th position from a byte-
resolution hidden state" (v11's h_a_blocks[:, :, K-1, :] trick), since
there's no byte-resolution hidden state at level 0 anymore.

Levels 1 and 2 (E_1/D_1/codelm_1, E_2/D_2/codelm_2) are UNCHANGED from
v11 — still operate on c_0/c_1 exactly as before, so every validated v11
mechanism (detach-teacher-force fix / e_ntp_weight, both amortization
strategies, share_across_levels) carries over untouched, just sitting on
top of a cheaper level 0.

Effective receptive field: with Ks[0]=4, E_0/D_0's OWN attention (windowed
or dense) now operates over only 256 positions instead of 1024, so
covering the full 1024-byte span through E_0/D_0's own attention alone
is far cheaper than it was in v11 (a window or dense pass over 256
tokens, not 1024) — "match bytelm's 1024-byte effective field" falls out
directly from block-level attention reach, not from needing wide windows
over raw bytes. The coarser levels (1, 2) still exist and still provide
cheap longer-range access on top of this, unchanged from v11 — this file
does NOT yet implement genuinely EXTENDING reach beyond 1024 bytes via
levels 1/2 (i.e. having c_1/c_2 look further back in time than level 0's
own window, rather than just re-summarizing the same 1024-byte span at
coarser resolution) — that's a further, bigger architectural change
(would need a persistent cross-window code buffer) not attempted here;
this file's scope is specifically the level-0 FLOP reduction.

No shared imports with qcutelm_vlt2/.../vlt11/qcutelm_pyramid.py (self-
contained-module convention) — Logger/Checkpointer/schedule helpers/
quantizers/Block/BitPredictHead/FetchHead duplicated verbatim.

    uv run python -m qcute.qcutelm_mergetoken_v1 --config configs/qcutelm_mergetoken_v1_<name>.py
"""
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
    Ks: tuple[int, ...] = (4, 4, 4)   # Ks[0] is now ALSO level 0's byte-block size (MTP factor) —
                                        # dual-purpose: input block granularity AND compression factor
                                        # feeding level 1, unlike v11 where these could in principle
                                        # differ (v11 doesn't actually decouple them either, but this
                                        # file makes the coupling structural, not incidental).
    dqs: tuple[int, ...] = (8, 8, 8)
    tier_d_models: tuple[int, ...] = (96, 96, 96)
    context_len: int = 1024
    quant_type: str = "ifsq"
    fsq_levels: int = 8
    vocab: int = 256
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    attn_window: int = -1     # applies to E_0/D_0's BLOCK sequence (context_len/Ks[0] positions, not
                                # context_len) and to E_1/D_1, E_2/D_2 same as v11. Default dense (-1)
                                # here since level 0's block sequence is already short (256 for the
                                # default Ks[0]=4) — dense attention over 256 is cheap; window is an
                                # opt-in, not required the way it was for v11's raw 1024-length level 0.
    fetch_n_heads: int = 2     # level-0 byte-chain MTP head's own small self-attention module
    fetch_gamma: float = 1.0   # Fetch's h_t^(j) = gamma*h_t + Emb(x_{t+j}) scaling
    tie_head0: bool = True     # level-0 byte prediction head tied to the byte embedding table
                                # (GPT-style) — matches qcute_fifo's own convention for its analogous head
    lm_d_model: int = 128
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    lm_attn_window: int = 16
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    bit_head_mode: str = "chain"   # levels 1/2 only (their next-code prediction head) — level 0 always
                                    # uses the byte-chain FetchHead now, this flag doesn't apply to it
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True
    e_ntp_weight: float = 0.0
    e_ntp_every: int = 1
    e_ntp_bit_head_mode: str | None = None
    share_across_levels: bool = False   # applies to levels 1/2 only (E_1/D_1/codelm_1 vs E_2/D_2/
                                    # codelm_2) — level 0 is architecturally unique now (byte-chain MTP
                                    # head, block merge) so it was never a candidate for cross-level
                                    # sharing with 1/2 even in v11's version of this flag.


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def fsq_quantize(v: torch.Tensor, levels: int, bound: str = "tanh") -> torch.Tensor:
    half_l = (levels - 1) / 2
    z = torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1)
    bounded = z * half_l
    z_hat = bounded + (torch.round(bounded) - bounded).detach()
    return z_hat / half_l


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.window is not None and T % self.window == 0 and T > self.window:
            y = self._forward_chunked(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = q.shape
        W = self.window
        n_chunks = T // W

        def to_chunks(t):
            return t.view(B, H, n_chunks, W, hd)

        qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
        zero_chunk = torch.zeros(B, H, 1, W, hd, device=q.device, dtype=q.dtype)
        kc_prev = torch.cat([zero_chunk, kc[:, :, :-1]], dim=2)
        vc_prev = torch.cat([zero_chunk, vc[:, :, :-1]], dim=2)
        k_local = torch.cat([kc_prev, kc], dim=3)
        v_local = torch.cat([vc_prev, vc], dim=3)

        i = torch.arange(W, device=q.device).view(W, 1)
        j_prev = torch.arange(W, device=q.device).view(1, W) - W
        j_cur = torch.arange(W, device=q.device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)
        mask_per_chunk = causal_window.unsqueeze(0).expand(n_chunks, W, 2 * W).clone()
        mask_per_chunk[0, :, 0:W] = False

        qb = qc.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = k_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)
        vb = v_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)
        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, window: int | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, window=window)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


def byte_to_bits(byte_ids: torch.Tensor) -> torch.Tensor:
    bits = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    return (2 * bits - 1) / math.sqrt(8)


class SmallSelfAttn(nn.Module):
    """Manual multi-head self-attention via F.scaled_dot_product_attention — NOT nn.MultiheadAttention,
    which this session found has severe MPS memory/perf pathology (8.82 GiB OOM after 2 forward calls
    at d_model=256, batch=16 — see qcutelm_pyramid.py's identical class for the full incident)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FetchHead(nn.Module):
    """Byte-chain MTP head — ported from qcutelm_pyramid.py's FetchHead (itself ported from
    qcute_fifo.py). Given a block position's hidden state h, predicts the `bandwidth` (= Ks[0]) raw
    bytes immediately following that block via the mathematically-exact chain rule (each subsequent
    byte's prediction additionally conditions on the true, teacher-forced previous byte(s) in the
    chain) — contrast BitPredictHead (qcutelm_vlt11.py), which chains over a CODE's BITS, not bytes."""

    def __init__(self, cfg: "Config", byte_emb: nn.Embedding, pred_head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = byte_emb
        self.pred_head = pred_head
        self.bandwidth = cfg.Ks[0]
        self.chain_pos_emb = nn.Embedding(self.bandwidth, byte_emb.embedding_dim)
        self.self_attn = SmallSelfAttn(byte_emb.embedding_dim, cfg.fetch_n_heads)

    def forward(self, h: torch.Tensor, target_bytes: torch.Tensor | None) -> torch.Tensor:
        """h: [N, D]. target_bytes: [N, bandwidth] true next bytes, teacher-forced (required —
        training only). -> logits [N, bandwidth, vocab]."""
        N, D = h.shape
        chain_vecs = [h + self.chain_pos_emb.weight[0]]
        logits_list = []
        for j in range(self.bandwidth):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self.self_attn(x)
            fetched = h + attn_out[:, -1, :]
            logits_list.append(self.pred_head(fetched))
            if j < self.bandwidth - 1:
                next_byte = target_bytes[:, j]
                chain_vecs.append(self.cfg.fetch_gamma * h + self.byte_emb(next_byte) + self.chain_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


class BitPredictHead(nn.Module):
    """Verbatim port of qcutelm_vlt11.BitPredictHead, for levels 1/2's next-code prediction only —
    level 0 uses FetchHead above instead."""

    def __init__(self, d_model: int, dq: int, mode: str, n_heads: int = 2, gamma: float = 1.0, chain_fixed_kernel: bool = True):
        super().__init__()
        assert mode in ("independent", "chain")
        self.dq = dq
        self.mode = mode
        self.gamma = gamma
        self.chain_fixed_kernel = chain_fixed_kernel
        if mode == "independent":
            self.head = nn.Linear(d_model, dq)
        else:
            self.head = nn.Linear(d_model, 1)
            self.bit_pos_emb = nn.Embedding(dq, d_model)
            self.bit_val_emb = nn.Embedding(2, d_model)
            self.self_attn = SmallSelfAttn(d_model, n_heads)
            causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "independent":
            return self.head(h)
        if self.chain_fixed_kernel and true_bits is not None:
            return self._forward_chain_fixed(h, true_bits)
        return self._forward_chain_loop(h, true_bits)

    def _forward_chain_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)
        zero_vec = val_embeds.new_zeros(N, 1, D)
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)
        pos = self.bit_pos_emb.weight.unsqueeze(0)
        h_scale = h.new_ones(1, self.dq, 1)
        if self.dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        x = h_scale * h.unsqueeze(1) + shifted + pos
        attn_out = self.self_attn(x, attn_mask=self.causal_mask)
        fetched = h.unsqueeze(1) + attn_out
        return self.head(fetched).squeeze(-1)

    def _forward_chain_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        chain_vecs = [h + self.bit_pos_emb.weight[0]]
        logits_list = []
        for j in range(self.dq):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self.self_attn(x)
            fetched = h + attn_out[:, -1, :]
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                chain_vecs.append(self.gamma * h + self.bit_val_emb(bit_val) + self.bit_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


class CodeLM(nn.Module):
    """Levels 1/2's own code forecaster — unchanged from qcutelm_vlt11.py."""

    def __init__(self, cfg: Config, dq: int):
        super().__init__()
        self.cfg = cfg
        self.dq = dq
        window = None if cfg.lm_attn_window == -1 else cfg.lm_attn_window
        self.in_proj = nn.Linear(dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=window) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        if factorized_softmax:
            self.pred_head = nn.Linear(cfg.lm_d_model, dq * cfg.fsq_levels)
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)
        else:
            self.pred_head = BitPredictHead(cfg.lm_d_model, dq, cfg.bit_head_mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        x = self.in_proj(z_hat)
        head_dim = cfg.lm_d_model // cfg.lm_n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)
        if cfg.quant_type in ("fsq", "ifsq"):
            raw = self.pred_head(h)
            B, T, _ = raw.shape
            logits = raw.view(B, T, self.dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1), logits
        B, T, D = h.shape
        h_flat = h.reshape(B * T, D)
        if cfg.bit_head_mode == "chain":
            if T > 1:
                h_bulk = h[:, :-1, :].reshape(B * (T - 1), D)
                true_bits_bulk = z_hat[:, 1:, :].reshape(B * (T - 1), self.dq)
                raw_bulk = self.pred_head(h_bulk, true_bits_bulk).reshape(B, T - 1, self.dq)
            else:
                raw_bulk = z_hat.new_zeros(B, 0, self.dq)
            raw_last = self.pred_head(h[:, -1, :], None).reshape(B, 1, self.dq)
            raw = torch.cat([raw_bulk, raw_last], dim=1)
        else:
            raw = self.pred_head(h_flat).reshape(B, T, self.dq)
        return 2 * torch.sigmoid(raw) - 1, raw


class DilateSandwichLM(nn.Module):
    """Level 0: block-merged input + FetchHead byte-chain MTP (this file's actual change).
    Levels 1/2: identical structure/mechanics to qcutelm_vlt11.RecursiveSandwichLM."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert self.n_levels >= 1
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        window = None if cfg.attn_window == -1 else cfg.attn_window

        # seq_lens[i] = level i's OWN sequence length. Ks[0] plays a DIFFERENT role than v11: it's
        # level 0's byte-block MERGE factor (raw bytes -> blocks), not a "pool level 0's own sequence"
        # factor — code_pre0 reads out ONE code per block position (no further pooling), so c_0's
        # length equals level 0's own block count exactly, and level 1's own sequence (= c_0) starts
        # at that SAME length, unpooled relative to level 0. From level 1 onward, Ks[i] resumes v11's
        # original meaning (level i's own pooling factor, producing level (i+1)'s length) — i.e. Ks[1]
        # pools level 1's sequence to produce level 2's length, Ks[2] pools level 2's to produce level
        # 3's (if it existed), etc. Ks[-1] (the last level's own K) is used only for that level's own
        # code_pre readout granularity, never to produce a further level.
        assert cfg.context_len % cfg.Ks[0] == 0, "context_len must be divisible by Ks[0] (level 0's block size)"
        n_blocks_0 = cfg.context_len // cfg.Ks[0]
        seq_lens = [n_blocks_0]
        if self.n_levels > 1:
            seq_lens.append(n_blocks_0)   # level 1's own input (c_0) starts at the same length as level
                                            # 0's own block count — level 0's readout doesn't pool further
            for k in cfg.Ks[1:-1]:
                assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide at every level"
                seq_lens.append(seq_lens[-1] // k)
            assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide at every level"
        self.seq_lens = seq_lens   # length n_levels

        for i, d in enumerate(cfg.tier_d_models):
            assert d % cfg.n_heads == 0, f"tier_d_models[{i}] ({d}) must be divisible by n_heads ({cfg.n_heads})"
        if window is not None:
            for i, L in enumerate(seq_lens):
                assert L % window == 0, f"attn_window ({window}) must divide level {i}'s sequence length ({L})"
        # (lm_attn_window's own divisibility is validated implicitly at construction time inside each
        # CodeLM's Block — an ill-fitting value fails loudly with a clear shape error at first forward()
        # call rather than a pre-flight assertion here; not yet mirrored from v11 for this file.)

        assert cfg.bit_head_mode in ("independent", "chain")

        D0 = cfg.tier_d_models[0]
        # level 0: byte-block merge (cheap, non-attentional, matches qcutelm_pyramid's local merge
        # formula minus the quantize() step — this is an INPUT embedding, not a code). Separate E_0/D_0
        # copies, matching v11's untied Pass1/Pass2 convention (qcutelm_vlt8's finding: sharing these
        # two DIFFERENT functions measurably hurt).
        self.byte_emb = nn.Embedding(cfg.vocab, D0)
        self.dec_byte_emb = nn.Embedding(cfg.vocab, D0)
        self.merge0 = nn.Linear(cfg.Ks[0] * D0, D0)
        self.dec_merge0 = nn.Linear(cfg.Ks[0] * D0, D0)

        self.e0_blocks = nn.ModuleList([Block(D0, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.e0_ln_f = nn.LayerNorm(D0)
        self.d0_blocks = nn.ModuleList([Block(D0, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.d0_ln_f = nn.LayerNorm(D0)

        self.code_pre0 = nn.Linear(D0, cfg.dqs[0])
        self.z_proj0 = nn.Linear(cfg.dqs[0], D0)
        self.codelm0 = CodeLM(cfg, cfg.dqs[0])

        self.head0 = nn.Linear(D0, cfg.vocab)
        if cfg.tie_head0:
            self.head0.weight = self.byte_emb.weight
        self.fetch_head0 = FetchHead(cfg, self.byte_emb, self.head0)
        if cfg.e_ntp_weight > 0:
            self.e_head0 = nn.Linear(D0, cfg.vocab)
            if cfg.tie_head0:
                self.e_head0.weight = self.byte_emb.weight
            self.e_fetch_head0 = FetchHead(cfg, self.byte_emb, self.e_head0)

        # levels 1+: identical structure/mechanics to qcutelm_vlt11.RecursiveSandwichLM, operating on
        # c_0, c_1, ... exactly as before. n_shared follows share_across_levels among levels 1+ only.
        self.n_upper = self.n_levels - 1
        if self.n_upper > 0:
            self.n_shared = self.n_upper if cfg.share_across_levels else self.n_upper
            if not cfg.share_across_levels:
                self.n_shared = self.n_upper
            else:
                assert len(set(cfg.tier_d_models[1:])) == 1, "share_across_levels requires uniform tier_d_models among levels 1+"
                assert len(set(cfg.dqs[1:])) == 1, "share_across_levels requires uniform dqs among levels 1+"
                self.n_shared = 1

            self.embed = nn.ModuleList([nn.Linear(cfg.dqs[i], cfg.tier_d_models[i + 1]) for i in range(self.n_shared)])
            self.dec_embed = nn.ModuleList([nn.Linear(cfg.dqs[i], cfg.tier_d_models[i + 1]) for i in range(self.n_shared)])
            self.e_blocks = nn.ModuleList([nn.ModuleList([Block(cfg.tier_d_models[i + 1], cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)]) for i in range(self.n_shared)])
            self.e_ln_f = nn.ModuleList([nn.LayerNorm(cfg.tier_d_models[i + 1]) for i in range(self.n_shared)])
            self.d_blocks = nn.ModuleList([nn.ModuleList([Block(cfg.tier_d_models[i + 1], cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)]) for i in range(self.n_shared)])
            self.d_ln_f = nn.ModuleList([nn.LayerNorm(cfg.tier_d_models[i + 1]) for i in range(self.n_shared)])
            self.code_pre = nn.ModuleList([nn.Linear(cfg.tier_d_models[i + 1], cfg.dqs[i + 1]) for i in range(self.n_shared)])
            self.z_proj = nn.ModuleList([nn.Linear(cfg.dqs[i + 1], cfg.tier_d_models[i + 1]) for i in range(self.n_shared)])
            self.codelm = nn.ModuleList([CodeLM(cfg, cfg.dqs[i + 1]) for i in range(self.n_shared)])
            self.head_code = nn.ModuleList([self._build_head_code(i, cfg.bit_head_mode) for i in range(self.n_shared)])
            if cfg.e_ntp_weight > 0:
                e_mode = cfg.e_ntp_bit_head_mode if cfg.e_ntp_bit_head_mode is not None else cfg.bit_head_mode
                self.head_e_code = nn.ModuleList([self._build_head_code(i, e_mode) for i in range(self.n_shared)])

    def _sel_upper(self, i: int) -> int:
        """i is an UPPER-level index (0 = level 1, 1 = level 2, ...). Returns index into the
        embed/e_blocks/... ModuleLists (length n_shared)."""
        return 0 if self.cfg.share_across_levels else i

    def _build_head_code(self, i: int, mode: str) -> nn.Module:
        cfg = self.cfg
        dq = cfg.dqs[i + 1]
        if cfg.quant_type in ("fsq", "ifsq"):
            return nn.Linear(cfg.tier_d_models[i + 1], dq * cfg.fsq_levels)
        return BitPredictHead(cfg.tier_d_models[i + 1], dq, mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def run_blocks(self, blocks: nn.ModuleList, ln_f: nn.LayerNorm, d_model: int, x: torch.Tensor) -> torch.Tensor:
        head_dim = d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in blocks:
            x = block(x, cos, sin)
        return ln_f(x)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, v.size(-1))
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        elif self.cfg.quant_type == "identity":
            return v
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def code_level_loss(self, raw_logits: torch.Tensor, true_code: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        true_code = true_code.detach()
        if cfg.quant_type in ("fsq", "ifsq"):
            half_l = (cfg.fsq_levels - 1) / 2
            true_level = torch.round(true_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
            dq = true_code.size(-1)
            logits = raw_logits.view(*raw_logits.shape[:-1], dq, cfg.fsq_levels)
            return F.cross_entropy(logits.reshape(-1, cfg.fsq_levels), true_level.reshape(-1))
        true_bits = (true_code > 0).float()
        return F.binary_cross_entropy_with_logits(raw_logits, true_bits, reduction="none").sum(-1).mean()

    def _merge_bytes(self, merge: nn.Linear, byte_emb: nn.Embedding, byte_seq: torch.Tensor) -> torch.Tensor:
        """byte_seq: [B, context_len] raw byte ids -> [B, n_blocks_0, D0] block embeddings."""
        cfg = self.cfg
        B = byte_seq.size(0)
        K0 = cfg.Ks[0]
        embeds = byte_emb(byte_seq)                              # [B, context_len, D0]
        blocks = embeds.view(B, -1, K0 * embeds.size(-1))         # [B, n_blocks_0, K0*D0]
        return merge(blocks)                                      # [B, n_blocks_0, D0]

    def _level0_ntp(self, h: torch.Tensor, byte_seq: torch.Tensor, is_e_side: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [B, n_blocks_0, D0] (D_0's or E_0's hidden state at every block position). byte_seq:
        [B, context_len] the true raw bytes. Each block position b predicts the Ks[0] bytes of block
        b+1 (the block immediately following it) — the last block has no "next block" to predict, so
        it's excluded from the loss (same "no target for the final position" pattern as any NTP loss)."""
        cfg = self.cfg
        K0 = cfg.Ks[0]
        B, n_blocks_0, D0 = h.shape
        h_bulk = h[:, :-1, :].reshape(-1, D0)                                  # [B*(n_blocks_0-1), D0]
        targets = byte_seq.view(B, n_blocks_0, K0)[:, 1:, :].reshape(-1, K0)   # [B*(n_blocks_0-1), K0]
        fetch = self.e_fetch_head0 if is_e_side else self.fetch_head0
        logits = fetch(h_bulk, targets)                                        # [N, K0, vocab]
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), targets.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean()
        return loss, acc

    def forward(self, ctx: torch.Tensor, step: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B = ctx.size(0)
        K0 = cfg.Ks[0]
        n_blocks_0 = self.seq_lens[0]

        compute_e_ntp = cfg.e_ntp_weight > 0 and (step is None or cfg.e_ntp_every <= 1 or step % cfg.e_ntp_every == 0)

        # LEVEL 0 — block-merged input, byte-chain MTP prediction (this file's actual change)
        e_in0 = self._merge_bytes(self.merge0, self.byte_emb, ctx)              # [B, n_blocks_0, D0]
        h_a0 = self.run_blocks(self.e0_blocks, self.e0_ln_f, cfg.tier_d_models[0], e_in0)

        h_a0_for_code = h_a0.detach() if cfg.e_ntp_weight > 0 else h_a0
        pre_q0 = self.code_pre0(h_a0_for_code)                                   # [B, n_blocks_0, dq0] —
                                                                                    # readout at EVERY
                                                                                    # position now (no
                                                                                    # block-slicing needed,
                                                                                    # every position already
                                                                                    # IS a block)
        c_0 = self.quantize(pre_q0)

        pred_soft0_full, raw_logits0_full = self.codelm0(c_0)
        pred_soft0 = pred_soft0_full[:, :-1, :]
        raw_logits0 = raw_logits0_full[:, :-1]
        true_next_c0 = c_0[:, 1:, :].detach()
        if cfg.quant_type in ("fsq", "ifsq"):
            half_l = (cfg.fsq_levels - 1) / 2
            true_level_idx = torch.round(true_next_c0 * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
            cm_loss0 = F.cross_entropy(raw_logits0.reshape(-1, cfg.fsq_levels), true_level_idx.reshape(-1))
        else:
            true_bits = (true_next_c0 > 0).float()
            cm_loss0 = F.binary_cross_entropy_with_logits(raw_logits0, true_bits, reduction="none").sum(-1).mean()

        d_in0 = self._merge_bytes(self.dec_merge0, self.dec_byte_emb, ctx)      # [B, n_blocks_0, D0]
        forecast_embed0 = self.z_proj0(pred_soft0)                              # [B, n_blocks_0-1, D0]
        d_in0 = torch.cat([d_in0[:, :1, :], forecast_embed0], dim=1)            # block 0 keeps its own
                                                                                    # real embedding
                                                                                    # (nothing to forecast
                                                                                    # from yet), blocks
                                                                                    # 1..n_blocks_0-1 get
                                                                                    # the causal forecast
        h_b0 = self.run_blocks(self.d0_blocks, self.d0_ln_f, cfg.tier_d_models[0], d_in0)

        loss0, acc0 = self._level0_ntp(h_b0, ctx, is_e_side=False)
        level_losses = [loss0]
        level_accs = [acc0]
        code_match_losses = [cm_loss0]
        e_ntp_losses, e_ntp_accs = [], []
        if compute_e_ntp:
            e_loss0, e_acc0 = self._level0_ntp(h_a0, ctx, is_e_side=True)
            e_ntp_losses.append(e_loss0)
            e_ntp_accs.append(e_acc0)

        byte_loss, byte_acc = loss0, acc0

        # LEVELS 1+ — unchanged from qcutelm_vlt11.py, operating on c_0, c_1, ...
        seq = c_0
        for i in range(self.n_upper):
            si = self._sel_upper(i)
            K = cfg.Ks[i + 1]
            L = self.seq_lens[i + 1]
            D = cfg.tier_d_models[i + 1]
            n_blocks = L // K

            e_in = self.embed[si](seq)
            h_a = self.run_blocks(self.e_blocks[si], self.e_ln_f[si], D, e_in)
            h_a_for_code = h_a.detach() if cfg.e_ntp_weight > 0 else h_a
            h_a_blocks = h_a_for_code.view(B, n_blocks, K, D)
            pre_q = self.code_pre[si](h_a_blocks[:, :, K - 1, :])
            c_i = self.quantize(pre_q)

            pred_soft_full, raw_logits_full = self.codelm[si](c_i)
            pred_soft = pred_soft_full[:, :-1, :]
            raw_logits = raw_logits_full[:, :-1]
            true_next_code = c_i[:, 1:, :].detach()
            if cfg.quant_type in ("fsq", "ifsq"):
                half_l = (cfg.fsq_levels - 1) / 2
                true_level_idx = torch.round(true_next_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                cm_loss = F.cross_entropy(raw_logits.reshape(-1, cfg.fsq_levels), true_level_idx.reshape(-1))
            else:
                true_bits = (true_next_code > 0).float()
                cm_loss = F.binary_cross_entropy_with_logits(raw_logits, true_bits, reduction="none").sum(-1).mean()
            code_match_losses.append(cm_loss)

            d_in = self.dec_embed[si](seq)
            d_in_blocks = d_in.view(B, n_blocks, K, D)
            forecast_embed = self.z_proj[si](pred_soft)
            d_in_blocks = torch.cat(
                [d_in_blocks[:, :1, :, :], torch.cat([forecast_embed.unsqueeze(2), d_in_blocks[:, 1:, 1:, :]], dim=2)],
                dim=1,
            )
            d_in = d_in_blocks.view(B, L, D)
            h_b = self.run_blocks(self.d_blocks[si], self.d_ln_f[si], D, d_in)

            h_flat = h_b[:, :-1, :].reshape(-1, D)
            true_next_seq = seq[:, 1:, :]
            head = self.head_code[si]
            if cfg.quant_type in ("fsq", "ifsq"):
                raw = head(h_flat).reshape(B, L - 1, -1)
            else:
                true_flat = true_next_seq.reshape(-1, true_next_seq.size(-1))
                raw_flat = head(h_flat, true_flat) if head.mode == "chain" else head(h_flat)
                raw = raw_flat.reshape(B, L - 1, -1)
            loss_i = self.code_level_loss(raw, true_next_seq)
            with torch.no_grad():
                if cfg.quant_type in ("fsq", "ifsq"):
                    half_l = (cfg.fsq_levels - 1) / 2
                    true_lvl = torch.round(true_next_seq.detach() * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                    pred_lvl = raw.view(*raw.shape[:-1], true_next_seq.size(-1), cfg.fsq_levels).argmax(-1)
                    acc_i = (pred_lvl == true_lvl).float().mean()
                else:
                    acc_i = ((raw > 0) == (true_next_seq.detach() > 0)).float().mean()
            level_losses.append(loss_i)
            level_accs.append(acc_i)

            if compute_e_ntp:
                head_e = self.head_e_code[si]
                if cfg.quant_type in ("fsq", "ifsq"):
                    raw_e = head_e(h_flat).reshape(B, L - 1, -1)
                else:
                    true_flat = true_next_seq.reshape(-1, true_next_seq.size(-1))
                    raw_flat_e = head_e(h_flat, true_flat) if head_e.mode == "chain" else head_e(h_flat)
                    raw_e = raw_flat_e.reshape(B, L - 1, -1)
                e_loss_i = self.code_level_loss(raw_e, true_next_seq)
                with torch.no_grad():
                    if cfg.quant_type in ("fsq", "ifsq"):
                        half_l = (cfg.fsq_levels - 1) / 2
                        true_lvl = torch.round(true_next_seq.detach() * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                        pred_lvl = raw_e.view(*raw_e.shape[:-1], true_next_seq.size(-1), cfg.fsq_levels).argmax(-1)
                        e_acc_i = (pred_lvl == true_lvl).float().mean()
                    else:
                        e_acc_i = ((raw_e > 0) == (true_next_seq.detach() > 0)).float().mean()
                e_ntp_losses.append(e_loss_i)
                e_ntp_accs.append(e_acc_i)

            seq = c_i

        loss = sum(level_losses) + cfg.code_match_weight * torch.stack(code_match_losses).sum()
        if compute_e_ntp:
            loss = loss + cfg.e_ntp_weight * torch.stack(e_ntp_losses).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "code_match_loss": torch.stack(code_match_losses).sum(),
            **{f"level{i}_loss": l for i, l in enumerate(level_losses)},
            **{f"level{i}_acc": a for i, a in enumerate(level_accs)},
            **{f"code_match_loss_L{i}": v for i, v in enumerate(code_match_losses)},
        }
        if compute_e_ntp:
            metrics["e_ntp_loss"] = torch.stack(e_ntp_losses).sum()
            metrics.update({f"e_ntp_loss_L{i}": l for i, l in enumerate(e_ntp_losses)})
            metrics.update({f"e_ntp_acc_L{i}": a for i, a in enumerate(e_ntp_accs)})
        return loss, metrics


def init_head_bias_to_unigram(model: DilateSandwichLM, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        model.head0.bias.copy_(log_freq.to(model.head0.bias.device))
        if model.cfg.e_ntp_weight > 0:
            model.e_head0.bias.copy_(log_freq.to(model.e_head0.bias.device))


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
def eval_model(model: DilateSandwichLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
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
    return result


def train(model: DilateSandwichLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_mergetoken_v1", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx, step=step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}", byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%")

        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), bpb=train_bpb, byte_acc=metrics["byte_acc"].item())

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="qcutelm_vlt11 with level 0 restructured for block-merge + byte-chain MTP", parents=[pre])
    p.add_argument("--Ks", type=_parse_int_tuple, default=(4, 4, 4))
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(96, 96, 96))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq", "identity"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--fetch_n_heads", type=int, default=2)
    p.add_argument("--fetch_gamma", type=float, default=1.0)
    p.add_argument("--tie_head0", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--lm_d_model", type=int, default=128)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=3)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--lm_attn_window", type=int, default=16)
    p.add_argument("--code_match_weight", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--bit_head_mode", type=str, default="chain", choices=["independent", "chain"])
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--e_ntp_weight", type=float, default=0.0)
    p.add_argument("--e_ntp_every", type=int, default=1)
    p.add_argument("--e_ntp_bit_head_mode", type=str, default=None, choices=[None, "independent", "chain"])
    p.add_argument("--share_across_levels", action="store_true")

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.Ks = _parse_int_tuple(args.Ks) if isinstance(args.Ks, str) else tuple(args.Ks)
    args.dqs = _parse_int_tuple(args.dqs) if isinstance(args.dqs, str) else tuple(args.dqs)
    args.tier_d_models = _parse_int_tuple(args.tier_d_models) if isinstance(args.tier_d_models, str) else tuple(args.tier_d_models)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, dqs=args.dqs, tier_d_models=args.tier_d_models, context_len=args.context_len,
        quant_type=args.quant_type, fsq_levels=args.fsq_levels, n_heads=args.n_heads, n_layers=args.n_layers,
        mlp_mult=args.mlp_mult, attn_window=args.attn_window, fetch_n_heads=args.fetch_n_heads,
        fetch_gamma=args.fetch_gamma, tie_head0=args.tie_head0,
        lm_d_model=args.lm_d_model, lm_n_heads=args.lm_n_heads, lm_n_layers=args.lm_n_layers,
        lm_mlp_mult=args.lm_mlp_mult, lm_attn_window=args.lm_attn_window,
        code_match_weight=args.code_match_weight, rope_base=args.rope_base,
        bit_head_mode=args.bit_head_mode, bit_chain_n_heads=args.bit_chain_n_heads,
        bit_chain_gamma=args.bit_chain_gamma, bit_chain_fixed_kernel=args.bit_chain_fixed_kernel,
        e_ntp_weight=args.e_ntp_weight, e_ntp_every=args.e_ntp_every, e_ntp_bit_head_mode=args.e_ntp_bit_head_mode,
        share_across_levels=args.share_across_levels,
    )
    model = DilateSandwichLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_mergetoken_v1_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} seq_lens={model.seq_lens} "
        f"context_len={cfg.context_len} quant_type={cfg.quant_type} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
