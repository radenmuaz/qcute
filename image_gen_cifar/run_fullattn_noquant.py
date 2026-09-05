"""No-quantization fork of run_fullattn.py, standalone codebase (no imports from
run_fullattn.py, run_causalattn.py, or qcute_lagcodec -- every primitive rewritten here).

Only difference from run_fullattn.py: each encoder level's code extraction is a plain
LINEAR PASSTHROUGH (pool -> Linear(D,D)) instead of pool -> code_head -> PQ categorical
STE quantization. The output is a continuous D-dimensional vector -- exactly d_model
wide, on purpose, because it is used with NO embedding table at all in either place a
discrete code used to need one: it IS the next level's input directly, and it IS the
decoder's per-row conditioning value directly. No code_vocab, no pq_chunks, no
quantize_hard, no code_embed, no codebook_utilization (nothing discrete left to measure
usage of) -- gradient flow is exact everywhere, no STE approximation needed since
nothing is ever discretized. Everything else (full-attention per-row encoder with no
cross-row leakage, no NTP, causal lag-1-row decoder, GQA, col_group_size, BOS bootstrap,
interleaved prompt-capable generation) is unchanged from run_fullattn.py -- see that
file's module docstring for the fuller rationale behind those pieces.
"""
import argparse
import json
import math
import os
import pickle
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent


# ---------------------------------------------------------------------------
# CIFAR-10 data
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def load_cifar10(data_root: Path) -> tuple:
    data_root.mkdir(parents=True, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        import urllib.request
        print(f"downloading {CIFAR10_URL} -> {tar_path}")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    extract_dir = data_root / "cifar-10-batches-py"
    if not extract_dir.exists():
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_root)

    def load_batch(fname: str) -> tuple:
        with open(extract_dir / fname, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
        labels = np.array(d[b"labels"], dtype=np.int64)
        return images, labels

    train_batches = [load_batch(f"data_batch_{i}") for i in range(1, 6)]
    train = np.concatenate([b[0] for b in train_batches], axis=0)
    train_labels = np.concatenate([b[1] for b in train_batches], axis=0)
    test, test_labels = load_batch("test_batch")  # doubles as val (no separate held-out split)
    return (train, train_labels), (test, test_labels)


class CIFARDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images  # (N,32,32,3) uint8
        self.labels = labels  # (N,) int64, CIFAR-10 class ids 0-9

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple:
        img = self.images[i]
        r = torch.from_numpy(img[:, :, 0].astype(np.int64))
        g = torch.from_numpy(img[:, :, 1].astype(np.int64))
        b = torch.from_numpy(img[:, :, 2].astype(np.int64))
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return r, g, b, y  # r,g,b each (32,32); y scalar


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 32
    d_model: int = 256
    n_layers: int = 1
    n_heads: int = 4
    n_kv_heads: int = None  # None = max(1, n_heads//4) GQA-by-default; == n_heads for plain MHA
    strides: tuple = (2, 4, 4)        # per-level downsample stride; product must == img_size
    code_extract_mode: str = "mean"   # "mean" (default, pool stride-window) or "last_idx"
    decoder_mode: str = "seq"         # "seq" (default, sequential R->G->B) or "mtp"
    rope_base: float = 10000.0
    mlp_mult: int = 4
    col_group_size: int = 1  # decoder column-track communication: 1=SISO (independent, default),
    # img_size=MIMO (full entanglement), else grouped (GQA-/group-conv-style) -- see ColumnMixAttention
    class_conditional: bool = False  # broadcast a learned per-class embedding into every row's
    # decoder conditioning -- same "inject a learned vector" mechanism as BOS, but data-dependent
    n_classes: int = 10

    def __post_init__(self):
        assert math.prod(self.strides) == self.img_size, \
            f"product(strides)={math.prod(self.strides)} must equal img_size={self.img_size}"
        assert self.d_model % self.n_heads == 0
        assert self.img_size % self.col_group_size == 0


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * d_model
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device) -> tuple:
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


def rope_cos_sin_pos(pos: int, head_dim: int, base: float, device: torch.device) -> tuple:
    """cos/sin for a single absolute position -- used by the decoder's KV-cached
    incremental decode path instead of recomputing a whole rope table per step."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None


class FullSelfAttention(nn.Module):
    """Qwen3-style RoPE + per-head QK-norm + GQA, but FULL (non-causal, bidirectional)
    attention -- every position attends to every other position in its own sequence. No
    incremental/KV-cache path: the encoder never autoregresses, always a single one-shot
    pass per row, so there's nothing to cache."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, d_model + 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, Hkv, T, hd = x.shape
        return x[:, :, None].expand(B, Hkv, self.n_rep, T, hd).reshape(B, Hkv * self.n_rep, T, hd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        Wq, Wk, Wv = self.qkv.weight[:D], self.qkv.weight[D:D + Hkv * hd], self.qkv.weight[D + Hkv * hd:]
        q = F.linear(x, Wq).view(B, T, H, hd).transpose(1, 2)
        k = F.linear(x, Wk).view(B, T, Hkv, hd).transpose(1, 2)
        v = F.linear(x, Wv).view(B, T, Hkv, hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class FullBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FullSelfAttention(d_model, n_heads, rope_base, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Same as run_causalattn.py's -- used only by the Decoder here, which is causal."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, d_model + 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, Hkv, T, hd = x.shape
        return x[:, :, None].expand(B, Hkv, self.n_rep, T, hd).reshape(B, Hkv * self.n_rep, T, hd)

    def _project_qkv(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        Wq, Wk, Wv = self.qkv.weight[:D], self.qkv.weight[D:D + Hkv * hd], self.qkv.weight[D + Hkv * hd:]
        q = F.linear(x, Wq).view(B, T, H, hd).transpose(1, 2)
        k = F.linear(x, Wk).view(B, T, Hkv, hd).transpose(1, 2)
        v = F.linear(x, Wv).view(B, T, Hkv, hd).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)

    def forward_incremental(self, x_new: torch.Tensor, pos: int, cache: KVCache) -> torch.Tensor:
        B, Tn, D = x_new.shape
        assert Tn == 1
        cos, sin = rope_cos_sin_pos(pos, self.head_dim, self.rope_base, x_new.device)
        q, k, v = self._project_qkv(x_new)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache.k = k if cache.k is None else torch.cat([cache.k, k], dim=2)
        cache.v = v if cache.v is None else torch.cat([cache.v, v], dim=2)
        k_full, v_full = self._repeat_kv(cache.k), self._repeat_kv(cache.v)
        y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        y = y.transpose(1, 2).reshape(B, 1, D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope_base, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_incremental(self, x_new: torch.Tensor, pos: int, cache: KVCache) -> torch.Tensor:
        x_new = x_new + self.attn.forward_incremental(self.norm1(x_new), pos, cache)
        x_new = x_new + self.mlp(self.norm2(x_new))
        return x_new


class ColumnMixAttention(nn.Module):
    """Non-causal self-attention across the COLUMN axis -- identical to run_causalattn.py's.
    group_size=1 (default) = SISO no-op, img_size = MIMO, in between = grouped."""

    def __init__(self, d_model: int, n_heads: int, group_size: int):
        super().__init__()
        self.group_size = group_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, row, col, D) -> same shape, column-mixed within groups."""
        g = self.group_size
        if g <= 1:
            return x
        B, R, C, D = x.shape
        assert C % g == 0, f"n_columns={C} must be divisible by group_size={g}"
        H, hd = self.n_heads, self.head_dim
        xn = self.norm(x)
        qkv = self.qkv(xn).view(B, R, C // g, g, 3, H, hd).permute(4, 0, 1, 2, 5, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2]
        shape5 = (B * R * (C // g), H, g, hd)
        q, k, v = q.reshape(shape5), k.reshape(shape5), v.reshape(shape5)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.view(B, R, C // g, H, g, hd).permute(0, 1, 2, 4, 3, 5).reshape(B, R, C, D)
        return x + self.out(y)


def blocks_step(blocks: nn.ModuleList, ln_f: nn.Module, x_new: torch.Tensor, pos: int, caches: list) -> torch.Tensor:
    h = x_new
    for blk, cache in zip(blocks, caches):
        h = blk.forward_incremental(h, pos, cache)
    return ln_f(h)


# ---------------------------------------------------------------------------
# Encoder: 3-level FULL-ATTENTION hierarchy, no NTP, no quantization, batched over ROWS
# ---------------------------------------------------------------------------

class FullAttnEncoderLevel(nn.Module):
    def __init__(self, cfg: Config, stride: int):
        super().__init__()
        self.cfg = cfg
        self.stride = stride
        self.blocks = nn.ModuleList([FullBlock(cfg.d_model, cfg.n_heads, cfg.mlp_mult, cfg.rope_base, cfg.n_kv_heads)
                                      for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.code_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)  # plain linear passthrough, no quantization

    def run(self, x_embed: torch.Tensor) -> torch.Tensor:
        """x_embed: (M, L, D), M = however many independent row-instances are batched
        together (B rows for a single generation step, or B*img_size rows for a full
        teacher-forced training pass) -- full attention within each row, no attention
        across the M axis at all (that's what keeps rows from seeing each other)."""
        B, L, D = x_embed.shape
        head_dim = D // self.cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, self.cfg.rope_base, x_embed.device)
        h = x_embed
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.ln_f(h)

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        s = self.stride
        h = h.view(B, L // s, s, D)
        return h.mean(2) if self.cfg.code_extract_mode == "mean" else h[:, :, -1, :]

    def encode(self, x_embed: torch.Tensor) -> torch.Tensor:
        h = self.run(x_embed)
        pooled = self.pool(h)
        return self.code_head(pooled)  # (..., D) continuous, no discretization


class ImageEncoderFullAttn(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.r_embed = nn.Embedding(256, D)
        self.g_embed = nn.Embedding(256, D)
        self.b_embed = nn.Embedding(256, D)
        self.level0 = FullAttnEncoderLevel(cfg, cfg.strides[0])
        self.level1 = FullAttnEncoderLevel(cfg, cfg.strides[1])
        self.level2 = FullAttnEncoderLevel(cfg, cfg.strides[2])

    def encode_rows(self, r_rows: torch.Tensor, g_rows: torch.Tensor, b_rows: torch.Tensor) -> dict:
        """r_rows/g_rows/b_rows: (M, img_size) real bytes -- M independent rows (any mix
        of images/positions), each fully attended within itself, never across rows. Used
        both for one full row (M=B, generation) and for every row of a batch at once
        (M=B*img_size, training). code0/code1/code2 are continuous D-dim vectors, used
        directly as the next level's input -- no embedding table, nothing to look up."""
        x0 = self.r_embed(r_rows) + self.g_embed(g_rows) + self.b_embed(b_rows)
        code0 = self.level0.encode(x0)          # (M,16,D)
        code1 = self.level1.encode(code0)        # (M,4,D)
        code2 = self.level2.encode(code1)        # (M,1,D)
        return dict(code0=code0, code1=code1, code2=code2.squeeze(1))

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> dict:
        """Teacher-forced pass over a WHOLE real image at once (training): every row
        full-attended independently (rows folded into the batch dim), all in parallel."""
        B, img, _ = r.shape
        r_rows, g_rows, b_rows = r.reshape(B * img, img), g.reshape(B * img, img), b.reshape(B * img, img)
        out = self.encode_rows(r_rows, g_rows, b_rows)

        code0 = out["code0"].reshape(B, img * out["code0"].shape[1], -1)  # (B,L0,D)
        code1 = out["code1"].reshape(B, img * out["code1"].shape[1], -1)  # (B,L1,D)
        code2 = out["code2"].reshape(B, img, -1)                          # (B,img,D)
        return dict(code0=code0, code1=code1, code2=code2)


# ---------------------------------------------------------------------------
# Decoder: unchanged in kind from run_causalattn.py (causal, lag-1-row, GQA,
# column-batched SISO/MIMO/grouped, KV-cached, BOS-bootstrapped)
# ---------------------------------------------------------------------------

SLOT_L2, SLOT_L1, SLOT_L0, SLOT_R, SLOT_G, SLOT_B = range(6)
SLOT_L0_MTP, SLOT_RGB_MTP = 2, 3


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        # no dec_l2_embed/dec_l1_embed/dec_l0_embed -- codes are continuous D-dim vectors
        # already, used directly with no embedding table.
        self.byte_embed = nn.Embedding(256, D)
        n_slots = 4 if cfg.decoder_mode == "mtp" else 6
        self.slot_embed = nn.Embedding(n_slots, D)
        self.bos_l2 = nn.Parameter(torch.zeros(D))
        self.bos_l1 = nn.Parameter(torch.zeros(D))
        self.bos_l0 = nn.Parameter(torch.zeros(D))
        self.col_mix = ColumnMixAttention(D, cfg.n_heads, cfg.col_group_size)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.rope_base, cfg.n_kv_heads)
                                      for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(D)
        self.head_r = nn.Linear(D, 256, bias=False)
        self.head_g = nn.Linear(D, 256, bias=False)
        self.head_b = nn.Linear(D, 256, bias=False)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        head_dim = D // self.cfg.n_heads
        cos, sin = rope_cos_sin(T, head_dim, self.cfg.rope_base, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)

    def step(self, x_new: torch.Tensor, pos: int, caches: list) -> torch.Tensor:
        return blocks_step(self.blocks, self.ln_f, x_new, pos, caches)

    def new_caches(self) -> list:
        return [KVCache() for _ in range(self.cfg.n_layers)]

    def _lagged_code_embeds(self, code2, code1, code0, y_embed: torch.Tensor = None) -> tuple:
        """code2/code1/code0 are already continuous D-dim vectors (no embedding lookup)."""
        cfg = self.cfg
        img = cfg.img_size
        l2e = code2  # (B, img, D) -- one code2 per row
        l1e = code1.reshape(-1, img, code1.shape[1] // img, cfg.d_model)
        l0e = code0.reshape(-1, img, code0.shape[1] // img, cfg.d_model)
        B, D = l2e.shape[0], cfg.d_model
        l2e_lag = torch.cat([self.bos_l2.expand(B, 1, D), l2e[:, :-1]], dim=1)
        l1e_lag = torch.cat([self.bos_l1.expand(B, 1, l1e.shape[2], D), l1e[:, :-1]], dim=1)
        l0e_lag = torch.cat([self.bos_l0.expand(B, 1, l0e.shape[2], D), l0e[:, :-1]], dim=1)
        if y_embed is not None:
            l2e_lag = l2e_lag + y_embed.unsqueeze(1)
            l1e_lag = l1e_lag + y_embed.unsqueeze(1).unsqueeze(1)
            l0e_lag = l0e_lag + y_embed.unsqueeze(1).unsqueeze(1)
        return l2e_lag, l1e_lag, l0e_lag

    def _per_column_cond(self, code2, code1, code0, y_embed: torch.Tensor = None) -> tuple:
        cfg = self.cfg
        img = cfg.img_size
        l2e_lag, l1e_lag, l0e_lag = self._lagged_code_embeds(code2, code1, code0, y_embed)
        B, _, D = l2e_lag.shape
        n_l1_groups = l1e_lag.shape[2]
        n_l0_groups = l0e_lag.shape[2]
        cols = torch.arange(img, device=l2e_lag.device)
        l1_g = cols // (img // n_l1_groups)
        l0_g = cols // (img // n_l0_groups)
        l2e_col = l2e_lag.unsqueeze(2).expand(B, img, img, D).contiguous()
        l1e_col = l1e_lag[:, :, l1_g, :]
        l0e_col = l0e_lag[:, :, l0_g, :]
        l2e_col = self.col_mix(l2e_col)
        l1e_col = self.col_mix(l1e_col)
        l0e_col = self.col_mix(l0e_col)
        return l2e_col, l1e_col, l0e_col

    def row_cond_from_codes(self, code2_row, code1_row, code0_row, y_embed: torch.Tensor = None) -> tuple:
        """code2_row: (B,D); code1_row: (B,n_l1,D); code0_row: (B,n_l0,D) -- already
        continuous, no embedding lookup -- ONE row's own codes (real or just-realized) ->
        (l2e,l1e,l0e) each (B*img,D), ready to condition the NEXT row's decode. Same
        per-column broadcast/grouping/col_mix as _per_column_cond, just for a single row
        instead of all of them at once -- used by the interleaved generation loop, which
        computes each row's conditioning on the fly rather than having every row's codes
        precomputed."""
        cfg = self.cfg
        img = cfg.img_size
        D = cfg.d_model
        B = code2_row.shape[0]
        n_l1 = code1_row.shape[1]
        n_l0 = code0_row.shape[1]
        l2e = code2_row  # (B,D)
        l1e = code1_row  # (B,n_l1,D)
        l0e = code0_row  # (B,n_l0,D)
        cols = torch.arange(img, device=l2e.device)
        l1_g = cols // (img // n_l1)
        l0_g = cols // (img // n_l0)
        l2e_col = l2e.unsqueeze(1).expand(B, img, D)
        l1e_col = l1e[:, l1_g, :]
        l0e_col = l0e[:, l0_g, :]
        if y_embed is not None:
            l2e_col = l2e_col + y_embed.unsqueeze(1)
            l1e_col = l1e_col + y_embed.unsqueeze(1)
            l0e_col = l0e_col + y_embed.unsqueeze(1)
        l2e_col = self.col_mix(l2e_col.unsqueeze(1)).squeeze(1)
        l1e_col = self.col_mix(l1e_col.unsqueeze(1)).squeeze(1)
        l0e_col = self.col_mix(l0e_col.unsqueeze(1)).squeeze(1)
        return l2e_col.reshape(B * img, D), l1e_col.reshape(B * img, D), l0e_col.reshape(B * img, D)

    def bos_row_cond(self, B: int, device: torch.device, y_embed: torch.Tensor = None) -> tuple:
        """Row-0 bootstrap: BOS in place of a previous row's codes -- (l2e,l1e,l0e) each
        (B*img,D), uniform across columns (matches _lagged_code_embeds's row-0 handling
        exactly: bos_* is one vector broadcast to every group before col_mix)."""
        cfg = self.cfg
        img = cfg.img_size
        D = cfg.d_model
        outs = []
        for bos in (self.bos_l2, self.bos_l1, self.bos_l0):
            e = bos.view(1, 1, D).expand(B, img, D)
            if y_embed is not None:
                e = e + y_embed.unsqueeze(1)
            e = self.col_mix(e.unsqueeze(1)).squeeze(1)
            outs.append(e.reshape(B * img, D))
        return tuple(outs)

    def forward(self, code2, code1, code0, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor,
                y_embed: torch.Tensor = None) -> dict:
        """Teacher-forced training pass. r,g,b: (B,img,img) ground-truth bytes, [row,col]."""
        cfg = self.cfg
        img = cfg.img_size
        B, D = r.shape[0], cfg.d_model
        l2e_col, l1e_col, l0e_col = self._per_column_cond(code2, code1, code0, y_embed)
        r_e, g_e, b_e = self.byte_embed(r), self.byte_embed(g), self.byte_embed(b)

        if cfg.decoder_mode == "mtp":
            rgb_e = r_e + g_e + b_e
            slots = torch.stack([l2e_col, l1e_col, l0e_col, rgb_e], dim=3)
        else:
            slots = torch.stack([l2e_col, l1e_col, l0e_col, r_e, g_e, b_e], dim=3)
        n_slots = slots.shape[3]
        slots = slots + self.slot_embed.weight.view(1, 1, 1, n_slots, D)

        x = slots.permute(0, 2, 1, 3, 4).reshape(B * img, img * n_slots, D)
        h = self.run(x)
        h = h.view(B, img, img, n_slots, D).permute(0, 2, 1, 3, 4)

        if cfg.decoder_mode == "mtp":
            h_seed = h[:, :, :, SLOT_L0_MTP, :]
            logits_r, logits_g, logits_b = self.head_r(h_seed), self.head_g(h_seed), self.head_b(h_seed)
        else:
            logits_r = self.head_r(h[:, :, :, SLOT_L0, :])
            logits_g = self.head_g(h[:, :, :, SLOT_R, :])
            logits_b = self.head_b(h[:, :, :, SLOT_G, :])

        loss_r = F.cross_entropy(logits_r.reshape(-1, 256), r.reshape(-1))
        loss_g = F.cross_entropy(logits_g.reshape(-1, 256), g.reshape(-1))
        loss_b = F.cross_entropy(logits_b.reshape(-1, 256), b.reshape(-1))
        with torch.no_grad():
            acc = ((logits_r.argmax(-1) == r).float().mean()
                   + (logits_g.argmax(-1) == g).float().mean()
                   + (logits_b.argmax(-1) == b).float().mean()) / 3
        return dict(loss=(loss_r + loss_g + loss_b) / 3, acc=acc)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ImageGenCIFARFullAttn(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoderFullAttn(cfg)
        self.decoder = Decoder(cfg)
        if cfg.class_conditional:
            self.class_embed = nn.Embedding(cfg.n_classes, cfg.d_model)

    def _y_embed(self, y: torch.Tensor) -> torch.Tensor:
        return self.class_embed(y) if self.cfg.class_conditional else None

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, y: torch.Tensor = None) -> dict:
        y_embed = self._y_embed(y)
        enc = self.encoder(r, g, b)
        dec = self.decoder(enc["code2"], enc["code1"], enc["code0"], r, g, b, y_embed=y_embed)
        bpb = dec["loss"] / math.log(2)
        return dict(loss=dec["loss"], byte_loss=dec["loss"], bpb=bpb, acc=dec["acc"])

    @torch.no_grad()
    def generate(self, n: int, device: torch.device, greedy: bool = True,
                 y: "torch.Tensor | int | None" = None,
                 prompt_r: torch.Tensor = None, prompt_g: torch.Tensor = None,
                 prompt_b: torch.Tensor = None) -> torch.Tensor:
        """Genuinely interleaved per-row generation (no code-level prior exists to
        front-load, unlike run_causalattn.py): encode row r's REAL bytes (either
        `prompt_r/g/b`'s given rows, or the model's own row-r output once realized) ->
        that row's codes condition the decoder's row r+1 prediction -> repeat. Row 0
        with no prompt uses the decoder's learned BOS conditioning (bos_l2/l1/l0) as its
        bootstrap, same mechanism as run_causalattn.py.

        prompt_r/g/b: (n, n_prompt, img) real bytes for the first n_prompt rows, or None
        (n_prompt=0, fully unconditional from BOS)."""
        cfg = self.cfg
        img = cfg.img_size
        dec = self.decoder
        D = cfg.d_model
        n_prompt = prompt_r.shape[1] if prompt_r is not None else 0

        if cfg.class_conditional:
            if y is None:
                y = torch.randint(0, cfg.n_classes, (n,), device=device)
            elif isinstance(y, int):
                y = torch.full((n,), y, dtype=torch.long, device=device)
            else:
                y = y.to(device)
        y_embed = self._y_embed(y)  # (n,D) or None -- row_cond_from_codes/bos_row_cond broadcast internally

        def sample_from(logits):
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

        Bc = n * img
        slot_w = dec.slot_embed.weight
        caches = dec.new_caches()
        pos = 0
        r_out = torch.zeros(n, img, img, dtype=torch.long, device=device)
        g_out = torch.zeros(n, img, img, dtype=torch.long, device=device)
        b_out = torch.zeros(n, img, img, dtype=torch.long, device=device)

        l2e_prev, l1e_prev, l0e_prev = dec.bos_row_cond(n, device, y_embed)  # each (Bc,D)

        for row in range(img):
            dec.step((l2e_prev + slot_w[SLOT_L2]).unsqueeze(1), pos, caches); pos += 1
            dec.step((l1e_prev + slot_w[SLOT_L1]).unsqueeze(1), pos, caches); pos += 1
            l0_slot = slot_w[SLOT_L0_MTP if cfg.decoder_mode == "mtp" else SLOT_L0]
            h_seed = dec.step((l0e_prev + l0_slot).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1

            if row < n_prompt:
                r_row = prompt_r[:, row, :].reshape(Bc).to(device)
                g_row = prompt_g[:, row, :].reshape(Bc).to(device)
                b_row = prompt_b[:, row, :].reshape(Bc).to(device)
                if cfg.decoder_mode == "mtp":
                    rgb_e = (dec.byte_embed(r_row) + dec.byte_embed(g_row) + dec.byte_embed(b_row)
                             + slot_w[SLOT_RGB_MTP]).unsqueeze(1)
                    dec.step(rgb_e, pos, caches); pos += 1
                else:
                    dec.step((dec.byte_embed(r_row) + slot_w[SLOT_R]).unsqueeze(1), pos, caches); pos += 1
                    dec.step((dec.byte_embed(g_row) + slot_w[SLOT_G]).unsqueeze(1), pos, caches); pos += 1
                    dec.step((dec.byte_embed(b_row) + slot_w[SLOT_B]).unsqueeze(1), pos, caches); pos += 1
            elif cfg.decoder_mode == "mtp":
                r_row = sample_from(dec.head_r(h_seed))
                g_row = sample_from(dec.head_g(h_seed))
                b_row = sample_from(dec.head_b(h_seed))
                rgb_e = (dec.byte_embed(r_row) + dec.byte_embed(g_row) + dec.byte_embed(b_row)
                         + slot_w[SLOT_RGB_MTP]).unsqueeze(1)
                dec.step(rgb_e, pos, caches); pos += 1
            else:
                r_row = sample_from(dec.head_r(h_seed))
                h_r = dec.step((dec.byte_embed(r_row) + slot_w[SLOT_R]).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1
                g_row = sample_from(dec.head_g(h_r))
                h_g = dec.step((dec.byte_embed(g_row) + slot_w[SLOT_G]).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1
                b_row = sample_from(dec.head_b(h_g))
                dec.step((dec.byte_embed(b_row) + slot_w[SLOT_B]).unsqueeze(1), pos, caches); pos += 1

            r_out[:, row, :] = r_row.view(n, img)
            g_out[:, row, :] = g_row.view(n, img)
            b_out[:, row, :] = b_row.view(n, img)

            if row < img - 1:  # no need to encode the last row -- nothing left to condition
                enc_out = self.encoder.encode_rows(r_row.view(n, img), g_row.view(n, img), b_row.view(n, img))
                l2e_prev, l1e_prev, l0e_prev = dec.row_cond_from_codes(
                    enc_out["code2"], enc_out["code1"], enc_out["code0"], y_embed)

        return torch.stack([r_out, g_out, b_out], dim=-1).clamp(0, 255).to(torch.uint8)  # (n,img,img,3)


# ---------------------------------------------------------------------------
# Logging / checkpointing (hard-forked, minimal)
# ---------------------------------------------------------------------------

class _Tee:
    """stdout/stderr + file, eager flush -- tail -f <log_dir>/train.log to watch a run live."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    @property
    def encoding(self):
        return getattr(self.files[0], "encoding", "utf-8")

    def isatty(self):
        return self.files[0].isatty()


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.text_f = open(run_dir / "run.log", "a")
        self.json_f = open(run_dir / "run.jsonl", "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        line = f"[{format_hms(elapsed_s)}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        rec = {"elapsed_s": elapsed_s, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(rec) + "\n")
        self.json_f.flush()


class Checkpointer:
    def __init__(self, run_dir: Path, minimize: bool = True):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = run_dir / "best.pt"
        self.last_path = run_dir / "last.pt"
        self.minimize = minimize
        self.best_metric = float("inf") if minimize else float("-inf")

    def step(self, state: dict, metric: float) -> None:
        if math.isfinite(metric) and (metric < self.best_metric if self.minimize else metric > self.best_metric):
            self.best_metric = metric
            torch.save(state, self.best_path)
        torch.save(state, self.last_path)


# ---------------------------------------------------------------------------
# Training / inference entry point
# ---------------------------------------------------------------------------

def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def write_resolved_config(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items())]
    (run_dir / "resolved_config.py").write_text("\n".join(lines) + "\n")


def save_sample_grid(samples: np.ndarray, path: Path, pad: int = 2) -> None:
    """samples: (n,H,W,3) uint8 -- tile into a near-square grid on a white background."""
    n, h, w, c = samples.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i, img in enumerate(samples):
        row, col = divmod(i, cols)
        y, x = pad + row * (h + pad), pad + col * (w + pad)
        grid[y:y + h, x:x + w] = img
    Image.fromarray(grid).save(path)


CONFIG_FIELDS = ("img_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "code_extract_mode", "decoder_mode", "rope_base", "mlp_mult", "col_group_size",
                  "class_conditional", "n_classes")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None,
                      help="Python config file (image_gen_cifar/configs/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_fullattn_noquant_minimal")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=5, help="run val eval + qual-gen samples every N epochs")
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=int, default=Config.n_kv_heads)
    p.add_argument("--strides", type=lambda s: tuple(int(x) for x in s.split(",")), default=Config.strides)
    p.add_argument("--code_extract_mode", type=str, default=Config.code_extract_mode, choices=["mean", "last_idx"])
    p.add_argument("--decoder_mode", type=str, default=Config.decoder_mode, choices=["seq", "mtp"])
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--col_group_size", type=int, default=Config.col_group_size)
    p.add_argument("--class_conditional", type=lambda x: x.lower() != "false", default=Config.class_conditional)
    p.add_argument("--n_classes", type=int, default=Config.n_classes)

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        unknown = set(config_vars) - known
        if unknown:
            p.error(f"--config {pre_args.config} sets unknown field(s): {sorted(unknown)}")
        p.set_defaults(**config_vars)
    args = p.parse_args()

    device = get_device(args.device)
    data_root = Path(args.data_root)
    (train_np, train_labels), (val_np, val_labels) = load_cifar10(data_root)
    train_loader = DataLoader(CIFARDataset(train_np, train_labels), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(CIFARDataset(val_np, val_labels), batch_size=args.batch_size, shuffle=False, drop_last=True)

    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})
    model = ImageGenCIFARFullAttn(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = MODULE_DIR / "logs" / args.run_name
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(run_dir / "train.log", "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    logger = Logger(run_dir)
    ckpt = Checkpointer(run_dir)
    write_resolved_config(run_dir, args)
    if pre_args.config:
        (run_dir / f"config_{pre_args.config.name}").write_text(pre_args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    logger(f"params: total={((n_enc + n_dec) / 1e6):.2f}M encoder={n_enc / 1e6:.2f}M decoder={n_dec / 1e6:.2f}M device={device}",
           params_total=n_enc + n_dec, params_encoder=n_enc, params_decoder=n_dec)

    if args.checkpoint_path:
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        logger(f"loaded checkpoint {args.checkpoint_path}")

    def run_eval(loader, tag: str) -> float:
        model.eval()
        tot_bpb, tot_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for r, g, b, y in loader:
                r, g, b, y = r.to(device), g.to(device), b.to(device), y.to(device)
                out = model(r, g, b, y=y)
                tot_bpb += out["bpb"].item()
                tot_acc += out["acc"].item()
                n += 1
                if n >= 20:
                    break
        model.train()
        bpb, acc = tot_bpb / max(n, 1), tot_acc / max(n, 1)
        logger(f"{tag} bpb={bpb:.4f} acc={acc:.4f}", **{f"{tag}_bpb": bpb, f"{tag}_acc": acc})
        return bpb

    def run_qual_gen(epoch: int) -> None:
        model.eval()
        samples = model.generate(args.qual_gen_n, device, greedy=True)
        model.train()
        out_path = run_dir / f"samples_epoch{epoch}.png"
        save_sample_grid(samples.cpu().numpy(), out_path)
        logger(f"saved {args.qual_gen_n} qual-gen samples to {out_path}")

    if args.eval_only:
        run_eval(val_loader, "val")
        return

    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for r, g, b, y in pbar:
            r, g, b, y = r.to(device), g.to(device), b.to(device), y.to(device)
            out = model(r, g, b, y=y)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            step += 1

            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb={out['bpb'].item():.4f} acc={out['acc'].item():.4f}",
                       epoch=epoch, step=step, train_bpb=out["bpb"].item(), train_acc=out["acc"].item())
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            val_bpb = run_eval(val_loader, "val")
            ckpt.step({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "step": step}, val_bpb)
            run_qual_gen(epoch)
    logger("training done")


if __name__ == "__main__":
    main()
