"""ClockworkRNN-style decoder-only AR baseline, forked from run_causalattn.py, standalone
codebase (no imports from run_causalattn.py or qcute_lagcodec).

No encoder, no latent codes at all -- a plain (multi-rate) autoregressive pixel model, a
baseline to see what the hierarchical-code designs elsewhere in this directory are actually
buying over a simpler alternative.

The whole sequence runs at ROW (scanline) granularity, not per-pixel: each of the 32 rows
of a 32x32 image is pooled (mean of its 32 PQ-embedded pixels, 3x256 R/G/B tables summed)
into one row-embedding, and the model predicts the ENTIRE next row's 32x3 bytes in one
parallel shot from that row's hidden state -- an NTP objective "32 positions ahead", not
per-byte. Row 0 has no real previous row, so it's conditioned on a TRAINABLE PIXEL LINE (a
learned (img_size, embed_dim) tensor living directly in embedding space, pooled the same
way as a real row) -- chosen here (unlike the BOS-vector choice in the other forks) because
there's no encoder left to bypass or route through: a trainable line IS the natural,
un-special-cased way to seed this single-stack model.

Clockwork multi-rate hierarchy: `strides` (len == number of levels) sets each level's CLOCK
PERIOD over this row-sequence -- level i only runs its own transformer stack when
`row_idx % strides[i] == 0`; on every other row it does ZERO compute and its hidden state is
simply held constant from its last tick (true ClockworkRNN, not a cheaper approximation).
strides[0] must be 1 (fastest, ticks every row -- it's the level whose hidden state
actually drives the row prediction). Faster levels read every slower level's CURRENT (held)
state as additive conditioning before their own attention (matches ClockworkRNN's
connectivity: fast modules see slow ones, never the reverse) -- this is where the "save
attention cost" benefit actually comes from: a level with stride 4 only pays its attention+
MLP cost on 1/4 of the rows a flat transformer would. `d_model`/`n_layers`/`n_heads` are
per-level tuples (one entry per stride), so slower/coarser levels can be cheaper (or more
expensive) than the fast one independently.

Training: the whole strided hierarchy is computed in one teacher-forced parallel pass per
level (real rows are all known upfront, same AR-training trick used everywhere else in this
directory) -- each level only processes ITS OWN ticked subsequence (e.g. every-other-row for
stride=2), never touching the intermediate rows at all, exactly mirroring what the
KV-cached generation loop below actually does step by step.
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
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"labels"], dtype=np.int64)
        return images, labels

    train_batches = [load_batch(f"data_batch_{i}") for i in range(1, 6)]
    train = np.concatenate([b[0] for b in train_batches], axis=0)
    train_labels = np.concatenate([b[1] for b in train_batches], axis=0)
    test, test_labels = load_batch("test_batch")
    return (train, train_labels), (test, test_labels)


class CIFARDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple:
        img = self.images[i]
        r = torch.from_numpy(img[:, :, 0].astype(np.int64))
        g = torch.from_numpy(img[:, :, 1].astype(np.int64))
        b = torch.from_numpy(img[:, :, 2].astype(np.int64))
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return r, g, b, y


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    img_size: int = 32
    embed_dim: int = 256              # shared R/G/B pixel embedding width, pooled per row
    d_model: tuple = (256, 256, 128, 256)  # per clock-level width
    n_layers: tuple = (2, 2, 2, 2)         # per clock-level depth
    n_heads: tuple = (4, 4, 4, 4)     # per clock-level head count
    n_kv_heads: tuple = (None, None, None, None)  # per-level GQA kv heads; None -> max(1,n_heads[i]//4)
    strides: tuple = (1, 2, 4, 1)     # SANDWICH pattern: strides[0]==1 (fast input level), a
    # non-decreasing "filling" ramp in between (slower/coarser summary levels), strides[-1]==1
    # (fast COLLECTOR level, forced to tick every row) -- the collector is what actually drives
    # the row prediction now (not level0): it reads every other level unconditionally, including
    # same-speed level0, so it's the one place that must integrate the fast detail AND every
    # slow summary before predicting. Non-last levels keep the plain clockwork rule (read only
    # strictly-slower levels).
    mlp_mult: int = 4
    rope_base: float = 10000.0
    class_conditional: bool = False
    n_classes: int = 10

    def __post_init__(self):
        n = len(self.strides)
        assert n >= 2, "sandwich pattern needs at least [1, 1] (input level + collector level)"
        assert len(self.d_model) == n and len(self.n_layers) == n and len(self.n_heads) == n \
            and len(self.n_kv_heads) == n, "d_model/n_layers/n_heads/n_kv_heads must match len(strides)"
        assert self.strides[0] == 1, "level0 must tick every row (fast input level)"
        assert self.strides[-1] == 1, "last level must tick every row (fast COLLECTOR level)"
        assert all(self.strides[i] <= self.strides[i + 1] for i in range(n - 2)), \
            "the filling strides[1:-1] must be non-decreasing (ramping slower/coarser toward the middle)"
        resolved_kv = []
        for i in range(n):
            kv = self.n_kv_heads[i] if self.n_kv_heads[i] is not None else max(1, self.n_heads[i] // 4)
            assert self.n_heads[i] % kv == 0
            assert self.d_model[i] % self.n_heads[i] == 0
            resolved_kv.append(kv)
        self.n_kv_heads = tuple(resolved_kv)


# ---------------------------------------------------------------------------
# Common building blocks (hard-forked)
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
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None


class CausalSelfAttention(nn.Module):
    """Qwen3-style: RoPE + per-head QK-norm + GQA -- identical pattern to the other forks
    in this directory, hard-forked here rather than imported."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0
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


class LevelStack(nn.Module):
    """One clock-level's own small causal transformer, operating over ONLY that level's
    ticked positions (never sees the rows it skips)."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.d_model, self.n_heads, self.rope_base = d_model, n_heads, rope_base
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_mult, rope_base, n_kv_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,D), T = this level's own number of ticks so far -- full causal pass,
        used during training (whole strided subsequence known upfront)."""
        B, T, D = x.shape
        head_dim = D // self.n_heads
        cos, sin = rope_cos_sin(T, head_dim, self.rope_base, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)

    def step(self, x_new: torch.Tensor, tick_pos: int, caches: list) -> torch.Tensor:
        """x_new: (B,1,D), one new TICK (not row) -- only called on this level's own
        clock ticks; tick_pos is this level's own tick counter, not the row index."""
        h = x_new
        for blk, cache in zip(self.blocks, caches):
            h = blk.forward_incremental(h, tick_pos, cache)
        return self.ln_f(h)

    def new_caches(self) -> list:
        return [KVCache() for _ in self.blocks]


class ClockworkAR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.strides)
        self.r_embed = nn.Embedding(256, cfg.embed_dim)
        self.g_embed = nn.Embedding(256, cfg.embed_dim)
        self.b_embed = nn.Embedding(256, cfg.embed_dim)
        # trainable "virtual" first row, living directly in embedding space (not real bytes).
        # Small-noise init, NOT zeros: this feeds input_proj then straight into RMSNorm with
        # nothing else added first (unlike the other forks' BOS vectors, always summed with a
        # nonzero slot/code embedding before their first norm) -- an exactly-zero input makes
        # RMSNorm's rsqrt(mean(x^2)+eps) term numerically pathological (gradient dominated by
        # eps^-1/2 per norm layer, compounding across depth/levels into a real blow-up, observed
        # directly: ~1e11 grad norm on this parameter with zeros init).
        self.bootstrap_row = nn.Parameter(torch.randn(cfg.img_size, cfg.embed_dim) * 0.02)

        self.input_proj = nn.ModuleList([nn.Linear(cfg.embed_dim, cfg.d_model[i], bias=False)
                                          for i in range(self.n_levels)])
        self.levels = nn.ModuleList([
            LevelStack(cfg.d_model[i], cfg.n_layers[i], cfg.n_heads[i], cfg.n_kv_heads[i], cfg.mlp_mult, cfg.rope_base)
            for i in range(self.n_levels)
        ])
        # cond_proj[i][j]: projects level j's state into level i's width. Non-last levels
        # keep the plain clockwork rule (read only strictly-slower levels, among the
        # non-collector levels). The LAST level is the sandwich's fast COLLECTOR: it reads
        # every other level unconditionally (including same-speed level0) -- it's the one
        # place that must integrate the fast detail and every slow summary.
        self.collector = self.n_levels - 1
        self.cond_proj = nn.ModuleList()
        for i in range(self.n_levels):
            projs = nn.ModuleDict()
            for j in self._reads_of(i):
                projs[str(j)] = nn.Linear(cfg.d_model[j], cfg.d_model[i], bias=False)
            self.cond_proj.append(projs)

        Dc = cfg.d_model[self.collector]
        self.head_r = nn.Linear(Dc, cfg.img_size * 256, bias=False)
        self.head_g = nn.Linear(Dc, cfg.img_size * 256, bias=False)
        self.head_b = nn.Linear(Dc, cfg.img_size * 256, bias=False)
        if cfg.class_conditional:
            self.class_embed = nn.Embedding(cfg.n_classes, cfg.embed_dim)

    def pool_row(self, r_row: torch.Tensor, g_row: torch.Tensor, b_row: torch.Tensor) -> torch.Tensor:
        """r_row/g_row/b_row: (...,img_size) byte values -> (...,embed_dim) pooled row embedding."""
        e = self.r_embed(r_row) + self.g_embed(g_row) + self.b_embed(b_row)
        return e.mean(dim=-2)

    def _y_embed(self, y: torch.Tensor) -> torch.Tensor:
        return self.class_embed(y) if self.cfg.class_conditional else None

    def _reads_of(self, i: int) -> list:
        """Which levels level i reads as additive conditioning. The collector (last
        level) reads everyone else unconditionally; every other level keeps the plain
        clockwork rule (read only strictly-slower levels, among non-collector levels)."""
        if i == self.collector:
            return [j for j in range(self.n_levels) if j != i]
        return [j for j in range(self.n_levels) if j != self.collector and self.cfg.strides[j] > self.cfg.strides[i]]

    def _level_order(self) -> list:
        """Non-collector levels slowest-to-fastest first (so a faster one can read an
        already-computed slower one), collector strictly last (it needs everyone else
        computed first)."""
        non_collector = sorted(range(self.n_levels - 1), key=lambda i: -self.cfg.strides[i])
        return non_collector + [self.collector]

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, y: torch.Tensor = None) -> dict:
        """Teacher-forced training pass. r,g,b: (B,img,img) real bytes, [row,col]."""
        cfg = self.cfg
        B, img, _ = r.shape
        row_e = self.pool_row(r, g, b)  # (B,img,embed_dim) -- all 32 real rows
        boot = self.bootstrap_row.mean(dim=0).view(1, 1, -1).expand(B, 1, -1)
        y_embed = self._y_embed(y)
        if y_embed is not None:
            row_e = row_e + y_embed[:, None, :]
            boot = boot + y_embed[:, None, :]
        x_in = torch.cat([boot, row_e[:, :-1]], dim=1)  # (B,img,embed_dim) -- lag-1: predicts real row t from row t-1

        held = [None] * self.n_levels
        for i in self._level_order():
            stride_i = cfg.strides[i]
            idx = torch.arange(0, img, stride_i, device=x_in.device)
            xi = self.input_proj[i](x_in[:, idx])
            for j in self._reads_of(i):
                xi = xi + self.cond_proj[i][str(j)](held[j][:, idx])
            hi = self.levels[i].run(xi)
            held[i] = hi.repeat_interleave(stride_i, dim=1)[:, :img]  # hold constant between ticks

        h_out = held[self.collector]  # (B,img,d_model[-1]) -- collector, fresh every row
        logits_r = self.head_r(h_out).view(B, img, cfg.img_size, 256)
        logits_g = self.head_g(h_out).view(B, img, cfg.img_size, 256)
        logits_b = self.head_b(h_out).view(B, img, cfg.img_size, 256)

        loss_r = F.cross_entropy(logits_r.reshape(-1, 256), r.reshape(-1))
        loss_g = F.cross_entropy(logits_g.reshape(-1, 256), g.reshape(-1))
        loss_b = F.cross_entropy(logits_b.reshape(-1, 256), b.reshape(-1))
        with torch.no_grad():
            acc = ((logits_r.argmax(-1) == r).float().mean()
                   + (logits_g.argmax(-1) == g).float().mean()
                   + (logits_b.argmax(-1) == b).float().mean()) / 3
        loss = (loss_r + loss_g + loss_b) / 3
        return dict(loss=loss, bpb=loss / math.log(2), acc=acc)

    @torch.no_grad()
    def generate(self, n: int, device: torch.device, greedy: bool = True,
                 y: "torch.Tensor | int | None" = None) -> torch.Tensor:
        """KV-cached, genuinely clocked: level i only computes (and only advances its own
        cache) on rows where row_idx % strides[i] == 0 -- zero compute on off-ticks, its
        held state just carries forward unchanged."""
        cfg = self.cfg
        img = cfg.img_size
        if cfg.class_conditional:
            if y is None:
                y = torch.randint(0, cfg.n_classes, (n,), device=device)
            elif isinstance(y, int):
                y = torch.full((n,), y, dtype=torch.long, device=device)
            y_embed = self._y_embed(y.to(device))
        else:
            y_embed = None

        caches = [self.levels[i].new_caches() for i in range(self.n_levels)]
        held = [None] * self.n_levels
        tick_pos = [0] * self.n_levels

        def sample(logits):
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

        x_input = self.bootstrap_row.mean(dim=0).view(1, -1).expand(n, -1)
        if y_embed is not None:
            x_input = x_input + y_embed

        r_out = torch.zeros(n, img, img, dtype=torch.long, device=device)
        g_out = torch.zeros(n, img, img, dtype=torch.long, device=device)
        b_out = torch.zeros(n, img, img, dtype=torch.long, device=device)

        for t in range(img):
            for i in self._level_order():  # non-collector slowest-to-fastest, collector last
                if t % cfg.strides[i] == 0:
                    xi = self.input_proj[i](x_input)
                    for j in self._reads_of(i):
                        xi = xi + self.cond_proj[i][str(j)](held[j])
                    hi = self.levels[i].step(xi.unsqueeze(1), tick_pos[i], caches[i])[:, 0, :]
                    tick_pos[i] += 1
                    held[i] = hi
                # else: held[i] unchanged, zero compute this row for this level

            h_out = held[self.collector]
            logits_r = self.head_r(h_out).view(n, img, 256)
            logits_g = self.head_g(h_out).view(n, img, 256)
            logits_b = self.head_b(h_out).view(n, img, 256)
            row_r, row_g, row_b = sample(logits_r), sample(logits_g), sample(logits_b)
            r_out[:, t, :], g_out[:, t, :], b_out[:, t, :] = row_r, row_g, row_b

            x_input = self.pool_row(row_r, row_g, row_b)
            if y_embed is not None:
                x_input = x_input + y_embed

        return torch.stack([r_out, g_out, b_out], dim=-1).clamp(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Logging / checkpointing (hard-forked, minimal)
# ---------------------------------------------------------------------------

class _Tee:
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
    n, h, w, c = samples.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i, img in enumerate(samples):
        row, col = divmod(i, cols)
        y, x = pad + row * (h + pad), pad + col * (w + pad)
        grid[y:y + h, x:x + w] = img
    Image.fromarray(grid).save(path)


CONFIG_FIELDS = ("img_size", "embed_dim", "d_model", "n_layers", "n_heads", "n_kv_heads", "strides",
                  "mlp_mult", "rope_base", "class_conditional", "n_classes")


def _tuple_arg(s: str) -> tuple:
    return tuple(None if x.strip().lower() == "none" else int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None,
                      help="Python config file (image_gen_cifar/configs/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_ar_clockwork")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--embed_dim", type=int, default=Config.embed_dim)
    p.add_argument("--d_model", type=_tuple_arg, default=Config.d_model)
    p.add_argument("--n_layers", type=_tuple_arg, default=Config.n_layers)
    p.add_argument("--n_heads", type=_tuple_arg, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=_tuple_arg, default=Config.n_kv_heads)
    p.add_argument("--strides", type=_tuple_arg, default=Config.strides)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
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
    model = ClockworkAR(cfg).to(device)
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
    n_params = sum(p.numel() for p in model.parameters())
    logger(f"params: {n_params / 1e6:.2f}M, device={device}")

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
