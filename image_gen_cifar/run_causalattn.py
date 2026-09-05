"""Single-file hierarchical latent-AR image generator for CIFAR-10, hard-forked from
qcute_lagcodec's design (no imports from that package -- every primitive rewritten here).

Encoder: 3-level causal-LM hierarchy over a row-major-flattened 32x32 pixel scanline.
strides=(2,4,4) (product 32 == image width) so level2's 32 output codes land exactly
one-per-row. Each level's own code head mean-pools (default) or last-idx-pools its
stride-window of hidden states before the linear projection to a categorical code
(gumbel-hard STE). Levels 1/2 also carry an NTP head (auxiliary loss) so they can act as
free-running priors at generation time -- level0 has none: its generative role is the
Decoder itself.

Decoder: plain GPT-style decoder-only causal self-attention LM (no cross-attention, no
seed-token/block-folding machinery) -- codes are just embedded and concatenated into the
ordinary token sequence. Batched 32-ways over image COLUMNS (one clone per column, zero
inter-clone communication), each clone autoregressing top-to-bottom over its column's 32
rows: [level2_code, level1_code, level0_code(seed), R, G, B] per row (or
[level2,level1,level0-seed,RGB-packed] under --decoder_mode mtp). To keep this a valid
chain-rule NLL (not an ELBO-style reconstruction bound) every code fed into row r's decode
is LAGGED BY ONE ROW (computed from rows < r only) since the true encoder codes for row r
are themselves computed causally over pixels through the end of row r -- using them
unlagged would leak that row's own not-yet-generated pixels into its own conditioning.
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
    n_kv_heads: int = None  # None = max(1, n_heads//4) GQA-by-default (Llama3/Qwen3-style); == n_heads for plain MHA
    code_vocab: int = 16              # PER-CHUNK codebook width (product-quantized, see pq_chunks)
    pq_chunks: int = 4                # PQ groups per level's code; combinatorial capacity code_vocab**pq_chunks
    strides: tuple = (2, 4, 4)       # per-level downsample stride; product must == img_size
    code_extract_mode: str = "mean"  # "mean" (default, pool stride-window) or "last_idx"
    decoder_mode: str = "seq"        # "seq" (default, sequential R->G->B) or "mtp"
    rope_base: float = 10000.0
    mlp_mult: int = 4
    ntp_aux_weight: float = 1.0
    col_group_size: int = 1  # decoder column-track communication: 1=SISO (independent, default),
    # img_size=MIMO (full entanglement), else grouped (GQA-/group-conv-style) -- see ColumnMixAttention
    class_conditional: bool = False  # broadcast a learned per-class embedding into every row's
    # code conditioning (decoder) and every AR step (encoder's generative priors) -- same
    # "inject a learned vector where BOS would go" mechanism as the BOS tokens, but data-dependent
    # (on the label) and broadcast every step rather than only at the start, so the signal doesn't
    # dilute over the sequence. See CIFAR-10's 10 classes.
    n_classes: int = 10

    def __post_init__(self):
        assert math.prod(self.strides) == self.img_size, \
            f"product(strides)={math.prod(self.strides)} must equal img_size={self.img_size}"
        assert self.d_model % self.n_heads == 0
        assert self.img_size % self.col_group_size == 0


# ---------------------------------------------------------------------------
# Common building blocks (hard-forked from qcute_lagcodec_common.py)
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
    """cos/sin for a single absolute position -- O(head_dim), used by the KV-cached
    incremental decode path instead of recomputing a whole rope table per step."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None


class CausalSelfAttention(nn.Module):
    """Qwen3-style: RoPE + per-head QK-norm + GQA (n_kv_heads < n_heads repeats each
    KV head across n_heads//n_kv_heads query heads, Llama3/Qwen3-style), ported from
    qcute_lagcodec_common.py's CausalSelfAttention (same combined-qkv-weight-slicing /
    _repeat_kv pattern, hard-forked here rather than imported)."""

    def __init__(self, d_model: int, n_heads: int, rope_base: float, n_kv_heads: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        # combined Linear (not split wq/wk/wv), weight-sliced in _project_qkv -- mirrors
        # qcute_lagcodec_common.py's CausalSelfAttention exactly
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
        """x_new: exactly one new token, (B,1,D). Appends this step's K/V to `cache` in
        place -- cache stores at n_kv_heads width (the whole point of GQA's smaller KV
        cache), _repeat_kv happens only right before attention, not before caching. No
        causal mask needed: the single query is always the temporally-last position, so
        full attention over cache+self is already exactly causal."""
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
    """Non-causal self-attention across the COLUMN axis (not the row/time axis that the
    main Blocks handle) -- lets column-tracks' code conditioning communicate before each
    column proceeds independently down its own causal row-AR track. Grouped by
    `group_size`, GQA-/grouped-conv-style: columns commmunicate within their own group of
    `group_size` consecutive columns, groups never see each other. group_size=1 (default)
    is a true no-op (SISO: today's fully-independent column-tracks); group_size=img_size
    is full entanglement (MIMO: every column sees every other column); values in between
    are the grouped middle ground. Always safe to apply non-causally: it only ever mixes
    a row's CODE conditioning, which is fully known (from the encoder) before any byte in
    that row is predicted -- never mixes byte predictions themselves, so it can't leak a
    not-yet-generated pixel into another column's conditioning."""

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
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, R, C//g, H, g, hd)
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


def quantize_hard(logits: torch.Tensor) -> tuple:
    """Deterministic hard-argmax categorical with straight-through gradient (no gumbel noise)."""
    soft = F.softmax(logits, dim=-1)
    idx = soft.argmax(-1)
    hard = F.one_hot(idx, logits.shape[-1]).to(soft.dtype)
    return soft + (hard - soft).detach(), idx


@torch.no_grad()
def codebook_utilization(idx: torch.Tensor, vocab: int) -> torch.Tensor:
    """idx: (...,pq_chunks) long code ids in [0,vocab). Logged-only diagnostic (no
    gradient, not part of the loss) -- per-chunk usage-perplexity normalized to [0,1]
    (1.0 = batch uses every code in that chunk uniformly, ~0 = collapsed to a few
    codes), averaged over chunks. Each PQ chunk is its own codebook, so its usage
    marginal is computed separately, never pooled across chunks."""
    flat = idx.reshape(-1, idx.shape[-1])
    utils = []
    for c in range(flat.shape[-1]):
        counts = F.one_hot(flat[:, c], vocab).sum(0).float()
        probs = counts / counts.sum().clamp_min(1)
        ent = -(probs * probs.clamp_min(1e-9).log()).sum()
        utils.append(ent.exp() / vocab)
    return torch.stack(utils).mean()


def code_embed(code: torch.Tensor, table: nn.Embedding) -> torch.Tensor:
    """code: STE soft per-chunk one-hot (..., pq_chunks, V), training, or realized
    per-chunk ids (..., pq_chunks), generation -- sums the pq_chunks sub-embeddings into
    one combined vector (product-quantized code -> single embedding)."""
    if code.dtype in (torch.long, torch.int64):
        return table(code).sum(-2)
    return (code @ table.weight).sum(-2)


# ---------------------------------------------------------------------------
# Encoder: 3-level causal-LM hierarchy
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    def __init__(self, cfg: Config, stride: int, has_ntp: bool):
        super().__init__()
        self.cfg = cfg
        self.stride = stride
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, cfg.rope_base, cfg.n_kv_heads) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.code_head = nn.Linear(cfg.d_model, cfg.pq_chunks * cfg.code_vocab, bias=False)
        self.ntp_head = nn.Linear(cfg.d_model, cfg.pq_chunks * cfg.code_vocab, bias=False) if has_ntp else None

    def run(self, x_embed: torch.Tensor) -> torch.Tensor:
        B, L, D = x_embed.shape
        head_dim = D // self.cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, self.cfg.rope_base, x_embed.device)
        h = x_embed
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.ln_f(h)

    def step(self, x_new: torch.Tensor, pos: int, caches: list) -> torch.Tensor:
        return blocks_step(self.blocks, self.ln_f, x_new, pos, caches)

    def new_caches(self) -> list:
        return [KVCache() for _ in range(self.cfg.n_layers)]

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        s = self.stride
        h = h.view(B, L // s, s, D)
        return h.mean(2) if self.cfg.code_extract_mode == "mean" else h[:, :, -1, :]

    def _reshape_pq(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.view(*logits.shape[:-1], self.cfg.pq_chunks, self.cfg.code_vocab)

    def encode(self, x_embed: torch.Tensor) -> tuple:
        h = self.run(x_embed)
        pooled = self.pool(h)
        logits = self._reshape_pq(self.code_head(pooled))  # (...,pq_chunks,V)
        code_soft, code_idx = quantize_hard(logits)  # (...,pq_chunks,V), (...,pq_chunks)
        return code_soft, code_idx

    def ntp_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self._reshape_pq(self.ntp_head(h))

    def ntp_loss(self, x_embed: torch.Tensor, target_idx: torch.Tensor, cond: torch.Tensor = None,
                 y_embed: torch.Tensor = None) -> torch.Tensor:
        """target_idx: (...,pq_chunks) per-position PQ code ids. Each (position,chunk)
        pair is one classification instance for the cross-entropy (matches
        qcute_lagcodec's SimplexQuant.ntp_loss_acc chunking convention)."""
        if cond is not None:
            x_embed = x_embed + cond
        if y_embed is not None:
            x_embed = x_embed + y_embed.unsqueeze(1)  # broadcast class embedding into every position
        h = self.run(x_embed)
        logits = self.ntp_logits(h[:, :-1, :])  # (B,L-1,pq_chunks,V)
        return F.cross_entropy(logits.reshape(-1, self.cfg.code_vocab), target_idx[:, 1:].reshape(-1))


class ImageEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.r_embed = nn.Embedding(256, D)
        self.g_embed = nn.Embedding(256, D)
        self.b_embed = nn.Embedding(256, D)
        self.level0 = EncoderLevel(cfg, cfg.strides[0], has_ntp=False)
        self.code0_embed = nn.Embedding(cfg.code_vocab, D)
        self.level1 = EncoderLevel(cfg, cfg.strides[1], has_ntp=True)
        self.code1_embed = nn.Embedding(cfg.code_vocab, D)
        self.level2 = EncoderLevel(cfg, cfg.strides[2], has_ntp=True)
        self.level1_bos = nn.Parameter(torch.zeros(D))
        self.level2_bos = nn.Parameter(torch.zeros(D))

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, y_embed: torch.Tensor = None) -> dict:
        """Teacher-forced bottom-up pass (training): real pixels -> code0 -> code1 -> code2.
        y_embed (class conditioning) only matters for level1/level2's NTP heads -- those are
        the ones reused as generative priors in sample_codes(), so they're the ones that need
        to learn to use it; the encode()/code_head path just deterministically downsamples
        real data and has nothing to condition on."""
        B = r.shape[0]
        r, g, b = r.reshape(B, -1), g.reshape(B, -1), b.reshape(B, -1)  # row-major flatten
        x0 = self.r_embed(r) + self.g_embed(g) + self.b_embed(b)
        code0_soft, code0_idx = self.level0.encode(x0)

        x1 = code_embed(code0_soft, self.code0_embed)
        code1_soft, code1_idx = self.level1.encode(x1)
        cond1 = code_embed(code1_idx, self.code1_embed).repeat_interleave(self.cfg.strides[1], dim=1)
        ntp1 = self.level1.ntp_loss(x1, code0_idx, cond=cond1, y_embed=y_embed)

        x2 = code_embed(code1_soft, self.code1_embed)
        code2_soft, code2_idx = self.level2.encode(x2)
        ntp2 = self.level2.ntp_loss(x2, code1_idx, y_embed=y_embed)

        vocab = self.cfg.code_vocab
        util = dict(util0=codebook_utilization(code0_idx, vocab),
                    util1=codebook_utilization(code1_idx, vocab),
                    util2=codebook_utilization(code2_idx, vocab))
        return dict(code0_soft=code0_soft, code1_soft=code1_soft, code2_soft=code2_soft,
                    ntp_loss=ntp1 + ntp2, **util)

    @torch.no_grad()
    def sample_codes(self, B: int, device: torch.device, greedy: bool = True, use_cache: bool = True,
                      y_embed: torch.Tensor = None) -> tuple:
        """Top-down generative sampling: level2 free-runs unconditionally over the
        code1 alphabet (genuine AR bootstrap); level1 then free-runs over the code0
        alphabet conditioned on the just-sampled code1 (additive per-position cond,
        matching training's cond1); level0 has no NTP head -- the Decoder generates
        actual bytes conditioned on all three. KV-cached by default (O(T) per level);
        use_cache=False takes the O(T^2) recompute path, kept only as a correctness
        reference for --check_kv_cache."""
        cfg = self.cfg
        code0_len = cfg.img_size * cfg.img_size // cfg.strides[0]
        code1_len = code0_len // cfg.strides[1]

        def sample_from(logits):
            """logits: (B,pq_chunks,V) -> (B,pq_chunks) ids, one per PQ chunk."""
            if greedy:
                return logits.argmax(-1)
            probs = F.softmax(logits, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            return torch.multinomial(flat, 1).view(*probs.shape[:-1])

        # broadcast class conditioning into every AR step's input, matching ntp_loss's
        # every-position broadcast during training (so the NTP heads see it consistently)
        y_add = y_embed.unsqueeze(1) if y_embed is not None else 0.0
        pqc = cfg.pq_chunks

        if use_cache:
            caches = self.level2.new_caches()
            x_new = self.level2_bos.view(1, 1, -1).expand(B, 1, -1) + y_add
            code1_idx = torch.zeros(B, 0, pqc, dtype=torch.long, device=device)
            for t in range(code1_len):
                h = self.level2.step(x_new, t, caches)
                nxt = sample_from(self.level2.ntp_logits(h[:, 0, :]))  # (B,pqc)
                code1_idx = torch.cat([code1_idx, nxt.unsqueeze(1)], dim=1)
                x_new = code_embed(nxt, self.code1_embed).unsqueeze(1) + y_add

            caches = self.level1.new_caches()
            x_new = self.level1_bos.view(1, 1, -1).expand(B, 1, -1) + y_add
            code0_idx = torch.zeros(B, 0, pqc, dtype=torch.long, device=device)
            for t in range(code0_len):
                h = self.level1.step(x_new, t, caches)
                nxt = sample_from(self.level1.ntp_logits(h[:, 0, :]))
                code0_idx = torch.cat([code0_idx, nxt.unsqueeze(1)], dim=1)
                cond = code_embed(code1_idx[:, t // cfg.strides[1]], self.code1_embed).unsqueeze(1)
                x_new = code_embed(nxt, self.code0_embed).unsqueeze(1) + cond + y_add
        else:
            seq = self.level2_bos.expand(B, 1, -1) + y_add
            code1_idx = torch.zeros(B, 0, pqc, dtype=torch.long, device=device)
            for _ in range(code1_len):
                h = self.level2.run(seq)
                nxt = sample_from(self.level2.ntp_logits(h[:, -1, :]))
                code1_idx = torch.cat([code1_idx, nxt.unsqueeze(1)], dim=1)
                seq = torch.cat([seq, code_embed(nxt, self.code1_embed).unsqueeze(1) + y_add], dim=1)

            seq = self.level1_bos.expand(B, 1, -1) + y_add
            code0_idx = torch.zeros(B, 0, pqc, dtype=torch.long, device=device)
            for t in range(code0_len):
                h = self.level1.run(seq)
                nxt = sample_from(self.level1.ntp_logits(h[:, -1, :]))
                code0_idx = torch.cat([code0_idx, nxt.unsqueeze(1)], dim=1)
                cond = code_embed(code1_idx[:, t // cfg.strides[1]], self.code1_embed).unsqueeze(1)
                seq = torch.cat([seq, code_embed(nxt, self.code0_embed).unsqueeze(1) + cond + y_add], dim=1)

        x2 = code_embed(code1_idx, self.code1_embed)
        _, code2_idx = self.level2.encode(x2)
        return code0_idx, code1_idx, code2_idx


# ---------------------------------------------------------------------------
# Decoder: plain GPT-style causal LM, batched 32-ways over image columns
# ---------------------------------------------------------------------------

SLOT_L2, SLOT_L1, SLOT_L0, SLOT_R, SLOT_G, SLOT_B = range(6)
SLOT_L0_MTP, SLOT_RGB_MTP = 2, 3


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.dec_l2_embed = nn.Embedding(cfg.code_vocab, D)
        self.dec_l1_embed = nn.Embedding(cfg.code_vocab, D)
        self.dec_l0_embed = nn.Embedding(cfg.code_vocab, D)
        self.byte_embed = nn.Embedding(256, D)
        n_slots = 4 if cfg.decoder_mode == "mtp" else 6
        self.slot_embed = nn.Embedding(n_slots, D)
        self.bos_l2 = nn.Parameter(torch.zeros(D))
        self.bos_l1 = nn.Parameter(torch.zeros(D))
        self.bos_l0 = nn.Parameter(torch.zeros(D))
        self.col_mix = ColumnMixAttention(D, cfg.n_heads, cfg.col_group_size)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.rope_base, cfg.n_kv_heads) for _ in range(cfg.n_layers)])
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
        """code2/code1/code0: STE soft (B,L,V) during training or realized ids (B,L)
        during generation -- code_embed() normalizes either to (B,L,D). Shift by one row
        (row0 -> learned BOS) so row r's conditioning only ever depends on rows < r -- a
        genuine chain-rule NLL, not a reconstruction bound (row r's own encoder codes
        causally include row r itself). y_embed (class conditioning), when given, is
        broadcast-added to every row (not just row 0/BOS) so the signal doesn't dilute
        over the sequence -- same mechanism as the BOS injection, just data-dependent and
        applied everywhere instead of only where there's no real code yet."""
        cfg = self.cfg
        img = cfg.img_size
        l2e = code_embed(code2, self.dec_l2_embed)  # (B, img, D) -- one code2 per row
        l1e = code_embed(code1, self.dec_l1_embed).reshape(-1, img, code1.shape[1] // img, cfg.d_model)
        l0e = code_embed(code0, self.dec_l0_embed).reshape(-1, img, code0.shape[1] // img, cfg.d_model)
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
        """-> (l2e_col, l1e_col, l0e_col) each (B, row, col, D), lag-1-row applied."""
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

    def forward(self, code2, code1, code0, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor,
                y_embed: torch.Tensor = None) -> dict:
        """Teacher-forced training pass. r,g,b: (B,img,img) ground-truth bytes, [row,col]."""
        cfg = self.cfg
        img = cfg.img_size
        B, D = r.shape[0], cfg.d_model
        l2e_col, l1e_col, l0e_col = self._per_column_cond(code2, code1, code0, y_embed)  # (B,row,col,D)
        r_e, g_e, b_e = self.byte_embed(r), self.byte_embed(g), self.byte_embed(b)

        if cfg.decoder_mode == "mtp":
            rgb_e = r_e + g_e + b_e
            slots = torch.stack([l2e_col, l1e_col, l0e_col, rgb_e], dim=3)  # (B,row,col,4,D)
        else:
            slots = torch.stack([l2e_col, l1e_col, l0e_col, r_e, g_e, b_e], dim=3)  # (B,row,col,6,D)
        n_slots = slots.shape[3]
        slots = slots + self.slot_embed.weight.view(1, 1, 1, n_slots, D)

        x = slots.permute(0, 2, 1, 3, 4).reshape(B * img, img * n_slots, D)  # batch-fold columns
        h = self.run(x)
        h = h.view(B, img, img, n_slots, D).permute(0, 2, 1, 3, 4)  # -> (B,row,col,slot,D)

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

    @torch.no_grad()
    def generate(self, code2, code1, code0, greedy: bool = True, use_cache: bool = True,
                 y_embed: torch.Tensor = None) -> tuple:
        """Column-batched AR sampling, row by row down each column. KV-cached by
        default (O(T) per column); use_cache=False takes the O(T^2) full-recompute
        path (generate_nocache), kept only as a correctness reference."""
        if not use_cache:
            return self.generate_nocache(code2, code1, code0, greedy=greedy, y_embed=y_embed)

        cfg = self.cfg
        img = cfg.img_size
        device = code2.device if hasattr(code2, "device") else code0.device
        l2e_col, l1e_col, l0e_col = self._per_column_cond(code2, code1, code0, y_embed)  # (B,row,col,D)
        B, _, _, D = l2e_col.shape
        slot_w = self.slot_embed.weight

        def sample_from(logits):
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

        l2e = l2e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)
        l1e = l1e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)
        l0e = l0e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)

        Bc = B * img
        r_out = torch.zeros(Bc, img, dtype=torch.long, device=device)
        g_out = torch.zeros(Bc, img, dtype=torch.long, device=device)
        b_out = torch.zeros(Bc, img, dtype=torch.long, device=device)

        caches = self.new_caches()
        pos = 0
        for row in range(img):
            self.step((l2e[:, row] + slot_w[SLOT_L2]).unsqueeze(1), pos, caches); pos += 1
            self.step((l1e[:, row] + slot_w[SLOT_L1]).unsqueeze(1), pos, caches); pos += 1
            l0_slot = slot_w[SLOT_L0_MTP if cfg.decoder_mode == "mtp" else SLOT_L0]
            h_seed = self.step((l0e[:, row] + l0_slot).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1

            if cfg.decoder_mode == "mtp":
                r = sample_from(self.head_r(h_seed))
                g = sample_from(self.head_g(h_seed))
                b = sample_from(self.head_b(h_seed))
                rgb_e = (self.byte_embed(r) + self.byte_embed(g) + self.byte_embed(b) + slot_w[SLOT_RGB_MTP]).unsqueeze(1)
                self.step(rgb_e, pos, caches); pos += 1
            else:
                r = sample_from(self.head_r(h_seed))
                h_r = self.step((self.byte_embed(r) + slot_w[SLOT_R]).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1
                g = sample_from(self.head_g(h_r))
                h_g = self.step((self.byte_embed(g) + slot_w[SLOT_G]).unsqueeze(1), pos, caches)[:, 0, :]; pos += 1
                b = sample_from(self.head_b(h_g))
                self.step((self.byte_embed(b) + slot_w[SLOT_B]).unsqueeze(1), pos, caches); pos += 1
            r_out[:, row], g_out[:, row], b_out[:, row] = r, g, b

        def unfold(x):
            return x.view(B, img, img).permute(0, 2, 1)  # (Bc,row) -> (B,col,row) -> (B,row,col)

        return unfold(r_out), unfold(g_out), unfold(b_out)

    @torch.no_grad()
    def generate_nocache(self, code2, code1, code0, greedy: bool = True, y_embed: torch.Tensor = None) -> tuple:
        """Naive (no KV cache) column-batched AR sampling -- O(T^2) full recompute
        every step, kept only as a correctness reference for --check_kv_cache."""
        cfg = self.cfg
        img = cfg.img_size
        device = code2.device if hasattr(code2, "device") else code0.device
        l2e_col, l1e_col, l0e_col = self._per_column_cond(code2, code1, code0, y_embed)  # (B,row,col,D)
        B, _, _, D = l2e_col.shape
        slot_w = self.slot_embed.weight

        def sample_from(logits):
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

        # fold columns into batch: (B*img, ...)
        l2e = l2e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)
        l1e = l1e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)
        l0e = l0e_col.permute(0, 2, 1, 3).reshape(B * img, img, D)

        Bc = B * img
        r_out = torch.zeros(Bc, img, dtype=torch.long, device=device)
        g_out = torch.zeros(Bc, img, dtype=torch.long, device=device)
        b_out = torch.zeros(Bc, img, dtype=torch.long, device=device)
        seq = torch.zeros(Bc, 0, D, device=device)

        for row in range(img):
            base = torch.stack([l2e[:, row], l1e[:, row], l0e[:, row]], dim=1)  # (Bc,3,D)
            base = base + slot_w[:3].unsqueeze(0)
            seq = torch.cat([seq, base], dim=1)
            h = self.run(seq)
            if cfg.decoder_mode == "mtp":
                h_seed = h[:, -1, :]
                r = sample_from(self.head_r(h_seed))
                g = sample_from(self.head_g(h_seed))
                b = sample_from(self.head_b(h_seed))
                rgb_e = (self.byte_embed(r) + self.byte_embed(g) + self.byte_embed(b) + slot_w[3]).unsqueeze(1)
                seq = torch.cat([seq, rgb_e], dim=1)
            else:
                h_seed = h[:, -1, :]
                r = sample_from(self.head_r(h_seed))
                seq = torch.cat([seq, (self.byte_embed(r) + slot_w[3]).unsqueeze(1)], dim=1)
                h = self.run(seq)
                g = sample_from(self.head_g(h[:, -1, :]))
                seq = torch.cat([seq, (self.byte_embed(g) + slot_w[4]).unsqueeze(1)], dim=1)
                h = self.run(seq)
                b = sample_from(self.head_b(h[:, -1, :]))
                seq = torch.cat([seq, (self.byte_embed(b) + slot_w[5]).unsqueeze(1)], dim=1)
            r_out[:, row], g_out[:, row], b_out[:, row] = r, g, b

        def unfold(x):
            return x.view(B, img, img).permute(0, 2, 1)  # (Bc,row) -> (B,col,row) -> (B,row,col)

        return unfold(r_out), unfold(g_out), unfold(b_out)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ImageGenCIFAR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.decoder = Decoder(cfg)
        if cfg.class_conditional:
            self.class_embed = nn.Embedding(cfg.n_classes, cfg.d_model)

    def _y_embed(self, y: torch.Tensor) -> torch.Tensor:
        return self.class_embed(y) if self.cfg.class_conditional else None

    def forward(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, y: torch.Tensor = None) -> dict:
        y_embed = self._y_embed(y)
        enc = self.encoder(r, g, b, y_embed=y_embed)
        dec = self.decoder(enc["code2_soft"], enc["code1_soft"], enc["code0_soft"], r, g, b, y_embed=y_embed)
        total = dec["loss"] + self.cfg.ntp_aux_weight * enc["ntp_loss"]
        bpb = dec["loss"] / math.log(2)
        return dict(loss=total, byte_loss=dec["loss"], bpb=bpb, acc=dec["acc"], ntp_loss=enc["ntp_loss"],
                    util0=enc["util0"], util1=enc["util1"], util2=enc["util2"])

    @torch.no_grad()
    def generate(self, n: int, device: torch.device, greedy: bool = True, use_cache: bool = True,
                 y: "torch.Tensor | int | None" = None) -> torch.Tensor:
        """y: class id(s) when cfg.class_conditional -- an int (same class for all n samples),
        a (n,) tensor, or None (samples a random class per image). Ignored otherwise."""
        if self.cfg.class_conditional:
            if y is None:
                y = torch.randint(0, self.cfg.n_classes, (n,), device=device)
            elif isinstance(y, int):
                y = torch.full((n,), y, dtype=torch.long, device=device)
            else:
                y = y.to(device)
        y_embed = self._y_embed(y)
        code0_idx, code1_idx, code2_idx = self.encoder.sample_codes(n, device, greedy=greedy, use_cache=use_cache,
                                                                      y_embed=y_embed)
        r, g, b = self.decoder.generate(code2_idx, code1_idx, code0_idx, greedy=greedy, use_cache=use_cache,
                                         y_embed=y_embed)
        return torch.stack([r, g, b], dim=-1).clamp(0, 255).to(torch.uint8)  # (n,32,32,3)


def check_kv_cache_consistency(model: "ImageGenCIFAR", device: torch.device, n: int = 2) -> bool:
    """Greedy-decode with and without the KV cache and require byte-exact agreement --
    the two paths compute the identical causal function, so under greedy argmax they
    must select the same token at every step (mirrors summformer_jax's
    check_kv_cache_consistency convention, see CLAUDE.md)."""
    torch.manual_seed(0)
    cached = model.generate(n, device, greedy=True, use_cache=True)
    torch.manual_seed(0)
    nocache = model.generate(n, device, greedy=True, use_cache=False)
    match = (cached == nocache).float().mean().item()
    ok = match == 1.0
    return ok, match


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
        # proxy the real terminal's isatty so tqdm still renders in-place (\r) instead
        # of spamming a newline per update once stdout/stderr are wrapped
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
    """Load a Python config file as a dict of module-level variables. Values must
    already be the right type (tuples, ints, ...) -- argparse's `type=` conversion
    only applies to strings passed on the actual command line, not to defaults."""
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


CONFIG_FIELDS = ("img_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "code_vocab", "pq_chunks", "strides",
                  "code_extract_mode", "decoder_mode", "rope_base", "mlp_mult",
                  "ntp_aux_weight", "col_group_size",
                  "class_conditional", "n_classes")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None,
                      help="Python config file (image_gen_cifar/configs/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_lagcodec_minimal")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=5, help="run val eval + qual-gen samples every N epochs")
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--check_kv_cache", action="store_true",
                    help="verify KV-cached generation is byte-exact vs. the no-cache reference, then exit")
    # Config dataclass fields, individually overridable from CLI or --config:
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    p.add_argument("--n_kv_heads", type=int, default=Config.n_kv_heads,
                    help="GQA kv-head count (None = max(1, n_heads//4)); == n_heads for plain MHA")
    p.add_argument("--code_vocab", type=int, default=Config.code_vocab)
    p.add_argument("--pq_chunks", type=int, default=Config.pq_chunks)
    p.add_argument("--strides", type=lambda s: tuple(int(x) for x in s.split(",")), default=Config.strides)
    p.add_argument("--code_extract_mode", type=str, default=Config.code_extract_mode, choices=["mean", "last_idx"])
    p.add_argument("--decoder_mode", type=str, default=Config.decoder_mode, choices=["seq", "mtp"])
    p.add_argument("--rope_base", type=float, default=Config.rope_base)
    p.add_argument("--mlp_mult", type=int, default=Config.mlp_mult)
    p.add_argument("--ntp_aux_weight", type=float, default=Config.ntp_aux_weight)
    p.add_argument("--col_group_size", type=int, default=Config.col_group_size,
                    help="decoder column-track communication: 1=SISO (default), img_size=MIMO, else grouped")
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
    model = ImageGenCIFAR(cfg).to(device)
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

    if args.check_kv_cache:
        model.eval()
        ok, match = check_kv_cache_consistency(model, device)
        logger(f"check_kv_cache: {'PASS' if ok else 'FAIL'} (byte match rate={match:.6f})",
               check_kv_cache_pass=ok, check_kv_cache_match=match)
        return

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
        samples = model.generate(args.qual_gen_n, device, greedy=True, use_cache=True)
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
                logger(f"epoch={epoch} step={step} bpb={out['bpb'].item():.4f} acc={out['acc'].item():.4f} "
                       f"ntp_loss={out['ntp_loss'].item():.4f} util(l0/l1/l2)="
                       f"{out['util0'].item():.2f}/{out['util1'].item():.2f}/{out['util2'].item():.2f}",
                       epoch=epoch, step=step, train_bpb=out["bpb"].item(), train_acc=out["acc"].item(),
                       util0=out["util0"].item(), util1=out["util1"].item(), util2=out["util2"].item())
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            val_bpb = run_eval(val_loader, "val")
            ckpt.step({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "step": step}, val_bpb)
            run_qual_gen(epoch)
    logger("training done")


if __name__ == "__main__":
    main()
