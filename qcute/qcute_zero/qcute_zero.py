"""qcute_zero: a monolithic, single-LM alternative to qcute_v1's multi-encoder StackDecoder
lineage. One transformer LM (level0, byte space); every K bytes it summarizes its own hidden state
into a discrete code (STE hard sample via the tied embed/head), runs that code sequence through
the SAME shared blocks for a genuine code-NTP loss + contextualized K/V, then cross-attends that
K/V back into the byte-level query stream. Repeats per entry in `Ks` (a real cascade -- each
stage's codes are built from the PREVIOUS stage's own contextualized hidden state). Causality:
every code's boundary is its cumulative byte-span, never a local code-sequence index. Zero-KV sink
on every attention call (self- and cross-) keeps early/pre-boundary queries NaN-free with a
provably-zero no-op contribution. `Config.mtp_heads` + `generate_speculative` implement MTP-style
speculative decoding, verified exact against `generate_kv_cache`'s incremental stepper. Simplified
2026-08-25: every parallel-decode-strategy experiment (wavefront, blocklocal/GLAT, free rollout,
early exit, seed_query) pruned -- see backups/qcute_zero_parallel_attempt1.py for that lineage,
docs/status.md for the full history. Single file by design, primitives adapted from
qcute_v1_common.py rather than imported.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k.py
uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks221_overfit10k.py
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


# small shared utilities (copied/trimmed from qcute_v1_common.py)

def make_dict(**kwargs) -> dict:
    return kwargs


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
        if not math.isfinite(metric) or metric <= 0:
            return False
        return metric < self.best_metric if self.minimize else metric > self.best_metric

    def step(self, state: dict, metric: float) -> None:
        self._eval_count += 1
        if self.is_better(metric):
            self.best_metric = metric
            torch.save(state, self.best_path)
        if self._eval_count % self.save_every_n_evals == 0:
            torch.save(state, self.last_path)


def unpack_words(data: bytes, bits: int) -> list:
    if bits == 8:
        return list(data)
    words = []
    mask = (1 << bits) - 1
    for byte in data:
        for shift in range(8 - bits, -1, -bits):
            words += [(byte >> shift) & mask]
    return words


def pack_words(words: list, bits: int) -> bytes:
    if bits == 8:
        return bytes(words)
    words_per_byte = 8 // bits
    out = bytearray()
    for i in range(0, len(words) - len(words) % words_per_byte, words_per_byte):
        b = 0
        for j in range(words_per_byte):
            b = (b << bits) | words[i + j]
        out.append(b)
    return bytes(out)


def load_enwik8(path: Path, bits: int, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(unpack_words(data, bits), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple:
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


def load_config_module(path: Path) -> dict:
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


# RoPE + attention primitives

def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    """position_ids: (T,) shared across the batch, or (Bv, T) one row per batch element."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq  # (..., T, hd/2), generalizes torch.outer
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """cos/sin: (T, hd) shared across batch (broadcasts via [None, None]), or (Bv, T, hd)
    per-batch-row positions (broadcasts via [:, None] over the head dim only)."""
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


def sdpa_with_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Zero-value/zero-key sink prepended to every attention call -- guarantees >=1 valid key per
    query row, provably-zero contribution when it's the only one visible. attn_mask: bool, True=visible."""
    B, H, T, hd = q.shape
    sink_k = k.new_zeros(B, H, 1, hd)
    sink_v = v.new_zeros(B, H, 1, hd)
    k2 = torch.cat([sink_k, k], dim=2)
    v2 = torch.cat([sink_v, v], dim=2)
    sink_col = attn_mask.new_ones(attn_mask.shape[:-1] + (1,))
    mask2 = torch.cat([sink_col, attn_mask], dim=-1)
    return F.scaled_dot_product_attention(q, k2, v2, attn_mask=mask2)


def causal_mask(query_pos: torch.Tensor, key_pos: torch.Tensor, window: int | None) -> torch.Tensor:
    """(1, 1, T, S) bool mask, True=visible. window=None: unbounded causal (key_pos<=query_pos).
    window: also require (query_pos - key_pos) < window."""
    allow = key_pos.view(1, -1) <= query_pos.view(-1, 1)
    if window is not None:
        allow = allow & ((query_pos.view(-1, 1) - key_pos.view(1, -1)) < window)
    return allow.view(1, 1, *allow.shape)


def _fmt_bytes(t: torch.Tensor) -> str:
    """1D byte-id tensor -> printable latin-1 string, for verbose generation logging."""
    return bytes(int(b) & 0xFF for b in t.tolist()).decode("latin-1", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Attn(nn.Module):
    """Shared QKV/out projections for self- (forward) and cross-attention (forward_cross)."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        """x_new: only the NEW position(s). cache: None or (k_prev, v_prev). Mask uses LOCAL
        (call-relative) positions -- cos/sin (true absolute positions) is what encodes real distance."""
        B, Tn, D = x_new.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x_new).reshape(B, Tn, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        if cache is None:
            k_all, v_all, S_prev = k, v, 0
        else:
            k_prev, v_prev = cache
            k_all, v_all = torch.cat([k_prev, k], dim=2), torch.cat([v_prev, v], dim=2)
            S_prev = k_prev.shape[2]
        S = k_all.shape[2]
        new_pos = torch.arange(S_prev, S_prev + Tn, device=x_new.device)
        key_pos = torch.arange(S, device=x_new.device)
        mask = causal_mask(new_pos, key_pos, window)
        y = sdpa_with_sink(q, k_all, v_all, mask)
        out = self.out(y.transpose(1, 2).reshape(B, Tn, D))
        if window is not None and S > window:
            k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
        return out, (k_all, v_all)

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
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    """gate/up/down, no bias: down(silu(gate(x)) * up(x)). Replaces the plain Linear-GELU-Linear
    MLP everywhere in this file (Block and FuseStage both)."""
    def __init__(self, d_model: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * d_model
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Self-attention + MLP, shared across the byte-level pass and every fuse stage's own
    code-sequence NTP pass -- this IS the "single LM" the whole design hinges on."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        attn_out, new_cache = self.attn.forward_incremental(self.ln1(x_new), cos_new, sin_new, cache, window)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_cache


class FuseStage(nn.Module):
    """Cross-attention + MLP, one instance per fuse stage, own weights throughout. Cheap: called
    with the code sequence's length (L/cum_K), not the byte sequence's."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int):
        super().__init__()
        self.ln1 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.ModuleList([Attn(d_model, n_heads) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([SwiGLU(d_model, mlp_mult) for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, code_kv: torch.Tensor, cos_q, sin_q, cos_k, sin_k,
                attn_mask: torch.Tensor) -> torch.Tensor:
        for l in range(len(self.attn)):
            xn = self.ln1[l](x)
            coden = self.ln1[l](code_kv)
            x = x + self.attn[l].forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask)
            x = x + self.mlp[l](self.ln2[l](x))
        return x

    def readout(self, x: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        return F.linear(self.ln_out(x), embed_weight)


def gumbel_quantize(logits: torch.Tensor, tau: float, hard: bool = True, sample: bool = False) -> torch.Tensor:
    if sample:
        eps = torch.finfo(logits.dtype).tiny
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(u))
        soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    else:
        soft = F.softmax(logits / tau, dim=-1)
    if not hard:
        return soft
    hard_oh = F.one_hot(soft.argmax(-1), num_classes=logits.shape[-1]).to(soft.dtype)
    return soft + (hard_oh - soft).detach()


class Quantizer(nn.Module):
    """Pluggable per-fuse-stage code: categorical via gumbel-softmax STE, product-quantized.
    global_tie ties to the shared byte embed/head only when vocab==unit_vocab and pq_chunks==1."""
    def __init__(self, D: int, vocab: int, pq_chunks: int, gumbel_tau: float, code_hard: bool, code_sample: bool,
                 unit_vocab: int = 256, global_tie: bool = False,
                 tied_head_weight: torch.Tensor | None = None, tied_embed_weight: torch.Tensor | None = None):
        super().__init__()
        assert vocab ** pq_chunks >= unit_vocab, (
            f"quantizer code space too small: vocab**pq_chunks = {vocab}**{pq_chunks} = "
            f"{vocab ** pq_chunks} < unit_vocab ({unit_vocab}) -- must represent at least one real "
            f"trunk unit (byte/nibble/bit/etc., per input_preset)")
        self.vocab = vocab
        self.pq_chunks = pq_chunks
        self.width = vocab * pq_chunks
        self.unit_vocab = unit_vocab
        self.tau = gumbel_tau
        self.hard = code_hard
        self.sample = code_sample

        self.exact_tie = global_tie and vocab == unit_vocab and pq_chunks == 1
        if self.exact_tie:
            assert tied_head_weight is not None and tied_embed_weight is not None
            self.code_head = nn.Linear(D, self.width, bias=False)
            self.code_head.weight = tied_head_weight       # (width, D) == (unit_vocab, D), same shape as self.head
            self.code_predict = nn.Linear(D, self.width, bias=False)
            self.code_predict.weight = tied_head_weight
            self.code_embed = None                          # embedding done via direct matmul, see _embed()
            self._tied_embed_weight = tied_embed_weight      # (unit_vocab, D), same convention as self.embed.weight
        else:
            self.code_head = nn.Linear(D, self.width, bias=False)
            nn.init.normal_(self.code_head.weight, std=0.02)
            self.code_embed = nn.Linear(self.width, D, bias=False)
            nn.init.normal_(self.code_embed.weight, std=0.02)
            self.code_predict = nn.Linear(D, self.width, bias=False)
            nn.init.normal_(self.code_predict.weight, std=0.02)

    def _embed(self, onehot: torch.Tensor) -> torch.Tensor:
        if self.exact_tie:
            return onehot @ self._tied_embed_weight
        return self.code_embed(onehot)

    def _chunked(self, x: torch.Tensor) -> torch.Tensor:
        """(..., width) -> (..., pq_chunks, vocab) -- pure reshape, no data movement."""
        return x.reshape(*x.shape[:-1], self.pq_chunks, self.vocab)

    def extract(self, h: torch.Tensor) -> tuple:
        """h: (..., D) hidden state at a code boundary -> (code_repr (..., width) STE one-hot(s)
        concatenated, code_ids (...) combined integer id, code_embeds (..., D))."""
        logits = self.code_head(h)
        if self.pq_chunks == 1:
            onehot = gumbel_quantize(logits, self.tau, self.hard, self.sample)
        else:
            onehot = gumbel_quantize(self._chunked(logits), self.tau, self.hard, self.sample).reshape(logits.shape)
        return onehot, self.to_ids(onehot), self._embed(onehot)

    def extract_greedy(self, h: torch.Tensor) -> tuple:
        """Same as extract() but always hard=True/sample=False -- generation-time greedy extraction."""
        logits = self.code_head(h)
        if self.pq_chunks == 1:
            onehot = gumbel_quantize(logits, self.tau, hard=True, sample=False)
        else:
            onehot = gumbel_quantize(self._chunked(logits), self.tau, hard=True, sample=False).reshape(logits.shape)
        return onehot, self._embed(onehot)

    def to_ids(self, code_repr: torch.Tensor) -> torch.Tensor:
        chunk_ids = self._chunked(code_repr).argmax(-1)
        if self.pq_chunks == 1:
            return chunk_ids[..., 0]
        weights = self.vocab ** torch.arange(self.pq_chunks, device=code_repr.device)
        return (chunk_ids * weights).sum(-1)

    def ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor) -> tuple:
        """h_query: (..., D) code-sequence LM hidden state; target_repr: (..., width) the NEXT
        code's own one-hot representation (from extract())."""
        logits = self._chunked(self.code_predict(h_query)).reshape(-1, self.vocab)
        target = self._chunked(target_repr).argmax(-1).reshape(-1)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc


# Config + model

@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32, 1)       # same semantics as qcute_v1: cumulative periods, last
                                              # entry conventionally 1 (no further fuse stage after it)
    d_model: int = 256
    n_layers: int = 4                        # scalar -- shared "block regular", reused for every
                                              # fuse stage's own code-sequence NTP pass too
    fuse_n_layers: int | None = None         # defaults to n_layers if unset
    n_heads: int = 4
    mlp_mult: int = 4
    rope_base: float = 10000.0
    context_len: int = 256
    attn_window: int | None = None           # main byte self-attention window (None = unbounded)
    fuse_window: int | tuple | None = None   # per-fuse-stage cross-attn window, in BYTES; None/scalar/tuple
    input_preset: int = 8                    # byte alphabet bits -- vocab = 2**input_preset, shared
                                              # by codes (same embed/output head)
    gumbel_tau: float = 1.0
    code_hard: bool = True
    code_sample: bool = False
    quant_type: str = "simplex"              # only "simplex" ported so far (see Quantizer)
    vocab: int = 256                         # per-chunk code vocab size (Quantizer)
    pq_chunks: int = 1                       # product-quantization chunks; vocab**pq_chunks>=256
                                              # required (must represent at least a byte); default
                                              # 256/1 is functionally the old tied-embed code
    code_ntp_weight: float = 1.0             # weight for each fuse stage's own code-sequence NTP loss
    cond_weight: float = 1.0                 # weight for each stage's post-fusion byte NTP loss
    mtp_heads: int = 1                       # extra byte-ahead heads off the final hidden state
                                              # (1=disabled); generate_speculative drafts via these.
    mtp_weight: float = 1.0
    mtp_heads_code: int = 1                  # extra code-ahead heads off h_code (1=disabled)
    mtp_weight_code: float = 1.0
    mtp_heads_uncond: int = 1                # extra byte-ahead heads off pre-fusion h (1=disabled)
    mtp_weight_uncond: float = 1.0
    weight_tie: bool = False                 # True: head.weight literally refs embed.weight
    global_tie: bool = False                 # requires weight_tie=True; extends the tie to every
                                              # level's Quantizer (exact only if vocab==unit_vocab, pq_chunks==1)
    share_lm: bool = False                   # True ties every level to the same Block stack
    share_fuse: bool = False                 # True ties every fuse stage to fuse_stages[0]
    head_word_bits: int | None = None        # None = same as input_preset; see WordHead


class WordHead(nn.Module):
    """Output-head granularity decoupled from the trunk's word size: word_bits>unit_bits predicts
    multiple future positions jointly; word_bits<unit_bits factorizes one position into PQ-style sub-chunks."""
    def __init__(self, D: int, unit_bits: int, word_bits: int | None):
        super().__init__()
        self.unit_bits = unit_bits
        self.word_bits = word_bits if word_bits is not None else unit_bits
        assert self.word_bits % unit_bits == 0 or unit_bits % self.word_bits == 0, (
            f"head_word_bits ({self.word_bits}) must evenly divide, or be an even multiple of, "
            f"input_preset ({unit_bits})")
        self.unit_vocab = 2 ** unit_bits
        self.sub_vocab = 2 ** self.word_bits     # per-chunk width in BOTH cases (see class docstring)
        self.group_size = max(1, self.word_bits // unit_bits)     # >1: coarser (joint multi-position)
        self.n_sub = max(1, unit_bits // self.word_bits)          # >1: finer (PQ-style sub-word split)
        self.proj = nn.Linear(D, self.sub_vocab * self.n_sub, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def _chunked(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.n_sub, self.sub_vocab)

    def _split_unit_into_subchunks(self, unit_id: torch.Tensor) -> torch.Tensor:
        """Finer case only: (...,) one unit id in [0, unit_vocab) -> (..., n_sub) base-sub_vocab
        digits (place-value decomposition, inverse of combining n_sub sub-chunk argmaxes)."""
        ids = []
        rem = unit_id
        for _ in range(self.n_sub):
            ids.append(rem % self.sub_vocab)
            rem = rem // self.sub_vocab
        return torch.stack(ids, dim=-1)

    def combine_ids(self, unit_ids_grouped: torch.Tensor) -> torch.Tensor:
        """Coarser case only: (..., group_size) consecutive unit ids -> one combined place-value
        integer in [0, unit_vocab**group_size) == [0, sub_vocab)."""
        weights = self.unit_vocab ** torch.arange(self.group_size, device=unit_ids_grouped.device)
        return (unit_ids_grouped * weights).sum(-1)

    def split_id(self, combined_id: torch.Tensor) -> torch.Tensor:
        """Coarser case only: inverse of combine_ids -- (...,) combined integer -> (...,
        group_size) unit ids."""
        ids = []
        rem = combined_id
        for _ in range(self.group_size):
            ids.append(rem % self.unit_vocab)
            rem = rem // self.unit_vocab
        return torch.stack(ids, dim=-1)

    def loss_acc(self, h: torch.Tensor, unit_ids: torch.Tensor):
        """h: (B, L, D); unit_ids: (B, L) trunk-granularity ids (e.g. byte_ids). Returns (loss,
        acc), or (None, None) if the sequence is too short for even one prediction."""
        L = h.shape[1]
        if self.n_sub > 1:
            if L < 2:
                return None, None
            logits = self._chunked(self.proj(h[:, :-1, :]))               # (B, L-1, n_sub, sub_vocab)
            target = self._split_unit_into_subchunks(unit_ids[:, 1:])     # (B, L-1, n_sub)
            loss = F.cross_entropy(logits.reshape(-1, self.sub_vocab), target.reshape(-1))
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
            return loss, acc
        n_pos = L - self.group_size
        if n_pos < 1:
            return None, None
        logits = self.proj(h[:, :n_pos, :])
        windows = torch.stack([unit_ids[:, i + 1:i + 1 + n_pos] for i in range(self.group_size)], dim=-1)
        target = self.combine_ids(windows)
        loss = F.cross_entropy(logits.reshape(-1, self.sub_vocab), target.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def sample(self, h_last: torch.Tensor) -> torch.Tensor:
        """h_last: (B, D) or (B, 1, D) -> (B, group_size) next unit ids."""
        squeeze = h_last.dim() == 2
        h_last = h_last.unsqueeze(1) if squeeze else h_last
        logits = self.proj(h_last)                                        # (B, 1, sub_vocab*n_sub)
        if self.n_sub > 1:
            chunk_ids = self._chunked(logits).argmax(-1)                  # (B, 1, n_sub)
            weights = self.sub_vocab ** torch.arange(self.n_sub, device=logits.device)
            out = (chunk_ids * weights).sum(-1, keepdim=True)              # (B, 1) single unit id
        else:
            combined_id = logits.argmax(-1)                                # (B, 1)
            out = self.split_id(combined_id) if self.group_size > 1 else combined_id.unsqueeze(-1)
        return out.squeeze(1) if squeeze else out


def resolve_fuse_window(w, n_fuse: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_fuse
        return tuple(w)
    return (w,) * n_fuse


class QCuteZero(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        V = 2 ** cfg.input_preset
        self.vocab = V
        self.n_fuse = len(cfg.Ks) - 1
        assert D % cfg.n_heads == 0

        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)

        # level 0 = byte pass + refinement; level s+1 = fuse stage s's code-sequence NTP pass.
        n_lms = self.n_fuse + 1
        if cfg.share_lm:
            first = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
            self.lms = nn.ModuleList([first] * n_lms)
        else:
            self.lms = nn.ModuleList(
                [nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
                 for _ in range(n_lms)])
        if cfg.share_lm:
            first_ln = RMSNorm(D)
            self.ln_fs = nn.ModuleList([first_ln] * n_lms)
        else:
            self.ln_fs = nn.ModuleList([RMSNorm(D) for _ in range(n_lms)])

        assert not cfg.global_tie or cfg.weight_tie, "global_tie requires weight_tie=True"
        self.head = nn.Linear(D, V, bias=False)
        if cfg.weight_tie:
            assert cfg.head_word_bits is None or cfg.head_word_bits == cfg.input_preset, (
                "weight_tie=True ties self.head to self.embed, which only makes sense when the "
                "output head predicts the SAME format the input embeds (head_word_bits must match "
                f"input_preset, e.g. byte-in/byte-out) -- got input_preset={cfg.input_preset}, "
                f"head_word_bits={cfg.head_word_bits}")
            self.head.weight = self.embed.weight
        else:
            nn.init.normal_(self.head.weight, std=0.02)

        fuse_layers = cfg.fuse_n_layers if cfg.fuse_n_layers is not None else cfg.n_layers
        if cfg.share_fuse:
            first_fs = FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers)
            self.fuse_stages = nn.ModuleList([first_fs] * self.n_fuse)
        else:
            self.fuse_stages = nn.ModuleList(
                [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        self.extra_heads = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads - 1))])
        self.extra_heads_uncond = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads_uncond - 1))])

        assert cfg.quant_type == "simplex", f"only quant_type='simplex' is ported so far, got {cfg.quant_type!r}"

        def _make_quantizer():
            return Quantizer(D, cfg.vocab, cfg.pq_chunks, cfg.gumbel_tau, cfg.code_hard, cfg.code_sample,
                              unit_vocab=V, global_tie=cfg.global_tie,
                              tied_head_weight=self.head.weight, tied_embed_weight=self.embed.weight)

        if cfg.share_lm:
            first_q = _make_quantizer()
            self.quantizers = nn.ModuleList([first_q] * max(1, self.n_fuse))
        else:
            self.quantizers = nn.ModuleList([_make_quantizer() for _ in range(max(1, self.n_fuse))])
        self.quantizer = self.quantizers[0]   # convenience alias, e.g. for .width below

        if cfg.share_lm:
            first_ehc = nn.ModuleList(
                [nn.Linear(D, self.quantizer.width, bias=False) for _ in range(max(0, cfg.mtp_heads_code - 1))])
            self.extra_heads_code_per_stage = nn.ModuleList([first_ehc] * max(1, self.n_fuse))
        else:
            self.extra_heads_code_per_stage = nn.ModuleList([nn.ModuleList(
                [nn.Linear(D, self.quantizer.width, bias=False) for _ in range(max(0, cfg.mtp_heads_code - 1))])
                for _ in range(max(1, self.n_fuse))])

        self.word_head = WordHead(D, cfg.input_preset, cfg.head_word_bits) if cfg.head_word_bits is not None else None

    def _run_blocks(self, level: int, x: torch.Tensor, cos, sin, attn_mask) -> torch.Tensor:
        for block in self.lms[level]:
            x = block(x, cos, sin, attn_mask)
        return self.ln_fs[level](x)

    def forward(self, byte_ids: torch.Tensor) -> tuple:
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = D // cfg.n_heads
        device = byte_ids.device
        V = self.vocab

        # --- byte-level pass ("block regular"), uncond ---
        byte_pos = torch.arange(L, device=device)
        cos_b, sin_b = rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base, device)
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        h = self._run_blocks(0, x0, cos_b, sin_b, byte_mask)

        if self.word_head is not None:
            uncond_loss, uncond_acc = self.word_head.loss_acc(h, byte_ids)
            uncond_loss = uncond_loss if uncond_loss is not None else h.new_zeros(())
            uncond_acc = uncond_acc if uncond_acc is not None else h.new_zeros(())
        else:
            uncond_logits = self.head(h[:, :-1, :])
            uncond_loss = F.cross_entropy(uncond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
            uncond_acc = (uncond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()

        # cheap/coarse extra byte-ahead heads off the pre-fusion hidden state h.
        uncond_mtp_losses, uncond_mtp_accs = [], []
        for i, head_u in enumerate(self.extra_heads_uncond):
            k = i + 2
            if L <= k:
                continue
            logits_u = head_u(h[:, :-k, :])
            targets_u = byte_ids[:, k:]
            uncond_mtp_losses.append(F.cross_entropy(logits_u.reshape(-1, V), targets_u.reshape(-1)))
            uncond_mtp_accs.append((logits_u.argmax(-1) == targets_u).float().mean())

        # --- cascade through fuse stages ---
        cur_h = h                # source hidden states to extract this stage's codes from
        x_cross = h              # running byte-level query stream, refined by each fuse stage
        cum_K = 1
        fuse_ntp_losses, fuse_ntp_accs = [], []
        cond_losses, cond_accs = [], []
        code_mtp_losses, code_mtp_accs = {}, {}   # keyed by (stage, k)

        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break

            quantizer = self.quantizers[s]
            # code extraction: pluggable per-stage Quantizer (own code_head/code_embed, STE hard sample)
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]
            onehot, code_ids, code_embeds = quantizer.extract(code_h)

            # this stage's own code-sequence NTP pass -- SAME shared blocks, causal, unbounded
            # (short sequence: n_blocks = cur_len // K_s)
            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(s + 1, code_embeds, cos_c, sin_c, code_mask)

            if n_blocks >= 2:
                code_ntp_loss, code_ntp_acc = quantizer.ntp_loss_acc(h_code[:, :-1, :], onehot[:, 1:, :])
                fuse_ntp_losses += [code_ntp_loss]
                fuse_ntp_accs += [code_ntp_acc]

            # extra code-ahead heads off h_code, this stage's own set.
            for i, head_c in enumerate(self.extra_heads_code_per_stage[s]):
                k = i + 2
                if n_blocks <= k:
                    continue
                logits_c = quantizer._chunked(head_c(h_code[:, :-k, :])).reshape(-1, quantizer.vocab)
                target_c = quantizer._chunked(onehot[:, k:, :]).argmax(-1).reshape(-1)
                code_mtp_losses[(s, k)] = F.cross_entropy(logits_c, target_c)
                code_mtp_accs[(s, k)] = (logits_c.argmax(-1) == target_c).float().mean()

            # causal boundary is the CUMULATIVE (absolute-byte) position, never a local code index.
            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_q, sin_q = cos_b, sin_b
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)

            x_cross = self.fuse_stages[s](x_cross, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)   # refinement pass

            if self.word_head is not None:
                h_normed = self.fuse_stages[s].ln_out(x_cross)
                cond_loss, cond_acc = self.word_head.loss_acc(h_normed, byte_ids)
                cond_loss = cond_loss if cond_loss is not None else x_cross.new_zeros(())
                cond_acc = cond_acc if cond_acc is not None else x_cross.new_zeros(())
            else:
                cond_logits_full = self.fuse_stages[s].readout(x_cross, self.head.weight)
                cond_logits = cond_logits_full[:, :-1, :]
                cond_loss = F.cross_entropy(cond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
                cond_acc = (cond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()
            cond_losses += [cond_loss]
            cond_accs += [cond_acc]

            cur_h = h_code

        # MTP heads: extra byte-ahead heads off the final post-cascade hidden state.
        final_h = x_cross
        mtp_losses, mtp_accs = [], []
        for i, head in enumerate(self.extra_heads):
            k = i + 2
            if L <= k:
                continue
            logits_k = F.linear(final_h[:, :-k, :], head.weight)
            targets_k = byte_ids[:, k:]
            mtp_losses.append(F.cross_entropy(logits_k.reshape(-1, V), targets_k.reshape(-1)))
            mtp_accs.append((logits_k.argmax(-1) == targets_k).float().mean())

        final_loss = cond_losses[-1] if cond_losses else uncond_loss
        final_acc = cond_accs[-1] if cond_accs else uncond_acc
        total_loss = (sum(cond_losses) * cfg.cond_weight if cond_losses else uncond_loss)
        if fuse_ntp_losses:
            total_loss = total_loss + cfg.code_ntp_weight * torch.stack(fuse_ntp_losses).sum()
        if mtp_losses:
            total_loss = total_loss + cfg.mtp_weight * torch.stack(mtp_losses).mean()
        if code_mtp_losses:
            total_loss = total_loss + cfg.mtp_weight_code * torch.stack(list(code_mtp_losses.values())).mean()
        if uncond_mtp_losses:
            total_loss = total_loss + cfg.mtp_weight_uncond * torch.stack(uncond_mtp_losses).mean()

        metrics = {
            "loss": total_loss, "final_loss": final_loss, "byte_acc": final_acc,
            "uncond_loss": uncond_loss, "uncond_acc": uncond_acc,
            **{f"cond{s}_loss": l for s, l in enumerate(cond_losses)},
            **{f"cond{s}_acc": a for s, a in enumerate(cond_accs)},
            **{f"fuse{s}_ntp_loss": l for s, l in enumerate(fuse_ntp_losses)},
            **{f"fuse{s}_ntp_acc": a for s, a in enumerate(fuse_ntp_accs)},
            **{f"mtp{i+2}_loss": l for i, l in enumerate(mtp_losses)},
            **{f"mtp{i+2}_acc": a for i, a in enumerate(mtp_accs)},
            **{f"mtp{k}_code{s}_loss": l for (s, k), l in code_mtp_losses.items()},
            **{f"mtp{k}_code{s}_acc": a for (s, k), a in code_mtp_accs.items()},
            **{f"mtp{i+2}_uncond_loss": l for i, l in enumerate(uncond_mtp_losses)},
            **{f"mtp{i+2}_uncond_acc": a for i, a in enumerate(uncond_mtp_accs)},
        }
        return total_loss, metrics

    @torch.no_grad()
    def _generate_cascade(self, byte_ids: torch.Tensor) -> tuple:
        """Shared no-grad full-recompute cascade for generation. Returns (cond_logits_full,
        code_kv_cache, final_h) -- the single source of truth _forward_next_byte_logits reads from."""
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = D // cfg.n_heads
        device = byte_ids.device
        byte_pos = torch.arange(L, device=device)
        cos_b, sin_b = rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base, device)
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        h = self._run_blocks(0, x0, cos_b, sin_b, byte_mask)

        cur_h = h
        x_cross = h
        cum_K = 1
        cond_logits_full = self.head(h)  # uncond fallback if n_fuse==0 -- h already normed by _run_blocks
        code_kv_cache = []
        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]
            onehot, code_embeds = self.quantizers[s].extract_greedy(code_h)

            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(s + 1, code_embeds, cos_c, sin_c, code_mask)

            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
            x_cross = self.fuse_stages[s](x_cross, h_code, cos_b, sin_b, cos_k, sin_k, fuse_mask)
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)
            cond_logits_full = self.fuse_stages[s].readout(x_cross, self.head.weight)
            code_kv_cache += [(h_code, code_pos_abs, window_s)]
            cur_h = h_code

        return cond_logits_full, code_kv_cache, x_cross

    @torch.no_grad()
    def _forward_next_byte_logits(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Full recompute over the whole sequence so far, returns logits for the NEXT byte
        (position L, i.e. the last position's post-fusion prediction)."""
        cond_logits_full, _, _ = self._generate_cascade(byte_ids)
        return cond_logits_full[:, -1, :]

    @torch.no_grad()
    def generate_no_cache(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Byte-by-byte, full recompute each step -- correctness reference. generate_kv_cache
        (below) produces the exact same argmax trajectory, incrementally, for actual use."""
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        for _ in range(n_new_bytes):
            logits = self._forward_next_byte_logits(all_bytes)
            next_byte = logits.argmax(-1, keepdim=True)
            all_bytes = torch.cat([all_bytes, next_byte], dim=1)
        if was_training:
            self.train()
        return all_bytes[0]

    def _make_incremental_stepper(self, Bsz: int, device_t: torch.device):
        """Factory for the incremental-KV-cache stepper, shared by generate_kv_cache/generate_speculative."""
        cfg = self.cfg
        D = cfg.d_model
        hd = D // cfg.n_heads

        byte_caches = [None] * cfg.n_layers
        refine_caches = [[None] * cfg.n_layers for _ in range(self.n_fuse)]
        h_hist = None                        # (Bsz, cur_L, D): raw byte hidden states so far
        stage_h_hist = [torch.zeros(Bsz, 0, D, device=device_t) for _ in range(self.n_fuse)]
        # per-stage backlog: while a stage is fully inactive, its input accumulates here so the
        # first activation can catch up in ONE priming call, matching a full recompute exactly.
        x_in_backlog = [None] * self.n_fuse
        cum_Ks = []
        cum = 1
        for K_s in cfg.Ks[:self.n_fuse]:
            cum *= K_s
            cum_Ks.append(cum)

        def step(byte_chunk: torch.Tensor, start_pos: int) -> torch.Tensor:
            nonlocal h_hist
            Tn = byte_chunk.shape[1]
            pos = torch.arange(start_pos, start_pos + Tn, device=device_t)
            cos_b, sin_b = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device_t)
            h_new = self.embed(byte_chunk)
            for l, block in enumerate(self.lms[0]):
                h_new, byte_caches[l] = block.forward_incremental(h_new, cos_b, sin_b, byte_caches[l], cfg.attn_window)
            h_new = self.ln_fs[0](h_new)
            h_hist = h_new if h_hist is None else torch.cat([h_hist, h_new], dim=1)

            x_in = h_new
            cur_h_hist = h_hist
            logits_full = self.head(x_in)  # uncond fallback if n_fuse==0
            for s in range(self.n_fuse):
                K_s = cfg.Ks[s]
                n_blocks = cur_h_hist.shape[1] // K_s
                if n_blocks > stage_h_hist[s].shape[1]:
                    # a new code boundary was crossed -- recompute this stage's short code
                    # sequence fresh (cheap: length n_blocks, not the full byte length)
                    code_h = cur_h_hist[:, K_s - 1::K_s, :][:, :n_blocks, :]
                    _, code_embeds = self.quantizers[s].extract_greedy(code_h)
                    code_local_pos = torch.arange(n_blocks, device=device_t)
                    cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device_t)
                    code_mask = causal_mask(code_local_pos, code_local_pos, None)
                    stage_h_hist[s] = self._run_blocks(s + 1, code_embeds, cos_c, sin_c, code_mask)
                h_code = stage_h_hist[s]
                n_blocks_now = h_code.shape[1]

                if n_blocks_now < 1:
                    # stage still fully inactive -- hard BREAK, matching forward()'s "if
                    # n_blocks<1: break" (a deeper stage can never be active while this one isn't).
                    x_in_backlog[s] = x_in if x_in_backlog[s] is None else torch.cat([x_in_backlog[s], x_in], dim=1)
                    break

                code_pos_abs = (torch.arange(n_blocks_now, device=device_t) + 1) * cum_Ks[s] - 1
                window_s = self.fuse_windows[s]
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device_t)

                if refine_caches[s][0] is None:
                    # first activation: prime with the FULL backlog (+ this chunk) in one shot,
                    # true absolute positions (this chunk's end always equals start_pos+Tn)
                    x_q = x_in if x_in_backlog[s] is None else torch.cat([x_in_backlog[s], x_in], dim=1)
                    x_in_backlog[s] = None
                else:
                    x_q = x_in
                q_len = x_q.shape[1]
                q_start = (start_pos + Tn) - q_len
                q_pos = torch.arange(q_start, q_start + q_len, device=device_t)
                cos_q, sin_q = rope_cos_sin_for_positions(q_pos, hd, cfg.rope_base, device_t)
                fuse_mask = causal_mask(q_pos, code_pos_abs, window_s)

                x_cross = self.fuse_stages[s](x_q, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
                for l, block in enumerate(self.lms[0]):
                    x_cross, refine_caches[s][l] = block.forward_incremental(
                        x_cross, cos_q, sin_q, refine_caches[s][l], cfg.attn_window)
                x_cross = self.ln_fs[0](x_cross)
                logits_full = self.fuse_stages[s].readout(x_cross, self.head.weight)
                x_in = x_cross
                cur_h_hist = h_code
            return logits_full

        return step

    @torch.no_grad()
    def generate_kv_cache(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """O(1) new attention work per new byte (vs generate_no_cache's full O(L) recompute),
        exact same argmax trajectory -- see check_kv_cache_consistency."""
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        step = self._make_incremental_stepper(prompt_bytes.shape[0], torch.device(device))

        all_bytes = prompt_bytes
        logits_all = step(all_bytes, 0)          # prime the caches with the whole prompt
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_bytes):
            next_byte = next_logits.argmax(-1, keepdim=True)
            all_bytes = torch.cat([all_bytes, next_byte], dim=1)
            logits_all = step(next_byte, all_bytes.shape[1] - 1)   # feed only the new byte
            next_logits = logits_all[:, -1, :]

        if was_training:
            self.train()
        return all_bytes[0]

    @torch.no_grad()
    def generate_speculative(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                              return_stats: bool = False, verbose: bool = False):
        """MTP-drafted speculative decode, verified vs the exact incremental stepper. Batch size 1 only."""
        cfg = self.cfg
        assert cfg.mtp_heads > 1, "generate_speculative requires cfg.mtp_heads > 1"
        device_t = torch.device(device)

        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        Bsz = prompt_bytes.shape[0]

        step = self._make_incremental_stepper(Bsz, device_t)
        all_bytes = prompt_bytes
        logits_all = step(all_bytes, 0)          # prime the verifier with the prompt
        next_logits = logits_all[:, -1, :]

        target_len = prompt_bytes.shape[1] + n_new_bytes
        n_accepted, n_checked, n_rounds, n_draft_passes = 0, 0, 0, 0

        while all_bytes.shape[1] < target_len:
            m = all_bytes.shape[1]
            draft_len = min(cfg.mtp_heads, target_len - m)
            n_rounds += 1

            # draft: extra_heads, ONE forward pass, no per-slot attention-stack cost.
            cond_logits_full, _, final_h = self._generate_cascade(all_bytes)
            n_draft_passes += 1
            draft_bytes = [cond_logits_full[:, -1, :].argmax(-1, keepdim=True)]
            last_h = final_h[:, -1:, :]
            for i in range(draft_len - 1):
                if i >= len(self.extra_heads):
                    break
                logits_i = F.linear(last_h, self.extra_heads[i].weight)
                draft_bytes.append(logits_i[:, -1, :].argmax(-1, keepdim=True))
            draft_bytes = torch.cat(draft_bytes, dim=1)  # (Bsz, <=draft_len)
            draft_len = draft_bytes.shape[1]
            if verbose:
                print(f"[mtp round {n_rounds}] draft_passes=1 draft={_fmt_bytes(draft_bytes[0])!r}")

            # --- verify: one real position at a time, against the SAME exact incremental stepper ---
            for i in range(draft_len):
                n_checked += 1
                real_byte = next_logits.argmax(-1, keepdim=True)   # the verifier's true choice
                draft_byte = draft_bytes[:, i:i + 1]
                agree = torch.equal(real_byte, draft_byte)
                accepted_byte = draft_byte if agree else real_byte
                if verbose:
                    tag = "ACCEPT" if agree else "REJECT"
                    print(f"    slot {i}: draft={_fmt_bytes(draft_byte[0])!r} "
                          f"real={_fmt_bytes(real_byte[0])!r} -> {tag}")
                if agree:
                    n_accepted += 1
                all_bytes = torch.cat([all_bytes, accepted_byte], dim=1)
                logits_all = step(accepted_byte, all_bytes.shape[1] - 1)
                next_logits = logits_all[:, -1, :]
                if not agree:
                    break   # reject: discard the rest of this round's draft, start a fresh one

        if was_training:
            self.train()
        seq = all_bytes[0]
        stats = {"accept_rate": n_accepted / max(1, n_checked), "n_draft_checks": n_checked,
                  "n_rounds": n_rounds, "n_draft_passes": n_draft_passes}
        if verbose:
            print(f"[mtp summary] rounds={n_rounds} draft_passes={n_draft_passes} "
                  f"verify_checks={n_checked} accepted={n_accepted} accept_rate={stats['accept_rate']:.3f}")
        if return_stats:
            return seq, stats
        return seq

    @torch.no_grad()
    def check_kv_cache_consistency(self, val_data: torch.Tensor, device: str,
                                    n_checks: int = 3, prompt_len: int = 8, n_new_bytes: int = 24) -> dict:
        """generate_no_cache vs generate_kv_cache MUST match bit-exact. Returns
        {"match_rate": float, "n_checks": int} -- should always be 1.0."""
        was_training = self.training
        self.eval()
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))  # vary length, incl. short
            start = torch.randint(0, max(1, val_data.shape[0] - pl - n_new_bytes), (1,)).item()
            prompt = val_data[start:start + pl].to(device)
            out_full = self.generate_no_cache(prompt, n_new_bytes, device)
            out_cache = self.generate_kv_cache(prompt, n_new_bytes, device)
            if torch.equal(out_full, out_cache):
                n_match += 1
        if was_training:
            self.train()
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}


# training loop

def eval_model(model, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    totals: dict = {}
    with torch.no_grad():
        for _ in range(n_batches):
            ctx = sample_context(data, batch_size, model.cfg.context_len, device)
            _, metrics = model(ctx)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v.item()
    model.train()
    return {k: v / n_batches for k, v in totals.items()}


def train(model, train_data, val_data, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.logs_dir / run_name, args.save_every_n_evals, minimize=True)
    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr
        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            scalars = {k: v.item() for k, v in metrics.items()}
            log(f"{pbar}", step=step, lr=lr, **scalars)

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["loss"])
            log(f"{pbar}  {val_str}  best_val_loss={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_loss=checkpointer.best_metric)

            if args.eval_decode_mtp_verify and model.cfg.mtp_heads > 1:
                # show NTP vs MTP-verified decode side by side -- should always be IDENTICAL.
                prompt = val_data[:args.eval_decode_prompt_len]
                n_new = model.cfg.mtp_heads   # one full draft round's worth -- MTP's own max
                out_ntp = model.generate_no_cache(prompt, n_new, device)
                out_mtp, spec_stats = model.generate_speculative(prompt, n_new, device, return_stats=True)
                text_ntp = pack_words(out_ntp.tolist(), model.cfg.input_preset).decode("latin-1", errors="replace")
                text_mtp = pack_words(out_mtp.tolist(), model.cfg.input_preset).decode("latin-1", errors="replace")
                match = "MATCH" if torch.equal(out_ntp, out_mtp) else "MISMATCH (bug!)"
                log(f"{pbar}  mtp_verify accept_rate={spec_stats['accept_rate']:.4f} "
                    f"n_draft_checks={spec_stats['n_draft_checks']}  ntp/mtp {match}\n"
                    f"    ntp: {text_ntp!r}\n"
                    f"    mtp: {text_mtp!r}",
                    step=step, mtp_accept_rate=spec_stats["accept_rate"],
                    mtp_n_draft_checks=spec_stats["n_draft_checks"],
                    ntp_text=text_ntp, mtp_text=text_mtp, ntp_mtp_match=(match == "MATCH"))


def build_argparser(description: str) -> tuple:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=description, parents=[pre])
    p.add_argument("--Ks", default=(32, 32, 1))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--fuse_n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--attn_window", default=None)
    p.add_argument("--fuse_window", default=None)
    p.add_argument("--input_preset", type=int, default=8)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--code_hard", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--code_sample", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--cond_weight", type=float, default=1.0)
    p.add_argument("--mtp_heads", type=int, default=1)
    p.add_argument("--mtp_weight", type=float, default=1.0)
    p.add_argument("--weight_tie", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--global_tie", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--quant_type", type=str, default="simplex")
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--pq_chunks", type=int, default=1)
    p.add_argument("--share_lm", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_fuse", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--mtp_heads_code", type=int, default=1)
    p.add_argument("--mtp_weight_code", type=float, default=1.0)
    p.add_argument("--mtp_heads_uncond", type=int, default=1)
    p.add_argument("--mtp_weight_uncond", type=float, default=1.0)
    p.add_argument("--head_word_bits", type=int, default=None)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=5)
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--eval_decode_mtp_verify", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--eval_decode_prompt_len", type=int, default=16)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    if isinstance(args.Ks, str):
        args.Ks = tuple(int(x) for x in args.Ks.split(","))
    else:
        args.Ks = tuple(args.Ks)
    return args, pre_args


def config_from_args(args) -> Config:
    return Config(
        Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers, fuse_n_layers=args.fuse_n_layers,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, rope_base=args.rope_base, context_len=args.context_len,
        attn_window=args.attn_window, fuse_window=args.fuse_window, input_preset=args.input_preset,
        gumbel_tau=args.gumbel_tau, code_hard=args.code_hard, code_sample=args.code_sample,
        code_ntp_weight=args.code_ntp_weight, cond_weight=args.cond_weight,
        mtp_heads=args.mtp_heads, mtp_weight=args.mtp_weight, weight_tie=args.weight_tie,
        global_tie=args.global_tie,
        quant_type=args.quant_type, vocab=args.vocab, pq_chunks=args.pq_chunks,
        share_lm=args.share_lm, share_fuse=args.share_fuse,
        mtp_heads_code=args.mtp_heads_code, mtp_weight_code=args.mtp_weight_code,
        mtp_heads_uncond=args.mtp_heads_uncond, mtp_weight_uncond=args.mtp_weight_uncond,
        head_word_bits=args.head_word_bits,
    )


def main() -> None:
    args, pre_args = build_argparser("qcute_zero: single-LM periodic-fusion architecture")
    torch.manual_seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config_from_args(args)
    model = QCuteZero(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_zero_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} n_fuse={model.n_fuse} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, cfg.input_preset, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
