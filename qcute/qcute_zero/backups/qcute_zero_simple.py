"""qcute_zero: a monolithic, single-LM alternative to qcute_v1's multi-encoder StackDecoder
lineage (see CLAUDE.md's Architecture section for qcute_v1; this is a separate lineage, not a
fork of it). Design, restated (chat 2026-08-22):

There is exactly ONE transformer LM (level0, byte space). Every K bytes it summarizes its own
just-produced hidden state into a discrete code -- via the SAME tied embed/output head bytes
already use (byte vocab and code vocab are the same space, so "extracting a code" is literally
"predict a byte-shaped distribution and take a differentiable (STE) hard sample of it"). That
code sequence is then run through the SAME shared blocks again (a second, much shorter forward
pass) to get (a) a genuine NTP loss on the code sequence itself (predict the next code from
previous codes, using the identical loss machinery as byte NTP -- "free" via weight reuse, no
separate per-level encoder needed) and (b) contextualized representations that become the K/V for
a cross-attention stage feeding back into the byte-level query stream. Repeat for every entry in
`Ks` (len(Ks)-1 "fuse" stages total, one per cumulative period Ks[0], Ks[0]*Ks[1], ... -- same Ks
semantics as qcute_v1) -- each stage's codes are built FROM the previous stage's own contextualized
hidden state, a genuine cascade, not independent re-samples of the raw byte hidden state.

Causality: every code's causal boundary is its CUMULATIVE byte-span (`cum_K*(block_idx+1)-1`, in
absolute byte-position coordinates), never its local index within whatever intermediate sequence
produced it -- getting this wrong (comparing local code-sequence indices directly against absolute
byte query positions) is the one way this design could accidentally become circular; using the
cumulative boundary throughout keeps every stage strictly non-circular (verified by hand, chat
2026-08-22: a code can only ever inform prediction of bytes strictly after every byte it was
itself computed from, never any byte it depends on).

Zero-KV sink (mandatory on every attention call, self- and cross-): a fixed (non-trainable) all-
zero key/value pair is always prepended and always visible, so every query row has >=1 valid key
even when every real key is masked out (e.g. a query before any periodic code's causal boundary
has been reached -- true for every query strictly before Ks[0]-1, and again before every deeper
stage's own first boundary). Softmax over a single visible key is always weight 1 regardless of
its score, so when the sink is the ONLY visible key the attention output is exactly zero -- a
provably clean no-op contribution, not an arbitrary bias, and immune to NaN. Because the sink's
value is exactly zero, whether it's "rotated" by RoPE is moot (a zero vector rotates to itself);
it's simplest to just prepend it after RoPE has been applied to the real keys.

No curriculum needed by design (unlike qcute_v1's max_srcs/curriculum_max_srcs hack): every fuse
stage's code source is the SAME shared, already-training backbone from step 1 (nothing is a fresh,
untouched, randomly-initialized module the way each qcute_v1 encoder level was), and the zero-sink
lets a stage's own freshly-initialized cross-attention weights learn to suppress themselves early
(put softmax weight on the sink) and gradually rely on real codes as those weights improve -- an
emergent, learned on-ramp instead of a hand-scheduled one. Expected, not yet proven -- the whole
point of the ks21/ks221-no-curriculum runs this file's plan calls for.

Query for "what predicts a new position" is the ordinary previous-token hidden state (no seed/BOS
token at all, unlike qcute_v1) -- pure standard AR continuation, causal by construction.

Real incremental KV caching (`generate_kv_cache`): byte-level self.blocks self-attention and each
fuse stage's post-cross-attn refinement self.blocks pass are cached across generation steps
(O(1) attention work per new byte instead of full O(L) recompute) -- see `Attn.forward_incremental`/
`Block.forward_incremental`. The short code-sequence self-attention (kvlm) pass and the fuse
cross-attention itself are still recomputed fresh whenever a new code appears (every Ks[s] bytes),
since code sequences are short (length ~ L/prod(Ks[:s+1])) -- not worth incrementally caching.
Produces the exact same argmax choices as `generate_no_cache` (verified by direct comparison),
just asymptotically cheaper for long generations.

MTP heads (`Config.mtp_heads`, 2026-08-22): optional extra `nn.Linear(D, V)` heads reading the
SAME final hidden state head0's own cond/uncond readout already uses -- pervasive (every position,
every step) and cheap (zero extra attention FLOPs), unlike the query_vec/`parallel_decode`
mechanism this superseded (one query_vec slot cost a full attention-stack pass, and only covered
`parallel_decode_n_blocks` sampled clusters per step). `generate_speculative` drafts via these
heads now. The query_vec idea itself is preserved as its own standalone testbed, forked onto the
simpler `qcute.bytelm` trunk: `qcute/bytelm_queryvec/bytelm_queryvec.py` (`qcute/qcute_zero_parallel/`,
the original fork of this file holding that mechanism, is now redundant/archived).

Single file by design for now (explicitly asked: "make thing single file first refactor later") --
copies/adapts primitives from qcute_v1_common.py (Block/RoPE/Logger/data-loading/train-loop shapes)
rather than importing them, since this is meant to stay a separate, prunable lineage.

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


# ----------------------------------------------------------------------------
# small shared utilities (copied/trimmed from qcute_v1_common.py)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# RoPE + attention primitives
# ----------------------------------------------------------------------------

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}  # theta only, no
                                                                              # Llama3.1 NTK-by-parts scaling


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    """position_ids: (T,) shared across the whole batch (the common case), or (Bv, T) -- one
    absolute-position row per batch element (block-folded parallel-decode training, where
    different folded blocks sit at different real byte positions)."""
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
    """Mandatory zero-value/zero-key sink, prepended to every attention call (self- and cross-) in
    this model: guarantees every query row has >=1 valid key (avoids NaN when a row's real keys
    are all masked False, e.g. before a periodic code's causal boundary is reached), and gives a
    provably clean zero contribution when it's the only visible key (softmax of one element is
    always weight 1, so output = 1*0 = 0) -- see chat 2026-08-22. attn_mask: bool, True=visible,
    shape (..., T, S) broadcastable to (B, H, T, S)."""
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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Attn(nn.Module):
    """Self- and cross-attention share this: same QKV/out projections, sdpa_with_sink mandatory
    either way. forward() = self-attention (Q,K,V all from x); forward_cross() = cross-attention
    (Q from x, K/V from a separate kv sequence). GQA (n_kv_heads < n_heads repeats each KV head
    across n_heads//n_kv_heads query heads) + optional Qwen3-style QK-norm (per-head RMSNorm on
    Q/K before RoPE) + optional decoupled head_dim (Qwen3-style: some of its smaller variants set
    head_dim independently of d_model//n_heads -- None here is a no-op, deriving head_dim the old
    way)."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, qk_norm: bool = True,
                 head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.d_model = d_model
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.attn_dim = n_heads * self.head_dim
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(self.attn_dim, d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, Hkv, T, hd = x.shape
        return x[:, :, None].expand(B, Hkv, self.n_rep, T, hd).reshape(B, Hkv * self.n_rep, T, hd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, Hkv, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, self.attn_dim))

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        """Incremental self-attention: x_new is only the NEW position(s) (Tn=1 per generation
        step, or the whole prompt on the priming call); cache is None (nothing yet) or (k_prev,
        v_prev) from earlier calls, stored PRE-repeat (n_kv_heads heads). Returns (out, new_cache)
        -- new_cache is trimmed to the last `window` entries when windowed, so a subsequent call
        only ever pays for what's visible. Mask uses LOCAL (call-relative) positions -- only
        relative order matters for causality, and cos/sin (computed from true absolute positions
        by the caller) is what actually encodes real distance, so this stays exactly consistent
        with the full-recompute path."""
        B, Tn, D = x_new.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x_new).view(B, Tn, H, hd).transpose(1, 2)
        k = self.wk(x_new).view(B, Tn, Hkv, hd).transpose(1, 2)
        v = self.wv(x_new).view(B, Tn, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
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
        y = sdpa_with_sink(q, self._repeat_kv(k_all), self._repeat_kv(v_all), mask)
        out = self.out(y.transpose(1, 2).reshape(B, Tn, self.attn_dim))
        if window is not None and S > window:
            k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
        return out, (k_all, v_all)

    def forward_cross(self, x_q: torch.Tensor, x_kv: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor,
                       cos_k: torch.Tensor, sin_k: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x_q).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x_kv).view(B, S, Hkv, hd).transpose(1, 2)
        v = self.wv(x_kv).view(B, S, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, self.attn_dim))


class SwiGLU(nn.Module):
    """gate/up/down, no bias: down(silu(gate(x)) * up(x)) -- Llama3-style MLP, replaces the plain
    Linear-GELU-Linear MLP everywhere in this file (Block and FuseStage both)."""
    def __init__(self, d_model: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * d_model
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """"block regular": self-attention + MLP. Shared (same weights) across the byte-level pass and
    every fuse stage's own code-sequence NTP pass -- this IS the "single LM" the whole design
    hinges on."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_kv_heads: int | None = None, qk_norm: bool = True,
                 head_dim: int | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads, n_kv_heads, qk_norm, head_dim)
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
    """"block fuse": cross-attention + MLP, one instance per periodic-fusion stage, own weights
    throughout (no cross-stage sharing) -- including this stage's own final LayerNorm feeding its
    own cond NTP readout (logits via the shared tied embed weight, passed in). Cheap: called with
    the code sequence's length (L/cum_K), not the byte sequence's."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int,
                 n_kv_heads: int | None = None, qk_norm: bool = True, head_dim: int | None = None):
        super().__init__()
        self.ln1 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.ModuleList([Attn(d_model, n_heads, n_kv_heads, qk_norm, head_dim) for _ in range(n_layers)])
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


# ----------------------------------------------------------------------------
# Config + model
# ----------------------------------------------------------------------------

@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32, 1)       # same semantics as qcute_v1: cumulative periods, last
                                              # entry conventionally 1 (no further fuse stage after it)
    d_model: int = 256
    n_layers: int = 4                        # scalar -- shared "block regular", reused for every
                                              # fuse stage's own code-sequence NTP pass too
    fuse_n_layers: int | None = None         # defaults to n_layers if unset
    n_heads: int = 4
    n_kv_heads: int | None = None            # None = max(1, n_heads//4) (Llama3/Qwen3-style GQA-by-default); set == n_heads for plain MHA
    head_dim: int | None = None              # None = d_model // n_heads (no-op, Llama3/big-Qwen3 style); set to
                                              # decouple from d_model/n_heads (small-Qwen3 style, e.g. head_dim=128)
    qk_norm: bool = True                    # Qwen3-style per-head RMSNorm on Q/K before RoPE
    mlp_mult: int = 4
    rope_base: float = 10000.0
    rope_preset: str | None = "qwen3"           # "llama2"/"llama3"/"qwen3" overrides rope_base (see ROPE_PRESETS)
    context_len: int = 256
    attn_window: int | None = None           # main byte self-attention window (None = unbounded)
    fuse_window: int | tuple | None = None   # per-fuse-stage cross-attn window, in BYTES; None/scalar/tuple
    input_preset: int = 8                    # byte alphabet bits -- vocab = 2**input_preset, shared
                                              # by codes (same embed/output head)
    gumbel_tau: float = 1.0
    code_hard: bool = True
    code_sample: bool = False
    code_ntp_weight: float = 1.0             # weight for each fuse stage's own code-sequence NTP loss
    cond_weight: float = 1.0                 # weight for each stage's post-fusion byte NTP loss
    mtp_heads: int = 1                       # extra byte-ahead heads reading the SAME final hidden
                                              # state (post-cascade), MTP-style (see qcute.bytelm) --
                                              # 1 = disabled (only the existing head0 next-byte
                                              # prediction). >1 heads predict t+2..t+mtp_heads.
    mtp_weight: float = 1.0                  # weight for the extra heads' mean loss
    mtp_heads_code: int = 1                  # extra code-ahead heads off h_code, per stage (1=disabled)
    mtp_weight_code: float = 1.0
    mtp_heads_uncond: int = 1                # extra byte-ahead heads off pre-fusion h (1=disabled)
    mtp_weight_uncond: float = 1.0
    weight_tie: bool = False                 # True: head.weight literally refs embed.weight
    share_lm: bool = False                   # True ties every level's LM stack to lms[0]
    share_fuse: bool = False                 # True ties every fuse stage to fuse_stages[0]


def resolve_fuse_window(w, n_fuse: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_fuse
        return tuple(w)
    return (w,) * n_fuse


class QCuteZero(nn.Module):
    def __init__(self, cfg: Config):
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.head_dim = cfg.head_dim if cfg.head_dim is not None else D // cfg.n_heads
        V = 2 ** cfg.input_preset
        self.vocab = V
        self.n_fuse = len(cfg.Ks) - 1
        assert D % cfg.n_heads == 0

        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)

        # level 0 = byte pass + refinement; level s+1 = fuse stage s's own code-sequence NTP pass.
        n_lms = self.n_fuse + 1
        if cfg.share_lm:
            first = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(cfg.n_layers)])
            self.lms = nn.ModuleList([first] * n_lms)
        else:
            self.lms = nn.ModuleList(
                [nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(cfg.n_layers)])
                 for _ in range(n_lms)])
        if cfg.share_lm:
            first_ln = RMSNorm(D)
            self.ln_fs = nn.ModuleList([first_ln] * n_lms)
        else:
            self.ln_fs = nn.ModuleList([RMSNorm(D) for _ in range(n_lms)])

        self.head = nn.Linear(D, V, bias=False)
        if cfg.weight_tie:
            self.head.weight = self.embed.weight
        else:
            nn.init.normal_(self.head.weight, std=0.02)

        fuse_layers = cfg.fuse_n_layers if cfg.fuse_n_layers is not None else cfg.n_layers
        if cfg.share_fuse:
            first_fs = FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim)
            self.fuse_stages = nn.ModuleList([first_fs] * self.n_fuse)
        else:
            self.fuse_stages = nn.ModuleList(
                [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        self.extra_heads = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads - 1))])
        self.extra_heads_uncond = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads_uncond - 1))])
        if cfg.share_lm:
            first_ehc = nn.ModuleList(
                [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads_code - 1))])
            self.extra_heads_code_per_stage = nn.ModuleList([first_ehc] * max(1, self.n_fuse))
        else:
            self.extra_heads_code_per_stage = nn.ModuleList([nn.ModuleList(
                [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads_code - 1))])
                for _ in range(max(1, self.n_fuse))])

    def _run_blocks(self, level: int, x: torch.Tensor, cos, sin, attn_mask) -> torch.Tensor:
        for block in self.lms[level]:
            x = block(x, cos, sin, attn_mask)
        return self.ln_fs[level](x)

    def forward(self, byte_ids: torch.Tensor) -> tuple:
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = self.head_dim
        device = byte_ids.device
        V = self.vocab

        # --- byte-level pass ("block regular"), uncond ---
        byte_pos = torch.arange(L, device=device)
        cos_b, sin_b = rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base, device)
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        h = self._run_blocks(0, x0, cos_b, sin_b, byte_mask)

        uncond_logits = F.linear(h[:, :-1, :], self.head.weight)
        uncond_loss = F.cross_entropy(uncond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
        uncond_acc = (uncond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()

        # cheap/coarse extra byte-ahead heads off the pre-fusion hidden state h.
        uncond_mtp_losses, uncond_mtp_accs = [], []
        for i, head_u in enumerate(self.extra_heads_uncond):
            k = i + 2
            if L <= k:
                continue
            logits_u = F.linear(h[:, :-k, :], head_u.weight)
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
        code_kv_cache = []       # (h_code_s, code_pos_abs, window) per stage

        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break

            # code extraction: same tied embed/output head bytes use, STE hard sample
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]
            code_logits = F.linear(code_h, self.head.weight)
            onehot = gumbel_quantize(code_logits, cfg.gumbel_tau, cfg.code_hard, cfg.code_sample)
            code_embeds = onehot @ self.embed.weight
            code_ids = onehot.argmax(-1)

            # this stage's own code-sequence NTP pass -- level s+1's own LM stack, causal,
            # unbounded (short sequence: n_blocks = cur_len // K_s)
            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(s + 1, code_embeds, cos_c, sin_c, code_mask)

            if n_blocks >= 2:
                code_ntp_logits = F.linear(h_code[:, :-1, :], self.head.weight)
                code_ntp_loss = F.cross_entropy(code_ntp_logits.reshape(-1, V), code_ids[:, 1:].reshape(-1))
                code_ntp_acc = (code_ntp_logits.argmax(-1) == code_ids[:, 1:]).float().mean()
                fuse_ntp_losses += [code_ntp_loss]
                fuse_ntp_accs += [code_ntp_acc]

            # extra code-ahead heads off h_code, this stage's own set.
            for i, head_c in enumerate(self.extra_heads_code_per_stage[s]):
                k = i + 2
                if n_blocks <= k:
                    continue
                logits_c = F.linear(h_code[:, :-k, :], head_c.weight)
                target_c = code_ids[:, k:]
                code_mtp_losses[(s, k)] = F.cross_entropy(logits_c.reshape(-1, V), target_c.reshape(-1))
                code_mtp_accs[(s, k)] = (logits_c.argmax(-1) == target_c).float().mean()

            # cross-attn: byte-level query stream attends into h_code, causal on CUMULATIVE
            # (absolute-byte) boundary, never this stage's local code-sequence index (chat
            # 2026-08-22: using the local index here would be the one way this becomes circular).
            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_q, sin_q = cos_b, sin_b
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
            x_cross = self.fuse_stages[s](x_cross, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
            # another pass through level 0's own self-attn+MLP LM blocks before this stage's own
            # cond readout (and before the next stage's cross-attn query input) -- i.e. fuse
            # cross-attn+own-mlp -> level0 self-attn/mlp -> this stage's own cond NTP head.
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)
            cond_logits_full = self.fuse_stages[s].readout(x_cross, self.head.weight)

            cond_logits = cond_logits_full[:, :-1, :]
            cond_loss = F.cross_entropy(cond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
            cond_acc = (cond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()
            cond_losses += [cond_loss]
            cond_accs += [cond_acc]

            code_kv_cache += [(h_code, code_pos_abs, window_s)]
            cur_h = h_code

        # --- optional: MTP heads, reading the SAME final hidden state (x_cross, post-cascade --
        # equal to h if n_fuse==0) that head0's own cond/uncond readout already uses. Each extra
        # head i (0-indexed here, predicting offset i+2 since head0 already covers offset+1) is a
        # separate untied nn.Linear(D, V) -- cheap (O(mtp_heads * D * V) params, zero extra
        # attention FLOPs), pervasive (computed at every position, not just sampled clusters),
        # unlike the pruned query_vec/parallel_decode mechanism (see qcute.bytelm_queryvec for
        # that preserved lineage) which consumed a full attention-stack pass per drafted position
        # and only covered `parallel_decode_n_blocks` sampled clusters per step.
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
        """Shared no-grad cascade for generation (full recompute, no incremental state): same
        computation as forward()'s cascade minus the loss terms. Returns (cond_logits_full,
        code_kv_cache, final_h) -- cond_logits_full is the final stage's full per-position logits
        (uncond fallback if n_fuse==0), code_kv_cache is the per-stage (h_code, code_pos_abs,
        window) list, final_h is the raw final hidden state (pre-readout) generate_speculative's
        MTP-head drafting reads from. Used by _forward_next_byte_logits so there is exactly one
        generation-time code path, not two drifting copies -- unlike qcute_v1's
        generate_no_cache/_stack_generate_blockwise split (see docs/status.md's 2026-08-21/22
        generation-bug entry for why that split is risky)."""
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = self.head_dim
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
            code_logits = F.linear(code_h, self.head.weight)
            onehot = gumbel_quantize(code_logits, cfg.gumbel_tau, hard=True, sample=False)
            code_embeds = onehot @ self.embed.weight

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

    @torch.no_grad()
    def generate_free_rollout(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """2026-08-23 PoC (docs/maths.md Part 12's missing piece, now built): every other
        generate_* method always extracts a fuse stage's code from the REAL trunk hidden state at
        that chunk's own last byte -- requiring that chunk's bytes to already exist, so free
        rollout (qcute_v1's own-chunk-code-before-its-bytes trick, Part 8) was never possible here.
        This instead samples stage 0's NEXT code from its own already-trained causal NTP
        (h_code[:, -1, :], same gumbel_quantize head used everywhere else) using only chunks that
        are already real, THEN decodes the new chunk's own K bytes one at a time cross-attending
        to that pre-sampled code (own-chunk code, but never derived from its own bytes). Full
        recompute per new byte (like generate_no_cache, not the fast KV-cache path) -- a
        correctness PoC, not optimized. n_fuse==1 only (single fuse stage); deeper cascades are
        future work. Needs at least K0 real prompt bytes (one whole chunk) to bootstrap the first
        sample -- a true from-nothing cold start would need an explicit null-code fallback for
        chunk 0, not added here (see docs/maths.md's mid-sentence-init discussion)."""
        assert self.n_fuse == 1, "generate_free_rollout PoC only supports a single fuse stage (n_fuse==1)"
        cfg = self.cfg
        hd = self.head_dim
        device_t = torch.device(device)
        was_training = self.training
        self.eval()
        K = cfg.Ks[0]

        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]
        assert all_bytes.shape[1] >= K, "generate_free_rollout needs at least K0 real prompt bytes"

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            n_blocks_prev = all_bytes.shape[1] // K
            chunk_start = n_blocks_prev * K

            # --- sample THIS new chunk's own code ahead of time, from already-real chunks only ---
            L0 = all_bytes.shape[1]
            pos0 = torch.arange(L0, device=device_t)
            cos0, sin0 = rope_cos_sin_for_positions(pos0, hd, cfg.rope_base, device_t)
            h0 = self._run_blocks(0, self.embed(all_bytes), cos0, sin0, causal_mask(pos0, pos0, cfg.attn_window))
            code_h = h0[:, K - 1::K, :][:, :n_blocks_prev, :]
            onehot = gumbel_quantize(F.linear(code_h, self.head.weight), cfg.gumbel_tau, hard=True, sample=False)
            code_embeds_past = onehot @ self.embed.weight
            cpos = torch.arange(n_blocks_prev, device=device_t)
            ccos, csin = rope_cos_sin_for_positions(cpos, hd, cfg.rope_base, device_t)
            h_code = self._run_blocks(1, code_embeds_past, ccos, csin, causal_mask(cpos, cpos, None))
            next_onehot = gumbel_quantize(F.linear(h_code[:, -1:, :], self.head.weight),
                                           cfg.gumbel_tau, hard=True, sample=False)
            next_code_embed = next_onehot @ self.embed.weight  # (B, 1, D) -- sampled, not extracted

            # --- decode this new chunk's K bytes one at a time, cross-attending to the ONE
            # pre-sampled code (own-chunk, but never derived from its own bytes) ---
            buf = torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], K)], dim=1)
            for t in range(K):
                L = buf.shape[1]
                pos = torch.arange(L, device=device_t)
                cos_b, sin_b = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device_t)
                byte_mask = causal_mask(pos, pos, cfg.attn_window)
                x = self._run_blocks(0, self.embed(buf), cos_b, sin_b, byte_mask)

                fuse_mask = (pos >= chunk_start).view(1, 1, L, 1)
                code_pos = torch.tensor([chunk_start], device=device_t)
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos, hd, cfg.rope_base, device_t)
                x_cross = self.fuse_stages[0](x, next_code_embed, cos_b, sin_b, cos_k, sin_k, fuse_mask)
                x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)
                logits = self.fuse_stages[0].readout(x_cross, self.head.weight)
                # predict-next convention (matches forward()/generate_no_cache: position p's
                # logits predict byte p+1) -- position chunk_start+t is buf's own not-yet-decided
                # placeholder (still zero) at this point, so predicting FROM it (as an earlier
                # version of this PoC did) uses a zero-embedded, undetermined hidden state instead
                # of the last real one -- confirmed via direct A/B (degenerate repetitive output
                # vs. plausible text) that this off-by-one was the actual bug, not an architecture
                # limitation (2026-08-23).
                next_byte = logits[:, chunk_start + t - 1, :].argmax(-1, keepdim=True)
                buf = buf.clone()
                buf[:, chunk_start + t] = next_byte[:, 0]
            all_bytes = buf

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            self.train()
        return all_bytes[0]

    @torch.no_grad()
    def _make_incremental_stepper(self, Bsz: int, device_t: torch.device):
        """Factory for the real incremental-KV-cache stepper: returns a `step(byte_chunk,
        start_pos) -> logits_full` closure carrying its own mutable state (byte-level self-attn
        cache, each fuse stage's refinement cache, code histories, backlogs). Shared by
        generate_kv_cache (drives it byte-by-byte) and generate_speculative (drives it with
        whatever byte value needs verifying, drafted or corrected -- same exact machinery either
        way, so verification is always ground truth, never an approximation of it)."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim

        byte_caches = [None] * cfg.n_layers
        refine_caches = [[None] * cfg.n_layers for _ in range(self.n_fuse)]
        h_hist = None                        # (Bsz, cur_L, D): raw byte hidden states so far
        stage_h_hist = [torch.zeros(Bsz, 0, D, device=device_t) for _ in range(self.n_fuse)]
        # per-stage backlog: while a stage is still fully inactive (n_blocks_now==0, matching
        # forward()'s own "if n_blocks<1: break" -- the WHOLE stage is skipped, not just some
        # positions), its input is accumulated here so the first activation can catch up on
        # everything it missed in ONE priming call, exactly matching a full recompute at that
        # point (an earlier version skipped this catch-up entirely -- confirmed via direct
        # generate_no_cache vs generate_kv_cache mismatch on short prompts, chat 2026-08-22).
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
                    code_logits = F.linear(code_h, self.head.weight)
                    onehot = gumbel_quantize(code_logits, cfg.gumbel_tau, hard=True, sample=False)
                    code_embeds = onehot @ self.embed.weight
                    code_local_pos = torch.arange(n_blocks, device=device_t)
                    cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device_t)
                    code_mask = causal_mask(code_local_pos, code_local_pos, None)
                    stage_h_hist[s] = self._run_blocks(s + 1, code_embeds, cos_c, sin_c, code_mask)
                h_code = stage_h_hist[s]
                n_blocks_now = h_code.shape[1]

                if n_blocks_now < 1:
                    # stage still fully inactive -- a hard BREAK, matching forward()'s own
                    # "if n_blocks<1: break" exactly: a deeper stage can never be active while
                    # this one isn't (its codes are derived FROM this stage's own h_code), so
                    # there is nothing further to accumulate downstream this step either (an
                    # earlier version used `continue` here, letting a later stage's backlog
                    # prematurely accumulate this stage's not-yet-final input -- double-counted
                    # once this stage later caught up, confirmed via direct logit comparison
                    # against _generate_cascade, chat 2026-08-22).
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
        """Real incremental KV cache: the byte-level self.blocks self-attention and each fuse
        stage's post-cross-attn refinement self.blocks pass are cached across steps (O(1) new
        attention work per new byte, vs generate_no_cache's full O(L) recompute). The short
        code-sequence self-attention (kvlm) pass and the fuse cross-attention itself are still
        recomputed fresh whenever a new code appears (every Ks[s] bytes) -- cheap, since those
        sequences are short (~L/prod(Ks[:s+1])), not worth incrementally caching. Produces the
        exact same argmax choices as generate_no_cache, just asymptotically cheaper for long
        generations (see check_kv_cache_consistency for the direct comparison)."""
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
                              return_stats: bool = False):
        """MTP-style speculative decoding: draft cfg.mtp_heads bytes per round from ONE forward
        pass's final hidden state (the SAME extra_heads trained in forward() -- cheap, no extra
        attention-stack cost per drafted position, unlike the pruned query_vec mechanism, see
        qcute.bytelm_queryvec for that preserved lineage), then VERIFY each drafted byte one at a
        time against the real, exact generate_kv_cache incremental stepper
        (_make_incremental_stepper) -- accept while the draft agrees with the model's own true
        greedy choice, and at the first disagreement use the model's own correct byte instead and
        discard the rest of that round's draft (standard accept/reject-to-first-divergence).
        Verification is always exact -- it's the same stepper generate_kv_cache uses, never an
        approximation of it. Requires cfg.mtp_heads > 1 and a checkpoint trained with it (otherwise
        extra_heads are untrained noise and every round rejects at position 0, degenerating to
        ordinary one-byte-at-a-time generation). Assumes batch size 1 (matches every other
        generate_* method's own effective assumption). Returns just the sequence, or (sequence,
        stats) with stats={"accept_rate", "n_draft_checks"} if return_stats=True."""
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
        n_accepted, n_checked = 0, 0

        while all_bytes.shape[1] < target_len:
            m = all_bytes.shape[1]
            draft_len = min(cfg.mtp_heads, target_len - m)

            # --- draft: extra_heads, ONE forward pass over the committed prefix, no per-slot
            # attention-stack cost -- head i (0-indexed) predicts offset i+2, so the immediate
            # next byte (offset+1, drafted position 0) comes from the SAME final hidden state
            # via the ordinary head0/cond readout already computed by _generate_cascade.
            cond_logits_full, _, final_h = self._generate_cascade(all_bytes)
            draft_bytes = [cond_logits_full[:, -1, :].argmax(-1, keepdim=True)]
            last_h = final_h[:, -1:, :]
            for i in range(draft_len - 1):
                if i >= len(self.extra_heads):
                    break
                logits_i = F.linear(last_h, self.extra_heads[i].weight)
                draft_bytes.append(logits_i[:, -1, :].argmax(-1, keepdim=True))
            draft_bytes = torch.cat(draft_bytes, dim=1)  # (Bsz, <=draft_len)
            draft_len = draft_bytes.shape[1]

            # --- verify: one real position at a time, against the SAME exact incremental stepper ---
            for i in range(draft_len):
                n_checked += 1
                real_byte = next_logits.argmax(-1, keepdim=True)   # the verifier's true choice
                draft_byte = draft_bytes[:, i:i + 1]
                agree = torch.equal(real_byte, draft_byte)
                accepted_byte = draft_byte if agree else real_byte
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
        if return_stats:
            return seq, {"accept_rate": n_accepted / max(1, n_checked), "n_draft_checks": n_checked}
        return seq

    @torch.no_grad()
    def check_kv_cache_consistency(self, val_data: torch.Tensor, device: str,
                                    n_checks: int = 3, prompt_len: int = 8, n_new_bytes: int = 24) -> dict:
        """Diagnostic: generate_no_cache vs generate_kv_cache MUST produce bit-exact identical
        greedy trajectories -- generate_kv_cache is a pure efficiency reformulation of the same
        computation, not an approximation. Checks n_checks random prompts sampled from val_data at
        varying lengths (short prompts specifically exercise the "stage not yet active" backlog
        path -- this is exactly where a real bug was caught and fixed, chat 2026-08-22). Returns
        {"match_rate": float, "n_checks": int} -- match_rate should always be 1.0; anything less
        means the two paths have desynced and needs debugging before trusting generate_kv_cache."""
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


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------

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
    p.add_argument("--n_kv_heads", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--qk_norm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--rope_preset", type=str, default="qwen3", choices=list(ROPE_PRESETS))
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
    p.add_argument("--mtp_heads_code", type=int, default=1)
    p.add_argument("--mtp_weight_code", type=float, default=1.0)
    p.add_argument("--mtp_heads_uncond", type=int, default=1)
    p.add_argument("--mtp_weight_uncond", type=float, default=1.0)
    p.add_argument("--weight_tie", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_lm", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_fuse", type=lambda x: x.lower() != "false", default=False)

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
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, qk_norm=args.qk_norm, head_dim=args.head_dim,
        mlp_mult=args.mlp_mult, rope_base=args.rope_base, rope_preset=args.rope_preset, context_len=args.context_len,
        attn_window=args.attn_window, fuse_window=args.fuse_window, input_preset=args.input_preset,
        gumbel_tau=args.gumbel_tau, code_hard=args.code_hard, code_sample=args.code_sample,
        code_ntp_weight=args.code_ntp_weight, cond_weight=args.cond_weight,
        mtp_heads=args.mtp_heads, mtp_weight=args.mtp_weight,
        mtp_heads_code=args.mtp_heads_code, mtp_weight_code=args.mtp_weight_code,
        mtp_heads_uncond=args.mtp_heads_uncond, mtp_weight_uncond=args.mtp_weight_uncond,
        weight_tie=args.weight_tie, share_lm=args.share_lm, share_fuse=args.share_fuse,
    )


def main() -> None:
    args, pre_args = build_argparser("qcute_zero: single-LM periodic-fusion architecture")
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
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
