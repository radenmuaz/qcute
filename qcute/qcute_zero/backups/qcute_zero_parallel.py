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

Default query for "what predicts a new position" is the ordinary previous-token hidden state (no
seed/BOS token at all, unlike qcute_v1) -- pure standard AR continuation, causal by construction.
`cfg.parallel_decode` (default False) is an OPTIONAL, separate mechanism: a single trainable query
vector, trained by predicting a WHOLE randomly-chosen Ks[0]-sized block in parallel (one random
block-aligned boundary per batch, all its bytes at once, cheap relative to a full per-position loss)
-- toward block-parallel local decode (predicting a block's bytes from strictly-prior codes only,
without needing their true sequential hidden states first). Every slot's cross-attn is masked to the
last GROUNDED position before the block (never any slot's own future position, matching the
causal invariant proven in docs/status.md), so this stays fully consistent with ordinary training --
not required for, or exercised by, the default training path.

Single file by design for now (explicitly asked: "make thing single file first refactor later") --
copies/adapts primitives from qcute_v1_common.py (Block/RoPE/Logger/data-loading/train-loop shapes)
rather than importing them, since this is meant to stay a separate, prunable lineage.

uv run python -m qcute.qcute_zero_parallel.qcute_zero_parallel --config configs/qcute_zero_parallel/ks21_overfit10k.py
uv run python -m qcute.qcute_zero_parallel.qcute_zero_parallel --config configs/qcute_zero_parallel/ks221_overfit10k.py
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
    (Q from x, K/V from a separate kv sequence)."""
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


class Block(nn.Module):
    """"block regular": self-attention + MLP. Shared (same weights) across the byte-level pass and
    every fuse stage's own code-sequence NTP pass -- this IS the "single LM" the whole design
    hinges on."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class FuseStage(nn.Module):
    """"block fuse": cross-attention + MLP, one instance per periodic-fusion stage, own weights
    throughout (no cross-stage sharing) -- including this stage's own final LayerNorm feeding its
    own cond NTP readout (logits via the shared tied embed weight, passed in). Cheap: called with
    the code sequence's length (L/cum_K), not the byte sequence's."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int):
        super().__init__()
        self.ln1 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.ModuleList([Attn(d_model, n_heads) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False), nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False))
            for _ in range(n_layers)])
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
    code_ntp_weight: float = 1.0             # weight for each fuse stage's own code-sequence NTP loss
    cond_weight: float = 1.0                 # weight for each stage's post-fusion byte NTP loss
    parallel_decode: bool = False            # trains a shared query vector to predict a WHOLE upcoming
    parallel_decode_weight: float = 1.0      # Ks[0]-sized block in parallel, off by default (see
                                              # docs/status.md's "parallel block decode brainstorm")
    parallel_decode_n_blocks: int = 1        # independently-sampled blocks trained per step, all
                                              # reusing the SAME code_kv_cache (folded into batch)


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
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(D)

        fuse_layers = cfg.fuse_n_layers if cfg.fuse_n_layers is not None else cfg.n_layers
        self.fuse_stages = nn.ModuleList(
            [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        if cfg.parallel_decode:
            self.query_vec = nn.Parameter(torch.zeros(D))

    def _run_blocks(self, x: torch.Tensor, cos, sin, attn_mask) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.ln_f(x)

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
        h = self._run_blocks(x0, cos_b, sin_b, byte_mask)

        uncond_logits = F.linear(h[:, :-1, :], self.embed.weight)
        uncond_loss = F.cross_entropy(uncond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
        uncond_acc = (uncond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()

        # --- cascade through fuse stages ---
        cur_h = h                # source hidden states to extract this stage's codes from
        x_cross = h              # running byte-level query stream, refined by each fuse stage
        cum_K = 1
        fuse_ntp_losses, fuse_ntp_accs = [], []
        cond_losses, cond_accs = [], []
        code_kv_cache = []       # (h_code_s, code_pos_abs, window) per stage, reused by parallel_decode

        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break

            # code extraction: same tied embed/output head bytes use, STE hard sample
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]
            code_logits = F.linear(code_h, self.embed.weight)
            onehot = gumbel_quantize(code_logits, cfg.gumbel_tau, cfg.code_hard, cfg.code_sample)
            code_embeds = onehot @ self.embed.weight
            code_ids = onehot.argmax(-1)

            # this stage's own code-sequence NTP pass -- SAME shared blocks, causal, unbounded
            # (short sequence: n_blocks = cur_len // K_s)
            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(code_embeds, cos_c, sin_c, code_mask)

            if n_blocks >= 2:
                code_ntp_logits = F.linear(h_code[:, :-1, :], self.embed.weight)
                code_ntp_loss = F.cross_entropy(code_ntp_logits.reshape(-1, V), code_ids[:, 1:].reshape(-1))
                code_ntp_acc = (code_ntp_logits.argmax(-1) == code_ids[:, 1:]).float().mean()
                fuse_ntp_losses += [code_ntp_loss]
                fuse_ntp_accs += [code_ntp_acc]

            # cross-attn: byte-level query stream attends into h_code, causal on CUMULATIVE
            # (absolute-byte) boundary, never this stage's local code-sequence index (chat
            # 2026-08-22: using the local index here would be the one way this becomes circular).
            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_q, sin_q = cos_b, sin_b
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
            x_cross = self.fuse_stages[s](x_cross, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
            # another pass through the SAME shared self-attn+MLP LM blocks before this stage's
            # own cond readout (and before the next stage's cross-attn query input) -- i.e. fuse
            # cross-attn+own-mlp -> shared self-attn/mlp -> this stage's own cond NTP head.
            x_cross = self._run_blocks(x_cross, cos_b, sin_b, byte_mask)
            cond_logits_full = self.fuse_stages[s].readout(x_cross, self.embed.weight)

            cond_logits = cond_logits_full[:, :-1, :]
            cond_loss = F.cross_entropy(cond_logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
            cond_acc = (cond_logits.argmax(-1) == byte_ids[:, 1:]).float().mean()
            cond_losses += [cond_loss]
            cond_accs += [cond_acc]

            code_kv_cache += [(h_code, code_pos_abs, window_s)]
            cur_h = h_code

        # --- optional: parallel-decode query vector, trained on `parallel_decode_n_blocks`
        # independently-sampled WHOLE Ks[0]-sized blocks per step, all reusing the SAME
        # code_kv_cache from this one forward pass (folded into an extended batch dim -- "Option B"
        # from the parallel-decode brainstorm: each sampled block becomes its own batch row, so
        # self-attn among a block's own K0 slots needs no new block-diagonal mask machinery, only
        # per-row RoPE positions and a per-row cross-attn clamp). Every slot's cross-attn is clamped
        # to ITS OWN block's last-grounded boundary -- never any slot's own future position -- so
        # this stays exactly consistent with the free-tier invariant (docs/status.md's
        # parallel-decode brainstorm section).
        parallel_decode_loss = None
        K0 = cfg.Ks[0] if cfg.Ks else None
        if cfg.parallel_decode and self.n_fuse > 0 and code_kv_cache and K0 and L >= 2 * K0:
            n_full_blocks = L // K0
            nb = max(1, min(cfg.parallel_decode_n_blocks, n_full_blocks - 1))
            bis = torch.randint(1, n_full_blocks, (nb,), device=device)      # bi>=1: prior block exists
            m_list = bis * K0                                                # (nb,)
            clamp_list = m_list - 1                                          # (nb,)
            offsets = torch.arange(K0, device=device)                       # (K0,) local slot index
            slot_pos_2d = m_list.view(nb, 1) + offsets.view(1, K0)          # (nb, K0) absolute positions

            # fold (real batch, sampled block) into one virtual batch axis Bv = B*nb, B-major/
            # nb-minor throughout, so orig_idx below stays consistent with every other expand
            Bv = B * nb
            orig_idx = torch.arange(B, device=device).view(B, 1).expand(B, nb).reshape(Bv)
            slot_pos_v = slot_pos_2d.unsqueeze(0).expand(B, nb, K0).reshape(Bv, K0)
            clamp_v = clamp_list.view(1, nb).expand(B, nb).reshape(Bv)       # (Bv,) one clamp per row
            targets = byte_ids[orig_idx.view(Bv, 1), slot_pos_v]            # (Bv, K0)

            cos_q1, sin_q1 = rope_cos_sin_for_positions(slot_pos_v, hd, cfg.rope_base, device)  # (Bv,K0,hd)
            self_mask_q = causal_mask(offsets, offsets, None)  # shared (1,1,K0,K0): rows are now
            # independent batch elements, so plain within-block causality needs no per-row variant
            xq = self.query_vec.view(1, 1, D).expand(Bv, K0, D)
            for s, (h_code, code_pos_abs, window_s) in enumerate(code_kv_cache):
                h_code_v = h_code.unsqueeze(1).expand(B, nb, *h_code.shape[1:]).reshape(Bv, *h_code.shape[1:])
                allow = code_pos_abs.view(1, -1) <= clamp_v.view(-1, 1)      # (Bv, n_codes_s)
                if window_s is not None:
                    allow = allow & ((clamp_v.view(-1, 1) - code_pos_abs.view(1, -1)) < window_s)
                mask_q = allow.view(Bv, 1, 1, -1)  # per-row clamp, broadcasts over K0 query rows & heads
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
                xq = self.fuse_stages[s](xq, h_code_v, cos_q1, sin_q1, cos_k, sin_k, mask_q)
                xq = self._run_blocks(xq, cos_q1, sin_q1, self_mask_q)
            pd_logits_full = self.fuse_stages[s].readout(xq, self.embed.weight)  # (Bv, K0, V)
            parallel_decode_loss = F.cross_entropy(pd_logits_full.reshape(-1, V), targets.reshape(-1))

        final_loss = cond_losses[-1] if cond_losses else uncond_loss
        final_acc = cond_accs[-1] if cond_accs else uncond_acc
        total_loss = (sum(cond_losses) * cfg.cond_weight if cond_losses else uncond_loss)
        if fuse_ntp_losses:
            total_loss = total_loss + cfg.code_ntp_weight * torch.stack(fuse_ntp_losses).sum()
        if parallel_decode_loss is not None:
            total_loss = total_loss + cfg.parallel_decode_weight * parallel_decode_loss

        metrics = {
            "loss": total_loss, "final_loss": final_loss, "byte_acc": final_acc,
            "uncond_loss": uncond_loss, "uncond_acc": uncond_acc,
            **{f"cond{s}_loss": l for s, l in enumerate(cond_losses)},
            **{f"cond{s}_acc": a for s, a in enumerate(cond_accs)},
            **{f"fuse{s}_ntp_loss": l for s, l in enumerate(fuse_ntp_losses)},
            **{f"fuse{s}_ntp_acc": a for s, a in enumerate(fuse_ntp_accs)},
        }
        if parallel_decode_loss is not None:
            metrics["parallel_decode_loss"] = parallel_decode_loss
            metrics["parallel_decode_acc"] = (pd_logits_full.argmax(-1) == targets).float().mean()
        return total_loss, metrics

    @torch.no_grad()
    def _generate_cascade(self, byte_ids: torch.Tensor) -> tuple:
        """Shared no-grad cascade for generation (full recompute, no incremental state): same
        computation as forward()'s cascade minus the loss terms. Returns (cond_logits_full,
        code_kv_cache) -- cond_logits_full is the final stage's full per-position logits (uncond
        fallback if n_fuse==0), code_kv_cache is the per-stage (h_code, code_pos_abs, window) list.
        Used by both _forward_next_byte_logits (byte-at-a-time) and generate_blockwise
        (block-at-a-time) so there is exactly one generation-time code path, not two drifting
        copies -- unlike qcute_v1's generate_no_cache/_stack_generate_blockwise split (see
        docs/status.md's 2026-08-21/22 generation-bug entry for why that split is risky)."""
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = D // cfg.n_heads
        device = byte_ids.device
        byte_pos = torch.arange(L, device=device)
        cos_b, sin_b = rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base, device)
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        h = self._run_blocks(x0, cos_b, sin_b, byte_mask)

        cur_h = h
        x_cross = h
        cum_K = 1
        cond_logits_full = F.linear(self.ln_f(h), self.embed.weight)  # uncond fallback if n_fuse==0
        code_kv_cache = []
        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]
            code_logits = F.linear(code_h, self.embed.weight)
            onehot = gumbel_quantize(code_logits, cfg.gumbel_tau, hard=True, sample=False)
            code_embeds = onehot @ self.embed.weight

            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(code_embeds, cos_c, sin_c, code_mask)

            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
            x_cross = self.fuse_stages[s](x_cross, h_code, cos_b, sin_b, cos_k, sin_k, fuse_mask)
            x_cross = self._run_blocks(x_cross, cos_b, sin_b, byte_mask)
            cond_logits_full = self.fuse_stages[s].readout(x_cross, self.embed.weight)
            code_kv_cache += [(h_code, code_pos_abs, window_s)]
            cur_h = h_code

        return cond_logits_full, code_kv_cache

    @torch.no_grad()
    def _forward_next_byte_logits(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Full recompute over the whole sequence so far, returns logits for the NEXT byte
        (position L, i.e. the last position's post-fusion prediction)."""
        cond_logits_full, _ = self._generate_cascade(byte_ids)
        return cond_logits_full[:, -1, :]

    @torch.no_grad()
    def generate_no_cache(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Byte-by-byte, full recompute each step -- correctness-first, matches qcute_v1's own
        current "not yet KV-cached" state (CLAUDE.md: "incrementally-correct (not yet KV-cached)
        generation"), same precedent. generate_kv_cache is aliased to this until real incremental
        caching is built -- the causal/static-shape design (chat 2026-08-22) is what makes that a
        future optimization, not a correctness fix."""
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

    generate_kv_cache = generate_no_cache

    @torch.no_grad()
    def generate_blockwise(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Free-tier block-parallel decode (docs/status.md's parallel-decode brainstorm, "free
        tier"): decides Ks[0] bytes per step via the trained query_vec instead of one byte at a
        time. Requires cfg.parallel_decode=True AND a checkpoint actually trained with it --
        otherwise query_vec is untrained noise and this will not produce coherent output. Reuses
        ONE prefix forward's code_kv_cache (via _generate_cascade) across all Ks[0] slots of the
        step -- the "kv cache reuse" this buys is per-BLOCK, not a real incremental cache across
        steps (the prefix itself is still fully recomputed each step, same as generate_no_cache --
        no true incremental KV cache exists yet, see that method's own docstring). No code
        drafting/accept-reject here -- that's the separate, unbuilt speculative tier."""
        cfg = self.cfg
        assert cfg.parallel_decode, "generate_blockwise requires a model trained with cfg.parallel_decode=True"
        K0 = cfg.Ks[0]
        D = cfg.d_model
        hd = D // cfg.n_heads

        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        target_len = prompt_bytes.shape[1] + n_new_bytes

        while all_bytes.shape[1] < target_len:
            m = all_bytes.shape[1]
            block_size = min(K0 - (m % K0), target_len - m)
            _, code_kv_cache = self._generate_cascade(all_bytes)

            slot_pos = torch.arange(m, m + block_size, device=device)
            cos_q, sin_q = rope_cos_sin_for_positions(slot_pos, hd, cfg.rope_base, device)
            self_mask = causal_mask(slot_pos, slot_pos, None)
            xq = self.query_vec.view(1, 1, D).expand(all_bytes.shape[0], block_size, D)

            if not code_kv_cache:  # n_fuse==0 -- no code KV at all, fall back to the uncond readout
                xq = self._run_blocks(xq, cos_q, sin_q, self_mask)
                logits = F.linear(self.ln_f(xq), self.embed.weight)
            else:
                clamp_pos_vec = torch.full((block_size,), m - 1, device=device)
                for s, (h_code, code_pos_abs, window_s) in enumerate(code_kv_cache):
                    mask_q = causal_mask(clamp_pos_vec, code_pos_abs, window_s)
                    cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)
                    xq = self.fuse_stages[s](xq, h_code, cos_q, sin_q, cos_k, sin_k, mask_q)
                    xq = self._run_blocks(xq, cos_q, sin_q, self_mask)
                logits = self.fuse_stages[s].readout(xq, self.embed.weight)

            new_bytes = logits.argmax(-1)  # (Bsz, block_size), all decided in one shot
            all_bytes = torch.cat([all_bytes, new_bytes], dim=1)

        if was_training:
            self.train()
        return all_bytes[0]


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
    p.add_argument("--parallel_decode", action="store_true", default=False)
    p.add_argument("--parallel_decode_weight", type=float, default=1.0)
    p.add_argument("--parallel_decode_n_blocks", type=int, default=1)

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
        parallel_decode=args.parallel_decode, parallel_decode_weight=args.parallel_decode_weight,
        parallel_decode_n_blocks=args.parallel_decode_n_blocks,
    )


def main() -> None:
    args, pre_args = build_argparser("qcute_zero: single-LM periodic-fusion architecture")
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config_from_args(args)
    model = QCuteZero(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_zero_parallel_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} n_fuse={model.n_fuse} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, cfg.input_preset, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
