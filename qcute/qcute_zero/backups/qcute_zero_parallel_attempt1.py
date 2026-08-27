"""qcute_zero: a monolithic, single-LM alternative to qcute_lagcodec's multi-encoder StackDecoder
lineage (see CLAUDE.md's Architecture section for qcute_lagcodec; this is a separate lineage, not a
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
semantics as qcute_lagcodec) -- each stage's codes are built FROM the previous stage's own contextualized
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

No curriculum needed by design (unlike qcute_lagcodec's max_srcs/curriculum_max_srcs hack): every fuse
stage's code source is the SAME shared, already-training backbone from step 1 (nothing is a fresh,
untouched, randomly-initialized module the way each qcute_lagcodec encoder level was), and the zero-sink
lets a stage's own freshly-initialized cross-attention weights learn to suppress themselves early
(put softmax weight on the sink) and gradually rely on real codes as those weights improve -- an
emergent, learned on-ramp instead of a hand-scheduled one. Expected, not yet proven -- the whole
point of the ks21/ks221-no-curriculum runs this file's plan calls for.

Query for "what predicts a new position" is the ordinary previous-token hidden state (no seed/BOS
token at all, unlike qcute_lagcodec) -- pure standard AR continuation, causal by construction.

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
copies/adapts primitives from qcute_lagcodec_common.py (Block/RoPE/Logger/data-loading/train-loop shapes)
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
# small shared utilities (copied/trimmed from qcute_lagcodec_common.py)
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


def wavefront_mask(timestep: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """(1,1,L,L) bool mask: j visible to i iff strictly-earlier timestep, or same timestep+region.
    Degenerates to causal_mask when every timestep is unique. See docs/status.md 2026-08-24."""
    earlier = timestep.view(-1, 1) > timestep.view(1, -1)
    same_step_same_region = ((timestep.view(-1, 1) == timestep.view(1, -1)) &
                              (region.view(-1, 1) == region.view(1, -1)))
    allow = earlier | same_step_same_region
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

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        """Incremental self-attention: x_new is only the NEW position(s) (Tn=1 per generation
        step, or the whole prompt on the priming call); cache is None (nothing yet) or (k_prev,
        v_prev) from earlier calls. Returns (out, new_cache) -- new_cache is trimmed to the last
        `window` entries when windowed, so a subsequent call only ever pays for what's visible.
        Mask uses LOCAL (call-relative) positions -- only relative order matters for causality,
        and cos/sin (computed from true absolute positions by the caller) is what actually encodes
        real distance, so this stays exactly consistent with the full-recompute path."""
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
    """"block regular": self-attention + MLP. Shared (same weights) across the byte-level pass and
    every fuse stage's own code-sequence NTP pass -- this IS the "single LM" the whole design
    hinges on."""
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
    """"block fuse": cross-attention + MLP, one instance per periodic-fusion stage, own weights
    throughout (no cross-stage sharing) -- including this stage's own final LayerNorm feeding its
    own cond NTP readout (logits via the shared tied embed weight, passed in). Cheap: called with
    the code sequence's length (L/cum_K), not the byte sequence's."""
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
    """Pluggable code representation for every fuse stage's own code (ported from
    qcute_lagcodec_common.py's SimplexQuant, quant_type="simplex" only for now -- adapted to this file's
    own module shapes: dedicated code_head/code_embed/code_predict instead of a stage_lm object's
    attributes). Categorical code via gumbel-softmax STE, product-quantized: pq_chunks independent
    vocab-way softmaxes concatenated (total code width = vocab*pq_chunks), combinatorial capacity
    vocab**pq_chunks. Constraint: vocab**pq_chunks >= unit_vocab (2**input_preset -- must be able
    to represent at least one real trunk unit, byte/nibble/bit/whatever input_preset is). Default
    vocab=256, pq_chunks=1 (with unit_vocab=256, i.e. byte trunk) is functionally the original
    single 256-way softmax, now with its own dedicated weights rather than literally reusing
    self.embed/self.head -- UNLESS global_tie requests otherwise (see below).

    global_tie (2026-08-24, `qcute_zero_simple`'s original design as a special case of this more
    general architecture): only representable as a literal shared-tensor tie when vocab==unit_vocab
    and pq_chunks==1 (width==unit_vocab exactly, matching self.embed/self.head's own shape) -- then
    code_head/code_predict directly reference tied_head_weight and code_embed multiplies directly
    against tied_embed_weight (matmul, not nn.Linear -- see _embed() below), exactly like the
    original pre-quantizer-refactor code (`onehot @ self.embed.weight`). For any other config
    (pq_chunks>1 or vocab!=unit_vocab), a literal tensor tie is impossible (code_head's per-chunk
    concatenated logit space and a flat unit_vocab-wide embed table are different shapes/objects,
    and solving code_embed's single linear layer to reproduce all unit_vocab embed rows via a
    sparse pq-chunk one-hot is a generally underdetermined system when chunks' index-ranges
    overlap across different unit values) -- global_tie is a no-op there: only the reserved-index
    *convention* holds (the first unit_vocab combinatorial ids, per to_ids()'s place-value order,
    are nominally "real unit values"; ids >= unit_vocab are free/bonus codes with no unit meaning),
    as documentation/bookkeeping, not an enforced numerical correspondence -- training decides
    what the reserved sub-range actually ends up encoding."""
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
        """Same as extract() but always hard=True/sample=False -- generation-time greedy code
        extraction, regardless of cfg.code_hard/code_sample (matches the old gumbel_quantize(...,
        hard=True, sample=False) calls every generate_* method used at code-extraction sites)."""
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

    def sample_next(self, h_query: torch.Tensor) -> tuple:
        """For generation: greedy per-chunk argmax over code_predict's logits -> STE one-hot ->
        code_embed. Used wherever generation needs to sample the NEXT code from the code-sequence
        LM's own hidden state (not extracted from real bytes) -- e.g. generate_free_rollout."""
        logits = self._chunked(self.code_predict(h_query))
        onehot = gumbel_quantize(logits, self.tau, hard=True, sample=False).reshape(*h_query.shape[:-1], self.width)
        return onehot, self._embed(onehot)


# ----------------------------------------------------------------------------
# Config + model
# ----------------------------------------------------------------------------

@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32, 1)       # same semantics as qcute_lagcodec: cumulative periods, last
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
    mtp_heads: int = 1                       # extra byte-ahead heads reading the SAME final hidden
                                              # state (post-cascade), MTP-style (see qcute.bytelm) --
                                              # 1 = disabled (only the existing head0 next-byte
                                              # prediction). >1 heads predict t+2..t+mtp_heads.
    mtp_weight: float = 1.0                  # weight for the extra heads' mean loss
    mtp_heads_code: int = 1                  # extra code-ahead heads reading the code-sequence
                                              # LM's own hidden state (h_code), predicting further
                                              # future codes (1 = disabled). Per-stage by default,
                                              # like the quantizer itself (share_lm=True shares both).
    mtp_weight_code: float = 1.0
    mtp_heads_uncond: int = 1                # extra byte-ahead heads reading the BASE trunk
                                              # hidden state h (before any fuse cross-attention --
                                              # the cheap/coarse "uncond" signal), predicting
                                              # future bytes (1 = disabled).
    mtp_weight_uncond: float = 1.0
    weight_tie: bool = False                 # default untied: byte output head is its own
                                              # nn.Linear(D, V, bias=False), separate params from
                                              # self.embed. True: head.weight literally refs
                                              # embed.weight (shared tensor, not a copy). "local"
                                              # tie -- level 0 only.
    global_tie: bool = False                 # requires weight_tie=True. Extends the tie to every
                                              # level's Quantizer too (qcute_zero_simple's original
                                              # one-shared-table design, as a special case of this
                                              # more general architecture) -- exact/literal only
                                              # when that stage's vocab==unit_vocab and pq_chunks==1
                                              # (e.g. v256pq1 for a byte trunk); otherwise a no-op
                                              # beyond the reserved-index convention (see Quantizer's
                                              # own docstring) -- no numerical weight forcing, since
                                              # that system is underdetermined once pq_chunks>1.
    seed_query_p: float = 0.0                # fraction of positions using the sparse context-free
                                              # seed-token auxiliary loss (0=disabled); see docs/status.md.
    seed_query_weight: float = 1.0
    wavefront_weight: float = 0.0            # training-time wavefront_mask loss (0=disabled), see
                                              # docs/status.md 2026-08-24 for the full design/rationale.
    wavefront_K: int = 8
    wavefront_n_waves: int = 2
    blocklocal_seed_weight: float = 0.0       # stack_local-style block-diagonal decode, shift-by-1
                                              # (0=disabled); see docs/status.md 2026-08-24.
    blocklocal_dual_mode: bool = True         # stage s>0: train both mask/rollout modes if True,
                                              # mask only if False; see docs/status.md 2026-08-24.
    blocklocal_glat_p: float = 0.0            # per-position prob of a GLAT-style second local-decode
                                              # pass with STE self-predicted bytes swapped in (0=off);
                                              # see docs/status.md 2026-08-24.
    share_lm: bool = False                   # default unshared: each level gets its own Block
                                              # stack; True ties every level to the same one (ALBERT-style).
    head_word_bits: int | None = None        # None = same as input_preset. Decouples output-head
                                              # granularity from trunk word size; see WordHead/docs/status.md.


class WordHead(nn.Module):
    """Pluggable output-head granularity, decoupled from the trunk's own per-position word size
    (unit_bits = input_preset). word_bits==unit_bits: plain linear head, unchanged. word_bits>
    unit_bits (coarser): predicts group_size=word_bits//unit_bits FUTURE unit positions jointly as
    one combined classification (place-value combination, same convention as Quantizer.to_ids'
    PQ combination). word_bits<unit_bits (finer): factorizes one position's own unit word into
    n_sub=unit_bits//word_bits independent sub-word chunks -- literally Quantizer's own PQ
    pattern, reused directly for the byte-output head instead of the code head."""
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
        """h_last: (B, D) or (B, 1, D) final hidden state -> (B, group_size) next unit ids
        (group_size==1 in the equal/finer cases -- a single next id, reconstructed via factorized
        argmax when n_sub>1, exactly like Quantizer.to_ids)."""
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

        # per-level LM stacks: level 0 = byte pass (+ every fuse stage's post-cross-attn
        # refinement pass), level s+1 = fuse stage s's own code-sequence NTP pass. Unshared by
        # default (each level its own independent Block stack); share_lm=True makes every entry
        # literally the same module instance (nn.ModuleList dedupes params by identity).
        n_lms = self.n_fuse + 1
        if cfg.share_lm:
            first = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
            self.lms = nn.ModuleList([first] * n_lms)
        else:
            self.lms = nn.ModuleList(
                [nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
                 for _ in range(n_lms)])
        # per-level final norm -- previously a single shared self.ln_f regardless of share_lm, an
        # oversight relative to per-level independence (each level's own blocks got independence,
        # its final norm didn't); now follows the exact same share_lm-controlled pattern as lms.
        if cfg.share_lm:
            first_ln = RMSNorm(D)
            self.ln_fs = nn.ModuleList([first_ln] * n_lms)
        else:
            self.ln_fs = nn.ModuleList([RMSNorm(D) for _ in range(n_lms)])

        # byte-output head: untied (own nn.Linear) by default; weight_tie=True makes it literally
        # reference self.embed.weight (shared tensor -- PyTorch's .parameters() dedupes by
        # identity, so this doesn't double-count params). Code extraction/code-NTP have their own
        # dedicated weights entirely separate from self.embed -- see Quantizer's code_head/
        # code_predict/code_embed below, wired in during the quantizer port.
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
        self.fuse_stages = nn.ModuleList(
            [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        self.extra_heads = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads - 1))])
        self.extra_heads_uncond = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads_uncond - 1))])

        self.seed_embed = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.seed_embed, std=0.02)

        # per-stage quantizer + code-MTP heads -- previously one global instance shared across every
        # fuse stage regardless of share_lm, the same oversight as ln_f above; now follows the same
        # share_lm-controlled pattern as lms/ln_fs.
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

        # uncond-level-0 MTP heads: cheap/coarse extra byte-ahead heads reading the BASE trunk
        # hidden state h, before any fuse cross-attention -- same pattern as the byte-level extra
        # heads below, just off a cheaper/earlier hidden state.
        uncond_mtp_losses, uncond_mtp_accs = [], []
        for i, head_u in enumerate(self.extra_heads_uncond):
            k = i + 2
            if L <= k:
                continue
            logits_u = head_u(h[:, :-k, :])
            targets_u = byte_ids[:, k:]
            uncond_mtp_losses.append(F.cross_entropy(logits_u.reshape(-1, V), targets_u.reshape(-1)))
            uncond_mtp_accs.append((logits_u.argmax(-1) == targets_u).float().mean())

        # training-time wavefront loss: tiles wavefront_mask across the whole sequence (one
        # K-block per tile). See docs/status.md 2026-08-24.
        wavefront_loss = h.new_zeros(())
        wavefront_acc = h.new_zeros(())
        if cfg.wavefront_weight > 0:
            Kw, n_waves = cfg.wavefront_K, cfg.wavefront_n_waves
            assert Kw % n_waves == 0, f"wavefront_K ({Kw}) must be evenly divisible by wavefront_n_waves ({n_waves})"
            region_len = Kw // n_waves
            n_full_blocks = L // Kw
            if region_len > 1 and n_full_blocks > 0:
                Lc = n_full_blocks * Kw
                wpos = torch.arange(Lc, device=device)
                local_i = wpos % Kw
                w_region = local_i // region_len
                local_j = local_i % region_len
                w_timestep = (wpos // Kw) * region_len + local_j
                w_mask = wavefront_mask(w_timestep, w_region)
                cos_w, sin_w = rope_cos_sin_for_positions(wpos, hd, cfg.rope_base, device)
                h_wave = self._run_blocks(0, self.embed(byte_ids[:, :Lc]), cos_w, sin_w, w_mask)
                logits_wave = self.head(h_wave)
                valid = local_j < (region_len - 1)         # exclude each region's own last local step
                target_idx = torch.clamp(wpos + 1, max=Lc - 1)
                target_wave = byte_ids[:, target_idx]
                logits_v = logits_wave[:, valid, :]
                target_v = target_wave[:, valid]
                wavefront_loss = F.cross_entropy(logits_v.reshape(-1, V), target_v.reshape(-1))
                wavefront_acc = (logits_v.argmax(-1) == target_v).float().mean()

        # --- cascade through fuse stages ---
        cur_h = h                # source hidden states to extract this stage's codes from
        x_cross = h              # running byte-level query stream, refined by each fuse stage
        cum_K = 1
        fuse_ntp_losses, fuse_ntp_accs = [], []
        cond_losses, cond_accs = [], []
        seed_losses, seed_accs = [], []
        code_mtp_losses, code_mtp_accs = {}, {}   # keyed by (stage, k) -- see below for why
        code_kv_cache = []       # (h_code_s, code_pos_abs, window) per stage
        cum_Ks_list = []         # cum_K AT each stage (byte span of one stage-s code)
        x_cross_pre_stage = []   # x_cross value entering stage s, BEFORE stage s's own cross-attn
                                  # -- i.e. reflects real ground-truth refinement from stages 0..s-1
                                  # only, never stage s's own code (blocklocal_seed_weight's "rollout"
                                  # mode input, see below)

        for s in range(self.n_fuse):
            x_cross_pre_stage.append(x_cross)
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cum_Ks_list.append(cum_K)
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

            # code-level MTP heads: extra code-ahead heads reading h_code, predicting further
            # future codes (offset i+2..) -- this stage's own set (per-stage, like the quantizer).
            for i, head_c in enumerate(self.extra_heads_code_per_stage[s]):
                k = i + 2
                if n_blocks <= k:
                    continue
                logits_c = quantizer._chunked(head_c(h_code[:, :-k, :])).reshape(-1, quantizer.vocab)
                target_c = quantizer._chunked(onehot[:, k:, :]).argmax(-1).reshape(-1)
                code_mtp_losses[(s, k)] = F.cross_entropy(logits_c, target_c)
                code_mtp_accs[(s, k)] = (logits_c.argmax(-1) == target_c).float().mean()

            # cross-attn: byte-level query stream attends into h_code, causal on CUMULATIVE
            # (absolute-byte) boundary, never this stage's local code-sequence index (chat
            # 2026-08-22: using the local index here would be the one way this becomes circular).
            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_q, sin_q = cos_b, sin_b
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)

            # sparse any-timestep seed-token auxiliary loss: sample a subset of positions, replace
            # their query with the learned seed embedding (no real byte context at all -- not even
            # self-attention), cross-attend to the SAME code KV/mask a real query at that position
            # would see, and predict that position's OWN byte (not p+1 -- there is no "from"
            # context here, this is a cold-start prediction of p given only the code).
            if cfg.seed_query_p > 0:
                n_sample = max(1, int(round(L * cfg.seed_query_p)))
                sel = torch.randperm(L, device=device)[:n_sample].sort().values
                seed_q = self.seed_embed.view(1, 1, D).expand(B, n_sample, D)
                cos_sq, sin_sq = cos_b[sel], sin_b[sel]
                mask_sq = fuse_mask[:, :, sel, :]
                seed_out = self.fuse_stages[s](seed_q, h_code, cos_sq, sin_sq, cos_k, sin_k, mask_sq)
                seed_logits = self.fuse_stages[s].readout(seed_out, self.head.weight)
                seed_targets = byte_ids[:, sel]
                seed_loss = F.cross_entropy(seed_logits.reshape(-1, V), seed_targets.reshape(-1))
                seed_acc = (seed_logits.argmax(-1) == seed_targets).float().mean()
                seed_losses += [seed_loss]
                seed_accs += [seed_acc]

            x_cross = self.fuse_stages[s](x_cross, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
            # another pass through the SAME shared self-attn+MLP LM blocks before this stage's
            # own cond readout (and before the next stage's cross-attn query input) -- i.e. fuse
            # cross-attn+own-mlp -> shared self-attn/mlp -> this stage's own cond NTP head.
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)

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

            code_kv_cache += [(h_code, code_pos_abs, window_s)]
            cur_h = h_code

        # Stack_local-style block-diagonal decode per fuse stage, seed cross-attends to code_{b-1}
        # (shift-by-1, exact/causal). mask_lower selects raw-embed vs x_cross_pre_stage self-attn
        # input for stage s>0. See docs/status.md 2026-08-24 for the full design/rationale.
        blocklocal_losses, blocklocal_accs = {}, {}   # keyed by (s, mask_lower) -- both modes trained
        blocklocal_seed_accs, blocklocal_local_accs = {}, {}
        blocklocal_glat_accs = {}
        if cfg.blocklocal_seed_weight > 0:
            for s in range(self.n_fuse):
                if s >= len(code_kv_cache):
                    break
                cum_K_s = cum_Ks_list[s]
                n_blocks_s = L // cum_K_s
                if n_blocks_s < 2:      # need >=2 blocks: block 0 has no code_{-1} to shift to
                    continue
                h_code_s, code_pos_abs_s, _window_s = code_kv_cache[s]
                Lc_s = n_blocks_s * cum_K_s
                seed_emb = self.seed_embed.view(1, 1, D).expand(B * n_blocks_s, 1, D)
                block_start_pos = torch.arange(n_blocks_s, device=device) * cum_K_s  # real byte position of each block's own start

                # s==0: both modes are identical (no lower level to mask/not-mask), run once
                modes = [True, False] if (s > 0 and cfg.blocklocal_dual_mode) else [True]
                for mask_lower in modes:
                    base = self.embed(byte_ids[:, :Lc_s]) if (mask_lower or s == 0) else x_cross_pre_stage[s][:, :Lc_s]
                    base_blocks = base.reshape(B * n_blocks_s, cum_K_s, D)
                    block_seq = torch.cat([seed_emb, base_blocks], dim=1)   # (B*n_blocks_s, cum_K_s+1, D)
                    local_pos = torch.arange(cum_K_s + 1, device=device)
                    cos_l, sin_l = rope_cos_sin_for_positions(local_pos, hd, cfg.rope_base, device)
                    local_mask = causal_mask(local_pos, local_pos, None)   # block-local causal, batched -- never crosses blocks
                    h_local = self._run_blocks(0, block_seq, cos_l, sin_l, local_mask)
                    h_local = h_local.view(B, n_blocks_s, cum_K_s + 1, D)

                    seed_h = h_local[:, 1:, 0, :]                # (B, n_blocks_s-1, D) -- blocks 1..n_blocks_s-1's seeds
                    # cross-attend block b's seed to code_{b-1} (SHIFTED, offset 1) + a window of
                    # further-back codes, controlled by attn_window
                    cos_q, sin_q = rope_cos_sin_for_positions(block_start_pos[1:], hd, cfg.rope_base, device)
                    cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs_s[:-1], hd, cfg.rope_base, device)
                    own_mask = causal_mask(block_start_pos[1:], code_pos_abs_s[:-1], cfg.attn_window)
                    seed_cross = self.fuse_stages[s](seed_h, h_code_s[:, :-1, :], cos_q, sin_q, cos_k, sin_k, own_mask)
                    seed_logits = self.fuse_stages[s].readout(seed_cross, self.head.weight)
                    seed_targets = byte_ids[:, cum_K_s:Lc_s:cum_K_s]        # blocks 1..n_blocks_s-1's own FIRST byte
                    seed_loss = F.cross_entropy(seed_logits.reshape(-1, V), seed_targets.reshape(-1))
                    seed_acc = (seed_logits.argmax(-1) == seed_targets).float().mean()

                    # rest of the block: block-local NTP (local step i+1 from step i), no cross-attn
                    if cum_K_s >= 2:
                        local_logits = self.head(h_local[:, :, 1:cum_K_s, :])
                        local_targets = byte_ids[:, :Lc_s].view(B, n_blocks_s, cum_K_s)[:, :, 1:]
                        local_loss = F.cross_entropy(local_logits.reshape(-1, V), local_targets.reshape(-1))
                        local_acc = (local_logits.argmax(-1) == local_targets).float().mean()
                    else:
                        local_loss, local_acc = h.new_zeros(()), h.new_zeros(())

                    # GLAT-style second pass: swap in STE self-predicted bytes (from local_logits
                    # above) at each within-block position w.p. blocklocal_glat_p, rerun the same
                    # block-local decode, add its loss unconditionally (additive, not skip-real --
                    # see docs/status.md 2026-08-24, mirrors encoder_ste_p's more-stable variant).
                    if cfg.blocklocal_glat_p > 0 and cum_K_s >= 2 and torch.is_grad_enabled():
                        pred_idx = local_logits.argmax(-1)                       # (B, n_blocks_s, cum_K_s-1)
                        pred_hard = self.embed(pred_idx)
                        pred_soft = F.softmax(local_logits, dim=-1) @ self.embed.weight
                        pred_ste = pred_hard.detach() + pred_soft - pred_soft.detach()
                        swap_mask = (torch.rand(pred_idx.shape, device=device) < cfg.blocklocal_glat_p).unsqueeze(-1)
                        base_view = base_blocks.view(B, n_blocks_s, cum_K_s, D)
                        swapped_tail = torch.where(swap_mask, pred_ste, base_view[:, :, 1:, :])
                        base_blocks_glat = torch.cat([base_view[:, :, :1, :], swapped_tail], dim=2).reshape(B * n_blocks_s, cum_K_s, D)
                        block_seq_glat = torch.cat([seed_emb, base_blocks_glat], dim=1)
                        h_local_glat = self._run_blocks(0, block_seq_glat, cos_l, sin_l, local_mask)
                        h_local_glat = h_local_glat.view(B, n_blocks_s, cum_K_s + 1, D)
                        local_logits_glat = self.head(h_local_glat[:, :, 1:cum_K_s, :])
                        local_loss_glat = F.cross_entropy(local_logits_glat.reshape(-1, V), local_targets.reshape(-1))
                        local_acc_glat = (local_logits_glat.argmax(-1) == local_targets).float().mean()
                    else:
                        local_loss_glat, local_acc_glat = h.new_zeros(()), h.new_zeros(())

                    blocklocal_losses[(s, mask_lower)] = seed_loss + local_loss + local_loss_glat
                    blocklocal_accs[(s, mask_lower)] = (seed_acc + local_acc) / 2
                    blocklocal_seed_accs[(s, mask_lower)] = seed_acc
                    blocklocal_glat_accs[(s, mask_lower)] = local_acc_glat
                    blocklocal_local_accs[(s, mask_lower)] = local_acc

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
        if seed_losses:
            total_loss = total_loss + cfg.seed_query_weight * torch.stack(seed_losses).sum()
        if code_mtp_losses:
            total_loss = total_loss + cfg.mtp_weight_code * torch.stack(list(code_mtp_losses.values())).mean()
        if uncond_mtp_losses:
            total_loss = total_loss + cfg.mtp_weight_uncond * torch.stack(uncond_mtp_losses).mean()
        if cfg.wavefront_weight > 0:
            total_loss = total_loss + cfg.wavefront_weight * wavefront_loss
        if blocklocal_losses:
            total_loss = total_loss + cfg.blocklocal_seed_weight * torch.stack(list(blocklocal_losses.values())).mean()

        metrics = {
            "loss": total_loss, "final_loss": final_loss, "byte_acc": final_acc,
            "uncond_loss": uncond_loss, "uncond_acc": uncond_acc,
            **{f"cond{s}_loss": l for s, l in enumerate(cond_losses)},
            **{f"cond{s}_acc": a for s, a in enumerate(cond_accs)},
            **{f"seed{s}_loss": l for s, l in enumerate(seed_losses)},
            **{f"seed{s}_acc": a for s, a in enumerate(seed_accs)},
            **{f"fuse{s}_ntp_loss": l for s, l in enumerate(fuse_ntp_losses)},
            **{f"fuse{s}_ntp_acc": a for s, a in enumerate(fuse_ntp_accs)},
            **{f"mtp{i+2}_loss": l for i, l in enumerate(mtp_losses)},
            **{f"mtp{i+2}_acc": a for i, a in enumerate(mtp_accs)},
            **{f"mtp{k}_code{s}_loss": l for (s, k), l in code_mtp_losses.items()},
            **{f"mtp{k}_code{s}_acc": a for (s, k), a in code_mtp_accs.items()},
            **{f"mtp{i+2}_uncond_loss": l for i, l in enumerate(uncond_mtp_losses)},
            **{f"mtp{i+2}_uncond_acc": a for i, a in enumerate(uncond_mtp_accs)},
            "wavefront_loss": wavefront_loss, "wavefront_acc": wavefront_acc,
            **{(f"blocklocal{s}_loss" if ml else f"blocklocal{s}_rollout_loss"): l
               for (s, ml), l in blocklocal_losses.items()},
            **{(f"blocklocal{s}_acc" if ml else f"blocklocal{s}_rollout_acc"): a
               for (s, ml), a in blocklocal_accs.items()},
            **{(f"blocklocal{s}_seed_acc" if ml else f"blocklocal{s}_rollout_seed_acc"): a
               for (s, ml), a in blocklocal_seed_accs.items()},
            **{(f"blocklocal{s}_local_acc" if ml else f"blocklocal{s}_rollout_local_acc"): a
               for (s, ml), a in blocklocal_local_accs.items()},
            **{(f"blocklocal{s}_glat_acc" if ml else f"blocklocal{s}_rollout_glat_acc"): a
               for (s, ml), a in blocklocal_glat_accs.items()},
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
        generation-time code path, not two drifting copies -- unlike qcute_lagcodec's
        generate_no_cache/_stack_generate_blockwise split (see docs/status.md's 2026-08-21/22
        generation-bug entry for why that split is risky)."""
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
    def _generate_cascade_early_exit(self, byte_ids: torch.Tensor, confidence_threshold: float) -> tuple:
        """Same cascade as _generate_cascade, stops early once a stage's cond prediction clears
        confidence_threshold. NO exactness guarantee, unlike generate_speculative/wavefront_mtp.
        Returns (logits_at_last_pos, exit_stage); -1=uncond. See docs/status.md 2026-08-24."""
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
        logits_last = self.head(h[:, -1:, :])
        exit_stage = -1
        if F.softmax(logits_last, dim=-1).max(-1).values.min() >= confidence_threshold:
            return logits_last[:, 0, :], exit_stage

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
            logits_last = self.fuse_stages[s].readout(x_cross[:, -1:, :], self.head.weight)
            exit_stage = s
            is_last_stage = (s == self.n_fuse - 1)
            if is_last_stage or F.softmax(logits_last, dim=-1).max(-1).values.min() >= confidence_threshold:
                break
            cur_h = h_code

        return logits_last[:, 0, :], exit_stage

    @torch.no_grad()
    def generate_early_exit(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                             confidence_threshold: float = 0.9, return_stats: bool = False,
                             verbose: bool = False) -> torch.Tensor:
        """Full-recompute generation using _generate_cascade_early_exit every step. Not
        guaranteed to match generate_no_cache -- compare directly to measure agreement."""
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        exit_counts = {}
        for _ in range(n_new_bytes):
            logits, exit_stage = self._generate_cascade_early_exit(all_bytes, confidence_threshold)
            next_byte = logits.argmax(-1, keepdim=True)
            exit_counts[exit_stage] = exit_counts.get(exit_stage, 0) + 1
            all_bytes = torch.cat([all_bytes, next_byte], dim=1)
            if verbose:
                print(f"    pos {all_bytes.shape[1]-1}: exit_stage={exit_stage} byte={_fmt_bytes(next_byte[0])!r}")
        if was_training:
            self.train()
        seq = all_bytes[0]
        if verbose or return_stats:
            n_stages_avail = self.n_fuse
            print(f"[early_exit summary] exit_stage histogram={exit_counts} "
                  f"(-1=uncond, 0..{n_stages_avail-1}=stage index, {n_stages_avail-1}=ran full cascade)")
        if return_stats:
            return seq, {"exit_counts": exit_counts}
        return seq

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
        """Block-local decode: seed cross-attends to code_{b-1} (shift-by-1, exact/causal), then
        block-local self-attention for the rest. Needs blocklocal_seed_weight>0 training,
        n_fuse==1, >=2*K0 prompt bytes. See docs/status.md 2026-08-24."""
        assert self.n_fuse == 1, "generate_free_rollout PoC only supports a single fuse stage (n_fuse==1)"
        cfg = self.cfg
        D = cfg.d_model
        hd = D // cfg.n_heads
        device_t = torch.device(device)
        was_training = self.training
        self.eval()
        K = cfg.Ks[0]

        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes[:, :prompt_bytes.shape[1] // K * K]
        assert all_bytes.shape[1] >= 2 * K, "generate_free_rollout (shift-by-1) needs at least 2*K0 real prompt bytes"
        Bsz = all_bytes.shape[0]

        # one-off pass to extract the prompt's own real codes (not a persistent cache)
        L = all_bytes.shape[1]
        pos0 = torch.arange(L, device=device_t)
        cos0, sin0 = rope_cos_sin_for_positions(pos0, hd, cfg.rope_base, device_t)
        h0 = self._run_blocks(0, self.embed(all_bytes), cos0, sin0, causal_mask(pos0, pos0, cfg.attn_window))
        n_blocks_prev = L // K
        code_h = h0[:, K - 1::K, :][:, :n_blocks_prev, :]
        _, code_embeds_past = self.quantizers[0].extract_greedy(code_h)

        # global byte-level cache: NOT used for decoding, only to keep code extraction on the
        # same global-hidden-state convention the code-level LM was trained on
        byte_caches = [None] * cfg.n_layers
        h_g = self.embed(all_bytes)
        for l, block in enumerate(self.lms[0]):
            h_g, byte_caches[l] = block.forward_incremental(h_g, cos0, sin0, byte_caches[l], cfg.attn_window)

        # code-level cache; hc_hist/cpos_hist keep every position's output (attn_window may reach
        # back further than one block, so the last one alone isn't enough)
        code_caches = [None] * cfg.n_layers
        cpos0 = torch.arange(n_blocks_prev, device=device_t)
        ccos0, csin0 = rope_cos_sin_for_positions(cpos0, hd, cfg.rope_base, device_t)
        hc = code_embeds_past
        for l, block in enumerate(self.lms[1]):
            hc, code_caches[l] = block.forward_incremental(hc, ccos0, csin0, code_caches[l], None)
        hc_hist = self.ln_fs[1](hc)                          # (B, n_blocks_prev, D)
        cpos_hist = cpos0 * K + (K - 1)                       # code_pos_abs convention, (n_blocks_prev,)

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            # own-block seed: cross-attend to code_{b-1} (shifted, real), block-local self-attn
            local_caches = [None] * cfg.n_layers
            block_start_pos = torch.tensor([n_blocks_prev * K], device=device_t)
            local_pos = torch.tensor([0], device=device_t)
            cos_l, sin_l = rope_cos_sin_for_positions(local_pos, hd, cfg.rope_base, device_t)
            x_seed = self.seed_embed.view(1, 1, D).expand(Bsz, 1, D)
            for l, block in enumerate(self.lms[0]):
                x_seed, local_caches[l] = block.forward_incremental(x_seed, cos_l, sin_l, local_caches[l], cfg.attn_window)
            seed_h = self.ln_fs[0](x_seed)

            cos_q, sin_q = rope_cos_sin_for_positions(block_start_pos, hd, cfg.rope_base, device_t)
            cos_k, sin_k = rope_cos_sin_for_positions(cpos_hist, hd, cfg.rope_base, device_t)
            own_mask = causal_mask(block_start_pos, cpos_hist, cfg.attn_window)
            seed_cross = self.fuse_stages[0](seed_h, hc_hist, cos_q, sin_q, cos_k, sin_k, own_mask)
            first_byte_logits = self.fuse_stages[0].readout(seed_cross, self.head.weight)
            first_byte = first_byte_logits[:, 0, :].argmax(-1, keepdim=True)

            # commit the first byte into the local cache (local position 1)
            pos_fb = torch.tensor([1], device=device_t)
            cos_fb, sin_fb = rope_cos_sin_for_positions(pos_fb, hd, cfg.rope_base, device_t)
            x_fb = self.embed(first_byte)
            for l, block in enumerate(self.lms[0]):
                x_fb, local_caches[l] = block.forward_incremental(x_fb, cos_fb, sin_fb, local_caches[l], cfg.attn_window)
            h_last = self.ln_fs[0](x_fb)
            all_bytes = torch.cat([all_bytes, first_byte], dim=1)

            # remaining K-1 bytes: block-local self-attention only, no cross-attention
            for t in range(1, K):
                next_byte = self.head(h_last).argmax(-1)
                all_bytes = torch.cat([all_bytes, next_byte], dim=1)
                pos_t = torch.tensor([t + 1], device=device_t)
                cos_t, sin_t = rope_cos_sin_for_positions(pos_t, hd, cfg.rope_base, device_t)
                x_t = self.embed(next_byte)
                for l, block in enumerate(self.lms[0]):
                    x_t, local_caches[l] = block.forward_incremental(x_t, cos_t, sin_t, local_caches[l], cfg.attn_window)
                h_last = self.ln_fs[0](x_t)

            # block now real: feed bytes through the global cache, extract code, grow code history
            new_block_bytes = all_bytes[:, -K:]
            pos_g = torch.arange(L, L + K, device=device_t)
            cos_g, sin_g = rope_cos_sin_for_positions(pos_g, hd, cfg.rope_base, device_t)
            x_g = self.embed(new_block_bytes)
            for l, block in enumerate(self.lms[0]):
                x_g, byte_caches[l] = block.forward_incremental(x_g, cos_g, sin_g, byte_caches[l], cfg.attn_window)
            h_g_last = self.ln_fs[0](x_g)[:, -1:, :]
            L += K
            n_blocks_prev += 1
            _, code_embed_new = self.quantizers[0].extract_greedy(h_g_last)
            cpos_new = torch.tensor([n_blocks_prev - 1], device=device_t)
            ccos_new, csin_new = rope_cos_sin_for_positions(cpos_new, hd, cfg.rope_base, device_t)
            hc_new = code_embed_new
            for l, block in enumerate(self.lms[1]):
                hc_new, code_caches[l] = block.forward_incremental(hc_new, ccos_new, csin_new, code_caches[l], None)
            hc_hist = torch.cat([hc_hist, self.ln_fs[1](hc_new)], dim=1)   # grow the cross-attn KV history
            cpos_hist = torch.cat([cpos_hist, cpos_new * K + (K - 1)], dim=0)

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            self.train()
        return all_bytes[0]

    @torch.no_grad()
    def _wavefront_draft_block(self, all_bytes: torch.Tensor, K: int, n_waves: int,
                                device_t: torch.device) -> torch.Tensor:
        """Shared by generate_wavefront/generate_wavefront_mtp: drafts K bytes via lockstep
        wavefront decode (MTP bootstrap + region_len-1 lockstep passes). buf is already in true
        left-to-right byte order, no reordering needed. See docs/status.md 2026-08-24."""
        cfg = self.cfg
        assert K % n_waves == 0, f"K ({K}) must be evenly divisible by n_waves ({n_waves})"
        region_len = K // n_waves
        max_offset = (n_waves - 1) * region_len + 1
        assert cfg.mtp_heads >= max_offset, (
            f"wavefront draft needs cfg.mtp_heads >= (n_waves-1)*region_len+1 = {max_offset} "
            f"to bootstrap the last wave's first token (got mtp_heads={cfg.mtp_heads})")
        hd = cfg.d_model // cfg.n_heads
        P = all_bytes.shape[1]

        # --- bootstrap every wave's first token from h_last (position P-1), no seed token ---
        pos_prefix = torch.arange(P, device=device_t)
        cos_p, sin_p = rope_cos_sin_for_positions(pos_prefix, hd, cfg.rope_base, device_t)
        h_prefix = self._run_blocks(0, self.embed(all_bytes), cos_p, sin_p,
                                    causal_mask(pos_prefix, pos_prefix, cfg.attn_window))
        h_last = h_prefix[:, -1:, :]                              # (B, 1, D)
        wave_firsts = [self.head(h_last).argmax(-1)]              # wave 0: ordinary head0
        for g in range(1, n_waves):
            offset = g * region_len + 1
            head_g = self.extra_heads[offset - 2]
            wave_firsts.append(F.linear(h_last, head_g.weight).argmax(-1))
        # buf layout: prefix ++ wave0's region_len slots ++ wave1's ++ ... ++ wave(n_waves-1)'s
        buf = torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], K)], dim=1)
        for g in range(n_waves):
            buf[:, P + g * region_len] = wave_firsts[g][:, 0]

        # --- logical clock: prefix keeps its own raw index; every wave shares the SAME
        # region_len-long timestep sequence (P, P+1, ..., P+region_len-1) -- lockstep ---
        per_wave_timestep = P + torch.arange(region_len, device=device_t)
        timestep = torch.cat([pos_prefix, per_wave_timestep.repeat(n_waves)])
        region = torch.cat([pos_prefix.new_full((P,), -1),
                             torch.arange(n_waves, device=device_t).repeat_interleave(region_len)])

        # --- lockstep decode: tau=2..region_len, every wave's tau-th token simultaneously ---
        for tau in range(2, region_len + 1):
            cos_t, sin_t = rope_cos_sin_for_positions(timestep, hd, cfg.rope_base, device_t)
            mask = wavefront_mask(timestep, region)
            h = self._run_blocks(0, self.embed(buf), cos_t, sin_t, mask)
            for g in range(n_waves):
                prev_idx = P + g * region_len + (tau - 1) - 1     # wave g's (tau-1)-th position
                next_byte = self.head(h[:, prev_idx:prev_idx + 1, :]).argmax(-1)
                buf[:, P + g * region_len + tau - 1] = next_byte[:, 0]

        return buf[:, P:P + K]

    @torch.no_grad()
    def generate_wavefront(self, prompt_bytes: torch.Tensor, K: int, n_waves: int, n_new_bytes: int,
                            device: str) -> torch.Tensor:
        """Byte-level-only, full-recompute, UNVERIFIED wavefront draft (see generate_wavefront_mtp
        for the verified counterpart). Degenerates to ordinary AR at n_waves==1."""
        device_t = torch.device(device)
        was_training = self.training
        self.eval()

        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes

        n_new_blocks = -(-n_new_bytes // K)
        for _ in range(n_new_blocks):
            draft_block = self._wavefront_draft_block(all_bytes, K, n_waves, device_t)
            all_bytes = torch.cat([all_bytes, draft_block], dim=1)

        all_bytes = all_bytes[:, :prompt_bytes.shape[1] + n_new_bytes]
        if was_training:
            self.train()
        return all_bytes[0]

    @torch.no_grad()
    def check_wavefront_consistency(self, val_data: torch.Tensor, device: str, n_checks: int = 3,
                                     prompt_len: int = 8, K: int = 8, n_new_bytes: int = 16) -> dict:
        """Diagnostic: generate_wavefront(n_waves=1) MUST match generate_no_cache bit-exactly --
        n_waves=1 has no two positions sharing a timestep, so wavefront_mask degenerates to plain
        causal_mask and the whole mechanism collapses to ordinary AR generation. Anything less
        than match_rate=1.0 means the timestep/region/mask arithmetic has a real bug."""
        was_training = self.training
        self.eval()
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            start = torch.randint(0, max(1, val_data.shape[0] - pl - n_new_bytes), (1,)).item()
            prompt = val_data[start:start + pl].to(device)
            out_full = self.generate_no_cache(prompt, n_new_bytes, device)
            out_wave = self.generate_wavefront(prompt, K, 1, n_new_bytes, device)
            if torch.equal(out_full, out_wave):
                n_match += 1
        if was_training:
            self.train()
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}

    @torch.no_grad()
    def generate_wavefront_mtp(self, prompt_bytes: torch.Tensor, K: int, n_waves: int, n_new_bytes: int,
                                device: str, return_stats: bool = False, verbose: bool = False):
        """Wavefront-drafted speculative decode: drafts a K-byte block via _wavefront_draft_block,
        verifies byte-by-byte against the exact incremental stepper (same guarantee as
        generate_speculative). Degenerates to plain MTP draft at region_len==1. n_fuse==0 only."""
        cfg = self.cfg
        assert self.n_fuse == 0, "generate_wavefront_mtp only supports byte-level-only configs (n_fuse==0) so far"
        device_t = torch.device(device)
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)

        step = self._make_incremental_stepper(prompt_bytes.shape[0], device_t)
        all_bytes = prompt_bytes
        logits_all = step(all_bytes, 0)          # prime the verifier with the prompt
        next_logits = logits_all[:, -1, :]

        target_len = prompt_bytes.shape[1] + n_new_bytes
        n_accepted, n_checked, n_rounds, n_draft_passes = 0, 0, 0, 0
        region_len = K // n_waves       # passes per round: 1 bootstrap + (region_len-1) lockstep

        while all_bytes.shape[1] < target_len:
            remaining = target_len - all_bytes.shape[1]
            draft_block = self._wavefront_draft_block(all_bytes, K, n_waves, device_t)
            draft_block = draft_block[:, :min(K, remaining)]
            n_rounds += 1
            n_draft_passes += region_len
            if verbose:
                print(f"[wavefront-mtp round {n_rounds}] draft_passes={region_len} "
                      f"(region_len={region_len}) draft={_fmt_bytes(draft_block[0])!r}")

            # --- verify: one real position at a time, in true order, against the SAME exact
            # incremental stepper generate_kv_cache/generate_speculative use ---
            for i in range(draft_block.shape[1]):
                n_checked += 1
                real_byte = next_logits.argmax(-1, keepdim=True)
                draft_byte = draft_block[:, i:i + 1]
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
        stats = {"accept_rate": n_accepted / max(1, n_checked), "n_draft_checks": n_checked,
                  "n_rounds": n_rounds, "n_draft_passes": n_draft_passes}
        if verbose:
            print(f"[wavefront-mtp summary] rounds={n_rounds} draft_passes={n_draft_passes} "
                  f"verify_checks={n_checked} accepted={n_accepted} accept_rate={stats['accept_rate']:.3f}")
        if return_stats:
            return all_bytes[0], stats
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
        hd = D // cfg.n_heads

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
                    _, code_embeds = self.quantizers[s].extract_greedy(code_h)
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
                              return_stats: bool = False, verbose: bool = False):
        """MTP-style speculative decoding: draft mtp_heads bytes from one forward pass, verify
        each against the exact incremental stepper (accept/reject-to-first-divergence). Requires
        mtp_heads>1 and a checkpoint trained with it. Batch size 1 only."""
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

            # --- draft: extra_heads, ONE forward pass over the committed prefix, no per-slot
            # attention-stack cost -- head i (0-indexed) predicts offset i+2, so the immediate
            # next byte (offset+1, drafted position 0) comes from the SAME final hidden state
            # via the ordinary head0/cond readout already computed by _generate_cascade.
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

            if args.eval_decode_mtp_verify and model.cfg.mtp_heads > 1:
                # MTP-drafted speculative decode, verified byte-by-byte against the exact NTP
                # stepper (generate_speculative already does this -- see its own docstring) --
                # show both NTP (generate_no_cache, the ground-truth reference) and the MTP-
                # verified decode side by side, plus the accept_rate, so MTP draft quality is
                # visible during training, not just at a separate manual generation step. The two
                # texts should be IDENTICAL (verification guarantees this) -- shown together as a
                # direct check of that guarantee, not just trusted from the accept_rate number.
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
    p.add_argument("--seed_query_p", type=float, default=0.0)
    p.add_argument("--seed_query_weight", type=float, default=1.0)
    p.add_argument("--wavefront_weight", type=float, default=0.0)
    p.add_argument("--wavefront_K", type=int, default=8)
    p.add_argument("--wavefront_n_waves", type=int, default=2)
    p.add_argument("--blocklocal_seed_weight", type=float, default=0.0)
    p.add_argument("--blocklocal_dual_mode", type=bool, default=True)
    p.add_argument("--blocklocal_glat_p", type=float, default=0.0)
    p.add_argument("--quant_type", type=str, default="simplex")
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--pq_chunks", type=int, default=1)
    p.add_argument("--share_lm", type=lambda x: x.lower() != "false", default=False)
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
        seed_query_p=args.seed_query_p, seed_query_weight=args.seed_query_weight,
        wavefront_weight=args.wavefront_weight, wavefront_K=args.wavefront_K,
        wavefront_n_waves=args.wavefront_n_waves,
        blocklocal_seed_weight=args.blocklocal_seed_weight,
        blocklocal_dual_mode=args.blocklocal_dual_mode,
        blocklocal_glat_p=args.blocklocal_glat_p,
        quant_type=args.quant_type, vocab=args.vocab, pq_chunks=args.pq_chunks,
        share_lm=args.share_lm,
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
