"""qcute.qcutelm_pyramid — single shared LM over a flat multi-resolution
sequence, BSQ-quantized codes inserted FIFO-style instead of forecast-
substituted (session: "do not separate E D codelm, just single lm, like
qcute fifo... the whole point of bsq is bsq is informative enough to give
compressed context").

This supersedes an earlier draft of this file (separate shared E/D/codelm
stacks with forecast-substitution, ported from qcutelm_vlt11) — replaced
after working through a real circularity in that design: `D_i` needed
"this block's own code" available AT the block's own start, which can't
exist yet (a summary of content not yet processed), hence v11's
`codelm`/forecast machinery to approximate it. A literal single causal LM
hits the same wall UNLESS code computation is decoupled from the main
attention pass — which is exactly how `qcute_fifo` avoids it: a span's
embedding there is NOT an output of causal attention over that span, it's
a fixed, cheap, non-attentional function (`linear(concat(children))`),
computable the instant the span's raw content is known, independent of
whether the main LM has "reached" that point yet.

Architecture:

1. **Local merge** (replaces `qcute_fifo`'s fixed linear-merge rule with a
   learned, data-driven one): level `i`'s code is
   `quantize(Linear(Ks[i] * d_model, code_dim)(concat of Ks[i] consecutive
   child embeddings)))`, then re-embedded to `d_model`
   (`code_in_embed[i]`). Purely local, non-causal WITHIN the block (a
   block, once complete, is entirely in the past for anyone using its
   summary — no leakage risk from attending across the block's own
   children), and parallel across blocks — no attention, no dependency on
   any other block, computed once per level in one shot.
2. **Flat sequence construction**: every level's embeddings (byte
   embeddings for level 0, code embeddings for levels 1..n_levels) are
   scattered into ONE flat sequence, each code token placed immediately
   after the span it summarizes (`_build_flat_layout`, precomputed once
   at __init__ — purely structural, depends only on `Ks`/`context_len`,
   not on data). This placement is what makes a PLAIN causal mask
   correct: a block's own code token always sits strictly after that
   block in flat order, so standard causal attention can never let a
   position see its own not-yet-complete summary, and CAN see any
   earlier-completed block's summary — verified by hand for `Ks=(4,)`,
   `L0=8` in this session (byte 4 sees code₀ correctly; byte 7 does not
   yet see code₁, which summarizes byte 7 itself).
3. **One shared causal LM** (the "single LM" — no E/D split, no
   `codelm`) runs ONE attention pass over the whole flat sequence.
   Level-0 positions are gathered back out (in original byte order) and
   the ONLY loss is ordinary byte-level NTP — no `code_match_loss`, no
   `e_ntp_weight`/cyclic-target-problem machinery, since codes are
   deterministic functions of already-known content, not autoregressive
   forecasts competing for gradient with anything.

Efficiency: `n_levels` cheap linear merges (negligible cost, no
attention) + ONE attention pass over the flat sequence, replacing the
earlier draft's (and `qcutelm_vlt11`'s) `3 * n_levels` separate attention
stacks per step — this is the pyramid efficiency goal from
`docs/vlt12_math.tex` achieved through a simpler mechanism (fixed local
merge + single global attention) than that doc's masked multi-level
attention scheme, made possible specifically by not needing forecasting
at all.

Not yet built: any generation/inference code (this file is training-only,
same convention as v11's byte_only/identity diagnostics this session);
`untie_levels=False` (default, sharing the merge/code_in_embed weights
across levels) requires all `Ks[i]` equal (documented at the assertion
site) since a shared `Linear(Ks[i]*d, code_dim)` isn't well-typed across
levels with different block sizes.

No shared imports with qcutelm_vlt2/.../vlt11 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers/quantizers/Block/
BitPredictHead duplicated verbatim from qcutelm_vlt11.py.

    uv run python -m qcute.qcutelm_pyramid --config configs/qcutelm_pyramid_<name>.py
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
    Ks: tuple[int, ...] = (4, 4, 4)   # per-level local merge factor. untie_levels=False requires all
                                        # entries equal (see model __init__'s assertion).
    vocab: int = 256           # byte vocabulary size — always 256, kept as a field (rather than a
                                # hardcoded literal) only because fifo_v1 mode's byte_emb/pred_head
                                # construction wants it explicitly, matching qcute_fifo.py's own Config.
    d_model: int = 256         # single shared width — bytes, codes, and the one main LM all live here
    code_dim: int | None = None   # None (default): code_dim = d_model (shared space, no projection
                                    # mismatch). int: codes live in their own space, bridged via a
                                    # plain linear map (code_in_embed), same "just add linear map" idea
                                    # as the earlier E/D/codelm draft's Config.code_dim.
    context_len: int = 1024
    quant_type: str = "ifsq"   # bsq/fsq/ifsq/identity — same 4-way choice and same quantize()/
                                # code_level_loss mechanism as qcutelm_vlt11, reused verbatim.
    fsq_levels: int = 8
    n_heads: int = 4
    n_layers: int = 4
    mlp_mult: int = 4
    attn_window: int = -1      # applies to the FLAT sequence length (must divide it) — dense (-1) by
                                # default here since the flat-sequence windowing story hasn't been
                                # worked out yet (unlike v11's per-level windows, a block-crossing
                                # window here could straddle a code-token insertion in a way that's not
                                # yet analyzed for correctness) — opt in deliberately, not the default.
    untie_levels: bool = False   # False (default): ONE shared `merge`/`code_in_embed` pair, reused at
                                # every level — requires all Ks[i] equal (see __init__). True:
                                # n_levels independent copies (Ks[i] may differ freely then). The main
                                # LM itself has no such flag — it is UNCONDITIONALLY single/shared,
                                # that's the whole point of this file (session: "just single lm").
    rope_base: float = 10000.0
    bit_head_mode: str = "chain"   # byte-prediction head's mode only (head0) — same BitPredictHead
                                    # mechanism as v11, unrelated to the merge/quantize mechanism above.
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True
    byte_only: bool = False    # True: skip all merging/insertion entirely — the main LM processes ONLY
                                # the raw byte sequence, no code tokens at all. Same diagnostic purpose
                                # as v11's Config.byte_only (isolate the base LM's own capacity), much
                                # cheaper to express here since there's no separate D to fall back to —
                                # it's just "don't build the flat sequence, use embed_0 directly."
    mode: str = "pyramid"      # "pyramid" (default): this file's own design above (deterministic flat
                                # multi-resolution sequence, BSQ-quantized merge, byte NTP only). "fifo_
                                # v1": a faithful port of the now-retired qcute_fifo.py's ACTUAL v1
                                # mechanism (session: "merge the current qcute_fifo greedy merge
                                # algorithm as flag then delete") — composition-SAMPLED window (one
                                # random valid non-increasing bandwidth sequence per step, not the
                                # deterministic full pyramid), UNQUANTIZED recursive linear merge (no
                                # BSQ/FSQ at all — qcute_fifo's own docstring: "no BSQ/FSQ anywhere in
                                # this file"), and EVERY slot (not just level-0/byte positions) predicts
                                # its own `bandwidth` upcoming bytes via a Fetch-style byte-chain MTP
                                # head, teacher-forced. The fields below (window/bandwidths/fetch_*/
                                # tie_head) are only consulted in this mode; Ks/d_model/code_dim/quant_
                                # type/etc. are only consulted in "pyramid" mode.
    window: int = 32           # fifo_v1 mode only — FIFO slot budget (sequence length in SLOT units,
                                # not bytes; a composition's achievable byte span is window*max(bandwidths))
    bandwidths: tuple[int, ...] = (1, 2)   # fifo_v1 mode only — allowed span sizes, each a power of 2
    fetch_n_heads: int = 2      # fifo_v1 mode only — FetchHead's own small self-attention module
    fetch_gamma: float = 1.0    # fifo_v1 mode only — Fetch's h_t^(j) = gamma*h_t + Emb(x_{t+j}) scaling
    tie_head: bool = True       # fifo_v1 mode only — prediction head weight-tied to the byte embedding


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


def rope_cos_sin_explicit(position_ids: torch.Tensor, head_dim: int, base: float):
    """Like rope_cos_sin but for EXPLICIT (non-contiguous) integer
    positions — fifo_v1 mode only, ported from qcute_fifo.py's own
    rope_cos_sin_at: a slot's RoPE position must be its own raw-byte END
    offset (so relative distances stay meaningful across slots of
    different bandwidth), not just its index 0..window-1."""
    device = position_ids.device
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def enumerate_compositions(window: int, bandwidths: tuple[int, ...]) -> list[tuple[int, ...]]:
    """fifo_v1 mode only — ported verbatim from qcute_fifo.py. All non-
    increasing length-`window` sequences using values from `bandwidths`
    (coarsest/oldest first, finest/newest last) — the discrete, tractable
    set of window "shapes" the real FIFO+merge cascade could ever
    produce. e.g. enumerate_compositions(4, (1, 2)) ->
        [(1,1,1,1), (2,1,1,1), (2,2,1,1), (2,2,2,1), (2,2,2,2)]"""
    bandwidths = tuple(sorted(bandwidths))
    n_levels = len(bandwidths)

    def counts_gen(remaining: int, levels_left: int):
        if levels_left == 1:
            yield (remaining,)
            return
        for c in range(remaining + 1):
            for rest in counts_gen(remaining - c, levels_left - 1):
                yield (c,) + rest

    comps = []
    for counts in counts_gen(window, n_levels):
        comp: list[int] = []
        for level in range(n_levels - 1, -1, -1):
            comp.extend([bandwidths[level]] * counts[level])
        comps.append(tuple(comp))
    return comps


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
        self._warned_dense_fallback = False
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
            if self.window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={self.window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window — falling back to DENSE attention (no windowing "
                      f"savings) for this layer. Only warns once per layer instance.")
                self._warned_dense_fallback = True
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
    """[*] long byte ids (0..255) -> [*, 8] float, each bit in {-1,+1}/sqrt(8)."""
    bits = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    return (2 * bits - 1) / math.sqrt(8)


def bits_to_byte(bits: torch.Tensor) -> torch.Tensor:
    """[*, 8] float -> [*] long byte ids. Inverse of byte_to_bits."""
    b = (bits > 0).long()
    powers = (2 ** torch.arange(8, device=bits.device))
    return (b * powers).sum(-1)


class SmallSelfAttn(nn.Module):
    """Manual multi-head self-attention (batch_first, no RoPE — chain-position
    info here comes from additive embeddings instead) via F.scaled_dot_product_attention.
    Replaces nn.MultiheadAttention in BitPredictHead/FetchHead: found this session to have
    severe MPS memory pathology — 8.82 GiB allocated and an out-of-memory crash after just 2
    forward calls at batch=16, d_model=256 (worked fine at v11's d_model=96, never stress-
    tested at this width before). Same math/semantics as nn.MultiheadAttention's default
    (single in-proj + out-proj, 1/sqrt(head_dim) scaling), matching this file's own
    CausalSelfAttention's existing preference for manual QKV+SDPA over the nn.MultiheadAttention
    module."""

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


class BitPredictHead(nn.Module):
    """Verbatim port of qcutelm_vlt11.BitPredictHead — see that class's
    docstring for the full chain-mode/fixed-kernel rationale. Used here
    only for head0 (byte prediction) — the merge/quantize mechanism above
    doesn't use this class at all (no chain-of-bits prediction needed for
    a deterministic merge).

    KNOWN MPS ISSUE (this session): mode="chain"'s BACKWARD pass becomes
    pathologically slow (~12s+ per call, vs ~0.6s in isolation on a fresh
    leaf tensor) once chained through this file's full upstream computation
    graph (main attention stack + merges + scatter/gather) — root cause not
    isolated (suspected MPS backward-kernel behavior specific to this
    class's SmallSelfAttn + embedding-concatenation pattern under a long
    upstream graph). mode="independent" (plain Linear, no chain) does not
    have this problem — verified stable ~0.8-1.3s/step end-to-end at this
    file's default config scale. Use "independent" here until diagnosed;
    qcutelm_vlt11.py's own use of this same class (at narrower d_model, no
    equally long upstream graph) has not shown this issue."""

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


class FetchHead(nn.Module):
    """fifo_v1 mode only — ported verbatim from qcute_fifo.py. Given a
    slot's hidden state h and its own bandwidth, predicts that many
    upcoming BYTES (not bits — contrast BitPredictHead above) via the
    Fetch mechanism: a single shared self-attention hop combining h with
    gamma*h + Emb(previous TRUE/teacher-forced byte in the chain), then
    h + that attention output feeds the (optionally tied) prediction
    head. Every slot goes through this same path uniformly, including
    bandwidth-1 slots (a length-1 chain) — qcute_fifo's "always mtp"."""

    def __init__(self, cfg: Config, byte_emb: nn.Embedding, pred_head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = byte_emb
        self.pred_head = pred_head
        self.max_chain = max(cfg.bandwidths)
        self.chain_pos_emb = nn.Embedding(self.max_chain, cfg.d_model)
        self.self_attn = SmallSelfAttn(cfg.d_model, cfg.fetch_n_heads)

    def forward(self, h: torch.Tensor, bandwidth: int, target_bytes: torch.Tensor | None) -> torch.Tensor:
        """h: [N, D]. target_bytes: [N, bandwidth] true next bytes, teacher-forced. -> logits [N, bandwidth, vocab]."""
        N, D = h.shape
        chain_vecs = [h + self.chain_pos_emb.weight[0]]
        logits_list = []
        for j in range(bandwidth):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self.self_attn(x)
            fetched = h + attn_out[:, -1, :]
            logits_list.append(self.pred_head(fetched))
            if j < bandwidth - 1:
                next_byte = target_bytes[:, j]
                chain_vecs.append(self.cfg.fetch_gamma * h + self.byte_emb(next_byte) + self.chain_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


def _build_flat_layout(Ks: tuple[int, ...], L0: int) -> tuple[list[int], list[int]]:
    """Pure-Python, one-time (per Config) structural computation — no
    tensors, no data dependency. Returns (level_id, orig_idx), both
    length flat_len: level_id[p] = which level flat position p belongs to
    (0 = raw byte, 1..len(Ks) = code level i, matching Ks[i-1]'s block
    factor); orig_idx[p] = that level's own index at flat position p.
    Construction: start with level-0 positions in order; at each level
    transition, walk the FULL accumulated sequence so far (which already
    contains every earlier level's tokens interleaved) and, counting ONLY
    occurrences of the immediately-previous level (ignoring anything
    older interspersed among them), insert one new level-`lvl` token
    immediately after every Ks[lvl-1]-th such occurrence. This placement
    (new token strictly after everything it summarizes) is what makes a
    plain causal mask correct with zero special-casing — see module
    docstring. (Grouping the whole accumulated sequence instead of just
    the previous level's own elements was a real bug caught by this
    session's own smoke test — level 2 must group level 1's 64 elements
    into blocks of 4, NOT every 4 elements of the already-320-long
    level-0+level-1 interleaved sequence.)"""
    flat = [(0, i) for i in range(L0)]
    for lvl in range(1, len(Ks) + 1):
        K = Ks[lvl - 1]
        prev_count = sum(1 for (l, _) in flat if l == lvl - 1)
        assert prev_count % K == 0, f"Ks={Ks} must evenly divide level {lvl-1}'s own sequence length ({prev_count})"
        new_flat = []
        seen = 0
        block_idx = 0
        for item in flat:
            new_flat.append(item)
            if item[0] == lvl - 1:
                seen += 1
                if seen % K == 0:
                    new_flat.append((lvl, block_idx))
                    block_idx += 1
        flat = new_flat
    level_id = [x[0] for x in flat]
    orig_idx = [x[1] for x in flat]
    return level_id, orig_idx


class FlatPyramidLM(nn.Module):
    """Single shared causal LM over a flat multi-resolution sequence — see
    module docstring for the full construction and why it's causally
    correct with a plain mask."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.mode in ("pyramid", "fifo_v1")
        assert cfg.d_model % cfg.n_heads == 0, f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        if cfg.mode == "fifo_v1":
            self._init_fifo_v1()
            return
        self.n_levels = len(cfg.Ks)
        self.code_dim = cfg.code_dim if cfg.code_dim is not None else cfg.d_model
        self.n_shared = self.n_levels if cfg.untie_levels else 1
        if not cfg.untie_levels:
            assert len(set(cfg.Ks)) == 1, (
                f"untie_levels=False requires all Ks entries equal (a shared Linear(K*d, code_dim) "
                f"isn't well-typed across levels with different block sizes) — got Ks={cfg.Ks}. "
                f"Use untie_levels=True for per-level Ks."
            )

        level_id, orig_idx = _build_flat_layout(cfg.Ks, cfg.context_len)
        self.flat_len = len(level_id)
        # per-level flat positions, in original order (monotonic by construction — insertions never
        # reorder existing elements) — used to scatter each level's embeddings into the flat sequence
        # and to gather level 0's hidden states back out for the NTP loss.
        flat_pos_of: list[list[int]] = [[] for _ in range(self.n_levels + 1)]
        for p, lvl in enumerate(level_id):
            flat_pos_of[lvl].append(p)
        for lvl in range(self.n_levels + 1):
            self.register_buffer(f"flat_pos_of_{lvl}", torch.tensor(flat_pos_of[lvl], dtype=torch.long), persistent=False)

        window = None if cfg.attn_window == -1 else cfg.attn_window
        if window is not None:
            assert self.flat_len % window == 0, f"attn_window ({window}) must divide flat_len ({self.flat_len})"

        assert cfg.bit_head_mode in ("independent", "chain")
        self.byte_embed_in = nn.Linear(8, cfg.d_model)

        # local merge: level i's Ks[i] children (d_model-dim) -> code_dim pre-quantize value. Shared
        # (n_shared=1) requires uniform Ks (asserted above); untied gives one merge per level.
        self.merge = nn.ModuleList([nn.Linear(cfg.Ks[i % len(cfg.Ks)] * cfg.d_model, self.code_dim) for i in range(self.n_shared)]) \
            if cfg.untie_levels else nn.ModuleList([nn.Linear(cfg.Ks[0] * cfg.d_model, self.code_dim)])
        self.code_in_embed = nn.ModuleList([nn.Linear(self.code_dim, cfg.d_model) for _ in range(self.n_shared)])

        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.head0 = BitPredictHead(cfg.d_model, 8, cfg.bit_head_mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def _init_fifo_v1(self) -> None:
        """fifo_v1 mode — ported from qcute_fifo.py's CombinatorialLM.__init__. ONE shared byte
        embedding table doubles as the (optionally tied) prediction head's output weight — "one shared
        vocabulary, just recursively pooled" (no BSQ/FSQ, unlike pyramid mode)."""
        cfg = self.cfg
        assert all(b & (b - 1) == 0 for b in cfg.bandwidths), "bandwidths must all be powers of 2 (binary merge tree)"
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
        self.merge_proj = nn.ModuleDict({str(b): nn.Linear(2 * cfg.d_model, cfg.d_model) for b in cfg.bandwidths if b > 1})
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window=None) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        vocab = cfg.vocab
        self.pred_head = nn.Linear(cfg.d_model, vocab)
        if cfg.tie_head:
            self.pred_head.weight = self.byte_emb.weight
        self.fetch_head = FetchHead(cfg, self.byte_emb, self.pred_head)

    def embed_span(self, byte_span: torch.Tensor, bandwidth: int) -> torch.Tensor:
        """fifo_v1 mode only — recursive PQ-like binary merge, ported from qcute_fifo.py verbatim
        (same formula pyramid mode's `_merge_level` uses, minus the quantize() step — this is
        deliberately the "FIFO's own fixed merge rule, unquantized" reference point)."""
        if bandwidth == 1:
            return self.byte_emb(byte_span[:, 0])
        half = bandwidth // 2
        left = self.embed_span(byte_span[:, :half], half)
        right = self.embed_span(byte_span[:, half:], half)
        return self.merge_proj[str(bandwidth)](torch.cat([left, right], dim=-1))

    def run_blocks_fifo(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin_explicit(position_ids, head_dim, self.cfg.rope_base)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def build_batch_fifo(self, data: torch.Tensor, batch_size: int, composition: tuple[int, ...], device: str):
        """fifo_v1 mode only — ported from qcute_fifo.py's build_batch verbatim (one composition
        applied across the whole batch, per-slot targets padded with -100 for slots whose own
        bandwidth < max_bandwidth)."""
        cfg = self.cfg
        W = len(composition)
        max_bw = max(composition)
        cum = 0
        needed = 0
        for b in composition:
            cum += b
            needed = max(needed, cum + b)
        starts = torch.randint(0, max(1, len(data) - needed), (batch_size,))
        ctx = torch.stack([data[s:s + needed] for s in starts]).to(device)

        embeds = []
        targets = torch.full((batch_size, W, max_bw), -100, dtype=torch.long, device=device)
        pos_ids = []
        cum = 0
        for i, b in enumerate(composition):
            span = ctx[:, cum:cum + b]
            embeds.append(self.embed_span(span, b))
            pos_ids.append(cum + b - 1)
            tgt = ctx[:, cum + b:cum + b + b]
            targets[:, i, :b] = tgt
            cum += b
        slot_embeds = torch.stack(embeds, dim=1)
        position_ids = torch.tensor(pos_ids, device=device, dtype=torch.long)
        return slot_embeds, position_ids, targets

    def forward_fifo_v1(self, data: torch.Tensor, batch_size: int, composition: tuple[int, ...], device: str) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        slot_embeds, position_ids, targets = self.build_batch_fifo(data, batch_size, composition, device)
        h = self.run_blocks_fifo(slot_embeds, position_ids)
        B, W, D = h.shape

        losses = []
        n_correct, n_total = 0, 0
        for i, b in enumerate(composition):
            h_i = h[:, i, :]
            tgt_i = targets[:, i, :b]
            logits_i = self.fetch_head(h_i, b, tgt_i)
            loss_i = F.cross_entropy(logits_i.reshape(-1, cfg.vocab), tgt_i.reshape(-1))
            losses.append(loss_i)
            with torch.no_grad():
                n_correct += (logits_i.argmax(-1) == tgt_i).sum().item()
                n_total += tgt_i.numel()
        loss = torch.stack(losses).mean()
        acc = n_correct / max(1, n_total)
        return loss, {"loss": loss, "byte_loss": loss, "byte_acc": torch.tensor(acc)}

    def _sel(self, i: int) -> int:
        return i if self.cfg.untie_levels else 0

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

    def _merge_level(self, i: int, child_embed: torch.Tensor) -> torch.Tensor:
        """child_embed: [B, n_children, d_model] where n_children is a multiple of Ks[i]. Returns
        code_embed: [B, n_children/Ks[i], d_model] — the quantized code for each block, re-embedded
        back into d_model for insertion into the flat sequence / use as the next level's children."""
        cfg = self.cfg
        K = cfg.Ks[i]
        B, T, D = child_embed.shape
        blocks = child_embed.view(B, T // K, K * D)
        pre_q = self.merge[self._sel(i)](blocks)
        code = self.quantize(pre_q)
        return self.code_in_embed[self._sel(i)](code)

    def forward(self, ctx: torch.Tensor, step: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B = ctx.size(0)
        embed_0 = self.byte_embed_in(byte_to_bits(ctx))   # [B, L0, d_model]

        if cfg.byte_only:
            h = self._run_main(embed_0)
            loss, acc = self._byte_loss(h, ctx)
            return loss, {"loss": loss, "byte_loss": loss, "byte_acc": acc}

        level_embeds = [embed_0]
        for i in range(self.n_levels):
            level_embeds.append(self._merge_level(i, level_embeds[-1]))

        flat = embed_0.new_zeros(B, self.flat_len, cfg.d_model)
        for lvl in range(self.n_levels + 1):
            idx = getattr(self, f"flat_pos_of_{lvl}")
            flat[:, idx, :] = level_embeds[lvl]

        h_flat = self._run_main(flat)
        h_byte = h_flat[:, self.flat_pos_of_0, :]   # [B, L0, d_model], in original byte order
        loss, acc = self._byte_loss(h_byte, ctx)
        return loss, {"loss": loss, "byte_loss": loss, "byte_acc": acc}

    def _run_main(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def _byte_loss(self, h: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_flat = h[:, :-1, :].reshape(-1, h.size(-1))
        true_bits_flat = byte_to_bits(ctx[:, 1:]).reshape(-1, 8)
        logits = self.head0(h_flat, true_bits_flat) if self.head0.mode == "chain" else self.head0(h_flat)
        loss = F.binary_cross_entropy_with_logits(logits, (true_bits_flat > 0).float(), reduction="none").sum(-1).mean()
        with torch.no_grad():
            acc = ((logits > 0) == (true_bits_flat > 0)).float().mean()
        return loss, acc


def init_head_bias_to_unigram(model: FlatPyramidLM, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    if model.cfg.mode == "fifo_v1":
        # ported from qcute_fifo.py's own init_head_bias_to_unigram — plain 256-way softmax bias,
        # no bit-chain (fifo_v1's FetchHead predicts whole bytes via ordinary cross-entropy, not bits).
        log_freq = torch.log(counts / counts.sum())
        with torch.no_grad():
            model.pred_head.bias.copy_(log_freq.to(model.pred_head.bias.device))
        return
    freq = counts / counts.sum()
    with torch.no_grad():
        byte_ids = torch.arange(256)
        bits = byte_to_bits(byte_ids)
        p_bit1 = (freq.unsqueeze(-1) * (bits > 0).float()).sum(0).clamp(1e-4, 1 - 1e-4)
        logit_bit = torch.log(p_bit1 / (1 - p_bit1))
        head = model.head0
        if head.mode == "independent":
            head.head.bias.copy_(logit_bit.to(head.head.bias.device))
        else:
            head.head.bias.copy_(logit_bit.mean().view(1).to(head.head.bias.device))


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
def eval_model(model: FlatPyramidLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str, compositions: list | None = None) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for i in range(n_batches):
        if model.cfg.mode == "fifo_v1":
            composition = compositions[i % len(compositions)]   # cycle through the full discrete set
            loss, metrics = model.forward_fifo_v1(data, batch_size, composition, device)
        else:
            ctx = sample_context(data, batch_size, model.cfg.context_len, device)
            loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    return result


def train(model: FlatPyramidLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    compositions = None
    if model.cfg.mode == "fifo_v1":
        compositions = enumerate_compositions(model.cfg.window, model.cfg.bandwidths)
        log(f"compositions: {len(compositions)} valid (window={model.cfg.window}, bandwidths={model.cfg.bandwidths})")

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt12", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        if model.cfg.mode == "fifo_v1":
            composition = compositions[torch.randint(0, len(compositions), (1,)).item()]
            loss, metrics = model.forward_fifo_v1(train_data, args.batch_size, composition, device)
        else:
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
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device, compositions=compositions)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Single shared LM over a flat multi-resolution sequence, BSQ codes FIFO-style", parents=[pre])
    p.add_argument("--Ks", type=_parse_int_tuple, default=(4, 4, 4))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--code_dim", type=int, default=None)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq", "identity"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--untie_levels", action="store_true")
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--bit_head_mode", type=str, default="chain", choices=["independent", "chain"])
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--byte_only", action="store_true")
    p.add_argument("--mode", type=str, default="pyramid", choices=["pyramid", "fifo_v1"])
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--bandwidths", type=_parse_int_tuple, default=(1, 2))
    p.add_argument("--fetch_n_heads", type=int, default=2)
    p.add_argument("--fetch_gamma", type=float, default=1.0)
    p.add_argument("--tie_head", type=lambda x: x.lower() != "false", default=True)

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
    args.bandwidths = _parse_int_tuple(args.bandwidths) if isinstance(args.bandwidths, str) else tuple(args.bandwidths)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, d_model=args.d_model, code_dim=args.code_dim, context_len=args.context_len,
        quant_type=args.quant_type, fsq_levels=args.fsq_levels, n_heads=args.n_heads, n_layers=args.n_layers,
        mlp_mult=args.mlp_mult, attn_window=args.attn_window, untie_levels=args.untie_levels,
        rope_base=args.rope_base, bit_head_mode=args.bit_head_mode, bit_chain_n_heads=args.bit_chain_n_heads,
        bit_chain_gamma=args.bit_chain_gamma, bit_chain_fixed_kernel=args.bit_chain_fixed_kernel,
        byte_only=args.byte_only, mode=args.mode, window=args.window, bandwidths=args.bandwidths,
        fetch_n_heads=args.fetch_n_heads, fetch_gamma=args.fetch_gamma, tie_head=args.tie_head,
    )
    model = FlatPyramidLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_pyramid_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    if cfg.mode == "fifo_v1":
        log(f"mode=fifo_v1 window={cfg.window} bandwidths={cfg.bandwidths} d_model={cfg.d_model} "
            f"params={n_params/1e6:.3f}M device={device}")
    else:
        log(f"Ks={cfg.Ks} d_model={cfg.d_model} code_dim={model.code_dim} untie_levels={cfg.untie_levels} "
            f"flat_len={model.flat_len} context_len={cfg.context_len} quant_type={cfg.quant_type} "
            f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
