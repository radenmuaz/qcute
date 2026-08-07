"""qcute.qcute_refine — pure recursive NTP tower + joint-chain MTP
detokenizer. A simplified redesign vs. qcutelm_vlt11's E_i/D_i "sandwich":
no separate producer/consumer pair per level, no codelm forecast-
substitution machinery. Instead:

ENCODER TOWER (levels 0..N-1): each level is nothing but a plain causal
transformer doing next-token-prediction (NTP) on its OWN input sequence
(bytes at level 0, level i-1's emitted code at level i>0) — this NTP loss
is always on, directly supervising every level (no moving/collapsing-
target problem, since there's no competing objective reshaping the same
hidden state — this is what "so that targets are stable" means). Every
K[i] positions, that level's hidden state is projected and BSQ-quantized
into a code c_i; the resulting (L/K[i])-length code sequence is level
i+1's entire input — genuine sequence-length compression at each level,
same shrinking-tower idea as qcutelm_vlt11.

Quantizer is BSQ everywhere by default (the "pure" choice — no FSQ/IFSQ
branching). Level 0's own representation is the native 8-bit BSQ-shaped
byte (qcutelm_vlt11's byte_repr="bits" convention, always on here — a
byte IS its own 8-dim code, no embedding table), and its default emitted
code width (dqs[0]) is also 8 — same width preserved through the first
compression instead of an arbitrary pick.

DETOKENIZER TOWER — also recursive, the structural MIRROR of the encoder
tower, run in reverse. Detokenizer_i is the direct inverse of
EncoderLevel_i's own compression step, nothing more: given level i's own
code sequence c_i (the exact thing EncoderLevel_i emitted), it decodes
ONLY the block of K level-i-INPUT units (bytes at i=0, else level i-1's
code) that were compressed into each code element — "only to [the] block
that inputs [that] code." Stacked top-down across all N levels in REVERSE
level order, this is a full recursive decoder: the first one applied
decodes the LAST (coarsest, top) code c_{N-1} into level N-2's code; the
last one applied, Detokenizer_0, decodes c_0 into bytes. (Implementation
note: `self.detokenizers` is still indexed by level i like the encoder
list, for direct pairing with `self.encoders[i]`/`c_i` — "first"/"last"
above describes execution ORDER of a full top-down decode, i.e. reverse
index order, not the list's storage order.)

Each Detokenizer_i runs its own small causal transformer over the CODE
sequence itself (length n_blocks[i] — unlike the encoder trunk, which
only ever READS every K-th hidden state to emit a code, the detokenizer
trunk uses EVERY code position, "no skip" — so a block's decode also
draws on causal context from earlier code blocks, the long-range-
dependency channel), then applies a joint chain-rule multi-token-
prediction head at every code position, predicting that whole K-element
block AT ONCE via one exact chain-rule factorization over all K*in_dq
bits (reusing BitPredictHead's chain machinery — flatten the K target
units into one longer bit vector and chain over all of them, rather than
assuming the K positions are independent). Trained by teacher-forcing
against the real (detached) code and the real target block — a
reconstruction/decode objective, not a generative one.

The detokenizer stack decodes an ALREADY-KNOWN code's ALREADY-EXISTING
children — useful as a training signal (keeps every level's code
decodable) and as a standalone decode pathway, but it is NOT part of the
generative loop: producing a genuinely NEW next byte still has to go
through causal history alone, which only EncoderLevel_0's own NTP head
does. `generate_no_cache()`/`generate_kv_cache()` therefore use that head
directly, one byte at a time (validated to agree exactly via
`validate_generation()`, same convention as qcutelm_vlt11) — see their
own docstrings.

No shared imports with any other qcutelm_vlt*/qcutelm fork (self-
contained-module convention) — Logger/Checkpointer/attention/BitPredictHead
etc. duplicated from qcutelm_vlt11.py, the closest relative.

    uv run python -m qcute.qcute_refine --config configs/qcute_refine_v1.py
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
    Ks: tuple[int, ...] = (4, 4, 4)              # per-level LOCAL compression factor (level i's own
                                                    # sequence shrinks by Ks[i]) — uniform by default (all
                                                    # entries equal); pass a non-uniform tuple to override.
    dqs: tuple[int, ...] = (8, 8, 8)              # this level's OWN emitted BSQ code width. dqs[0]=8
                                                    # (default) matches level 0's native 8-bit byte
                                                    # representation — the "pure" choice of preserving
                                                    # width through the first compression rather than
                                                    # picking an arbitrary one.
    tier_d_models: tuple[int, ...] = (96, 96, 96)  # one entry per level — encoder AND that level's own
                                                    # detokenizer share this width (separate weights).
    tier_n_layers: tuple[int, ...] = (1, 1, 1)     # default 1 layer per encoder level unless overridden —
                                                    # keeps each level's own NTP target (its hidden state
                                                    # at block-end positions) simple/stable rather than
                                                    # letting a deep per-level stack move it around a lot
                                                    # during early training.
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int = 64      # encoder trunks' windowed attention (-1 = dense). Must evenly divide
                                # every level's own (shrinking) sequence length, or be >= it.
    rope_base: float = 10000.0
    detok_d_model: int = 96
    detok_n_heads: int = 4
    detok_n_layers: int = 2
    detok_mlp_mult: int = 4
    detok_attn_window: int = 64   # detokenizer trunks run DENSE over every position of their own level's
                                    # CODE sequence (length n_blocks[i] = seq_lens[i] // Ks[i] — the SAME
                                    # sequence that level's own EncoderLevel emits, much shorter than the
                                    # raw input sequence it decodes from), so this must divide (or be >=)
                                    # n_blocks[i] at every level, not seq_lens[i].
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True
    detok_weight: float = 1.0     # scales the summed detokenizer MTP losses relative to the encoder
                                    # tower's summed NTP losses in the total loss. detok_weight==0.0 SKIPS
                                    # every Detokenizer's forward entirely (not just zero-weights it) —
                                    # a real speed lever: session diagnosis found BitPredictHead's chain
                                    # mode ~200-1800x slower than a plain nn.Linear at dq=8..32 (quadratic-
                                    # ish in dq; the detokenizer's own dq=K*in_dq compounds this further
                                    # than the encoder's own dq=in_dq), so skipping detokenizer forward
                                    # calls is the single biggest lever available without touching the
                                    # encoder tower at all.
    code_ntp_weight: float = 1.0  # scales levels>0's own NTP loss (level 0's byte_loss is NEVER scaled
                                    # by this — it's the one metric with a direct bpb reading and stays
                                    # unconditionally on). code_ntp_weight==0.0 SKIPS those levels' own
                                    # ntp_head forward entirely, same "skip the expensive chain call, not
                                    # just zero-weight it" rationale as detok_weight above. Combined with
                                    # detok_weight=0.0 this gives a "pure last-layer byte NTP" ablation —
                                    # total loss reduces to just level 0's own byte NTP, isolating whether
                                    # the auxiliary code-level/detokenizer objectives are helping or
                                    # competing (same spirit as qcutelm_vlt11's byte_only diagnostic).
    quant_type: str = "bsq"       # "bsq" (default, the "pure" choice — see module docstring): parameter-
                                    # free hypersphere-corner STE quantizer, bounded {-1,+1}/sqrt(dq) codes.
                                    # "identity": NO quantization — code_pre's raw continuous output IS
                                    # c_i, unbounded (qcutelm_vlt11's own ceiling-baseline diagnostic,
                                    # ported here). Training-only, and only SOUND in combination with
                                    # code_ntp_weight=0.0 and detok_weight=0.0 — every downstream consumer
                                    # of a code (a later level's own BitPredictHead-based ntp_head, or any
                                    # Detokenizer's mtp_head) assumes {-1,+1}-ish bit semantics; with both
                                    # weights at 0 those consumers are never invoked at all (skipped, per
                                    # the two flags above), so the mismatch never manifests. Not asserted
                                    # in code — a documented precondition, not an enforced one.
    vocab: int = 256


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_at(pos_id: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.tensor([[float(pos_id)]], device=device) * inv_freq
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
                      f"T % window == 0 and T > window — falling back to DENSE attention for this layer. "
                      f"Only warns once per layer instance.")
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

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        B, _, D = x_new.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        y = F.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(B, 1, D)), new_k, new_v


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

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        attn_out, new_k, new_v = self.attn.forward_step(self.ln1(x_new), cos_new, sin_new, cache_k, cache_v)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_k, new_v


def byte_to_bits(byte_ids: torch.Tensor) -> torch.Tensor:
    """[*] long byte ids (0..255) -> [*, 8] float, each bit in {-1,+1}/sqrt(8) — LSB-first, deterministic,
    no learned parameters. A byte losslessly IS its own 8-bit BSQ-shaped code."""
    bits = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    return (2 * bits - 1) / math.sqrt(8)


def bits_to_byte(bits: torch.Tensor) -> torch.Tensor:
    b = (bits > 0).long()
    powers = (2 ** torch.arange(8, device=bits.device))
    return (b * powers).sum(-1)


class BitPredictHead(nn.Module):
    """Predicts `dq` chained bits from a hidden vector via Fetch-style causal
    self-attention over the bit sequence — the exact chain-rule
    factorization of the joint dq-bit distribution (ported from
    qcutelm_vlt11.py; see its own docstring for the full derivation). Used
    here for two different jobs at two different scales: an encoder
    level's own next-input-element NTP (dq = that level's native
    representation width), and a detokenizer's joint multi-token
    prediction (dq = K * that level's native representation width — the K
    future steps flattened into one long bit vector, chained across all of
    them at once, not just within one step)."""

    def __init__(self, d_model: int, dq: int, n_heads: int = 2, gamma: float = 1.0, fixed_kernel: bool = True):
        super().__init__()
        self.dq = dq
        self.gamma = gamma
        self.fixed_kernel = fixed_kernel
        self.head = nn.Linear(d_model, 1)
        self.bit_pos_emb = nn.Embedding(dq, d_model)
        self.bit_val_emb = nn.Embedding(2, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        """h: [N, D]. true_bits: [N, dq] float in {-1,+1}-ish (teacher-forcing) or None (greedy chain
        decode at inference). -> raw_logits [N, dq]."""
        if self.fixed_kernel and true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
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
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self.causal_mask, need_weights=False)
        fetched = h.unsqueeze(1) + attn_out
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        chain_vecs = [h + self.bit_pos_emb.weight[0]]
        logits_list = []
        for j in range(self.dq):
            x = torch.stack(chain_vecs, dim=1)
            attn_out, _ = self.self_attn(x, x, x, need_weights=False)
            fetched = h + attn_out[:, -1, :]
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                chain_vecs.append(self.gamma * h + self.bit_val_emb(bit_val) + self.bit_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


def chain_bce_loss(raw_logits: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
    """Sum over the bit dim (nats per predicted unit), then mean over
    everything else — matches qcutelm_vlt11's own convention (reduction=
    'mean' would silently average over bits too)."""
    return F.binary_cross_entropy_with_logits(raw_logits, (true_bits > 0).float(), reduction="none").sum(-1).mean()


class EncoderLevel(nn.Module):
    """One level of the recursive NTP tower: embeds its own input sequence
    (bytes for level 0, level-1's emitted code for level>0), runs a small
    causal transformer, and does two things with the resulting hidden
    state — (1) always-on direct NTP loss on its OWN next input element
    (own head, own target: this level's stability doesn't depend on any
    other level's loss), and (2) every K-th position, projects+BSQ-
    quantizes into this level's own emitted code, handed to the next level
    (or unused, at the top level) and to this level's own detokenizer."""

    def __init__(self, cfg: Config, level: int, in_dq: int):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.cfg = cfg
        D = cfg.tier_d_models[level]
        self.embed = nn.Linear(in_dq, D)
        window = None if cfg.attn_window == -1 else cfg.attn_window
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.tier_n_layers[level])])
        self.ln_f = nn.LayerNorm(D)
        self.ntp_head = BitPredictHead(D, in_dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)
        self.code_pre = nn.Linear(D, cfg.dqs[level])

    def forward(self, seq_repr: torch.Tensor, compute_ntp: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """seq_repr: [B, L, in_dq] (this level's own input, already in
        bit/BSQ-code representation). compute_ntp=False SKIPS the
        ntp_head call entirely (not just its loss weight — the call
        itself: session diagnosis found BitPredictHead's chain mode
        ~200-1800x slower than a plain nn.Linear at dq=8..32, so this is a
        real speed lever, not a no-op) — returns zero placeholders for
        ntp_loss/ntp_acc in that case; level 0 must never pass False here
        (byte_loss always needs a real value). Returns (c_i [B, n_blocks,
        dqs[level]], ntp_loss, ntp_acc, h [B, L, D])."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        B, L, _ = seq_repr.shape
        D = cfg.tier_d_models[self.level]
        n_blocks = L // K

        x = self.embed(seq_repr)
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            true_flat = seq_repr[:, 1:, :].reshape(-1, self.in_dq)
            raw = self.ntp_head(h_flat, true_flat)
            ntp_loss = chain_bce_loss(raw, true_flat)
            with torch.no_grad():
                ntp_acc = ((raw > 0) == (true_flat > 0)).float().mean()
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        h_blocks = h.view(B, n_blocks, K, D)
        pre_q = self.code_pre(h_blocks[:, :, K - 1, :])
        if cfg.quant_type == "bsq":
            c_i = bsq_quantize(pre_q, cfg.dqs[self.level])
        elif cfg.quant_type == "identity":
            c_i = pre_q   # no discretization at all — see Config.quant_type's docstring for the
                            # soundness precondition (only valid alongside code_ntp_weight=detok_weight=0)
        else:
            raise ValueError(f"unknown quant_type {cfg.quant_type!r}")
        return c_i, ntp_loss, ntp_acc, h


class Detokenizer(nn.Module):
    """The structural inverse of EncoderLevel `level`'s own compression
    step: given that level's own code sequence c_i (length n_blocks[i]),
    decodes exactly the block of K level-i-INPUT units (bytes at level 0,
    else level i-1's code) each code element was compressed from — "only
    to the block that inputs that code," nothing dense or past-block-
    conditioned. Runs its own small causal transformer over the CODE
    sequence itself (every code position used, unlike the encoder trunk
    which only READS every K-th hidden state), then a joint chain-rule
    multi-token-prediction head at every code position predicts that
    block's own K children AT ONCE via one exact chain-rule factorization
    over all K*in_dq bits (not K independent single-step guesses)."""

    def __init__(self, cfg: Config, level: int, in_dq: int, own_dq: int):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.K = cfg.Ks[level]
        self.cfg = cfg
        D = cfg.detok_d_model
        self.code_embed = nn.Linear(own_dq, D)
        window = None if cfg.detok_attn_window == -1 else cfg.detok_attn_window
        self.blocks = nn.ModuleList([Block(D, cfg.detok_n_heads, cfg.detok_mlp_mult, window) for _ in range(cfg.detok_n_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.mtp_head = BitPredictHead(D, self.K * in_dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def forward(self, c_i: torch.Tensor, seq_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c_i: [B, n_blocks, own_dq] this level's own code (EncoderLevel
        `level`'s output). seq_repr: [B, n_blocks*K, in_dq] this level's
        own INPUT sequence — the true values each code block was
        compressed from (the reconstruction target). Returns (loss, acc)."""
        cfg = self.cfg
        K = self.K
        B, n_blocks, _ = c_i.shape
        D = cfg.detok_d_model

        x = self.code_embed(c_i)
        head_dim = D // cfg.detok_n_heads
        cos, sin = rope_cos_sin(n_blocks, head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)                                             # [B, n_blocks, D]

        h_flat = h.reshape(-1, D)
        target = seq_repr.view(B, n_blocks, K, self.in_dq).reshape(B, n_blocks, K * self.in_dq)
        target_flat = target.reshape(-1, K * self.in_dq)

        raw = self.mtp_head(h_flat, target_flat)
        loss = chain_bce_loss(raw, target_flat)
        with torch.no_grad():
            acc = ((raw > 0) == (target_flat > 0)).float().mean()
        return loss, acc


class RefineLM(nn.Module):
    """N-level pure recursive tower: N EncoderLevels (bytes -> code_0 ->
    code_1 -> ...), each paired with its own Detokenizer_i decoding that
    SAME level's code back into the block of inputs that produced it (the
    structural mirror, run in reverse — see module docstring). Only
    EncoderLevel_0's own NTP head is generative; the detokenizer stack is
    a reconstruction/decode pathway."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        assert len(cfg.tier_n_layers) == self.n_levels

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens
        code_seq_lens = [seq_lens[i] // cfg.Ks[i] for i in range(self.n_levels)]   # n_blocks[i] — length
        self.code_seq_lens = code_seq_lens                                          # of level i's own code
                                                                                      # sequence, i.e. what
                                                                                      # Detokenizer_i's trunk
                                                                                      # actually runs over

        for i, d in enumerate(cfg.tier_d_models):
            assert d % cfg.n_heads == 0, f"tier_d_models[{i}] ({d}) must be divisible by n_heads ({cfg.n_heads})"
            assert cfg.detok_d_model % cfg.detok_n_heads == 0
        window = None if cfg.attn_window == -1 else cfg.attn_window
        detok_window = None if cfg.detok_attn_window == -1 else cfg.detok_attn_window
        for i, L in enumerate(seq_lens):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
        for i, Lc in enumerate(code_seq_lens):
            if detok_window is not None:
                assert Lc % detok_window == 0 or Lc <= detok_window, f"detok_attn_window ({detok_window}) must divide level {i}'s code sequence length ({Lc}), or be >= it"

        in_dqs = [8] + list(cfg.dqs[:-1])   # level i's own input width: 8 (byte bits) at i=0, else dqs[i-1]
        self.in_dqs = in_dqs
        self.encoders = nn.ModuleList([EncoderLevel(cfg, i, in_dqs[i]) for i in range(self.n_levels)])
        self.detokenizers = nn.ModuleList([Detokenizer(cfg, i, in_dqs[i], cfg.dqs[i]) for i in range(self.n_levels)])

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        compute_detok = cfg.detok_weight > 0        # skip Detokenizer forward entirely when unweighted —
                                                        # see Config.detok_weight's docstring (real speed
                                                        # lever, chain heads are ~200-1800x a plain Linear)
        seq_repr = byte_to_bits(byte_ids)   # level 0's own input
        ntp_losses, ntp_accs = [], []
        mtp_losses, mtp_accs = [], []
        byte_loss = byte_acc = None

        for i in range(self.n_levels):
            compute_ntp = i == 0 or cfg.code_ntp_weight > 0   # level 0's byte_loss is never skippable
            c_i, ntp_loss, ntp_acc, _h = self.encoders[i](seq_repr, compute_ntp=compute_ntp)
            ntp_losses.append(ntp_loss)
            ntp_accs.append(ntp_acc)
            if i == 0:
                byte_loss, byte_acc = ntp_loss, ntp_acc

            if compute_detok:
                mtp_loss, mtp_acc = self.detokenizers[i](c_i.detach(), seq_repr)
            else:
                mtp_loss, mtp_acc = c_i.new_zeros(()), c_i.new_zeros(())
            mtp_losses.append(mtp_loss)
            mtp_accs.append(mtp_acc)

            seq_repr = c_i

        # ntp_loss_total/mtp_loss_total are RAW (unweighted) sums, for logging — same convention
        # mtp_loss_total already had (detok_weight is applied only inside `loss` below, never baked into
        # the reported total). code_ntp_weight==0.0 alongside compute_ntp=False above means levels>0's own
        # ntp_losses are already exactly zero (not just small), so `loss`'s own weighting is a genuine
        # zero-out, not merely "close enough."
        ntp_total = torch.stack(ntp_losses).sum()
        mtp_total = torch.stack(mtp_losses).sum()
        code_ntp_total = torch.stack(ntp_losses[1:]).sum() if self.n_levels > 1 else byte_loss.new_zeros(())
        loss = byte_loss + cfg.code_ntp_weight * code_ntp_total + cfg.detok_weight * mtp_total
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "ntp_loss_total": ntp_total, "mtp_loss_total": mtp_total,
            **{f"level{i}_ntp_loss": l for i, l in enumerate(ntp_losses)},
            **{f"level{i}_ntp_acc": a for i, a in enumerate(ntp_accs)},
            **{f"level{i}_mtp_loss": l for i, l in enumerate(mtp_losses)},
            **{f"level{i}_mtp_acc": a for i, a in enumerate(mtp_accs)},
        }
        return loss, metrics


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor) -> torch.Tensor:
    """h_last: [B, tier_d_models[0]] EncoderLevel_0's hidden state at the
    newest position -> greedy-chain-decoded next byte ids [B]. Shared by
    both generate_no_cache and generate_kv_cache so they're guaranteed to
    sample identically given identical hidden states — the whole point of
    validating one against the other (see validate_generation below)."""
    logits = model.encoders[0].ntp_head(h_last, true_bits=None)   # chain mode greedy-decodes internally
    return bits_to_byte(logits)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Reference (slow, obviously-correct) byte-by-byte generation:
    recomputes EncoderLevel_0 from scratch over the WHOLE sequence every
    new byte — no cache, no windowing complications, easy to trust. Only
    EncoderLevel_0's own NTP head is generative in this architecture (see
    module docstring); the detokenizer stack and levels 1+ never
    participate — they're reconstruction/training-time-only."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encoders[0]
    cfg = model.cfg
    D = cfg.tier_d_models[0]

    for _ in range(n_new_bytes):
        L = all_bytes.size(1)
        x = enc0.embed(byte_to_bits(all_bytes))
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in enc0.blocks:
            x = block(x, cos, sin)
        h = enc0.ln_f(x)
        next_byte = _sample_next_byte(model, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """KV-cache-efficient generation: EncoderLevel_0's per-layer K/V grow
    by exactly one entry per new byte, never recomputed — works precisely
    because, like qcutelm_vlt11, this architecture is not destructive
    (every position's hidden state, once computed, is a fixed function of
    causal history, never mutated later). Dense-attention only
    (attn_window must be -1) — caching + windowed chunked attention aren't
    implemented together (Block.forward_step always attends densely over
    the full cache), same limitation as qcutelm_vlt11's own
    generate_kv_cache."""
    cfg = model.cfg
    assert cfg.attn_window == -1, "generate_kv_cache only supports dense attention (attn_window=-1) — see docstring"
    enc0 = model.encoders[0]
    D = cfg.tier_d_models[0]
    n_layers = len(enc0.blocks)
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)

    cache_k: list[torch.Tensor | None] = [None] * n_layers
    cache_v: list[torch.Tensor | None] = [None] * n_layers

    def step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = enc0.embed(byte_to_bits(byte_id)).unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(enc0.blocks):
            x, cache_k[li], cache_v[li] = block.forward_step(x, cos_new, sin_new, cache_k[li], cache_v[li])
        return enc0.ln_f(x).squeeze(1)   # [B, D]

    L0 = prompt_bytes.size(1)
    last_h = None
    for pos in range(L0):
        last_h = step(prompt_bytes[:, pos], pos)

    out_bytes = [prompt_bytes]
    for i in range(n_new_bytes):
        next_byte = _sample_next_byte(model, last_h)
        out_bytes.append(next_byte.unsqueeze(1))
        last_h = step(next_byte, L0 + i)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def validate_generation(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
    """Sanity check: generate_no_cache and generate_kv_cache must produce
    IDENTICAL byte sequences given the same model/prompt (both are
    deterministic greedy decodes of the same math, just at different
    speeds) — same validation convention as qcutelm_vlt11. Returns True
    and prints nothing on match; raises AssertionError with both
    sequences on mismatch."""
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


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
    return result


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine", dynamic_ncols=True)
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
            mtp_loss=f"{metrics['mtp_loss_total'].item():.4f}",
        )

        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), bpb=train_bpb, byte_acc=metrics["byte_acc"].item())

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def _broadcast_int_tuple(s, n: int) -> tuple[int, ...]:
    """Accepts a single value (broadcast to n copies — "uniform unless
    overridden") or an explicit comma-separated/list value of length n."""
    t = _parse_int_tuple(s)
    if len(t) == 1:
        return t * n
    assert len(t) == n, f"expected 1 (broadcast) or {n} values, got {len(t)}: {t}"
    return t


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Pure recursive NTP tower + joint-chain MTP detokenizer (qcute_refine)", parents=[pre])
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--Ks", default=(4, 4, 4))            # single int (broadcast) or explicit tuple
    p.add_argument("--tier_n_layers", default=(1, 1, 1))  # single int (broadcast) or explicit tuple
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(96, 96, 96))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=64)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--detok_d_model", type=int, default=96)
    p.add_argument("--detok_n_heads", type=int, default=4)
    p.add_argument("--detok_n_layers", type=int, default=2)
    p.add_argument("--detok_mlp_mult", type=int, default=4)
    p.add_argument("--detok_attn_window", type=int, default=64)
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--detok_weight", type=float, default=1.0)

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
    args.dqs = _parse_int_tuple(args.dqs)
    n_levels = len(args.dqs)
    args.Ks = _broadcast_int_tuple(args.Ks, n_levels)
    args.tier_n_layers = _broadcast_int_tuple(args.tier_n_layers, n_levels)
    args.tier_d_models = _parse_int_tuple(args.tier_d_models)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, dqs=args.dqs, tier_d_models=args.tier_d_models, tier_n_layers=args.tier_n_layers,
        context_len=args.context_len, n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window,
        rope_base=args.rope_base, detok_d_model=args.detok_d_model, detok_n_heads=args.detok_n_heads,
        detok_n_layers=args.detok_n_layers, detok_mlp_mult=args.detok_mlp_mult, detok_attn_window=args.detok_attn_window,
        bit_chain_n_heads=args.bit_chain_n_heads, bit_chain_gamma=args.bit_chain_gamma,
        bit_chain_fixed_kernel=args.bit_chain_fixed_kernel, detok_weight=args.detok_weight,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} tier_n_layers={cfg.tier_n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
