"""qcute.qcute_refine_v2 — same recursive NTP encoder tower as
qcute_refine.py, with the block-local joint-chain-MTP Detokenizer replaced
by a cross-attention decoder that REUSES the encoder tower's own already-
computed hidden states instead of running any new self-attention trunk of
its own.

ENCODER TOWER: unchanged from qcute_refine.py — EncoderLevel[i] embeds its
own input sequence x^(i) (bytes at i=0, else c_{i-1}), runs a causal
transformer (own weights, tier_n_layers[i] deep), keeps an always-on
unconditioned NTP loss on its own next input element (own head, own
target — "so that targets are stable"), and every K[i] positions BSQ-
quantizes into its own emitted code c^(i), which becomes level i+1's
entire input. EncoderLevel[i].forward now returns its post-ln_f hidden
state h^(i) as a first-class output (qcute_refine.py already computed
this and discarded it; v2 is what actually uses it).

TOKENIZER (decoder), one per ADJACENT level pair (i, i+1), i = 0..N-2 —
NOT one per level like qcute_refine.py's Detokenizer_i (there is no
decoder above the top level; nothing coarser exists to cross-attend to):

    Q  = h^(i)    ("previous level['s] code LM" — EncoderLevel[i]'s own
                    hidden states, the finer sequence being decoded)
    KV = h^(i+1)  ("current level['s] code LM" — EncoderLevel[i+1]'s own
                    hidden states; EncoderLevel[i+1]'s OWN INPUT is
                    exactly c^(i), so its hidden state at code-block index
                    b is already, by construction, "as of" real time
                    (b+1)*K[i] — no new computation needed to get this)

Both h^(i) and h^(i+1) are DETACHED before use — this decoder's loss must
not reshape either EncoderLevel's own hidden state (which stays trained
purely by its own unconditioned NTP loss; two objectives competing over
the same hidden state is exactly the moving-target failure mode this
whole design avoids elsewhere).

A single cross-attention TRANSFORMER BLOCK (cross-attn sublayer + MLP
sublayer, each pre-norm + residual, mirroring this file's own causal
`Block`'s shape) combines Q and KV. Causal safety: query position t may
only attend to KV block b once b is FULLY complete, i.e. b < (t+1)//K[i]
(the same "past, already-resolved block" rule qcute_refine.py's module
docstring worked through) — enforced via an explicit boolean attention
mask, not RoPE (Q and KV live at different granularities/lengths, so a
shared rotary basis doesn't apply). A single learned "null" KV slot is
prepended and always visible, so early positions (before any KV block is
complete) still get a well-defined attention distribution instead of an
all-masked row — the same "zero-KV is load-bearing, not just a
regularizer" mechanism documented for qcute_refine's own variable-length-
tokenizer sibling forks.

DECODE HEAD: predicts x^(i)'s own NEXT token — single position, not a
joint K-block prediction (the old Detokenizer's whole "multi-token
prediction" mechanism is gone, not merely cheapened). Defaults to a
single plain nn.Linear (`tok_head_mode="linear"`, independent per-bit
logits — cheap, mirrors bytelm's own parallel-Linear-head style);
`tok_head_mode="chain"` swaps in the exact chain-rule BitPredictHead for
comparison, at the ~200-1800x per-call cost qcute_refine.py's own session
diagnosis measured.

Generation is unaffected: only EncoderLevel[0]'s own NTP head is
generative (a genuinely new byte requires nothing but causal history;
DecoderLevel's own KV side needs an ALREADY-COMPLETE code block, so it
can decode existing children but never propose new ones) —
generate_no_cache/generate_kv_cache/validate_generation are copied
unchanged from qcute_refine.py.

No shared imports with qcute_refine.py or any qcutelm_vlt* fork (self-
contained-module convention, matching how qcutelm_vlt7->vlt8->...
evolved) — everything duplicated.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_2level.py
    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_3level.py
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
    Ks: tuple[int, ...] = (2, 2, 2)
    dqs: tuple[int, ...] = (8, 8, 8)
    tier_d_models: tuple[int, ...] = (96, 96, 96)
    tier_n_layers: tuple[int, ...] = (1, 1, 1)
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int | tuple[int, ...] = 128   # single int: broadcast to every level (backward-
                                    # compatible default). Per-level tuple (length n_levels): lets the
                                    # TOP level get its OWN, genuinely-sub-full window instead of always
                                    # inheriting whatever value the finer levels use — e.g. with a shared
                                    # scalar, a top level whose own sequence length happens to equal the
                                    # window falls back to dense (T>window is False, see
                                    # CausalSelfAttention) purely by coincidence, not by choice. -1 in
                                    # either form means dense for that level.
    rope_base: float = 10000.0
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True
    code_ntp_weight: float = 1.0  # scales levels>0's own NTP loss (level 0's byte_loss never scaled).
                                    # ==0.0 SKIPS those levels' ntp_head forward entirely (real speed
                                    # lever — see qcute_refine.py's own session diagnosis).
    quant_type: str = "bsq"       # "bsq" (default) or "identity" (ceiling-baseline diagnostic, training-
                                    # only, sound only alongside code_ntp_weight=tok_weight=0.0) — see
                                    # qcute_refine.py's Config docstring for the full rationale, unchanged.
    # --- DecoderLevel (cross-attention decoder) ---
    tok_d_model: int = 96          # shared cross-attention working width — h^(i)/h^(i+1) (which may have
                                    # different tier_d_models[i]/[i+1]) are each linearly projected into
                                    # this common space before cross-attending.
    tok_n_heads: int = 4
    tok_mlp_mult: int = 4
    tok_head_mode: str = "linear"  # "linear" (default): single plain nn.Linear, independent per-bit
                                    # logits — cheap. "chain": exact chain-rule BitPredictHead instead
                                    # (~200-1800x more expensive per call, per qcute_refine.py's own
                                    # measured diagnosis) — opt-in comparison, not the default.
    tok_weight: float = 1.0        # scales the summed DecoderLevel losses. ==0.0 SKIPS every
                                    # DecoderLevel's forward entirely (not just zero-weights it).
    layer_warmup_steps: tuple[int, ...] = ()   # LAYERWISE CURRICULUM (queued ablation, not yet run):
                                    # length must be n_levels-1 (one entry per level-activation gap,
                                    # like `tokenizers` itself) or empty (=all-zeros, i.e. every level
                                    # active from step 0 — the default, backward-compatible behavior).
                                    # layer_warmup_steps[i] = how many steps level i trains ALONE (with
                                    # levels >i entirely absent from the forward pass — not just zero-
                                    # weighted, genuinely not run) before level i+1 turns on. Reason:
                                    # let the lower LM's own BSQ codes become stable before handing them
                                    # to the level above as ITS training target — feeding a still-
                                    # collapsing/shifting code upward immediately trains the upper level
                                    # on a moving target, the same instability class this file's "always-
                                    # on direct NTP loss, own head own target" design otherwise guards
                                    # against within a single level. See RefineLM.n_active_levels/
                                    # activation_steps and train()'s per-stage param groups below.
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


class CrossBlock(nn.Module):
    """Single cross-attention transformer block: cross-attn sublayer (Q
    from one sequence, K/V from another) + MLP sublayer, each pre-norm +
    residual — same shape as this file's own causal `Block`, with
    self-attention swapped for cross-attention and no RoPE (Q/KV live at
    different granularities, so a shared rotary basis doesn't apply;
    causal safety comes entirely from the explicit boolean attn_mask)."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        # manual QKV + F.scaled_dot_product_attention instead of nn.MultiheadAttention — see
        # BitPredictHead's own comment for the MPS NaN-gradient finding this avoids.
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """attn_mask: bool [Lq, Lkv], True = BLOCKED (DecoderLevel's own "disallow" convention,
        matching nn.MultiheadAttention's convention) — inverted internally since
        F.scaled_dot_product_attention's boolean convention is the opposite (True = may attend)."""
        qn, kvn = self.ln_q(q), self.ln_kv(kv)
        B, Lq, D = qn.shape
        Lkv = kvn.shape[1]
        H, hd = self.n_heads, self.head_dim
        qh = self.q_proj(qn).reshape(B, Lq, H, hd).transpose(1, 2)
        kvp = self.kv_proj(kvn).reshape(B, Lkv, 2, H, hd).permute(2, 0, 3, 1, 4)
        kh, vh = kvp[0], kvp[1]
        sdpa_mask = ~attn_mask if attn_mask is not None else None
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=sdpa_mask)
        attn_out = self.out_proj(y.transpose(1, 2).reshape(B, Lq, D))
        q = q + attn_out
        q = q + self.mlp(self.ln2(q))
        return q


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
    qcutelm_vlt11.py via qcute_refine.py). Used for EncoderLevel's own
    unconditioned NTP head (always), and optionally
    (tok_head_mode="chain") for DecoderLevel's own decode head."""

    def __init__(self, d_model: int, dq: int, n_heads: int = 2, gamma: float = 1.0, fixed_kernel: bool = True):
        super().__init__()
        self.dq = dq
        self.gamma = gamma
        self.fixed_kernel = fixed_kernel
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.head = nn.Linear(d_model, 1)
        self.bit_pos_emb = nn.Embedding(dq, d_model)
        self.bit_val_emb = nn.Embedding(2, d_model)
        # manual QKV + F.scaled_dot_product_attention instead of nn.MultiheadAttention — session found
        # nn.MultiheadAttention's MPS backward produces NaN gradients at d_model=256 (confirmed: identical
        # run stable on CPU, NaN only on MPS, isolated via named_parameters() to exactly this submodule's
        # out_proj.weight.grad) despite being fine at the earlier d_model=96 configs' scale. Every other
        # attention op in this codebase (CausalSelfAttention, CrossBlock) already uses manual SDPA and has
        # been stable all session — this makes BitPredictHead consistent with that, not a new mechanism.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _mha(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x: [N, T, D]. attn_mask: None, or additive float [T, T] (SDPA accepts the same -inf/0
        convention causal_mask is already built in, no conversion needed). -> [N, T, D]."""
        N, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv_proj(x).reshape(N, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(y.transpose(1, 2).reshape(N, T, D))

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
        attn_out = self._mha(x, attn_mask=self.causal_mask)
        fetched = h.unsqueeze(1) + attn_out
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        chain_vecs = [h + self.bit_pos_emb.weight[0]]
        logits_list = []
        for j in range(self.dq):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self._mha(x, attn_mask=None)
            fetched = h + attn_out[:, -1, :]
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                chain_vecs.append(self.gamma * h + self.bit_val_emb(bit_val) + self.bit_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


def chain_bce_loss(raw_logits: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
    """Sum over the bit dim (nats per predicted unit), then mean over
    everything else — matches qcutelm_vlt11/qcute_refine's own convention."""
    return F.binary_cross_entropy_with_logits(raw_logits, (true_bits > 0).float(), reduction="none").sum(-1).mean()


class EncoderLevel(nn.Module):
    """Unchanged from qcute_refine.py, except forward() now genuinely uses
    its own returned h (previously discarded by every caller) — one level
    of the recursive NTP tower: embeds its own input sequence, runs a
    small causal transformer, always-on direct NTP loss on its own next
    input element (own head, own target), and every K-th position BSQ-
    quantizes into this level's own emitted code."""

    def __init__(self, cfg: Config, level: int, in_dq: int, window: int | None):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.cfg = cfg
        D = cfg.tier_d_models[level]
        self.embed = nn.Linear(in_dq, D)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.tier_n_layers[level])])
        self.ln_f = nn.LayerNorm(D)
        self.ntp_head = BitPredictHead(D, in_dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)
        self.code_pre = nn.Linear(D, cfg.dqs[level])

    def forward(self, seq_repr: torch.Tensor, compute_ntp: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """seq_repr: [B, L, in_dq]. compute_ntp=False SKIPS the ntp_head
        call entirely (real speed lever — level 0 must never pass False).
        Returns (c_i [B, n_blocks, dqs[level]], ntp_loss, ntp_acc,
        h [B, L, D] — now a first-class output, consumed by this level's
        and the next level's DecoderLevel)."""
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
            c_i = pre_q
        else:
            raise ValueError(f"unknown quant_type {cfg.quant_type!r}")
        return c_i, ntp_loss, ntp_acc, h


class DecoderLevel(nn.Module):
    """Decodes EncoderLevel[level]'s own input x^(level) by cross-
    attending from EncoderLevel[level]'s own hidden states (Q, "previous
    level['s] code LM") to EncoderLevel[level+1]'s own hidden states (KV,
    "current level['s] code LM" — EncoderLevel[level+1]'s OWN INPUT is
    exactly c^(level), so its hidden state IS the thing to attend to; no
    separate trunk is built here at all). See module docstring for the
    full causal-mask/null-KV/detach rationale."""

    def __init__(self, cfg: Config, level: int, in_dq: int):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.K = cfg.Ks[level]
        self.cfg = cfg
        D = cfg.tok_d_model
        self.q_proj = nn.Linear(cfg.tier_d_models[level], D)
        self.kv_proj = nn.Linear(cfg.tier_d_models[level + 1], D)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.null_kv, std=0.02)
        self.cross_block = CrossBlock(D, cfg.tok_n_heads, cfg.tok_mlp_mult)
        if cfg.tok_head_mode == "linear":
            self.head = nn.Linear(D, in_dq)
        elif cfg.tok_head_mode == "chain":
            self.head = BitPredictHead(D, in_dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)
        else:
            raise ValueError(f"unknown tok_head_mode {cfg.tok_head_mode!r}")

    def forward(self, h_prev: torch.Tensor, h_curr: torch.Tensor, seq_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_prev: [B, L, tier_d_models[level]] EncoderLevel[level]'s own
        hidden states (caller must have already .detach()'d this).
        h_curr: [B, n_blocks, tier_d_models[level+1]] EncoderLevel[level+1]'s
        own hidden states (also pre-detached by caller). seq_repr:
        [B, L, in_dq] level `level`'s own true input (decode target).
        Returns (loss, acc)."""
        cfg = self.cfg
        K = self.K
        B, L, _ = h_prev.shape
        n_blocks = h_curr.size(1)
        D = cfg.tok_d_model

        q = self.q_proj(h_prev)
        kv = self.kv_proj(h_curr)
        null = self.null_kv.expand(B, 1, D)
        kv = torch.cat([null, kv], dim=1)   # [B, 1+n_blocks, D] — null slot always visible

        t_idx = torch.arange(L, device=h_prev.device).unsqueeze(1)          # [L, 1]
        b_idx = torch.arange(n_blocks, device=h_prev.device).unsqueeze(0)   # [1, n_blocks]
        visible = b_idx < ((t_idx + 1) // K)                                 # [L, n_blocks] bool: block b
                                                                                # complete & visible at t
        null_col = torch.ones(L, 1, dtype=torch.bool, device=h_prev.device)
        visible = torch.cat([null_col, visible], dim=1)                      # [L, 1+n_blocks]
        disallow = ~visible                                                   # nn.MultiheadAttention bool
                                                                                # mask convention: True=blocked

        h_dec = self.cross_block(q, kv, attn_mask=disallow)                   # [B, L, D]

        h_flat = h_dec[:, :-1, :].reshape(-1, D)
        true_flat = seq_repr[:, 1:, :].reshape(-1, self.in_dq)
        if cfg.tok_head_mode == "chain":
            raw = self.head(h_flat, true_flat)
        else:
            raw = self.head(h_flat)
        loss = chain_bce_loss(raw, true_flat)
        with torch.no_grad():
            acc = ((raw > 0) == (true_flat > 0)).float().mean()
        return loss, acc


class RefineLM(nn.Module):
    """N-level recursive NTP tower (N EncoderLevels) + N-1 DecoderLevels,
    one per adjacent level pair, each a cross-attention decoder reusing
    the tower's own already-computed hidden states — see module docstring."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        assert len(cfg.tier_n_layers) == self.n_levels
        assert cfg.tok_d_model % cfg.tok_n_heads == 0

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens
        self.code_seq_lens = [seq_lens[i] // cfg.Ks[i] for i in range(self.n_levels)]

        for i, d in enumerate(cfg.tier_d_models):
            assert d % cfg.n_heads == 0, f"tier_d_models[{i}] ({d}) must be divisible by n_heads ({cfg.n_heads})"

        # resolve attn_window into one value per level — single int broadcasts (backward compatible),
        # a tuple must have length n_levels (lets e.g. the top level get its own, smaller, genuinely-
        # sub-full window instead of always inheriting whatever the finer levels use — see Config's
        # own docstring for the "coincidental dense fallback" problem this fixes).
        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels, f"attn_window tuple must have length n_levels={self.n_levels}, got {len(raw_windows)}"
        windows = [None if w == -1 else w for w in raw_windows]
        self.windows = windows
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] ({window}) must divide level {i}'s sequence length ({L}), or be >= it"

        in_dqs = [8] + list(cfg.dqs[:-1])
        self.in_dqs = in_dqs
        self.encoders = nn.ModuleList([EncoderLevel(cfg, i, in_dqs[i], windows[i]) for i in range(self.n_levels)])
        self.decoders = nn.ModuleList([DecoderLevel(cfg, i, in_dqs[i]) for i in range(self.n_levels - 1)])

        lw = cfg.layer_warmup_steps if cfg.layer_warmup_steps else (0,) * (self.n_levels - 1)
        assert len(lw) == self.n_levels - 1, (
            f"layer_warmup_steps must have length n_levels-1={self.n_levels - 1} (or be empty for "
            f"'no curriculum'), got {len(lw)}: {lw}"
        )
        self.layer_warmup_steps = lw
        activation_steps = [0]
        for w in lw:
            activation_steps.append(activation_steps[-1] + w)
        self.activation_steps = activation_steps   # length n_levels; activation_steps[0] == 0 always

    def n_active_levels(self, step: int | None) -> int:
        """step=None (eval's default, and every call site before this
        feature existed): all levels active, matching prior behavior
        exactly. Otherwise: level 0 is always active; level i (i>=1)
        becomes active once `step >= self.activation_steps[i]` — the
        layerwise curriculum (see Config.layer_warmup_steps)."""
        if step is None or not any(self.layer_warmup_steps):
            return self.n_levels
        n = 1
        for i in range(1, self.n_levels):
            if step >= self.activation_steps[i]:
                n += 1
            else:
                break
        return n

    def forward(self, byte_ids: torch.Tensor, step: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        n_active = self.n_active_levels(step)
        seq_repr = byte_to_bits(byte_ids)
        ntp_losses, ntp_accs = [], []
        h_list, x_list = [], []
        byte_loss = byte_acc = None

        for i in range(n_active):
            compute_ntp = i == 0 or cfg.code_ntp_weight > 0
            c_i, ntp_loss, ntp_acc, h_i = self.encoders[i](seq_repr, compute_ntp=compute_ntp)
            ntp_losses.append(ntp_loss)
            ntp_accs.append(ntp_acc)
            if i == 0:
                byte_loss, byte_acc = ntp_loss, ntp_acc
            h_list.append(h_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        compute_tok = cfg.tok_weight > 0
        tok_losses, tok_accs = [], []
        for i in range(n_active - 1):
            if compute_tok:
                tl, ta = self.decoders[i](h_list[i].detach(), h_list[i + 1].detach(), x_list[i])
            else:
                tl, ta = h_list[i].new_zeros(()), h_list[i].new_zeros(())
            tok_losses.append(tl)
            tok_accs.append(ta)

        ntp_total = torch.stack(ntp_losses).sum()
        tok_total = torch.stack(tok_losses).sum() if tok_losses else byte_loss.new_zeros(())
        code_ntp_total = torch.stack(ntp_losses[1:]).sum() if len(ntp_losses) > 1 else byte_loss.new_zeros(())
        loss = byte_loss + cfg.code_ntp_weight * code_ntp_total + cfg.tok_weight * tok_total
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "ntp_loss_total": ntp_total, "tok_loss_total": tok_total,
            "n_active_levels": byte_loss.new_tensor(float(n_active)),   # tensor, not plain int — every
                                                                          # metrics value gets .item()'d
                                                                          # downstream (eval_model/train)
            **{f"level{i}_ntp_loss": l for i, l in enumerate(ntp_losses)},
            **{f"level{i}_ntp_acc": a for i, a in enumerate(ntp_accs)},
            **{f"pair{i}_tok_loss": l for i, l in enumerate(tok_losses)},
            **{f"pair{i}_tok_acc": a for i, a in enumerate(tok_accs)},
        }
        return loss, metrics


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor) -> torch.Tensor:
    logits = model.encoders[0].ntp_head(h_last, true_bits=None)
    return bits_to_byte(logits)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Reference (slow, obviously-correct) byte-by-byte generation:
    recomputes EncoderLevel_0 from scratch over the WHOLE sequence every
    new byte. Only EncoderLevel_0's own NTP head is generative (see module
    docstring) — DecoderLevel never participates."""
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
    """KV-cache-efficient generation, identical mechanism to qcute_refine.py's
    own generate_kv_cache — dense-attention only for level 0 specifically
    (the only level generation ever touches; other levels' own windows
    don't matter here)."""
    cfg = model.cfg
    assert model.windows[0] is None, "generate_kv_cache only supports dense attention at level 0 (attn_window[0] must be -1) — see docstring"
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
        return enc0.ln_f(x).squeeze(1)

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
def eval_model(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str, step: int | None = None) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx, step=step)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    return result


def build_param_groups(model: RefineLM) -> list[dict]:
    """One param group per activation STAGE (stage 0 = encoders[0] alone;
    stage i>=1 = encoders[i] + tokenizers[i-1], the pair that turns on
    together once level i activates — see Config.layer_warmup_steps) —
    lets train() give each stage its own reset warmup schedule. With no
    curriculum (layer_warmup_steps empty), every stage activates at step 0
    and this is behaviorally identical to one global param group."""
    groups = [{"params": list(model.encoders[0].parameters()), "stage": 0}]
    for i in range(1, model.n_levels):
        params = list(model.encoders[i].parameters()) + list(model.decoders[i - 1].parameters())
        groups.append({"params": params, "stage": i})
    return groups


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v2", dynamic_ncols=True)
    for step in pbar:
        # each stage gets its OWN warmup, reset to 0 at that stage's own activation_steps[stage] — the
        # SAME shared lr_at/lr_at_warmup_constant_cosine functions every other run uses, just called with
        # a per-stage relative step instead of the global one. Stages not yet active get lr=0 (their
        # params also aren't in this step's forward graph at all, so grad is None -> AdamW skips them
        # regardless; lr=0 here is belt-and-suspenders, not load-bearing).
        for g in opt.param_groups:
            rel_step = step - model.activation_steps[g["stage"]]
            if rel_step < 0:
                lr = 0.0
            elif args.cosine_decay:
                lr = lr_at_warmup_constant_cosine(rel_step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
            else:
                lr = lr_at(rel_step, args.warmup_steps, args.lr_peak)
            g["lr"] = lr
        lr = opt.param_groups[0]["lr"]   # stage-0 lr, for logging/postfix — always active, always representative

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx, step=step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
            tok_loss=f"{metrics['tok_loss_total'].item():.4f}",
        )

        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), bpb=train_bpb, byte_acc=metrics["byte_acc"].item())

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device, step=step)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def _broadcast_int_tuple(s, n: int) -> tuple[int, ...]:
    t = _parse_int_tuple(s)
    if len(t) == 1:
        return t * n
    assert len(t) == n, f"expected 1 (broadcast) or {n} values, got {len(t)}: {t}"
    return t


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Recursive NTP tower + cross-attention DecoderLevel (qcute_refine_v2)", parents=[pre])
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--Ks", default=(2, 2, 2))
    p.add_argument("--tier_n_layers", default=(1, 1, 1))
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(96, 96, 96))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=128)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "identity"])
    p.add_argument("--tok_d_model", type=int, default=96)
    p.add_argument("--tok_n_heads", type=int, default=4)
    p.add_argument("--tok_mlp_mult", type=int, default=4)
    p.add_argument("--tok_head_mode", type=str, default="linear", choices=["linear", "chain"])
    p.add_argument("--tok_weight", type=float, default=1.0)
    p.add_argument("--layer_warmup_steps", type=lambda s: () if s in ("", "()") else _parse_int_tuple(s), default=())

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
        rope_base=args.rope_base, bit_chain_n_heads=args.bit_chain_n_heads, bit_chain_gamma=args.bit_chain_gamma,
        bit_chain_fixed_kernel=args.bit_chain_fixed_kernel, code_ntp_weight=args.code_ntp_weight,
        quant_type=args.quant_type, tok_d_model=args.tok_d_model, tok_n_heads=args.tok_n_heads,
        tok_mlp_mult=args.tok_mlp_mult, tok_head_mode=args.tok_head_mode, tok_weight=args.tok_weight,
        layer_warmup_steps=args.layer_warmup_steps,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v2_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} tier_n_layers={cfg.tier_n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} tok_head_mode={cfg.tok_head_mode} "
        f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
