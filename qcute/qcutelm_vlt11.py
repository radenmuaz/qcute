"""qcute.qcutelm_vlt11 — recursive Pass1/Pass2 sandwich: at every level,
an UNCONDITIONAL encoder (E_i) compresses its own sequence into a code,
and a SEPARATE, conditioned decoder (D_i) — own weights, own embedding,
NOT reading E_i's hidden states — predicts the NEXT element of that same
sequence using the code as periodic conditioning. Level i+1's entire
input sequence is level i's code — genuine compression in sequence
length at every level (not just in per-position content), matching
qcutelm_vlt7/vlt8's original narrow/wide compute argument, applied
recursively instead of just once.

Confirmed via session graph-check (arrows only, no prose, ending "yes"):

    b_t ──E0──▶ h0a_t ──q──▶ c0_t
                              │
           c0_t ──D0──▶ h0b_t ──▶ b_(t+1)    (vs b_(t+1))

    c0_t ──E1──▶ h1a_t ──q──▶ c1_t
                              │
           c1_t ──D1──▶ h1b_t ──▶ c0_(t+1)   (vs c0_(t+1))

Level 0: E0 = unconditional causal byte LM (own byte_emb) -> code_pre[0]
  reads E0's hidden state at every Ks[0]-th position -> quantize -> c0 (a
  sequence of length context_len/Ks[0] — level 1's ENTIRE input). D0 =
  a SEPARATE tier (its own byte_emb, own attention weights — NOT E0's
  hidden states, "no h0") that embeds the SAME raw bytes but with c0's
  CAUSAL FORECAST (codelm[0], built from past c0 values only — never the
  true c0, which would leak this-block's-own future bytes) substituted at
  each block-start position -> predicts the NEXT BYTE, genuine NTP,
  teacher-forced against the true next byte. This is exactly
  qcutelm_vlt7/vlt8's Pass1 (unconditional/producer) + Pass2
  (conditional/consumer) pair — with separate weights by construction,
  matching vlt8's own finding that sharing weights between these two
  different *functions* measurably hurt.

Level i>0: identical shape, one level up. E_i is an unconditional causal
  LM whose entire sequence is level (i-1)'s code, c_{i-1} (own linear
  embed of that continuous code — NOT E_{i-1}'s hidden states). D_i is a
  separate tier, its own embed of the SAME c_{i-1} sequence, conditioned
  on c_i's causal forecast, predicting the NEXT c_{i-1} value (NTP one
  level of abstraction up — loss is the same BCE/bit or CE/dim mechanism
  every earlier fork's code_match_loss uses, since the "vocabulary" here
  is a continuous quantized code, not a discrete byte).

Each level's own D_i loss trains that level directly; the compounding
hierarchical effect on BYTE prediction is indirect (level i+1's loss
backpropagates through c_i, which is E_i's own output, shaping E_i's —
and therefore c_i's — learned representations) rather than a direct
cross-level conditioning chain into D_0. This was an explicit, confirmed
design choice this session, not an oversight: D_i is conditioned ONLY on
its own level's code, nothing coarser.

Genuine compute savings, unlike this file's earlier (qcutelm_vlt10-
style) version: level i's sequence length is context_len /
(Ks[0]*...*Ks[i-1]) — actually shrinking at each level, not staying at
byte-length throughout. This is what "no h0" bought architecturally: once
D_i has its own embedding of its own level's sequence, it never needs
anything from a finer level's hidden states, so there's no reason to keep
every level at byte-length the way qcutelm_vlt10's substitution mechanism
required.

No shared imports with qcutelm_vlt/vlt2/.../vlt10 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers/quantizers duplicated.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_<name>.py
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
    Ks: tuple[int, ...] = (4, 4, 4)           # LOCAL compression factor per level (level i's OWN
                                                # sequence shrinks by Ks[i]) — not cumulative, unlike
                                                # qcutelm_vlt10's periods
    dqs: tuple[int, ...] = (8, 8, 8)           # one entry per level — this level's own code dimension
    tier_d_models: tuple[int, ...] = (96, 96, 96)  # one entry per level (len(Ks)) — E_i and D_i share
                                                # this dim (separate weights, same width)
    context_len: int = 1024
    quant_type: str = "ifsq"  # "bsq"/"fsq"/"ifsq": real discretized codes (STE gradient). "identity":
                                # NO quantization at all — code_pre's raw continuous output IS c_i,
                                # unbounded, no rounding. Ceiling-baseline diagnostic (session: "quant
                                # identity first as ceiling baseline... highest repr power") — isolates
                                # whether hard discretization is the slow-convergence bottleneck by
                                # removing it entirely; not a real operating mode (codes aren't
                                # comparable/transmittable the way bsq/fsq's bounded, roundable codes
                                # are). generate_no_cache/generate_kv_cache do NOT support it — training-
                                # only, for the convergence-speed comparison in docs/status.md.
    fsq_levels: int = 8
    vocab: int = 256
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    attn_window: int = 64     # windowed by DEFAULT now (-1 = dense is the opt-out flag). Effective reach
                                # per layer is 2*window (each chunk attends to itself + the previous
                                # chunk only, not a rolling window), roughly 2*window*n_layers after
                                # stacking — NOT the full sequence length; long-range signal beyond that
                                # is expected to travel through the code hierarchy, not raw attention.
                                # Must evenly divide every level's own (shrinking) sequence length; 64 is
                                # the largest value that divides [1024,256,64] (the default Ks/context_len
                                # config's tier lengths) — gcd(1024,256,64)=64.
    lm_d_model: int = 128
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    lm_attn_window: int = 16   # windowed by default; 16 = gcd(256,64,16), the largest value dividing
                                # the default config's per-level codelm sequence lengths. -1 = dense.
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    byte_repr: str = "bits"       # "bits" (default): level 0 treats a byte as its own native 8-bit BSQ-
                                    # like code (deterministic bit-extraction, no learned embedding table;
                                    # D0 predicts 8 independent/chained bit-logits, BCE) — unifies level 0
                                    # with every coarser level's own representation, instead of a separate
                                    # 256-way categorical special case. "softmax": the original nn.Embedding
                                    # + 256-way CE fallback.
    bit_head_mode: str = "chain"   # "chain" (default): Fetch-style causal self-attention chain across
                                    # the dq bits (ported from qcute_fifo/qcute_bytepool's FetchHead) —
                                    # each bit additionally conditions on previously-decided bits, the
                                    # mathematically exact chain-rule factorization instead of assuming
                                    # independence. Applies to D0's byte-bit head (byte_repr="bits") and
                                    # to bsq-mode coarser-level heads (D_i i>0, and CodeLM) — fsq/ifsq
                                    # levels stay softmax-per-dim regardless, this flag doesn't apply
                                    # there. "independent": dq parallel sigmoid bits, no cross-bit
                                    # conditioning (every earlier fork's BSQ convention; cheaper, no
                                    # per-bit self-attention loop).
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True   # True (default): BitPredictHead's chain mode uses one fixed-
                                    # shape, causally-masked self-attention call per teacher-forced
                                    # position instead of a dq-step Python loop — same math, much better
                                    # kernel utilization at training scale. False: the old incremental
                                    # loop (kept for direct comparison). Only affects training (true_bits
                                    # given) — greedy generation always uses the loop, since future bits
                                    # genuinely aren't known yet.
    e_ntp_weight: float = 0.0     # 0.0 (default): off, matching every other optional-loss convention in
                                    # this codebase. >0: gives each E_i its own direct, UNCONDITIONED NTP
                                    # loss (own head, same targets D_i uses) — the "detach-teacher-force"
                                    # fix for the cyclic-target problem (session discussion: E_i has no
                                    # loss of its own otherwise, trained only indirectly through 3 paths
                                    # that all compete to shape the same hidden state, none of which
                                    # directly constrains c_i to stay informative rather than collapsing —
                                    # observed live in qcutelm_vlt11_k2_l3_full's own training: level1/2
                                    # accuracy hit 100% by step 499 while byte-level accuracy sat at 78%).
                                    # When >0, h_a is stop-gradiented specifically at the code_pre input
                                    # (see forward()'s comment) so E_i's own NTP objective and the code-
                                    # matching objectives don't compete over the same hidden state.
    e_ntp_every: int = 1          # 1 (default): compute e_ntp_loss every training step. >1: only every
                                    # N-th step (amortization strategy — see docs/status.md's "Strategies
                                    # to amortize the multi-loss cost": running e_ntp_weight>0 every step
                                    # was measured to plateau training at ~0.5-0.55it/s on MPS, roughly
                                    # half of the depth-4 bytelm baseline's 1.13it/s, despite qcutelm_vlt11
                                    # doing 4-9x FEWER FLOPs/step — i.e. it's overhead-bound (many small
                                    # sub-network/kernel-launch calls), not compute-bound, so skipping the
                                    # extra head's forward/backward on most steps directly cuts wall-clock
                                    # without touching FLOPs/step on the steps that do run it. Has no
                                    # effect when e_ntp_weight==0. eval_model() always computes it (step=
                                    # None passed to forward()) regardless of this — eval runs infrequently
                                    # (every eval_every steps) so its cost doesn't need amortizing, and
                                    # skipping it there would make val metrics noisier without saving much.
    e_ntp_bit_head_mode: str | None = None   # None (default): head_e0/head_e_code use the same
                                    # bit_head_mode as D_i's heads (head0/head_code) — original behavior.
                                    # "independent"/"chain": override just the E-side heads to a cheaper
                                    # (or different) mode than D_i's, independent of bit_head_mode. Second
                                    # amortization strategy: e_ntp_loss's job is only to prevent c_i
                                    # collapse, not to nail exact bit-chain calibration, so a cheaper head
                                    # there (no per-bit chain self-attention) is a reasonable quality-for-
                                    # speed trade, and directly targets BitPredictHead's chain-mode self-
                                    # attention (already a many-small-ops cost even for D_i alone) being
                                    # invoked a SECOND time per level as the suspected dominant overhead
                                    # source behind e_ntp_weight's wall-clock hit.
    byte_only: bool = False       # False (default): the full recursive sandwich. True: train D_0
                                    # ALONE as a plain dense byte LM — no E_0, no code_pre/quantize/
                                    # codelm[0], no forecast substitution into D_0's input (pure
                                    # teacher-forced real embeddings, exactly matching bytelm's
                                    # problem), no levels 1+ at all. Session: "restart and disable all
                                    # other losses, full unconstrained but byte ntp loss" — a stricter
                                    # isolation than quant_type="identity" (which still runs the full
                                    # hierarchy/substitution, just without discretization): this mode
                                    # isolates the "tier_d_models[0]=96 is under half of bytelm's
                                    # d_model=256" capacity hypothesis on its own, with zero
                                    # multi-objective competition and zero substitution-corruption
                                    # confound. Ignores quant_type/e_ntp_weight/code_match_weight
                                    # entirely (nothing downstream of D_0 ever runs). Training-only
                                    # diagnostic — generate_no_cache/generate_kv_cache don't special-
                                    # case it and would break (they assume the full hierarchy exists).
    share_across_levels: bool = False   # False (default, ORIGINAL v11 behavior, unchanged): each
                                    # level gets its own independent E_i/D_i/codelm[i]/code_pre[i]/
                                    # z_proj[i]/head_code[i] — tier_d_models[i]/dqs[i] may differ freely
                                    # per level. True: ONE shared copy of each, reused at every level
                                    # (qcute_fifo's "one shared model" idea folded directly into v11,
                                    # session: "the 3 separate e d codelm, merge to v11" — as opposed to
                                    # qcutelm_vlt12.py's more radical single-LM-no-forecast redesign,
                                    # this keeps v11's whole Pass1(E)/Pass2(D)/codelm-forecast mechanism
                                    # exactly as-is, just ties the SAME weights across levels instead of
                                    # giving each level its own). Requires tier_d_models and dqs to each
                                    # be uniform across all levels (asserted in __init__ — "constrain
                                    # each layer must be same dim hparam"); the shared width/code-dim is
                                    # then simply tier_d_models[0]/dqs[0] — no separate code_dim
                                    # override flag here (unlike qcutelm_vlt12.py's Config.code_dim);
                                    # keep dqs itself uniform if a different code size is wanted.


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


def rope_cos_sin_at(pos_id: int, head_dim: int, base: float, device: torch.device):
    """Same as rope_cos_sin but for a single, explicit (non-contiguous-
    from-0) position — needed for KV-cached single-step generation."""
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

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        """KV-cached single-position step: x_new [B,1,D], cos_new/sin_new for
        this exact RoPE position. cache_k/cache_v: [B,H,T_so_far,hd] or None
        (first step). Dense attention over the whole growing cache — ignores
        self.window (caching+windowing not implemented together in this
        fork; see generate_kv_cache's docstring). Returns (out [B,1,D],
        new_cache_k, new_cache_v)."""
        B, _, D = x_new.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        y = F.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)   # q = only the new position;
                                                                                  # attending to all of (past, self)
                                                                                  # in the cache is exactly causal
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
    """[*] long byte ids (0..255) -> [*, 8] float, each bit in {-1,+1}/sqrt(8)
    (BSQ's own normalization convention, dq=8) — LSB-first, deterministic,
    no learned parameters. A byte losslessly IS its own 8-bit BSQ-shaped
    code; this makes that literal instead of routing it through a learned
    256-way embedding table."""
    bits = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    return (2 * bits - 1) / math.sqrt(8)


def bits_to_byte(bits: torch.Tensor) -> torch.Tensor:
    """[*, 8] float (positive=bit 1, non-positive=bit 0) -> [*] long byte ids. Inverse of byte_to_bits."""
    b = (bits > 0).long()
    powers = (2 ** torch.arange(8, device=bits.device))
    return (b * powers).sum(-1)


class BitPredictHead(nn.Module):
    """Predicts `dq` independent-or-chained bits from a hidden vector.
    mode="independent": one Linear(d_model, dq) — dq parallel sigmoid
    logits, no cross-bit conditioning (every earlier fork's BSQ default).
    mode="chain": Fetch-style causal self-attention chain (ported from
    qcute_fifo.FetchHead / qcute_bytepool.FetchHead) — bit j's logit
    additionally conditions on bits 0..j-1, the mathematically exact
    chain-rule factorization of the joint dq-bit distribution instead of
    an independence assumption.

    chain_fixed_kernel=True (default): when `true_bits` is given (training,
    teacher-forced), builds ONE fixed-length dq-position sequence up front
    — position 0 = a zero vector (nothing precedes the first bit), position
    j>0 = the embedding of true bit j-1 — and runs a SINGLE causally-masked
    self-attention call over it, predicting all dq bits at once. This
    replaces the mathematically-equivalent but much slower dq-step Python
    loop (dq separate variable-shaped self-attention calls per item) with
    one fixed-shape, kernel-friendly batched call — the loop version can't
    batch across chain steps at all, which matters a lot at training scale
    (every position of every item in the batch pays dq sequential calls).
    Falls back to the incremental loop whenever true_bits is None (greedy
    generation — future bits genuinely aren't known yet, so there's nothing
    to batch) or when chain_fixed_kernel=False (kept for direct comparison/
    debugging)."""

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
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        """h: [N, D]. true_bits: [N, dq] float in {-1,+1}-ish (teacher-forcing,
        training) or None (greedy at inference, chain mode only). ->
        raw_logits [N, dq]."""
        if self.mode == "independent":
            return self.head(h)
        if self.chain_fixed_kernel and true_bits is not None:
            return self._forward_chain_fixed(h, true_bits)
        return self._forward_chain_loop(h, true_bits)

    def _forward_chain_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        bit_ids = (true_bits > 0).long()                              # [N, dq]
        val_embeds = self.bit_val_emb(bit_ids)                         # [N, dq, D]
        zero_vec = val_embeds.new_zeros(N, 1, D)
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)  # position j holds bit j-1's
                                                                          # embedding, or a zero vector
                                                                          # preceding position 0
        pos = self.bit_pos_emb.weight.unsqueeze(0)                     # [1, dq, D]
        # position 0 gets UNSCALED h (matches _forward_chain_loop's chain_vecs[0] = h + pos_emb[0], no
        # gamma there); positions 1..dq-1 get gamma-scaled h (matches the loop's appended entries) —
        # these two implementations must compute the identical function whenever gamma != 1.0, or the
        # "same math, faster kernel" equivalence this class relies on silently breaks.
        h_scale = h.new_ones(1, self.dq, 1)
        if self.dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        x = h_scale * h.unsqueeze(1) + shifted + pos                    # [N, dq, D]
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self.causal_mask, need_weights=False)
        fetched = h.unsqueeze(1) + attn_out                             # [N, dq, D]
        return self.head(fetched).squeeze(-1)                           # [N, dq]

    def _forward_chain_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
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


class CodeLM(nn.Module):
    """Separate weights per level, operates strictly on that level's own
    short code sequence. Forecasts code[i+1] from codes[:i] (causal). No
    loss of its own; code_match_loss (computed in the parent model,
    against a detached target) trains it — identical mechanism to every
    earlier fork's CodeLM."""

    def __init__(self, cfg: Config, dq: int):
        super().__init__()
        self.cfg = cfg
        self.dq = dq
        window = None if cfg.lm_attn_window == -1 else cfg.lm_attn_window
        self.in_proj = nn.Linear(dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=window) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        if factorized_softmax:
            self.pred_head = nn.Linear(cfg.lm_d_model, dq * cfg.fsq_levels)
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)
        elif cfg.quant_type == "identity":
            # continuous unbounded regression head — no bits, no levels, just predict the code
            # vector directly. See Config.quant_type's docstring for why this exists (ceiling
            # baseline: max representational power, isolates whether discretization itself is
            # the slow-convergence bottleneck).
            self.pred_head = nn.Linear(cfg.lm_d_model, dq)
        else:
            self.pred_head = BitPredictHead(cfg.lm_d_model, dq, cfg.bit_head_mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        x = self.in_proj(z_hat)
        head_dim = cfg.lm_d_model // cfg.lm_n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)
        if cfg.quant_type in ("fsq", "ifsq"):
            raw = self.pred_head(h)
            B, T, _ = raw.shape
            logits = raw.view(B, T, self.dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1), logits
        if cfg.quant_type == "identity":
            raw = self.pred_head(h)   # [B, T, dq] — the prediction itself IS the soft code, no
            return raw, raw            # squashing function needed for an unbounded regression target
        B, T, D = h.shape
        h_flat = h.reshape(B * T, D)
        if cfg.bit_head_mode == "chain":
            # positions 0..T-2 predict z_hat[t+1], which exists -> teacher-force properly. Position
            # T-1's target does NOT exist yet (it's the genuine forecast, used both by generation and,
            # during training, sliced off by callers before any loss) — it must be greedy-decoded
            # (true_bits=None), NOT zero-padded: padding with zeros would silently compute the WRONG
            # forecast there (this was a real bug — generate_no_cache used to diverge from
            # generate_kv_cache's correct greedy decode at exactly this position, caught by comparing
            # the two directly).
            if T > 1:
                h_bulk = h[:, :-1, :].reshape(B * (T - 1), D)
                true_bits_bulk = z_hat[:, 1:, :].reshape(B * (T - 1), self.dq)
                raw_bulk = self.pred_head(h_bulk, true_bits_bulk).reshape(B, T - 1, self.dq)
            else:
                raw_bulk = z_hat.new_zeros(B, 0, self.dq)
            raw_last = self.pred_head(h[:, -1, :], None).reshape(B, 1, self.dq)
            raw = torch.cat([raw_bulk, raw_last], dim=1)
        else:
            raw = self.pred_head(h_flat).reshape(B, T, self.dq)
        return 2 * torch.sigmoid(raw) - 1, raw


class RecursiveSandwichLM(nn.Module):
    """N-level recursive Pass1(E)/Pass2(D) stack. See module docstring for
    the full "no h0" rationale and the confirmed session graph."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        self.n_shared = 1 if cfg.share_across_levels else self.n_levels
        if cfg.share_across_levels:
            assert len(set(cfg.tier_d_models)) == 1, (
                f"share_across_levels=True requires uniform tier_d_models — got {cfg.tier_d_models}"
            )
            assert len(set(cfg.dqs)) == 1, f"share_across_levels=True requires uniform dqs — got {cfg.dqs}"
        window = None if cfg.attn_window == -1 else cfg.attn_window

        # level-i sequence lengths, shrinking: L_0 = context_len, L_i = L_{i-1} / Ks[i-1]
        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens   # length n_levels

        for i, d in enumerate(cfg.tier_d_models):
            assert d % cfg.n_heads == 0, f"tier_d_models[{i}] ({d}) must be divisible by n_heads ({cfg.n_heads})"
        if window is not None:
            for i, L in enumerate(seq_lens):
                # L <= window is fine too: CausalSelfAttention.forward only takes the chunked path when
                # T > self.window, else it falls back to dense — so a window wider than this level's own
                # sequence just means "dense at this level," not an error (see the pyramid fork's own
                # windowing-silent-fallback finding for why this distinction matters to get right).
                assert L % window == 0 or L <= window, (
                    f"attn_window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
                )
        if cfg.lm_attn_window != -1:
            for i, L in enumerate(seq_lens):
                n_blocks_i = L // cfg.Ks[i]
                assert n_blocks_i % cfg.lm_attn_window == 0, f"lm_attn_window must divide level {i}'s n_blocks ({n_blocks_i})"

        assert cfg.byte_repr in ("bits", "softmax")
        assert cfg.bit_head_mode in ("independent", "chain")

        # level 0: byte representation — "bits" (default): deterministic 8-bit BSQ-shaped code, no
        # learned embedding table, just a linear projection of byte_to_bits(); "softmax": the original
        # nn.Embedding lookup (fallback/ablation). E0's and D0's own copies stay separate either way,
        # matching qcutelm_vlt8's untied Pass1/Pass2 finding.
        if cfg.byte_repr == "bits":
            self.byte_bits_proj = nn.Linear(8, cfg.tier_d_models[0])
            self.dec_byte_bits_proj = nn.Linear(8, cfg.tier_d_models[0])
        else:
            self.byte_emb = nn.Embedding(cfg.vocab, cfg.tier_d_models[0])
            self.dec_byte_emb = nn.Embedding(cfg.vocab, cfg.tier_d_models[0])
        # level i>0: linear embed of c_{i-1} (continuous, dqs[i-1]-dim) — separate E_i/D_i embeds.
        # embed[0]/dec_embed[0] stay unused None placeholders either way (level 0 uses byte_bits_proj/
        # byte_emb instead); share_across_levels=True collapses the i>=1 entries down to ONE shared
        # module at index 1, indexed via self._sel1(i) rather than i directly.
        embed_range = range(1, 2) if cfg.share_across_levels else range(1, self.n_levels)
        self.embed = nn.ModuleList([None] + [nn.Linear(cfg.dqs[i - 1], cfg.tier_d_models[i]) for i in embed_range])
        self.dec_embed = nn.ModuleList([None] + [nn.Linear(cfg.dqs[i - 1], cfg.tier_d_models[i]) for i in embed_range])

        self.e_blocks = nn.ModuleList([nn.ModuleList([Block(cfg.tier_d_models[i], cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)]) for i in range(self.n_shared)])
        self.e_ln_f = nn.ModuleList([nn.LayerNorm(cfg.tier_d_models[i]) for i in range(self.n_shared)])
        self.d_blocks = nn.ModuleList([nn.ModuleList([Block(cfg.tier_d_models[i], cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)]) for i in range(self.n_shared)])
        self.d_ln_f = nn.ModuleList([nn.LayerNorm(cfg.tier_d_models[i]) for i in range(self.n_shared)])

        self.code_pre = nn.ModuleList([nn.Linear(cfg.tier_d_models[i], cfg.dqs[i]) for i in range(self.n_shared)])
        self.z_proj = nn.ModuleList([nn.Linear(cfg.dqs[i], cfg.tier_d_models[i]) for i in range(self.n_shared)])
        self.codelm = nn.ModuleList([CodeLM(cfg, cfg.dqs[i]) for i in range(self.n_shared)])

        # level 0's decode head predicts bytes; level i>0's decode head predicts next c_{i-1} (raw
        # logits, same BCE/bit-or-CE/dim shape as code_match_loss). "bits"/bsq-mode heads use
        # BitPredictHead (independent-or-chain, per bit_head_mode); fsq/ifsq stay plain softmax-per-dim
        # (the chain flag doesn't apply there — a "level" isn't a bit). head_code/head_e_code follow the
        # same None-at-0-plus-shared-at-1 convention as embed/dec_embed above.
        self.head0 = self._build_head0(cfg.bit_head_mode)
        head_code_range = range(1, 2) if cfg.share_across_levels else range(1, self.n_levels)
        self.head_code = nn.ModuleList([None] + [self._build_head_code(i, cfg.bit_head_mode) for i in head_code_range])
        if cfg.e_ntp_weight > 0:
            # E_i's own unconditioned NTP head — separate weights from D_i's, same shape. Mode can be
            # overridden cheaper via e_ntp_bit_head_mode (see Config docstring — amortization strategy).
            e_mode = cfg.e_ntp_bit_head_mode if cfg.e_ntp_bit_head_mode is not None else cfg.bit_head_mode
            self.head_e0 = self._build_head0(e_mode)
            self.head_e_code = nn.ModuleList([None] + [self._build_head_code(i, e_mode) for i in head_code_range])

    def _sel(self, i: int) -> int:
        """Index into a length-n_levels (untied) or length-1 (shared) ModuleList — e_blocks/e_ln_f/
        d_blocks/d_ln_f/code_pre/z_proj/codelm all follow this convention (real entries starting at 0)."""
        return 0 if self.cfg.share_across_levels else i

    def _sel1(self, i: int) -> int:
        """Same idea, for the None-at-0-plus-real-entries-from-1 convention (embed/dec_embed/head_code/
        head_e_code) — only ever called with i>=1."""
        return 1 if self.cfg.share_across_levels else i

    def _build_head0(self, mode: str) -> nn.Module:
        cfg = self.cfg
        if cfg.byte_repr == "bits":
            return BitPredictHead(cfg.tier_d_models[0], 8, mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)
        return nn.Linear(cfg.tier_d_models[0], cfg.vocab)

    def _build_head_code(self, i: int, mode: str) -> nn.Module:
        cfg = self.cfg
        if cfg.quant_type in ("fsq", "ifsq"):
            return nn.Linear(cfg.tier_d_models[i], cfg.dqs[i - 1] * cfg.fsq_levels)
        if cfg.quant_type == "identity":
            return nn.Linear(cfg.tier_d_models[i], cfg.dqs[i - 1])
        return BitPredictHead(cfg.tier_d_models[i], cfg.dqs[i - 1], mode, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel)

    def run_blocks(self, blocks: nn.ModuleList, ln_f: nn.LayerNorm, level: int, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.tier_d_models[level] // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in blocks:
            x = block(x, cos, sin)
        return ln_f(x)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, v.size(-1))
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        elif self.cfg.quant_type == "identity":
            # no discretization at all — pure passthrough, unbounded continuous code. Ceiling
            # baseline: isolates whether hard quantization is the slow-convergence bottleneck by
            # removing it entirely (max representational power a code_pre readout could have).
            return v
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def code_level_loss(self, raw_logits: torch.Tensor, true_code: torch.Tensor) -> torch.Tensor:
        """Same mechanism as code_match_loss (BCE/bit for bsq, CE/dim for
        fsq/ifsq) — used for D_i's (i>0) next-code loss too, since its
        target is also a continuous quantized code, not a discrete byte."""
        cfg = self.cfg
        true_code = true_code.detach()
        if cfg.quant_type in ("fsq", "ifsq"):
            half_l = (cfg.fsq_levels - 1) / 2
            true_level = torch.round(true_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
            dq = true_code.size(-1)
            logits = raw_logits.view(*raw_logits.shape[:-1], dq, cfg.fsq_levels)
            return F.cross_entropy(logits.reshape(-1, cfg.fsq_levels), true_level.reshape(-1))
        if cfg.quant_type == "identity":
            return F.mse_loss(raw_logits, true_code)
        true_bits = (true_code > 0).float()
        return F.binary_cross_entropy_with_logits(raw_logits, true_bits, reduction="none").sum(-1).mean()

    def _ntp_loss_and_acc(self, h: torch.Tensor, i: int, seq: torch.Tensor, is_e_side: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by D_i's own (conditioned) NTP loss and, when
        e_ntp_weight>0, E_i's own (unconditioned) NTP loss — structurally
        identical except which hidden state / head pair is used. h: [B, L,
        tier_d_models[i]]. seq: this level's own input sequence (byte ids
        for i=0, codes for i>0) — targets are seq shifted by one."""
        cfg = self.cfg
        B, L, D = h.shape
        h_flat = h[:, :-1, :].reshape(-1, D)
        if i == 0:
            head = self.head_e0 if is_e_side else self.head0
            if cfg.byte_repr == "bits":
                true_bits_flat = byte_to_bits(seq[:, 1:]).reshape(-1, 8)
                logits = head(h_flat, true_bits_flat) if head.mode == "chain" else head(h_flat)
                # sum over the 8-bit dim first (nats PER BYTE), THEN mean over positions — reduction=
                # 'mean' would average over bits too, silently reporting nats-per-bit as nats-per-byte.
                loss = F.binary_cross_entropy_with_logits(logits, (true_bits_flat > 0).float(), reduction="none").sum(-1).mean()
                with torch.no_grad():
                    acc = ((logits > 0) == (true_bits_flat > 0)).float().mean()
            else:
                logits = head(h)                                           # [B, L, vocab]
                target = torch.full((B, L), -100, dtype=torch.long, device=seq.device)
                target[:, :-1] = seq[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), target.reshape(-1), ignore_index=-100)
                with torch.no_grad():
                    valid = target != -100
                    acc = (logits.argmax(-1) == target)[valid].float().mean()
            return loss, acc

        head = self.head_e_code[self._sel1(i)] if is_e_side else self.head_code[self._sel1(i)]
        true_next_seq = seq[:, 1:, :]                                      # [B, L-1, dqs[i-1]]
        if cfg.quant_type in ("fsq", "ifsq"):
            raw = head(h_flat).reshape(B, L - 1, -1)
        elif cfg.quant_type == "identity":
            raw = head(h_flat).reshape(B, L - 1, -1)   # plain Linear, no true_bits/chain arg
        else:
            true_flat = true_next_seq.reshape(-1, true_next_seq.size(-1))
            raw_flat = head(h_flat, true_flat) if head.mode == "chain" else head(h_flat)
            raw = raw_flat.reshape(B, L - 1, -1)
        loss = self.code_level_loss(raw, true_next_seq)
        with torch.no_grad():
            if cfg.quant_type in ("fsq", "ifsq"):
                half_l = (cfg.fsq_levels - 1) / 2
                true_lvl = torch.round(true_next_seq.detach() * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                pred_lvl = raw.view(*raw.shape[:-1], true_next_seq.size(-1), cfg.fsq_levels).argmax(-1)
                acc = (pred_lvl == true_lvl).float().mean()
            elif cfg.quant_type == "identity":
                # no natural "accuracy" for unbounded regression — report fraction of variance
                # explained (R^2-like) as a proxy, purely diagnostic.
                true_d = true_next_seq.detach()
                mse = F.mse_loss(raw, true_d)
                acc = 1 - mse / (true_d.var() + 1e-8)
            else:
                acc = ((raw > 0) == (true_next_seq.detach() > 0)).float().mean()
        return loss, acc

    def forward(self, ctx: torch.Tensor, step: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B = ctx.size(0)
        seq = ctx   # level 0's sequence: byte ids [B, L_0]

        if cfg.byte_only:
            # D_0 alone — plain teacher-forced dense byte LM, no E_0/code_pre/quantize/codelm[0], no
            # forecast substitution, no levels 1+. See Config.byte_only's docstring.
            if cfg.byte_repr == "bits":
                d_in = self.dec_byte_bits_proj(byte_to_bits(seq))
            else:
                d_in = self.dec_byte_emb(seq)
            h_b = self.run_blocks(self.d_blocks[0], self.d_ln_f[0], 0, d_in)
            loss, acc = self._ntp_loss_and_acc(h_b, 0, seq, is_e_side=False)
            return loss, {"loss": loss, "byte_loss": loss, "byte_acc": acc}

        # e_ntp_every amortization: step=None (eval_model's call site) always computes it — eval runs
        # infrequently so its cost doesn't need amortizing. During training, only every e_ntp_every-th
        # step pays for the extra head forward/backward; see Config.e_ntp_every's docstring.
        compute_e_ntp = cfg.e_ntp_weight > 0 and (step is None or cfg.e_ntp_every <= 1 or step % cfg.e_ntp_every == 0)

        level_losses = []
        level_accs = []
        code_match_losses = []
        e_ntp_losses = []
        e_ntp_accs = []
        byte_loss = None
        byte_acc = None

        for i in range(self.n_levels):
            K = cfg.Ks[i]
            L = self.seq_lens[i]
            D = cfg.tier_d_models[i]
            n_blocks = L // K

            # ENCODE — unconditional, own weights
            if i == 0:
                e_in = self.byte_bits_proj(byte_to_bits(seq)) if cfg.byte_repr == "bits" else self.byte_emb(seq)
            else:
                e_in = self.embed[self._sel1(i)](seq)
            h_a = self.run_blocks(self.e_blocks[self._sel(i)], self.e_ln_f[self._sel(i)], i, e_in)

            # detach-teacher-force fix (session: formalized, now implemented): when E_i has its own
            # direct NTP loss, stop-gradient h_a specifically at the code_pre input, so code_match_loss/
            # D_i's substitution — both downstream of c_i — cannot ALSO reshape h_a. Without this, two
            # different objectives compete to shape the same hidden state, and code_match_loss's target
            # is itself a model output (unlike the E-side NTP loss's real-data target), so the joint
            # optimum can be a collapsed/degenerate c_i that still looks fine to code_match_loss alone
            # (the qcutelm_vlt8_bsq_dense_supervision failure mode). With e_ntp_weight=0 there is no
            # competing objective, so the detach would only cut E_i's one remaining gradient path — skip
            # it then.
            h_a_for_code = h_a.detach() if cfg.e_ntp_weight > 0 else h_a
            h_a_blocks = h_a_for_code.view(B, n_blocks, K, D)
            pre_q = self.code_pre[self._sel(i)](h_a_blocks[:, :, K - 1, :])
            c_i = self.quantize(pre_q)                                     # [B, n_blocks, dqs[i]]

            # codelm: causal forecast of c_i's own future, from c_i's own past
            pred_soft_full, raw_logits_full = self.codelm[self._sel(i)](c_i)
            pred_soft = pred_soft_full[:, :-1, :]
            raw_logits = raw_logits_full[:, :-1]
            true_next_code = c_i[:, 1:, :].detach()
            if cfg.quant_type in ("fsq", "ifsq"):
                # raw_logits here is already [B, T, dq, fsq_levels] (CodeLM.forward's own internal
                # reshape) — NOT the flat [B, T, dq*fsq_levels] code_level_loss's fsq/ifsq branch
                # expects (that shape comes from D_i's plain-Linear head instead), so this can't
                # reuse code_level_loss directly despite the near-identical math.
                half_l = (cfg.fsq_levels - 1) / 2
                true_level_idx = torch.round(true_next_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                cm_loss = F.cross_entropy(raw_logits.reshape(-1, cfg.fsq_levels), true_level_idx.reshape(-1))
            elif cfg.quant_type == "identity":
                cm_loss = F.mse_loss(raw_logits, true_next_code)
            else:
                true_bits = (true_next_code > 0).float()
                cm_loss = F.binary_cross_entropy_with_logits(raw_logits, true_bits, reduction="none").sum(-1).mean()
            code_match_losses.append(cm_loss)

            # DECODE — separate weights, own embedding of the SAME seq (never reads h_a: "no h0")
            if i == 0:
                d_in = self.dec_byte_bits_proj(byte_to_bits(seq)) if cfg.byte_repr == "bits" else self.dec_byte_emb(seq)
            else:
                d_in = self.dec_embed[self._sel1(i)](seq)
            d_in_blocks = d_in.view(B, n_blocks, K, D)
            forecast_embed = self.z_proj[self._sel(i)](pred_soft)          # [B, n_blocks-1, D]
            # block 0 has nothing to forecast from (no prior block exists) — rather than substitute a
            # placeholder there, just leave D_i's own plain embedding of the real first element in
            # place (nothing to inject, nothing to hide); blocks 1..n_blocks-1 substitute the real
            # causal forecast, unchanged.
            d_in_blocks = torch.cat(
                [d_in_blocks[:, :1, :, :], torch.cat([forecast_embed.unsqueeze(2), d_in_blocks[:, 1:, 1:, :]], dim=2)],
                dim=1,
            )
            d_in = d_in_blocks.view(B, L, D)
            h_b = self.run_blocks(self.d_blocks[self._sel(i)], self.d_ln_f[self._sel(i)], i, d_in)

            loss_i, acc_i = self._ntp_loss_and_acc(h_b, i, seq, is_e_side=False)
            level_losses.append(loss_i)
            level_accs.append(acc_i)
            if i == 0:
                byte_loss, byte_acc = loss_i, acc_i

            if compute_e_ntp:
                e_loss_i, e_acc_i = self._ntp_loss_and_acc(h_a, i, seq, is_e_side=True)
                e_ntp_losses.append(e_loss_i)
                e_ntp_accs.append(e_acc_i)

            seq = c_i   # level i+1's entire input sequence

        loss = sum(level_losses) + cfg.code_match_weight * torch.stack(code_match_losses).sum()
        if compute_e_ntp:
            loss = loss + cfg.e_ntp_weight * torch.stack(e_ntp_losses).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "code_match_loss": torch.stack(code_match_losses).sum(),
            **{f"level{i}_loss": l for i, l in enumerate(level_losses)},
            **{f"level{i}_acc": a for i, a in enumerate(level_accs)},
            **{f"code_match_loss_L{i}": v for i, v in enumerate(code_match_losses)},
        }
        if compute_e_ntp:
            metrics["e_ntp_loss"] = torch.stack(e_ntp_losses).sum()
            metrics.update({f"e_ntp_loss_L{i}": l for i, l in enumerate(e_ntp_losses)})
            metrics.update({f"e_ntp_acc_L{i}": a for i, a in enumerate(e_ntp_accs)})
        return loss, metrics


def init_head_bias_to_unigram(model: RecursiveSandwichLM, data: torch.Tensor) -> None:
    cfg = model.cfg
    counts = torch.bincount(data, minlength=256).float() + 1.0
    freq = counts / counts.sum()
    heads = [model.head0] + ([model.head_e0] if cfg.e_ntp_weight > 0 else [])
    with torch.no_grad():
        for head in heads:
            if cfg.byte_repr == "softmax":
                head.bias.copy_(torch.log(freq).to(head.bias.device))
            else:
                # per-bit marginal P(bit_k=1) over the byte distribution -> logit bias, same "start near
                # the right unconditional guess" idea as the softmax case, just bit-by-bit.
                byte_ids = torch.arange(256)
                bits = byte_to_bits(byte_ids)                                  # [256, 8], +/-1-ish
                p_bit1 = (freq.unsqueeze(-1) * (bits > 0).float()).sum(0).clamp(1e-4, 1 - 1e-4)   # [8]
                logit_bit = torch.log(p_bit1 / (1 - p_bit1))
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


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def _sample_next_byte(model: "RecursiveSandwichLM", last_h: torch.Tensor) -> torch.Tensor:
    """last_h: [B, tier_d_models[0]] (D0's hidden state at the newest
    position) -> sampled next byte ids [B]. Shared by both generate_no_cache
    and generate_kv_cache so they're guaranteed to sample identically given
    identical hidden states — the whole point of validating one against
    the other."""
    cfg = model.cfg
    if cfg.byte_repr == "bits":
        logits = model.head0(last_h)          # chain mode with true_bits=None already greedy-decodes internally
        return bits_to_byte(logits)
    return model.head0(last_h).argmax(-1)


@torch.no_grad()
def generate_no_cache(model: "RecursiveSandwichLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Reference (slow, obviously-correct) generation: recomputes E_0/D_0
    from scratch over the WHOLE sequence every new byte — no cache, no
    windowing complications, easy to trust. Only level 0 (E_0, codelm[0],
    D_0) ever participates in byte generation — levels 1+ are pure
    training-time auxiliary signal (see module docstring: D_i is
    conditioned only on its own level's code, never coarser), so this
    function never touches them."""
    cfg = model.cfg
    K = cfg.Ks[0]
    D = cfg.tier_d_models[0]
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.size(0)
    all_bytes = prompt_bytes

    for _ in range(n_new_bytes):
        L = all_bytes.size(1)
        e_in = model.byte_bits_proj(byte_to_bits(all_bytes)) if cfg.byte_repr == "bits" else model.byte_emb(all_bytes)
        h_a = model.run_blocks(model.e_blocks[0], model.e_ln_f[0], 0, e_in)

        d_in = model.dec_byte_bits_proj(byte_to_bits(all_bytes)) if cfg.byte_repr == "bits" else model.dec_byte_emb(all_bytes)
        d_in = d_in.clone()   # block 0's own plain embedding stays in place — nothing to inject there
        n_complete = L // K
        if n_complete >= 1:
            complete_len = n_complete * K
            h_a_blocks = h_a[:, :complete_len, :].view(B, n_complete, K, D)
            pre_q = model.code_pre[0](h_a_blocks[:, :, K - 1, :])
            c0 = model.quantize(pre_q)
            pred_soft_full, _ = model.codelm[0](c0)
            for j in range(1, n_complete + 1):
                pos = j * K
                if pos < L:
                    d_in[:, pos, :] = model.z_proj[0](pred_soft_full[:, j - 1, :])
        h_b = model.run_blocks(model.d_blocks[0], model.d_ln_f[0], 0, d_in)
        next_byte = _sample_next_byte(model, h_b[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RecursiveSandwichLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """KV-cache-efficient generation: E_0's and D_0's per-layer K/V grow by
    exactly one entry per new byte (never recomputed), and codelm[0]'s own
    K/V grow by one entry per completed block — none of E_0/D_0/codelm[0]
    ever re-processes a position it has already seen. This works precisely
    because this architecture is NOT destructive the way qcute_fifo's v1
    merge-based design is (see docs/fifo_v2.md section 8): every position's
    hidden state, once computed, is a fixed function of causal history and
    never gets mutated by anything that happens later.

    Limitation: dense-attention only (attn_window must be -1) — caching +
    windowed chunked attention aren't implemented together in this fork
    (Block.forward_step always attends densely over the full cache)."""
    cfg = model.cfg
    assert cfg.attn_window == -1, "generate_kv_cache only supports dense attention (attn_window=-1) — see docstring"
    K = cfg.Ks[0]
    D = cfg.tier_d_models[0]
    n_e_layers = len(model.e_blocks[0])
    n_d_layers = len(model.d_blocks[0])
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.size(0)

    e_cache_k = [None] * n_e_layers
    e_cache_v = [None] * n_e_layers
    d_cache_k = [None] * n_d_layers
    d_cache_v = [None] * n_d_layers
    lm_cache_k = [None] * len(model.codelm[0].blocks)
    lm_cache_v = [None] * len(model.codelm[0].blocks)

    h_a_hist: list[torch.Tensor] = []    # E0's hidden state at every processed position (for readout)
    c0_hist: list[torch.Tensor] = []     # true c0 values, one per completed block
    pending_forecast: torch.Tensor | None = None   # z_proj'd forecast waiting to be used at the NEXT block start

    def e0_step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = model.byte_bits_proj(byte_to_bits(byte_id)) if cfg.byte_repr == "bits" else model.byte_emb(byte_id)
        x = x.unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(model.e_blocks[0]):
            x, e_cache_k[li], e_cache_v[li] = block.forward_step(x, cos_new, sin_new, e_cache_k[li], e_cache_v[li])
        return model.e_ln_f[0](x).squeeze(1)   # [B, D]

    def d0_step(byte_id: torch.Tensor, pos: int, override_embed: torch.Tensor | None) -> torch.Tensor:
        if override_embed is not None:
            x = override_embed
        else:
            x = model.dec_byte_bits_proj(byte_to_bits(byte_id)) if cfg.byte_repr == "bits" else model.dec_byte_emb(byte_id)
        x = x.unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(model.d_blocks[0]):
            x, d_cache_k[li], d_cache_v[li] = block.forward_step(x, cos_new, sin_new, d_cache_k[li], d_cache_v[li])
        return model.d_ln_f[0](x).squeeze(1)   # [B, D]

    def codelm0_step(code: torch.Tensor, pos: int) -> torch.Tensor:
        x = model.codelm[0].in_proj(code).unsqueeze(1)
        lm_head_dim = cfg.lm_d_model // cfg.lm_n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, lm_head_dim, cfg.rope_base, device)
        for li, block in enumerate(model.codelm[0].blocks):
            x, lm_cache_k[li], lm_cache_v[li] = block.forward_step(x, cos_new, sin_new, lm_cache_k[li], lm_cache_v[li])
        h = model.codelm[0].ln_f(x).squeeze(1)
        raw = model.codelm[0].pred_head(h)   # BitPredictHead(h, true_bits=None) greedy-chains at inference when applicable
        if cfg.quant_type in ("fsq", "ifsq"):
            logits = raw.view(raw.size(0), model.codelm[0].dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * model.codelm[0].level_values).sum(-1)
        return 2 * torch.sigmoid(raw) - 1

    def process_position(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        """Runs E0+D0 (+codelm0 at block ends) for exactly one new position,
        given the caches built so far. Returns D0's hidden state there."""
        nonlocal pending_forecast
        h_a = e0_step(byte_id, pos)
        h_a_hist.append(h_a)
        override = pending_forecast   # None at pos==0 (nothing to forecast from yet) -> d0_step falls
                                        # back to its own plain embedding of byte_id, nothing injected
        pending_forecast = None
        h_b = d0_step(byte_id, pos, override)
        if (pos + 1) % K == 0:
            block_idx = pos // K
            pre_q = model.code_pre[0](h_a_hist[pos])
            c0 = model.quantize(pre_q)
            c0_hist.append(c0)
            pred_soft = codelm0_step(c0, block_idx)
            pending_forecast = model.z_proj[0](pred_soft)
        return h_b

    # prime the cache with the prompt, teacher-forced (positions 0..L0-1)
    L0 = prompt_bytes.size(1)
    last_h_b = None
    for pos in range(L0):
        last_h_b = process_position(prompt_bytes[:, pos], pos)

    # generate new bytes one at a time, each fed back in as the next position
    out_bytes = [prompt_bytes]
    for step in range(n_new_bytes):
        next_byte = _sample_next_byte(model, last_h_b)
        out_bytes.append(next_byte.unsqueeze(1))
        last_h_b = process_position(next_byte, L0 + step)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


@torch.no_grad()
def eval_model(model: RecursiveSandwichLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
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


def train(model: RecursiveSandwichLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt11", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx, step=step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        postfix = dict(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}",
            bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
        )
        if "code_match_loss" in metrics:   # absent in byte_only mode — no code hierarchy runs at all
            postfix["code_match_loss"] = f"{metrics['code_match_loss'].item():.4f}"
        pbar.set_postfix(**postfix)

        # lightweight train-bpb-only log, separate from the heavier eval_every val_ block below —
        # matches bytelm.py/bpelm.py's log_every convention. Logger.__call__ drops `msg` entirely
        # whenever `record` kwargs are given (see its docstring/source), so bpb must be passed
        # explicitly as a kwarg here or it never reaches run.jsonl — the eval_every block below only
        # ever passed val_* kwargs, which is why train bpb was previously text-only (run.log), absent
        # from run.jsonl.
        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), bpb=train_bpb, byte_acc=metrics["byte_acc"].item())

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Recursive Pass1/Pass2 sandwich, per-level (fork of qcute.qcutelm_vlt7/vlt8's narrow/wide split, applied recursively)", parents=[pre])
    p.add_argument("--Ks", type=_parse_int_tuple, default=(4, 4, 4))
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(96, 96, 96))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=64)
    p.add_argument("--lm_d_model", type=int, default=128)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=3)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--lm_attn_window", type=int, default=16)
    p.add_argument("--code_match_weight", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--byte_repr", type=str, default="bits", choices=["bits", "softmax"])
    p.add_argument("--bit_head_mode", type=str, default="chain", choices=["independent", "chain"])
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--e_ntp_weight", type=float, default=0.0)
    p.add_argument("--e_ntp_every", type=int, default=1)
    p.add_argument("--e_ntp_bit_head_mode", type=str, default=None, choices=[None, "independent", "chain"])
    p.add_argument("--byte_only", action="store_true")
    p.add_argument("--share_across_levels", action="store_true")

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
    args.dqs = _parse_int_tuple(args.dqs) if isinstance(args.dqs, str) else tuple(args.dqs)
    args.tier_d_models = _parse_int_tuple(args.tier_d_models) if isinstance(args.tier_d_models, str) else tuple(args.tier_d_models)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, dqs=args.dqs, tier_d_models=args.tier_d_models, context_len=args.context_len,
        quant_type=args.quant_type, fsq_levels=args.fsq_levels, n_heads=args.n_heads, n_layers=args.n_layers,
        mlp_mult=args.mlp_mult, attn_window=args.attn_window, lm_d_model=args.lm_d_model, lm_n_heads=args.lm_n_heads,
        lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult, lm_attn_window=args.lm_attn_window,
        code_match_weight=args.code_match_weight, rope_base=args.rope_base,
        byte_repr=args.byte_repr, bit_head_mode=args.bit_head_mode, bit_chain_n_heads=args.bit_chain_n_heads,
        bit_chain_gamma=args.bit_chain_gamma, bit_chain_fixed_kernel=args.bit_chain_fixed_kernel,
        e_ntp_weight=args.e_ntp_weight, e_ntp_every=args.e_ntp_every, e_ntp_bit_head_mode=args.e_ntp_bit_head_mode,
        byte_only=args.byte_only, share_across_levels=args.share_across_levels,
    )
    model = RecursiveSandwichLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt11_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} seq_lens={model.seq_lens} "
        f"context_len={cfg.context_len} quant_type={cfg.quant_type} params={n_params/1e6:.3f}M "
        f"(codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
