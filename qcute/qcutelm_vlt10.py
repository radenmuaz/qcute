"""qcute.qcutelm_vlt10 — Clockwork-RNN-inspired multi-timescale tokenizer,
generalized to N levels: codelm is not a separate side-module
(qcutelm_vlt7/vlt8) nor resolved via a genuine sequential AR bootstrap loop
(qcutelm_vlt9) — it is a literal MIDDLE LAYER of a tiered stack, invoked
sparsely (every K bytes, "slow clock"), sandwiched between a tokenizer tier
below it (every byte, "fast clock") and one above it (also every byte,
re-synced from the slow clock at every block boundary). Forked from
qcutelm_vlt8. New fork (v10), does NOT replace qcutelm_vlt9 — that lineage
is a distinct, still-live experiment in genuine block-by-block symmetry;
this one instead asks whether architecturally embedding codelm as a real
layer (rather than reconciling it with the tokenizer only via auxiliary
losses) gives tighter code<->codelm coupling than qcutelm_vlt8's
aux_recon_weight/encode_match_weight without qcutelm_vlt9's O(n_blocks^2)
sequential prefill cost.

Motivation (session finding, observed live in qcutelm_vlt8_bsq_dense_supervision):
code_match_loss collapsed to ~0.0000 while aux_recon_acc stayed far below
no_code_acc/code_conditioned_acc — a mutual-collapse pattern consistent with
codelm's prediction and the encoder's code converging toward a low-diversity
trivial solution, reconciled only through loss terms with a long backprop
path. This fork's bet: make codelm's output structurally PART OF the
computation that produces the final logits (not a separately-computed thing
pulled into alignment after the fact) — the coupling is then load-bearing by
construction, not just loss-encouraged.

Architecture, N levels (Config.Ks = (K_1, K_2, ..., K_N), one entry per
level; len(Ks) codelms sandwiched between len(Ks)+1 tokenizer tiers,
tier_0 .. tier_N):

    bytes -> tier_0 (every byte, windowed attn)
    for level = 1..N:
        period_level = K_1 * K_2 * ... * K_level (cumulative, in RAW BYTES)
        strided readout from tier_{level-1}'s hidden state, every
          period_level bytes (last byte of each period_level-block) ->
          code_pre[level] + quantize -> codes z_hat_level
          [B, context_len/period_level, dq]
        codelm[level] (windowed attn over this level's own short code
          sequence) -> causal forecast, trained via code_match_loss_level
          (same mechanism as every earlier fork's CodeLM, no new loss
          invented, now one term per level, summed)
        tier_level input[t] = tier_{level-1}.h[t]  for t not a
                                 period_level-block start
                             = z_proj[level](forecast for that block)
                                 for t = j*period_level, j = 1..n_blocks-1
                             = z_proj[level](bootstrap[level])  for t = 0
          (every level's bootstrap is its own parameter/zero — no forecast
          exists yet for the very first block AT THAT LEVEL, same
          qcutelm_vlt9-style role, one per level)
        tier_level = run tier_level's own windowed-attn blocks over that
          input (SEPARATE weights per tier)
    final logits = head(tier_N's output) — the ONLY byte-level loss

N=1 (Config.Ks=(4,) by default) reduces exactly to the originally-built
2-level version (tier_0="lower", tier_1="upper", codelm[0]=the only
codelm) — this file no longer has a special-cased 2-level implementation,
the N=1 case of the general loop below IS that implementation.

No tier below tier_N carries its own byte-NTP loss/head: at every position
that is not a level-1 block-start, tier_1's input is tier_0's hidden state
verbatim, and so on up the stack — those positions structurally "skip" that
level's codelm and pass straight through. Since tier_N is what actually
produces every final prediction, earlier tiers having their own loss would
train a representation nothing downstream is forced to use consistently
(session-confirmed default, ported unchanged from the 2-level version: "at
timesteps not modulo k, skip to upper layer"). No free "no-code baseline"
comparison as a result — same tradeoff qcutelm_vlt9 already accepted.

Why this stays fast at any N, unlike qcutelm_vlt9's sequential loop:
nothing here requires waiting on any codelm block-by-block. Each tier's
forward pass depends only on the tier below it (already fully computed, one
vectorized pass); each codelm's forecast at block j depends only on codes
<j, already available from that one pass; each tier's substituted input is
assembled functionally (cat, no in-place mutation, autograd-safe) and run
in one more vectorized pass. Total: 2*N + 1 full-sequence passes (N tiers'
attention + N codelm passes, tier_0 counted once), no python loop over
n_blocks anywhere — O(L) work per level, not O(n_blocks^2).

Lead/lag, explicit tradeoff, same at every level: codelm[level]'s forecast
substituted at block j's start position was built from codes <j at that
level (a forecast FOR block j, available exactly when block j begins).
Every non-boundary position at every level gets no codelm assistance from
that level at all — a real capacity cost accepted deliberately, compounding
with N (coarser levels' codelms fire even less often).

No shared imports with qcutelm_vlt/vlt2/.../vlt9 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers/quantizers duplicated.

    uv run python -m qcute.qcutelm_vlt10 --config configs/qcutelm_vlt10_<name>.py
"""
import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, field
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
    Ks: tuple[int, ...] = (4,)   # one entry per level. level i's readout period in RAW BYTES is the
                                  # cumulative product Ks[0]*...*Ks[i-1] (1-indexed i). len(Ks) codelms,
                                  # len(Ks)+1 tokenizer tiers. Ks=(4,) reproduces the original 2-level
                                  # (1 codelm) design exactly.
    context_len: int = 1024
    dq: int = 6
    quant_type: str = "ifsq"   # "bsq" -> factorized sigmoids; "fsq"/"ifsq" -> factorized softmax
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 96          # uniform across ALL tokenizer tiers (tier_0 .. tier_N)
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    attn_window: int = -1      # uniform across ALL tokenizer tiers. -1 = dense. N>0 = must divide context_len.
    lm_d_model: int = 256      # uniform across ALL codelm levels (wide, each touches only its own short
                                # code sequence)
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    lm_attn_window: int = -1   # uniform across ALL codelm levels. -1 = dense. N>0 = must divide each
                                # level's own n_blocks (context_len // that level's cumulative period).
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    trainable_bootstrap: bool = False  # each level's block-0 marker (no codelm forecast exists yet at
                                        # that level) — False: zero vector. True: one learned [d_model]
                                        # parameter PER LEVEL. Same role as qcutelm_vlt9's bootstrap_slot.


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    """Full-causal SDPA + RoPE. window=None: plain O(T^2) dense. window=W:
    O(T*W) chunked (ported from qcutelm_vlt6/vlt8's verified
    _forward_chunked_no_sink) when T is a multiple of W and T > W; silently
    falls back to dense otherwise (needed for generate()'s growing,
    not-always-window-divisible sequence lengths)."""

    def __init__(self, d_model: int, n_heads: int, window: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
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


class CodeLM(nn.Module):
    """Separate weights per level, operates strictly on that level's own
    short [B, n_blocks_level, dq] code sequence, never on raw bytes.
    Forecasts code[i+1] from codes[:i] (causal, optionally windowed — see
    Config.lm_attn_window). No loss of its own; code_match_loss (computed
    in the parent model, against a detached target) trains it — identical
    mechanism to every earlier fork's CodeLM."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        window = None if cfg.lm_attn_window == -1 else cfg.lm_attn_window
        self.in_proj = nn.Linear(cfg.dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=window) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        self.pred_head = nn.Linear(cfg.lm_d_model, cfg.dq * cfg.fsq_levels if factorized_softmax else cfg.dq)
        if factorized_softmax:
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z_hat: [B, n, dq] (true codes seen so far, this level) ->
        (pred_soft, raw_logits), each predicting position i+1 from
        positions [:i]."""
        cfg = self.cfg
        x = self.in_proj(z_hat)
        head_dim = cfg.lm_d_model // cfg.lm_n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)
        raw = self.pred_head(h)
        if cfg.quant_type in ("fsq", "ifsq"):
            B, T, _ = raw.shape
            logits = raw.view(B, T, cfg.dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1), logits
        return 2 * torch.sigmoid(raw) - 1, raw


class ClockworkLM(nn.Module):
    """N-level Clockwork sandwich: tier_0 (bytes) -> codelm[0] (period
    Ks[0]) -> tier_1 -> codelm[1] (period Ks[0]*Ks[1]) -> tier_2 -> ... ->
    tier_N (owns the model's only loss). See module docstring for the full
    rationale; N=1 (default Ks=(4,)) is exactly the originally-built
    2-level design."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        n_tiers = self.n_levels + 1
        window = None if cfg.attn_window == -1 else cfg.attn_window
        if window is not None:
            assert cfg.context_len % window == 0, (
                f"attn_window ({window}) must divide context_len ({cfg.context_len})."
            )

        periods = []
        cum = 1
        for k in cfg.Ks:
            cum *= k
            assert cfg.context_len % cum == 0, (
                f"cumulative period {cum} (from Ks={cfg.Ks}) must divide context_len ({cfg.context_len})."
            )
            periods.append(cum)
        self.periods = periods   # periods[level] = level's own readout period in raw bytes (0-indexed level)

        if cfg.lm_attn_window != -1:
            for level, period in enumerate(periods):
                n_blocks_level = cfg.context_len // period
                assert n_blocks_level % cfg.lm_attn_window == 0, (
                    f"lm_attn_window ({cfg.lm_attn_window}) must divide level {level}'s n_blocks ({n_blocks_level})."
                )

        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.tier_blocks = nn.ModuleList([
            nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window=window) for _ in range(cfg.n_layers)])
            for _ in range(n_tiers)
        ])
        self.tier_ln_f = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(n_tiers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab)   # applied only to the LAST tier's output

        self.code_pre = nn.ModuleList([nn.Linear(cfg.d_model, cfg.dq) for _ in range(self.n_levels)])
        self.z_proj = nn.ModuleList([nn.Linear(cfg.dq, cfg.d_model) for _ in range(self.n_levels)])
        self.codelm = nn.ModuleList([CodeLM(cfg) for _ in range(self.n_levels)])
        if cfg.trainable_bootstrap:
            self.bootstrap_embed = nn.ParameterList([nn.Parameter(torch.zeros(cfg.d_model)) for _ in range(self.n_levels)])

    def bootstrap_slot(self, level: int, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Level `level`'s block-0 marker — no codelm forecast exists yet
        at that level. Zero (default) or a learned per-level parameter
        (trainable_bootstrap). [B, 1, D]."""
        if self.cfg.trainable_bootstrap:
            return self.bootstrap_embed[level].to(dtype).view(1, 1, -1).expand(B, 1, -1)
        return torch.zeros(B, 1, self.cfg.d_model, device=device, dtype=dtype)

    def run_tier(self, tier_idx: int, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in self.tier_blocks[tier_idx]:
            x = block(x, cos, sin)
        return self.tier_ln_f[tier_idx](x)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, self.cfg.dq)
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def build_tier_input(self, h_prev: torch.Tensor, bootstrap: torch.Tensor, forecast_embed: torch.Tensor, period: int) -> torch.Tensor:
        """h_prev: [B, L, D] (tier below's hidden state), bootstrap:
        [B, 1, D], forecast_embed: [B, n_blocks-1, D] (already z_proj'd) ->
        [B, L, D] with EVERY period-block's FIRST byte position replaced —
        block 0 gets the bootstrap marker, every later block gets this
        level's codelm forecast; h_prev's own value at those positions is
        never used (a genuine "codelm tier" input, not a fallback). Fully
        functional (cat, no in-place mutation) so it stays autograd-safe."""
        B, L, D = h_prev.shape
        n_blocks = L // period
        h_blocks = h_prev.view(B, n_blocks, period, D)
        subst_per_block = torch.cat([bootstrap, forecast_embed], dim=1)      # [B, n_blocks, D]
        tier_in_blocks = torch.cat([subst_per_block.unsqueeze(2), h_blocks[:, :, 1:, :]], dim=2)
        return tier_in_blocks.view(B, L, D)

    def _targets(self, ctx: torch.Tensor) -> torch.Tensor:
        """Plain shift-by-one next-byte target over the whole sequence —
        no reserved slots, since every level's forecast is substituted
        directly onto an existing byte position rather than an inserted
        extra token. Last position has no target."""
        B, L = ctx.shape
        target = torch.full((B, L), -100, dtype=torch.long, device=ctx.device)
        target[:, :-1] = ctx[:, 1:]
        return target

    def _loss_and_metrics(self, logits: torch.Tensor, target: torch.Tensor, finest_period: int) -> tuple[torch.Tensor, dict]:
        """finest_period = periods[0] (level-1's, the most frequent
        substitution) — used to report code_conditioned_acc/
        within_block_acc at the same granularity the original 2-level
        version did, regardless of how many coarser levels sit above it."""
        cfg = self.cfg
        B, L = logits.shape[0], logits.shape[1]
        n_blocks = L // finest_period
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), target.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            valid = target != -100
            pred = logits.argmax(-1)
            acc = (pred == target)[valid].float().mean() if valid.any() else torch.zeros(())
            block_start_mask = torch.zeros_like(target, dtype=torch.bool).view(B, n_blocks, finest_period)
            block_start_mask[:, :, 0] = True
            block_start_mask = block_start_mask.view(B, L)
            cc_valid = block_start_mask & valid
            code_conditioned_acc = (pred == target)[cc_valid].float().mean() if cc_valid.any() else torch.zeros(())
            within_valid = valid & ~block_start_mask
            within_block_acc = (pred == target)[within_valid].float().mean() if within_valid.any() else torch.zeros(())
        return loss, {"loss": loss, "acc": acc, "code_conditioned_acc": code_conditioned_acc, "within_block_acc": within_block_acc}

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        D = cfg.d_model

        h = self.run_tier(0, self.byte_emb(ctx))   # tier_0

        code_match_losses = []
        for level in range(self.n_levels):
            period = self.periods[level]
            n_blocks = L // period
            h_blocks = h.view(B, n_blocks, period, D)
            slot_hidden = h_blocks[:, :, period - 1, :]
            pre_q = self.code_pre[level](slot_hidden)
            z_hat = self.quantize(pre_q)                       # [B, n_blocks, dq]

            pred_soft_full, raw_logits_full = self.codelm[level](z_hat)
            pred_soft = pred_soft_full[:, :-1, :]                # forecast for blocks 1..n_blocks-1
            raw_logits = raw_logits_full[:, :-1]
            true_next_code = z_hat[:, 1:, :].detach()
            if cfg.quant_type in ("fsq", "ifsq"):
                half_l = (cfg.fsq_levels - 1) / 2
                true_level_idx = torch.round(true_next_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
                cm_loss = F.cross_entropy(raw_logits.reshape(-1, cfg.fsq_levels), true_level_idx.reshape(-1))
            else:
                true_bits = (true_next_code > 0).float()
                cm_loss = F.binary_cross_entropy_with_logits(raw_logits, true_bits)
            code_match_losses.append(cm_loss)

            forecast_embed = self.z_proj[level](pred_soft)      # [B, n_blocks-1, D]
            bootstrap = self.bootstrap_slot(level, B, ctx.device, h.dtype)
            tier_in = self.build_tier_input(h, bootstrap, forecast_embed, period)
            h = self.run_tier(level + 1, tier_in)

        logits = self.head(h)
        target = self._targets(ctx)
        ntp_loss, metrics = self._loss_and_metrics(logits, target, self.periods[0])

        code_match_loss = torch.stack(code_match_losses).sum()
        loss = ntp_loss + cfg.code_match_weight * code_match_loss
        metrics = {
            "loss": loss, "code_match_loss": code_match_loss,
            **{f"code_match_loss_L{i + 1}": v for i, v in enumerate(code_match_losses)},
            **metrics,
        }
        return loss, metrics


def init_head_bias_to_unigram(model: ClockworkLM, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        model.head.bias.copy_(log_freq.to(model.head.bias.device))


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
def generate(model: ClockworkLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """prompt_bytes: [L0] -> [L0 + n_new_bytes]. No KV cache (recomputes the
    full tiered pipeline from scratch every generated byte, consistent with
    this lineage's existing simplicity tradeoff) — works for ANY prefix
    length: each level's codes/forecasts are only computed for that level's
    currently-complete blocks, and the block-start substitution is applied
    at every position that exists within the current prefix, at every
    level in turn."""
    cfg = model.cfg
    D = cfg.d_model
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.size(0)
    all_bytes = prompt_bytes

    for _ in range(n_new_bytes):
        L = all_bytes.size(1)
        h = model.run_tier(0, model.byte_emb(all_bytes))
        for level in range(model.n_levels):
            period = model.periods[level]
            n_complete = L // period
            tier_in = h.clone()   # no_grad context — in-place is fine here
            tier_in[:, 0, :] = model.bootstrap_slot(level, B, device, h.dtype).squeeze(1)
            if n_complete >= 1:
                complete_len = n_complete * period
                blocks_complete = h[:, :complete_len, :].view(B, n_complete, period, D)
                slot_hidden = blocks_complete[:, :, period - 1, :]
                z_hat = model.quantize(model.code_pre[level](slot_hidden))
                pred_soft_full, _ = model.codelm[level](z_hat)
                for j in range(1, n_complete + 1):
                    pos = j * period
                    if pos < L:
                        tier_in[:, pos, :] = model.z_proj[level](pred_soft_full[:, j - 1, :])
            h = model.run_tier(level + 1, tier_in)
        logits = model.head(h[:, -1, :])
        next_byte = logits.argmax(-1)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def qualitative_gen(model: ClockworkLM, data: torch.Tensor, prompt_len: int, n_new_bytes: int, device: str, log, step: int) -> None:
    period_max = model.periods[-1]
    prompt_len = max(period_max, (prompt_len // period_max) * period_max)
    n_new_bytes = max(period_max, (n_new_bytes // period_max) * period_max)
    if len(data) < prompt_len + n_new_bytes:
        return
    prompt = data[:prompt_len]
    ground_truth = data[prompt_len:prompt_len + n_new_bytes]
    gen_full = generate(model, prompt, n_new_bytes, device)
    generated = gen_full[prompt_len:]
    match = (generated.cpu() == ground_truth).float().mean().item()
    log(f"[gen@step{step}] prompt={_bytes_repr(prompt)}", step=step)
    log(f"[gen@step{step}] generated={_bytes_repr(generated)}", step=step)
    log(f"[gen@step{step}] ground_truth={_bytes_repr(ground_truth)}  byte_match={match*100:.2f}%",
        step=step, gen_byte_match=match)


@torch.no_grad()
def eval_model(model: ClockworkLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        accum.setdefault("total_loss", []).append(loss.item())
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["loss"] / math.log(2)
    return result


def train(model: ClockworkLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt10", dynamic_ncols=True)
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

        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}",
            bpb=f"{metrics['loss'].item()/math.log(2):.4f}",
            code_acc=f"{metrics['code_conditioned_acc'].item()*100:.2f}%",
            within_acc=f"{metrics['within_block_acc'].item()*100:.2f}%",
            code_match_loss=f"{metrics['code_match_loss'].item():.4f}",
        )

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])

        if args.gen_every > 0 and (step % args.gen_every == 0 or step == args.steps):
            qualitative_gen(model, val_data, args.gen_prompt_len, args.gen_new_bytes, device, log, step)


def _parse_ks(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Clockwork-RNN-inspired multi-timescale tokenizer, N levels: codelm as a sparse middle layer, chained (fork of qcute.qcutelm_vlt8)", parents=[pre])
    p.add_argument("--Ks", type=_parse_ks, default=(4,), help="comma-separated periods, one per level, e.g. 4,4")
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--dq", type=int, default=6)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--lm_d_model", type=int, default=256)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=3)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--lm_attn_window", type=int, default=-1)
    p.add_argument("--code_match_weight", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--trainable_bootstrap", type=lambda x: x.lower() != "false", default=False)

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
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--gen_every", type=int, default=1000, help="0 = off")
    p.add_argument("--gen_prompt_len", type=int, default=64)
    p.add_argument("--gen_new_bytes", type=int, default=64)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    if isinstance(args.Ks, str):
        args.Ks = _parse_ks(args.Ks)
    else:
        args.Ks = tuple(args.Ks)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        Ks=args.Ks, context_len=args.context_len, dq=args.dq, quant_type=args.quant_type,
        fsq_levels=args.fsq_levels, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, attn_window=args.attn_window, lm_d_model=args.lm_d_model,
        lm_n_heads=args.lm_n_heads, lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult,
        lm_attn_window=args.lm_attn_window, code_match_weight=args.code_match_weight, rope_base=args.rope_base,
        trainable_bootstrap=args.trainable_bootstrap,
    )
    model = ClockworkLM(cfg).to(device)
    n_tokenizer = sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith("codelm"))
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))
    n_params = n_tokenizer + n_codelm

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt10_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} periods(bytes)={model.periods} context_len={cfg.context_len} dq={cfg.dq} "
        f"quant_type={cfg.quant_type} params={n_params/1e6:.3f}M (tokenizer={n_tokenizer/1e6:.3f}M codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
