"""qcute.qcutelm_vlt11 — recursive per-level clockwork sandwich: byte ->
code -> code substituted as the NEXT (coarser) level's native input ->
... -> final next-token prediction. Rewritten from an earlier version of
this file (see "REPLACED DESIGN" below) to fix a fundamental
self-consistency problem by dropping the separate self-consistency loss
entirely, in favor of qcutelm_vlt10's substitution mechanism — generalized
here so each level can use its OWN code dimension (`dqs[level]`) and its
own tier hidden dimension (`tier_d_models[level]`), which qcutelm_vlt10
does not support (it keeps d_model/dq uniform across all tiers).

Architecture — literally "byte-code-code-next byte", recursively:
  tier_0 (dim tier_d_models[0]): bytes in, windowed causal attn
  for level in 0..n_levels-1:
    strided readout from tier_level's hidden state, every periods[level]
      bytes (last byte of each period-block) -> code_pre[level]
      (tier_d_models[level] -> dqs[level]) -> quantize -> code
    codelm[level] (operates in dqs[level]-dim space): causal forecast of
      the NEXT code, trained via code_match_loss[level] (same mechanism
      as every earlier fork's CodeLM)
    that forecast code IS what becomes tier_{level+1}'s native input at
      each block-start position (z_proj[level]: dqs[level] ->
      tier_d_models[level+1]) — literally "the code becomes the native
      code for the upper, coarser layer", not a separate re-derivation
    tier_{level+1} (dim tier_d_models[level+1]): windowed causal attn over
      that substituted-input sequence
  final logits = head(tier_N's output) — the ONLY byte-level loss

Every level is thus a genuine two-layer sandwich (tier_level "encodes" ->
code -> tier_{level+1} "decodes", conditioned on that code) — but unlike a
literal autoencoder, the decode side's target is the NEXT token, not a
reconstruction of the same input ("basically code autoencoder but out is
next token"). Because the SAME code produced by codelm[level]'s own
forecast is what gets substituted into tier_{level+1} (undetached,
gradient flows through it directly, exactly matching qcutelm_vlt8's
Pass-2/qcutelm_vlt10's substitution — not qcutelm_vlt6/vlt7's TRUE-code
teacher-forcing), there is no separate "predicted vs true code" pair to
reconcile with an auxiliary loss: the final NTP loss backpropagates
through every level's own code directly. This is what "fixes self
consistency" — there was never a second, independently-produced code that
needed reconciling with the first in this design, so no
self_consistency_weight knob exists here at all.

REPLACED DESIGN (previous version of this file, kept only as motivating
history): a genuine hierarchical POOL/UN-POOL autoencoder — encode side
recursively pooled K children into 1 parent code (PoolAttn, query-vector
cross-attention), decode side recursively un-pooled via BOS-style
DecodeBlock, with an explicit self_consistency_weight loss reconciling
decode's predicted child codes against encode's true child codes at every
paired level. That design's first documented bug (caught via a missing-
gradient check): substituting the TRUE child code as input to the next
finer level (rather than decode's own prediction) let the final loss's
gradient skip every level except the finest whenever
self_consistency_weight=0. The fix at the time (using decode's own
predicted soft code to descend) worked, but the underlying two-network-
reconciled-by-an-auxiliary-loss shape was the deeper issue this rewrite
removes altogether, in favor of qcutelm_vlt10's proven single-pass
substitution mechanism.

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
    Ks: tuple[int, ...] = (2, 2, 2)          # one entry per level; level i's readout period in raw bytes
                                               # is the cumulative product Ks[0]*...*Ks[i]
    dqs: tuple[int, ...] = (8, 8, 8)          # one entry per level — EACH level's own code dimension,
                                               # unlike qcutelm_vlt10's single global dq
    tier_d_models: tuple[int, ...] = (64, 64, 64, 64)   # one entry per tier (len(Ks)+1) — EACH tier's own
                                               # hidden dimension, unlike qcutelm_vlt10's single global d_model
    context_len: int = 1024
    quant_type: str = "ifsq"
    fsq_levels: int = 8
    vocab: int = 256
    n_heads: int = 4        # uniform across tiers; each tier_d_models[i] must be divisible by n_heads
    n_layers: int = 2
    mlp_mult: int = 4
    attn_window: int = -1
    lm_d_model: int = 128    # uniform across codelm levels (wide, each touches only its own short code sequence)
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    lm_attn_window: int = -1
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    trainable_bootstrap: bool = False


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
    short [B, n_blocks_level, dqs[level]] code sequence, never on raw
    bytes. Forecasts code[i+1] from codes[:i] (causal, optionally
    windowed). No loss of its own; code_match_loss (computed in the
    parent model, against a detached target) trains it — identical
    mechanism to every earlier fork's CodeLM."""

    def __init__(self, cfg: Config, dq: int):
        super().__init__()
        self.cfg = cfg
        self.dq = dq
        window = None if cfg.lm_attn_window == -1 else cfg.lm_attn_window
        self.in_proj = nn.Linear(dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=window) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        self.pred_head = nn.Linear(cfg.lm_d_model, dq * cfg.fsq_levels if factorized_softmax else dq)
        if factorized_softmax:
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            logits = raw.view(B, T, self.dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1), logits
        return 2 * torch.sigmoid(raw) - 1, raw


class RecursiveClockworkLM(nn.Module):
    """N-level recursive clockwork sandwich, per-level dq/tier-dim. See
    module docstring for the full "byte-code-code-next byte" rationale."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert len(cfg.dqs) == self.n_levels, "dqs must have one entry per level (len(Ks))"
        assert len(cfg.tier_d_models) == self.n_levels + 1, "tier_d_models must have len(Ks)+1 entries"
        n_tiers = self.n_levels + 1
        window = None if cfg.attn_window == -1 else cfg.attn_window
        if window is not None:
            assert cfg.context_len % window == 0, (
                f"attn_window ({window}) must divide context_len ({cfg.context_len})."
            )
        for d in cfg.tier_d_models:
            assert d % cfg.n_heads == 0, f"every tier_d_models entry ({d}) must be divisible by n_heads ({cfg.n_heads})"

        periods = []
        cum = 1
        for k in cfg.Ks:
            cum *= k
            assert cfg.context_len % cum == 0, (
                f"cumulative period {cum} (from Ks={cfg.Ks}) must divide context_len ({cfg.context_len})."
            )
            periods.append(cum)
        self.periods = periods

        if cfg.lm_attn_window != -1:
            for level, period in enumerate(periods):
                n_blocks_level = cfg.context_len // period
                assert n_blocks_level % cfg.lm_attn_window == 0, (
                    f"lm_attn_window ({cfg.lm_attn_window}) must divide level {level}'s n_blocks ({n_blocks_level})."
                )

        self.byte_emb = nn.Embedding(cfg.vocab, cfg.tier_d_models[0])
        self.tier_blocks = nn.ModuleList([
            nn.ModuleList([Block(cfg.tier_d_models[t], cfg.n_heads, cfg.mlp_mult, window=window) for _ in range(cfg.n_layers)])
            for t in range(n_tiers)
        ])
        self.tier_ln_f = nn.ModuleList([nn.LayerNorm(cfg.tier_d_models[t]) for t in range(n_tiers)])
        self.head = nn.Linear(cfg.tier_d_models[-1], cfg.vocab)   # applied only to the LAST tier's output

        # level i: reads FROM tier_i (dim tier_d_models[i]) -> dqs[i]; codelm[i] forecasts in dqs[i]-space;
        # z_proj[i] maps dqs[i] -> tier_{i+1}'s dim (tier_d_models[i+1]) for substitution there.
        self.code_pre = nn.ModuleList([nn.Linear(cfg.tier_d_models[i], cfg.dqs[i]) for i in range(self.n_levels)])
        self.z_proj = nn.ModuleList([nn.Linear(cfg.dqs[i], cfg.tier_d_models[i + 1]) for i in range(self.n_levels)])
        self.codelm = nn.ModuleList([CodeLM(cfg, cfg.dqs[i]) for i in range(self.n_levels)])
        if cfg.trainable_bootstrap:
            self.bootstrap_embed = nn.ParameterList([nn.Parameter(torch.zeros(cfg.tier_d_models[i + 1])) for i in range(self.n_levels)])

    def bootstrap_slot(self, level: int, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        D = self.cfg.tier_d_models[level + 1]
        if self.cfg.trainable_bootstrap:
            return self.bootstrap_embed[level].to(dtype).view(1, 1, -1).expand(B, 1, -1)
        return torch.zeros(B, 1, D, device=device, dtype=dtype)

    def run_tier(self, tier_idx: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        head_dim = cfg.tier_d_models[tier_idx] // cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, cfg.rope_base, x.device)
        for block in self.tier_blocks[tier_idx]:
            x = block(x, cos, sin)
        return self.tier_ln_f[tier_idx](x)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, v.size(-1))
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def _targets(self, ctx: torch.Tensor) -> torch.Tensor:
        B, L = ctx.shape
        target = torch.full((B, L), -100, dtype=torch.long, device=ctx.device)
        target[:, :-1] = ctx[:, 1:]
        return target

    def _loss_and_metrics(self, logits: torch.Tensor, target: torch.Tensor, finest_period: int) -> tuple[torch.Tensor, dict]:
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

        h = self.run_tier(0, self.byte_emb(ctx))   # tier_0

        code_match_losses = []
        for level in range(self.n_levels):
            period = self.periods[level]
            n_blocks = L // period
            D_prev = cfg.tier_d_models[level]
            D_next = cfg.tier_d_models[level + 1]
            h_blocks = h.view(B, n_blocks, period, D_prev)

            # every position's OWN local code (D_prev -> dq), used both for readout (last position of
            # each block -> z_hat, the level's true code) and as the pass-through dim-bridge into D_next
            # for non-boundary positions (every position, not just block-ends, gets projected through the
            # same code_pre/z_proj pathway — the only dim-bridging path this level has).
            local_pre = self.code_pre[level](h_blocks)                    # [B, n_blocks, period, dq]
            local_code = self.quantize(local_pre)
            local_embed = self.z_proj[level](local_code)                   # [B, n_blocks, period, D_next]

            z_hat = local_code[:, :, period - 1, :]                        # this level's TRUE code — same
                                                                              # code_pre call as local_pre, no need
                                                                              # to recompute on slot_hidden separately

            pred_soft_full, raw_logits_full = self.codelm[level](z_hat)
            pred_soft = pred_soft_full[:, :-1, :]
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

            forecast_embed = self.z_proj[level](pred_soft)                 # [B, n_blocks-1, D_next] — "the code
                                                                              # becomes the native code for the
                                                                              # upper, coarser layer"
            bootstrap = self.bootstrap_slot(level, B, ctx.device, h.dtype)  # [B, 1, D_next]
            subst_per_block = torch.cat([bootstrap, forecast_embed], dim=1)  # [B, n_blocks, D_next]

            tier_in_blocks = torch.cat([subst_per_block.unsqueeze(2), local_embed[:, :, 1:, :]], dim=2)
            tier_in = tier_in_blocks.view(B, L, D_next)
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


def init_head_bias_to_unigram(model: RecursiveClockworkLM, data: torch.Tensor) -> None:
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
def generate(model: RecursiveClockworkLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """No KV cache (recomputes the full tiered pipeline from scratch every
    generated byte, consistent with this lineage's existing simplicity
    tradeoff) — works for any prefix length."""
    cfg = model.cfg
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
            D_prev = cfg.tier_d_models[level]
            D_next = cfg.tier_d_models[level + 1]
            n_complete = L // period
            local_pre = model.code_pre[level](h.view(B, L, D_prev))
            local_code = model.quantize(local_pre)
            local_embed = model.z_proj[level](local_code)   # [B, L, D_next]

            tier_in = local_embed.clone()
            tier_in[:, 0, :] = model.bootstrap_slot(level, B, device, h.dtype).squeeze(1)
            if n_complete >= 1:
                complete_len = n_complete * period
                blocks_complete = h[:, :complete_len, :].view(B, n_complete, period, D_prev)
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


@torch.no_grad()
def plan_coarse_codes(model: RecursiveClockworkLM, prompt_bytes: torch.Tensor, n_new_blocks: int, device: str, plan_level: int | None = None) -> torch.Tensor:
    """The "free-rolling" half: encode the prompt through tiers 0..
    plan_level ONCE to seed plan_level's TRUE code history, then roll
    codelm[plan_level] forward autoregressively for n_new_blocks steps —
    PURE code-space, zero byte computation, no other tier touched during
    the rollout itself. plan_level defaults to the coarsest level
    (n_levels-1). Returns [B, n_prompt_blocks + n_new_blocks,
    dqs[plan_level]] — the extended code history, ready for
    `detokenize_from_plan`. See module docstring's honest limitation:
    this is genuinely free, but turning the result into bytes still
    needs the full tier stack (that part is NOT free)."""
    cfg = model.cfg
    plan_level = model.n_levels - 1 if plan_level is None else plan_level
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B, L = prompt_bytes.shape

    h = model.run_tier(0, model.byte_emb(prompt_bytes))
    z_hat_plan = None
    for level in range(plan_level + 1):
        period = model.periods[level]
        D_prev = cfg.tier_d_models[level]
        n_complete = L // period
        local_code = model.quantize(model.code_pre[level](h.view(B, L, D_prev)))
        complete_len = n_complete * period
        blocks_complete = local_code[:, :complete_len, :].view(B, n_complete, period, cfg.dqs[level])
        z_hat = blocks_complete[:, :, period - 1, :]                       # [B, n_complete, dqs[level]]

        if level == plan_level:
            z_hat_plan = z_hat
            break

        local_embed = model.z_proj[level](local_code)
        tier_in = local_embed.clone()
        tier_in[:, 0, :] = model.bootstrap_slot(level, B, device, h.dtype).squeeze(1)
        pred_soft_full, _ = model.codelm[level](z_hat)
        for j in range(1, n_complete + 1):
            pos = j * period
            if pos < L:
                tier_in[:, pos, :] = model.z_proj[level](pred_soft_full[:, j - 1, :])
        h = model.run_tier(level + 1, tier_in)

    code_hist = z_hat_plan                                                 # [B, n_prompt_blocks, dqs[plan_level]]
    for _ in range(n_new_blocks):
        pred_soft_full, _ = model.codelm[plan_level](code_hist)
        next_code = model.quantize(pred_soft_full[:, -1:, :])
        code_hist = torch.cat([code_hist, next_code], dim=1)

    if was_training:
        model.train()
    return code_hist


@torch.no_grad()
def detokenize_from_plan(model: RecursiveClockworkLM, prompt_bytes: torch.Tensor, planned_codes: torch.Tensor, plan_level: int, device: str) -> torch.Tensor:
    """The "late detokenize" half: generate the actual bytes for the
    n_new_blocks beyond the prompt (planned_codes.size(1) -
    n_prompt_blocks), using the SAME full byte-by-byte tier pipeline as
    `generate()` (still needs every tier — see module docstring) — except
    at plan_level's block-boundary positions that fall in the NEWLY
    planned range, the pre-decided code from `plan_coarse_codes` is
    substituted instead of a fresh per-step forecast, so the earlier
    free-rolled plan is respected rather than re-decided on the fly.
    Boundaries within the ORIGINAL prompt range still use a fresh
    codelm forecast (matching training/`generate()`'s own behavior
    exactly there — planned_codes' prompt-range entries are the TRUE
    codes, not forecasts, so substituting them there would be off-
    distribution)."""
    cfg = model.cfg
    plan_period = model.periods[plan_level]
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.size(0)
    n_prompt_blocks = prompt_bytes.size(1) // plan_period
    n_new_blocks = planned_codes.size(1) - n_prompt_blocks
    n_new_bytes = n_new_blocks * plan_period
    all_bytes = prompt_bytes

    for _ in range(n_new_bytes):
        L = all_bytes.size(1)
        h = model.run_tier(0, model.byte_emb(all_bytes))
        for level in range(model.n_levels):
            period = model.periods[level]
            D_prev = cfg.tier_d_models[level]
            n_complete = L // period
            local_code = model.quantize(model.code_pre[level](h.view(B, L, D_prev)))
            local_embed = model.z_proj[level](local_code)

            tier_in = local_embed.clone()
            tier_in[:, 0, :] = model.bootstrap_slot(level, B, device, h.dtype).squeeze(1)
            if n_complete >= 1:
                complete_len = n_complete * period
                blocks_complete = local_code[:, :complete_len, :].view(B, n_complete, period, cfg.dqs[level])
                z_hat = blocks_complete[:, :, period - 1, :]
                pred_soft_full, _ = model.codelm[level](z_hat)
                for j in range(1, n_complete + 1):
                    pos = j * period
                    if pos < L:
                        if level == plan_level and j >= n_prompt_blocks:
                            code_to_use = planned_codes[:, j, :]           # pre-decided, respect the plan
                        else:
                            code_to_use = pred_soft_full[:, j - 1, :]      # fresh forecast, matches training
                        tier_in[:, pos, :] = model.z_proj[level](code_to_use)
            h = model.run_tier(level + 1, tier_in)
        logits = model.head(h[:, -1, :])
        next_byte = logits.argmax(-1)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def qualitative_gen(model: RecursiveClockworkLM, data: torch.Tensor, prompt_len: int, n_new_bytes: int, device: str, log, step: int) -> None:
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
def eval_model(model: RecursiveClockworkLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
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


def train(model: RecursiveClockworkLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
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


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Recursive per-level clockwork sandwich: byte->code->code (native input to coarser level)->...->next byte, per-level dim/dq (fork of qcute.qcutelm_vlt10's mechanism)", parents=[pre])
    p.add_argument("--Ks", type=_parse_int_tuple, default=(2, 2, 2))
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(64, 64, 64, 64))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--lm_d_model", type=int, default=128)
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
        code_match_weight=args.code_match_weight, rope_base=args.rope_base, trainable_bootstrap=args.trainable_bootstrap,
    )
    model = RecursiveClockworkLM(cfg).to(device)
    n_tokenizer = sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith("codelm"))
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))
    n_params = n_tokenizer + n_codelm

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt11_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} periods(bytes)={model.periods} "
        f"context_len={cfg.context_len} quant_type={cfg.quant_type} params={n_params/1e6:.3f}M "
        f"(tokenizer={n_tokenizer/1e6:.3f}M codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
