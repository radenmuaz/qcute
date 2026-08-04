"""qcute.qcutelm_vlt9 — TRUE architectural symmetry: encode and decode are
now the SAME function (a single conditional LM, one set of weights), both
consuming codelm's code every K steps, resolved via a genuine block-by-
block bootstrap loop. Forked from qcutelm_vlt8.

Session finding this fork answers ("i found a flaw in v8 for symmetry"):
qcutelm_vlt7/vlt8 were never actually symmetric between encode and decode
despite the "symmetric" framing. Decode is a genuine CONSUMER of codelm's
output — every block's byte generation is conditioned on a predicted
code. Encode is a pure PRODUCER — it computes z_hat from raw bytes alone
and never consumes codelm's output in return. That's a one-way pipeline
(encode -> codelm -> decode) reusing the same weights for two of its
three stages, not a symmetric pair — which is why qcutelm_vlt8 changed
its default to shared_tokenizer_phases=False (untied weights): if
encode and decode are different functions (unconditional vs. conditional
LM), sharing weights between them serves two incompatible jobs.

TRUE symmetry means encode ALSO consumes codelm's forecast as
conditioning, exactly like decode. For block 0 (nothing precedes it) the
conditioning code is a bootstrap marker (zero, or a learned parameter —
trainable_bootstrap). For every block i>0, encode's own computation needs
codelm's forecast (built from codes 0..i-1), which needs codelm to have
already seen block i-1's TRUE code, which needs encode to have already
finished processing block i-1 — a genuine block-by-block AR handshake
between encoder and codelm. There is no way to vectorize this away: block
i's input literally cannot be constructed until block i-1's true code is
known, so — unlike qcutelm_vlt7/vlt8, where encode is one fully
vectorized pass over the whole sequence — encode (and therefore the
combined encode+decode single pass, since they're now the same function)
must run as a genuine n_blocks-iteration Python loop, recomputing the
growing sequence's hidden states at every step (no KV cache, consistent
with this file's existing simplicity tradeoff elsewhere in this lineage).

PROBLEM, flagged going in, not discovered after the fact: prefill is
slow. qcutelm_vlt7/vlt8 process an entire known context (prefill, or a
full training batch) in ~2-3 vectorized forward passes. qcutelm_vlt9 must
process it in n_blocks sequential steps (256 for context_len=1024, K=4),
each recomputing attention over the growing sequence — real, expected,
unavoidable overhead for the property being tested (does genuine
end-to-end symmetry, where NOTHING about the code space depends on which
"phase" produced it, change quality enough to justify the cost).

Structural change: codes are now a PREFIX to each block (bos-style, code
then K bytes) rather than a suffix (K bytes then a slot, qcutelm_vlt7/
vlt8's build_interleaved) — the natural structure once every block is
uniformly "conditioned then generated," no more distinction between a
"no-code" reserved slot and a "forecast" slot. One consequence: the free
same-weights no-code-vs-code baseline ablation qcutelm_vlt7/vlt8 got for
free (no_code_acc vs code_conditioned_acc from two passes at zero extra
cost) doesn't exist here — adding a real "no code at all" comparison
pass would reintroduce a second full pass, undoing the point of this
fork. Traded away deliberately, not an oversight.

codelm's own training signal (code_match_loss, aux_recon_weight,
encode_match_weight) all still apply, computed via ONE final vectorized
codelm(z_hat_full) call after the loop — codelm's own attention is
causal, so its prediction at position i depends only on codes <i
regardless of whether the full sequence or just a prefix was fed in, so
this final call's per-position predictions are identical to what the
incremental in-loop calls would have given, without needing to cache
per-step codelm state.

No shared imports with qcutelm_vlt/vlt2/.../vlt8 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers/quantizers duplicated.

    uv run python -m qcute.qcutelm_vlt9 --config configs/qcutelm_vlt9_<name>.py
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
    K: int = 4
    context_len: int = 1024
    dq: int = 6
    quant_type: str = "ifsq"   # "bsq" -> factorized sigmoids; "fsq"/"ifsq" -> factorized softmax
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 96          # the ONE stack — encode and decode are now the same function
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    lm_d_model: int = 256      # codelm tier (wide, touches ONLY the short n_blocks-length code sequence)
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    trainable_bootstrap: bool = False  # False: literal zero vector marks block 0's code prefix (nothing
                                        # precedes it, no codelm forecast exists yet). True: a single
                                        # learned [d_model] parameter instead.
    aux_recon_weight: float = 0.0      # short, direct gradient path for code_pre: decode(z_hat_enc) vs the
                                        # SAME block's own bytes, block-local (K-length, batched, zero
                                        # cross-block attention). Mirrors qcutelm_vlt6/vlt8's aux_recon_weight.
    encode_match_weight: float = 0.0   # mutual-consistency: code_pre's raw (pre-quantization) output pulled
                                        # toward codelm's own prediction (detached) — reverse direction of
                                        # code_match_loss. See qcutelm_vlt8's Config docstring for the full
                                        # tokenizer/detokenizer-free-decoding motivation, unchanged here.


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
    """Plain full-causal SDPA + RoPE. No windowing in this fork — the
    growing, recomputed-every-step sequence this file's forward()/generate()
    use doesn't have a stable period to window against the way qcutelm_vlt8's
    fixed interleaved format did; prefill cost is already dominated by the
    O(n_blocks) sequential loop itself, not by attention's own O(T^2) term
    within each step."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
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
    """The ONLY wide component — separate weights, operates strictly on
    the short [B, n, dq] code sequence, never on raw bytes. Forecasts
    code[i+1] from codes[:i] (causal). No loss of its own; code_match_loss
    (computed in the parent model, against a detached target) trains it —
    identical mechanism to qcutelm_vlt6/vlt7/vlt8's CodeLM, duplicated per
    this repo's self-contained-module convention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        self.pred_head = nn.Linear(cfg.lm_d_model, cfg.dq * cfg.fsq_levels if factorized_softmax else cfg.dq)
        if factorized_softmax:
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z_hat: [B, n, dq] (true codes seen so far) -> (pred_soft,
        raw_logits), each predicting position i+1 from positions [:i]."""
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


class SymmetricLM(nn.Module):
    """ONE stack, ONE function: every block is [code_prefix, K bytes],
    where code_prefix is a bootstrap marker (block 0) or codelm's forecast
    (every later block) — genuinely the same conditional-LM role for both
    what qcutelm_vlt7/vlt8 called "encode" and "decode". See module
    docstring for the full rationale and the prefill-cost tradeoff."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)
        self.code_pre = nn.Linear(cfg.d_model, cfg.dq)
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.codelm = CodeLM(cfg)
        if cfg.trainable_bootstrap:
            self.bootstrap_embed = nn.Parameter(torch.zeros(cfg.d_model))

    def bootstrap_slot(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Block 0's code prefix — nothing precedes it, no codelm forecast
        exists yet. Zero (default) or a learned parameter (trainable_bootstrap)."""
        if self.cfg.trainable_bootstrap:
            return self.bootstrap_embed.to(dtype).view(1, 1, -1).expand(B, 1, -1)
        return torch.zeros(B, 1, self.cfg.d_model, device=device, dtype=dtype)

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, self.cfg.dq)
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def decode_block_local(self, code: torch.Tensor, target_block: torch.Tensor) -> torch.Tensor:
        """code: [N, dq], target_block: [N, K] -> logits: [N, K, vocab].
        Block-local, batched, zero cross-block attention — aux_recon_weight's
        short direct path for code_pre. Ported from qcutelm_vlt8, single
        set of weights here (no dec_* split — there's only one stack)."""
        K = target_block.size(1)
        bos = self.z_proj(code).unsqueeze(1)
        if K > 1:
            dec_in = torch.cat([bos, self.byte_emb(target_block[:, :-1])], dim=1)
        else:
            dec_in = bos
        dec_h = self.run_blocks(dec_in)
        return self.head(dec_h)

    def _targets(self, ctx: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """[B, n_blocks*(K+1)] — code IS a prefix now (not a suffix), so
        local position p (0..K-1) predicts ctx_blocks[:,:,p] directly
        (position 0 = the code, predicting this block's own byte 0;
        positions 1..K-1 = within-block next-byte). Local position K (the
        block's LAST byte) has no target — its "next" input is the
        FOLLOWING block's code, not a byte."""
        cfg = self.cfg
        K = cfg.K
        B = ctx.size(0)
        ctx_blocks = ctx.view(B, n_blocks, K)
        target = torch.full((B, n_blocks, K + 1), -100, dtype=torch.long, device=ctx.device)
        target[:, :, 0:K] = ctx_blocks[:, :, 0:K]
        return target.view(B, n_blocks * (K + 1))

    def _loss_and_metrics(self, logits: torch.Tensor, target_flat: torch.Tensor, n_blocks: int, prefix: str) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        K = cfg.K
        B = logits.size(0)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), target_flat.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            valid = target_flat != -100
            pred = logits.argmax(-1)
            acc = (pred == target_flat)[valid].float().mean() if valid.any() else torch.zeros(())
            slot_mask = torch.zeros_like(target_flat, dtype=torch.bool).view(B, n_blocks, K + 1)
            slot_mask[:, :, 0] = True   # code-prefix position, every block (no boundary exception this time)
            slot_mask = slot_mask.view(B, n_blocks * (K + 1))
            slot_valid = slot_mask & valid
            slot_acc = (pred == target_flat)[slot_valid].float().mean() if slot_valid.any() else torch.zeros(())
            within_valid = valid & ~slot_mask
            within_acc = (pred == target_flat)[within_valid].float().mean() if within_valid.any() else torch.zeros(())
        return loss, {f"{prefix}_loss": loss, f"{prefix}_acc": acc, f"{prefix}_slot_acc": slot_acc, f"{prefix}_within_acc": within_acc}

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        K = cfg.K
        n_blocks = L // K
        D = cfg.d_model

        byte_embed = self.byte_emb(ctx)
        byte_blocks = byte_embed.view(B, n_blocks, K, D)

        code_hist: list[torch.Tensor] = []     # true z_hat per block
        pre_q_hist: list[torch.Tensor] = []    # pre-quantization value per block (for encode_match_weight)
        running_seq = None
        h = None
        for i in range(n_blocks):
            if i == 0:
                code_prefix = self.bootstrap_slot(B, ctx.device, byte_embed.dtype)
            else:
                code_hist_tensor = torch.stack(code_hist, dim=1)   # [B, i, dq] — codes 0..i-1
                pred_soft_i, _ = self.codelm(code_hist_tensor)
                code_prefix = self.z_proj(pred_soft_i[:, -1, :]).unsqueeze(1)   # forecast for block i

            block_in = torch.cat([code_prefix, byte_blocks[:, i, :, :]], dim=1)   # [B, K+1, D]
            running_seq = block_in if running_seq is None else torch.cat([running_seq, block_in], dim=1)

            h = self.run_blocks(running_seq)   # recomputed in full every iteration — the prefill cost
            slot_hidden_i = h[:, -1, :]          # hidden state at block i's LAST byte
            pre_q_i = self.code_pre(slot_hidden_i)
            z_hat_i = self.quantize(pre_q_i)
            code_hist.append(z_hat_i)
            pre_q_hist.append(pre_q_i)

        z_hat_full = torch.stack(code_hist, dim=1)     # [B, n_blocks, dq]
        pre_q_full = torch.stack(pre_q_hist, dim=1)     # [B, n_blocks, dq]

        # Main NTP loss over the WHOLE final sequence — h (from the last loop iteration) already
        # covers every position causally-correctly, no need to recompute.
        logits = self.head(h)
        target_flat = self._targets(ctx, n_blocks)
        ntp_loss, ntp_metrics = self._loss_and_metrics(logits, target_flat, n_blocks, "ntp")

        # CodeLM supervision: one final vectorized causal pass over the complete true-code sequence.
        # codelm's own attention is causal, so its prediction at position i depends only on codes <i
        # regardless of whether fed the full sequence or just a prefix — identical to the in-loop
        # per-step predictions, no need to have cached them.
        pred_soft_full, raw_logits_full = self.codelm(z_hat_full)
        pred_soft = pred_soft_full[:, :-1, :]
        raw_logits = raw_logits_full[:, :-1]
        true_next_code = z_hat_full[:, 1:, :].detach()
        if cfg.quant_type in ("fsq", "ifsq"):
            half_l = (cfg.fsq_levels - 1) / 2
            true_level = torch.round(true_next_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
            code_match_loss = F.cross_entropy(raw_logits.reshape(-1, cfg.fsq_levels), true_level.reshape(-1))
        else:
            true_bits = (true_next_code > 0).float()
            code_match_loss = F.binary_cross_entropy_with_logits(raw_logits, true_bits)

        aux_recon_loss = torch.zeros((), device=ctx.device)
        aux_recon_acc = torch.zeros((), device=ctx.device)
        if cfg.aux_recon_weight > 0:
            all_blocks = ctx.view(B, n_blocks, K)
            z_flat = z_hat_full.reshape(B * n_blocks, cfg.dq)
            blocks_flat = all_blocks.reshape(B * n_blocks, K)
            aux_logits = self.decode_block_local(z_flat, blocks_flat)
            aux_recon_loss = F.cross_entropy(aux_logits.reshape(-1, cfg.vocab), blocks_flat.reshape(-1))
            aux_recon_acc = (aux_logits.argmax(-1) == blocks_flat).float().mean()

        encode_match_loss = torch.zeros((), device=ctx.device)
        if cfg.encode_match_weight > 0:
            encode_match_loss = F.mse_loss(pre_q_full[:, 1:, :], pred_soft.detach())

        loss = (ntp_loss + cfg.code_match_weight * code_match_loss
                + cfg.aux_recon_weight * aux_recon_loss + cfg.encode_match_weight * encode_match_loss)
        metrics = {
            "loss": loss,
            "code_match_loss": code_match_loss,
            "code_conditioned_acc": ntp_metrics["ntp_slot_acc"],   # predicts THIS block's own 1st byte from its code prefix
            "within_block_acc": ntp_metrics["ntp_within_acc"],
            **ntp_metrics,
        }
        if cfg.aux_recon_weight > 0:
            metrics["aux_recon_loss"] = aux_recon_loss
            metrics["aux_recon_acc"] = aux_recon_acc
        if cfg.encode_match_weight > 0:
            metrics["encode_match_loss"] = encode_match_loss
        return loss, metrics


def init_head_bias_to_unigram(model: SymmetricLM, data: torch.Tensor) -> None:
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
def generate(model: SymmetricLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """prompt_bytes: [L0] (L0 a positive multiple of K) -> [L0 + n_new_bytes]
    generated continuation. Same block-by-block loop as forward() — the
    prompt is processed teacher-forced through it (real bytes, still one
    block at a time since codelm's forecast for block i genuinely needs
    block i-1's TRUE code first), then new blocks are generated the same
    way with sampled/argmax bytes instead of ground truth."""
    cfg = model.cfg
    K = cfg.K
    D = cfg.d_model
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    assert prompt_bytes.size(1) % K == 0 and prompt_bytes.size(1) >= K, "prompt length must be a positive multiple of K"
    B = prompt_bytes.size(0)
    n_prompt_blocks = prompt_bytes.size(1) // K

    byte_embed = model.byte_emb(prompt_bytes).view(B, n_prompt_blocks, K, D)
    code_hist: list[torch.Tensor] = []
    running_seq = None
    for i in range(n_prompt_blocks):
        if i == 0:
            code_prefix = model.bootstrap_slot(B, device, byte_embed.dtype)
        else:
            code_hist_t = torch.stack(code_hist, dim=1)
            pred_soft_i, _ = model.codelm(code_hist_t)
            code_prefix = model.z_proj(pred_soft_i[:, -1, :]).unsqueeze(1)
        block_in = torch.cat([code_prefix, byte_embed[:, i, :, :]], dim=1)
        running_seq = block_in if running_seq is None else torch.cat([running_seq, block_in], dim=1)
        h = model.run_blocks(running_seq)
        z_i = model.quantize(model.code_pre(h[:, -1, :]))
        code_hist.append(z_i)

    code_hist_t = torch.stack(code_hist, dim=1)
    pred_soft_i, _ = model.codelm(code_hist_t)
    code_prefix = model.z_proj(pred_soft_i[:, -1, :]).unsqueeze(1)   # forecast for the FIRST new block
    running_seq = torch.cat([running_seq, code_prefix], dim=1)

    out_bytes = [prompt_bytes]
    cur_block_bytes = []
    n_generated = 0
    while n_generated < n_new_bytes:
        h = model.run_blocks(running_seq)
        logits = model.head(h[:, -1, :])
        next_byte = logits.argmax(-1)
        out_bytes.append(next_byte.unsqueeze(1))
        running_seq = torch.cat([running_seq, model.byte_emb(next_byte).unsqueeze(1)], dim=1)
        cur_block_bytes.append(next_byte)
        n_generated += 1
        if len(cur_block_bytes) == K:
            h2 = model.run_blocks(running_seq)
            z_new = model.quantize(model.code_pre(h2[:, -1, :]))
            code_hist.append(z_new)
            code_hist_t = torch.stack(code_hist, dim=1)
            pred_soft_i, _ = model.codelm(code_hist_t)
            code_prefix_new = model.z_proj(pred_soft_i[:, -1, :]).unsqueeze(1)
            running_seq = torch.cat([running_seq, code_prefix_new], dim=1)
            cur_block_bytes = []
    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def qualitative_gen(model: SymmetricLM, data: torch.Tensor, prompt_len: int, n_new_bytes: int, device: str, log, step: int) -> None:
    K = model.cfg.K
    prompt_len = max(K, (prompt_len // K) * K)
    n_new_bytes = max(K, (n_new_bytes // K) * K)
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
def eval_model(model: SymmetricLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
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
    result["bpb"] = result["ntp_loss"] / math.log(2)
    return result


def train(model: SymmetricLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt9", dynamic_ncols=True)
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

        postfix = dict(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}",
            bpb=f"{metrics['ntp_loss'].item()/math.log(2):.4f}",
            code_acc=f"{metrics['code_conditioned_acc'].item()*100:.2f}%",
            within_acc=f"{metrics['within_block_acc'].item()*100:.2f}%",
            code_match_loss=f"{metrics['code_match_loss'].item():.4f}",
        )
        if "aux_recon_loss" in metrics:
            postfix["aux_recon_acc"] = f"{metrics['aux_recon_acc'].item()*100:.2f}%"
        if "encode_match_loss" in metrics:
            postfix["encode_match_loss"] = f"{metrics['encode_match_loss'].item():.4f}"
        pbar.set_postfix(**postfix)

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])

        if args.gen_every > 0 and (step % args.gen_every == 0 or step == args.steps):
            qualitative_gen(model, val_data, args.gen_prompt_len, args.gen_new_bytes, device, log, step)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="TRUE symmetric encode/decode via block-by-block AR bootstrap (fork of qcute.qcutelm_vlt8) — slow prefill, tests genuine symmetry", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--dq", type=int, default=6)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--lm_d_model", type=int, default=256)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=3)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--code_match_weight", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--trainable_bootstrap", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--aux_recon_weight", type=float, default=0.0)
    p.add_argument("--encode_match_weight", type=float, default=0.0)

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

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        K=args.K, context_len=args.context_len, dq=args.dq, quant_type=args.quant_type,
        fsq_levels=args.fsq_levels, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, lm_d_model=args.lm_d_model,
        lm_n_heads=args.lm_n_heads, lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult,
        code_match_weight=args.code_match_weight, rope_base=args.rope_base,
        trainable_bootstrap=args.trainable_bootstrap,
        aux_recon_weight=args.aux_recon_weight, encode_match_weight=args.encode_match_weight,
    )
    model = SymmetricLM(cfg).to(device)
    n_tokenizer = sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith("codelm"))
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))
    n_params = n_tokenizer + n_codelm

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt9_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} context_len={cfg.context_len} dq={cfg.dq} quant_type={cfg.quant_type} "
        f"params={n_params/1e6:.3f}M (tokenizer={n_tokenizer/1e6:.3f}M codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
