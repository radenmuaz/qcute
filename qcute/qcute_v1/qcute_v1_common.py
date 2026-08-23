import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def make_dict(**kwargs) -> dict:
    return kwargs


def resolve_per_level(value, n_levels: int) -> tuple:
    if isinstance(value, (tuple, list)):
        assert len(value) == n_levels, f"expected {n_levels} per-level values, got {len(value)}: {value!r}"
        return tuple(value)
    return (value,) * n_levels


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


WORD_PRESET_BITS = (1, 4, 8)


@dataclass
class Config:
    quant_type: str
    vocab: int
    binary_bits: int
    grid_dq: int
    grid_levels: int
    gmm_k: int
    gmm_dq: int
    input_preset: int
    output_preset: int
    decoder_type: str = "concat"
    Ks: tuple[int, ...] = (32, 32)
    d_model: int | tuple = 256
    n_layers: int | tuple = 2
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int | tuple = 32
    rope_base: float = 10000.0
    byte_ntp_weight: float = 1.0
    code_ntp_weight: float = 1.0
    decode_ntp_weight: float | tuple = 1.0
    gumbel_tau: float = 1.0
    code_extract_mode: str = "last_h"
    code_head_tied: bool = False
    ntp_head_tied: bool = False
    binary_lfq: bool = False
    entropy_reg_weight: float = 0.0
    grid_bound: str = "sigmoid"
    grid_logistic_scale: float = 0.5
    code_hard: bool = True
    code_sample: bool = False
    gmm_bpb_precision_bits: int = 8
    quant_dropout_p0: float = 0.0
    quant_dropout_decay_steps: int | None = None
    quant_dropout_schedule: str = "linear"
    decode_cross_stage_layers: int | None = None
    decoder_own_stage_mode: str = "shared"  # analogous to kv_lm_mode (2026-08-23, was
    # share_encode_decode_self: bool = False, i.e. "copy" was the old default): level i's own-stage
    # LM (track0's bb0 in make_own_stage/StackDecoder, own-code stage in ConcatDecoder) either
    # reuses encoders[i].lm directly ("shared", default) or is an independently-initialized/trained
    # LM ("copy"). Combined with kv_lm_mode's own default flip to "shared", the only genuinely NEW
    # (non-encoder-reused) decoder parameters left by default are the cross-attention machinery
    # itself (cross_attn_stage's per-layer cross-attn+MLP -- no encoder equivalent to reuse) and
    # each level's own NTP/prediction head.
    cond_depth: int = -1
    pq_chunks: int = 1
    track_dropout_p0: float = 0.0
    track_dropout_ramp_steps: int | None = None
    track_dropout_schedule: str = "linear"
    use_self_code: bool = False
    detach_ss_sample: bool = False  # detaches sample_next()'s STE sample (used by encoder_ste_p in
    # either mode) so gradient never reaches the level-above encoder that produced it
    encoder_ste_p: float = 0.0  # probability per forward pass of resampling every non-top level's
    # code from the level-above's own NTP prediction (sample_next(), STE unless detach_ss_sample) --
    # unifies the old separate scheduled_sampling_p mechanism (removed 2026-08-23) into one knob;
    # see encoder_ste_skip_real below for the two modes. Not "consistency" in a reconstruction-
    # comparison sense: it's STE training of the level-above's own NTP head via decode's
    # reconstruction loss as feedback. 1.0 = every step.
    encoder_ste_skip_real: bool = False  # False (default, additive): the real-code decode pass
    # always runs; when encoder_ste_p also fires, a SEPARATE second decode pass with the
    # self-sampled code runs too, added UNWEIGHTED on top (encoder_ste_total) -- decode's own
    # gradient path is unaffected, only the code-producer sees new signal. More stable empirically
    # (2026-08-23 comparison). True (skip, formerly scheduled_sampling_p's behavior): the
    # self-sampled code REPLACES the real-code pass entirely this step, mutually exclusive -- more
    # faithful to the true (100% self-sampled) generation-time distribution, but empirically less
    # stable (destabilizes the dominant training signal itself). See docs/status.md's "code-level
    # consistency training" discipline (#2/#3) for the motivating discussion.
    byte_consistency_p: float = 0.0  # probability per forward pass of a SECOND, whole-model pass:
    # argmax level0's own byte-level reconstruction logits, detach, feed that self-predicted byte
    # sequence through the WHOLE model again (self-supervised as always, so it reconstructs ITS OWN
    # input) -- tests whole-model idempotence/stability under one round of self-feeding, unlike
    # encoder_ste_p's code-level-only swap. 1.0 = every step. See docs/status.md's 2026-08-23 entry.
    uncertainty_weighting: bool = False
    active_srcs_mode: int | None | tuple = None  # NOTE: excluding the topmost level's code from
    # conditioning (the old "notoplevel" curriculum use, e.g. (1, None) for ks21 / (2, 1, None) for
    # ks221) is now baked into StackDecoder.__init__ unconditionally (2026-08-23, see its
    # docstring) -- these values are now no-ops for that purpose. Still useful for other max_srcs
    # ablations (e.g. forcing max_srcs=1 to drop ALL upper conditioning, not just the top level).
    active_srcs_until_step: int = 0
    kv_lm_mode: str = "shared"  # reuse the coarser level's own (already-trained) encoder LM as the
    # kv_lm by default (2026-08-23, was "identity") -- reasoning: level i's cross-attn stage
    # already has its own dedicated submodule (cross_attn_stage's LM) to actually consume the
    # code, so kv_lm's job is just producing a good K/V *representation* of that code, which the
    # encoder that already models it well is the natural source for, rather than training a
    # redundant "copy" LM from scratch. "identity" (no transform) and "copy" (renamed from
    # "fresh" 2026-08-23 -- a separate LM, independently initialized/trained, not tied to the
    # encoder) remain as alternatives.
    kv_lm_layers: int | None = None
    decode_scope: str = "level0_only"  # which levels' OWN decode_level gets computed/trained
    # (2026-08-23, default changed from implicit-pervasive) -- "level0_only" (default): only
    # level0's decode runs, since it's the only one anything downstream actually consumes (final
    # byte output); level i>0's OWN reconstruction of its own domain is never used by anything else
    # -- level0's upper-track cross-attention conditions on c_list[j] (the ENCODER's output)
    # directly, which stays fully available/trained via encode_losses regardless of whether level
    # j's OWN decode runs (decode_level(i=0)'s j in decode_derived_c fallback already handles this,
    # see qcute_v1.py). "pervasive": every level's own decode_level runs and contributes to
    # decode_total, the original (pre-2026-08-23) behavior -- still useful if you actually want
    # level i>0's own reconstruction quality as a training signal/diagnostic in its own right.
    byte_head_tied: bool = False  # mirrors the existing cfg.ntp_head_tied (which already governs
    # the CODE-level NTP head inside the quant classes, self.code_predict) but for the BYTE-level
    # case, which had no equivalent knob at all -- ntp_loss_acc's is_byte_level branch always
    # hardcoded F.linear(h, self.embed.weight), standard input/output tying, no alternative.
    # Default False (untied, LM.byte_head is a genuine nn.Linear(D, V, bias=False)) so that every
    # LM's byte-output projection is a free parameter by default, independent of whatever
    # input-embedding sharing is happening at the whole-LM level (decoder_own_stage_mode/
    # kv_lm_mode) -- 2026-08-23, chat: "instead of hard code F.linear use bb0.embed.weight / put
    # tie weight false, make by default nn.Linear bias false a free param even in copy or no
    # weight share". Motivated by a real gap: with decoder_own_stage_mode="shared", the encoder's
    # own unconditional byte-NTP loss and decode's code-conditioned reconstruction loss landed on
    # the EXACT SAME embed.weight tensor -- no head existed specific to "predict a byte from this
    # code-informed hidden state," even though cross-attention+MLP had already transformed that
    # hidden state into something semantically different from the unconditional case. True
    # (opt-in): ties LM.byte_head to embed.weight, the old only-possible behavior. WARNING (see
    # StackDecoder.__init__): combining byte_head_tied=True with decoder_own_stage_mode="shared"
    # collapses track0's conditional and unconditional heads into one tensor -- allowed, but a
    # printed warning recommends byte_head_tied=False whenever kv_lm_mode/decoder_own_stage_mode
    # reuse an encoder, so at least the output head stays independent.

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


def bsq_quantize(v: torch.Tensor, dq: int, hard: bool = True, sample: bool = False, lfq: bool = False) -> torch.Tensor:
    v_eff = v if lfq else F.normalize(v, dim=-1)
    scale = 1.0 if lfq else 1.0 / math.sqrt(dq)
    if not hard:
        return v_eff * scale
    if sample:
        probs = torch.sigmoid(v_eff)
        u = torch.rand_like(v_eff)
        hard_v = torch.where(u < probs, torch.ones_like(v_eff), -torch.ones_like(v_eff))
    else:
        hard_v = torch.sign(v_eff)
    return (v_eff + (hard_v - v_eff).detach()) * scale


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def bsq_entropy_reg(v: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(v)
    per_example = bernoulli_entropy(probs).sum(-1).mean()
    batch_avg = probs.reshape(-1, probs.size(-1)).mean(0)
    batch = bernoulli_entropy(batch_avg).sum()
    return per_example - batch


def softmax_entropy_reg(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    per_example = -(probs * logp).sum(-1).mean()
    batch_avg = probs.reshape(-1, probs.size(-1)).mean(0)
    batch = -(batch_avg * batch_avg.clamp_min(1e-9).log()).sum()
    return per_example - batch


def fsq_quantize(v: torch.Tensor, L: int, hard: bool = True, sample: bool = False, bound: str = "sigmoid",
                  logistic_scale: float = 0.5) -> torch.Tensor:
    """sample=True injects reparameterized logistic noise (PixelCNN++-style discretized-logistic
    dequantization) around z_bounded before either rounding (hard=True) or returning directly
    (hard=False) -- gives hard=False/sample=True a genuine stochastic soft sample, previously
    dead code (fsq_quantize used to return the deterministic z_bounded/half_l for any hard=False
    call, ignoring sample entirely)."""
    half_l = (L - 1) / 2
    z_bounded = half_l * (torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1))
    z_eff = z_bounded
    if sample:
        eps = torch.finfo(v.dtype).tiny
        u = torch.rand_like(z_bounded).clamp(min=eps, max=1.0 - eps)
        noise = logistic_scale * (torch.log(u) - torch.log(1.0 - u))
        z_eff = (z_bounded + noise).clamp(-half_l, half_l)
    if not hard:
        return z_eff / half_l
    z_rounded = torch.round(z_eff).clamp(-half_l, half_l)
    z_hat = z_bounded + (z_rounded - z_bounded).detach()
    return z_hat / half_l


MAX_PQ_TABLE_DQ = 16


class CodeEmbed(nn.Module):
    def __init__(self, dq: int, D: int):
        super().__init__()
        assert dq <= MAX_PQ_TABLE_DQ
        self.table = nn.Embedding(2 ** dq, D)
        self.proxy = nn.Linear(dq, D)
        self.register_buffer("_powers", (2 ** torch.arange(dq)).long(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = ((x > 0).long() * self._powers).sum(-1)
        hard = self.table(idx)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


class FSQEmbed(nn.Module):
    def __init__(self, dq: int, L: int, D: int):
        super().__init__()
        self.dq, self.L = dq, L
        self.table = nn.Parameter(torch.zeros(dq, L, D))
        nn.init.normal_(self.table, std=0.02)
        self.proxy = nn.Linear(dq, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bound = (self.L - 1) / 2
        levels = (x * bound + bound).round().long().clamp(0, self.L - 1)
        dq_idx = torch.arange(self.dq, device=x.device)
        hard = self.table[dq_idx, levels].sum(dim=-2)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


class QuantScheme:
    def __init__(self):
        self.quant_dropout_p = 0.0
        self.detach_ss_sample = False  # set from cfg.detach_ss_sample by make_quant()

    def _effective_hard_sample(self) -> tuple:
        """Quant Noise (Fan et al. 2020): with probability quant_dropout_p, take the plain
        continuous identity/soft pass (hard=False, sample=False -- exact gradient, no STE bias,
        no injected noise) instead of the configured hard/sample setting, but ONLY during real
        training forward passes (torch.is_grad_enabled()) -- every eval/generation call in this
        codebase already runs under torch.no_grad(), so this is a no-op there regardless of
        quant_dropout_p, guaranteeing deployment always sees the true self.hard/self.sample
        behavior. Schedule (quant_dropout_p itself) is driven externally by the training loop."""
        if self.quant_dropout_p > 0 and torch.is_grad_enabled() and torch.rand(()).item() < self.quant_dropout_p:
            return False, False
        return self.hard, self.sample

    def init_modules(self, D: int, V: int, code_head_tied: bool) -> tuple:
        raise NotImplementedError

    def quantize(self, pre_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_ids(self, source_c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed_for_decode(self, stage_lm: nn.Module, source_c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ntp_loss_acc(self, stage_lm: nn.Module, h_query: torch.Tensor, target_repr: torch.Tensor) -> tuple:
        raise NotImplementedError

    def embed_input(self, stage_lm: nn.Module, seq_repr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_next(self, stage_lm: nn.Module, h_query: torch.Tensor, vocab: int) -> torch.Tensor:
        raise NotImplementedError

    def entropy_reg(self, pre_q: torch.Tensor):
        return None


class SimplexQuant(QuantScheme):
    """pq_chunks > 1: product-quantized categorical code, standard PQ-literature convention
    (matches GMMQuant/GMMDiagQuant's own pq_chunks semantics) -- vocab (V) is the FIXED
    per-chunk codebook size, pq_chunks is a pure multiplier: total code width is V * pq_chunks
    (pq_chunks independent V-way softmaxes concatenated), combinatorial capacity V^pq_chunks.
    pq_chunks=1 (default) is exactly the original single V-way softmax."""

    def __init__(self, tau: float, hard: bool = True, sample: bool = False, ntp_head_tied: bool = False,
                 pq_chunks: int = 1):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.sample = sample
        self.ntp_head_tied = ntp_head_tied
        self.pq_chunks = pq_chunks

    def init_modules(self, D, V, code_head_tied):
        if code_head_tied and self.pq_chunks > 1:
            raise NotImplementedError(
                "code_head_tied=True is not supported with pq_chunks>1 (the tied embedding "
                "table is V-wide, not the V*pq_chunks-wide total code)")
        self.V_sub = V  # fixed per-chunk width, matches GMM's gmm_k role
        self.V = V * self.pq_chunks  # total code width
        code_head = None if code_head_tied else nn.Linear(D, self.V, bias=False)
        if code_head is not None:
            nn.init.normal_(code_head.weight, std=0.02)
        ntp_head = None if self.ntp_head_tied else nn.Linear(D, self.V, bias=False)
        if ntp_head is not None:
            nn.init.normal_(ntp_head.weight, std=0.02)
        # code_embed: dedicated categorical-code embedding table (V=cfg.vocab, matching code_head's
        # output width), separate from stage_lm.embed (that backbone's own INPUT alphabet table --
        # only equal to V when this level's input word width matches cfg.vocab, e.g. level0 under a
        # non-byte input_preset diverges). BSQ/FSQ already carry their own dedicated code_embed
        # module for the same reason; softmax previously reused stage_lm.embed.weight as a shortcut,
        # safe only when every level shared one global vocab.
        code_embed = nn.Linear(self.V, D, bias=False)
        nn.init.normal_(code_embed.weight, std=0.02)
        return code_head, code_embed, ntp_head

    def _chunked(self, x: torch.Tensor) -> torch.Tensor:
        """(..., V*pq_chunks) -> (..., pq_chunks, V) -- pure reshape, no data movement."""
        return x.reshape(*x.shape[:-1], self.pq_chunks, self.V_sub)

    def quantize(self, pre_q):
        hard, sample = self._effective_hard_sample()
        if self.pq_chunks == 1:
            return gumbel_quantize(pre_q, self.tau, hard, sample)
        return gumbel_quantize(self._chunked(pre_q), self.tau, hard, sample).reshape(pre_q.shape)

    def to_ids(self, source_c):
        chunk_ids = self._chunked(source_c).argmax(-1)  # (..., pq_chunks)
        if self.pq_chunks == 1:
            return chunk_ids[..., 0]
        weights = self.V_sub ** torch.arange(self.pq_chunks, device=source_c.device)
        return (chunk_ids * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def _ntp_logits(self, stage_lm, h_query):
        if self.ntp_head_tied:
            return F.linear(h_query, stage_lm.embed.weight)
        return stage_lm.code_predict(h_query)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        logits = self._chunked(self._ntp_logits(stage_lm, h_query)).reshape(-1, self.V_sub)
        target = self._chunked(target_repr).argmax(-1).reshape(-1)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        # `vocab` (cfg.vocab, the caller-passed per-chunk width) is unused here -- self.V
        # (the true total code width, V_sub * pq_chunks) is what the returned code must match.
        # Same hard/sample setting as quantize() (via _effective_hard_sample), so scheduled
        # sampling's substitute code goes through the identical STE path -- gradient flows back
        # to the level-above encoder unless detach_ss_sample forces the old detached behavior.
        logits = self._chunked(self._ntp_logits(stage_lm, h_query))  # (..., pq_chunks, V_sub)
        hard, sample = self._effective_hard_sample()
        onehot = gumbel_quantize(logits, self.tau, hard, sample)
        if self.detach_ss_sample:
            onehot = onehot.detach()
        return onehot.reshape(*h_query.shape[:-1], self.V)

    def entropy_reg(self, pre_q):
        chunked = self._chunked(pre_q)
        if self.pq_chunks == 1:
            return softmax_entropy_reg(chunked[..., 0, :])
        # per-chunk regularization -- each chunk is an independent codebook, so its own usage
        # marginal (the H(E[p]) anti-collapse term) must stay separate, not pooled across chunks.
        return torch.stack([softmax_entropy_reg(chunked[..., m, :]) for m in range(self.pq_chunks)]).sum()


class BinaryQuant(QuantScheme):
    def __init__(self, binary_bits: int, hard: bool = True, sample: bool = False, lfq: bool = False):
        super().__init__()
        self.binary_bits = binary_bits
        self.hard = hard
        self.sample = sample
        self.lfq = lfq

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.binary_bits, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = CodeEmbed(self.binary_bits, D)
        code_predict = nn.Linear(D, self.binary_bits, bias=False)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        hard, sample = self._effective_hard_sample()
        return bsq_quantize(pre_q, self.binary_bits, hard, sample, self.lfq)

    def to_ids(self, source_c):
        bits = (source_c > 0).long()
        weights = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        return (bits * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target_bits = (target_repr.reshape(-1, self.binary_bits) > 0).float()
        pred = stage_lm.code_predict(h_query)
        loss = F.binary_cross_entropy_with_logits(pred, target_bits)
        with torch.no_grad():
            acc = ((pred > 0).float() == target_bits).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        pred = stage_lm.code_predict(h_query)
        hard, sample = self._effective_hard_sample()
        out = bsq_quantize(pred, self.binary_bits, hard, sample, self.lfq)
        return out.detach() if self.detach_ss_sample else out

    def entropy_reg(self, pre_q):
        return bsq_entropy_reg(pre_q)


class GridQuant(QuantScheme):
    def __init__(self, dq: int, L: int, hard: bool = True, sample: bool = False, bound: str = "sigmoid",
                 logistic_scale: float = 0.5):
        super().__init__()
        self.dq, self.L, self.hard, self.sample, self.bound = dq, L, hard, sample, bound
        self.logistic_scale = logistic_scale

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.dq, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = FSQEmbed(self.dq, self.L, D)
        code_predict = nn.Linear(D, self.dq * self.L)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        hard, sample = self._effective_hard_sample()
        return fsq_quantize(pre_q, self.L, hard, sample, self.bound, self.logistic_scale)

    def to_ids(self, source_c):
        bound = (self.L - 1) / 2
        levels = (source_c * bound + bound).round().long().clamp(0, self.L - 1)
        weights = self.L ** torch.arange(self.dq, device=levels.device)
        return (levels * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        bound = (self.L - 1) / 2
        target_levels = (target_repr.reshape(-1, self.dq) * bound + bound).round().long().clamp(0, self.L - 1)
        pred = stage_lm.code_predict(h_query).reshape(-1, self.dq, self.L)
        loss = F.cross_entropy(pred.reshape(-1, self.L), target_levels.reshape(-1))
        with torch.no_grad():
            acc = (pred.argmax(-1) == target_levels).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        # No sampling noise here (unlike quantize()'s self.sample) -- there's no well-defined
        # sampling distribution over FSQ's L levels the way there is for a softmax; always a
        # plain deterministic STE pick (hard argmax forward, soft-softmax gradient backward),
        # tau=1.0 is arbitrary since GridQuant has no tau of its own. detach_ss_sample forces
        # the old fully-detached (no gradient to the level-above encoder) behavior.
        bound = (self.L - 1) / 2
        pred = stage_lm.code_predict(h_query).reshape(*h_query.shape[:-1], self.dq, self.L)
        onehot = gumbel_quantize(pred, 1.0, hard=True, sample=False)
        levels = (onehot * torch.arange(self.L, device=pred.device, dtype=pred.dtype)).sum(-1)
        if self.detach_ss_sample:
            levels = levels.detach()
        return (levels - bound) / bound


def solve_upper_triangular(U: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve U y = b for upper-triangular U via manual back-substitution -- plain
    elementwise ops only, no torch.linalg call (those have poor/no MPS support)."""
    dq = U.shape[-1]
    ys = [None] * dq
    for i in reversed(range(dq)):
        acc = b[..., i]
        for j in range(i + 1, dq):
            acc = acc - U[..., i, j] * ys[j]
        ys[i] = acc / U[..., i, i]
    return torch.stack(ys, dim=-1)


class GMMCodebook(nn.Module):
    """Full-covariance GMM codebook, optionally product-quantized: pq_chunks independent
    K-component codebooks, each over its own dq/pq_chunks-dim sub-vector, instead of one shared
    K-component codebook over the whole dq-dim code -- combinatorial capacity K^pq_chunks
    instead of a flat K (same PQ-VAE-style chunking as GridQuant/BSQ already get for free from
    their own per-dimension factorization; GMM's shared table doesn't factorize on its own,
    hence this explicit chunking). pq_chunks=1 (default) is exactly the original single-table
    behavior. Covariance parameterized via its precision Cholesky factor A (Lambda = A A^T) so
    the NLL's Mahalanobis term is a plain matmul (y = A^T(x-mu)) -- no solve/inverse needed.
    Sampling (z = mu + A^-T eps) needs one small manual triangular solve instead."""

    def __init__(self, K: int, dq: int, D: int, pq_chunks: int = 1):
        super().__init__()
        assert dq % pq_chunks == 0, f"gmm_dq ({dq}) must be divisible by pq_chunks ({pq_chunks})"
        self.K, self.dq, self.pq_chunks = K, dq, pq_chunks
        self.dq_sub = dq // pq_chunks
        M, dqs = pq_chunks, self.dq_sub
        self.mu = nn.Parameter(torch.empty(M, K, dqs).normal_(std=1.0))
        self.chol_raw = nn.Parameter(torch.empty(M, K, dqs, dqs).normal_(std=0.1))
        self.proj = nn.Linear(dq, D, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.register_buffer("_tril_off", torch.tril(torch.ones(dqs, dqs), diagonal=-1), persistent=False)
        self.register_buffer("_eye", torch.eye(dqs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def precision_chol(self) -> torch.Tensor:
        diag = F.softplus(self.chol_raw.diagonal(dim1=-2, dim2=-1)) + 1e-4
        return self.chol_raw * self._tril_off + diag.unsqueeze(-1) * self._eye

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., pq_chunks, dq_sub) -> (..., pq_chunks, K)."""
        A = self.precision_chol()
        diff = x.unsqueeze(-2) - self.mu
        y = torch.einsum("...mkd,mkde->...mke", diff, A)
        maha = (y * y).sum(-1)
        log_diag_sum = A.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).log().sum(-1)
        return log_diag_sum - 0.5 * (self.dq_sub * math.log(2 * math.pi) + maha)

    def sample(self, k_idx: torch.Tensor) -> torch.Tensor:
        """k_idx: (..., pq_chunks) per-chunk selected component -> (..., pq_chunks, dq_sub)."""
        A = self.precision_chol()
        M = self.pq_chunks
        A_sel = torch.stack([A[m][k_idx[..., m]] for m in range(M)], dim=-3)
        mu_sel = torch.stack([self.mu[m][k_idx[..., m]] for m in range(M)], dim=-2)
        eps = torch.randn(*k_idx.shape, self.dq_sub, device=self.mu.device, dtype=self.mu.dtype)
        y = solve_upper_triangular(A_sel.transpose(-1, -2), eps)
        return mu_sel + y

    def sample_all(self, batch_shape: tuple) -> torch.Tensor:
        """Reparam-sample every component of every chunk at once -- one precision_chol() call
        and one batched triangular solve, instead of calling sample() K times per chunk (each
        redundantly recomputing precision_chol() -- O(K^2) work for what should be O(K))."""
        A = self.precision_chol()
        eps = torch.randn(*batch_shape, self.pq_chunks, self.K, self.dq_sub,
                           device=self.mu.device, dtype=self.mu.dtype)
        y = solve_upper_triangular(A.transpose(-1, -2), eps)
        return self.mu + y


class GMMDiagCodebook(nn.Module):
    """Diagonal-covariance GMM codebook -- same PQ chunking / interface as GMMCodebook, plain
    elementwise NLL/reparam (no triangular solve needed either direction)."""

    def __init__(self, K: int, dq: int, D: int, pq_chunks: int = 1):
        super().__init__()
        assert dq % pq_chunks == 0, f"gmm_dq ({dq}) must be divisible by pq_chunks ({pq_chunks})"
        self.K, self.dq, self.pq_chunks = K, dq, pq_chunks
        self.dq_sub = dq // pq_chunks
        M, dqs = pq_chunks, self.dq_sub
        self.mu = nn.Parameter(torch.empty(M, K, dqs).normal_(std=1.0))
        self.logvar = nn.Parameter(torch.zeros(M, K, dqs))
        self.proj = nn.Linear(dq, D, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.mu
        maha = (diff * diff * (-self.logvar).exp()).sum(-1)
        return -0.5 * (self.dq_sub * math.log(2 * math.pi) + self.logvar.sum(-1) + maha)

    def sample(self, k_idx: torch.Tensor) -> torch.Tensor:
        M = self.pq_chunks
        mu_sel = torch.stack([self.mu[m][k_idx[..., m]] for m in range(M)], dim=-2)
        logvar_sel = torch.stack([self.logvar[m][k_idx[..., m]] for m in range(M)], dim=-2)
        std = (0.5 * logvar_sel).exp()
        return mu_sel + std * torch.randn_like(std)

    def sample_all(self, batch_shape: tuple) -> torch.Tensor:
        std = (0.5 * self.logvar).exp()
        eps = torch.randn(*batch_shape, self.pq_chunks, self.K, self.dq_sub,
                           device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + std * eps


class GMMHead(nn.Module):
    def __init__(self, D: int, dq: int, K: int, pq_chunks: int = 1):
        super().__init__()
        assert dq % pq_chunks == 0
        self.dq, self.pq_chunks, self.K = dq, pq_chunks, K
        self.proj = nn.Linear(D, dq + pq_chunks * K, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> tuple:
        out = self.proj(h)
        query = out[..., :self.dq]
        gating = out[..., self.dq:].reshape(*h.shape[:-1], self.pq_chunks, self.K)
        return query, gating


class _GMMQuantBase(QuantScheme):
    codebook_cls = None

    def __init__(self, K: int, dq: int, hard: bool = True, sample: bool = False, pq_chunks: int = 1):
        super().__init__()
        assert dq % pq_chunks == 0, f"gmm_dq ({dq}) must be divisible by pq_chunks ({pq_chunks})"
        self.K, self.dq, self.pq_chunks = K, dq, pq_chunks
        self.dq_sub = dq // pq_chunks
        self.hard = hard
        self.sample = sample
        self._codebook = None

    def init_modules(self, D, V, code_head_tied):
        code_head = GMMHead(D, self.dq, self.K, self.pq_chunks)
        code_predict = GMMHead(D, self.dq, self.K, self.pq_chunks)
        self._codebook = self.codebook_cls(self.K, self.dq, D, self.pq_chunks)
        return code_head, self._codebook, code_predict

    def _reshape_query(self, query: torch.Tensor) -> torch.Tensor:
        return query.reshape(*query.shape[:-1], self.pq_chunks, self.dq_sub)

    def _flatten_code(self, code_chunks: torch.Tensor) -> torch.Tensor:
        return code_chunks.reshape(*code_chunks.shape[:-2], self.dq)

    def _posterior_logits(self, query, gating_logits):
        """query: (..., dq) flat; gating_logits: (..., pq_chunks, K) -> (..., pq_chunks, K)."""
        return F.log_softmax(gating_logits, dim=-1) + self._codebook.log_prob(self._reshape_query(query))

    def _select(self, query, gating_logits, hard, sample):
        cb = self._codebook
        M = self.pq_chunks
        log_post = self._posterior_logits(query, gating_logits)  # (..., M, K)
        r = F.softmax(log_post, dim=-1)
        soft = torch.einsum("...mk,mkd->...md", r, cb.mu)  # (..., M, dq_sub)
        if not hard and not sample:
            return self._flatten_code(soft)
        if sample:
            eps = torch.finfo(log_post.dtype).tiny
            u = torch.rand_like(log_post).clamp(min=eps, max=1.0 - eps)
            weights = F.softmax(log_post - torch.log(-torch.log(u)), dim=-1)
        else:
            weights = r
        if not hard:
            samples = cb.sample_all(query.shape[:-1])  # (..., M, K, dq_sub)
            return self._flatten_code((weights.unsqueeze(-1) * samples).sum(-2))
        k_star = weights.argmax(-1)  # (..., M)
        if sample:
            hard_code = cb.sample(k_star)
        else:
            hard_code = torch.stack([cb.mu[m][k_star[..., m]] for m in range(M)], dim=-2)
        code = soft + (hard_code - soft).detach()
        return self._flatten_code(code)

    def quantize(self, pre_q):
        query, gating_logits = pre_q
        hard, sample = self._effective_hard_sample()
        return self._select(query, gating_logits, hard, sample)

    def _chunk_ids(self, chunks: torch.Tensor) -> torch.Tensor:
        """chunks: (..., pq_chunks, dq_sub) -> (..., pq_chunks) nearest-component id per chunk
        (uncombined -- used directly for per-chunk cross-entropy in ntp_loss_acc)."""
        dists = ((chunks.unsqueeze(-2) - self._codebook.mu) ** 2).sum(-1)
        return dists.argmin(-1)

    def to_ids(self, source_c):
        chunk_ids = self._chunk_ids(self._reshape_query(source_c))
        if self.pq_chunks == 1:
            return chunk_ids[..., 0]
        weights = self.K ** torch.arange(self.pq_chunks, device=source_c.device)
        return (chunk_ids * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        logits = self._posterior_logits(query_pred, gating_pred).reshape(-1, self.pq_chunks, self.K)
        target_id = self._chunk_ids(self._reshape_query(target_repr.reshape(-1, self.dq)))
        loss = F.cross_entropy(logits.reshape(-1, self.K), target_id.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(-1) == target_id).float().mean()
        return loss, acc

    def sample_next(self, stage_lm, h_query, vocab):
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        hard, sample = self._effective_hard_sample()
        out = self._select(query_pred, gating_pred, hard, sample)
        return out.detach() if self.detach_ss_sample else out

    def bpb_bound(self, stage_lm, h_query, target_repr, precision_bits: int) -> torch.Tensor:
        """Achievable bpb bound for code_hard=False / code_sample=True, where ntp_loss_acc's
        K-way cross-entropy against to_ids() undercounts (it only charges for "which of K
        components", discarding the continuous residual those modes actually emit). Uses the
        true mixture density's NLL of the exact target_repr (nats -> bits), summed over
        independent chunks (log joint = sum of per-chunk log-likelihoods), plus a stated
        per-dim quantization-precision correction (differential entropy alone isn't bits without
        one, same reason RealNVP/Glow-style bits/dim reporting adds a fixed dequantization
        constant) -- an honest, achievable upper bound, not exact."""
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        cb = self._codebook
        log_pi = F.log_softmax(gating_pred.reshape(-1, self.pq_chunks, self.K), dim=-1)
        log_lik = cb.log_prob(self._reshape_query(target_repr.reshape(-1, self.dq)))
        nll_nats_per_chunk = -torch.logsumexp(log_pi + log_lik, dim=-1)  # (-1, pq_chunks)
        nll_nats = nll_nats_per_chunk.sum(-1).mean()
        return nll_nats / math.log(2) + self.dq * precision_bits

    def entropy_reg(self, pre_q):
        query, gating_logits = pre_q
        logits = self._posterior_logits(query, gating_logits)  # (..., M, K)
        if self.pq_chunks == 1:
            return softmax_entropy_reg(logits[..., 0, :])
        # per-chunk regularization -- same reasoning as SimplexQuant.entropy_reg's pq_chunks
        # branch: each chunk is an independent codebook, must not be pooled with the others.
        return torch.stack([softmax_entropy_reg(logits[..., m, :]) for m in range(self.pq_chunks)]).sum()


class GMMQuant(_GMMQuantBase):
    codebook_cls = GMMCodebook


class GMMDiagQuant(_GMMQuantBase):
    codebook_cls = GMMDiagCodebook


def make_quant(cfg: Config) -> QuantScheme:
    if cfg.quant_type == "binary":
        quant = BinaryQuant(cfg.binary_bits, cfg.code_hard, cfg.code_sample, cfg.binary_lfq)
    elif cfg.quant_type == "grid":
        quant = GridQuant(cfg.grid_dq, cfg.grid_levels, cfg.code_hard, cfg.code_sample, cfg.grid_bound,
                           cfg.grid_logistic_scale)
    elif cfg.quant_type == "gmm":
        quant = GMMQuant(cfg.gmm_k, cfg.gmm_dq, cfg.code_hard, cfg.code_sample, cfg.pq_chunks)
    elif cfg.quant_type == "gmm_diag":
        quant = GMMDiagQuant(cfg.gmm_k, cfg.gmm_dq, cfg.code_hard, cfg.code_sample, cfg.pq_chunks)
    else:
        quant = SimplexQuant(cfg.gumbel_tau, cfg.code_hard, cfg.code_sample, cfg.ntp_head_tied, cfg.pq_chunks)
    quant.detach_ss_sample = cfg.detach_ss_sample
    return quant


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def chunked_windowed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    B, H, T, hd = q.shape
    w = window
    device = q.device
    if T <= w:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if T % w != 0:
        pos = torch.arange(T, device=device)
        ti, tj = pos.unsqueeze(1), pos.unsqueeze(0)
        attn_mask = ((tj <= ti) & (ti - tj < w)).view(1, 1, T, T)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    n_chunks = T // w
    qb = q.view(B, H, n_chunks, w, hd)
    kb = k.view(B, H, n_chunks, w, hd)
    vb = v.view(B, H, n_chunks, w, hd)
    pad_k = torch.zeros(B, H, 1, w, hd, device=device, dtype=k.dtype)
    pad_v = torch.zeros(B, H, 1, w, hd, device=device, dtype=v.dtype)
    k_ext = torch.cat([pad_k, kb], dim=2)
    v_ext = torch.cat([pad_v, vb], dim=2)

    idx = torch.arange(n_chunks, device=device).view(n_chunks, 1) + torch.arange(2, device=device).view(1, 2)
    k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)
    v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)

    pos = torch.arange(T, device=device)
    pos_b = pos.view(n_chunks, w)
    pad_pos = torch.full((1, w), -10 ** 9, device=device, dtype=pos.dtype)
    pos_ext = torch.cat([pad_pos, pos_b], dim=0)
    pos_win = pos_ext[idx].reshape(n_chunks, 2 * w)

    ti = pos_b.unsqueeze(-1)
    tj = pos_win.unsqueeze(1)
    allow = (tj <= ti) & (ti - tj < w)
    mask_flat = allow.view(1, n_chunks, 1, w, 2 * w).expand(B, n_chunks, 1, w, 2 * w).reshape(B * n_chunks, 1, w, 2 * w)

    qb_flat = qb.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, w, hd)
    k_win_flat = k_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)
    v_win_flat = v_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)

    y = F.scaled_dot_product_attention(qb_flat, k_win_flat, v_win_flat, attn_mask=mask_flat)
    return y.view(B, n_chunks, H, w, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None:
            y = chunked_windowed_attention(q, k, v, window)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        a = self.attn(self.ln1(x), cos, sin, window)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_cross(self, x: torch.Tensor, code_kv: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor,
                       cos_k: torch.Tensor, sin_k: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        xn = self.ln1(x)
        coden = self.ln1(code_kv)
        a = self.attn.forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


_warned_thin_window: set = set()


def warn_thin_window(tracks: list, window: int, min_codes: int = 2) -> None:
    key = tuple(K for _, K, _ in tracks) + (window,)
    if key in _warned_thin_window:
        return
    total_codes = sum(max(0, window // K) for _, K, _ in tracks)
    if total_codes < min_codes:
        _warned_thin_window.add(key)
        Ks_str = ",".join(str(K) for _, K, _ in tracks)
        print(f"WARNING: windowed decode window={window} covers only ~{total_codes} cumulative code(s) "
              f"across tracks Ks=({Ks_str}) -- below min_codes={min_codes}.")


class LM(nn.Module):
    def __init__(self, cfg: Config, d_model: int, n_layers: int, vocab: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.vocab = vocab
        self.quant = make_quant(cfg)
        D = d_model
        V = vocab
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        # byte_head: mirrors code_predict's ntp_head_tied pattern above, but for the byte-level
        # output projection (ntp_loss_acc's is_byte_level branch) -- None (tied to self.embed.weight,
        # standard input/output tying) when cfg.byte_head_tied, else a genuine independent
        # nn.Linear(D, V, bias=False), the default (2026-08-23).
        self.byte_head = None if cfg.byte_head_tied else nn.Linear(D, V, bias=False)
        if self.byte_head is not None:
            nn.init.normal_(self.byte_head.weight, std=0.02)
        # trainable per-block SEED TOKEN, prepended before every K-block in qcute_v1's non-top-level
        # decode (see docs/qcute_v1_plan.md) -- not "BOS" (a single sequence-start marker) or a
        # "sink" (a passive fallback key that itself predicts nothing): it's a full token, going
        # through self-attn/MLP/cross-attn like any real byte and genuinely predicting its block's
        # own first byte from that block's own code (see StackDecoderV1, own_block_cross_attn_decode).
        # Each block's self-attention chain is freshly seeded by this same constant rather than
        # inheriting state from the previous block, since decode no longer fuses any code into
        # self-attention at all (own-level code is cross-attended instead).
        self.self_code_const = nn.Parameter(torch.zeros(D))
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D)
        # code_head/code_predict always produce/consume cfg.vocab-wide codes (the categorical code
        # convention shared by every level going UP the hierarchy) -- independent of `vocab` above,
        # which only sizes this backbone's own INPUT embedding table (word alphabet, level0 only
        # differs from cfg.vocab when input_preset != 8).
        self.code_head, self.code_embed, self.code_predict = self.quant.init_modules(D, cfg.vocab, cfg.code_head_tied)
        self.code_query = self.code_out = self.query_embed = None
        if cfg.code_extract_mode == "light_query_attn":
            self.code_query = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.code_query, std=0.02)
            self.code_out = nn.Linear(D, D, bias=False)
        elif cfg.code_extract_mode == "query_embed":
            self.query_embed = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.query_embed, std=0.02)

    def classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.code_head(pooled) if self.code_head is not None else F.linear(pooled, self.embed.weight)

    def query_embed_pool(self, x0: torch.Tensor, K: int, n_blocks: int, window: int | None) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device

        x0_blocks = x0.view(B, n_blocks, K, D)
        q_tok = self.query_embed.view(1, 1, 1, D).expand(B, n_blocks, 1, D)
        xe = torch.cat([x0_blocks, q_tok], dim=2).view(B, n_blocks * (K + 1), D)
        Le = n_blocks * (K + 1)

        slot = torch.arange(K + 1, device=device).repeat(n_blocks)
        is_query = slot == K
        block_of = torch.arange(n_blocks, device=device).repeat_interleave(K + 1)
        within_block_pos = torch.where(is_query, torch.full_like(slot, K - 1), slot)
        real_pos = block_of * K + within_block_pos

        cos, sin = rope_cos_sin_for_positions(real_pos, hd, cfg.rope_base, device)

        win = window if window is not None else L
        ti = real_pos.unsqueeze(1)
        tj = real_pos.unsqueeze(0)
        causal = tj <= ti
        windowed = (ti - tj) < win
        query_only_sees_own_block = (~is_query).unsqueeze(0) | (block_of.unsqueeze(1) == block_of.unsqueeze(0))
        real_never_sees_query = ~(is_query.unsqueeze(0) & (~is_query).unsqueeze(1))
        allow = causal & windowed & query_only_sees_own_block & real_never_sees_query
        attn_mask = allow.view(1, 1, Le, Le)

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_blocks = he.view(B, n_blocks, K + 1, D)
        return he_blocks[:, :, K, :]

    @property
    def byte_output_weight(self) -> torch.Tensor:
        """The byte-level output projection -- self.byte_head.weight (independent, default) or
        self.embed.weight (tied, cfg.byte_head_tied=True). The single source of truth every
        byte-output call site (ntp_loss_acc, generation sampling, embed_weight dict fields) should
        read from instead of hardcoding self.embed.weight (2026-08-23 fix)."""
        return self.embed.weight if self.byte_head is None else self.byte_head.weight

    def ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor, is_byte_level: bool) -> tuple:
        if is_byte_level:
            target = target_repr.reshape(-1)
            logits = F.linear(h_query, self.byte_output_weight)
            loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
            return loss, acc
        return self.quant.ntp_loss_acc(self, h_query, target_repr)

    def embed_input(self, seq_repr: torch.Tensor, is_byte_level: bool) -> torch.Tensor:
        if is_byte_level:
            return self.embed(seq_repr)
        return self.quant.embed_input(self, seq_repr)

    def extract_code(self, h: torch.Tensor, x0: torch.Tensor, K: int, window: int | None) -> dict:
        cfg = self.cfg
        B, L, D = h.shape
        n_blocks = L // K
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        if cfg.code_extract_mode == "last_h":
            pooled = h_blocks[:, :, K - 1, :]
        elif cfg.code_extract_mode == "softmax_pool":
            q_implicit = h_blocks[:, :, K - 1, :]
            scores = (h_blocks * q_implicit.unsqueeze(2)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
        elif cfg.code_extract_mode == "light_query_attn":
            scores = (h_blocks * self.code_query.view(1, 1, 1, D)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
            pooled = self.code_out(pooled)
        elif cfg.code_extract_mode == "query_embed":
            pooled = self.query_embed_pool(x0, K, n_blocks, window)
        else:
            raise ValueError(f"unknown code_extract_mode {cfg.code_extract_mode!r}")
        pre_q = self.classify(pooled)
        code = self.quant.quantize(pre_q)
        entropy_reg = self.quant.entropy_reg(pre_q)
        return make_dict(code=code, entropy_reg=entropy_reg)


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


_warned_short_data: set = set()


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    if len(data) < context_len and id(data) not in _warned_short_data:
        _warned_short_data.add(id(data))
        print(f"WARNING: sample_context data ({len(data)} bytes) is shorter than context_len ({context_len})")
    n = max(1, len(data) - context_len)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + context_len] for s in starts]).to(device)


def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def lr_at_warmup_constant_cosine(step: int, warmup: int, constant_steps: int, peak: float,
                                  total_steps: int, min_lr_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_start = warmup + constant_steps
    if step < decay_start:
        return peak
    min_lr = peak * min_lr_frac
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def quant_dropout_p_at(step: int, p0: float, decay_steps: int, schedule: str = "linear") -> float:
    """schedule="step": AE-Warm-Up style (Zhao et al. 2026, "Continuous First, Discrete Later") --
    quantization fully disabled (p=p0, typically 1.0) for decay_steps, then instantly fully
    enabled (p=0) -- a hard phase switch, not a ramp. schedule="linear" (default, unchanged):
    the original gradual Quant Noise-style decay."""
    if p0 <= 0 or decay_steps <= 0:
        return 0.0
    if schedule == "step":
        return p0 if step < decay_steps else 0.0
    return max(0.0, p0 * (1.0 - step / decay_steps))


def track_dropout_p_at(step: int, p0: float, ramp_steps: int, schedule: str = "linear") -> float:
    """Dense->sparse curriculum for decode's cross-level tracks (see CLAUDE.md's "Latent-AR /
    parallel-block-local-decode investigation") -- the OPPOSITE ramp direction from
    quant_dropout_p_at: starts at 0 (fully dense, today's behavior, eases early training the
    same way DenseNet's dense skips do) and ramps UP toward p0 (probability of pruning every
    track except self+topmost) over ramp_steps. schedule="step": AE-Warm-Up-style hard switch
    (dense until ramp_steps, then instantly p0). schedule="linear" (default): gradual ramp."""
    if p0 <= 0 or ramp_steps <= 0:
        return 0.0
    if schedule == "step":
        return 0.0 if step < ramp_steps else p0
    return min(p0, p0 * step / ramp_steps)


def apply_track_dropout(tracks: list, p: float) -> list:
    """Returns a list of (original_index, track) pairs -- ALWAYS, even when nothing is pruned --
    since StackDecoderV1's per-track stage weights (self.stage_lms[i][t]) are keyed by each
    track's ORIGINAL position (0=self, n_sources-1=topmost), not its position within whatever
    survives pruning; losing that would run topmost-track data through the wrong stage's
    weights. With probability p, prunes every track except the first (self, j==i) and the last
    (topmost coarser level) -- one flip per call (matches quant_dropout's own per-call
    granularity), not per-track. No-op when there are <=2 tracks already (nothing "middle" to
    drop) or the flip doesn't fire. Caller gates this to training only via torch.is_grad_enabled()."""
    indexed = list(enumerate(tracks))
    if len(tracks) <= 2 or p <= 0:
        return indexed
    if torch.rand(()).item() < p:
        return [indexed[0], indexed[-1]]
    return indexed


def set_track_dropout_p(model, p: float) -> None:
    model.track_dropout_p = p


def self_code_active(level: int, n_levels: int, use_self_code: bool) -> bool:
    """Top level of a real (n_levels>=2) hierarchy always keeps genuine self-code -- it's the
    only cross-block memory a level with nothing above it has, irreplaceable. Every other level
    (including the sole level of an n_levels==1 config) follows cfg.use_self_code (default
    False): inactive self-code is replaced by a trainable constant seed (LM.self_code_const),
    NOT dropped -- except at n_levels==1, where there's no coarser track to fall back on either,
    so decode_level returns None and the model falls back to the encoder's own unconditioned
    NTP loss (see qcute_v1.QCuteLM.forward's decode_losses[0] fallback)."""
    return use_self_code or (n_levels >= 2 and level == n_levels - 1)


_warned_no_self_code: set = set()


def warn_degenerate_self_code(level: int) -> None:
    if level in _warned_no_self_code:
        return
    _warned_no_self_code.add(level)
    print(f"WARNING: level{level} use_self_code=False with n_levels==1 -- decode degenerates to "
          f"an unconditioned LM (encode-only fallback, equivalent to qcute_v1_wordlm). Confirm intended.")


def _walk_lms(obj):
    if hasattr(obj, "quant"):
        yield obj
    else:
        for item in obj:
            yield from _walk_lms(item)


def compute_effective_dim(x: torch.Tensor) -> float:
    """Participation ratio (sum(eigvals)^2 / sum(eigvals^2)) of x's (N, D) covariance -- a
    standard "effective rank" proxy for the dimensional-collapse diagnostic in Zhao et al. 2026
    ("Continuous First, Discrete Later"): ranges [1, D], 1 for a fully collapsed (rank-1)
    representation, D for an isotropic one. Their own criterion (a water-filling capacity bound
    from their Theorem 1) needs the exact target rate/codebook size and isn't reproduced here --
    this is a lighter, standard stand-in for MONITORING ONLY (see train()'s
    --monitor_effective_dim: prints when it plateaus, does not drive any schedule). Runs on CPU
    (torch.linalg.eigvalsh has poor/no MPS support) -- fine since this is called at most once per
    --eval_every, not in the hot per-step path."""
    x = x.detach().to("cpu", dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    s = eigvals.sum().item()
    if s <= 0:
        return 0.0
    return (s ** 2) / (eigvals ** 2).sum().item()


def set_quant_dropout_p(model, p: float) -> None:
    for enc in model.encoders:
        enc.quant.quant_dropout_p = p
    for lm in _walk_lms(model.decoder.stage_lms):
        lm.quant.quant_dropout_p = p


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


@torch.no_grad()
def add_per_level_bpb(result: dict) -> dict:
    for k in list(result.keys()):
        if k.endswith("_ntp_loss_encode") or k.endswith("_ntp_loss_decode"):
            result[k.replace("_ntp_loss_", "_bpb_")] = result[k] / math.log(2)
    return result


def eval_model(model, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    result["bpb_full"] = result["byte_loss_full"] / math.log(2)
    return add_per_level_bpb(result)


def parse_eval_sample(s):
    if isinstance(s, (int, float)):
        return s
    s = str(s)
    return float(s) if "." in s else int(s)


@torch.no_grad()
def eval_model_full(model, data: torch.Tensor, batch_size: int, device: str, sample: float | int = 1.0) -> dict:
    """Iterates the whole val set by default (sample=1.0). sample: float -> fraction of windows,
    int -> absolute window count (e.g. sample=0.1 or sample=500 for a faster partial full-eval).
    batch_size is the per-forward-pass CHUNK size within that selection -- -1 means no chunking
    (single giant batch, the old default): full_val_eval's own single-shot giant-batch forward
    pass was implicated in a recurring MPS eval glitch (decode-side metrics silently coming back
    as exact 0.0 on some eval rounds, confirmed via clean CPU replay of the same checkpoint/data --
    not a logic bug), so callers should pass a bounded chunk size by default, not -1."""
    model.eval()
    context_len = model.cfg.context_len
    n_windows_total = len(data) // context_len
    n_windows = max(1, round(n_windows_total * sample)) if isinstance(sample, float) else max(1, min(sample, n_windows_total))
    step_batch = n_windows if batch_size == -1 else batch_size
    accum: dict = {}
    total_n = 0
    for start in range(0, n_windows, step_batch):
        idxs = range(start, min(start + step_batch, n_windows))
        ctx = torch.stack([data[i * context_len:(i + 1) * context_len] for i in idxs]).to(device)
        _, metrics = model(ctx)
        bsz = ctx.size(0)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v.item() * bsz
        total_n += bsz
    model.train()
    result = {k: v / total_n for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    result["bpb_full"] = result["byte_loss_full"] / math.log(2)
    return add_per_level_bpb(result)


def build_param_groups(model) -> list:
    seen: set = set()
    params = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params += [p]
    return [{"params": params}]


def parse_int_tuple(s) -> tuple:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def parse_scalar_or_tuple(s):
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    if isinstance(s, int):
        return s
    s = str(s)
    return tuple(int(x) for x in s.split(",")) if "," in s else int(s)


def train(model, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.logs_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    last_effective_dim = None
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr
        set_quant_dropout_p(model, quant_dropout_p_at(step, args.quant_dropout_p0,
                                                        args.quant_dropout_decay_steps or args.steps,
                                                        args.quant_dropout_schedule))
        set_track_dropout_p(model, track_dropout_p_at(step, args.track_dropout_p0,
                                                        args.track_dropout_ramp_steps or args.steps,
                                                        args.track_dropout_schedule))

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        cur_max_srcs = (args.active_srcs_mode if step < args.active_srcs_until_step else None)
        loss, metrics = model(ctx, max_srcs=cur_max_srcs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
                          byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%")
        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb
            uncertainty_str = ""
            if args.uncertainty_weighting:
                uncertainty_str = "  " + "  ".join(
                    f"sigma_{k[len('uncertainty_sigma_'):]}={v:.4f}"
                    for k, v in train_scalars.items() if k.startswith("uncertainty_sigma_"))
            log(f"{pbar}{uncertainty_str}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            if args.full_val_eval:
                val = eval_model_full(model, val_data, args.eval_chunk_size, device, args.eval_sample)
            else:
                val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.monitor_effective_dim:
                with torch.no_grad():
                    dim_ctx = sample_context(val_data, args.batch_size, model.cfg.context_len, device)
                    dim_result = model._run(dim_ctx, compute_ntp=False)
                code0 = dim_result["c_list"][0].reshape(-1, dim_result["c_list"][0].shape[-1])
                effective_dim = compute_effective_dim(code0)
                plateaued = (last_effective_dim is not None
                             and abs(effective_dim - last_effective_dim) / max(last_effective_dim, 1e-8)
                             < args.dim_monitor_plateau_tol)
                plateau_note = "  PLATEAU (consider switching quant on now, AE-Warm-Up style)" if plateaued else ""
                log(f"{pbar}  effective_dim={effective_dim:.4f}{plateau_note}",
                    step=step, effective_dim=effective_dim)
                last_effective_dim = effective_dim

            if args.qual_gen_bytes > 0:
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    model.decoder.qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                                         window[args.qual_prompt_bytes:], device, log=log, label=label)
                    model.decoder.check_gen_consistency(model, window, device, prompt_len=args.qual_prompt_bytes,
                                                          log=log, label=label)
                    model.decoder.check_roundtrip_consistency(model, window, device, log=log, label=label)
                    model.decoder.check_decode_modes(model, window, device, log=log, label=label)


def build_argparser(description: str) -> tuple:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=description, parents=[pre])
    p.add_argument("--decoder_type", type=str, default="concat",
                    choices=["concat", "stack_v1", "stack", "stack_local", "stack_sync"])
    p.add_argument("--Ks", default=(32, 32))
    p.add_argument("--d_model", type=parse_scalar_or_tuple, default=256)
    p.add_argument("--n_layers", type=parse_scalar_or_tuple, default=2)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", default=32)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--decode_ntp_weight", type=float, default=1.0)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--code_hard", type=lambda x: x.lower() != "false", default=True,
                    help="True: hard code with straight-through (round/argmax/sign). False: continuous relaxed code")
    p.add_argument("--code_sample", action="store_true",
                    help="inject stochastic noise before quantizing (gumbel for simplex, bernoulli/uniform dither for binary/grid)")
    p.add_argument("--code_extract_mode", type=str, default="last_h",
                    choices=["last_h", "softmax_pool", "light_query_attn", "query_embed"])
    p.add_argument("--code_head_tied", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--vocab", type=int, default=None)
    p.add_argument("--input_preset", type=int, default=None, choices=list(WORD_PRESET_BITS),
                    help="level0 input word width in bits: 1 (bit), 4 (nibble), or 8 (byte)")
    p.add_argument("--output_preset", type=int, default=None, choices=list(WORD_PRESET_BITS),
                    help="level0 output/generation word width in bits: 1 (bit), 4 (nibble), or 8 (byte) -- "
                         "must currently equal --input_preset, asymmetric presets not yet implemented")
    p.add_argument("--quant_type", type=str, default=None, choices=["simplex", "binary", "grid", "gmm", "gmm_diag"])
    p.add_argument("--binary_bits", type=int, default=None)
    p.add_argument("--binary_lfq", action="store_true")
    p.add_argument("--entropy_reg_weight", type=float, default=0.0)
    p.add_argument("--ntp_head_tied", action="store_true")
    p.add_argument("--byte_head_tied", action="store_true",
                    help="ties every LM's byte-level output projection to its own input embedding "
                         "table (default False: independent nn.Linear(D, V, bias=False) byte_head). "
                         "Mirrors --ntp_head_tied, which already does this for the code-level head.")
    p.add_argument("--decode_cross_stage_layers", type=int, default=None)
    p.add_argument("--decoder_own_stage_mode", choices=["copy", "shared"], default="shared",
                    help="level i's own-stage decode LM (default since 2026-08-23: 'shared' -- "
                         "reuses encoders[i].lm directly, was 'copy'/share_encode_decode_self=False). "
                         "'copy': an independently-initialized/trained LM instead.")
    p.add_argument("--cond_depth", type=int, default=-1,
                    help="StackDecoder (--decoder_type stack) only: how many levels above own "
                         "code each non-top level conditions on. -1 (default) = pervasive, every "
                         "level above. 1 = one level up only, the minimal own-code-plus-one-track shape.")
    p.add_argument("--grid_dq", type=int, default=None)
    p.add_argument("--grid_levels", type=int, default=None)
    p.add_argument("--grid_bound", type=str, default="sigmoid", choices=["sigmoid", "tanh"])
    p.add_argument("--grid_logistic_scale", type=float, default=0.5,
                    help="scale of the reparameterized logistic dequantization noise used when "
                         "quant_type=grid and code_sample=True")
    p.add_argument("--gmm_k", type=int, default=None, help="number of shared GMM codebook components")
    p.add_argument("--gmm_dq", type=int, default=None, help="GMM code dimensionality")
    p.add_argument("--gmm_bpb_precision_bits", type=int, default=8,
                    help="per-dim quantization-precision correction added to the achievable bpb bound "
                         "reported for code_hard=False/code_sample=True (differential NLL alone isn't bits)")
    p.add_argument("--quant_dropout_p0", type=float, default=0.0,
                    help="Quant Noise (Fan et al. 2020): probability of taking a plain continuous "
                         "identity/soft quantize pass instead of the configured hard/sample setting, "
                         "during training only (torch.is_grad_enabled()) -- linearly decayed to 0 over "
                         "--quant_dropout_decay_steps. 0.0 (default) is a no-op.")
    p.add_argument("--quant_dropout_decay_steps", type=int, default=None,
                    help="steps to decay --quant_dropout_p0 to 0 over (schedule=linear) or steps to hold "
                         "it fully on before an instant switch to 0 (schedule=step); defaults to --steps")
    p.add_argument("--quant_dropout_schedule", type=str, default="linear", choices=["linear", "step"],
                    help="linear: gradual Quant Noise-style decay (default). step: AE-Warm-Up-style hard "
                         "switch (Zhao et al. 2026) -- quant_dropout_p0=1.0 with schedule=step reproduces "
                         "their 'train as plain unquantized autoencoder for T_wu steps, then quantize' scheme")
    p.add_argument("--monitor_effective_dim", action="store_true",
                    help="print level0 code's effective dimension (participation ratio) each --eval_every "
                         "round, and flag when it plateaus -- diagnostic only, does not affect training. "
                         "Off by default: needs an extra forward pass + CPU eigendecomposition, heavier "
                         "than the rest of eval.")
    p.add_argument("--track_dropout_p0", type=float, default=0.0,
                    help="dense->sparse decode-track curriculum: target probability (reached by "
                         "--track_dropout_ramp_steps) of pruning every decode track except self + "
                         "topmost. Ramps UP from 0 (opposite direction from --quant_dropout_p0) -- "
                         "0.0 (default) is a no-op, always dense (today's behavior).")
    p.add_argument("--track_dropout_ramp_steps", type=int, default=None,
                    help="steps to ramp --track_dropout_p0 up from 0 over (schedule=linear) or "
                         "steps to hold fully dense before an instant switch to p0 (schedule=step); "
                         "defaults to --steps")
    p.add_argument("--track_dropout_schedule", type=str, default="linear", choices=["linear", "step"],
                    help="linear: gradual ramp (default). step: AE-Warm-Up-style hard switch -- "
                         "fully dense for track_dropout_ramp_steps, then instantly sparse")
    p.add_argument("--pq_chunks", type=int, default=1,
                    help="product-quantize simplex/gmm/gmm_diag codes: a pure multiplier on top of a "
                         "FIXED per-chunk codebook size (--vocab for simplex, --gmm_k for gmm/gmm_diag) "
                         "-- combinatorial capacity is (per-chunk size)^pq_chunks instead of a flat "
                         "single table, standard product-quantization convention. 1 (default) is the "
                         "original single-table behavior. No effect on grid/binary, which are already "
                         "fully per-dimension factorized.")
    p.add_argument("--use_self_code", type=lambda x: x.lower() != "false", default=False,
                    help="False (default): non-top decode levels use a trainable constant seed instead "
                         "of real self-code (breaks cross-block recurrence, see self_code_active); the "
                         "top level of a multi-level hierarchy always keeps real self-code regardless. "
                         "n_levels==1 with this False degenerates decode to an unconditioned LM (warns).")
    p.add_argument("--detach_ss_sample", action="store_true", default=False,
                    help="encoder_ste_p's self-sampled code (sample_next()) goes through the same STE "
                         "path as normal quantize() by default, so its decode loss backprops into the "
                         "level-above encoder; set this to fall back to the old fully-detached (no "
                         "gradient to that encoder) behavior.")
    p.add_argument("--encoder_ste_p", type=float, default=0.0,
                    help="probability (0=never, default; 1.0=every step) of resampling every non-top "
                         "level's code from the level-above's own NTP prediction (sample_next()) instead "
                         "of the ground-truth code. See --encoder_ste_skip_real for the two modes "
                         "(additive second pass vs. substituting the real pass, formerly a separate "
                         "scheduled_sampling_p flag, unified 2026-08-23).")
    p.add_argument("--encoder_ste_skip_real", action="store_true", default=False,
                    help="False (default, additive): the real-code decode pass always runs; when "
                         "encoder_ste_p also fires, a separate second decode pass with the self-sampled "
                         "code runs too, added unweighted on top. True (skip): the self-sampled code "
                         "replaces the real-code pass entirely this step (mutually exclusive) -- the "
                         "old scheduled_sampling_p behavior.")
    p.add_argument("--byte_consistency_p", type=float, default=0.0,
                    help="probability (0=never, default; 1.0=every step) of a second, whole-model "
                         "forward pass on level0's own argmax'd (detached) byte-level reconstruction, "
                         "fed back through the entire model self-supervised -- tests whole-model "
                         "idempotence under self-feeding, unlike encoder_ste_p's code-level-only swap.")
    p.add_argument("--uncertainty_weighting", action="store_true", default=False,
                    help="Kendall/Gal/Cipolla 2018 homoscedastic uncertainty weighting: learn one "
                         "log-variance per NTP task (each level's own encode loss, each level's decode "
                         "loss, and the bundled decode_stage_extra loss) instead of the fixed "
                         "byte_ntp_weight/code_ntp_weight/decode_ntp_weight scalars, so per-task scale "
                         "differences (e.g. wider-vocab levels naturally producing bigger raw losses) "
                         "self-balance via gradient descent rather than manual tuning. Logged each "
                         "log_every step as train_uncertainty_sigma_<task>.")
    p.add_argument("--active_srcs_mode", type=int, default=None,
                    help="max_srcs passed to model(...) for steps < active_srcs_until_step -- a phased "
                         "mode restricting which conditioning tracks/modules are active early in "
                         "training (default None: no phasing, always full conditioning). Scalar (e.g. "
                         "2, via CLI) broadcasts to every level's decode_level call; a per-level tuple "
                         "(config-file only, e.g. (2, 1, None) on a Ks=(2,2,1) model) is required to "
                         "genuinely drop ALL conditioning on a given coarser level -- a scalar cap "
                         "can't do this since a level's OWN nearest upper track always survives any "
                         "cap >=2, e.g. level1 (only 1 upper track: level2) still sees level2 under a "
                         "global max_srcs=2 meant to hide it from level0. NOTE: excluding the topmost "
                         "level specifically no longer needs this -- StackDecoder now hard-excludes it "
                         "unconditionally (2026-08-23, see its docstring); this flag remains for other "
                         "phased-conditioning ablations.")
    p.add_argument("--active_srcs_until_step", type=int, default=0,
                    help="step at which active_srcs_mode stops applying and training "
                         "switches to full max_srcs=None (default 0: mode never active).")
    p.add_argument("--kv_lm_mode", choices=["identity", "copy", "shared"], default="shared",
                    help="how upper-track cross-attention K/V is built from a coarser level's code "
                         "(StackDecoder only). 'shared' (default since 2026-08-23, was 'identity'): "
                         "a causal pass over the embedded code sequence reusing the producing level's "
                         "OWN encoder LM weights (encoders[j].lm) -- cheaper than 'copy', and level i's "
                         "cross_attn_stage LM already has its own dedicated submodule to consume the "
                         "code, so kv_lm just needs a good K/V representation, which the encoder that "
                         "already models this code well is the natural source for (see chat 2026-08-22 "
                         "for the gradient-interference concern this raises, still open). 'identity': "
                         "raw per-position code embedding, no interaction between code positions (the "
                         "old default). 'copy' (renamed from 'fresh' 2026-08-23): a new small LM "
                         "(kv_lm_layers, own independently-trained weights, NOT tied to the encoder) "
                         "causally self-attends over the embedded code sequence first, contextualizing "
                         "each position from earlier codes in the same track, before it's used as K/V. "
                         "'shared'/'copy' both require d_model to match across the levels involved.")
    p.add_argument("--kv_lm_layers", type=int, default=None,
                    help="n_layers for kv_lm_mode='copy's dedicated LM (default: same as this "
                         "level's own n_layers). Ignored for 'identity'/'shared'.")
    p.add_argument("--decode_scope", choices=["level0_only", "pervasive"], default="level0_only",
                    help="which levels' own decode_level runs (default since 2026-08-23: "
                         "level0_only -- only level0's decode is trained/computed, since it's the "
                         "only one anything downstream consumes; level i>0's own reconstruction of "
                         "its own domain trains nothing that isn't already trained via its "
                         "encode_loss + level0's cross-attention into c_list[i]. 'pervasive': every "
                         "level's own decode runs and contributes to decode_total, the original "
                         "(pre-2026-08-23) behavior.")
    p.add_argument("--dim_monitor_plateau_tol", type=float, default=0.01,
                    help="relative change in effective_dim below which two consecutive --eval_every "
                         "measurements are flagged as a plateau")

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--full_val_eval", action="store_true",
                    help="each --eval_every round runs eval_model_full over the whole val set instead of eval_model's sampled batches")
    p.add_argument("--eval_chunk_size", type=int, default=64,
                    help="per-forward-pass chunk size for eval_model_full -- -1 means no chunking (single "
                         "giant batch covering the whole selection at once, the old default, implicated in "
                         "a recurring MPS eval glitch on larger configs)")
    p.add_argument("--eval_sample", type=parse_eval_sample, default=1.0,
                    help="fraction (float, e.g. 0.1) or absolute count (int, e.g. 500) of val windows "
                         "eval_model_full evaluates -- default 1.0 iterates the whole val set")
    p.add_argument("--qual_gen_bytes", type=int, default=0)
    p.add_argument("--qual_prompt_bytes", type=int, default=64)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--compile", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_split", choices=["train", "val"], default="val")
    p.add_argument("--checkpoint_path", type=Path, default=None)

    _missing_from_cli = {f.name for f in dataclass_fields(Config)} - {a.dest for a in p._actions}
    assert not _missing_from_cli, f"Config field(s) {_missing_from_cli} have no matching --arg registered"

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.Ks = parse_int_tuple(args.Ks)
    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")
    if args.quant_type is None:
        p.error("--quant_type has no default -- set it explicitly (--quant_type or config file: simplex|binary|grid|gmm|gmm_diag)")
    if args.vocab is None:
        p.error("--vocab has no default -- set it explicitly (--vocab or config file)")
    if args.quant_type == "binary" and args.binary_bits is None:
        p.error("--binary_bits has no default -- set it explicitly when --quant_type binary")
    if args.quant_type == "grid" and args.grid_dq is None:
        p.error("--grid_dq has no default -- set it explicitly when --quant_type grid")
    if args.quant_type == "grid" and args.grid_levels is None:
        p.error("--grid_levels has no default -- set it explicitly when --quant_type grid")
    if args.quant_type in ("gmm", "gmm_diag") and args.gmm_k is None:
        p.error("--gmm_k has no default -- set it explicitly when --quant_type gmm|gmm_diag")
    if args.quant_type in ("gmm", "gmm_diag") and args.gmm_dq is None:
        p.error("--gmm_dq has no default -- set it explicitly when --quant_type gmm|gmm_diag")
    if args.quant_type in ("gmm", "gmm_diag") and args.gmm_dq is not None and args.gmm_dq % args.pq_chunks != 0:
        p.error(f"--gmm_dq ({args.gmm_dq}) must be divisible by --pq_chunks ({args.pq_chunks})")
    if args.quant_type == "simplex" and args.code_head_tied and args.pq_chunks > 1:
        p.error("--code_head_tied is not supported with --pq_chunks > 1 for quant_type simplex")
    if args.input_preset is None:
        p.error("--input_preset has no default -- set it explicitly (--input_preset or config file: 1|4|8)")
    if args.output_preset is None:
        p.error("--output_preset has no default -- set it explicitly (--output_preset or config file: 1|4|8)")
    return args, pre_args


def config_from_args(args) -> Config:
    return Config(
        decoder_type=args.decoder_type, Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers,
        context_len=args.context_len, n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window,
        rope_base=args.rope_base, byte_ntp_weight=args.byte_ntp_weight, code_ntp_weight=args.code_ntp_weight,
        decode_ntp_weight=args.decode_ntp_weight, gumbel_tau=args.gumbel_tau,
        code_extract_mode=args.code_extract_mode, code_head_tied=args.code_head_tied,
        vocab=args.vocab, quant_type=args.quant_type, binary_bits=args.binary_bits, binary_lfq=args.binary_lfq,
        input_preset=args.input_preset, output_preset=args.output_preset,
        entropy_reg_weight=args.entropy_reg_weight, ntp_head_tied=args.ntp_head_tied,
        byte_head_tied=args.byte_head_tied,
        decode_cross_stage_layers=args.decode_cross_stage_layers,
        decoder_own_stage_mode=args.decoder_own_stage_mode,
        cond_depth=args.cond_depth,
        grid_dq=args.grid_dq, grid_levels=args.grid_levels, grid_bound=args.grid_bound,
        grid_logistic_scale=args.grid_logistic_scale,
        gmm_k=args.gmm_k, gmm_dq=args.gmm_dq, gmm_bpb_precision_bits=args.gmm_bpb_precision_bits,
        quant_dropout_p0=args.quant_dropout_p0, quant_dropout_decay_steps=args.quant_dropout_decay_steps,
        quant_dropout_schedule=args.quant_dropout_schedule,
        code_hard=args.code_hard,
        code_sample=args.code_sample,
        pq_chunks=args.pq_chunks,
        track_dropout_p0=args.track_dropout_p0,
        track_dropout_ramp_steps=args.track_dropout_ramp_steps,
        track_dropout_schedule=args.track_dropout_schedule,
        use_self_code=args.use_self_code,
        detach_ss_sample=args.detach_ss_sample,
        encoder_ste_p=args.encoder_ste_p,
        encoder_ste_skip_real=args.encoder_ste_skip_real,
        byte_consistency_p=args.byte_consistency_p,
        uncertainty_weighting=args.uncertainty_weighting,
        active_srcs_mode=args.active_srcs_mode,
        active_srcs_until_step=args.active_srcs_until_step,
        kv_lm_mode=args.kv_lm_mode,
        kv_lm_layers=args.kv_lm_layers,
        decode_scope=args.decode_scope,
    )


def run_main(QCuteLM) -> None:
    args, pre_args = build_argparser("qcute_v1: shared Encoder, pluggable concat/stack Decoder")
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config_from_args(args)
    model = QCuteLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_v1_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"decoder_type={cfg.decoder_type} Ks={cfg.Ks} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, cfg.input_preset, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_words={len(train_data)}  val_words={len(val_data)}  input_preset={cfg.input_preset}bit "
        f"output_preset={cfg.output_preset}bit level0_vocab={2 ** cfg.input_preset}")

    if args.eval_only:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        eval_data = train_data if args.eval_split == "train" else val_data
        result = eval_model_full(model, eval_data, args.eval_chunk_size, device, args.eval_sample)
        result_str = "  ".join(f"{args.eval_split}_{k}={v:.4f}" for k, v in result.items())
        log(f"eval_only_full_{args.eval_split}set  {result_str}",
            **{f"{args.eval_split}_{k}": v for k, v in result.items()})
        return

    train(model, train_data, val_data, args, log, run_name, device)
