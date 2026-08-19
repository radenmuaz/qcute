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
    code_hard: bool = True
    code_sample: bool = False
    gmm_bpb_precision_bits: int = 8
    decode_cross_stage_layers: int | None = None
    share_encode_decode_self: bool = False


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


def fsq_quantize(v: torch.Tensor, L: int, hard: bool = True, sample: bool = False, bound: str = "sigmoid") -> torch.Tensor:
    half_l = (L - 1) / 2
    z_bounded = half_l * (torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1))
    if not hard:
        return z_bounded / half_l
    if sample:
        noise = torch.rand_like(z_bounded) - 0.5
        z_rounded = torch.round(z_bounded + noise).clamp(-half_l, half_l)
    else:
        z_rounded = torch.round(z_bounded)
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
    def __init__(self, tau: float, hard: bool = True, sample: bool = False, ntp_head_tied: bool = False):
        self.tau = tau
        self.hard = hard
        self.sample = sample
        self.ntp_head_tied = ntp_head_tied

    def init_modules(self, D, V, code_head_tied):
        code_head = None if code_head_tied else nn.Linear(D, V, bias=False)
        if code_head is not None:
            nn.init.normal_(code_head.weight, std=0.02)
        ntp_head = None if self.ntp_head_tied else nn.Linear(D, V, bias=False)
        if ntp_head is not None:
            nn.init.normal_(ntp_head.weight, std=0.02)
        # code_embed: dedicated categorical-code embedding table (V=cfg.vocab, matching code_head's
        # output width), separate from stage_lm.embed (that backbone's own INPUT alphabet table --
        # only equal to V when this level's input word width matches cfg.vocab, e.g. level0 under a
        # non-byte input_preset diverges). BSQ/FSQ already carry their own dedicated code_embed
        # module for the same reason; softmax previously reused stage_lm.embed.weight as a shortcut,
        # safe only when every level shared one global vocab.
        code_embed = nn.Linear(V, D, bias=False)
        nn.init.normal_(code_embed.weight, std=0.02)
        return code_head, code_embed, ntp_head

    def quantize(self, pre_q):
        return gumbel_quantize(pre_q, self.tau, self.hard, self.sample)

    def to_ids(self, source_c):
        return source_c.argmax(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def _ntp_logits(self, stage_lm, h_query):
        if self.ntp_head_tied:
            return F.linear(h_query, stage_lm.embed.weight)
        return stage_lm.code_predict(h_query)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target = target_repr.argmax(-1).reshape(-1)
        logits = self._ntp_logits(stage_lm, h_query)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        logits = self._ntp_logits(stage_lm, h_query)
        next_id = logits.argmax(-1)
        return F.one_hot(next_id, num_classes=vocab).to(h_query.dtype)

    def entropy_reg(self, pre_q):
        return softmax_entropy_reg(pre_q)


class BinaryQuant(QuantScheme):
    def __init__(self, binary_bits: int, hard: bool = True, sample: bool = False, lfq: bool = False):
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
        return bsq_quantize(pre_q, self.binary_bits, self.hard, self.sample, self.lfq)

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
        return bsq_quantize(pred, self.binary_bits, self.hard, self.sample, self.lfq)

    def entropy_reg(self, pre_q):
        return bsq_entropy_reg(pre_q)


class GridQuant(QuantScheme):
    def __init__(self, dq: int, L: int, hard: bool = True, sample: bool = False, bound: str = "sigmoid"):
        self.dq, self.L, self.hard, self.sample, self.bound = dq, L, hard, sample, bound

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.dq, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = FSQEmbed(self.dq, self.L, D)
        code_predict = nn.Linear(D, self.dq * self.L)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        return fsq_quantize(pre_q, self.L, self.hard, self.sample, self.bound)

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
        bound = (self.L - 1) / 2
        pred = stage_lm.code_predict(h_query).reshape(*h_query.shape[:-1], self.dq, self.L)
        if self.sample:
            probs = F.softmax(pred, dim=-1)
            levels = torch.multinomial(probs.reshape(-1, self.L), 1).reshape(pred.shape[:-1])
        else:
            levels = pred.argmax(-1)
        return (levels.float() - bound) / bound


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
    """Shared full-covariance GMM codebook: K components over a dq-dim code space.
    Covariance parameterized via its precision Cholesky factor A (Lambda = A A^T) so the
    NLL's Mahalanobis term is a plain matmul (y = A^T(x-mu)) -- no solve/inverse needed.
    Sampling (z = mu + A^-T eps) needs one small manual triangular solve instead."""

    def __init__(self, K: int, dq: int, D: int):
        super().__init__()
        self.K, self.dq = K, dq
        self.mu = nn.Parameter(torch.empty(K, dq).normal_(std=1.0))
        self.chol_raw = nn.Parameter(torch.empty(K, dq, dq).normal_(std=0.1))
        self.proj = nn.Linear(dq, D, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.register_buffer("_tril_off", torch.tril(torch.ones(dq, dq), diagonal=-1), persistent=False)
        self.register_buffer("_eye", torch.eye(dq), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def precision_chol(self) -> torch.Tensor:
        diag = F.softplus(self.chol_raw.diagonal(dim1=-2, dim2=-1)) + 1e-4
        return self.chol_raw * self._tril_off + diag.unsqueeze(-1) * self._eye

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        A = self.precision_chol()
        diff = x.unsqueeze(-2) - self.mu
        y = torch.einsum("...kd,kde->...ke", diff, A)
        maha = (y * y).sum(-1)
        log_diag_sum = A.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).log().sum(-1)
        return log_diag_sum - 0.5 * (self.dq * math.log(2 * math.pi) + maha)

    def sample(self, k_idx: torch.Tensor) -> torch.Tensor:
        A = self.precision_chol()[k_idx]
        eps = torch.randn(*k_idx.shape, self.dq, device=self.mu.device, dtype=self.mu.dtype)
        y = solve_upper_triangular(A.transpose(-1, -2), eps)
        return self.mu[k_idx] + y

    def sample_all(self, batch_shape: tuple) -> torch.Tensor:
        """Reparam-sample every component at once -- one precision_chol() call and one
        batched triangular solve (K,dq,dq broadcasting against batch_shape+(K,dq)),
        instead of calling sample() K times (each redundantly recomputing precision_chol()
        over all K components -- O(K^2) work for what should be O(K))."""
        A = self.precision_chol()
        eps = torch.randn(*batch_shape, self.K, self.dq, device=self.mu.device, dtype=self.mu.dtype)
        y = solve_upper_triangular(A.transpose(-1, -2), eps)
        return self.mu + y


class GMMDiagCodebook(nn.Module):
    """Shared diagonal-covariance GMM codebook -- same interface as GMMCodebook, plain
    elementwise NLL/reparam (no triangular solve needed either direction)."""

    def __init__(self, K: int, dq: int, D: int):
        super().__init__()
        self.K, self.dq = K, dq
        self.mu = nn.Parameter(torch.empty(K, dq).normal_(std=1.0))
        self.logvar = nn.Parameter(torch.zeros(K, dq))
        self.proj = nn.Linear(dq, D, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.mu
        maha = (diff * diff * (-self.logvar).exp()).sum(-1)
        return -0.5 * (self.dq * math.log(2 * math.pi) + self.logvar.sum(-1) + maha)

    def sample(self, k_idx: torch.Tensor) -> torch.Tensor:
        std = (0.5 * self.logvar[k_idx]).exp()
        return self.mu[k_idx] + std * torch.randn_like(std)

    def sample_all(self, batch_shape: tuple) -> torch.Tensor:
        std = (0.5 * self.logvar).exp()
        eps = torch.randn(*batch_shape, self.K, self.dq, device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + std * eps


class GMMHead(nn.Module):
    def __init__(self, D: int, dq: int, K: int):
        super().__init__()
        self.dq = dq
        self.proj = nn.Linear(D, dq + K, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> tuple:
        out = self.proj(h)
        return out[..., :self.dq], out[..., self.dq:]


class _GMMQuantBase(QuantScheme):
    codebook_cls = None

    def __init__(self, K: int, dq: int, hard: bool = True, sample: bool = False):
        self.K, self.dq = K, dq
        self.hard = hard
        self.sample = sample
        self._codebook = None

    def init_modules(self, D, V, code_head_tied):
        code_head = GMMHead(D, self.dq, self.K)
        code_predict = GMMHead(D, self.dq, self.K)
        self._codebook = self.codebook_cls(self.K, self.dq, D)
        return code_head, self._codebook, code_predict

    def _posterior_logits(self, query, gating_logits):
        return F.log_softmax(gating_logits, dim=-1) + self._codebook.log_prob(query)

    def _select(self, query, gating_logits):
        cb = self._codebook
        log_post = self._posterior_logits(query, gating_logits)
        r = F.softmax(log_post, dim=-1)
        soft = torch.einsum("...k,kd->...d", r, cb.mu)
        if not self.hard and not self.sample:
            return soft
        if self.sample:
            eps = torch.finfo(log_post.dtype).tiny
            u = torch.rand_like(log_post).clamp(min=eps, max=1.0 - eps)
            weights = F.softmax(log_post - torch.log(-torch.log(u)), dim=-1)
        else:
            weights = r
        if not self.hard:
            samples = cb.sample_all(query.shape[:-1])
            return (weights.unsqueeze(-1) * samples).sum(-2)
        k_star = weights.argmax(-1)
        hard_code = cb.sample(k_star) if self.sample else cb.mu[k_star]
        return soft + (hard_code - soft).detach()

    def quantize(self, pre_q):
        query, gating_logits = pre_q
        return self._select(query, gating_logits)

    def to_ids(self, source_c):
        dists = ((source_c.unsqueeze(-2) - self._codebook.mu) ** 2).sum(-1)
        return dists.argmin(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        logits = self._posterior_logits(query_pred, gating_pred).reshape(-1, self.K)
        target_id = self.to_ids(target_repr.reshape(-1, self.dq))
        loss = F.cross_entropy(logits, target_id)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target_id).float().mean()
        return loss, acc

    def sample_next(self, stage_lm, h_query, vocab):
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        return self._select(query_pred, gating_pred)

    def bpb_bound(self, stage_lm, h_query, target_repr, precision_bits: int) -> torch.Tensor:
        """Achievable bpb bound for code_hard=False / code_sample=True, where ntp_loss_acc's
        K-way cross-entropy against to_ids() undercounts (it only charges for "which of K
        components", discarding the continuous residual those modes actually emit). Uses the
        true mixture density's NLL of the exact target_repr (nats -> bits) plus a stated
        per-dim quantization-precision correction (differential entropy alone isn't bits without
        one, same reason RealNVP/Glow-style bits/dim reporting adds a fixed dequantization
        constant) -- an honest, achievable upper bound, not exact."""
        query_pred, gating_pred = stage_lm.code_predict(h_query)
        cb = self._codebook
        log_pi = F.log_softmax(gating_pred.reshape(-1, self.K), dim=-1)
        log_lik = cb.log_prob(target_repr.reshape(-1, self.dq))
        nll_nats = -torch.logsumexp(log_pi + log_lik, dim=-1).mean()
        return nll_nats / math.log(2) + self.dq * precision_bits

    def entropy_reg(self, pre_q):
        query, gating_logits = pre_q
        return softmax_entropy_reg(self._posterior_logits(query, gating_logits))


class GMMQuant(_GMMQuantBase):
    codebook_cls = GMMCodebook


class GMMDiagQuant(_GMMQuantBase):
    codebook_cls = GMMDiagCodebook


def make_quant(cfg: Config) -> QuantScheme:
    if cfg.quant_type == "binary":
        return BinaryQuant(cfg.binary_bits, cfg.code_hard, cfg.code_sample, cfg.binary_lfq)
    if cfg.quant_type == "grid":
        return GridQuant(cfg.grid_dq, cfg.grid_levels, cfg.code_hard, cfg.code_sample, cfg.grid_bound)
    if cfg.quant_type == "gmm":
        return GMMQuant(cfg.gmm_k, cfg.gmm_dq, cfg.code_hard, cfg.code_sample)
    if cfg.quant_type == "gmm_diag":
        return GMMDiagQuant(cfg.gmm_k, cfg.gmm_dq, cfg.code_hard, cfg.code_sample)
    return SimplexQuant(cfg.gumbel_tau, cfg.code_hard, cfg.code_sample, cfg.ntp_head_tied)


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

    def ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor, is_byte_level: bool) -> tuple:
        if is_byte_level:
            target = target_repr.reshape(-1)
            logits = F.linear(h_query, self.embed.weight)
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


@torch.no_grad()
def eval_model_full(model, data: torch.Tensor, batch_size: int, device: str) -> dict:
    model.eval()
    context_len = model.cfg.context_len
    n_windows = len(data) // context_len
    batch_size = n_windows if batch_size == -1 else batch_size
    accum: dict = {}
    total_n = 0
    for start in range(0, n_windows, batch_size):
        idxs = range(start, min(start + batch_size, n_windows))
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
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
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
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
                          byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%")
        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            if args.full_val_eval:
                val = eval_model_full(model, val_data, -1, device)
            else:
                val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.qual_gen_bytes > 0:
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    model.decoder.qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                                         window[args.qual_prompt_bytes:], device, log=log, label=label)
                    model.decoder.check_gen_consistency(model, window, device, prompt_len=args.qual_prompt_bytes,
                                                          log=log, label=label)


def build_argparser(description: str) -> tuple:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=description, parents=[pre])
    p.add_argument("--decoder_type", type=str, default="concat", choices=["concat", "stack"])
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
    p.add_argument("--decode_cross_stage_layers", type=int, default=None)
    p.add_argument("--share_encode_decode_self", action="store_true")
    p.add_argument("--grid_dq", type=int, default=None)
    p.add_argument("--grid_levels", type=int, default=None)
    p.add_argument("--grid_bound", type=str, default="sigmoid", choices=["sigmoid", "tanh"])
    p.add_argument("--gmm_k", type=int, default=None, help="number of shared GMM codebook components")
    p.add_argument("--gmm_dq", type=int, default=None, help="GMM code dimensionality")
    p.add_argument("--gmm_bpb_precision_bits", type=int, default=8,
                    help="per-dim quantization-precision correction added to the achievable bpb bound "
                         "reported for code_hard=False/code_sample=True (differential NLL alone isn't bits)")

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
                    help="each --eval_every round runs eval_model_full over the whole val set (batch_size=-1) instead of eval_model's sampled batches")
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
        decode_cross_stage_layers=args.decode_cross_stage_layers,
        share_encode_decode_self=args.share_encode_decode_self,
        grid_dq=args.grid_dq, grid_levels=args.grid_levels, grid_bound=args.grid_bound,
        gmm_k=args.gmm_k, gmm_dq=args.gmm_dq, gmm_bpb_precision_bits=args.gmm_bpb_precision_bits,
        code_hard=args.code_hard,
        code_sample=args.code_sample,
    )


def run_main(QCuteLM) -> None:
    args, pre_args = build_argparser("qcute_v5: shared Encoder, pluggable concat/stack Decoder")
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config_from_args(args)
    model = QCuteLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_v5_{int(time.time())}")
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
        result = eval_model_full(model, eval_data, args.batch_size, device)
        result_str = "  ".join(f"{args.eval_split}_{k}={v:.4f}" for k, v in result.items())
        log(f"eval_only_full_{args.eval_split}set  {result_str}",
            **{f"{args.eval_split}_{k}": v for k, v in result.items()})
        return

    train(model, train_data, val_data, args, log, run_name, device)
