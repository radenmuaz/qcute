"""qcute.bytelm_tpu — single-file fork of qcute.bytelm for CPU/TPU (torch_xla) use.

Standalone on purpose (not an import of qcute.bytelm) so it can be read/debugged/deployed to a
TPU VM without dragging in the rest of the repo's device-selection assumptions.

**Optional flash-attention kernel (`--use_flash_attention`), default off.**
`torch_xla.experimental.custom_kernel.flash_attention` (a JAX Pallas TPU kernel under
torch_xla's hood) needs `jax` installed *and* `libtpu>=0.0.44` — the pinned stable
`torch_xla==2.9.0` install (see docs/bytelm_tpu_setup.md) locks `libtpu==0.0.21`, and bumping
libtpu alone against that pin is a confirmed hard break (`RuntimeError: Unexpected
PJRT_ExecuteOptions size: expected 112, got 80` — the plugin/framework PJRT API versions
disagree). What does work, confirmed directly on a v4-8 node: `torch==2.10.0.dev0` +
`torch_xla==2.10.0.dev0` (from the GCS nightly wheel URLs — `pip install
UV_SKIP_WHEEL_FILENAME_CHECK=1 ... https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch{,_xla}-2.10.0.dev-cp312-cp312-linux_x86_64.whl`,
the "no date suffix" rolling-latest nightly), `libtpu` and `jax`/`jaxlib` left unpinned (resolve
to 0.0.46 automatically), plus the system package `libopenblas0` (nightly torch's wheel needs it,
`sudo apt-get install -y libopenblas0` — the stable release doesn't need this). Flash-attention
output matches plain SDPA closely (max abs diff ~0.011, mean ~0.00016 against a ~0.04-magnitude
reference) once that's all in place. **This is a nightly-only feature** — `--use_flash_attention`
on the stable pin will just raise ImportError-guarded fallback to plain SDPA (`_HAS_FLASH_ATTENTION`
stays False), not crash. Only used in the full-sequence training `forward` path — `forward_step`
(single-query KV-cache decode) stays on plain SDPA regardless, since flash_attention's kernel
assumes a self-attention causal pattern (q_len == kv_len), not a growing single-query cache.

**Optional multi-chip data-parallel training (`--multichip`), default off.** Spawns one process
per locally-addressable TPU device via `torch_xla.launch` (confirmed on the same v4-8 node:
`torch_xla.runtime.addressable_runtime_device_count()` reports 4 — a "v4-8" slice has 8
TensorCores across 4 chips, but each v4 chip's 2 cores run combined as one "megacore" logical
device by default, hence 4 addressable devices, not 8). Gradient sync uses
`xm.optimizer_step(opt, barrier=True)` (calls `reduce_gradients` — an all-reduce average across
replicas — then `opt.step()`, then the execution barrier) in place of a bare `opt.step()`; this is
the same call whether running single- or multi-process, since reducing over a replica group of 1
is a no-op. Only the rank-0 process does file I/O (Logger, Checkpointer, qualitative
generation/benchmarking) — every rank still runs the forward/backward/optimizer_step loop since
gradient all-reduce is a collective operation all replicas must participate in. `--batch_size` is
per-process (local batch); effective global batch is `batch_size * world_size`, logged as such
when `--multichip` is on. Each process's `batch_iter` draws independently-random batches (spawned
processes get independent RNG state), so this is genuine data parallelism, not repeated work.

Differences from qcute.bytelm:

  - Device is cpu or xla only — no mps/cuda. `--device {cpu,xla}` (default: auto-detect, xla if
    torch_xla+TPU is importable/visible, else cpu).
  - Training step uses `torch_xla.core.xla_model.optimizer_step` (opt.step() + the XLA graph
    execution barrier) on xla; eval loops call `xm.mark_step()` after every batch so the lazy
    XLA graph doesn't grow unbounded across a no-grad loop. All hot-path tensor shapes (batch,
    context, mtp_heads) are fixed for the whole run, so this should not recompile per step.
  - Checkpointer.is_better guards against a non-finite or non-positive metric (copied from
    qcute_v1_common.py's version — bf16 TPU training can occasionally spike to nan/inf; a bad
    checkpoint from that is worse than skipping a save).
  - CausalSelfAttention.zero_kv_sink option (default ON): prepends one all-zero key/value
    ("sink") token before every real token, visible to every query unconditionally, before SDPA.
    A static (non-learned) attention sink, sometimes used to relieve softmax's requirement that
    attention weights sum to 1 even when no real key deserves much mass. Disable via
    `LMConfig(zero_kv_sink=False)` / `--no_zero_kv_sink`. **Compatible with use_flash_attention
    only when `(context+1) % 1024 == 0`** (e.g. context=8191, not 8192) — the Pallas kernel's
    causal=True needs q_len==kv_len exactly (satisfied by padding q with one dummy leading row
    to match the sink-prepended kv_len=T+1, verified: max abs diff ~0.007-0.01 vs. the
    explicit-mask reference, same as flash-attention's normal variance) *and* that common length
    divisible by its internal block size (1024, observed). A context that doesn't satisfy this
    falls back to the explicit-mask SDPA path (O(T^2) memory) instead, with a startup warning —
    still correct, just not the fast path.
  - RMSNorm (not LayerNorm) and SwiGLU MLP (gate+up+down, not a single-branch silu MLP) —
    LLaMA-style stack, all Linear layers already bias-free. Adopted 2026-08-22 to converge
    faster as a baseline, independent of whether any enwik8-bpb literature result used them.

Otherwise the model (plain pre-norm transformer, RoPE, MTP heads), training loop, and generation
code (plain AR / KV-cache / self-speculative) are the same as qcute.bytelm — see that module's
own docstring for the full design rationale (handover doc §5/§1.6 baseline framing).

## sd preset (8 layers, ~101M params) full-enwik8 TPU run

`PRESETS["sd"]` (d_model=1024, n_layers=8, n_heads=16, context=2048, mtp_heads=8) is the target
config for this module: ~101M non-embedding params, aimed at sub-1.0 bpb on full enwik8
(datasets/enwik8.gz, 100,000,000 bytes) within a 12h budget on a single TPU chip.

FLOPs-vs-data budget check (see qcute/bytelm.py's own docstring for the underlying FLOPs grid):
6*N_params*tokens (standard fwd+bwd approximation) means a single v6e-1 chip at a conservative
~40% MFU of its ~918 TFLOPS/s bf16 peak processes on the order of 10^10 tokens in 12h — one to
two orders of magnitude more than the ~2*10^9 tokens (~20 epochs over enwik8's ~95M-byte train
split) that compute-optimal scaling (~20 tokens/param) would call for. In other words: for this
model size, the full-enwik8 corpus (not compute) is almost certainly the binding constraint —
convergence to sub-1.0 bpb should be a training-stability/epochs question, not a raw-FLOPs one.
That said, this estimate is a priori (no real TPU torch_xla throughput measured yet for this
module) — watch actual it/s on the very first run and retune `--steps` from there, per this
repo's standing "long runs have shown unpredictable throughput" caution (CLAUDE.md).

If a single TPU chip can't reach sub-1.0 bpb in 12h in practice, the natural next step is a
4-chip pod — but this module is single-process/single-device only (no torch_xla
SPMD/multiprocessing data-parallel wiring); that would be new infra, not implemented here.

    uv run python -m qcute.bytelm_tpu --config configs/bytelm/bytelm_tpu_sd_full_enwik8.py
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    _HAS_XLA = True
except ImportError:
    torch_xla = None
    xm = None
    xr = None
    _HAS_XLA = False

try:
    from torch_xla.experimental.custom_kernel import flash_attention as _xla_flash_attention
    _HAS_FLASH_ATTENTION = True
except ImportError:
    _xla_flash_attention = None
    _HAS_FLASH_ATTENTION = False


def is_xla_device(device) -> bool:
    return getattr(device, "type", None) == "xla"


def resolve_device(name: str | None) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "xla":
        if not _HAS_XLA:
            raise RuntimeError("--device xla requested but torch_xla is not importable")
        return torch_xla.device()
    if not _HAS_XLA:
        return torch.device("cpu")
    try:
        return torch_xla.device()
    except Exception:
        return torch.device("cpu")


def autocast_ctx(device: torch.device):
    # bf16 autocast on xla, no-op on cpu — halves-plus activation memory vs. plain fp32 (the
    # default otherwise), which is what actually gates batch_size/context on a fixed-HBM chip,
    # not attention memory (confirmed: flash-attention alone left batch_size=4 OOMing at
    # context=4096 on a 67M-param model / 32GB v6e-1 chip, all in fp32).
    if is_xla_device(device):
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def mark_step(device: torch.device) -> None:
    # Lazy XLA ops otherwise queue up without executing — required at least once per training
    # step and once per eval batch, or the unexecuted graph grows unboundedly and per-step wall
    # time climbs without bound (confirmed directly: omitting this made every step slower than
    # the last).
    if is_xla_device(device):
        torch_xla.sync()


def optimizer_step(opt: torch.optim.Optimizer, device: torch.device) -> None:
    if is_xla_device(device):
        # reduce_gradients (all-reduce average across replicas) + opt.step() + the execution
        # barrier — the same call whether running single- or multi-process (--multichip): an
        # all-reduce over a replica group of 1 is a no-op, so this doesn't need branching.
        xm.optimizer_step(opt, barrier=True)
    else:
        opt.step()


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    """logs/<run_name>/run.log (human `tail -f`-able text) + run.jsonl (structured, for
    scripts/plot_run.py). Only writes at the log_every/eval_every cadence — tqdm's own \\r-redraw
    progress bar never touches either file."""

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
    """checkpoints/<run_name>/{best,last}.pt. is_better rejects a non-finite or non-positive
    metric (bf16 TPU training can spike to nan/inf; a checkpoint saved on that spike is worse
    than just skipping the save) — copied from qcute_v1_common.py's Checkpointer."""

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


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


@dataclass
class LMConfig:
    vocab: int = 256
    d_model: int = 1024
    n_layers: int = 8
    n_heads: int = 16
    context: int = 2048
    mlp_mult: int = 4
    rope_base: float = 10000.0
    mtp_heads: int = 8  # n parallel next-byte heads (bandwidth-matched to qcute.qcutelm's K)
    zero_kv_sink: bool = True  # prepend one all-zero K/V token, always attendable, before SDPA
    use_flash_attention: bool = False  # nightly-only (see module docstring); ignored if unavailable

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS: dict[str, LMConfig] = {
    # ~(4 + 2*mlp_mult) * d_model^2 * n_layers non-embedding params (vocab=256 is negligible)
    "tiny": LMConfig(d_model=128, n_layers=4, n_heads=4, context=512, mtp_heads=4),  # ~0.9M — pipeline sanity check
    "xs": LMConfig(d_model=256, n_layers=4, n_heads=4, context=256, mtp_heads=4),  # ~3.7M, quick local/CPU runs
    "sm": LMConfig(d_model=256, n_layers=8, n_heads=4, mlp_mult=2, context=1024, mtp_heads=4),  # ~4.3M — narrow/deep, not wide
    "fast": LMConfig(d_model=512, n_layers=8, n_heads=8, mlp_mult=2, context=1024, mtp_heads=4),  # ~16.9M — narrow/deep, not wide
    "d512x16": LMConfig(d_model=512, n_layers=16, n_heads=8, mlp_mult=4, context=4096, mtp_heads=1),  # ~67M, SwiGLU — v6e-1 single-chip saturation target
    "sd": LMConfig(d_model=1024, n_layers=8, n_heads=16, context=2048),   # ~101M — the full-enwik8 TPU target
    "md": LMConfig(d_model=2048, n_layers=8, n_heads=16, context=2048),   # ~403M
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [T, head_dim]
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
    # x: [B, H, T, head_dim], cos/sin: [T, head_dim]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Reused (not reallocated per forward call) for zero_kv_sink's dummy sink K/V and dummy
        # leading Q row — a single [1,1,1,hd] buffer, cast + expand()ed (view, no copy) to the
        # actual [B,H,1,hd] shape needed each call, instead of a fresh torch.zeros(...) every time.
        self.register_buffer("_zero_sink", torch.zeros(1, 1, 1, cfg.head_dim), persistent=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)  # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        use_flash = self.cfg.use_flash_attention and _HAS_FLASH_ATTENTION and is_xla_device(x.device)
        if self.cfg.zero_kv_sink:
            zero_kv = self._zero_sink.to(dtype=k.dtype).expand(B, H, 1, hd)
            k = torch.cat([zero_kv, k], dim=2)
            v = torch.cat([zero_kv, v], dim=2)
            # Only take the flash path if (T+1) is block-aligned (see below) -- otherwise the
            # Pallas kernel raises its own ValueError rather than silently miscomputing anything,
            # but crashing mid-run is still worse than a graceful fallback to the mask path, which
            # is always correct regardless of T. (Bug fixed 2026-08-23: this check used to live
            # only in main()'s startup warning, which printed but never actually forced the
            # fallback -- confirmed crashing with context=8192 before this fix.)
            if use_flash and (T + 1) % 1024 != 0:
                use_flash = False
            if use_flash:
                # Padding q with one dummy leading row makes q_len==kv_len==T+1 exactly, which
                # is required for the Pallas kernel's causal=True to compute the right diagonal
                # (verified: max abs diff ~0.007-0.01 vs. the explicit-mask reference below, same
                # as flash-attention's normal algorithmic variance). But the kernel ALSO requires
                # that common length divisible by its internal block size (1024, observed) — so
                # this only works when T+1 is itself a multiple of 1024 (e.g. context=8191, not
                # 8192). Caller's responsibility to pick a compatible context; a bad choice fails
                # loudly with the kernel's own "q_seq_len should be divisible by block_q_dq"
                # ValueError rather than silently computing something wrong.
                zero_q = self._zero_sink.to(dtype=q.dtype).expand(B, H, 1, hd)
                q_padded = torch.cat([zero_q, q], dim=2)
                y = _xla_flash_attention(q_padded, k, v, causal=True, sm_scale=1.0 / (hd ** 0.5))[:, :, 1:, :]
            else:
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                sink_col = torch.ones(T, 1, dtype=torch.bool, device=x.device)
                attn_mask = torch.cat([sink_col, causal], dim=1)  # [T, T+1] — sink always attendable
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif use_flash:
            y = _xla_flash_attention(q, k, v, causal=True, sm_scale=1.0 / (hd ** 0.5))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # [B, H, T, hd]
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        """Single-new-position forward, growing an explicit K/V cache. x_new: [B, 1, D]. Returns
        (y [B, 1, D], new_cache_k, new_cache_v). With zero_kv_sink, the sink is prepended once
        into an empty cache (position 0) and simply persists as the first cached K/V from then
        on — no extra masking needed since forward_step already attends over the full cache."""
        B, _, D = x_new.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        if self.cfg.zero_kv_sink and cache_k is None:
            cache_k = torch.zeros(B, H, 1, hd, device=x_new.device, dtype=k.dtype)
            cache_v = torch.zeros(B, H, 1, hd, device=x_new.device, dtype=v.dtype)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        y = F.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)   # single query, full past KV — no mask needed
        y = y.transpose(1, 2).reshape(B, 1, D)
        return self.out(y), new_k, new_v


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class MLP(nn.Module):
    """SwiGLU (LLaMA-style): gate and up are separate projections, unlike a plain Swish MLP —
    hidden width (cfg.mlp_mult * d_model) is unchanged, so this has ~50% more MLP params than a
    single-branch silu MLP at the same mlp_mult."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

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


class ByteLM(nn.Module):
    """MTP baseline (handover §1.6): n parallel softmax heads predict bytes t+1..t+n from the
    same trunk hidden state, bandwidth-matched to qcute.qcutelm's K. Head 0 (immediate next-byte)
    is weight-tied to the input embedding as usual; the other n-1 heads are untied (standard
    MTP)."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab, bias=False) for _ in range(cfg.mtp_heads)]
        )
        self.heads[0].weight = self.tok_emb.weight  # weight tying, head 0 only
        self.apply(self._init_weights)
        # GPT-2-style residual scaling: keeps activation growth in check with depth
        for block in self.blocks:
            for proj in (block.attn.out, block.mlp.down):
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T] long -> logits [n_heads, B, T, vocab]
        B, T = tokens.shape
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, tokens.device)
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        return torch.stack([head(x) for head in self.heads], dim=0)


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return nats / math.log(2)


def mtp_loss(logits: torch.Tensor, tokens: torch.Tensor, context: int):
    """logits: [n_heads, B, context, vocab] from model(tokens[:, :context]).
    tokens: [B, context + n_heads]. Returns (mean loss over all heads, head-0 bpb — the standard
    next-byte metric comparable to qcute.bytelm/qcute.qcutelm)."""
    n_heads = logits.size(0)
    losses = []
    for i in range(n_heads):
        targets_i = tokens[:, i + 1 : i + 1 + context]
        losses.append(F.cross_entropy(logits[i].reshape(-1, logits.size(-1)), targets_i.reshape(-1)))
    losses = torch.stack(losses)
    head0_bpb = losses[0] / math.log(2)
    return losses.mean(), head0_bpb


def batch_iter(data: torch.Tensor, batch_size: int, context: int, n_heads: int, device: torch.device):
    seq_len = context + n_heads  # n_heads bytes of lookahead beyond the context window
    n = (len(data) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.to(device)


def split_train_val_test(
    data: torch.Tensor, val_frac: float, test_frac: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chronological split: test (if test_frac>0) is the trailing test_frac of bytes, val is the
    val_frac before that, train is everything before both — so val/test never overlap regardless
    of which fractions are requested. test is None (not just empty) when test_frac<=0, so callers
    can tell "no test split configured" apart from "test split came out empty"."""
    n_val = max(1, int(len(data) * val_frac))
    if test_frac <= 0:
        return data[:-n_val], data[-n_val:], None
    n_test = max(1, int(len(data) * test_frac))
    return data[: -(n_val + n_test)], data[-(n_val + n_test) : -n_test], data[-n_test:]


@torch.no_grad()
def eval_bpb(model: nn.Module, data_iter, context: int, n_batches: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch = next(data_iter)
        with autocast_ctx(device):
            logits = model(batch[:, :context])
        _, head0_bpb = mtp_loss(logits, batch, context)
        total += head0_bpb.item()
        mark_step(device)
    model.train()
    return total / n_batches


@torch.no_grad()
def eval_bpb_full(model: nn.Module, data: torch.Tensor, batch_size: int, context: int, n_heads: int,
                   device: torch.device, desc: str | None = None) -> float:
    """Deterministic full-val-set pass: non-overlapping seq_len windows, walked in fixed
    chronological order starting at byte 0 (never random), each byte scored exactly once.
    `desc`, if given, shows a live tqdm bar over the eval batches (e.g. "val_full"/"test_full") —
    off by default since a full pass is normally a few seconds to tens of seconds, but useful to
    see it's actually progressing (not stuck) on a slow/large eval."""
    model.eval()
    seq_len = context + n_heads
    n_windows = (len(data) - 1) // seq_len
    batch_size = n_windows if batch_size == -1 else batch_size
    total, total_n = 0.0, 0
    starts = range(0, n_windows, batch_size)
    if desc is not None:
        starts = tqdm(starts, desc=desc, leave=False, dynamic_ncols=True)
    for start in starts:
        idxs = range(start, min(start + batch_size, n_windows))
        batch = torch.stack([data[i * seq_len:(i + 1) * seq_len] for i in idxs]).to(device)
        with autocast_ctx(device):
            logits = model(batch[:, :context])
        _, head0_bpb = mtp_loss(logits, batch, context)
        bsz = batch.size(0)
        total += head0_bpb.item() * bsz
        total_n += bsz
        mark_step(device)
    model.train()
    return total / total_n


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


# ---------------------------------------------------------------------------
# Generation — plain AR vs. self-speculative (MTP heads as draft). Fixed-shape training/eval is
# what matters for TPU throughput; these run only optionally, post-training, and (KV-cache path
# especially) will recompile per new sequence length on xla — not optimized for that here.
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_ar(model: ByteLM, prompt: torch.Tensor, n_new_bytes: int, temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    cfg = model.cfg
    tokens = prompt.clone()
    for _ in range(n_new_bytes):
        ctx = tokens[:, -cfg.context :]
        logits = model(ctx)[0][:, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
    model.train()
    return tokens


@torch.no_grad()
def generate_no_cache(model: ByteLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: torch.device) -> torch.Tensor:
    """Reference (slow, obviously-correct) GREEDY decode: recomputes the whole trunk from scratch
    over the whole sequence every new byte."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    cfg = model.cfg

    for _ in range(n_new_bytes):
        ctx = all_bytes[:, -cfg.context:]
        logits = model(ctx)[0][:, -1]   # head 0 (immediate next-byte) only
        next_byte = logits.argmax(-1)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: ByteLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: torch.device) -> torch.Tensor:
    """KV-cache-efficient GREEDY decode — a per-layer cache_k/cache_v list, advanced one position
    at a time via each Block's forward_step."""
    cfg = model.cfg
    n_layers = len(model.blocks)
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)

    cache_k: list[torch.Tensor | None] = [None] * n_layers
    cache_v: list[torch.Tensor | None] = [None] * n_layers

    def step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = model.tok_emb(byte_id).unsqueeze(1)
        cos_new, sin_new = rope_cos_sin_at(pos, cfg.head_dim, cfg.rope_base, device)
        for li, block in enumerate(model.blocks):
            x, cache_k[li], cache_v[li] = block.forward_step(x, cos_new, sin_new, cache_k[li], cache_v[li])
        return model.ln_f(x).squeeze(1)

    L0 = prompt_bytes.size(1)
    last_h = None
    for pos in range(L0):
        last_h = step(prompt_bytes[:, pos], pos)

    out_bytes = [prompt_bytes]
    for i in range(n_new_bytes):
        logits = model.heads[0](last_h)
        next_byte = logits.argmax(-1)
        out_bytes.append(next_byte.unsqueeze(1))
        last_h = step(next_byte, L0 + i)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def validate_generation(model: ByteLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: torch.device) -> bool:
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


@torch.no_grad()
def generate_speculative(
    model: ByteLM, prompt: torch.Tensor, n_new_bytes: int, temperature: float = 1.0
) -> tuple[torch.Tensor, list[int]]:
    """Self-speculative decoding, draft = the model's own MTP heads. Batch size 1 only
    (acceptance length varies per sequence)."""
    assert prompt.size(0) == 1, "generate_speculative supports batch size 1"
    model.eval()
    cfg = model.cfg
    n_heads = cfg.mtp_heads
    tokens = prompt.clone()
    accept_lengths: list[int] = []
    generated = 0

    while generated < n_new_bytes:
        ctx = tokens[:, -cfg.context :]
        draft_logits = model(ctx)[:, :, -1, :]                      # [n_heads, 1, vocab]
        draft_probs = F.softmax(draft_logits / temperature, dim=-1)
        draft_tokens = torch.multinomial(draft_probs.squeeze(1), 1).squeeze(-1)  # [n_heads]
        candidate = torch.cat([tokens, draft_tokens.unsqueeze(0)], dim=1)

        verify_ctx = candidate[:, -cfg.context :]
        verify_logits = model(verify_ctx)[0]                        # head-0, true causal: [1, T, vocab]
        target_logits = verify_logits[:, -(n_heads + 1) : -1]       # p(x_{t+i} | x_{<t+i}), i=1..n_heads
        target_probs = F.softmax(target_logits / temperature, dim=-1).squeeze(0)  # [n_heads, vocab]

        accepted = 0
        for i in range(n_heads):
            tok = draft_tokens[i].item()
            p_target = target_probs[i, tok].item()
            p_draft = draft_probs[i, 0, tok].item()
            if torch.rand(()).item() < min(1.0, p_target / max(p_draft, 1e-8)):
                accepted += 1
            else:
                break

        if accepted > 0:
            tokens = torch.cat([tokens, draft_tokens[:accepted].unsqueeze(0)], dim=1)
        if accepted < n_heads:
            resid = (target_probs[accepted] - draft_probs[accepted, 0]).clamp_min(0)
            resid = resid if resid.sum() > 0 else target_probs[accepted]
            next_tok = torch.multinomial(resid / resid.sum(), 1)
        else:
            bonus_probs = F.softmax(verify_logits[:, -1] / temperature, dim=-1)
            next_tok = torch.multinomial(bonus_probs.squeeze(0), 1).unsqueeze(0)
        tokens = torch.cat([tokens, next_tok.reshape(1, 1)], dim=1)

        accept_lengths.append(accepted)
        generated += accepted + 1

    model.train()
    return tokens, accept_lengths


def benchmark_generation(model: ByteLM, prompt: torch.Tensor, n_bytes: int, temperature: float = 1.0, log=print):
    t0 = time.perf_counter()
    generate_ar(model, prompt, n_bytes, temperature)
    ar_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, accept_lengths = generate_speculative(model, prompt, n_bytes, temperature)
    spec_time = time.perf_counter() - t0

    avg_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    log(
        f"generation benchmark ({n_bytes} bytes): "
        f"plain_ar={ar_time:.2f}s ({n_bytes/ar_time:.1f} B/s)  "
        f"speculative={spec_time:.2f}s ({n_bytes/spec_time:.1f} B/s)  "
        f"avg_accept_len={avg_accept:.2f}/{model.cfg.mtp_heads}  "
        f"speedup={ar_time/spec_time:.2f}x"
    )


@torch.no_grad()
def score_continuation_bpb(model: ByteLM, full_bytes: bytes, prompt_len: int, device: torch.device) -> float:
    model.eval()
    seq = torch.tensor([list(full_bytes)], dtype=torch.long, device=device)
    inputs, targets = seq[:, :-1], seq[:, 1:]
    logits = model(inputs)[0]  # head 0: [1, T, vocab]
    cont_logits = logits[:, prompt_len - 1 :]
    cont_targets = targets[:, prompt_len - 1 :]
    nats = F.cross_entropy(cont_logits.reshape(-1, cont_logits.size(-1)), cont_targets.reshape(-1))
    model.train()
    return (nats / math.log(2)).item()


def qualitative_generate(
    model: ByteLM, prompt_bytes: bytes, gen_len: int, ground_truth: bytes | None, device: torch.device,
    temperature: float = 1.0, log=print,
) -> None:
    prompt = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    out, _ = generate_speculative(model, prompt, gen_len, temperature)
    gen_bytes = bytes(out[0, prompt.size(1):].tolist())

    log(f"qual_prompt:       {prompt_bytes!r}")
    log(f"qual_generated:    {gen_bytes!r}")
    if ground_truth is not None:
        log(f"qual_ground_truth: {ground_truth!r}")
        bpb = score_continuation_bpb(model, prompt_bytes + ground_truth, len(prompt_bytes), device)
        log(f"qual_bpb_on_ground_truth: {bpb:.4f}", qual_bpb_on_ground_truth=bpb)


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Python config file (configs/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Byte-level causal transformer + MTP-head LM baseline (BPB), CPU/TPU only", parents=[pre]
    )
    p.add_argument("--preset", choices=list(PRESETS), default="sd")
    p.add_argument("--device", choices=["cpu", "xla"], default=None, help="default: auto (xla if available, else cpu)")
    p.add_argument("--context", type=int, default=None, help="override preset's context length")
    p.add_argument("--mtp_heads", type=int, default=None, help="override preset's MTP head count")
    p.add_argument("--n_layers", type=int, default=None, help="override preset's transformer layer count")
    p.add_argument("--no_zero_kv_sink", action="store_true", help="disable the (default-on) all-zero, always-attendable K/V sink token")
    p.add_argument("--no_torch_compile", action="store_true", help="disable the (default-on) torch.compile wrap (backend=openxla on xla, inductor on cpu)")
    p.add_argument("--use_flash_attention", action="store_true",
                    help="use torch_xla's Pallas flash-attention kernel in the training forward pass (nightly-only, see module docstring; silently falls back to plain SDPA if unavailable)")
    p.add_argument("--multichip", action="store_true",
                    help="data-parallel across all locally-addressable TPU devices via torch_xla.launch (--batch_size is per-process; global batch = batch_size * world_size)")
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None, help="prefix of the corpus to load (default: all)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.0,
                    help="held-out test fraction, chronologically before val (default 0: no test split)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=1000, help="steps held at peak LR before cosine decay begins")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--full_val_eval", action="store_true")
    p.add_argument("--benchmark_generate_bytes", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_split", choices=["train", "val", "test"], default="val")
    p.add_argument("--checkpoint_path", type=Path, default=None)
    p.add_argument("--qual_gen_bytes", type=int, default=0)
    p.add_argument("--qual_source", choices=["train", "val", "user"], default="val")
    p.add_argument("--qual_prompt_bytes", type=int, default=64)
    p.add_argument("--qual_user_text", type=str, default=None)

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()
    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")
    if args.qual_gen_bytes > 0 and args.qual_source == "user" and not args.qual_user_text:
        p.error("--qual_source user requires --qual_user_text")
    if args.eval_split == "test" and args.test_frac <= 0:
        p.error("--eval_split test requires --test_frac > 0")
    if args.multichip and not _HAS_XLA:
        p.error("--multichip requires torch_xla")

    if args.multichip:
        torch_xla.launch(_run, args=(args, pre_args))
    else:
        _run(0, args, pre_args)


def _run(index: int, args: argparse.Namespace, pre_args: argparse.Namespace) -> None:
    is_master = index == 0
    device = resolve_device(args.device)

    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        cfg = LMConfig(**ckpt["cfg"])
        model = ByteLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        if not args.no_torch_compile:
            model = torch.compile(model, backend="openxla" if is_xla_device(device) else "inductor")
    else:
        cfg = PRESETS[args.preset]
        if args.context is not None:
            cfg.context = args.context
        if args.mtp_heads is not None:
            cfg.mtp_heads = args.mtp_heads
        if args.n_layers is not None:
            cfg.n_layers = args.n_layers
        if args.no_zero_kv_sink:
            cfg.zero_kv_sink = False
        if args.use_flash_attention:
            cfg.use_flash_attention = True
        if cfg.zero_kv_sink and cfg.use_flash_attention and (cfg.context + 1) % 1024 != 0:
            print(f"WARNING: zero_kv_sink+use_flash_attention needs (context+1) divisible by "
                  f"1024 to use the fast path (got context={cfg.context}, context+1="
                  f"{cfg.context + 1}) — falling back to the explicit-mask SDPA path (O(T^2) "
                  f"memory) for this run. Pick context=1024*k-1 (e.g. 8191, not 8192) to actually "
                  f"get flash-attention's memory savings with the sink on.")
        model = ByteLM(cfg).to(device)
        start_step = 0
        if not args.no_torch_compile:
            model = torch.compile(model, backend="openxla" if is_xla_device(device) else "inductor")

    if args.run_name:
        run_name = args.run_name
    elif pre_args.config:
        run_name = pre_args.config.stem
    else:
        run_name = f"bytelm_tpu_{args.preset}_{int(time.time())}"

    if is_master:
        log = Logger(args.logs_dir / run_name)
        print(f"run_name={run_name}  logging to {log.text_path} (raw text) / {log.json_path} (JSONL) — tail -f {log.text_path}")
    else:
        log = lambda msg="", **record: None  # noqa: E731 — non-master ranks do no file/console I/O
    preset_label = f"loaded_from={args.checkpoint_path} (step {start_step})" if args.checkpoint_path else f"preset={args.preset}"
    # xr.addressable_runtime_device_count() returns the CALLING process's own local device count
    # (1, inside an already-spawned multichip worker) -- xr.world_size() is the real replica
    # count across all processes. Confirmed bug (logged world_size=1 despite 4 workers actually
    # running) on 2026-08-23 while verifying --multichip works on the stable torch_xla==2.9.0
    # pin (see docs/bytelm_tpu_setup.md).
    world_size = xr.world_size() if (args.multichip and _HAS_XLA) else 1
    log(
        f"{preset_label}  params={count_params(model)/1e6:.1f}M  device={device}  xla={_HAS_XLA}"
        f"  context={cfg.context}  mtp_heads={cfg.mtp_heads}  zero_kv_sink={cfg.zero_kv_sink}"
        f"  use_flash_attention={cfg.use_flash_attention} (available={_HAS_FLASH_ATTENTION})"
        f"  torch_compile={not args.no_torch_compile}"
        + (f"  multichip=True world_size={world_size} global_batch={args.batch_size * world_size}" if args.multichip else "")
    )

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data, test_data = split_train_val_test(data, args.val_frac, args.test_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}"
        + (f"  test_bytes={len(test_data)}" if test_data is not None else "  test_bytes=0 (no --test_frac)"))
    if not args.eval_only:
        seq_len = cfg.context + cfg.mtp_heads
        steps_per_epoch = len(train_data) / (args.batch_size * seq_len)
        epochs = args.steps / steps_per_epoch
        log(f"~{steps_per_epoch:.1f} steps/epoch  ~{epochs:.1f} epochs over train_bytes "
            f"(steps={args.steps} batch_size={args.batch_size} seq_len={seq_len}, "
            f"random-with-replacement sampling — see batch_iter)")
    val_iter = batch_iter(val_data, args.batch_size, cfg.context, cfg.mtp_heads, device)

    if args.eval_only:
        if is_master:
            eval_data = {"train": train_data, "val": val_data, "test": test_data}[args.eval_split]
            eval_bpb_val = eval_bpb_full(model, eval_data, args.batch_size, cfg.context, cfg.mtp_heads, device, desc=f"{args.eval_split}_full")
            log(f"eval_only_full_{args.eval_split}set  {args.eval_split}_bpb {eval_bpb_val:.4f}",
                **{f"{args.eval_split}_bpb": eval_bpb_val})
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        train_iter = batch_iter(train_data, args.batch_size, cfg.context, cfg.mtp_heads, device)
        checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True) if is_master else None

        model.train()
        # Every rank must run this loop in lockstep (optimizer_step's gradient all-reduce is a
        # collective op) — only master creates the tqdm bar / does eval+checkpoint I/O, so
        # non-master ranks don't clutter shared stdout with duplicate progress bars.
        step_iter = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True) if is_master else range(1, args.steps + 1)
        for step in step_iter:
            if step == 1 and is_master:
                # First step pays a one-time trace+compile cost (XLA lazy compilation, plus
                # torch.compile's own tracing if enabled) that dwarfs steady-state step time —
                # timed and logged separately so it isn't mistaken for the real per-step rate.
                compile_t0 = time.perf_counter()
            if args.cosine_decay:
                lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
            else:
                lr = lr_at(step, args.warmup_steps, args.lr_peak)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = next(train_iter)
            with autocast_ctx(device):
                logits = model(batch[:, : cfg.context])
            loss, head0_bpb = mtp_loss(logits, batch, cfg.context)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer_step(opt, device)

            if not is_master:
                continue
            if step == 1:
                first_step_s = time.perf_counter() - compile_t0
                log(f"first_step_compile_s {first_step_s:.1f}", first_step_compile_s=first_step_s)
            step_iter.set_postfix(lr=f"{lr:.2e}", mtp_loss=f"{loss.item():.4f}", bpb=f"{head0_bpb.item():.4f}")
            if step % args.log_every == 0:
                log(f"{step_iter}", step=step, lr=lr, mtp_loss=loss.item(), bpb=head0_bpb.item())
            if step % args.eval_every == 0 or step == args.steps:
                if args.full_val_eval:
                    # Chunked through args.batch_size (never a single-giant-batch -1 pass) to
                    # avoid OOM — same per-process batch size already proven safe in the training
                    # forward pass itself, and eval has no backward-pass memory to worry about.
                    # desc= shows a live tqdm bar (visible progress + timing, not stuck-vs-slow
                    # guesswork) and elapsed time gets logged alongside the bpb numbers below.
                    t0 = time.perf_counter()
                    val_bpb = eval_bpb_full(model, val_data, args.batch_size, cfg.context, cfg.mtp_heads, device, desc="val_full")
                    val_eval_s = time.perf_counter() - t0
                else:
                    val_bpb = eval_bpb(model, val_iter, cfg.context, args.eval_batches, device)
                    val_eval_s = None
                log_line = f"step {step:5d}  val_bpb {val_bpb:.4f}"
                log_fields = {"step": step, "val_bpb": val_bpb}
                if val_eval_s is not None:
                    log_line += f"  val_eval_s {val_eval_s:.1f}"
                    log_fields["val_eval_s"] = val_eval_s
                if args.full_val_eval and test_data is not None:
                    # Logged for observability only — never drives checkpoint selection (that
                    # stays val_bpb-only below), so this doesn't compromise test as a genuinely
                    # held-out number; it just gets watched throughout training instead of only
                    # once at the end.
                    t0 = time.perf_counter()
                    test_bpb_now = eval_bpb_full(model, test_data, args.batch_size, cfg.context, cfg.mtp_heads, device, desc="test_full")
                    test_eval_s = time.perf_counter() - t0
                    log_line += f"  test_bpb {test_bpb_now:.4f}  test_eval_s {test_eval_s:.1f}"
                    log_fields["test_bpb"] = test_bpb_now
                    log_fields["test_eval_s"] = test_eval_s
                log(log_line, **log_fields)
                checkpointer.step(
                    {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg), "val_bpb": val_bpb},
                    val_bpb,
                )
        if is_master:
            log(
                f"checkpoints: best={checkpointer.best_path} (val_bpb {checkpointer.best_metric:.4f})  last={checkpointer.last_path}"
            )
        if test_data is not None and is_master:
            # Held out from both training and every val_bpb-driven checkpoint/LR decision, so
            # unlike val_bpb this number was never used to pick anything during the run — load
            # the val-selected best checkpoint back and score it on test exactly once, at the end.
            best_ckpt = torch.load(checkpointer.best_path, map_location="cpu")
            model.load_state_dict(best_ckpt["model"])
            test_bpb = eval_bpb_full(model, test_data, args.batch_size, cfg.context, cfg.mtp_heads, device, desc="final_test_full")
            log(f"final_test_bpb (best-val checkpoint, step {best_ckpt['step']})  test_bpb {test_bpb:.4f}",
                test_bpb=test_bpb, test_bpb_from_step=best_ckpt["step"])

    if not is_master:
        return

    if args.benchmark_generate_bytes > 0:
        prompt = next(val_iter)[:1, :1]  # one real byte as prompt, batch size 1
        benchmark_generation(model, prompt, args.benchmark_generate_bytes, log=log)

    if args.qual_gen_bytes > 0:
        if args.qual_source == "user":
            prompt_bytes = args.qual_user_text.encode("utf-8")
            ground_truth = None
        else:
            src_data = train_data if args.qual_source == "train" else val_data
            total_len = args.qual_prompt_bytes + args.qual_gen_bytes
            start = torch.randint(0, len(src_data) - total_len, (1,)).item()
            window = src_data[start : start + total_len].tolist()
            prompt_bytes = bytes(window[: args.qual_prompt_bytes])
            ground_truth = bytes(window[args.qual_prompt_bytes :])
        qualitative_generate(model, prompt_bytes, args.qual_gen_bytes, ground_truth, device, log=log)


if __name__ == "__main__":
    main()
