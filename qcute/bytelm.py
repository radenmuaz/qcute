"""qcute.bytelm — Phase 0 reference baseline: byte-level causal transformer LM with
RoPE and an MTP head, bandwidth-matched to qcute.qcutelm.

Handover doc §5 Phase 0 calls for reference baselines ("numbers to beat")
before the continuous tokenizer is built, and §1.6 names BPE+MTP as the
*strong* baseline: n parallel softmax heads get bandwidth at ~1x convergence,
no VAE overhead. This is that baseline, byte-level: plain pre-norm
transformer, rotary position embeddings, causal self-attention, `mtp_heads`
parallel output heads (head 0 weight-tied, matching standard practice; the
rest untied) predicting bytes t+1..t+n from the same trunk hidden state.
`mtp_heads` targets matched bandwidth against qcute.qcutelm's `K` and
qcute.bpelm's bytes/token — 4 for `xs` (tiny-corpus-scale, see PRESETS
comment for why 8 was a mismatch there), 8 for `sd`/`md` (full-corpus scale,
matching the handover doc's §6.1 default). Reports exact bits-per-byte from
head 0 (the standard next-byte metric) — no ELBO needed, unlike the FSQ/BSQ
bottleneck in qcute.qcutelm.

Three power-of-2-friendly presets (see PRESETS below):
  xs  ~3.7M params  : d_model=256,  layers=4, heads=4,  head_dim=64,  ctx=256,  mtp_heads=4
  sd  ~100M params  : d_model=1024, layers=8, heads=16, head_dim=64,  ctx=2048, mtp_heads=8
  md  ~400M params  : d_model=2048, layers=8, heads=16, head_dim=128, ctx=2048, mtp_heads=8
`--context`/`--mtp_heads` override the preset for quick experiments.

Deliberately monolithic (one module, no internal submodules) for now.

Duplication vs. qcute.qcutelm is a deliberate split, not oversight: pure
infra with zero model-specific coupling (Logger, Checkpointer,
load_config_module, load_enwik8, split_train_val, lr_at, the RoPE math
rope_cos_sin/rotate_half/apply_rope) is a real "should share into a
qcute/utils.py" candidate — not yet done, tracked as a pending decision.
Architecture-bearing code (CausalSelfAttention/Block/MLP, the generation
functions, eval_bpb) stays duplicated on purpose: it *is* the model each
file exists to let you read and debug standalone, and qcutelm's generation
is a structurally different shape (encode→code→decode, not byte-AR) so a
shared abstraction there wouldn't be honest.
"""
from __future__ import annotations

import argparse
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


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    """Terminal: human-readable lines via tqdm.write, at the usual
    log_every/eval_every interval (doesn't clobber the progress bar). Every
    line is prefixed with elapsed time since the Logger was created, as
    [HH:MM:SS], and the record also carries elapsed_s (int) / elapsed_hms.

    Writes into its own run directory (logs/<run_name>/), not loose files in
    logs/ — keeps everything for one run (plus its checkpoints, in the
    matching checkpoints/<run_name>/) findable by run_name alone:
      run.log   raw terminal text, exactly what's printed to stdout —
                `tail -f` this for a human-readable live view.
      run.jsonl structured, one JSON record per line — for later plotting.
    Both written only at the log_every/eval_every interval — tqdm's live bar
    (constant \\r-redraws) never touches either file.
    """

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
        # msg is redundant once structured fields are present (they're the parsed-out
        # version of the same text); keep it only for plain informational lines.
        json_record = {"elapsed_s": elapsed_s, "elapsed_hms": elapsed_hms, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(json_record) + "\n")
        self.json_f.flush()


class Checkpointer:
    """Saves two files in its own run directory (checkpoints/<run_name>/,
    matching the Logger's logs/<run_name>/): `best.pt` is overwritten only
    when the tracked val metric improves; `last.pt` is overwritten every
    `save_every_n_evals` eval calls (default 1, i.e. every eval). Each
    checkpoint carries the model/optimizer state, step, cfg (as a dict, to
    rebuild the model architecture on load), and the metric."""

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

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS: dict[str, LMConfig] = {
    # ~12 * d_model^2 * n_layers non-embedding params (vocab=256 is negligible)
    # xs: mtp_heads=4, not the default 8 — on a tiny corpus, BPE (qcute.bpelm's
    # fair comparison point) only reaches ~4 bytes/token before larger vocabs
    # start memorizing phrases rather than generalizing (see scripts/train_bpe.py),
    # so targeting 8 bytes/timestep here would be an unfair bandwidth mismatch.
    # sd/md keep mtp_heads=8, matched to qcute.qcutelm's K=8 and the handover
    # doc's full-corpus-scale default (§6.1) — 8 becomes achievable there.
    "xs": LMConfig(d_model=256, n_layers=4, n_heads=4, context=256, mtp_heads=4),  # ~3.7M, for quick local runs
    "sd": LMConfig(d_model=1024, n_layers=8, n_heads=16, context=2048),   # ~101M
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)  # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # [B, H, T, hd]
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class ByteLM(nn.Module):
    """MTP baseline (handover §1.6): n parallel softmax heads predict bytes
    t+1..t+n from the same trunk hidden state, bandwidth-matched to
    qcute.qcutelm's K so the two are a fair BPB-at-matched-bandwidth
    comparison. Head 0 (immediate next-byte) is weight-tied to the input
    embedding as usual; the other n-1 heads are untied (standard MTP)."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
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
    tokens: [B, context + n_heads] (extra n_heads-1 bytes of lookahead target).
    Returns (mean loss over all heads, head-0 bpb i.e. the standard next-byte
    metric comparable to qcute.bytelm without MTP and to qcute.qcutelm's bpb)."""
    n_heads = logits.size(0)
    losses = []
    for i in range(n_heads):
        targets_i = tokens[:, i + 1 : i + 1 + context]
        losses.append(F.cross_entropy(logits[i].reshape(-1, logits.size(-1)), targets_i.reshape(-1)))
    losses = torch.stack(losses)
    head0_bpb = losses[0] / math.log(2)
    return losses.mean(), head0_bpb


def batch_iter(data: torch.Tensor, batch_size: int, context: int, n_heads: int, device: str):
    seq_len = context + n_heads  # n_heads bytes of lookahead beyond the context window
    n = (len(data) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.to(device)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


@torch.no_grad()
def eval_bpb(model: nn.Module, data_iter, context: int, n_batches: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch = next(data_iter)
        logits = model(batch[:, :context])
        _, head0_bpb = mtp_loss(logits, batch, context)
        total += head0_bpb.item()
    model.train()
    return total / n_batches


def lr_at(step: int, warmup: int, peak: float) -> float:
    """Linear warmup, then constant at peak — same schedule used by both
    qcute.bytelm and qcute.qcutelm, for a fair comparison between the two."""
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


# ---------------------------------------------------------------------------
# Generation — plain AR vs. self-speculative (MTP heads as draft), for a
# latency benchmark comparable to qcute.qcutelm's K-bytes-per-LM-step decode.
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_ar(model: ByteLM, prompt: torch.Tensor, n_new_bytes: int, temperature: float = 1.0) -> torch.Tensor:
    """Naive autoregressive decode, one byte per forward pass (head 0 only)."""
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
def generate_speculative(
    model: ByteLM, prompt: torch.Tensor, n_new_bytes: int, temperature: float = 1.0
) -> tuple[torch.Tensor, list[int]]:
    """Self-speculative decoding (Leviathan et al. 2023-style), draft = the
    model's own MTP heads. mtp_heads propose bytes t+1..t+n from a single
    trunk pass at t; one verification forward pass (head 0, true causal)
    computes exact p(x_{t+i} | x_{<t+i}) for all i; accept/reject via
    standard speculative rejection sampling. Lets one verification step emit
    up to mtp_heads bytes — the fair latency comparison against
    qcute.qcutelm, which emits K bytes per LM step by construction.
    Batch size 1 only (acceptance length varies per sequence).
    """
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
    log(
        "compare against qcute.qcutelm's decode throughput "
        "(K bytes emitted per single LM+decoder step, unconditionally, no rejection)."
    )


@torch.no_grad()
def score_continuation_bpb(model: ByteLM, full_bytes: bytes, prompt_len: int, device: str) -> float:
    """Teacher-forced head-0 bpb on just the continuation (bytes[prompt_len:]),
    given the real prompt as context. Assumes len(full_bytes) <= cfg.context + 1."""
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
    model: ByteLM, prompt_bytes: bytes, gen_len: int, ground_truth: bytes | None, device: str,
    temperature: float = 1.0, log=print,
) -> None:
    """Generate a continuation from a prompt (dataset-drawn or user-supplied)
    and, if a real ground-truth continuation is available (dataset-drawn),
    show it alongside the model's guess plus the model's bpb on the truth —
    a qualitative complement to the aggregate val_bpb number."""
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
    """Load a Python config file (e.g. configs/bytelm_xs_tiny.py) as a dict of
    module-level variables. Values must already be the right type (Path(...),
    int, float, ...) — argparse's `type=` conversion only applies to strings
    passed on the actual command line, not to defaults."""
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
        description="Byte-level causal transformer + MTP-head LM baseline (BPB)", parents=[pre]
    )
    p.add_argument("--preset", choices=list(PRESETS), default="sd")
    p.add_argument("--context", type=int, default=None, help="override preset's context length")
    p.add_argument("--mtp_heads", type=int, default=None, help="override preset's MTP head count")
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8.gz"))
    p.add_argument("--n_bytes", type=int, default=20_000_000, help="prefix of enwik8 to load")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument(
        "--benchmark_generate_bytes", type=int, default=0,
        help="if >0, after training benchmark plain-AR vs. self-speculative (MTP-head draft) generation latency"
    )
    p.add_argument(
        "--run_name", type=str, default=None,
        help="run directory name under logs/ and checkpoints/; falls back to the --config filename, then bytelm_<preset>_<timestamp>"
    )
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1, help="write the 'last' checkpoint every N eval() calls")
    p.add_argument("--eval_only", action="store_true", help="skip training; load --checkpoint_path and just evaluate")
    p.add_argument("--checkpoint_path", type=Path, default=None, help="required with --eval_only; also usable to warm-start training")
    p.add_argument(
        "--qual_gen_bytes", type=int, default=0,
        help="if >0, after training/eval generate this many bytes qualitatively and log prompt/generated/(ground truth)"
    )
    p.add_argument("--qual_source", choices=["train", "val", "user"], default="val")
    p.add_argument("--qual_prompt_bytes", type=int, default=64, help="prompt length when --qual_source is train/val")
    p.add_argument("--qual_user_text", type=str, default=None, help="prompt text when --qual_source user (utf-8 encoded)")

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()
    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")
    if args.qual_gen_bytes > 0 and args.qual_source == "user" and not args.qual_user_text:
        p.error("--qual_source user requires --qual_user_text")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        cfg = LMConfig(**ckpt["cfg"])
        model = ByteLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
    else:
        cfg = PRESETS[args.preset]
        if args.context is not None:
            cfg.context = args.context
        if args.mtp_heads is not None:
            cfg.mtp_heads = args.mtp_heads
        model = ByteLM(cfg).to(device)
        start_step = 0

    if args.run_name:
        run_name = args.run_name
    elif pre_args.config:
        run_name = pre_args.config.stem
    else:
        run_name = f"bytelm_{args.preset}_{int(time.time())}"
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} (raw text) / {log.json_path} (JSONL) — tail -f {log.text_path}")
    preset_label = f"loaded_from={args.checkpoint_path} (step {start_step})" if args.checkpoint_path else f"preset={args.preset}"
    log(
        f"{preset_label}  params={count_params(model)/1e6:.1f}M  device={device}"
        f"  context={cfg.context}  mtp_heads={cfg.mtp_heads}"
    )

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")
    val_iter = batch_iter(val_data, args.batch_size, cfg.context, cfg.mtp_heads, device)

    if args.eval_only:
        val_bpb = eval_bpb(model, val_iter, cfg.context, args.eval_batches)
        log(f"eval_only  val_bpb {val_bpb:.4f}", val_bpb=val_bpb)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
        train_iter = batch_iter(train_data, args.batch_size, cfg.context, cfg.mtp_heads, device)
        checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

        model.train()
        pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
        for step in pbar:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = next(train_iter)
            logits = model(batch[:, : cfg.context])
            loss, head0_bpb = mtp_loss(logits, batch, cfg.context)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            pbar.set_postfix(lr=f"{lr:.2e}", mtp_loss=f"{loss.item():.4f}", bpb=f"{head0_bpb.item():.4f}")
            if step % args.log_every == 0:
                log(f"{pbar}", step=step, lr=lr, mtp_loss=loss.item(), bpb=head0_bpb.item())
            if step % args.eval_every == 0 or step == args.steps:
                val_bpb = eval_bpb(model, val_iter, cfg.context, args.eval_batches)
                log(f"step {step:5d}  val_bpb {val_bpb:.4f}", step=step, val_bpb=val_bpb)
                checkpointer.step(
                    {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg), "val_bpb": val_bpb},
                    val_bpb,
                )
        log(
            f"checkpoints: best={checkpointer.best_path} (val_bpb {checkpointer.best_metric:.4f})  last={checkpointer.last_path}"
        )

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
