"""qcute.bpelm — BPE baseline: byte-level BPE tokenization (sentencepiece) +
a plain causal transformer, single next-token head (no MTP — bandwidth here
comes purely from BPE merging, per user direction, not stacked with MTP).

Handover doc §1.6 names BPE+MTP as the strong baseline; this module is the
BPE half alone (qcute.bytelm is the separate byte+MTP half) — together they
let you compare bandwidth mechanisms independently. `--vocab_size` targets
~qcute.qcutelm's K=8 bytes/timestep (see scripts/train_bpe.py for why that's
only approximately reachable on the tiny corpus).

Architecture mirrors qcute.bytelm's trunk (pre-norm transformer, RoPE,
weight-tied output head, same GPT-2-style init) minus the MTP heads — same
duplication-is-deliberate rationale as bytelm/qcutelm (see their docstrings
and docs/architecture.md): this module's forward pass and BPB accounting
(token-level CE reweighted by each token's actual byte length, not treated
as 1 unit) are specific enough to this tokenization scheme that sharing
would obscure more than it'd save.

Requires a tokenizer trained with scripts/train_bpe.py first:
    uv run python scripts/train_bpe.py --data datasets/enwik8_tiny.gz
    uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_tiny_8192.model
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sentencepiece as spm
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
    """Same pattern as qcute.bytelm / qcute.qcutelm — see their docstrings.
    Writes into its own run directory (logs/<run_name>/run.{log,jsonl})."""

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
    """Same pattern as qcute.bytelm / qcute.qcutelm — see their docstrings.
    Writes into its own run directory (checkpoints/<run_name>/{best,last}.pt)."""

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
class BpeLMConfig:
    vocab: int = 8192       # overwritten from the sentencepiece model at load time
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    context: int = 256       # tokens, not bytes
    mlp_mult: int = 4
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


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
    def __init__(self, cfg: BpeLMConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: BpeLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class BpeLM(nn.Module):
    """Plain causal transformer over BPE token ids — single next-token head,
    no MTP. Weight-tied embedding/output head, GPT-2-style init (same
    load-bearing reasoning as qcute.bytelm: default Embedding init blows up
    init-time loss)."""

    def __init__(self, cfg: BpeLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            for proj in (block.attn.out, block.mlp[-1]):
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, tokens.device)
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_byte_len_table(sp: spm.SentencePieceProcessor) -> torch.Tensor:
    """Exact UTF-8 byte length of each token id's decoded text — used to
    convert token-level cross-entropy into bits-per-*byte*, the number
    that's actually comparable to qcute.bytelm / qcute.qcutelm's bpb.

    Requires a lossless tokenizer (scripts/train_bpe.py's identity
    normalization + no whitespace collapsing + byte_fallback) — verified by
    that script's roundtrip check at training time. Byte-fallback pieces
    (sentencepiece's literal `<0xXX>` fallback for any char outside the
    learned vocab) need special-casing here: each one is exactly 1 raw
    byte, not the 6 UTF-8 bytes its literal string form would encode to."""
    lens = []
    for i in range(sp.vocab_size()):
        piece = sp.IdToPiece(i)
        if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            lens.append(1)
        else:
            lens.append(max(1, len(piece.replace("▁", " ").encode("utf-8"))))
    return torch.tensor(lens, dtype=torch.long)


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor, byte_len_table: torch.Tensor) -> torch.Tensor:
    """Exact: sum of per-token nats / sum of those tokens' real byte lengths
    / ln2 — not `mean_nats_per_token / avg_bytes_per_token`, which would be
    a biased estimate whenever token frequency correlates with token length
    (it does: common short tokens vs. rare long ones)."""
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
    n_bytes = byte_len_table[targets.reshape(-1)].sum()
    return nats / n_bytes.clamp(min=1) / math.log(2)


def encode_corpus(sp: spm.SentencePieceProcessor, path: Path, n_bytes: int | None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        raw = f.read(n_bytes) if n_bytes else f.read()
    text = raw.decode("utf-8", errors="replace")
    ids = sp.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def batch_iter(data: torch.Tensor, batch_size: int, context: int, device: str):
    seq_len = context + 1
    n = (len(data) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.to(device)


@torch.no_grad()
def eval_bpb(model: nn.Module, data_iter, byte_len_table: torch.Tensor, n_batches: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        batch = next(data_iter)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        total += bits_per_byte(model(inputs), targets, byte_len_table).item()
    model.train()
    return total / n_batches


def lr_at(step: int, warmup: int, peak: float) -> float:
    """Linear warmup, then constant at peak — same schedule as qcute.bytelm
    / qcute.qcutelm, for a fair comparison across all three baselines."""
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


@torch.no_grad()
def generate_ar(model: BpeLM, prompt_ids: torch.Tensor, n_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    """Naive autoregressive decode, one BPE token per forward pass. Unlike
    qcute.bytelm this has no MTP heads to draft with, so there's no
    speculative variant here — this is the only decode path."""
    model.eval()
    cfg = model.cfg
    tokens = prompt_ids.clone()
    for _ in range(n_new_tokens):
        ctx = tokens[:, -cfg.context :]
        logits = model(ctx)[:, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
    model.train()
    return tokens


@torch.no_grad()
def score_continuation_bpb(
    model: BpeLM, full_ids: torch.Tensor, prompt_token_len: int, byte_len_table: torch.Tensor, device: str
) -> float:
    """Teacher-forced bpb on just the continuation tokens (full_ids[prompt_token_len:]),
    given the real prompt as context — same idea as qcute.bytelm's function of
    the same name, byte-weighted instead of token-averaged (see bits_per_byte)."""
    model.eval()
    seq = full_ids.to(device)
    inputs, targets = seq[:, :-1], seq[:, 1:]
    logits = model(inputs)
    cont_logits = logits[:, prompt_token_len - 1 :]
    cont_targets = targets[:, prompt_token_len - 1 :]
    bpb = bits_per_byte(cont_logits, cont_targets, byte_len_table)
    model.train()
    return bpb.item()


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

    p = argparse.ArgumentParser(description="qcute BPE baseline: sentencepiece BPE + plain causal LM", parents=[pre])
    p.add_argument("--sp_model", type=Path, default=None, help="sentencepiece .model from scripts/train_bpe.py (required, via CLI or --config)")
    p.add_argument("--context", type=int, default=256, help="context length in tokens")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_tiny.gz"))
    p.add_argument("--n_bytes", type=int, default=None, help="prefix of the corpus to load (default: all)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--run_name", type=str, default=None, help="run directory name under logs/ and checkpoints/; falls back to --config filename, then bpelm_<timestamp>")
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint_path", type=Path, default=None)

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()
    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")
    if args.sp_model is None:
        p.error("--sp_model is required (directly or via --config)")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=str(args.sp_model))
    byte_len_table = build_byte_len_table(sp)

    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        cfg = BpeLMConfig(**ckpt["cfg"])
        model = BpeLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
    else:
        cfg = BpeLMConfig(
            vocab=sp.vocab_size(), context=args.context, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
        model = BpeLM(cfg).to(device)
        start_step = 0
    byte_len_table = byte_len_table.to(device)

    if args.run_name:
        run_name = args.run_name
    elif pre_args.config:
        run_name = pre_args.config.stem
    else:
        run_name = f"bpelm_{int(time.time())}"
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} (raw text) / {log.json_path} (JSONL) — tail -f {log.text_path}")
    preset_label = f"loaded_from={args.checkpoint_path} (step {start_step})" if args.checkpoint_path else "fresh_init"
    log(
        f"{preset_label}  vocab={cfg.vocab}  params={count_params(model)/1e6:.2f}M  device={device}"
        f"  context={cfg.context} tokens"
    )

    data = encode_corpus(sp, args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_tokens={len(train_data)}  val_tokens={len(val_data)}")
    val_iter = batch_iter(val_data, args.batch_size, cfg.context, device)

    if args.eval_only:
        val_bpb = eval_bpb(model, val_iter, byte_len_table, args.eval_batches)
        log(f"eval_only  val_bpb {val_bpb:.4f}", val_bpb=val_bpb)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
    train_iter = batch_iter(train_data, args.batch_size, cfg.context, device)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(train_iter)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        bpb = bits_per_byte(logits, targets, byte_len_table)

        opt.zero_grad()
        bpb.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        pbar.set_postfix(lr=f"{lr:.2e}", bpb=f"{bpb.item():.4f}")
        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, bpb=bpb.item())
        if step % args.eval_every == 0 or step == args.steps:
            val_bpb = eval_bpb(model, val_iter, byte_len_table, args.eval_batches)
            log(f"step {step:5d}  val_bpb {val_bpb:.4f}", step=step, val_bpb=val_bpb)
            checkpointer.step(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg), "val_bpb": val_bpb},
                val_bpb,
            )
    log(f"checkpoints: best={checkpointer.best_path} (val_bpb {checkpointer.best_metric:.4f})  last={checkpointer.last_path}")


if __name__ == "__main__":
    main()
