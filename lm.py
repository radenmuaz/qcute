"""lm.py — Phase 0 reference baseline: byte-level causal transformer LM with RoPE.

Handover doc §5 Phase 0 calls for a byte-softmax LM baseline ("numbers to beat")
before the continuous tokenizer is built. This is that baseline: plain pre-norm
transformer, rotary position embeddings, causal self-attention, weight-tied
output head. Reports exact bits-per-byte (softmax over 256 raw bytes -> no
ELBO needed, unlike the FSQ/vMF bottlenecks in qcute.py).

Two power-of-2-friendly presets (see PRESETS below):
  sd  ~100M params : d_model=1024, layers=8,  heads=16, head_dim=64,  ctx=2048
  md  ~400M params : d_model=2048, layers=8,  heads=16, head_dim=128, ctx=2048

Deliberately monolithic; factorize once this needs to share code with qcute.py
beyond the enwik8 loader.
"""
from __future__ import annotations

import argparse
import gzip
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


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

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS: dict[str, LMConfig] = {
    # ~12 * d_model^2 * n_layers non-embedding params (vocab=256 is negligible)
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
        return self.down(F.gelu(self.up(x)))


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
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
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
        # tokens: [B, T] long -> logits [B, T, vocab]
        B, T = tokens.shape
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, tokens.device)
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return nats / math.log(2)


def batch_iter(data: torch.Tensor, batch_size: int, context: int, device: str):
    seq_len = context + 1  # +1 for the next-byte target shift
    n = (len(data) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.to(device)


def lr_at(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * peak * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    p = argparse.ArgumentParser(description="Byte-level causal transformer LM baseline (BPB)")
    p.add_argument("--preset", choices=list(PRESETS), default="sd")
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8.gz"))
    p.add_argument("--n_bytes", type=int, default=20_000_000, help="prefix of enwik8 to load")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PRESETS[args.preset]
    model = ByteLM(cfg).to(device)
    print(f"preset={args.preset}  params={count_params(model)/1e6:.1f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)

    data = load_enwik8(args.data, args.n_bytes)
    data_iter = batch_iter(data, args.batch_size, cfg.context, device)

    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.warmup_steps, args.steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(data_iter)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        bpb = bits_per_byte(logits, targets)

        opt.zero_grad()
        bpb.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            print(f"step {step:5d}  lr {lr:.2e}  bpb {bpb.item():.4f}")


if __name__ == "__main__":
    main()
