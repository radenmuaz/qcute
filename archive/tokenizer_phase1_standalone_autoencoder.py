"""ARCHIVED 2026-08-02 — superseded by qcute/tokenizer.py's end-to-end
encoder+bottleneck+LM+decoder (interface Option A, handover §2.1). Kept for
reference (streaming-causal encoder design); not run or maintained.

qcute.tokenizer — Phase 1: standalone byte-chunk autoencoder (FSQ bottleneck).

Minimal implementation following docs/continuous_tokenizer_handover.md
sections 1.2.2 (FSQ), 1.3 (encoder), 1.4.2/1.4.3a (Phase-1 NAT decoder).

Encoder: causal recurrent body (GRU stand-in for a Mamba-style SSM) emitting one
latent every K bytes. Decoder: memoryless NAT block conditioned on z_t via FiLM,
trained with one-shot factorized cross-entropy. No LM yet — validates the
bottleneck alone per Phase 1's go/no-go (reconstruction > 99.5%).

Deliberately monolithic (one module, no internal submodules) for now;
split further once Phase 2 needs to share pieces with qcute.lm.
"""
from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    K: int = 8              # bytes per chunk / latent emission rate
    d_model: int = 256       # encoder/decoder width
    enc_layers: int = 2
    dec_layers: int = 2
    dec_heads: int = 4
    dq: int = 6              # FSQ dims
    L: int = 8               # FSQ levels per dim -> codebook 8**6
    vocab: int = 256         # raw bytes
    max_bytes: int = 4096    # positional embedding cap for encoder


class FSQ(nn.Module):
    """Finite scalar quantization bottleneck (handover §1.2.2)."""

    def __init__(self, d_in: int, dq: int, L: int):
        super().__init__()
        self.L = L
        self.proj = nn.Linear(d_in, dq)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        z_pre = self.proj(u)
        bound = (self.L - 1) / 2
        z_bounded = bound * torch.tanh(z_pre)
        z_rounded = torch.round(z_bounded)
        z_hat = z_bounded + (z_rounded - z_bounded).detach()  # straight-through
        return z_hat


class CausalByteEncoder(nn.Module):
    """Causal-over-bytes body emitting one latent every K bytes (handover §1.3).

    Uses a GRU as a simple, correct stand-in for a Mamba-style SSM: both are
    causal, O(N) recurrent bodies. Swap in a real SSM kernel later if needed.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_bytes, cfg.d_model)
        self.body = nn.GRU(cfg.d_model, cfg.d_model, num_layers=cfg.enc_layers, batch_first=True)
        self.fsq = FSQ(cfg.d_model, cfg.dq, cfg.L)

    def forward(self, bytes_: torch.Tensor) -> torch.Tensor:
        # bytes_: [B, T*K] long
        B, N = bytes_.shape
        pos = torch.arange(N, device=bytes_.device).clamp_max(self.cfg.max_bytes - 1)
        x = self.byte_emb(bytes_) + self.pos_emb(pos)
        h, _ = self.body(x)                                   # [B, N, d_model], causal by construction
        h_boundaries = h[:, self.cfg.K - 1 :: self.cfg.K, :]  # h_{tK} for t=1..T: [B, T, d_model]
        return self.fsq(h_boundaries)                          # [B, T, dq]


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [N, K, d_model], cond: [N, cond_dim]
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class NATDecoder(nn.Module):
    """Memoryless parallel decoder, Phase-1 baseline (handover §1.4.2).

    Given z_t alone, expands to K byte predictions via a small bidirectional
    transformer with per-layer FiLM conditioning on z_t. One-shot factorized
    training (handover §1.4.3a): independent per-position cross-entropy.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.slot_pos = nn.Embedding(cfg.K, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model, nhead=cfg.dec_heads, batch_first=True, norm_first=True
                )
                for _ in range(cfg.dec_layers)
            ]
        )
        self.films = nn.ModuleList([FiLM(cfg.dq, cfg.d_model) for _ in range(cfg.dec_layers)])
        self.out = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, dq] -> logits [B, T, K, vocab]
        B, T, _ = z.shape
        cond = self.z_proj(z).reshape(B * T, self.cfg.d_model)
        x = cond.unsqueeze(1) + self.slot_pos.weight.unsqueeze(0)  # [B*T, K, d_model]
        cond_z = z.reshape(B * T, -1)
        for layer, film in zip(self.layers, self.films):
            x = film(layer(x), cond_z)
        logits = self.out(x)
        return logits.reshape(B, T, self.cfg.K, self.cfg.vocab)


class QcuteAutoencoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = CausalByteEncoder(cfg)
        self.decoder = NATDecoder(cfg)

    def forward(self, byte_chunks: torch.Tensor):
        # byte_chunks: [B, T, K] long
        B, T, K = byte_chunks.shape
        z = self.encoder(byte_chunks.reshape(B, T * K))
        logits = self.decoder(z)
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab), byte_chunks.reshape(-1))
        acc = (logits.argmax(-1) == byte_chunks).float().mean()
        return loss, acc, z


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def batch_iter(data: torch.Tensor, batch_size: int, seq_chunks: int, K: int, device: str):
    seq_len = seq_chunks * K
    n = (len(data) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([data[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.reshape(batch_size, seq_chunks, K).to(device)


def main():
    p = argparse.ArgumentParser(description="qcute Phase 1: FSQ autoencoder on byte chunks")
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8.gz"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_chunks", type=int, default=32, help="latents per training sequence")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--n_bytes", type=int, default=2_000_000, help="prefix of enwik8 to load")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = QcuteAutoencoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = load_enwik8(args.data, args.n_bytes)
    data_iter = batch_iter(data, args.batch_size, args.seq_chunks, cfg.K, device)

    model.train()
    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        loss, acc, _ = model(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  recon_acc {acc.item()*100:.2f}%")


if __name__ == "__main__":
    main()
