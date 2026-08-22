"""qcute.bytelm_queryvec — fork of qcute.bytelm preserving the query-vec parallel-block-decode
idea (originally built/explored inside qcute_zero, pruned from there 2026-08-22 in favor of
regular MTP heads reading the same per-step hidden state — see qcute_zero's own module docstring
and docs/status.md). Kept as its own standalone testbed on the simplest possible trunk (plain
byte-level, no fuse stages / no codes), since the flaw diagnosed in qcute_zero (one query_vec slot
consumes one full attention-layer-stack pass, unlike MTP's cheap "reuse one h, many linear heads"
density) is a property of the query_vec mechanism itself, not of qcute_zero's architecture —
worth keeping isolated and testable independent of the fuse-stage plumbing.

Design (adapted from qcute_zero's Config.parallel_decode / forward()'s parallel-decode block,
Ks[0]-less since bytelm has no code hierarchy): a single shared trained vector `query_vec`
(nn.Parameter(D)) stands in for "no real previous-byte hidden state available here." Training
samples `query_vec_n_blocks` independent clusters of `query_vec_cluster_len` contiguous slots per
step, each at a random (unaligned) start position; every slot cross-attends into the REAL trunk's
own per-layer K/V (computed once per batch, shared across all sampled clusters), masked to only
the real positions strictly before its own cluster's start, plus block-causal self-attention among
its own cluster's earlier slots — never any other cluster's slots, never a real position at or
after its own start. This is denser context than qcute_zero's version (raw per-position byte K/V,
not compressed codes) since there's no fuse-stage compression step here.

uv run python -m qcute.bytelm_queryvec.bytelm_queryvec --config configs/bytelm_queryvec/xs_overfit10k.py
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


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


@dataclass
class LMConfig:
    vocab: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    context: int = 256
    mlp_mult: int = 4
    rope_base: float = 10000.0
    mtp_heads: int = 4
    query_vec_cluster_len: int = 2     # C: contiguous query_vec slots drafted per cluster
    query_vec_n_blocks: int = 4        # independently-sampled clusters trained per step
    query_vec_weight: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS: dict[str, LMConfig] = {
    "xs": LMConfig(d_model=256, n_layers=4, n_heads=4, context=256, mtp_heads=4),
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    """Like rope_cos_sin but for arbitrary (possibly batched) position ids -- position_ids can be
    (T,) (shared across batch) or (Bv, T) (one set of positions per batch row, needed since every
    sampled cluster has its own random start position)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, head_dim]; cos/sin: [T, head_dim] (shared) or [B, T, head_dim] (per-row)
    if cos.dim() == 2:
        return x * cos[None, None] + rotate_half(x) * sin[None, None]
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def qkv_split(self, x: torch.Tensor):
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv_split(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)

    def forward_step(self, x_new, cos_new, sin_new, cache_k, cache_v):
        B, _, D = x_new.shape
        q, k, v = self.qkv_split(x_new)
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        y = F.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, 1, D)
        return self.out(y), new_k, new_v

    def forward_query_vec_step(self, x_q, cos_q, sin_q, k_real, v_real, allow_real, self_mask):
        """One layer's worth of query-slot attention: combined KV = real trunk's own per-layer
        K/V (masked per-row by `allow_real`, i.e. only positions strictly before this row's
        cluster start) ++ this cluster's own earlier query slots (block-causal via `self_mask`).
        x_q: (Bv, C, D). k_real/v_real: (Bv, H, L, hd), already expanded to the Bv rows.
        allow_real: (Bv, L) bool. self_mask: (1, 1, C, C) bool, block-causal among C slots."""
        Bv, C, D = x_q.shape
        q, k_q, v_q = self.qkv_split(x_q)
        q, k_q = apply_rope(q, cos_q, sin_q), apply_rope(k_q, cos_q, sin_q)
        k_cat = torch.cat([k_real, k_q], dim=2)
        v_cat = torch.cat([v_real, v_q], dim=2)
        mask_real = allow_real.view(Bv, 1, 1, -1).expand(Bv, 1, C, -1)
        mask_query = self_mask.expand(Bv, 1, C, C)
        mask = torch.cat([mask_real, mask_query], dim=-1)
        y = F.scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=mask)
        y = y.transpose(1, 2).reshape(Bv, C, D)
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

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_step(self, x_new, cos_new, sin_new, cache_k, cache_v):
        attn_out, new_k, new_v = self.attn.forward_step(self.ln1(x_new), cos_new, sin_new, cache_k, cache_v)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_k, new_v


class ByteLMQueryVec(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab, bias=False) for _ in range(cfg.mtp_heads)]
        )
        self.heads[0].weight = self.tok_emb.weight
        self.query_vec = nn.Parameter(torch.zeros(cfg.d_model))
        self.apply(self._init_weights)
        for block in self.blocks:
            for proj in (block.attn.out, block.mlp.down):
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
        x = self.ln_f(x)
        return torch.stack([head(x) for head in self.heads], dim=0)

    def forward_query_vec_loss(self, tokens: torch.Tensor):
        """Trunk forward (real path, ordinary causal), collecting each layer's own post-RoPE K/V
        so query_vec clusters can cross-attend into them -- then query_vec_n_blocks independent
        clusters of query_vec_cluster_len slots each, folded into a Bv=B*n_blocks batch axis
        (qcute_zero's "Option B"), predict their own cluster's bytes blind (no real hidden state,
        only query_vec + masked access to strictly-prior real K/V + own earlier cluster slots)."""
        cfg = self.cfg
        B, L = tokens.shape
        D, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        C = cfg.query_vec_cluster_len
        nb = max(1, cfg.query_vec_n_blocks)
        device = tokens.device
        assert L >= C + 1, "context too short for query_vec_cluster_len"

        cos, sin = rope_cos_sin(L, hd, cfg.rope_base, device)
        h = self.tok_emb(tokens)
        real_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            xn = block.ln1(h)
            q, k, v = block.attn.qkv_split(xn)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, L, D)
            h = h + block.attn.out(y)
            h = h + block.mlp(block.ln2(h))
            real_kv.append((k, v))   # (B, H, L, hd), post-RoPE

        m_list = torch.randint(1, L - C + 1, (nb,), device=device)     # m>=1: some prior context exists
        clamp_list = m_list - 1
        offsets = torch.arange(C, device=device)
        slot_pos_2d = m_list.view(nb, 1) + offsets.view(1, C)

        Bv = B * nb
        orig_idx = torch.arange(B, device=device).view(B, 1).expand(B, nb).reshape(Bv)
        slot_pos_v = slot_pos_2d.unsqueeze(0).expand(B, nb, C).reshape(Bv, C)
        clamp_v = clamp_list.view(1, nb).expand(B, nb).reshape(Bv)
        targets = tokens[orig_idx.view(Bv, 1), slot_pos_v]

        cos_q, sin_q = rope_cos_sin_for_positions(slot_pos_v, hd, cfg.rope_base, device)
        self_mask = (offsets.view(-1, 1) >= offsets.view(1, -1)).view(1, 1, C, C)   # block-causal within cluster
        allow_real = torch.arange(L, device=device).view(1, -1) <= clamp_v.view(-1, 1)  # (Bv, L)

        xq = self.query_vec.view(1, 1, D).expand(Bv, C, D)
        for li, block in enumerate(self.blocks):
            k_real, v_real = real_kv[li]
            k_real_v = k_real.unsqueeze(1).expand(B, nb, H, L, hd).reshape(Bv, H, L, hd)
            v_real_v = v_real.unsqueeze(1).expand(B, nb, H, L, hd).reshape(Bv, H, L, hd)
            xn = block.ln1(xq)
            attn_out = block.attn.forward_query_vec_step(xn, cos_q, sin_q, k_real_v, v_real_v, allow_real, self_mask)
            xq = xq + attn_out
            xq = xq + block.mlp(block.ln2(xq))
        xq = self.ln_f(xq)
        logits = F.linear(xq, self.tok_emb.weight)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), targets.reshape(-1))
        acc = (logits.argmax(-1) == targets).float().mean()
        return loss, acc

    @torch.no_grad()
    def generate_blockwise(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Free-tier block-parallel decode: query_vec_cluster_len bytes per step via the trained
        query_vec, reusing one fresh full-recompute real-trunk pass per step (not incrementally
        cached -- same non-incremental design qcute_zero's own generate_blockwise had)."""
        cfg = self.cfg
        C = cfg.query_vec_cluster_len
        D, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        target_len = prompt_bytes.shape[1] + n_new_bytes

        while all_bytes.shape[1] < target_len:
            m = all_bytes.shape[1]
            block_size = min(C, target_len - m)
            L = m
            B = all_bytes.shape[0]
            cos, sin = rope_cos_sin(L, hd, cfg.rope_base, device)
            h = self.tok_emb(all_bytes)
            real_kv = []
            for block in self.blocks:
                xn = block.ln1(h)
                q, k, v = block.attn.qkv_split(xn)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y = y.transpose(1, 2).reshape(B, L, D)
                h = h + block.attn.out(y)
                h = h + block.mlp(block.ln2(h))
                real_kv.append((k, v))

            slot_pos = torch.arange(m, m + block_size, device=device)
            cos_q, sin_q = rope_cos_sin_for_positions(slot_pos, hd, cfg.rope_base, device)
            offsets = torch.arange(block_size, device=device)
            self_mask = (offsets.view(-1, 1) >= offsets.view(1, -1)).view(1, 1, block_size, block_size)
            allow_real = torch.ones(B, L, dtype=torch.bool, device=device)  # every real position < m is visible

            xq = self.query_vec.view(1, 1, D).expand(B, block_size, D)
            for li, block in enumerate(self.blocks):
                k_real, v_real = real_kv[li]
                xn = block.ln1(xq)
                attn_out = block.attn.forward_query_vec_step(xn, cos_q, sin_q, k_real, v_real, allow_real, self_mask)
                xq = xq + attn_out
                xq = xq + block.mlp(block.ln2(xq))
            xq = self.ln_f(xq)
            logits = F.linear(xq, self.tok_emb.weight)
            new_bytes = logits.argmax(-1)
            all_bytes = torch.cat([all_bytes, new_bytes], dim=1)

        if was_training:
            self.train()
        return all_bytes[0]


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return nats / math.log(2)


def mtp_loss(logits: torch.Tensor, tokens: torch.Tensor, context: int):
    n_heads = logits.size(0)
    losses = []
    for i in range(n_heads):
        targets_i = tokens[:, i + 1 : i + 1 + context]
        losses.append(F.cross_entropy(logits[i].reshape(-1, logits.size(-1)), targets_i.reshape(-1)))
    losses = torch.stack(losses)
    head0_bpb = losses[0] / math.log(2)
    return losses.mean(), head0_bpb


def batch_iter(data: torch.Tensor, batch_size: int, context: int, n_heads: int, device: str):
    seq_len = context + n_heads
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
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="bytelm + query_vec parallel-decode testbed", parents=[pre])
    p.add_argument("--preset", choices=list(PRESETS), default="xs")
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--mtp_heads", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--query_vec_cluster_len", type=int, default=None)
    p.add_argument("--query_vec_n_blocks", type=int, default=None)
    p.add_argument("--query_vec_weight", type=float, default=None)
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=5)
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PRESETS[args.preset]
    if args.context is not None:
        cfg.context = args.context
    if args.mtp_heads is not None:
        cfg.mtp_heads = args.mtp_heads
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.query_vec_cluster_len is not None:
        cfg.query_vec_cluster_len = args.query_vec_cluster_len
    if args.query_vec_n_blocks is not None:
        cfg.query_vec_n_blocks = args.query_vec_n_blocks
    if args.query_vec_weight is not None:
        cfg.query_vec_weight = args.query_vec_weight
    model = ByteLMQueryVec(cfg).to(device)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"bytelm_queryvec_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"params={count_params(model)/1e6:.3f}M  device={device}  context={cfg.context}  "
        f"mtp_heads={cfg.mtp_heads}  query_vec_cluster_len={cfg.query_vec_cluster_len}  "
        f"query_vec_n_blocks={cfg.query_vec_n_blocks}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    val_iter = batch_iter(val_data, args.batch_size, cfg.context, cfg.mtp_heads, device)
    train_iter = batch_iter(train_data, args.batch_size, cfg.context, cfg.mtp_heads, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
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
        qv_loss, qv_acc = model.forward_query_vec_loss(batch[:, : cfg.context])
        total_loss = loss + cfg.query_vec_weight * qv_loss

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        pbar.set_postfix(lr=f"{lr:.2e}", bpb=f"{head0_bpb.item():.4f}", qv_acc=f"{qv_acc.item():.4f}")
        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, mtp_loss=loss.item(), bpb=head0_bpb.item(),
                query_vec_loss=qv_loss.item(), query_vec_acc=qv_acc.item())
        if step % args.eval_every == 0 or step == args.steps:
            val_bpb = eval_bpb(model, val_iter, cfg.context, args.eval_batches)
            log(f"step {step:5d}  val_bpb {val_bpb:.4f}", step=step, val_bpb=val_bpb)
            checkpointer.step(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg), "val_bpb": val_bpb},
                val_bpb,
            )
    log(f"checkpoints: best={checkpointer.best_path} (val_bpb {checkpointer.best_metric:.4f})  last={checkpointer.last_path}")


if __name__ == "__main__":
    main()
