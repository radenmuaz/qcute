"""qcute.qcutelm_vlt2 — fork of qcute.qcutelm_vlt with a different decoder
input design (no shared imports, per this repo's self-contained-module
convention). Encoder (PoolerEncoder) is unchanged from qcutelm_vlt: causal
"pooler", byte_emb only (still NoPE on the encoder side), take the last
timestep, project, BSQ quantize.

Decoder input changed, instead of NoPE + identical-broadcast-z (the
qcutelm_vlt design): position 0's input is ALWAYS the BSQ code embedding
(z_proj(z)) — nothing else, no position embedding added to it. Positions
1..T-1's input are TRAINABLE POSITION EMBEDDINGS (a real [K-1, d_model]
table, one distinct learned vector per position, no z content at all).
So the decoder's input sequence is heterogeneous by construction:
[z_proj(z), pos_emb[0], pos_emb[1], ..., pos_emb[T-2]] — the code occupies
position 0 like a prepended context/summary token, and positions 1..T-1
start from genuine, always-distinguishable position identities (not
broadcast-identical the way qcutelm_vlt's positions were) and must pick up
the code's influence purely through causal attention flowing backward from
position 0 (visible to every later position under the causal mask).

This sidesteps qcutelm_vlt's core mechanism entirely: there, the decoder
was positionally blind in its first layer without zero-KV, because every
position's input (and hence Q/K/V) was identical. Here, positions 1..T-1
are never identical to each other (distinct pos_emb rows) or to position 0
(code, not a position vector) — real per-position differentiation from the
first layer, the same way qcute.qcutelm's redesigned ChunkDecoder gets it
(see that module's pos_emb), just realized as a causal sequence with the
code prepended rather than a fixed-K set with the code broadcast-added
everywhere.

Zero-KV (see ZeroKVCausalSelfAttention) is still present — as an "escape
hatch" (Miller 2023, "Attention Is Off By One") rather than the
load-bearing mechanism it was in qcutelm_vlt — but fixed at 1 sink slot,
not configurable (qcutelm_vlt's Config.n_sink experiment doesn't apply
here; position differentiation no longer depends on it).

Curriculum, replay, forgetting-check, first-byte-accuracy tracking,
point-estimate gating, early-stop: all unchanged from qcutelm_vlt — see
that module's train_curriculum docstring.
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    """Same contract as qcute.qcutelm.Logger — see its docstring."""

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
    """Same contract as qcute.qcutelm.Checkpointer — see its docstring."""

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
    K: int = 4                  # max chunk length (curriculum trains 1..K)
    dq: int = 18                # BSQ code dims — see build_config's entropy-matching comment
    lfq: bool = False           # regress BSQ to plain LFQ (skip L2-normalize) — see qcute.qcutelm.bsq_quantize
    vocab: int = 256
    d_model: int = 128          # shared encoder/decoder width
    n_heads: int = 4
    n_layers_enc: int = 1
    n_layers_dec: int = 1
    mlp_mult: int = 4           # MLP hidden = mlp_mult * d_model


def build_config(dq: int | None, **kwargs) -> Config:
    if dq is None:
        dq = 18
    return Config(dq=dq, **kwargs)


def bsq_quantize(v: torch.Tensor, dq: int, lfq: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical math to qcute.qcutelm.bsq_quantize — duplicated per this
    repo's self-contained-module convention, not an oversight."""
    if lfq:
        z_hat = v + (torch.sign(v) - v).detach()
        targets = (v > 0).float().detach()
        return z_hat, targets
    v_unit = F.normalize(v, dim=-1)
    z_hat = (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)
    targets = (v_unit > 0).float().detach()
    return z_hat, targets


class ZeroKVCausalSelfAttention(nn.Module):
    """Causal self-attention with a single zero key/value pair concatenated
    before SDPA (Miller 2023, "Attention Is Off By One") — an escape hatch
    here, not load-bearing for position differentiation the way it was in
    qcutelm_vlt (see this module's top docstring): the decoder's inputs
    are never identical across positions in this design, so real per-
    position differentiation exists from the first layer regardless."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, hd]
        zero_kv = torch.zeros(B, H, 1, hd, device=x.device, dtype=x.dtype)
        k = torch.cat([zero_kv, k], dim=2)  # [B, H, 1+T, hd]
        v = torch.cat([zero_kv, v], dim=2)
        attn_mask = torch.zeros(T, 1 + T, dtype=torch.bool, device=x.device)
        attn_mask[:, 0] = True  # sink slot always visible to every query
        attn_mask[:, 1:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Pre-norm residual: zero-KV causal self-attention + MLP. Shared by
    both the pooler encoder and unpooler decoder."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ZeroKVCausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PoolerEncoder(nn.Module):
    """bytes[1..T] -> causal transformer (still NoPE, unchanged from
    qcutelm_vlt — only the decoder's design changed in this fork) -> last
    timestep's hidden state -> project -> BSQ."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)  # no pos_emb — NoPE
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers_enc)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.dq)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # chunk: [B, T] long, T in [1, cfg.K] -> z_hat/targets: [B, dq]
        h = self.byte_emb(chunk)
        for block in self.blocks:
            h = block(h)
        pooled = self.ln_f(h[:, -1, :])  # last timestep only
        return bsq_quantize(self.proj(pooled), self.cfg.dq, self.cfg.lfq)


class UnpoolerDecoder(nn.Module):
    """Position 0's input is always z_proj(z) (the BSQ code embedding) —
    never combined with a position embedding. Positions 1..T-1's input are
    trainable position embeddings (pos_emb[0..T-2], a real [K-1, d_model]
    table) — never combined with z. One forward pass, not autoregressive:
    T is just how many total positions to build (1 code slot + T-1
    position-embedding slots), no separate decode loop."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max(cfg.K - 1, 1), cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers_dec)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: [B, dq] -> logits: [B, T, vocab]
        B = z.size(0)
        z_tok = self.z_proj(z).unsqueeze(1)  # [B, 1, d_model] — position 0
        if T > 1:
            pos_toks = self.pos_emb[:T - 1].unsqueeze(0).expand(B, -1, -1)  # [B, T-1, d_model]
            h = torch.cat([z_tok, pos_toks], dim=1)
        else:
            h = z_tok
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


class VLTokenizer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = PoolerEncoder(cfg)
        self.decoder = UnpoolerDecoder(cfg)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # chunk: [B, T] long, T <= cfg.K (this step's curriculum stage)
        B, T = chunk.shape
        z_hat, _ = self.encoder(chunk)
        logits = self.decoder(z_hat, T)
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab), chunk.reshape(-1))
        correct = logits.argmax(-1) == chunk
        acc = correct.float().mean()
        # Position 0 here holds the code directly (not a broadcast/blind
        # signal the way qcutelm_vlt's was) — tracked separately anyway,
        # for a direct before/after comparison against that design.
        first_byte_acc = correct[:, 0].float().mean()
        return loss, {"loss": loss, "recon_acc": acc, "first_byte_acc": first_byte_acc}


def init_decoder_bias_to_unigram(decoder: UnpoolerDecoder, data: torch.Tensor) -> None:
    """Same trick as qcute.qcutelm.init_decoder_bias_to_unigram — see its
    docstring. Free head start, no architecture change."""
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        decoder.head.bias.copy_(log_freq.to(decoder.head.bias.device))


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def sample_chunks(data: torch.Tensor, batch_size: int, T: int, device: str) -> torch.Tensor:
    n = max(1, len(data) - T)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + T] for s in starts]).to(device)


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
def eval_recon_acc(model: VLTokenizer, data: torch.Tensor, T: int, batch_size: int, n_batches: int, device: str) -> tuple[float, float]:
    """Returns (whole_chunk_acc, first_byte_acc)."""
    model.eval()
    correct = total = 0
    first_correct = first_total = 0
    for _ in range(n_batches):
        batch = sample_chunks(data, batch_size, T, device)
        z_hat, _ = model.encoder(batch)
        logits = model.decoder(z_hat, T)
        match = logits.argmax(-1) == batch
        correct += match.float().sum().item()
        total += batch.numel()
        first_correct += match[:, 0].float().sum().item()
        first_total += match.size(0)
    model.train()
    return correct / total, first_correct / first_total


def train_curriculum(model: VLTokenizer, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    """Same curriculum/replay/forgetting-check/point-estimate-gating logic
    as qcute.qcutelm_vlt.train_curriculum — see that module's docstring."""
    cfg = model.cfg
    init_decoder_bias_to_unigram(model.decoder, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    T = cfg.K if args.no_curriculum else 1
    stage_step = 0
    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt2", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        is_eval_step = step % args.eval_every == 0 or step == args.steps
        T_step = T
        if not args.no_curriculum and not is_eval_step and T > 1 and args.replay_frac > 0 and torch.rand(()).item() < args.replay_frac:
            T_step = int(torch.randint(1, T + 1, (1,)).item())
        batch = sample_chunks(train_data, args.batch_size, T_step, device)
        loss, metrics = model(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        acc = metrics["recon_acc"].item()
        first_acc = metrics["first_byte_acc"].item()
        pbar.set_postfix(lr=f"{lr:.2e}", T=T, T_step=T_step, loss=f"{loss.item():.4f}",
                          train_recon_acc=f"{acc*100:.2f}%", first_byte_acc=f"{first_acc*100:.2f}%")
        stage_step += 1

        if is_eval_step:
            val_acc, val_first_acc = eval_recon_acc(model, val_data, T, args.batch_size, args.eval_batches, device)
            # forgetting-check across earlier stages is meaningless without curriculum
            # (no earlier stages were ever trained) — skip the extra eval passes entirely.
            prior_accs = {} if args.no_curriculum else {
                t: eval_recon_acc(model, val_data, t, args.batch_size, args.eval_batches, device)[0]
                for t in range(1, T)
            }
            prior_str = "  ".join(f"T{t}_acc={a*100:.2f}%" for t, a in prior_accs.items())
            log(f"{pbar}  T={T}  val_acc={val_acc*100:.2f}%  val_first_byte_acc={val_first_acc*100:.2f}%" + (f"  {prior_str}" if prior_str else ""),
                step=step, T=T, train_recon_acc=acc, first_byte_acc=first_acc, val_recon_acc=val_acc,
                val_first_byte_acc=val_first_acc, **{f"T{t}_val_acc": a for t, a in prior_accs.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "T": T, "val_recon_acc": val_acc}, 1.0 - val_acc)
            if not args.no_curriculum and T < cfg.K and (
                acc >= args.curriculum_target_acc or stage_step >= args.curriculum_max_steps_per_stage
            ):
                reason = "target_acc" if acc >= args.curriculum_target_acc else "max_steps_per_stage"
                T += 1
                stage_step = 0
                log(f"curriculum advance ({reason}): T={T}", step=step, T=T)
            elif T == cfg.K and acc >= args.curriculum_target_acc:
                log(f"target reached at final stage T={T}: train_recon_acc {acc*100:.2f}% >= "
                    f"{args.curriculum_target_acc*100:.1f}% at step {step}", step=step, T=T)
                return


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Variable-length causal tokenizer, code-prefix decoder (fork of qcute.qcutelm_vlt)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--dq", type=int, default=None)
    p.add_argument("--lfq", action="store_true")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers_enc", type=int, default=1)
    p.add_argument("--n_layers_dec", type=int, default=1)
    p.add_argument("--mlp_mult", type=int, default=4)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)

    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--replay_frac", type=float, default=0.2)
    p.add_argument("--curriculum_target_acc", type=float, default=0.95)
    p.add_argument("--curriculum_max_steps_per_stage", type=int, default=2000)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config(
        args.dq, K=args.K, lfq=args.lfq, d_model=args.d_model, n_heads=args.n_heads,
        n_layers_enc=args.n_layers_enc, n_layers_dec=args.n_layers_dec, mlp_mult=args.mlp_mult,
    )
    model = VLTokenizer(cfg).to(device)
    n_enc = sum(p_.numel() for p_ in model.encoder.parameters())
    n_dec = sum(p_.numel() for p_ in model.decoder.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt2_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} dq={cfg.dq} params={(n_enc+n_dec)/1e6:.3f}M (encoder={n_enc/1e6:.3f}M decoder={n_dec/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train_curriculum(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
