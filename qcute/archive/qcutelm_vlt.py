"""qcute.qcutelm_vlt — variable-length causal tokenizer, forked from
qcute.qcutelm to try a structurally different encoder/decoder design (no
shared imports, per this repo's self-contained-module convention — see
qcute/qcutelm.py's own docstring for why).

Encoder = causal-transformer "pooler": byte_emb (no positional embedding
at all — NoPE, see below) -> N causal self-attention+MLP layers over
however many bytes (1..K) this training step's curriculum stage uses ->
take ONLY the last timestep's hidden state (causal attention means it has,
in principle, attended to every earlier position, so it's a genuine
running summary of the whole 1..T prefix, not just the last byte) ->
project -> BSQ quantize. This makes the encoder's *chunk length* a
training-time choice, not a fixed architectural constant the way
qcute.qcutelm's ChunkEncoder's K is.

Decoder = causal-transformer "unpooler", NOT autoregressive: the single
code z is projected and *broadcast identically* to T timesteps (T = the
target chunk length, communicated implicitly by how many positions get
broadcast to, not encoded anywhere explicit), then N causal self-attention
+MLP layers, then a per-position vocab head — one forward pass produces
all T bytes at once, mirroring qcute.qcutelm's post-refactor code-only
ChunkDecoder (see docs/status.md), just built from a causal transformer
instead of a small fixed-K mixer.

NoPE (no positional embeddings anywhere, encoder or decoder): order info
comes only from the causal mask. This is well-precedented for causal
transformers generally (Haviv et al. 2022, "Transformer Language Models
without Positional Encodings Still Learn Positional Information" — the
causal mask alone lets position be recovered implicitly), but it is
*load-bearing*, not just a nice-to-have, for this decoder specifically:
every timestep's *input* is the identical broadcast z, so query/key/value
projections are identical at every position too, and real-key attention
*scores* end up identical across position pairs as a result (q_t . k_s is
the same scalar for every t, s). Under plain causal softmax over n_t
identical-valued, identical-scored real keys, the weights are uniform and
sum to 1, so the output is just the mean of n_t identical values — i.e.
IDENTICAL across every timestep, regardless of position. Zero-KV fixes
this: concatenating a zero key (always scores 0, i.e. contributes a fixed
exp(0)=1 term to softmax's denominator) and zero value (contributes
nothing to the weighted sum) means each position's real-key weight becomes
n_t*exp(c) / (1 + n_t*exp(c)) for some real-key score c — a genuine
function of n_t (= the causal step index), which *does* vary by position.
That scalar-valued per-position signal is exactly what depth+nonlinearity
(the post-attention MLP) then has room to turn into rich, non-scalar-
multiple per-position differentiation. Without zero-KV, this specific
identical-broadcast-input decoder design would be positionally blind in
its very first layer, no matter how deep the stack gets afterward.

Curriculum: train_curriculum() starts chunk length T=1 (trivial — see
qcute.qcutelm's K=1 sanity check, docs/status.md) and advances T by 1 once
the current stage's val recon_acc clears --curriculum_target_acc (or a
max-steps-per-stage fallback fires), up to K. Variable T is why NoPE
matters practically, too: an absolute positional embedding table would
need to be sized for the largest K up front; a purely causal-mask-driven
scheme has no such constraint.
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
    d_model: int = 128          # shared encoder/decoder width (no separate d_byte/d_enc/d_dec split here)
    n_heads: int = 4
    n_layers_enc: int = 1
    n_layers_dec: int = 1
    mlp_mult: int = 4           # MLP hidden = mlp_mult * d_model
    # Number of zero-KV sink slots (see ZeroKVCausalSelfAttention) — 1
    # (default) is the original design. For the decoder's identical-
    # broadcast-input case specifically, the position-differentiation
    # signal derived from "how many real keys are visible" (n_t) is only
    # n_sink-dimensional in the first layer (n_sink=1 -> a single scalar
    # per position); >1 gives a richer basis for the post-attention MLP to
    # expand on, without reintroducing any actual position embedding —
    # every query still sees the same n_sink sink keys, only the *count*
    # of visible real keys varies by position, same mechanism as n_sink=1,
    # just higher-rank.
    n_sink: int = 1


def build_config(dq: int | None, **kwargs) -> Config:
    # Same entropy-matching reasoning as qcute.qcutelm.build_config: at
    # K=4, a chunk's raw entropy ceiling is log2(256^4)=32 bits, English
    # text's actual entropy is ~4.5-5 bits/byte (~18-20 bits for 4 bytes)
    # — dq=18 targets that real redundancy, not the uniform-byte ceiling.
    if dq is None:
        dq = 18
    return Config(dq=dq, **kwargs)


def bsq_quantize(v: torch.Tensor, dq: int, lfq: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical math to qcute.qcutelm.bsq_quantize — duplicated per this
    repo's self-contained-module convention, not an oversight. See that
    function's docstring for the full STE/normalize rationale."""
    if lfq:
        z_hat = v + (torch.sign(v) - v).detach()
        targets = (v > 0).float().detach()
        return z_hat, targets
    v_unit = F.normalize(v, dim=-1)
    z_hat = (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)
    targets = (v_unit > 0).float().detach()
    return z_hat, targets


class ZeroKVCausalSelfAttention(nn.Module):
    """Causal self-attention with n_sink zero key/value pairs concatenated
    before SDPA — see this module's top docstring for why this is
    load-bearing (not just a regularizer) for the decoder's identical-
    broadcast-input design specifically, and a generic "escape hatch"
    (Miller 2023, "Attention Is Off By One") for the encoder. No
    positional embedding anywhere (NoPE) — order info comes only from the
    causal mask (+ zero-KV's effect on it, for the decoder). n_sink=1 is
    the original design (a single scalar per-position differentiation
    signal in the first layer, for the decoder's identical-broadcast-input
    case); n_sink>1 gives a richer, higher-rank basis — see Config.n_sink."""

    def __init__(self, d_model: int, n_heads: int, n_sink: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_sink = n_sink
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd, S = self.n_heads, self.head_dim, self.n_sink
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, hd]
        zero_kv = torch.zeros(B, H, S, hd, device=x.device, dtype=x.dtype)
        k = torch.cat([zero_kv, k], dim=2)  # [B, H, S+T, hd]
        v = torch.cat([zero_kv, v], dim=2)
        attn_mask = torch.zeros(T, S + T, dtype=torch.bool, device=x.device)
        attn_mask[:, :S] = True  # sink slots always visible to every query
        attn_mask[:, S:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Pre-norm residual: zero-KV causal self-attention + MLP. Shared by
    both the pooler encoder and unpooler decoder."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ZeroKVCausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_sink)
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
    """bytes[1..T] -> causal transformer -> last timestep's hidden state
    (a running causal summary of the whole prefix) -> project -> BSQ."""

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
    """z -> broadcast to T timesteps (identical input every position) ->
    causal transformer (zero-KV is what actually differentiates positions
    here — see module docstring) -> per-position vocab logits. One-shot,
    not autoregressive: T is just how many positions the caller broadcasts
    z to, there's no separate decode loop."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers_dec)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: [B, dq] -> logits: [B, T, vocab]
        h = self.z_proj(z).unsqueeze(1).expand(-1, T, -1)
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
        # Position 0 has the fewest causally-visible zero-KV sink keys
        # (n_t=1, just itself — see ZeroKVCausalSelfAttention's docstring),
        # so it's the position with the least differentiation signal in
        # the first layer and plausibly the hardest to get right — tracked
        # separately from the whole-chunk average for any T.
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
    """Random contiguous T-byte windows — simpler than qcute.qcutelm's
    batch_iter since there's no seq_chunks structure here, just one
    T-length window per example."""
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
    """Returns (whole_chunk_acc, first_byte_acc) — see VLTokenizer.forward's
    docstring for why position 0 is tracked separately."""
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
    """Curriculum over chunk length T: start at T=1 (trivial — see
    qcute.qcutelm's K=1 sanity check), advance T by 1 once the current
    stage's val recon_acc clears --curriculum_target_acc, or once
    --curriculum_max_steps_per_stage fires (fallback so a stuck stage
    doesn't stall the whole run), up to cfg.K. --no_curriculum trains at
    T=cfg.K throughout, for a direct ablation.

    --replay_frac > 0: once T > 1, each training step has that probability
    of sampling a *random* T' in [1, T] instead of the current stage's T —
    rehearsal against catastrophic forgetting of earlier (shorter-chunk)
    stages, since the same shared encoder/decoder weights serve every T
    and nothing otherwise stops later stages from overwriting what earlier
    ones learned. Eval (and the curriculum-advance decision) always stays
    at the current stage's T regardless — replay only touches training
    batches, not the signal that decides when to advance."""
    cfg = model.cfg
    init_decoder_bias_to_unigram(model.decoder, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    T = cfg.K if args.no_curriculum else 1
    stage_step = 0
    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        # No replay on eval-check steps: the curriculum-advance/early-stop
        # decision below reads `acc` from *this* step's batch, and it must
        # reflect the current stage's T, not a lucky replay batch at some
        # easier T' < T scoring high by chance (a real bug this caught:
        # a T'=1 replay batch hit 100% and falsely triggered "target
        # reached" at T=4 while actual T=4 accuracy was only ~51%).
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
            # Forgetting check: also eval every earlier stage T'<T, not
            # just the current one — replay_frac only rehearses during
            # training, it doesn't guarantee earlier stages stay solved;
            # this is the only place that actually measures it.
            prior_accs = {
                t: eval_recon_acc(model, val_data, t, args.batch_size, args.eval_batches, device)[0]
                for t in range(1, T)
            }
            prior_str = "  ".join(f"T{t}_acc={a*100:.2f}%" for t, a in prior_accs.items())
            log(f"{pbar}  T={T}  val_acc={val_acc*100:.2f}%  val_first_byte_acc={val_first_acc*100:.2f}%" + (f"  {prior_str}" if prior_str else ""),
                step=step, T=T, train_recon_acc=acc, first_byte_acc=first_acc, val_recon_acc=val_acc,
                val_first_byte_acc=val_first_acc, **{f"T{t}_val_acc": a for t, a in prior_accs.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "T": T, "val_recon_acc": val_acc}, 1.0 - val_acc)
            # Curriculum gates on the current step's own training-batch
            # accuracy (a point estimate, not a multi-batch average) — the
            # curriculum advances one way or another regardless (target hit
            # or stage_step fallback), so a noisier but free signal is fine
            # here; no separate eval pass over train_data.
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

    p = argparse.ArgumentParser(description="Variable-length causal tokenizer (fork of qcute.qcutelm)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--dq", type=int, default=None)
    p.add_argument("--lfq", action="store_true")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers_enc", type=int, default=1)
    p.add_argument("--n_layers_dec", type=int, default=1)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--n_sink", type=int, default=1)

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

    p.add_argument("--no_curriculum", action="store_true", help="train at T=K throughout, instead of the 1->K curriculum")
    p.add_argument("--replay_frac", type=float, default=0.2, help="probability of sampling a random T'<=T (rehearsal) instead of the current stage's T, once T>1")
    p.add_argument("--curriculum_target_acc", type=float, default=0.95, help="val recon_acc to clear before advancing T")
    p.add_argument("--curriculum_max_steps_per_stage", type=int, default=2000, help="advance T anyway after this many steps, even if target_acc isn't hit")

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
        n_sink=args.n_sink,
    )
    model = VLTokenizer(cfg).to(device)
    n_enc = sum(p_.numel() for p_ in model.encoder.parameters())
    n_dec = sum(p_.numel() for p_ in model.decoder.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} dq={cfg.dq} params={(n_enc+n_dec)/1e6:.3f}M (encoder={n_enc/1e6:.3f}M decoder={n_dec/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train_curriculum(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
