"""qcute.qcutelm — end-to-end latent autoregression: encoder + bottleneck +
LM + decoder, trained jointly, generation loop included.

Simplified vs. the handover doc's recommended architecture on purpose:
encoder and decoder are plain MLPs over a fixed K-byte chunk (non-streaming,
no causal-SSM byte-level context — handover §1.3/§1.4 calls this out as the
"naive" chunk-local design with boundary artifacts, but it's the fastest path
to a working end-to-end system). The LM ↔ tokenizer interface is Option A,
pure latent autoregression (handover §2.1): the LM predicts the next
bottleneck code and that sampled code is fed back directly, no re-encoding.

Bottleneck is FSQ (handover §1.2.2) or BSQ (§1.2.3), selectable via
--bottleneck; training loss is reconstruction CE (decoder) + next-code
prediction CE/BCE (LM), per the joint-training pseudocode in handover §7.1.

Deliberately monolithic and independent of qcute.bytelm (no shared imports) —
see docs/architecture.md for why. The old streaming-causal-encoder Phase 1
autoencoder this replaces is archived at
archive/tokenizer_phase1_standalone_autoencoder.py.

Duplication vs. qcute.bytelm is a deliberate split, not oversight: pure
infra with zero model-specific coupling (Logger, Checkpointer,
load_config_module, load_enwik8, split_train_val, lr_at, the RoPE math
rope_cos_sin/rotate_half/apply_rope) is a real "should share into a
qcute/utils.py" candidate — not yet done, tracked as a pending decision.
Architecture-bearing code (CausalSelfAttention/Block/MLP, generate(),
eval_metrics) stays duplicated on purpose: it *is* the model each file
exists to let you read and debug standalone, and this module's generation
is a structurally different shape from bytelm's byte-AR loop (encode→code→
decode), so a shared abstraction there wouldn't be honest.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, field
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

    Writes into its own run directory (logs/<run_name>/), matching the
    Checkpointer's checkpoints/<run_name>/ — keeps everything for one run
    findable by run_name alone:
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


@dataclass
class Config:
    # K=4, not the handover doc's K=8 default: on this tiny 500KB corpus,
    # qcute.bpelm (the fair comparison point) only reaches ~3-4 bytes/token
    # before larger vocabs start memorizing phrases (see scripts/train_bpe.py)
    # — targeting K=8 here would be a bandwidth mismatch at this corpus
    # scale. Revisit at full-enwik8 scale, where K=8 becomes achievable
    # (see qcute/bytelm.py's PRESETS comment for the same reasoning).
    K: int = 4                   # bytes per chunk
    bottleneck: str = "fsq"     # "fsq" or "bsq"
    dq: int = 6                 # bottleneck dims (FSQ default; BSQ default is 18, see build_config)
    L: int = 8                  # FSQ levels per dim (unused for BSQ)
    d_byte: int = 64            # byte embedding dim (encoder/decoder)
    d_enc: int = 256            # encoder MLP width
    d_dec: int = 256            # decoder MLP width
    d_model: int = 256          # LM width
    n_layers: int = 4
    n_heads: int = 4
    rope_base: float = 10000.0
    vocab: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def build_config(bottleneck: str, dq: int | None, **kwargs) -> Config:
    if dq is None:
        dq = 18 if bottleneck == "bsq" else 6
    return Config(bottleneck=bottleneck, dq=dq, **kwargs)


# ---------------------------------------------------------------------------
# Bottlenecks (handover §1.2.2 FSQ, §1.2.3 BSQ)
# ---------------------------------------------------------------------------


class FSQ(nn.Module):
    """Finite scalar quantization. Targets are level indices in [0, L)."""

    def __init__(self, d_in: int, dq: int, L: int):
        super().__init__()
        self.dq, self.L = dq, L
        self.proj = nn.Linear(d_in, dq)

    def forward(self, u: torch.Tensor):
        bound = (self.L - 1) / 2
        z_bounded = bound * torch.tanh(self.proj(u))
        z_rounded = torch.round(z_bounded)
        z_hat = z_bounded + (z_rounded - z_bounded).detach()      # STE
        targets = (z_rounded + bound).long().clamp(0, self.L - 1)  # [.., dq] in [0, L)
        return z_hat, targets

    @staticmethod
    def levels_to_z(levels: torch.Tensor, L: int) -> torch.Tensor:
        return levels.float() - (L - 1) / 2


class BSQ(nn.Module):
    """Binary spherical quantization. Targets are sign bits in {0, 1}."""

    def __init__(self, d_in: int, dq: int):
        super().__init__()
        self.dq = dq
        self.proj = nn.Linear(d_in, dq)

    def forward(self, u: torch.Tensor):
        v = self.proj(u)
        v_unit = F.normalize(v, dim=-1)
        z_hat = (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(self.dq)  # STE
        targets = (v_unit > 0).float()
        return z_hat, targets

    @staticmethod
    def bits_to_z(bits: torch.Tensor, dq: int) -> torch.Tensor:
        return (2 * bits - 1) / math.sqrt(dq)


def make_bottleneck(cfg: Config) -> nn.Module:
    if cfg.bottleneck == "fsq":
        return FSQ(cfg.d_enc, cfg.dq, cfg.L)
    if cfg.bottleneck == "bsq":
        return BSQ(cfg.d_enc, cfg.dq)
    raise ValueError(f"unknown bottleneck: {cfg.bottleneck}")


# ---------------------------------------------------------------------------
# Encoder / decoder — plain MLPs over one chunk, non-streaming (see module docstring)
# ---------------------------------------------------------------------------


class ChunkEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_byte)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.K * cfg.d_byte, cfg.d_enc), nn.GELU(), nn.Linear(cfg.d_enc, cfg.d_enc)
        )
        self.bottleneck = make_bottleneck(cfg)

    def forward(self, chunk: torch.Tensor):
        # chunk: [N, K] long -> z_hat: [N, dq], targets: [N, dq]
        N = chunk.size(0)
        u = self.mlp(self.byte_emb(chunk).reshape(N, -1))
        return self.bottleneck(u)


class ChunkDecoder(nn.Module):
    """Memoryless: given z alone, produce K byte logits (handover §1.4.2/1.4.3a)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dq, cfg.d_dec), nn.GELU(), nn.Linear(cfg.d_dec, cfg.K * cfg.vocab)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, dq] -> logits [N, K, vocab]
        N = z.size(0)
        return self.mlp(z).reshape(N, self.cfg.K, self.cfg.vocab)


# ---------------------------------------------------------------------------
# Latent LM — causal transformer over the code sequence (interface Option A, §2.1)
# ---------------------------------------------------------------------------


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
    def __init__(self, cfg: Config):
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class LatentLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.bottleneck == "fsq":
            self.head = nn.Linear(cfg.d_model, cfg.dq * cfg.L)
        else:
            self.head = nn.Linear(cfg.d_model, cfg.dq)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, dq] -> logits: FSQ [B, T, dq, L] or BSQ [B, T, dq]
        B, T, _ = z.shape
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, z.device)
        x = self.in_proj(z)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.ln_f(x))
        if self.cfg.bottleneck == "fsq":
            logits = logits.reshape(B, T, self.cfg.dq, self.cfg.L)
        return logits


# ---------------------------------------------------------------------------
# End-to-end model
# ---------------------------------------------------------------------------


class QCuteLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ChunkEncoder(cfg)
        self.decoder = ChunkDecoder(cfg)
        self.lm = LatentLM(cfg)

    def forward(self, byte_chunks: torch.Tensor):
        # byte_chunks: [B, T, K] long
        B, T, K = byte_chunks.shape
        z_hat, targets = self.encoder(byte_chunks.reshape(B * T, K))
        z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)

        # reconstruction (every chunk decoded from its own code)
        rec_logits = self.decoder(z_hat.reshape(B * T, -1)).reshape(B, T, K, self.cfg.vocab)
        rec_loss = F.cross_entropy(rec_logits.reshape(-1, self.cfg.vocab), byte_chunks.reshape(-1))
        recon_acc = (rec_logits.argmax(-1) == byte_chunks).float().mean()

        # next-code prediction (teacher-forced, shifted by one chunk)
        pred_logits = self.lm(z_hat)[:, :-1]        # predict z_{t+1} from z_{<=t}
        pred_targets = targets[:, 1:]
        if self.cfg.bottleneck == "fsq":
            pred_loss = F.cross_entropy(
                pred_logits.reshape(-1, self.cfg.L), pred_targets.reshape(-1)
            )
            latent_acc = (pred_logits.argmax(-1) == pred_targets).float().mean()
        else:
            pred_loss = F.binary_cross_entropy_with_logits(pred_logits, pred_targets)
            latent_acc = ((pred_logits > 0).float() == pred_targets).float().mean()

        loss = rec_loss + pred_loss

        # Exact BPB (handover §1.7): encoder is deterministic (FSQ/BSQ are point
        # masses), so the ELBO collapses to decoder-NLL + LM code-NLL, tight when
        # reconstruction is near-lossless. rec_loss is already nats/byte (mean CE
        # over all K positions); pred_loss is nats/dim, so *dq gives nats/chunk
        # before dividing by the K bytes that chunk represents.
        bpb_rec = rec_loss / math.log(2)
        bpb_pred = (pred_loss * self.cfg.dq) / (K * math.log(2))
        bpb_total = bpb_rec + bpb_pred  # full ELBO
        # LM-only proxy (valid once recon >= 99.9%, handover §1.7 boxed eq.) —
        # what you'd report at eval time once the decoder isn't needed.

        return loss, {
            "rec_loss": rec_loss,
            "pred_loss": pred_loss,
            "recon_acc": recon_acc,
            "latent_acc": latent_acc,
            "bpb_total": bpb_total,
            "bpb_lm_only": bpb_pred,
        }

    @torch.no_grad()
    def generate(self, prompt_chunks: torch.Tensor, n_chunks: int, temperature: float = 1.0) -> torch.Tensor:
        """Option A (pure latent autoregression, handover §2.1): the LM's sampled
        code is fed back directly, no re-encoding of decoded bytes."""
        self.eval()
        cfg = self.cfg
        B, T0, K = prompt_chunks.shape
        z_hat, _ = self.encoder(prompt_chunks.reshape(B * T0, K))
        z_history = z_hat.reshape(B, T0, -1)
        out_chunks = [prompt_chunks]

        for _ in range(n_chunks):
            logits = self.lm(z_history)[:, -1]  # last position's prediction
            if cfg.bottleneck == "fsq":
                probs = F.softmax(logits / temperature, dim=-1)              # [B, dq, L]
                levels = torch.multinomial(probs.reshape(-1, cfg.L), 1).reshape(B, cfg.dq)
                z_next = FSQ.levels_to_z(levels, cfg.L)
            else:
                bits = (torch.sigmoid(logits / temperature) > torch.rand_like(logits)).float()
                z_next = BSQ.bits_to_z(bits, cfg.dq)

            byte_logits = self.decoder(z_next)                                # [B, K, vocab]
            next_chunk = byte_logits.argmax(-1)                               # [B, K]
            out_chunks.append(next_chunk.unsqueeze(1))
            z_history = torch.cat([z_history, z_next.unsqueeze(1)], dim=1)

        return torch.cat(out_chunks, dim=1)


@torch.no_grad()
def score_continuation_bpb(model: QCuteLM, full_chunks: torch.Tensor, n_prompt_chunks: int, device: str) -> float:
    """Bpb restricted to the continuation region (chunks[n_prompt_chunks:]):
    decoder reconstruction CE for those chunks, plus LM next-code CE/BCE for
    the code predictions whose *target* chunk lies in the continuation
    (index n_prompt_chunks-1 in the shifted pred/target pair onward).
    full_chunks: [1, T, K], T = n_prompt_chunks + n_continuation_chunks."""
    model.eval()
    cfg = model.cfg
    B, T, K = full_chunks.shape
    z_hat, targets = model.encoder(full_chunks.reshape(B * T, K))
    z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)

    rec_logits = model.decoder(z_hat.reshape(B * T, -1)).reshape(B, T, K, cfg.vocab)
    cont_rec_logits = rec_logits[:, n_prompt_chunks:]
    cont_rec_targets = full_chunks[:, n_prompt_chunks:]
    rec_nats = F.cross_entropy(cont_rec_logits.reshape(-1, cfg.vocab), cont_rec_targets.reshape(-1))

    pred_logits = model.lm(z_hat)[:, :-1]
    pred_targets = targets[:, 1:]
    start = max(0, n_prompt_chunks - 1)  # first prediction whose target is a continuation chunk
    cont_pred_logits, cont_pred_targets = pred_logits[:, start:], pred_targets[:, start:]
    if cfg.bottleneck == "fsq":
        pred_nats = F.cross_entropy(cont_pred_logits.reshape(-1, cfg.L), cont_pred_targets.reshape(-1))
    else:
        pred_nats = F.binary_cross_entropy_with_logits(cont_pred_logits, cont_pred_targets)

    bpb_rec = rec_nats / math.log(2)
    bpb_pred = (pred_nats * cfg.dq) / (K * math.log(2))
    model.train()
    return (bpb_rec + bpb_pred).item()


def qualitative_generate(
    model: QCuteLM, prompt_chunks: torch.Tensor, n_gen_chunks: int,
    ground_truth_chunks: torch.Tensor | None, device: str, log=print,
) -> None:
    """Generate a continuation from a prompt (dataset-drawn or user-supplied)
    and, if a real ground-truth continuation is available (dataset-drawn),
    show it alongside the model's guess plus the model's bpb on the truth —
    a qualitative complement to the aggregate val_bpb number."""
    out = model.generate(prompt_chunks, n_gen_chunks)
    prompt_bytes = bytes(prompt_chunks[0].reshape(-1).tolist())
    gen_bytes = bytes(out[0, prompt_chunks.size(1):].reshape(-1).tolist())

    log(f"qual_prompt:       {prompt_bytes!r}")
    log(f"qual_generated:    {gen_bytes!r}")
    if ground_truth_chunks is not None:
        gt_bytes = bytes(ground_truth_chunks[0].reshape(-1).tolist())
        log(f"qual_ground_truth: {gt_bytes!r}")
        full_chunks = torch.cat([prompt_chunks, ground_truth_chunks], dim=1)
        bpb = score_continuation_bpb(model, full_chunks, prompt_chunks.size(1), device)
        log(f"qual_bpb_on_ground_truth: {bpb:.4f}", qual_bpb_on_ground_truth=bpb)


# ---------------------------------------------------------------------------
# Data + training loop
# ---------------------------------------------------------------------------


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


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


@torch.no_grad()
def eval_metrics(model: "QCuteLM", data_iter, n_batches: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for _ in range(n_batches):
        batch = next(data_iter)
        _, metrics = model(batch)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v.item()
    model.train()
    return {k: v / n_batches for k, v in totals.items()}


def lr_at(step: int, warmup: int, peak: float) -> float:
    """Linear warmup, then constant at peak — same schedule as qcute.bytelm,
    for a fair comparison between the two."""
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def load_config_module(path: Path) -> dict:
    """Load a Python config file (e.g. configs/qcutelm_bsq_tiny.py) as a dict
    of module-level variables. Values must already be the right type
    (Path(...), int, float, ...) — argparse's `type=` conversion only
    applies to strings passed on the actual command line, not to defaults."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def qualitative_generation_block(
    model: QCuteLM, cfg: Config, args, train_data: torch.Tensor, val_data: torch.Tensor, device: str, log,
) -> None:
    """Draws a prompt (dataset window or user text), rounds to whole chunks,
    and runs qualitative_generate. No-op if --qual_gen_bytes <= 0."""
    if args.qual_gen_bytes <= 0:
        return
    n_gen_chunks = max(1, args.qual_gen_bytes // cfg.K)

    if args.qual_source == "user":
        raw = args.qual_user_text.encode("utf-8")
        raw += b"\x00" * ((-len(raw)) % cfg.K)  # pad to a whole number of chunks
        n_prompt_chunks = max(1, len(raw) // cfg.K)
        prompt_chunks = torch.tensor([list(raw)], dtype=torch.long, device=device).reshape(1, n_prompt_chunks, cfg.K)
        ground_truth_chunks = None
    else:
        n_prompt_chunks = max(1, args.qual_prompt_bytes // cfg.K)
        src_data = train_data if args.qual_source == "train" else val_data
        total_chunks = n_prompt_chunks + n_gen_chunks
        seq_len = total_chunks * cfg.K
        start = torch.randint(0, len(src_data) - seq_len, (1,)).item()
        window = src_data[start : start + seq_len].reshape(total_chunks, cfg.K)
        prompt_chunks = window[:n_prompt_chunks].unsqueeze(0).to(device)
        ground_truth_chunks = window[n_prompt_chunks:].unsqueeze(0).to(device)

    qualitative_generate(model, prompt_chunks, n_gen_chunks, ground_truth_chunks, device, log=log)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Python config file (configs/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="qcute end-to-end tokenizer+LM (FSQ/BSQ, Option A)", parents=[pre])
    p.add_argument("--bottleneck", choices=["fsq", "bsq"], default="fsq")
    p.add_argument("--dq", type=int, default=None, help="defaults: 6 for fsq, 18 for bsq")
    p.add_argument("--K", type=int, default=4)  # see Config.K's comment for why 4, not the doc's 8
    p.add_argument("--data", type=Path, default=Path("datasets/enwik8.gz"))
    p.add_argument("--n_bytes", type=int, default=2_000_000, help="prefix of enwik8 to load")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_chunks", type=int, default=32, help="latents per training sequence")
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument(
        "--run_name", type=str, default=None,
        help="run directory name under logs/ and checkpoints/; falls back to the --config filename, then qcutelm_<bottleneck>_<timestamp>"
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
    p.add_argument("--qual_prompt_bytes", type=int, default=64, help="prompt length when --qual_source is train/val (rounded down to a multiple of K)")
    p.add_argument("--qual_user_text", type=str, default=None, help="prompt text when --qual_source user (utf-8 encoded, padded to a multiple of K)")

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
        cfg = Config(**ckpt["cfg"])
        model = QCuteLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
    else:
        cfg = build_config(args.bottleneck, args.dq, K=args.K)
        model = QCuteLM(cfg).to(device)
        start_step = 0
    n_params = sum(p_.numel() for p_ in model.parameters())

    if args.run_name:
        run_name = args.run_name
    elif pre_args.config:
        run_name = pre_args.config.stem
    else:
        run_name = f"qcutelm_{args.bottleneck}_{int(time.time())}"
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} (raw text) / {log.json_path} (JSONL) — tail -f {log.text_path}")
    loaded_note = f"  loaded_from={args.checkpoint_path} (step {start_step})" if args.checkpoint_path else ""
    log(f"bottleneck={cfg.bottleneck} dq={cfg.dq} params={n_params/1e6:.2f}M device={device}" + loaded_note)

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")
    val_iter = batch_iter(val_data, args.batch_size, args.seq_chunks, cfg.K, device)

    if args.eval_only:
        val = eval_metrics(model, val_iter, args.eval_batches)
        log(
            f"eval_only  val_recon_acc {val['recon_acc']*100:.2f}%  val_latent_acc {val['latent_acc']*100:.2f}%"
            f"  val_bpb {val['bpb_total']:.4f}  val_bpb_lm_only {val['bpb_lm_only']:.4f}",
            **{f"val_{k}": v for k, v in val.items()},
        )
        qualitative_generation_block(model, cfg, args, train_data, val_data, device, log)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak)
    train_iter = batch_iter(train_data, args.batch_size, args.seq_chunks, cfg.K, device)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(train_iter)
        loss, metrics = model(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(
            loss=f"{loss.item():.3f}", recon_acc=f"{metrics['recon_acc'].item()*100:.1f}%",
            bpb=f"{metrics['bpb_total'].item():.3f}",
        )
        if step % args.log_every == 0:
            msg = (
                f"lr {lr:.2e}  loss {loss.item():.4f}  rec {metrics['rec_loss'].item():.4f}"
                f"  pred {metrics['pred_loss'].item():.4f}  recon_acc {metrics['recon_acc'].item()*100:.2f}%"
                f"  latent_acc {metrics['latent_acc'].item()*100:.2f}%"
                f"  bpb {metrics['bpb_total'].item():.4f}  bpb_lm_only {metrics['bpb_lm_only'].item():.4f}"
            )
            log(f"{msg}  {pbar}", step=step, lr=lr, **{k: v.item() for k, v in metrics.items()})
        if step % args.eval_every == 0 or step == args.steps:
            val = eval_metrics(model, val_iter, args.eval_batches)
            msg = (
                f"step {step:5d}  val_recon_acc {val['recon_acc']*100:.2f}%  val_latent_acc {val['latent_acc']*100:.2f}%"
                f"  val_bpb {val['bpb_total']:.4f}  val_bpb_lm_only {val['bpb_lm_only']:.4f}"
            )
            log(msg, step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": asdict(cfg), **{f"val_{k}": v for k, v in val.items()}},
                val["bpb_total"],
            )
    log(
        f"checkpoints: best={checkpointer.best_path} (val_bpb {checkpointer.best_metric:.4f})  last={checkpointer.last_path}"
    )

    qualitative_generation_block(model, cfg, args, train_data, val_data, device, log)


if __name__ == "__main__":
    main()
