"""qcute.qcutelm_vlt5 — fork of qcute.qcutelm_vlt4 with a redesigned
reconstruction path, aimed specifically at getting tokenizer reconstruction
match to 95% before anything else (no joint latent-LM training here — see
qcutelm_vlt4's --joint_lm for that; this fork is pretraining only).

Two changes from qcutelm_vlt4's plain (non-joint_lm) path:

1. Sliding-window attention (attn_window, default 16) is now the intended
   default for this fork's whole design, not just an optional flag — see
   qcutelm_vlt4.py's ZeroKVCausalSelfAttention docstring for the mechanism.

2. THE reconstruction path is continuous across the whole context, not
   reset per K-byte block. qcutelm_vlt4's decode() treats every block as
   an independent fresh sequence: [code, byte0, byte1, ..., byte(K-2)],
   causal attention starting from nothing every time — a real detokenizer
   used by a downstream LM would instead have every previously-decoded
   byte and code available as context, continuously. This fork trains
   that way directly: build ONE long sequence spanning the whole context
   by, for every K-byte block, inserting that block's own code right
   before the block starts and DROPPING the block's own last byte (the
   byte the code was computed FROM) from the explicit input — its
   information only survives implicitly via the code. Run this whole
   sequence through the SAME shared Block stack in one continuous pass
   (same weights as the NTP path, windowed attention, no artificial reset
   between blocks) and predict the ENTIRE original byte sequence,
   position-for-position.

   Concretely, for K=4 bytes 1,2,3,4,5,6,7,8 with codes c4 (from position
   4, i.e. summarizing bytes 1-4) and c8 (from position 8, summarizing
   bytes 5-8):

       recon input:  c4, 1, 2, 3, c8, 5, 6, 7
       target:        1, 2, 3, 4,  5, 6, 7, 8

   Position 5 (predicting byte 5, the first byte of block 2) sees the
   ENTIRE reconstructed block 1 (c4,1,2,3) plus its own fresh code c8 —
   not just c8 in isolation the way qcutelm_vlt4's per-block decode would
   give it. Later blocks get progressively more accumulated context,
   within the sliding window, exactly matching how a real downstream LM
   would use this as a detokenizer: decoding continuously, never resetting.

No shared imports with qcutelm_vlt/vlt2/vlt3/vlt4 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers duplicated verbatim.
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
    K: int = 4
    context_len: int = 128
    attn_window: int = 16       # -1 = full causal. Default 16 here (not -1) — see module docstring.
    dq: int = 18
    lfq: bool = False
    quant_type: str = "bsq"     # "bsq", "fsq", or "ifsq"
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_mult: int = 4
    code_net_layers: int = 0
    ntp_loss_weight: float = 1.0
    recon_loss_weight: float = 1.0

    # --- CodeLM: a causal transformer sitting between code production and
    # decode, operating on the CONTINUOUS code embedding sequence (no
    # code_to_index/classification loss — see module docstring). Residual
    # + zero-initialized output projection means CodeLM is EXACTLY the
    # identity function at initialization: plain pretraining (NTP +
    # recon-directly-from-code) is the literal use_code_lm=False /
    # zero-init special case of this same architecture, not a different
    # mode. No detach anywhere — recon_loss's gradient flows freely back
    # through CodeLM into code_net/the tokenizer's own blocks.
    use_code_lm: bool = True
    lm_d_model: int = 128
    lm_n_heads: int = 4
    lm_n_layers: int = 4
    lm_mlp_mult: int = 4


def build_config(dq: int | None, **kwargs) -> Config:
    if dq is None:
        dq = 18
    return Config(dq=dq, **kwargs)


def bsq_quantize(v: torch.Tensor, dq: int, lfq: bool = False) -> torch.Tensor:
    if lfq:
        return v + (torch.sign(v) - v).detach()
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def fsq_quantize(v: torch.Tensor, levels: int, bound: str = "tanh") -> torch.Tensor:
    half_l = (levels - 1) / 2
    z = torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1)
    bounded = z * half_l
    z_hat = bounded + (torch.round(bounded) - bounded).detach()
    return z_hat / half_l


class ZeroKVCausalSelfAttention(nn.Module):
    """Causal self-attention with a single zero key/value pair (Miller
    2023, "Attention Is Off By One") — escape-hatch role. window=None:
    full causal. window=N: banded, last N real keys only — see
    qcute.qcutelm_vlt4's identical mechanism for the full rationale."""

    def __init__(self, d_model: int, n_heads: int, window: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, hd]
        if self.window is not None and T % self.window == 0 and T > self.window:
            y = self._forward_chunked(q, k, v)
        else:
            y = self._forward_dense(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def _forward_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """O(T^2) dense-masked path — used when window is None (full
        causal) or T isn't a clean multiple of window (chunking needs
        T % window == 0). See _forward_chunked for the O(T*window) path."""
        B, H, T, hd = q.shape
        zero_kv = torch.zeros(B, H, 1, hd, device=q.device, dtype=q.dtype)
        k = torch.cat([zero_kv, k], dim=2)
        v = torch.cat([zero_kv, v], dim=2)
        attn_mask = torch.zeros(T, 1 + T, dtype=torch.bool, device=q.device)
        attn_mask[:, 0] = True
        if self.window is None:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        else:
            i = torch.arange(T, device=q.device).view(T, 1)
            j = torch.arange(T, device=q.device).view(1, T)
            causal = (j <= i) & (i - j < self.window)
        attn_mask[:, 1:] = causal
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """O(T*window) sliding-window attention via reshape: split T into
        non-overlapping chunks of size W=window; each chunk's queries only
        ever need this chunk's + the previous chunk's keys/values (a
        window of size W reaching back from anywhere in chunk c can reach
        at most into chunk c-1). The needed causal+window mask is then a
        single fixed [W, 2W] pattern — identical for every chunk/batch/
        layer — instead of a full [T, T] mask. Mathematically identical
        result to _forward_dense at the same window; just avoids ever
        materializing the O(T^2) score matrix."""
        B, H, T, hd = q.shape
        W = self.window
        n_chunks = T // W

        def to_chunks(t):
            return t.view(B, H, n_chunks, W, hd)

        qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
        zero_chunk = torch.zeros(B, H, 1, W, hd, device=q.device, dtype=q.dtype)
        kc_prev = torch.cat([zero_chunk, kc[:, :, :-1]], dim=2)  # chunk c-1, zeros for c=0
        vc_prev = torch.cat([zero_chunk, vc[:, :, :-1]], dim=2)
        k_local = torch.cat([kc_prev, kc], dim=3)  # [B, H, n_chunks, 2W, hd]
        v_local = torch.cat([vc_prev, vc], dim=3)
        zero_kv_sink = torch.zeros(B, H, n_chunks, 1, hd, device=q.device, dtype=q.dtype)
        k_local = torch.cat([zero_kv_sink, k_local], dim=3)  # [B, H, n_chunks, 1+2W, hd] — sink always visible
        v_local = torch.cat([zero_kv_sink, v_local], dim=3)

        i = torch.arange(W, device=q.device).view(W, 1)          # query local offset within current chunk
        j_prev = torch.arange(W, device=q.device).view(1, W) - W  # prev-chunk key offsets: -W..-1
        j_cur = torch.arange(W, device=q.device).view(1, W)       # cur-chunk key offsets: 0..W-1
        key_offset = torch.cat([j_prev, j_cur], dim=1)            # [1, 2W]
        diff = i - key_offset                                     # [W, 2W]
        causal_window = (diff >= 0) & (diff < W)                  # W == self.window by construction here
        mask = torch.cat([torch.ones(W, 1, dtype=torch.bool, device=q.device), causal_window], dim=1)  # [W, 1+2W]

        # chunk 0 has no real previous chunk (kc_prev/vc_prev are zero-padding, not real content) — the
        # general mask still marks some prev-chunk slots "allowed" for chunk 0's queries, which would let
        # the softmax denominator include fake zero-key contributions. Override chunk 0's prev-chunk columns
        # to always-False so it only ever attends to the sink + its own chunk, matching _forward_dense exactly.
        mask_per_chunk = mask.unsqueeze(0).expand(n_chunks, W, 1 + 2 * W).clone()
        mask_per_chunk[0, :, 1:1 + W] = False

        # fold (B, n_chunks) into one batch dim for a single batched SDPA call
        qb = qc.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = k_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 1 + 2 * W, hd)
        vb = v_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 1 + 2 * W, hd)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 1 + 2 * W).reshape(B * n_chunks, 1, W, 1 + 2 * W)
        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)  # broadcasts over H
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, window: int | None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ZeroKVCausalSelfAttention(d_model, n_heads, window=window)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model), nn.SiLU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CodeLM(nn.Module):
    """Causal transformer over the SEQUENCE of code embeddings (one per
    K-byte block), sitting between code production and decode. Full
    causal (n_blocks is already much shorter than the raw byte context,
    no windowing needed). No separate code-level classification loss —
    trained purely by whatever backprops into it, which here is
    recon_loss (see module docstring). Residual with a zero-initialized
    output projection: forward(x) == x exactly at initialization, so
    plain pretraining (no CodeLM contribution) is the literal starting
    point this learns away from, not a separate code path."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.in_proj = nn.Identity() if cfg.lm_d_model == cfg.d_model else nn.Linear(cfg.d_model, cfg.lm_d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=None) for _ in range(cfg.lm_n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        self.out_proj = nn.Linear(cfg.lm_d_model, cfg.d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, code_emb: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(code_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return code_emb + self.out_proj(x)  # identity at init (out_proj is zero-initialized)


class ContinuousReconTokenizer(nn.Module):
    """Regular windowed-attention causal LM (byte_emb + shared Block
    stack, NoPE) whose every-K-th-timestep hidden states are read off by a
    small code_net to produce codes — same as qcutelm_vlt4's
    StridedReadoutTokenizer. The reconstruction path is what's different:
    continuous across the whole context (see module docstring), not reset
    per block."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.context_len % cfg.K == 0, "context_len must be a multiple of K"
        self.cfg = cfg
        window = None if cfg.attn_window == -1 else cfg.attn_window
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)  # shared: NTP head AND reconstruction head
        if cfg.code_net_layers <= 0:
            self.code_net = nn.Linear(cfg.d_model, cfg.dq)
        else:
            layers = []
            d = cfg.d_model
            for _ in range(cfg.code_net_layers):
                layers += [nn.Linear(d, cfg.d_model), nn.SiLU()]
                d = cfg.d_model
            layers.append(nn.Linear(d, cfg.dq))
            self.code_net = nn.Sequential(*layers)
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.code_lm = CodeLM(cfg) if cfg.use_code_lm else None

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def quantize(self, raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "fsq":
            return fsq_quantize(raw, self.cfg.fsq_levels, bound="tanh")
        if self.cfg.quant_type == "ifsq":
            return fsq_quantize(raw, self.cfg.fsq_levels, bound="sigmoid")
        return bsq_quantize(raw, self.cfg.dq, self.cfg.lfq)

    def lm_forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.run_blocks(self.byte_emb(ctx))
        ntp_logits = self.head(h[:, :-1, :])
        return h, ntp_logits

    def codes_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        K = self.cfg.K
        stride_h = h[:, K - 1::K, :]
        return self.quantize(self.code_net(stride_h))

    def build_continuous_recon_seq(self, ctx: torch.Tensor, code_embs: torch.Tensor) -> torch.Tensor:
        """ctx: [B, L], code_embs: [B, L//K, d_model] (post-CodeLM if
        use_code_lm, else z_proj(z_hat) directly) -> recon input embeddings
        [B, L, d_model]. Per block: [code_emb, byte_emb(block[0]), ...,
        byte_emb(block[K-2])] — code replaces the block's own last byte's
        slot, first K-1 real bytes shift into positions 1..K-1. See module
        docstring for the worked example."""
        cfg = self.cfg
        B, L = ctx.shape
        K = cfg.K
        n_blocks = L // K
        byte_embs = self.byte_emb(ctx).view(B, n_blocks, K, cfg.d_model)
        prefix_embs = byte_embs[:, :, :K - 1, :]              # drop each block's own last byte
        code_embs = code_embs.unsqueeze(2)                     # [B, n_blocks, 1, d_model]
        recon_blocks = torch.cat([code_embs, prefix_embs], dim=2)  # [B, n_blocks, K, d_model]
        return recon_blocks.view(B, L, cfg.d_model)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h, ntp_logits = self.lm_forward(ctx)
        ntp_targets = ctx[:, 1:]
        # NTP loss stays entirely pre-CodeLM: computed straight from the tokenizer's own hidden
        # states, never touches code_lm — "pure ntp on tokenizer pre codelm level".
        ntp_loss = F.cross_entropy(ntp_logits.reshape(-1, self.cfg.vocab), ntp_targets.reshape(-1))

        z_hat = self.codes_from_hidden(h)     # [B, n_blocks, dq]
        code_embs = self.z_proj(z_hat)         # [B, n_blocks, d_model]
        if self.code_lm is not None:
            code_embs = self.code_lm(code_embs)  # post-codelm — "the recon with code post codelm"
        recon_in = self.build_continuous_recon_seq(ctx, code_embs)  # [B, L, d_model]
        h_recon = self.run_blocks(recon_in)   # SAME shared weights, one continuous pass, no reset
        recon_logits = self.head(h_recon)     # [B, L, vocab] — predicts the ENTIRE original ctx

        # No separate code-level classification loss anywhere ("no supervision on code") — recon_loss
        # is the only thing that trains code_lm, and nothing is detached: gradient flows straight
        # back through code_lm into code_net/the tokenizer's own blocks.
        recon_loss = F.cross_entropy(recon_logits.reshape(-1, self.cfg.vocab), ctx.reshape(-1))
        recon_acc = (recon_logits.argmax(-1) == ctx).float().mean()

        loss = self.cfg.ntp_loss_weight * ntp_loss + self.cfg.recon_loss_weight * recon_loss
        return loss, {
            "loss": loss, "ntp_loss": ntp_loss, "recon_loss": recon_loss, "recon_acc": recon_acc,
            "recon_logits": recon_logits,
        }


def init_head_bias_to_unigram(model: ContinuousReconTokenizer, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        model.head.bias.copy_(log_freq.to(model.head.bias.device))


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    n = max(1, len(data) - context_len)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + context_len] for s in starts]).to(device)


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
def eval_model(model: ContinuousReconTokenizer, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> tuple[float, float, list[float]]:
    """Returns (val_bpb, val_recon_acc, per_chunk_acc). per_chunk_acc[i] is
    accuracy for chunk i specifically (0 = only its own code, no prior
    accumulated context; later chunks see progressively more via the
    continuous recon path) — the aggregate val_recon_acc pools all chunks
    together, which can hide a chunk-0-specific shortfall if later chunks
    (easier, more context) are pulling the average up. See session notes."""
    model.eval()
    K = model.cfg.K
    n_blocks = model.cfg.context_len // K
    ntp_losses, recon_accs = [], []
    chunk_correct = torch.zeros(n_blocks)
    chunk_total = torch.zeros(n_blocks)
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        ntp_losses.append(metrics["ntp_loss"].item())
        recon_accs.append(metrics["recon_acc"].item())
        match = (metrics["recon_logits"].argmax(-1) == ctx).float()  # [B, L]
        match_blocks = match.view(-1, n_blocks, K).mean(-1)  # [B, n_blocks] — per-chunk accuracy
        chunk_correct += match_blocks.sum(0).cpu()
        chunk_total += match_blocks.size(0)
    model.train()
    bpb = (sum(ntp_losses) / len(ntp_losses)) / math.log(2)
    per_chunk_acc = (chunk_correct / chunk_total).tolist()
    return bpb, sum(recon_accs) / len(recon_accs), per_chunk_acc


def train(model: ContinuousReconTokenizer, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    """Pretraining only, focused on reconstruction match. No curriculum:
    context_len/K fixed by config, every batch exercises the full task."""
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt5", dynamic_ncols=True)
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

        train_bpb = metrics["ntp_loss"].item() / math.log(2)
        acc = metrics["recon_acc"].item()
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", ntp_loss=f"{metrics['ntp_loss'].item():.4f}",
                          train_bpb=f"{train_bpb:.4f}", recon_loss=f"{metrics['recon_loss'].item():.4f}",
                          recon_acc=f"{acc*100:.2f}%")

        if step % args.eval_every == 0 or step == args.steps:
            val_bpb, val_recon_acc, per_chunk_acc = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            # chunk 0 has zero accumulated context (only its own code) — the true hard case, easy for the
            # pooled val_recon_acc to hide if later chunks (more context) are pulling the average up.
            chunk0_acc = per_chunk_acc[0]
            rest_mean_acc = sum(per_chunk_acc[1:]) / max(1, len(per_chunk_acc) - 1)
            log(f"{pbar}  val_bpb={val_bpb:.4f}  val_recon_acc={val_recon_acc*100:.2f}%  "
                f"chunk0_acc={chunk0_acc*100:.2f}%  chunk1plus_mean_acc={rest_mean_acc*100:.2f}%  "
                f"chunk_last_acc={per_chunk_acc[-1]*100:.2f}%",
                step=step, train_bpb=train_bpb, train_recon_acc=acc, val_bpb=val_bpb, val_recon_acc=val_recon_acc,
                chunk0_acc=chunk0_acc, chunk1plus_mean_acc=rest_mean_acc, chunk_last_acc=per_chunk_acc[-1])
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step, "val_recon_acc": val_recon_acc}, 1.0 - val_recon_acc)
            if val_recon_acc >= args.recon_target_acc:
                log(f"recon target reached: val_recon_acc {val_recon_acc*100:.2f}% >= {args.recon_target_acc*100:.1f}% at step {step}", step=step)
                return


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Continuous (non-reset) reconstruction tokenizer (fork of qcute.qcutelm_vlt4)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--context_len", type=int, default=128)
    p.add_argument("--attn_window", type=int, default=16)
    p.add_argument("--dq", type=int, default=None)
    p.add_argument("--lfq", action="store_true")
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--code_net_layers", type=int, default=0)
    p.add_argument("--ntp_loss_weight", type=float, default=1.0)
    p.add_argument("--recon_loss_weight", type=float, default=1.0)
    p.add_argument("--use_code_lm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--lm_d_model", type=int, default=128)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=4)
    p.add_argument("--lm_mlp_mult", type=int, default=4)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--recon_target_acc", type=float, default=0.95)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config(
        args.dq, K=args.K, context_len=args.context_len, attn_window=args.attn_window, lfq=args.lfq,
        quant_type=args.quant_type, fsq_levels=args.fsq_levels, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, code_net_layers=args.code_net_layers,
        ntp_loss_weight=args.ntp_loss_weight, recon_loss_weight=args.recon_loss_weight,
        use_code_lm=args.use_code_lm, lm_d_model=args.lm_d_model, lm_n_heads=args.lm_n_heads,
        lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult,
    )
    model = ContinuousReconTokenizer(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt5_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} context_len={cfg.context_len} attn_window={cfg.attn_window} dq={cfg.dq} "
        f"quant_type={cfg.quant_type} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
