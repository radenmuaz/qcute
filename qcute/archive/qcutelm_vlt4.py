"""qcute.qcutelm_vlt4 — fork of qcute.qcutelm_vlt3 replacing the trainable
"code-query" token with a strided readout over a regular, large-context
causal LM.

Design (as specified in conversation): qcutelm_vlt3's code-query token has
to learn pooling from scratch AND only ever sees a K-byte window (no real
context) — its representation is entirely shaped by the tokenizer's own
narrow reconstruction objective. This fork instead trains the shared
Block stack as a REGULAR byte-level causal LM over a much larger context
(`context_len`, e.g. 128) with a standard next-token-prediction loss as
the primary objective — the same unconstrained training bytelm.py gets,
so the hidden states are rich, context-informed representations, not
representations warped to serve a bottleneck from step 1.

Codes are then read off this already-good LM for free: take the hidden
state at every K-th timestep (the position right at the end of each
K-byte block — under the causal mask it has seen that whole block, plus
everything before it), and pass it through a small separate net
(`code_net` — plain linear by default, optionally a 1-hidden-layer MLP)
to produce the code. This keeps the LM's own representations from being
distorted much by the bottleneck (only a small net sits between "LM
hidden state" and "code"), while the code itself is derived from a much
more informative representation than qcutelm_vlt3's code-query token ever
had access to.

Each K-byte block's code is then used exactly like qcutelm_vlt3's
code-as-BOS decode stage: z_proj(z_hat) becomes a content-dependent BOS
token, followed by teacher-forced bytes, run through the SAME shared
Block stack, reconstructing that same K-byte block. All blocks across a
context window are decoded in one flattened batched call (not one at a
time) — still just 2 "stages" per training step (one LM forward pass
providing both NTP logits and every block's readout hidden state, one
decode forward pass over all blocks flattened into the batch dimension),
same efficiency shape as qcutelm_vlt3.

No curriculum here — context_len and K are both fixed by config; the LM
half of the objective doesn't need staging the way isolated K-byte
reconstruction did, and every block in every training batch already
exercises the full K-length reconstruction task from step 1.

No shared imports with qcutelm_vlt/vlt2/vlt3 (self-contained-module
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
    K: int = 4                  # code/decode block length (each code summarizes K bytes)
    context_len: int = 128      # LM training context — must be a multiple of K
    dq: int = 18
    lfq: bool = False
    quant_type: str = "bsq"     # "bsq", "fsq", or "ifsq"
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_mult: int = 4
    code_net_layers: int = 0    # 0 = plain nn.Linear readout, 1+ = MLP with that many hidden layers
    ntp_loss_weight: float = 1.0    # weight on the LM's own NTP loss (was hardcoded implicit 1, now explicit/tunable)
    recon_loss_weight: float = 1.0  # weight on the block-reconstruction loss
    self_distill_weight: float = 0.0  # 0 = off. >0: distill LM's (full-context) NTP logits into decode's (code-only) logits
    attn_window: int = -1       # -1 = full causal (unbounded). Else banded sliding window width —
                                 # set equal to context_len so streaming inference over long documents
                                 # reproduces exactly the "<= context_len bytes of context" distribution
                                 # training saw, at every position (see ZeroKVCausalSelfAttention docstring)

    # --- joint latent-LM training (train_joint_lm / forward_joint_lm) ---
    # Hierarchical context: the tokenizer only needs a small LOCAL window
    # (context_len) to produce a faithful code per K-byte block — the LONG
    # effective range comes from CodeLM stacking many codes, not from the
    # tokenizer's own attention span. lm_context_bytes (must be a multiple
    # of context_len) is split into lm_context_bytes // context_len
    # independent tokenizer windows, each producing context_len // K codes;
    # all windows' codes are concatenated into one sequence of length
    # lm_context_bytes // K for CodeLM — e.g. context_len=16, K=4,
    # lm_context_bytes=256 -> 16 independent 16-byte windows x 4 codes each
    # = 64 codes of CodeLM context, reaching the full 256 bytes, while each
    # individual tokenizer forward pass only ever attends over 16 bytes.
    lm_context_bytes: int = 256
    lm_d_model: int = 128       # CodeLM width (separate from the tokenizer's d_model — different token space)
    lm_n_heads: int = 4
    lm_n_layers: int = 4
    lm_mlp_mult: int = 4
    code_lm_weight: float = 1.0
    code_lm_warmup_steps: int = 5000  # code_lm_weight ramps 0 -> code_lm_weight linearly over this many steps.
                                       # The code space is a moving target early on (tokenizer weights still
                                       # changing under recon_loss every step) — training CodeLM hard against
                                       # it from step 0 wastes its gradient chasing a target that won't hold
                                       # still; letting the tokenizer settle first gives CodeLM something
                                       # learnable to predict (see module docstring / session notes).
    code_lm_detach: bool = True  # explicit stop-gradient into the tokenizer from the code-LM loss (see module
                                  # docstring for why). Note code_to_index()'s round()/.long() are inherently
                                  # non-differentiable regardless — setting this False would not actually let
                                  # gradient through with this index-based CodeLM; kept for documentation intent
                                  # and as a hook if a future continuous/soft CodeLM input replaces the hard index.


def build_config(dq: int | None, **kwargs) -> Config:
    if dq is None:
        dq = 18
    return Config(dq=dq, **kwargs)


def bsq_quantize(v: torch.Tensor, dq: int, lfq: bool = False) -> torch.Tensor:
    """Identical math to qcute.qcutelm.bsq_quantize — duplicated per this
    repo's self-contained-module convention, not an oversight. Operates on
    the last dim regardless of leading batch dims."""
    if lfq:
        return v + (torch.sign(v) - v).detach()
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def fsq_quantize(v: torch.Tensor, levels: int, bound: str = "tanh") -> torch.Tensor:
    """Same as qcute.qcutelm_vlt3.fsq_quantize — see that module's
    docstring. bound="sigmoid" is the iFSQ variant."""
    half_l = (levels - 1) / 2
    z = torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1)
    bounded = z * half_l
    z_hat = bounded + (torch.round(bounded) - bounded).detach()
    return z_hat / half_l


class ZeroKVCausalSelfAttention(nn.Module):
    """Causal self-attention with a single zero key/value pair concatenated
    before SDPA (Miller 2023, "Attention Is Off By One") — escape-hatch
    role, as in qcutelm_vlt2/vlt3.

    window=None: plain causal (unbounded), the qcutelm_vlt2/vlt3 behavior —
    fine for training on fixed-length context_len samples, but mismatched
    for inference over documents longer than context_len: naive chunking
    gives boundary positions less context than training taught them to
    expect, while plain full-causal attention over a whole long document
    gives later positions MORE context than any training example ever had.
    window=N bounds every position to its last N real keys (banded mask,
    tril AND NOT tril-shifted-by-N) — this makes an arbitrarily long
    single forward pass reproduce, at every position, exactly the same
    "<= N bytes of true local context" conditional distribution training
    saw with fixed-length windows. Pair with window=context_len so
    training and streaming inference match exactly."""

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
        q, k, v = qkv[0], qkv[1], qkv[2]
        zero_kv = torch.zeros(B, H, 1, hd, device=x.device, dtype=x.dtype)
        k = torch.cat([zero_kv, k], dim=2)
        v = torch.cat([zero_kv, v], dim=2)
        attn_mask = torch.zeros(T, 1 + T, dtype=torch.bool, device=x.device)
        attn_mask[:, 0] = True
        if self.window is None:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        else:
            i = torch.arange(T, device=x.device).view(T, 1)
            j = torch.arange(T, device=x.device).view(1, T)
            causal = (j <= i) & (i - j < self.window)  # banded: last `window` real keys only
        attn_mask[:, 1:] = causal
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Generic pre-norm residual block — takes explicit dims (not a Config)
    so it's reusable for both the tokenizer's byte-level stack and CodeLM's
    code-level stack, which operate on different widths/token spaces."""

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


def code_to_index(z_hat: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Codes are already indices into a fixed, deterministic vocabulary —
    no learned codebook needed. BSQ: each of the dq dims is a sign bit,
    index = sum(bit_i * 2^i), vocab_size = 2^dq. FSQ/iFSQ: each dim is one
    of `fsq_levels` integers, index = sum(level_i * levels^i), vocab_size
    = levels^dq. z_hat: [..., dq] -> index: [...] long."""
    if cfg.quant_type in ("fsq", "ifsq"):
        half_l = (cfg.fsq_levels - 1) / 2
        level = torch.round(z_hat * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
        basis = cfg.fsq_levels ** torch.arange(cfg.dq, device=z_hat.device)
        return (level * basis).sum(-1)
    bit = (z_hat > 0).long()
    basis = 2 ** torch.arange(cfg.dq, device=z_hat.device)
    return (bit * basis).sum(-1)


def code_vocab_size(cfg: Config) -> int:
    if cfg.quant_type in ("fsq", "ifsq"):
        return cfg.fsq_levels ** cfg.dq
    return 2 ** cfg.dq


class CodeLM(nn.Module):
    """Separate small causal transformer over the SEQUENCE of code indices
    (one per K-byte block) — literally qcute.bytelm's recipe applied to
    code-space instead of byte-space: embedding table sized to the code
    vocabulary, causal Block stack, softmax next-code-prediction loss. Own
    weights, own width (lm_d_model) — different token space from the
    tokenizer's byte-level stack, so no weight sharing here."""

    def __init__(self, cfg: Config):
        super().__init__()
        vocab_size = code_vocab_size(cfg)
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.lm_d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=None) for _ in range(cfg.lm_n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        self.head = nn.Linear(cfg.lm_d_model, vocab_size)

    def forward(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # indices: [B, n_blocks] long -> (loss, acc) next-code prediction
        x = self.tok_emb(indices)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, :-1, :])  # [B, n_blocks-1, vocab_size]
        targets = indices[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        acc = (logits.argmax(-1) == targets).float().mean()
        return loss, acc


class StridedReadoutTokenizer(nn.Module):
    """Regular large-context causal LM (byte_emb + shared Block stack,
    NoPE) whose every-K-th-timestep hidden states are read off by a small
    `code_net` to produce codes, each reconstructing its own K-byte block
    via the shared weights' code-as-BOS decode path."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.context_len % cfg.K == 0, "context_len must be a multiple of K"
        assert cfg.lm_context_bytes % cfg.context_len == 0, "lm_context_bytes must be a multiple of context_len"
        self.cfg = cfg
        window = None if cfg.attn_window == -1 else cfg.attn_window
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)  # shared: NTP head AND decode-reconstruction head
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
        self.code_lm = CodeLM(cfg)  # only used by forward_joint_lm / train_joint_lm

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
        """ctx: [B, L] -> (h: [B, L, D], ntp_logits: [B, L-1, vocab])"""
        h = self.run_blocks(self.byte_emb(ctx))
        ntp_logits = self.head(h[:, :-1, :])
        return h, ntp_logits

    def codes_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, D] -> z_hat: [B, L//K, dq], one code per K-byte block,
        read from the hidden state at the end of each block (index K-1,
        2K-1, ... under 0-indexing)."""
        K = self.cfg.K
        stride_h = h[:, K - 1::K, :]  # [B, L//K, D]
        return self.quantize(self.code_net(stride_h))

    def decode(self, z_hat: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """z_hat: [N, dq], block: [N, K] (teacher forcing targets) -> logits: [N, K, vocab].
        N is typically B*(L//K) — all blocks flattened into the batch dim."""
        N, K = block.shape
        bos = self.z_proj(z_hat).unsqueeze(1)
        if K > 1:
            dec_in = torch.cat([bos, self.byte_emb(block[:, :-1])], dim=1)
        else:
            dec_in = bos
        dec_h = self.run_blocks(dec_in)
        return self.head(dec_h)

    @torch.no_grad()
    def decode_greedy(self, z_hat: torch.Tensor, K: int) -> torch.Tensor:
        N = z_hat.size(0)
        bos = self.z_proj(z_hat).unsqueeze(1)
        seq = bos
        out_bytes = []
        for _ in range(K):
            h = self.run_blocks(seq)
            logits = self.head(h[:, -1, :])
            next_byte = logits.argmax(-1)
            out_bytes.append(next_byte)
            seq = torch.cat([seq, self.byte_emb(next_byte).unsqueeze(1)], dim=1)
        return torch.stack(out_bytes, dim=1)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # ctx: [B, context_len] long
        B, L = ctx.shape
        K = self.cfg.K
        h, ntp_logits = self.lm_forward(ctx)
        ntp_targets = ctx[:, 1:]
        ntp_loss = F.cross_entropy(ntp_logits.reshape(-1, self.cfg.vocab), ntp_targets.reshape(-1))

        z_hat = self.codes_from_hidden(h)  # [B, L//K, dq]
        n_blocks = L // K
        blocks = ctx.view(B, n_blocks, K)
        z_hat_flat = z_hat.reshape(B * n_blocks, self.cfg.dq)
        blocks_flat = blocks.reshape(B * n_blocks, K)
        dec_logits = self.decode(z_hat_flat, blocks_flat)  # [B*n_blocks, K, vocab]

        recon_loss = F.cross_entropy(dec_logits.reshape(-1, self.cfg.vocab), blocks_flat.reshape(-1))
        recon_acc = (dec_logits.argmax(-1) == blocks_flat).float().mean()

        if self.cfg.self_distill_weight > 0:
            distill_loss = self.self_distill_loss(ntp_logits, dec_logits, B, L, n_blocks, K)
        else:
            distill_loss = torch.zeros((), device=ctx.device)

        loss = self.cfg.ntp_loss_weight * ntp_loss + self.cfg.recon_loss_weight * recon_loss + self.cfg.self_distill_weight * distill_loss
        return loss, {
            "loss": loss, "ntp_loss": ntp_loss, "recon_loss": recon_loss, "recon_acc": recon_acc,
            "distill_loss": distill_loss,
        }

    def forward_joint_lm(self, ctx: torch.Tensor, code_lm_weight: float | None = None) -> tuple[torch.Tensor, dict]:
        """Joint training with a real latent LM, from random init. No raw-
        byte NTP loss (removed entirely — the LM objective now lives at
        the code level via CodeLM). Only the required losses: recon_loss
        (shapes the code space) + code_lm_loss (the actual generative
        objective this whole project is aiming at). Simplified metrics:
        just the two accuracies that matter — recon_acc and code_lm_acc.

        ctx: [B, lm_context_bytes] — split into lm_context_bytes //
        context_len independent LOCAL tokenizer windows (batched together,
        one cheap forward pass, no cross-window attention needed); each
        window's codes are concatenated back into one long per-example
        code sequence for CodeLM. See Config.lm_context_bytes docstring."""
        cfg = self.cfg
        B, Lm = ctx.shape
        K = cfg.K
        window_len = cfg.context_len
        n_windows = Lm // window_len
        blocks_per_window = window_len // K
        total_blocks = n_windows * blocks_per_window

        ctx_windows = ctx.view(B * n_windows, window_len)  # independent local windows, batched
        h = self.run_blocks(self.byte_emb(ctx_windows))    # [B*n_windows, window_len, d_model]
        z_hat_w = self.codes_from_hidden(h)                 # [B*n_windows, blocks_per_window, dq]
        z_hat = z_hat_w.reshape(B, total_blocks, cfg.dq)    # reassembled, byte-order-consistent per example

        blocks_flat = ctx_windows.view(B * total_blocks, K)
        z_hat_flat = z_hat_w.reshape(B * total_blocks, cfg.dq)
        dec_logits = self.decode(z_hat_flat, blocks_flat)

        recon_loss = F.cross_entropy(dec_logits.reshape(-1, cfg.vocab), blocks_flat.reshape(-1))
        recon_acc = (dec_logits.argmax(-1) == blocks_flat).float().mean()

        indices = code_to_index(z_hat, cfg)  # [B, total_blocks] — spans the full lm_context_bytes
        code_lm_input = indices.detach() if cfg.code_lm_detach else indices
        code_lm_loss, code_lm_acc = self.code_lm(code_lm_input)

        w = cfg.code_lm_weight if code_lm_weight is None else code_lm_weight
        loss = recon_loss + w * code_lm_loss
        return loss, {
            "loss": loss, "recon_loss": recon_loss, "recon_acc": recon_acc,
            "code_lm_loss": code_lm_loss, "code_lm_acc": code_lm_acc, "code_lm_weight": w,
        }

    def self_distill_loss(self, ntp_logits: torch.Tensor, dec_logits: torch.Tensor, B: int, L: int, n_blocks: int, K: int) -> torch.Tensor:
        """Distills the LM stage's own next-token logits (which see real
        left-context, up to context_len-1 bytes) into the decode stage's
        logits (which only see the compressed code) at the same byte
        positions — a soft-target signal on top of the hard reconstruction
        cross-entropy, using the model's better-informed self as the
        teacher. Position 0 of the whole context has no left context (no
        NTP prediction exists for it) and is excluded."""
        vocab = self.cfg.vocab
        teacher_logits = F.pad(ntp_logits, (0, 0, 1, 0))  # [B, L, vocab] — position 0 is a zero placeholder
        teacher_blocks = teacher_logits.view(B, n_blocks, K, vocab).reshape(B * n_blocks, K, vocab)
        valid = torch.ones(B, L, dtype=torch.bool, device=ntp_logits.device)
        valid[:, 0] = False
        valid_blocks = valid.view(B, n_blocks, K).reshape(B * n_blocks, K)

        student_logp = F.log_softmax(dec_logits, dim=-1)
        teacher_p = F.softmax(teacher_blocks.detach(), dim=-1)
        per_tok = -(teacher_p * student_logp).sum(-1)  # [B*n_blocks, K] — soft-target cross-entropy
        denom = valid_blocks.float().sum().clamp(min=1)
        return (per_tok * valid_blocks.float()).sum() / denom


def init_head_bias_to_unigram(model: StridedReadoutTokenizer, data: torch.Tensor) -> None:
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
def eval_model(model: StridedReadoutTokenizer, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> tuple[float, float, float]:
    """Returns (val_bpb, val_recon_acc, val_loss)."""
    model.eval()
    ntp_losses, recon_accs = [], []
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        ntp_losses.append(metrics["ntp_loss"].item())
        recon_accs.append(metrics["recon_acc"].item())
    model.train()
    mean_ntp = sum(ntp_losses) / len(ntp_losses)
    bpb = mean_ntp / math.log(2)
    return bpb, sum(recon_accs) / len(recon_accs), mean_ntp


def train(model: StridedReadoutTokenizer, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    """No curriculum: context_len/K are both fixed by config, every batch
    already exercises the full task (LM NTP + all-blocks reconstruction)."""
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt4", dynamic_ncols=True)
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
        postfix = dict(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", ntp_loss=f"{metrics['ntp_loss'].item():.4f}",
                       train_bpb=f"{train_bpb:.4f}", recon_loss=f"{metrics['recon_loss'].item():.4f}",
                       recon_acc=f"{acc*100:.2f}%")
        if model.cfg.self_distill_weight > 0:
            postfix["distill_loss"] = f"{metrics['distill_loss'].item():.4f}"
        pbar.set_postfix(**postfix)

        if step % args.eval_every == 0 or step == args.steps:
            val_bpb, val_recon_acc, val_ntp_loss = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            log(f"{pbar}  val_bpb={val_bpb:.4f}  val_recon_acc={val_recon_acc*100:.2f}%",
                step=step, train_bpb=train_bpb, train_recon_acc=acc, val_bpb=val_bpb, val_recon_acc=val_recon_acc)
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step, "val_recon_acc": val_recon_acc}, 1.0 - val_recon_acc)
            if val_recon_acc >= args.recon_target_acc:
                log(f"recon target reached: val_recon_acc {val_recon_acc*100:.2f}% >= {args.recon_target_acc*100:.1f}% at step {step}", step=step)
                return


@torch.no_grad()
def eval_joint_lm(model: StridedReadoutTokenizer, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> tuple[float, float]:
    """Returns (val_recon_acc, val_code_lm_acc) — the only two metrics that matter for this objective."""
    model.eval()
    recon_accs, code_lm_accs = [], []
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.lm_context_bytes, device)
        loss, metrics = model.forward_joint_lm(ctx)
        recon_accs.append(metrics["recon_acc"].item())
        code_lm_accs.append(metrics["code_lm_acc"].item())
    model.train()
    return sum(recon_accs) / len(recon_accs), sum(code_lm_accs) / len(code_lm_accs)


def train_joint_lm(model: StridedReadoutTokenizer, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    """Joint training with the real latent LM, from random init. Only the
    required losses (recon_loss + code_lm_loss — no byte-level NTP, no
    self-distill) and only the metrics that matter (recon_acc, code_lm_acc)."""
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt4_joint_lm", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        code_lm_w = model.cfg.code_lm_weight * min(1.0, step / max(1, model.cfg.code_lm_warmup_steps))
        ctx = sample_context(train_data, args.batch_size, model.cfg.lm_context_bytes, device)
        loss, metrics = model.forward_joint_lm(ctx, code_lm_weight=code_lm_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", recon_loss=f"{metrics['recon_loss'].item():.4f}",
                          recon_acc=f"{metrics['recon_acc'].item()*100:.2f}%",
                          code_lm_loss=f"{metrics['code_lm_loss'].item():.4f}",
                          code_lm_acc=f"{metrics['code_lm_acc'].item()*100:.2f}%",
                          code_lm_w=f"{code_lm_w:.3f}")

        if step % args.eval_every == 0 or step == args.steps:
            val_recon_acc, val_code_lm_acc = eval_joint_lm(model, val_data, args.batch_size, args.eval_batches, device)
            log(f"{pbar}  val_recon_acc={val_recon_acc*100:.2f}%  val_code_lm_acc={val_code_lm_acc*100:.2f}%",
                step=step, val_recon_acc=val_recon_acc, val_code_lm_acc=val_code_lm_acc)
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step, "val_recon_acc": val_recon_acc}, 1.0 - val_recon_acc)
            if val_recon_acc >= args.recon_target_acc:
                log(f"recon target reached: val_recon_acc {val_recon_acc*100:.2f}% >= {args.recon_target_acc*100:.1f}% at step {step}", step=step)
                return


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Strided-readout tokenizer: regular large-context LM + small code_net (fork of qcute.qcutelm_vlt3)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--context_len", type=int, default=128)
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
    p.add_argument("--self_distill_weight", type=float, default=0.0)
    p.add_argument("--attn_window", type=int, default=-1, help="-1 = full causal; else banded sliding window width (recommend = context_len)")

    p.add_argument("--joint_lm", action="store_true", help="train_joint_lm: random init, recon_loss + code_lm_loss only, no byte NTP")
    p.add_argument("--lm_context_bytes", type=int, default=256)
    p.add_argument("--lm_d_model", type=int, default=128)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=4)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--code_lm_weight", type=float, default=1.0)
    p.add_argument("--code_lm_detach", type=lambda x: x.lower() != "false", default=True)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=20000)
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
        args.dq, K=args.K, context_len=args.context_len, lfq=args.lfq, quant_type=args.quant_type,
        fsq_levels=args.fsq_levels, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        mlp_mult=args.mlp_mult, code_net_layers=args.code_net_layers, ntp_loss_weight=args.ntp_loss_weight,
        recon_loss_weight=args.recon_loss_weight, self_distill_weight=args.self_distill_weight,
        attn_window=args.attn_window, lm_context_bytes=args.lm_context_bytes, lm_d_model=args.lm_d_model, lm_n_heads=args.lm_n_heads,
        lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult, code_lm_weight=args.code_lm_weight,
        code_lm_detach=args.code_lm_detach,
    )
    model = StridedReadoutTokenizer(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt4_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} context_len={cfg.context_len} dq={cfg.dq} quant_type={cfg.quant_type} "
        f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    if args.joint_lm:
        train_joint_lm(model, train_data, val_data, args, log, run_name, device)
    else:
        train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
