"""qcute.qcutelm_vlt6 — the tokenizer IS an AR latent LM, single loss.

Pipeline (per conversation): bytes -> encoder -> code -> codelm -> codepred
(factorized softmax/sigmoids) -> next code -> decoder -> bytes. Only ONE
loss: byte-level next-token-prediction cross-entropy, evaluated at the
decoder's output (the "last layer") — no reconstruction loss, no
code-level classification loss, no auxiliary byte-level NTP loss on the
encoder's own hidden states (qcutelm_vlt4/vlt5 both had this; removed
here). Nothing else trains any part of this pipeline.

This is a deliberate departure from every earlier variant in this file's
lineage, which all trained the tokenizer as an AUTOENCODER: code(block i)
is computed FROM block i's own bytes, then used to reconstruct THOSE SAME
bytes (qcutelm_vlt/vlt2/vlt3/vlt4/vlt5's decode/recon paths). Here the
decoder never sees the bytes it's asked to predict — code(i) is still
computed from block i's own bytes (same strided-readout encoder as
before), but codelm processes the sequence of TRUE codes causally
(teacher-forced, like a normal LM) and codepred predicts a soft/
differentiable representation of code(i+1) *before* ever seeing block
i+1 — that predicted code is what the decoder reconstructs from, and the
loss compares against block i+1's REAL bytes. Genuine held-out
next-block prediction, not autoencoding — this is the actual generative
test the whole "continuous tokenizer" project is aiming at, done as one
single end-to-end objective instead of a staged tokenizer-then-LM
pipeline.

codepred has no loss of its own ("no other loss") — its bit-logits
(BSQ/LFQ: dq independent sigmoids) or per-dimension logits (FSQ/iFSQ: dq
independent softmaxes over fsq_levels) are converted to a soft,
differentiable code representation (matching bsq_quantize's/
fsq_quantize's own value range, so it's in-distribution for the decoder)
and fed straight into decode — codepred's parameters get ALL their
gradient indirectly, through the decoder's byte NTP loss backpropping
through the whole chain (encoder -> code -> codelm -> codepred -> decode).
No detach anywhere.

Context (per conversation, referring to earlier session history):
qcutelm.py's original design first tried a "loosely coupled" tokenizer+LM
combination (weighted-sum of separate losses) and, in this session's
`qcutelm_vlt4`/`vlt5` --joint_lm/CodeLM experiments, a "tightly coupled"
version that still kept a reconstruction loss as the dominant signal
shaping the code space (CodeLM only ever refined/contextualized a code
computed from the SAME block it reconstructs). This fork removes the
reconstruction crutch entirely — the tokenizer itself IS the AR LM, code
quality is judged purely by whether the pipeline can predict genuinely
unseen future bytes.

No shared imports with qcutelm_vlt/vlt2/vlt3/vlt4/vlt5 (self-contained-
module convention) — Logger/Checkpointer/schedule helpers, quantizers,
and the chunked sliding-window attention (verified in qcutelm_vlt5.py)
duplicated verbatim.
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


@dataclass
class Config:
    K: int = 4
    context_len: int = 512
    attn_window: int = 16
    dq: int = 18
    lfq: bool = False
    quant_type: str = "bsq"     # "bsq"/"lfq" -> factorized sigmoids; "fsq"/"ifsq" -> factorized softmax
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_mult: int = 4
    code_net_layers: int = 0
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
    """Causal self-attention with a zero-KV sink; window=None -> dense
    O(T^2) path, window=N -> chunked O(T*N) path (see qcutelm_vlt5.py,
    where both paths were numerically verified to match exactly)."""

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
        if self.window is not None and T % self.window == 0 and T > self.window:
            y = self._forward_chunked(q, k, v)
        else:
            y = self._forward_dense(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def _forward_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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
        B, H, T, hd = q.shape
        W = self.window
        n_chunks = T // W

        def to_chunks(t):
            return t.view(B, H, n_chunks, W, hd)

        qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
        zero_chunk = torch.zeros(B, H, 1, W, hd, device=q.device, dtype=q.dtype)
        kc_prev = torch.cat([zero_chunk, kc[:, :, :-1]], dim=2)
        vc_prev = torch.cat([zero_chunk, vc[:, :, :-1]], dim=2)
        k_local = torch.cat([kc_prev, kc], dim=3)
        v_local = torch.cat([vc_prev, vc], dim=3)
        zero_kv_sink = torch.zeros(B, H, n_chunks, 1, hd, device=q.device, dtype=q.dtype)
        k_local = torch.cat([zero_kv_sink, k_local], dim=3)
        v_local = torch.cat([zero_kv_sink, v_local], dim=3)

        i = torch.arange(W, device=q.device).view(W, 1)
        j_prev = torch.arange(W, device=q.device).view(1, W) - W
        j_cur = torch.arange(W, device=q.device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)
        mask = torch.cat([torch.ones(W, 1, dtype=torch.bool, device=q.device), causal_window], dim=1)
        mask_per_chunk = mask.unsqueeze(0).expand(n_chunks, W, 1 + 2 * W).clone()
        mask_per_chunk[0, :, 1:1 + W] = False  # chunk 0 has no real previous chunk

        qb = qc.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = k_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 1 + 2 * W, hd)
        vb = v_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 1 + 2 * W, hd)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 1 + 2 * W).reshape(B * n_chunks, 1, W, 1 + 2 * W)
        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
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
    """Causal transformer over the sequence of TRUE codes (teacher-forced,
    like a normal LM) + a codepred head producing a soft, differentiable
    representation of the NEXT code — factorized sigmoids (BSQ/LFQ: dq
    independent bits), factorized softmax (FSQ/iFSQ: dq independent
    per-dimension categoricals over fsq_levels), or a plain unbounded
    linear prediction (quant_type="none": no quantization/bottleneck at
    all, fully continuous code). No loss of its own; see module docstring."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Identity() if cfg.lm_d_model == cfg.d_model else nn.Linear(cfg.d_model, cfg.lm_d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult, window=None) for _ in range(cfg.lm_n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        self.pred_head = nn.Linear(cfg.lm_d_model, cfg.dq * cfg.fsq_levels if factorized_softmax else cfg.dq)
        if factorized_softmax:
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l  # matches fsq_quantize's (-1,1) rescale
            self.register_buffer("level_values", level_values)

    def forward(self, code_embed_true: torch.Tensor) -> torch.Tensor:
        """code_embed_true: [B, n_blocks, d_model] (z_proj of the TRUE
        codes) -> pred_soft: [B, n_blocks, dq], a soft/differentiable
        prediction of the NEXT position's code (pred_soft[:, i] predicts
        code[i+1]; caller drops the last, target-less position)."""
        x = self.in_proj(code_embed_true)
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        raw = self.pred_head(h)
        if self.cfg.quant_type == "none":
            return raw  # unbounded continuous prediction — matches quantize()'s identity passthrough
        if self.cfg.quant_type in ("fsq", "ifsq"):
            B, T, _ = raw.shape
            logits = raw.view(B, T, self.cfg.dq, self.cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1)  # [B, T, dq], soft expected level, range (-1, 1)
        return 2 * torch.sigmoid(raw) - 1  # [B, T, dq], soft bit values, range (-1, 1) — matches bsq_quantize's sign()


class ARLatentTokenizer(nn.Module):
    """The tokenizer IS the AR LM — see module docstring for the full
    pipeline and single-loss rationale."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.context_len % cfg.K == 0, "context_len must be a multiple of K"
        self.cfg = cfg
        window = None if cfg.attn_window == -1 else cfg.attn_window
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)  # shared: encoder pass AND decode pass
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
        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)  # true code AND predicted code both go through this
        self.codelm = CodeLM(cfg)

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def quantize(self, raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "none":
            return raw  # no bottleneck at all — fully continuous code, identity passthrough
        if self.cfg.quant_type == "fsq":
            return fsq_quantize(raw, self.cfg.fsq_levels, bound="tanh")
        if self.cfg.quant_type == "ifsq":
            return fsq_quantize(raw, self.cfg.fsq_levels, bound="sigmoid")
        return bsq_quantize(raw, self.cfg.dq, self.cfg.lfq)

    def codes_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        K = self.cfg.K
        stride_h = h[:, K - 1::K, :]
        return self.quantize(self.code_net(stride_h))

    def decode_block(self, code_soft: torch.Tensor, target_block: torch.Tensor) -> torch.Tensor:
        """code_soft: [N, dq] (a TRUE or PREDICTED code, same z_proj either
        way), target_block: [N, K] teacher-forcing targets -> logits: [N, K, vocab]."""
        N, K = target_block.shape
        bos = self.z_proj(code_soft).unsqueeze(1)
        if K > 1:
            dec_in = torch.cat([bos, self.byte_emb(target_block[:, :-1])], dim=1)
        else:
            dec_in = bos
        dec_h = self.run_blocks(dec_in)
        return self.head(dec_h)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        K = cfg.K
        n_blocks = L // K

        h = self.run_blocks(self.byte_emb(ctx))          # encoder pass (windowed/chunked attn)
        z_hat = self.codes_from_hidden(h)                  # [B, n_blocks, dq] — TRUE codes, one per block

        code_embed_true = self.z_proj(z_hat)                # [B, n_blocks, d_model] — codelm's own input (teacher-forced)
        pred_soft = self.codelm(code_embed_true)[:, :-1, :]  # [B, n_blocks-1, dq] — predicts codes[1:] from codes[:-1]

        target_blocks = ctx.view(B, n_blocks, K)[:, 1:, :]  # [B, n_blocks-1, K] — the blocks being predicted
        pred_soft_flat = pred_soft.reshape(B * (n_blocks - 1), cfg.dq)
        target_flat = target_blocks.reshape(B * (n_blocks - 1), K)

        dec_logits = self.decode_block(pred_soft_flat, target_flat)  # never saw target_flat's bytes before this
        loss = F.cross_entropy(dec_logits.reshape(-1, cfg.vocab), target_flat.reshape(-1))  # THE only loss
        acc = (dec_logits.argmax(-1) == target_flat).float().mean()
        return loss, {"loss": loss, "next_block_acc": acc}


def init_head_bias_to_unigram(model: ARLatentTokenizer, data: torch.Tensor) -> None:
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
def decode_greedy_block(model: ARLatentTokenizer, code_soft: torch.Tensor, K: int) -> torch.Tensor:
    """code_soft: [B, dq] (a predicted, never-quantized code) -> generated
    bytes: [B, K], byte-by-byte greedy autoregressive decode from that
    code as BOS — the actual generation-time counterpart of decode_block's
    teacher-forced training-time version (no future bytes available here)."""
    B = code_soft.size(0)
    bos = model.z_proj(code_soft).unsqueeze(1)
    seq = bos
    out_bytes = []
    for _ in range(K):
        h = model.run_blocks(seq)
        logits = model.head(h[:, -1, :])
        next_byte = logits.argmax(-1)
        out_bytes.append(next_byte)
        seq = torch.cat([seq, model.byte_emb(next_byte).unsqueeze(1)], dim=1)
    return torch.stack(out_bytes, dim=1)


@torch.no_grad()
def generate(model: ARLatentTokenizer, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """prompt_bytes: [L0] (L0 a positive multiple of K) -> [L0 + n_new_blocks*K]
    generated continuation. Block-level AR (predict next code from purely
    causal past codes via codelm, same as training) nested with byte-level
    AR (decode_greedy_block) — genuine generation, no ground truth used
    anywhere in this function."""
    cfg = model.cfg
    K = cfg.K
    was_training = model.training
    model.eval()
    ctx = prompt_bytes.to(device)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    assert ctx.size(1) % K == 0 and ctx.size(1) >= K, "prompt length must be a positive multiple of K"
    n_new_blocks = max(1, n_new_bytes // K)
    for _ in range(n_new_blocks):
        h = model.run_blocks(model.byte_emb(ctx))
        z_hat = model.codes_from_hidden(h)
        code_embed_true = model.z_proj(z_hat)
        codelm_out = model.codelm(code_embed_true)  # [1, n_blocks, dq]
        pred_next = codelm_out[:, -1, :]              # prediction for the block AFTER the last known one
        new_block = decode_greedy_block(model, pred_next, K)
        ctx = torch.cat([ctx, new_block], dim=1)
    if was_training:
        model.train()
    return ctx[0]


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def qualitative_gen(model: ARLatentTokenizer, data: torch.Tensor, prompt_len: int, n_new_bytes: int, device: str, log, step: int) -> None:
    """Fixed-offset (start=0 of `data`) prompt for reproducibility across
    evals — same seed every call, so generation quality over training is
    directly comparable step to step. Logs prompt, generated continuation,
    and the REAL ground-truth continuation side by side."""
    K = model.cfg.K
    prompt_len = max(K, (prompt_len // K) * K)
    n_new_bytes = max(K, (n_new_bytes // K) * K)
    if len(data) < prompt_len + n_new_bytes:
        return
    prompt = data[:prompt_len]
    ground_truth = data[prompt_len:prompt_len + n_new_bytes]
    gen_full = generate(model, prompt, n_new_bytes, device)
    generated = gen_full[prompt_len:]
    match = (generated.cpu() == ground_truth).float().mean().item()
    log(f"[gen@step{step}] prompt={_bytes_repr(prompt)}", step=step)
    log(f"[gen@step{step}] generated={_bytes_repr(generated)}", step=step)
    log(f"[gen@step{step}] ground_truth={_bytes_repr(ground_truth)}  byte_match={match*100:.2f}%",
        step=step, gen_byte_match=match)


@torch.no_grad()
def eval_model(model: ARLatentTokenizer, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> tuple[float, float]:
    """Returns (val_bpb, val_next_block_acc)."""
    model.eval()
    losses, accs = [], []
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        losses.append(loss.item())
        accs.append(metrics["next_block_acc"].item())
    model.train()
    bpb = (sum(losses) / len(losses)) / math.log(2)
    return bpb, sum(accs) / len(accs)


def train(model: ARLatentTokenizer, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt6", dynamic_ncols=True)
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

        train_bpb = loss.item() / math.log(2)
        acc = metrics["next_block_acc"].item()
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", train_bpb=f"{train_bpb:.4f}",
                          next_block_acc=f"{acc*100:.2f}%")

        if step % args.eval_every == 0 or step == args.steps:
            val_bpb, val_acc = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            log(f"{pbar}  val_bpb={val_bpb:.4f}  val_next_block_acc={val_acc*100:.2f}%",
                step=step, train_bpb=train_bpb, train_next_block_acc=acc, val_bpb=val_bpb, val_next_block_acc=val_acc)
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step, "val_bpb": val_bpb}, val_bpb)

        if args.gen_every > 0 and (step % args.gen_every == 0 or step == args.steps):
            qualitative_gen(model, val_data, args.gen_prompt_len, args.gen_new_bytes, device, log, step)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Tokenizer-as-AR-LM, single byte-NTP loss (fork of qcute.qcutelm_vlt5)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--attn_window", type=int, default=16)
    p.add_argument("--dq", type=int, default=None)
    p.add_argument("--lfq", action="store_true")
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "fsq", "ifsq", "none"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--code_net_layers", type=int, default=0)
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
    p.add_argument("--gen_every", type=int, default=1000, help="0 = off")
    p.add_argument("--gen_prompt_len", type=int, default=64)
    p.add_argument("--gen_new_bytes", type=int, default=64)

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
        lm_d_model=args.lm_d_model, lm_n_heads=args.lm_n_heads, lm_n_layers=args.lm_n_layers,
        lm_mlp_mult=args.lm_mlp_mult,
    )
    model = ARLatentTokenizer(cfg).to(device)
    n_tokenizer = sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith("codelm"))
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))
    n_params = n_tokenizer + n_codelm

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt6_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    quant_str = f"{cfg.quant_type}(levels={cfg.fsq_levels})" if cfg.quant_type in ("fsq", "ifsq") else cfg.quant_type
    log(f"K={cfg.K} context_len={cfg.context_len} attn_window={cfg.attn_window} dq={cfg.dq} "
        f"quant_type={quant_str} params={n_params/1e6:.3f}M "
        f"(tokenizer={n_tokenizer/1e6:.3f}M codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
