"""qcute.qcute_bytepool — hierarchical byte-grouping + multi-token
prediction, no BSQ/FSQ anywhere (session: "rename v12 v13 to bytepool,
since no bsq"). Two variants, selected via `--variant`/`Config.variant`,
both in this one file (session: "include v12 in same file"):

  v13 (default) — BytepoolV13LM, described below: three INDEPENDENT LMs
    (byte/pair/quad) connected by cross-attention, a coarse-to-fine
    speculative-decoding-shaped cascade.
  v12 — BytepoolV12LM (K=2 only, see its own docstring): the ORIGINAL,
    earlier worked example — a 3-block PIPELINE (pool -> joint-predict ->
    genuine BOS-autoregressive decode) rather than 3 parallel layers; no
    cross-attention, no bandwidth-4 level.

BytepoolV13LM: three independent LMs sharing one base byte-embedding
table, each operating at its own fixed bandwidth (1, 2, 4 bytes/position),
connected by an added cross-attention hop so coarser layers can hand
context down to finer ones — a coarse-to-fine speculative-decoding-shaped
cascade, not a separate encoder/decoder split (contrast qcutelm_vlt11) and
not a single shared stack (contrast qcutelm_vlt10/qcute_fifo).

Architecture (3 layers, matching the session's worked example):
  layer1 (finest):  every byte,        bandwidth=1
  layer2:            every byte-pair,  bandwidth=2, embedding = linear(concat(byte, byte))
  layer3 (coarsest): every 4 bytes,    bandwidth=4, embedding = linear(concat(pair, pair))
Every layer's input embedding is built compositionally from the SAME base
byte table (PQ-like binary merge, identical mechanism to qcute_fifo's
`embed_span`) — "share embedding weights... each higher layer basically do
pq with first layer byte embedding table."

Each layer independently runs its own causal windowed self-attention stack
over its own (pooled) sequence length, THEN — for layer2 and layer1 only —
an added cross-attention sub-layer lets it attend into the coarser layer(s)'
hidden states (layer2 -> layer3; layer1 -> layer2 AND layer3), masked so
query position i may only see coarser key position j when key j's own byte
span is entirely <= query i's own byte span (no future leakage: `(j+1)*
k_bandwidth <= (i+1)*q_bandwidth`). This is the "additional cross attention
layer then softmax head(s) in coop mode" candidate from the session
discussion, concretized as the 3-layer draft-and-refine cascade: "layer3
act as speculative decoder... layer2 has additional cross attention layer
that mix with last output of 3rd... layer1 ... mix with last output of 3rd,
and last output of 2nd." Training uses the coarser layers' CONTINUOUS
hidden states for this mixing (not sampled/discrete draft tokens — that
distinction only matters for actual speculative-decoding inference, not
for a differentiable training pass).

Every layer ALWAYS uses an MTP head (same "always mtp" convention as
qcute_fifo) via the identical Fetch-style chain-probability self-attention
mechanism (ported from qcute_fifo.FetchHead) — layer3 chain-predicts its
own next 4 bytes, layer2 its next 2, layer1 its next 1 (a length-1 chain,
not special-cased). Each layer's loss is independent, ordinary teacher-
forced next-bytes cross-entropy at that layer's own resolution; the total
loss is their (configurably weighted) sum. Embedding table shared/tied
(byte_emb.weight, reused as every layer's respective FetchHead output
weight when `tie_heads=True`) per the session's "tied" design, using the
same re-init-to-avoid-blowup fix from qcute_fifo (nn.Embedding's default
std=1 is far too large once shared with a Linear's output role).

NOT YET BUILT, explicitly out of scope for this file (documented, session-
described but deferred):
  - Self-distillation (finer layers' cross-attention-refined logits as a
    soft KL teacher for the coarser layer's own one-shot draft logits, to
    improve future speculative-decoding accept rates).
  - The "conditional prob linear" chain ablation (autoregressive-within-
    block sampling with lightweight per-step linears, as an alternative to
    each layer's joint one-shot bandwidth-b prediction — note this is
    actually already partially present via FetchHead's own chain
    mechanism, since "always mtp" already chain-factors each layer's own
    b-byte target; the ablation would be about whether layer2/layer3's
    OWN predictions should also condition on FINER layers' outputs before
    finalizing, which isn't implemented here).
  - N-layer generalization (fixed at 3 layers / bandwidths (1,2,4) for
    this first build, matching "all layer must have same hidden dim e.g.
    256 for first trial").
  - A real generate() — the cross-attention wiring here is trained on
    teacher-forced ground truth at every layer; actual autoregressive
    (speculative-decoding-style) sampling would need each layer's SAMPLED
    draft to become the next layer's input, not built yet.

No shared imports with qcutelm_vlt/vlt2/.../vlt11/qcute_fifo (self-
contained-module convention) — Logger/Checkpointer/schedule helpers/RoPE
duplicated (FetchHead's chain mechanism ported/adapted, not imported).

    uv run python -m qcute.qcute_bytepool --config configs/qcute_bytepool_<name>.py
"""
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
    variant: str = "v13"        # "v13" (cross-attention cascade, 3 independent layers) or "v12"
                                  # (pool -> joint-predict -> BOS-decode pipeline, K=2 only — see
                                  # BytepoolV12LM's docstring)
    context_len: int = 256     # must be a multiple of 4 (layer3's bandwidth) for v13; a multiple of
                                 # 4 for v12 too (needs >=2 pair-groups for "predict the NEXT pair")
    vocab: int = 256
    d_model: int = 256          # uniform across all layers, per session spec
    n_heads: int = 4
    n_layers: int = 2           # self-attention depth, per layer/block
    mlp_mult: int = 4
    attn_window: int = -1       # uniform per-layer windowed attention; -1 = dense. must divide each
                                  # layer's own sequence length (context_len, context_len/2, context_len/4)
    cross_n_heads: int = 4
    fetch_n_heads: int = 2
    fetch_gamma: float = 1.0
    pool_n_heads: int = 4        # v12 only — PoolAttn's query-vector cross-attention heads
    rope_base: float = 10000.0
    tie_heads: bool = True
    layer1_weight: float = 1.0
    layer2_weight: float = 1.0
    layer3_weight: float = 1.0
    block1_weight: float = 1.0   # v12 only
    block2_weight: float = 1.0   # v12 only
    block3_weight: float = 1.0   # v12 only


def rope_cos_sin_at(position_ids: torch.Tensor, head_dim: int, base: float):
    """Explicit (non-contiguous) integer positions — each layer's RoPE
    position is its own raw-byte END offset, so relative distances stay
    meaningful across layers of different bandwidth (same convention as
    qcute_fifo)."""
    device = position_ids.device
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.window is not None and T % self.window == 0 and T > self.window:
            y = self._forward_chunked(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

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

        i = torch.arange(W, device=q.device).view(W, 1)
        j_prev = torch.arange(W, device=q.device).view(1, W) - W
        j_cur = torch.arange(W, device=q.device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)
        mask_per_chunk = causal_window.unsqueeze(0).expand(n_chunks, W, 2 * W).clone()
        mask_per_chunk[0, :, 0:W] = False

        qb = qc.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = k_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)
        vb = v_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)
        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, window: int | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, window=window)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


def build_cross_mask(q_len: int, k_len: int, q_bandwidth: int, k_bandwidth: int, device) -> torch.Tensor:
    """[q_len, k_len] bool mask, True = allowed. Query position i (own
    span [i*q_bandwidth, (i+1)*q_bandwidth)) may attend to coarser key
    position j (own span [j*k_bandwidth, (j+1)*k_bandwidth)) iff key j's
    span is entirely within query i's own already-seen input range:
    (j+1)*k_bandwidth <= (i+1)*q_bandwidth — no future-byte leakage."""
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    return ((j + 1) * k_bandwidth) <= ((i + 1) * q_bandwidth)


class CrossAttnBlock(nn.Module):
    """The "additional cross attention layer" that lets a finer layer mix
    in a coarser layer's hidden states — one hop, residual-added, LayerNorm
    pre-norm on the query side (coarser side is already normalized by its
    own ln_f). Masked via `build_cross_mask` (precomputed once, passed in)
    so no future information leaks across the boundary."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, Tq, D = x.shape
        Tk = kv_source.size(1)
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(self.ln(x)).view(B, Tq, H, hd).transpose(1, 2)
        kv = self.kv_proj(kv_source).view(B, Tk, 2, H, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return x + self.out(out.transpose(1, 2).reshape(B, Tq, D))


class FetchHead(nn.Module):
    """Ported from qcute_fifo.FetchHead — chain-probability MTP head, fixed
    `bandwidth` for this layer (no per-call bandwidth arg, unlike
    qcute_fifo's combinatorial version, since every layer here always
    predicts the same number of bytes)."""

    def __init__(self, cfg: Config, bandwidth: int, byte_emb: nn.Embedding, pred_head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        self.bandwidth = bandwidth
        self.byte_emb = byte_emb
        self.pred_head = pred_head
        self.chain_pos_emb = nn.Embedding(max(1, bandwidth), cfg.d_model)
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.fetch_n_heads, batch_first=True)

    def forward(self, h: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
        """h: [N, D], target_bytes: [N, bandwidth] -> logits [N, bandwidth, vocab]."""
        b = self.bandwidth
        chain_vecs = [h + self.chain_pos_emb.weight[0]]
        logits_list = []
        for j in range(b):
            x = torch.stack(chain_vecs, dim=1)
            attn_out, _ = self.self_attn(x, x, x, need_weights=False)
            fetched = h + attn_out[:, -1, :]
            logits_list.append(self.pred_head(fetched))
            if j < b - 1:
                next_byte = target_bytes[:, j]
                chain_vecs.append(self.cfg.fetch_gamma * h + self.byte_emb(next_byte) + self.chain_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


class BytepoolV13LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.context_len % 4 == 0, "context_len must be a multiple of 4 (layer3's bandwidth)"
        window = None if cfg.attn_window == -1 else cfg.attn_window

        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)   # see qcute_fifo — avoids
                                                                                     # tie_heads blowing up logits
        self.pair_merge = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.quad_merge = nn.Linear(2 * cfg.d_model, cfg.d_model)

        self.layer1_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.layer1_ln_f = nn.LayerNorm(cfg.d_model)
        self.layer2_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.layer2_ln_f = nn.LayerNorm(cfg.d_model)
        self.layer3_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.layer3_ln_f = nn.LayerNorm(cfg.d_model)

        self.cross_1_to_2 = CrossAttnBlock(cfg.d_model, cfg.cross_n_heads)   # layer1 attends layer2
        self.cross_1_to_3 = CrossAttnBlock(cfg.d_model, cfg.cross_n_heads)   # layer1 attends layer3
        self.cross_2_to_3 = CrossAttnBlock(cfg.d_model, cfg.cross_n_heads)   # layer2 attends layer3

        self.pred_head = nn.Linear(cfg.d_model, cfg.vocab)
        if cfg.tie_heads:
            self.pred_head.weight = self.byte_emb.weight
        self.fetch1 = FetchHead(cfg, 1, self.byte_emb, self.pred_head)
        self.fetch2 = FetchHead(cfg, 2, self.byte_emb, self.pred_head)
        self.fetch3 = FetchHead(cfg, 4, self.byte_emb, self.pred_head)

    def run_stack(self, blocks: nn.ModuleList, ln_f: nn.LayerNorm, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin_at(position_ids, head_dim, self.cfg.rope_base)
        for block in blocks:
            x = block(x, cos, sin)
        return ln_f(x)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        D = cfg.d_model
        device = ctx.device

        byte_embed = self.byte_emb(ctx)                                             # [B, L, D]
        pair_embed = self.pair_merge(byte_embed.view(B, L // 2, 2, D).flatten(-2))    # [B, L/2, D]
        quad_embed = self.quad_merge(pair_embed.view(B, L // 4, 2, D).flatten(-2))    # [B, L/4, D]

        pos1 = torch.arange(L, device=device)
        pos2 = torch.arange(1, L // 2 + 1, device=device) * 2 - 1
        pos3 = torch.arange(1, L // 4 + 1, device=device) * 4 - 1

        h3 = self.run_stack(self.layer3_blocks, self.layer3_ln_f, quad_embed, pos3)   # [B, L/4, D]
        h2_self = self.run_stack(self.layer2_blocks, self.layer2_ln_f, pair_embed, pos2)
        mask_2_3 = build_cross_mask(L // 2, L // 4, 2, 4, device)
        h2 = self.cross_2_to_3(h2_self, h3, mask_2_3)                                 # [B, L/2, D]
        h1_self = self.run_stack(self.layer1_blocks, self.layer1_ln_f, byte_embed, pos1)
        mask_1_2 = build_cross_mask(L, L // 2, 1, 2, device)
        mask_1_3 = build_cross_mask(L, L // 4, 1, 4, device)
        h1 = self.cross_1_to_2(h1_self, h2, mask_1_2)
        h1 = self.cross_1_to_3(h1, h3, mask_1_3)                                      # [B, L, D]

        def layer_loss(h: torch.Tensor, fetch: FetchHead, bandwidth: int) -> tuple[torch.Tensor, torch.Tensor]:
            """h[:, i] covers own span [i*bandwidth, (i+1)*bandwidth); its
            target is the NEXT span, [(i+1)*bandwidth, (i+2)*bandwidth) —
            always drop the last position, whose target would run past L."""
            n = h.size(1) - 1
            h_flat = h[:, :n, :].reshape(B * n, D)
            tgt_list = [ctx[:, (i + 1) * bandwidth:(i + 2) * bandwidth] for i in range(n)]
            tgt = torch.stack(tgt_list, dim=1).reshape(B * n, bandwidth)   # [B*n, bandwidth]
            logits = fetch(h_flat, tgt)                                    # [B*n, bandwidth, vocab]
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), tgt.reshape(-1))
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt).float().mean()
            return loss, acc

        loss1, acc1 = layer_loss(h1, self.fetch1, 1)
        loss2, acc2 = layer_loss(h2, self.fetch2, 2)
        loss3, acc3 = layer_loss(h3, self.fetch3, 4)

        loss = cfg.layer1_weight * loss1 + cfg.layer2_weight * loss2 + cfg.layer3_weight * loss3
        metrics = {
            "loss": loss, "bpb_loss": loss1,   # layer1 = plain byte-level NTP, comparable to bytelm/bpelm
            "layer1_loss": loss1, "layer1_acc": acc1,
            "layer2_loss": loss2, "layer2_acc": acc2,
            "layer3_loss": loss3, "layer3_acc": acc3,
        }
        return loss, metrics


class PoolAttn(nn.Module):
    """v12 only — encode-side pooling: a single learned query vector
    attends (one cross-attention hop) over its 2 children -> 1 pooled
    vector. Ported from qcutelm_vlt11.PoolAttn (Perceiver-style, block-
    local: called with children already reshaped to [N, 2, D])."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=d_model ** -0.5)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        N = children.size(0)
        q = self.query.view(1, 1, -1).expand(N, 1, -1)
        out, _ = self.attn(q, children, children, need_weights=False)
        return out.squeeze(1)


class DecodeBlock(nn.Module):
    """v12 only — BOS-style block-local byte decode: parent (1 embedding)
    as a prefix, K byte children generated teacher-forced via a tiny
    causal stack. Ported from qcutelm_vlt11.DecodeBlock's is_byte_level
    branch (v12 has no code-child case — no BSQ/FSQ anywhere in this
    file)."""

    def __init__(self, cfg: Config, byte_emb: nn.Embedding, pred_head: nn.Linear, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = byte_emb
        self.pred_head = pred_head
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        pos = torch.arange(x.size(1), device=x.device)
        cos, sin = rope_cos_sin_at(pos, head_dim, self.cfg.rope_base)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def forward(self, parent_emb: torch.Tensor, target_children: torch.Tensor) -> torch.Tensor:
        """parent_emb: [N, D]. target_children: [N, K] byte ids -> logits: [N, K, vocab]."""
        K = target_children.size(1)
        bos = parent_emb.unsqueeze(1)
        child_in = self.byte_emb(target_children[:, :-1]) if K > 1 else target_children.new_zeros(target_children.size(0), 0, self.cfg.d_model)
        dec_in = torch.cat([bos, child_in], dim=1) if K > 1 else bos
        h = self.run_blocks(dec_in)
        return self.pred_head(h)


class BytepoolV12LM(nn.Module):
    """v12 (K=2 only — see module docstring's "NOT YET BUILT" note for why
    K=4's deeper recursive generalization isn't built here): pool -> joint-
    predict -> BOS-decode, matching the session's original worked example
    exactly, as 3 explicit blocks:

    block1: byte embeddings, causal self-attn -> h1. Plain byte NTP
        (bandwidth=1, via FetchHead for "always mtp" consistency with
        v13) at every position. ALSO: pool each adjacent pair of h1
        positions (PoolAttn, query-vector cross-attention) -> ab_1
        [B, L/2, D] ("emit emb ab_1 with lm query vec").
    block2: input = ab_1, causal self-attn over pair-groups -> h2 = ab_2
        ("emit emb ab_2" — here just block2's own output hidden state; no
        further pooling needed since K=2 has only one pooling level).
        JOINTLY predicts the NEXT pair-group's 2 bytes in one FetchHead
        call (bandwidth=2, chain-factored) — "first softmax ntp c, second
        softmax ntp d in single timestep".
    block3: ab_2 (=h2) as BOS -> DecodeBlock (K=2 byte children) ->
        genuine autoregressive teacher-forced byte decode of the SAME 2
        target bytes block2 predicted jointly — "use ab_2 as bos, byte
        level ntp c,d". This is the actual generative test: can a real
        BOS-decode reconstruct the same targets block2 only guessed at
        jointly.

    All three blocks' losses are independent (own targets, teacher-forced,
    (configurably) weighted, summed) — same pattern as v13's 3 layers."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.context_len % 4 == 0, "context_len must be a multiple of 4 (need >=2 pair-groups so block2/3 have a next-group target)"
        window = None if cfg.attn_window == -1 else cfg.attn_window

        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)

        self.block1_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.block1_ln_f = nn.LayerNorm(cfg.d_model)
        self.pool_attn = PoolAttn(cfg.d_model, cfg.pool_n_heads)
        self.block2_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.n_layers)])
        self.block2_ln_f = nn.LayerNorm(cfg.d_model)

        self.pred_head = nn.Linear(cfg.d_model, cfg.vocab)
        if cfg.tie_heads:
            self.pred_head.weight = self.byte_emb.weight
        self.fetch1 = FetchHead(cfg, 1, self.byte_emb, self.pred_head)
        self.fetch2 = FetchHead(cfg, 2, self.byte_emb, self.pred_head)
        self.block3 = DecodeBlock(cfg, self.byte_emb, self.pred_head, cfg.n_layers)

    def run_stack(self, blocks: nn.ModuleList, ln_f: nn.LayerNorm, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin_at(position_ids, head_dim, self.cfg.rope_base)
        for block in blocks:
            x = block(x, cos, sin)
        return ln_f(x)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        D = cfg.d_model
        device = ctx.device

        byte_embed = self.byte_emb(ctx)
        pos1 = torch.arange(L, device=device)
        h1 = self.run_stack(self.block1_blocks, self.block1_ln_f, byte_embed, pos1)   # [B, L, D]

        n_pairs = L // 2
        ab_1 = self.pool_attn(h1.view(B * n_pairs, 2, D)).view(B, n_pairs, D)          # [B, L/2, D]
        pos2 = torch.arange(1, n_pairs + 1, device=device) * 2 - 1
        h2 = self.run_stack(self.block2_blocks, self.block2_ln_f, ab_1, pos2)          # [B, L/2, D] = ab_2

        # block1: plain byte NTP at every position (bandwidth=1)
        h1_flat = h1[:, :-1, :].reshape(B * (L - 1), D)
        tgt1 = ctx[:, 1:].reshape(B * (L - 1), 1)
        logits1 = self.fetch1(h1_flat, tgt1)
        loss1 = F.cross_entropy(logits1.reshape(-1, cfg.vocab), tgt1.reshape(-1))
        with torch.no_grad():
            acc1 = (logits1.argmax(-1) == tgt1).float().mean()

        # block2: joint one-shot prediction of the NEXT pair-group's 2 bytes
        n_use = n_pairs - 1
        h2_flat = h2[:, :n_use, :].reshape(B * n_use, D)
        tgt23 = torch.stack([ctx[:, 2 * (i + 1):2 * (i + 1) + 2] for i in range(n_use)], dim=1).reshape(B * n_use, 2)
        logits2 = self.fetch2(h2_flat, tgt23)
        loss2 = F.cross_entropy(logits2.reshape(-1, cfg.vocab), tgt23.reshape(-1))
        with torch.no_grad():
            acc2 = (logits2.argmax(-1) == tgt23).float().mean()

        # block3: genuine BOS-autoregressive decode of the SAME target bytes
        logits3 = self.block3(h2_flat, tgt23)
        loss3 = F.cross_entropy(logits3.reshape(-1, cfg.vocab), tgt23.reshape(-1))
        with torch.no_grad():
            acc3 = (logits3.argmax(-1) == tgt23).float().mean()

        loss = cfg.block1_weight * loss1 + cfg.block2_weight * loss2 + cfg.block3_weight * loss3
        metrics = {
            "loss": loss, "bpb_loss": loss1,
            "block1_loss": loss1, "block1_acc": acc1,
            "block2_loss": loss2, "block2_acc": acc2,
            "block3_loss": loss3, "block3_acc": acc3,
        }
        return loss, metrics


def init_head_bias_to_unigram(model: nn.Module, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        model.pred_head.bias.copy_(log_freq.to(model.pred_head.bias.device))


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
def eval_model(model: nn.Module, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["bpb_loss"] / math.log(2)   # byte-level (finest) NTP, comparable to bytelm/bpelm
    return result


def train(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc=f"train_bytepool_{model.cfg.variant}", dynamic_ncols=True)
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

        postfix = dict(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{metrics['bpb_loss'].item()/math.log(2):.4f}")
        for k, v in metrics.items():
            if k.endswith("_acc"):
                postfix[k] = f"{v.item()*100:.1f}%"
        pbar.set_postfix(**postfix)

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Hierarchical byte-grouping + MTP, no BSQ/FSQ — v13 (cross-attention cascade) or v12 (pool->joint-predict->BOS-decode) (qcute_bytepool)", parents=[pre])
    p.add_argument("--variant", type=str, default="v13", choices=["v12", "v13"])
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--cross_n_heads", type=int, default=4)
    p.add_argument("--fetch_n_heads", type=int, default=2)
    p.add_argument("--fetch_gamma", type=float, default=1.0)
    p.add_argument("--pool_n_heads", type=int, default=4)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--tie_heads", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--layer1_weight", type=float, default=1.0)
    p.add_argument("--layer2_weight", type=float, default=1.0)
    p.add_argument("--layer3_weight", type=float, default=1.0)
    p.add_argument("--block1_weight", type=float, default=1.0)
    p.add_argument("--block2_weight", type=float, default=1.0)
    p.add_argument("--block3_weight", type=float, default=1.0)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        variant=args.variant, context_len=args.context_len, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, attn_window=args.attn_window, cross_n_heads=args.cross_n_heads,
        fetch_n_heads=args.fetch_n_heads, fetch_gamma=args.fetch_gamma, pool_n_heads=args.pool_n_heads,
        rope_base=args.rope_base, tie_heads=args.tie_heads, layer1_weight=args.layer1_weight,
        layer2_weight=args.layer2_weight, layer3_weight=args.layer3_weight, block1_weight=args.block1_weight,
        block2_weight=args.block2_weight, block3_weight=args.block3_weight,
    )
    model = (BytepoolV12LM(cfg) if cfg.variant == "v12" else BytepoolV13LM(cfg)).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_bytepool_{cfg.variant}_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"variant={cfg.variant} context_len={cfg.context_len} d_model={cfg.d_model} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
