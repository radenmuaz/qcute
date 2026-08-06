"""qcute.qcute_fifo — "bytelm with a combinatorially expanded vocabulary":
ONE shared LM (not multiple tiers like v10-v13), but its input positions
("slots") can each represent a span of 1, 2, 4, ... raw bytes. Every span's
embedding is built compositionally from the SAME base byte-embedding table
via PQ-like binary merging (a 2-byte embed = linear(concat(byte, byte)); a
4-byte embed = linear(concat(2byte, 2byte)), recursively) — so there is
exactly one vocabulary, just recursively pooled, not a separate discrete
codebook (no BSQ/FSQ anywhere in this file).

Runtime picture (NOT implemented as an actual streaming queue in this file —
see "Not yet built" below): a FIFO of exactly `window` slots. New raw bytes
enter as fresh 1-byte slots. Whenever the slot count would exceed `window`,
the OLDEST two same-bandwidth slots merge into one higher-bandwidth slot
(carry-propagation, like a binary counter) — recent context stays byte-
resolution, old context is progressively coarsened, all within a FIXED slot
budget. Deliberately traded away: standard KV-caching does not work here,
since a merged slot's embedding is a brand-new composite input each time —
old keys/values can't be reused across a merge, they must be recomputed.
Accepted cost, not a bug to fix.

Head: ALWAYS multi-token/"JTP" (session-confirmed: "1) always mtp", not
just for spans >1) via a Fetch-style chain-probability self-attention hop
(ported from the "Fetch" mechanism description: h_t^(j) = gamma*h_t +
Emb(x_{t+j}); Fetch = h_t + SelfAttn(h_t^(0..j)); feed to the head) — every
slot predicts `bandwidth` bytes immediately following its OWN span via the
mathematically-correct chain rule (each subsequent byte's prediction
additionally conditions on the true, teacher-forced previous byte(s) in the
chain), not a naive per-byte-independent joint softmax. A bandwidth-1 slot
is a chain of length 1 (reduces to a normal single NTP step, but still
routed through the same Fetch/head code path for uniformity, per "always
mtp"). Targets are just the next `bandwidth` raw bytes after that slot's
own span — ordinary next-token-prediction, computed at every slot position,
sized to that slot's own bandwidth. Session-confirmed: "2) yes" — merges/
spans are purely a context-compression device, never something with its own
reconstruction target; only actual upcoming raw bytes are ever predicted.

Training: rather than simulating the live FIFO-merge process, each training
step samples ONE composition (a length-`window` sequence of bandwidths,
non-increasing — coarsest/oldest first, finest/newest last, matching what
the real merge cascade could produce) from the full DISCRETE SET of valid
compositions for the configured `bandwidths` alphabet and `window` size
(`enumerate_compositions`, session-requested: "3) yes... write this code"
— not per-slot random-int sampling, for tractability: for a 2-bandwidth
alphabet {1,2} and window W there are only W+1 valid compositions (a
"staircase" of some number of 2s followed by 1s), not an exponential
combinatorial blowup). Exposes the model to every window-worth-of-history
shape (from all-finest, `window` raw bytes, "minimum `window`", to
all-coarsest, `window * max(bandwidths)` raw bytes, "maximum
max(bandwidths) * window") the real inference-time merge cascade could
produce, letting it interpolate to compositions never explicitly sampled
during training (session: "let model interpolate untrained combination").

First trial scope (session-specified): `window=32`, `bandwidths=(1, 2)`
("downsample 2" — only byte and byte-pair spans, no 4-byte level yet).

Not yet built (out of scope for this file, noted for later): the literal
streaming FIFO+cascading-merge inference-time queue algorithm described in
the module's runtime picture above — `generate()` here re-embeds a freshly
sampled composition from scratch every step (no merge-state carried across
steps), consistent with every other fork in this lineage's existing no-KV-
cache simplicity tradeoff, and sufficient to validate whether the
composition-conditioned training signal is learnable at all before building
real incremental queue maintenance.

No shared imports with qcutelm_vlt/vlt2/.../vlt10 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers duplicated.

v2 design sketch (not implemented — multi-composition training via a
shared pyramid pass, genuine KV-cache compatibility via append-only
per-level caches, a survey of alternative merge-scheduling policies, and
expressivity rankings including a structured-data/audio-image lens):
see [docs/fifo_v2.md](../docs/fifo_v2.md).

    uv run python -m qcute.qcute_fifo --config configs/qcute_fifo_<name>.py
"""
import argparse
import gzip
import itertools
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
    window: int = 32           # FIFO slot budget = training sequence length in SLOT units
    bandwidths: tuple[int, ...] = (1, 2)   # allowed span sizes, ascending, each a power of 2
                                             # (2x downsample per level, matching the binary-merge tree)
    vocab: int = 256
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    fetch_n_heads: int = 2      # small self-attention module for the chain-prob MTP head
    fetch_gamma: float = 1.0    # Fetch's h_t^(j) = gamma*h_t + Emb(x_{t+j}) scaling
    rope_base: float = 10000.0
    tie_head: bool = True       # prediction head weight-tied to byte_emb (GPT-style), per the
                                 # "combinatorial expanded vocab, one shared table" design


def enumerate_compositions(window: int, bandwidths: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All non-increasing length-`window` sequences using values from
    `bandwidths` (coarsest/oldest first, finest/newest last) — the discrete,
    tractable set of window "shapes" the real FIFO+merge cascade could ever
    produce. For a 2-value alphabet this is just `window+1` "staircases"
    (k copies of the larger value then window-k of the smaller); for L
    values it's C(window+L-1, L-1) — polynomial in window, not exponential,
    so the full set can just be enumerated and sampled from directly
    (session: "for efficiency", not per-slot random-int sampling).

    e.g. enumerate_compositions(4, (1, 2)) ->
        [(1,1,1,1), (2,1,1,1), (2,2,1,1), (2,2,2,1), (2,2,2,2)]
    """
    bandwidths = tuple(sorted(bandwidths))
    n_levels = len(bandwidths)

    def counts_gen(remaining: int, levels_left: int):
        if levels_left == 1:
            yield (remaining,)
            return
        for c in range(remaining + 1):
            for rest in counts_gen(remaining - c, levels_left - 1):
                yield (c,) + rest

    comps = []
    for counts in counts_gen(window, n_levels):   # counts[i] = #slots at bandwidths[i], ascending order
        comp: list[int] = []
        for level in range(n_levels - 1, -1, -1):  # emit coarsest (highest bandwidth) first
            comp.extend([bandwidths[level]] * counts[level])
        comps.append(tuple(comp))
    return comps


def rope_cos_sin_at(position_ids: torch.Tensor, head_dim: int, base: float):
    """Like the other forks' rope_cos_sin, but for EXPLICIT (non-contiguous)
    integer positions — needed here since a slot's RoPE position must be its
    own raw-byte END offset (so relative distances stay meaningful across
    slots of different bandwidth), not just its index 0..window-1."""
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
    """Dense O(window^2) causal SDPA + RoPE — window IS the attention span
    by construction here (the FIFO slot budget), so no separate chunking
    mechanism is needed the way other forks' attn_window provides."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
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


class FetchHead(nn.Module):
    """The chain-prob MTP head: given a slot's hidden state h and its own
    bandwidth, predicts that many upcoming bytes via the Fetch mechanism —
    a single shared (weight-tied across every chain step and every slot)
    small self-attention hop combining h with gamma*h + Emb(previous
    TRUE/teacher-forced byte in the chain), then h + that attention output
    feeds the (tied) prediction head. Step 1 (no prior chain bytes yet)
    still runs through this same path for uniformity ("always mtp"), just
    with a length-1 self-attention input — not special-cased."""

    def __init__(self, cfg: Config, byte_emb: nn.Embedding, pred_head: nn.Linear):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = byte_emb
        self.pred_head = pred_head
        self.max_chain = max(cfg.bandwidths)
        self.chain_pos_emb = nn.Embedding(self.max_chain, cfg.d_model)   # distinguishes chain position
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.fetch_n_heads, batch_first=True)

    def forward(self, h: torch.Tensor, bandwidth: int, target_bytes: torch.Tensor | None) -> torch.Tensor:
        """h: [N, D] (one slot's hidden state, N = B * (however many slots
        share this bandwidth, batched together)). target_bytes: [N,
        bandwidth] true next bytes, teacher-forced (required — training
        only; use `generate_chain` for autoregressive sampling). ->
        logits: [N, bandwidth, vocab]."""
        N, D = h.shape
        chain_vecs = [h + self.chain_pos_emb.weight[0]]
        logits_list = []
        for j in range(bandwidth):
            x = torch.stack(chain_vecs, dim=1)              # [N, j+1, D]
            attn_out, _ = self.self_attn(x, x, x, need_weights=False)
            fetched = h + attn_out[:, -1, :]
            logits_list.append(self.pred_head(fetched))
            if j < bandwidth - 1:
                next_byte = target_bytes[:, j]
                chain_vecs.append(self.cfg.fetch_gamma * h + self.byte_emb(next_byte) + self.chain_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def generate_chain(self, h: torch.Tensor, bandwidth: int) -> torch.Tensor:
        """Same as forward but samples (argmax) each chain step instead of
        teacher-forcing — used at inference. Returns sampled bytes [N, bandwidth]."""
        N, D = h.shape
        chain_vecs = [h + self.chain_pos_emb.weight[0]]
        sampled = []
        for j in range(bandwidth):
            x = torch.stack(chain_vecs, dim=1)
            attn_out, _ = self.self_attn(x, x, x, need_weights=False)
            fetched = h + attn_out[:, -1, :]
            next_byte = self.pred_head(fetched).argmax(-1)
            sampled.append(next_byte)
            if j < bandwidth - 1:
                chain_vecs.append(self.cfg.fetch_gamma * h + self.byte_emb(next_byte) + self.chain_pos_emb.weight[j + 1])
        return torch.stack(sampled, dim=1)


class CombinatorialLM(nn.Module):
    """ONE shared stack. Slot embeddings are built recursively from the
    base byte table (PQ-like binary merge); the stack runs dense causal
    attention (RoPE keyed on each slot's true end-of-span byte offset, not
    its raw slot index) over exactly `window` slots; every slot's hidden
    state feeds the shared FetchHead, predicting `bandwidth` bytes
    immediately following that slot's own span."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert all(b & (b - 1) == 0 for b in cfg.bandwidths), "bandwidths must all be powers of 2 (binary merge tree)"
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        # nn.Embedding's default init (std=1/elem) is ~sqrt(d_model)x too large once this table
        # doubles as pred_head's output weight (tie_head=True) — a plain Linear(d_model, vocab)
        # would default to std ~= d_model**-0.5. Re-init so tying doesn't blow up initial logits.
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
        self.merge_proj = nn.ModuleDict({
            str(b): nn.Linear(2 * cfg.d_model, cfg.d_model) for b in cfg.bandwidths if b > 1
        })
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.pred_head = nn.Linear(cfg.d_model, cfg.vocab)
        if cfg.tie_head:
            self.pred_head.weight = self.byte_emb.weight
        self.fetch_head = FetchHead(cfg, self.byte_emb, self.pred_head)

    def embed_span(self, byte_span: torch.Tensor, bandwidth: int) -> torch.Tensor:
        """byte_span: [B, bandwidth] -> [B, D], built recursively:
        bandwidth=1 -> byte_emb directly; bandwidth=b -> merge_proj[b](cat(
        embed_span(first half, b/2), embed_span(second half, b/2)))."""
        if bandwidth == 1:
            return self.byte_emb(byte_span[:, 0])
        half = bandwidth // 2
        left = self.embed_span(byte_span[:, :half], half)
        right = self.embed_span(byte_span[:, half:], half)
        return self.merge_proj[str(bandwidth)](torch.cat([left, right], dim=-1))

    def run_blocks(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin_at(position_ids, head_dim, self.cfg.rope_base)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def build_batch(self, data: torch.Tensor, batch_size: int, composition: tuple[int, ...], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample `batch_size` examples matching `composition` (session:
        one composition per training step, applied across the whole batch
        — kept simple/tractable rather than a different composition per
        example, which would force ragged per-example context lengths).
        Returns (slot_embeds [B,W,D], position_ids [W], targets [B,W,max_bandwidth]
        padded with -100 for slots whose own bandwidth < max_bandwidth)."""
        cfg = self.cfg
        W = len(composition)
        max_bw = max(composition)

        # how many raw bytes past the window's own span (sum(composition)) do we need, to cover
        # every slot's own "predict `bandwidth` bytes immediately after me" target
        cum = 0
        needed = 0
        for b in composition:
            cum += b
            needed = max(needed, cum + b)
        starts = torch.randint(0, max(1, len(data) - needed), (batch_size,))
        ctx = torch.stack([data[s:s + needed] for s in starts]).to(device)   # [B, needed]

        embeds = []
        targets = torch.full((batch_size, W, max_bw), -100, dtype=torch.long, device=device)
        pos_ids = []
        cum = 0
        for i, b in enumerate(composition):
            span = ctx[:, cum:cum + b]
            embeds.append(self.embed_span(span, b))
            pos_ids.append(cum + b - 1)   # RoPE position = this slot's own last raw-byte offset
            tgt = ctx[:, cum + b:cum + b + b]   # next `b` bytes immediately after this slot's span
            targets[:, i, :b] = tgt
            cum += b
        slot_embeds = torch.stack(embeds, dim=1)                              # [B, W, D]
        position_ids = torch.tensor(pos_ids, device=device, dtype=torch.long)  # [W]
        return slot_embeds, position_ids, targets

    def forward(self, data: torch.Tensor, batch_size: int, composition: tuple[int, ...], device: str) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        slot_embeds, position_ids, targets = self.build_batch(data, batch_size, composition, device)
        h = self.run_blocks(slot_embeds, position_ids)   # [B, W, D]
        B, W, D = h.shape

        losses = []
        n_correct, n_total = 0, 0
        for i, b in enumerate(composition):
            h_i = h[:, i, :]                        # [B, D]
            tgt_i = targets[:, i, :b]                # [B, b]
            logits_i = self.fetch_head(h_i, b, tgt_i)   # [B, b, vocab]
            loss_i = F.cross_entropy(logits_i.reshape(-1, cfg.vocab), tgt_i.reshape(-1))
            losses.append(loss_i)
            with torch.no_grad():
                n_correct += (logits_i.argmax(-1) == tgt_i).sum().item()
                n_total += tgt_i.numel()
        loss = torch.stack(losses).mean()
        acc = n_correct / max(1, n_total)
        metrics = {"loss": loss, "acc": torch.tensor(acc)}
        return loss, metrics


def init_head_bias_to_unigram(model: CombinatorialLM, data: torch.Tensor) -> None:
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
def eval_model(model: CombinatorialLM, data: torch.Tensor, batch_size: int, n_batches: int, compositions: list[tuple[int, ...]], device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for i in range(n_batches):
        composition = compositions[i % len(compositions)]   # cycle through the full discrete set for eval
        loss, metrics = model(data, batch_size, composition, device)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item() if torch.is_tensor(v) else v)
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["loss"] / math.log(2)
    return result


def train(model: CombinatorialLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)
    compositions = enumerate_compositions(model.cfg.window, model.cfg.bandwidths)
    log(f"compositions: {len(compositions)} valid (window={model.cfg.window}, bandwidths={model.cfg.bandwidths})")

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_fifo", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        composition = compositions[torch.randint(0, len(compositions), (1,)).item()]
        loss, metrics = model(train_data, args.batch_size, composition, device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{loss.item()/math.log(2):.4f}",
                          acc=f"{metrics['acc'].item()*100:.2f}%", n2s=sum(1 for b in composition if b > 1))

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, compositions, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_bandwidths(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="bytelm with a combinatorially expanded vocabulary (FIFO-window multi-bandwidth slots, Fetch-style chain MTP head)", parents=[pre])
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--bandwidths", type=_parse_bandwidths, default=(1, 2))
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--fetch_n_heads", type=int, default=2)
    p.add_argument("--fetch_gamma", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--tie_head", type=lambda x: x.lower() != "false", default=True)

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

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    if isinstance(args.bandwidths, str):
        args.bandwidths = _parse_bandwidths(args.bandwidths)
    else:
        args.bandwidths = tuple(args.bandwidths)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        window=args.window, bandwidths=args.bandwidths, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, fetch_n_heads=args.fetch_n_heads,
        fetch_gamma=args.fetch_gamma, rope_base=args.rope_base, tie_head=args.tie_head,
    )
    model = CombinatorialLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_fifo_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"window={cfg.window} bandwidths={cfg.bandwidths} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
