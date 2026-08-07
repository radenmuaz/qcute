"""qcute.qcutelm_vlt8 — hybrid: symmetric narrow tokenizer at the edges,
heavy separate-weight codelm in the middle, operating on the short code
sequence.

Lineage within qcutelm_vlt7.py (kept there, not re-forked, per "v7 direct"):
  v1: codes interleaved in-stream (bytes...code...bytes...code), one
      shared stack playing both encoder and decoder roles, one loss.
  v2 (folded in): reserved zero-vector slot in encoder mode too, so both
      modes share sequence length/RoPE positions exactly — fixed a real
      positional inconsistency the first version had.
  v3 (this version): a SEPARATE, wide codelm sandwiched between two calls
      of the SAME narrow tokenizer stack.

Why the fold to v3 was necessary (session discussion — "what is the point
vs bytelm then, should be structurally similar like v6 cheap tokenizer lm
first last, heavy code lm in middle"): v1/v2 had NO advantage over
bytelm at all. A single shared stack, run twice over ~the full
byte-length sequence, at one width, is strictly worse than a plain
byte-level LM of the same width run once — no compression benefit,
2x the compute. The reason: qcute's whole compute argument (matching
qcutelm_vlt6's design) comes from a WIDTH/LENGTH split — a narrow stack
touches the full byte-length sequence, a wide stack only ever touches
the short, K-fold-compressed code sequence (n_blocks = context_len/K
tokens). v1/v2 collapsed that split away in the name of architectural
symmetry, and lost the entire point in the process. Getting the split
back requires a component that does genuine forward FORECASTING over the
short code sequence — which necessarily means predicting a code before
its block's bytes exist, reintroducing exactly the code-prediction
machinery (code_match_loss, no new loss function needed — same mechanism
qcutelm_vlt6's CodeLM already uses) that v1/v2 had specifically avoided.
That tradeoff is unavoidable: "codes are never predicted" and "compute
savings from operating at code level" cannot both hold in one shared
stack.

Architecture now:
  Pass 1 (tokenizer tier, narrow d_model, "no-code" mode): causal, RoPE,
    zero vector at every reserved K-th slot — bytes: t1 t2 0 t3 t4 0 ...
    Gives (a) a deterministic readout of each block's TRUE code (strided
    read at the zero-slot's own hidden state, exactly as before) and (b)
    a free same-weights "no code at all" baseline loss/accuracy at the
    same slot positions.
  CodeLM tier (SEPARATE weights, wide lm_d_model, operates ONLY on the
    n_blocks-length sequence of true codes — never touches raw bytes at
    all): causal forecaster, predicts code[i+1] from codes[:i] (teacher-
    forced). Trained via code_match_loss — BCE/bit (bsq) or CE/dim
    (fsq/ifsq) against the tokenizer's own detached next code — the
    SAME mechanism qcutelm_vlt6's CodeLM/code_match_weight already uses,
    no new loss function invented for this.
  Pass 2 (tokenizer tier, SAME shared narrow weights as Pass 1,
    "forecast" mode): bytes: t1 t2 pred_c1 t3 t4 pred_c2 ... — the slot at
    each position holds CODELM's PREDICTED next code (not the block's own
    true code), so the loss here is the genuine generative test: can the
    narrow tokenizer decode a real future block using only codelm's
    forecast, the same shape of question qcutelm_vlt6's main_ntp answers
    via decode(codepred(codelm(...))) vs block[i+1]. Gradient flows back
    into codelm through this path too (pred_soft not detached here) —
    only the code_match_loss's OWN target is detached.

Where compute savings actually live now: codelm is the only WIDE
component, and it only ever processes n_blocks tokens (the codes), never
the full context_len-byte sequence — matching qcutelm_vlt6's own
narrow-encoder/wide-codelm split. The tokenizer tier stays narrow and
symmetric (single shared weights, single reserved-slot format, RoPE-
consistent between its two modes) — that's the part of v1/v2 worth
keeping; the part NOT worth keeping was pretending the whole model could
stay a single shared stack.

KNOWN GAP, fixed in qcutelm_vlt7: v1/v2/early-v3 had no windowing/
chunking on the tokenizer tier's O(T^2) attention. `attn_window` ports
qcutelm_vlt6's verified _forward_chunked_no_sink mechanism onto the
tokenizer tier's Blocks — O(T*window) instead of O(T^2). codelm's own
attention is left dense (its sequence is already short, windowing it
isn't the point).

v8 (this fork, from qcutelm_vlt7): fixes a real misalignment in that
windowing. The interleaved sequence is (K+1)-periodic (K bytes then one
code slot, per build_interleaved), but `attn_window` was an arbitrary
raw-position count with no relationship to that period — e.g.
context_len=1024, K=4, attn_window=64: block 12 spans positions
[60,64] (bytes 60-63, code slot at 64), and the chunk boundary at
position 64 SPLITS block 12's own bytes from its own code slot into two
different attention chunks. Not a correctness bug (the chunked mechanism
still lets a chunk attend to itself + the previous chunk, so the split
slot could still reach back to its own block's bytes) — but the window
boundary was accidental, not designed, and doesn't correspond to a clean
number of blocks the way a normal sequence LM's window corresponds to a
clean number of tokens. Fix: `attn_window` must now be a multiple of
`K+1` (enforced at Config construction, not just a documented
convention) — so every attention chunk covers exactly `attn_window/(K+1)`
whole blocks, code slot included, cleanly. No architecture change needed
— build_interleaved already reserves the LAST position of each K+1 group
for the code slot (session discussion: "reserve last to code embed"),
this fork just makes the windowing respect that structure instead of
ignoring it.

No shared imports with qcutelm_vlt/vlt2/.../vlt6 (self-contained-module
convention) — Logger/Checkpointer/schedule helpers/quantizers duplicated.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_<name>.py
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
    K: int = 4
    context_len: int = 1024
    dq: int = 6
    quant_type: str = "ifsq"   # "bsq" -> factorized sigmoids; "fsq"/"ifsq" -> factorized softmax
    fsq_levels: int = 8
    vocab: int = 256
    d_model: int = 96          # tokenizer tier (narrow, touches the full byte-length sequence)
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    attn_window: int = -1      # -1 = dense O(T^2) (original). N>0 = O(T*N) chunked (ported from qcutelm_vlt6),
                                # applied to the TOKENIZER tier's attention only — codelm's own sequence is
                                # already short (n_blocks), windowing it isn't the point.
    lm_d_model: int = 256      # codelm tier (wide, touches ONLY the short n_blocks-length code sequence)
    lm_n_heads: int = 4
    lm_n_layers: int = 3
    lm_mlp_mult: int = 4
    code_match_weight: float = 1.0
    rope_base: float = 10000.0
    trainable_slot_embed: bool = False  # False (original): literal zero vector marks the "no-code" reserved
                                         # slot in Pass 1. True: a single learned [B,n_blocks,d_model]-broadcast
                                         # parameter instead — ablation testing whether a genuine learned
                                         # "blank" marker (vs. a hardcoded constant) gives the model a better
                                         # signal for "this position has no code yet" / better code readout.
    shared_tokenizer_phases: bool = False  # False (default as of the session's symmetry-flaw finding):
                                           # Pass 1 ("no-code"/encode) and Pass 2 ("forecast"/decode) get
                                           # independent weights (same architecture, untied params). Encode is a
                                           # pure PRODUCER of codes (never consumes codelm's output); decode is a
                                           # pure CONSUMER (always conditioned on codelm's forecast) — these are
                                           # different functions (unconditional vs. conditional LM), not
                                           # symmetric roles, so sharing weights between them was asking one set
                                           # of weights to serve two incompatible jobs. True symmetry (encode
                                           # ALSO consuming codelm's output, block-by-block) is qcutelm_vlt9,
                                           # not this flag — see that module's docstring. True=original v7/v8
                                           # behavior, kept for direct comparison via config.
    aux_recon_weight: float = 0.0   # short, direct gradient path for code_pre: decode(z_hat_enc) vs the SAME
                                     # block's own bytes, block-local (K-length, batched, zero cross-block
                                     # attention — genuinely isolated, unlike Pass 1/Pass 2's windowed stack).
                                     # Mirrors qcutelm_vlt6's aux_recon_weight; motivated by the earlier-
                                     # diagnosed BSQ code_proj gradient weakness (10-28x, from F.normalize's
                                     # Jacobian) — without this, code_pre only ever gets gradient through the
                                     # long chain z_hat->codelm->pred_soft->decode->backprop.
    encode_match_weight: float = 0.0  # mutual-consistency: code_pre's raw (pre-quantization) output pulled
                                       # toward codelm's own prediction (detached) — the REVERSE direction of
                                       # code_match_loss. Motivated by the tokenizer/detokenizer-free codelm
                                       # decoding goal: free-rolling generation feeds codelm's own prediction
                                       # back as its next input, which only stays stable if codelm's
                                       # predictions and the encoder's true codes are already close — this
                                       # loss trains that closeness directly, instead of relying on
                                       # code_match_loss's one-directional (encoder is fixed, codelm chases it)
                                       # pull alone.


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def fsq_quantize(v: torch.Tensor, levels: int, bound: str = "tanh") -> torch.Tensor:
    half_l = (levels - 1) / 2
    z = torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1)
    bounded = z * half_l
    z_hat = bounded + (torch.round(bounded) - bounded).detach()
    return z_hat / half_l


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
    """Full-causal SDPA + RoPE. window=None: plain O(T^2) dense (original
    v1/v2 behavior). window=W: O(T*W) chunked (ported from qcutelm_vlt6's
    verified _forward_chunked_no_sink — same math, ~1e-7 max diff against
    the dense reference there) when T is a multiple of W and T > W;
    silently falls back to dense otherwise (needed since generate() runs
    on growing, not-always-window-divisible sequence lengths)."""

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
        k_local = torch.cat([kc_prev, kc], dim=3)  # [B, H, n_chunks, 2W, hd]
        v_local = torch.cat([vc_prev, vc], dim=3)

        i = torch.arange(W, device=q.device).view(W, 1)
        j_prev = torch.arange(W, device=q.device).view(1, W) - W
        j_cur = torch.arange(W, device=q.device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)  # [W, 2W]
        mask_per_chunk = causal_window.unsqueeze(0).expand(n_chunks, W, 2 * W).clone()
        mask_per_chunk[0, :, 0:W] = False  # chunk 0 has no real previous chunk

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


class CodeLM(nn.Module):
    """The ONLY wide component — separate weights from the tokenizer tier,
    operates strictly on the short [B, n_blocks, dq] code sequence, never
    on raw bytes. Forecasts code[i+1] from codes[:i] (teacher-forced,
    causal). No loss of its own; code_match_loss (computed in the parent
    model, against a detached target) is what trains it — identical
    mechanism to qcutelm_vlt6's CodeLM, duplicated per this repo's
    self-contained-module convention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.dq, cfg.lm_d_model)
        self.blocks = nn.ModuleList([Block(cfg.lm_d_model, cfg.lm_n_heads, cfg.lm_mlp_mult) for _ in range(cfg.lm_n_layers)])
        self.ln_f = nn.LayerNorm(cfg.lm_d_model)
        factorized_softmax = cfg.quant_type in ("fsq", "ifsq")
        self.pred_head = nn.Linear(cfg.lm_d_model, cfg.dq * cfg.fsq_levels if factorized_softmax else cfg.dq)
        if factorized_softmax:
            levels = cfg.fsq_levels
            half_l = (levels - 1) / 2
            level_values = (torch.arange(levels) - half_l) / half_l
            self.register_buffer("level_values", level_values)

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z_hat: [B, n_blocks, dq] (true codes, teacher-forced) ->
        (pred_soft, raw_logits). pred_soft: [B, n_blocks, dq], a soft
        prediction of the NEXT position's code (pred_soft[:,i] predicts
        z_hat[:,i+1]). raw_logits: pre-squash (per-bit or per-level
        logits), used for code_match_loss."""
        cfg = self.cfg
        x = self.in_proj(z_hat)
        head_dim = cfg.lm_d_model // cfg.lm_n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)
        raw = self.pred_head(h)
        if cfg.quant_type in ("fsq", "ifsq"):
            B, T, _ = raw.shape
            logits = raw.view(B, T, cfg.dq, cfg.fsq_levels)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.level_values).sum(-1), logits
        return 2 * torch.sigmoid(raw) - 1, raw


class InterleavedSymmetricLM(nn.Module):
    """Tokenizer tier: ONE narrow shared stack plays both "no-code" (Pass
    1) and "forecast" (Pass 2) modes — same weights, same reserved-slot
    format, RoPE-consistent. CodeLM tier: separate wide weights, short
    sequence only. See module docstring for the full rationale."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        window = None if cfg.attn_window == -1 else cfg.attn_window
        if window is not None:
            assert window % (cfg.K + 1) == 0, (
                f"attn_window ({window}) must be a multiple of K+1 ({cfg.K + 1}) so attention chunks align "
                f"to whole blocks (code slot included) — see module docstring's v8 rationale."
            )
        # Pass 1 ("no-code"/encoder role) — always its own weights.
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window=window) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)
        self.code_pre = nn.Linear(cfg.d_model, cfg.dq)   # hidden state -> pre-quantization vector (Pass 1 only)

        # Pass 2 ("forecast"/decode role) — aliases Pass 1's weights when shared_tokenizer_phases=True
        # (original behavior), independent copies otherwise (same architecture, untied params).
        if cfg.shared_tokenizer_phases:
            self.dec_byte_emb = self.byte_emb
            self.dec_blocks = self.blocks
            self.dec_ln_f = self.ln_f
            self.dec_head = self.head
        else:
            self.dec_byte_emb = nn.Embedding(cfg.vocab, cfg.d_model)
            self.dec_blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, window=window) for _ in range(cfg.n_layers)])
            self.dec_ln_f = nn.LayerNorm(cfg.d_model)
            self.dec_head = nn.Linear(cfg.d_model, cfg.vocab)

        self.z_proj = nn.Linear(cfg.dq, cfg.d_model)     # code (true or predicted) -> embedding, for injection
        self.codelm = CodeLM(cfg)
        if cfg.trainable_slot_embed:
            self.nocode_embed = nn.Parameter(torch.zeros(cfg.d_model))

    def nocode_slot(self, B: int, n_blocks: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The "no-code" reserved-slot marker for Pass 1 — either a literal
        zero vector (original) or a single learned parameter, broadcast to
        every slot (cfg.trainable_slot_embed=True ablation)."""
        if self.cfg.trainable_slot_embed:
            return self.nocode_embed.to(dtype).expand(B, n_blocks, -1)
        return torch.zeros(B, n_blocks, self.cfg.d_model, device=device, dtype=dtype)

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Pass 1 ("no-code"/encoder role) — always self.blocks/self.ln_f."""
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def run_dec_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Pass 2 ("forecast"/decode role) — self.dec_blocks/self.dec_ln_f,
        aliases Pass 1's own when shared_tokenizer_phases=True."""
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_cos_sin(x.size(1), head_dim, self.cfg.rope_base, x.device)
        for block in self.dec_blocks:
            x = block(x, cos, sin)
        return self.dec_ln_f(x)

    def decode_block_local(self, code: torch.Tensor, target_block: torch.Tensor) -> torch.Tensor:
        """code: [N, dq] (a true or predicted code), target_block: [N, K]
        teacher-forcing targets -> logits: [N, K, vocab]. Block-local,
        BATCHED (N = B*n_blocks), zero cross-block attention — genuinely
        isolated, unlike Pass 1/Pass 2's windowed multi-block stack.
        Ported from qcutelm_vlt6's decode_block. Reuses dec_byte_emb/
        run_dec_blocks/dec_head/z_proj — the chunked-window check in
        CausalSelfAttention automatically falls back to dense SDPA here
        since this sequence (length K) is far shorter than attn_window."""
        K = target_block.size(1)
        bos = self.z_proj(code).unsqueeze(1)
        if K > 1:
            dec_in = torch.cat([bos, self.dec_byte_emb(target_block[:, :-1])], dim=1)
        else:
            dec_in = bos
        dec_h = self.run_dec_blocks(dec_in)
        return self.dec_head(dec_h)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.cfg.quant_type == "bsq":
            return bsq_quantize(v, self.cfg.dq)
        elif self.cfg.quant_type == "fsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="tanh")
        elif self.cfg.quant_type == "ifsq":
            return fsq_quantize(v, self.cfg.fsq_levels, bound="sigmoid")
        raise ValueError(f"unknown quant_type {self.cfg.quant_type!r}")

    def build_interleaved(self, byte_blocks: torch.Tensor, slot_embed: torch.Tensor) -> torch.Tensor:
        """byte_blocks: [B, n_blocks, K, d_model], slot_embed: [B, n_blocks, d_model]
        (zero / true code / predicted code — same shape either way) -> [B,
        n_blocks*(K+1), d_model]: b0..b(K-1),slot, b0..b(K-1),slot, ..."""
        B, n_blocks, K, D = byte_blocks.shape
        return torch.cat([byte_blocks, slot_embed.unsqueeze(2)], dim=2).view(B, n_blocks * (K + 1), D)

    def _targets(self, ctx: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """[B, n_blocks*(K+1)] — within-block next-byte at local 0..K-2;
        slot (local K) predicts the FOLLOWING block's first byte; local
        K-1 (a block's last byte) has no target — ignore_index=-100 skips
        it. Identical for both passes: same length, same slot positions."""
        cfg = self.cfg
        K = cfg.K
        B = ctx.size(0)
        ctx_blocks = ctx.view(B, n_blocks, K)
        target = torch.full((B, n_blocks, K + 1), -100, dtype=torch.long, device=ctx.device)
        if K > 1:
            target[:, :, 0:K - 1] = ctx_blocks[:, :, 1:K]
        if n_blocks > 1:
            target[:, :n_blocks - 1, K] = ctx_blocks[:, 1:, 0]
        return target.view(B, n_blocks * (K + 1))

    def _loss_and_metrics(self, logits: torch.Tensor, target_flat: torch.Tensor, n_blocks: int, prefix: str) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        K = cfg.K
        B = logits.size(0)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), target_flat.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            valid = target_flat != -100
            pred = logits.argmax(-1)
            acc = (pred == target_flat)[valid].float().mean() if valid.any() else torch.zeros(())
            slot_mask = torch.zeros_like(target_flat, dtype=torch.bool).view(B, n_blocks, K + 1)
            slot_mask[:, : n_blocks - 1, K] = True
            slot_mask = slot_mask.view(B, n_blocks * (K + 1))
            slot_valid = slot_mask & valid
            slot_acc = (pred == target_flat)[slot_valid].float().mean() if slot_valid.any() else torch.zeros(())
            within_valid = valid & ~slot_mask
            within_acc = (pred == target_flat)[within_valid].float().mean() if within_valid.any() else torch.zeros(())
        return loss, {f"{prefix}_loss": loss, f"{prefix}_acc": acc, f"{prefix}_slot_acc": slot_acc, f"{prefix}_within_acc": within_acc}

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        B, L = ctx.shape
        K = cfg.K
        n_blocks = L // K
        D = cfg.d_model

        target_flat = self._targets(ctx, n_blocks)

        # Pass 1 — tokenizer tier, "no-code" mode: reserved-slot marker (zero or trainable).
        byte_embed = self.byte_emb(ctx)
        byte_blocks = byte_embed.view(B, n_blocks, K, D)
        zero_slot = self.nocode_slot(B, n_blocks, ctx.device, byte_embed.dtype)
        interleaved1 = self.build_interleaved(byte_blocks, zero_slot)
        h1 = self.run_blocks(interleaved1)
        logits1 = self.head(h1)
        loss_nocode, metrics1 = self._loss_and_metrics(logits1, target_flat, n_blocks, "nocode")

        # TRUE codes: deterministic readout of Pass 1's own hidden state at each zero slot.
        h1_view = h1.view(B, n_blocks, K + 1, D)
        slot_hidden = h1_view[:, :, K, :]
        pre_q = self.code_pre(slot_hidden)                   # [B, n_blocks, dq] — pre-quantization
        z_hat = self.quantize(pre_q)                          # [B, n_blocks, dq]

        # CodeLM tier — wide, separate weights, SHORT sequence only: forecast the next code.
        pred_soft_full, raw_logits_full = self.codelm(z_hat)
        pred_soft = pred_soft_full[:, :-1, :]                  # predicts z_hat[1:]
        raw_logits = raw_logits_full[:, :-1]

        true_next_code = z_hat[:, 1:, :].detach()               # stop-gradient target for code_match_loss only
        if cfg.quant_type in ("fsq", "ifsq"):
            half_l = (cfg.fsq_levels - 1) / 2
            true_level = torch.round(true_next_code * half_l + half_l).long().clamp(0, cfg.fsq_levels - 1)
            code_match_loss = F.cross_entropy(raw_logits.reshape(-1, cfg.fsq_levels), true_level.reshape(-1))
        else:
            true_bits = (true_next_code > 0).float()
            code_match_loss = F.binary_cross_entropy_with_logits(raw_logits, true_bits)

        # aux_recon: short, direct gradient path for code_pre — block-local reconstruction using the
        # SAME block's own true code (z_hat, NOT detached — this is what gives code_pre a direct signal
        # instead of only ever hearing from code_match_loss's long chain through codelm+Pass2).
        aux_recon_loss = torch.zeros((), device=ctx.device)
        aux_recon_acc = torch.zeros((), device=ctx.device)
        if cfg.aux_recon_weight > 0:
            all_blocks = ctx.view(B, n_blocks, K)
            z_flat = z_hat.reshape(B * n_blocks, cfg.dq)
            blocks_flat = all_blocks.reshape(B * n_blocks, K)
            aux_logits = self.decode_block_local(z_flat, blocks_flat)
            aux_recon_loss = F.cross_entropy(aux_logits.reshape(-1, cfg.vocab), blocks_flat.reshape(-1))
            aux_recon_acc = (aux_logits.argmax(-1) == blocks_flat).float().mean()

        # encode_match: mutual-consistency — pulls code_pre's raw output toward codelm's OWN prediction
        # (detached this time, reversing code_match_loss's direction). Directly trains the codelm-
        # prediction <-> true-code gap that tokenizer/detokenizer-free free-rolling depends on being small.
        encode_match_loss = torch.zeros((), device=ctx.device)
        if cfg.encode_match_weight > 0:
            encode_match_loss = F.mse_loss(pre_q[:, 1:, :], pred_soft.detach())

        # Pass 2 — tokenizer tier, "forecast" mode: slot holds CODELM's PREDICTED next code (not the
        # block's own true code) — the genuine generative test. Gradient flows back into codelm through
        # pred_soft (not detached here). Uses dec_byte_emb/run_dec_blocks/dec_head — aliases Pass 1's own
        # when shared_tokenizer_phases=True, independent weights otherwise.
        dec_byte_embed = self.dec_byte_emb(ctx)
        dec_byte_blocks = dec_byte_embed.view(B, n_blocks, K, D)
        pred_code_embed = self.z_proj(pred_soft)                # [B, n_blocks-1, D]
        pad = torch.zeros(B, 1, D, device=ctx.device, dtype=pred_code_embed.dtype)
        slot_embed_decode = torch.cat([pred_code_embed, pad], dim=1)   # last slot unused, masked out by target/-100

        interleaved2 = self.build_interleaved(dec_byte_blocks, slot_embed_decode)
        h2 = self.run_dec_blocks(interleaved2)
        logits2 = self.dec_head(h2)
        loss_decode, metrics2 = self._loss_and_metrics(logits2, target_flat, n_blocks, "code")

        loss = (loss_nocode + loss_decode + cfg.code_match_weight * code_match_loss
                + cfg.aux_recon_weight * aux_recon_loss + cfg.encode_match_weight * encode_match_loss)
        metrics = {
            "loss": loss,
            "code_match_loss": code_match_loss,
            "no_code_acc": metrics1["nocode_slot_acc"],           # Pass 1: predict next block's 1st byte, NO code — baseline
            "code_conditioned_acc": metrics2["code_slot_acc"],     # Pass 2: predict next block's 1st byte, WITH codelm's forecast
            "within_block_acc": metrics2["code_within_acc"],       # trivial short-range positions, expected to be easy/high
            **metrics1, **metrics2,
        }
        if cfg.aux_recon_weight > 0:
            metrics["aux_recon_loss"] = aux_recon_loss
            metrics["aux_recon_acc"] = aux_recon_acc
        if cfg.encode_match_weight > 0:
            metrics["encode_match_loss"] = encode_match_loss
        return loss, metrics


def init_head_bias_to_unigram(model: InterleavedSymmetricLM, data: torch.Tensor) -> None:
    counts = torch.bincount(data, minlength=256).float() + 1.0
    log_freq = torch.log(counts / counts.sum())
    with torch.no_grad():
        model.head.bias.copy_(log_freq.to(model.head.bias.device))
        if model.dec_head is not model.head:   # untied (shared_tokenizer_phases=False): dec_head is a
            model.dec_head.bias.copy_(log_freq.to(model.dec_head.bias.device))   # separate nn.Linear, else
                                                                                    # it stays randomly initialized


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
def generate(model: InterleavedSymmetricLM, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """prompt_bytes: [L0] (L0 a positive multiple of K) -> [L0 + n_new_bytes]
    generated continuation. Mirrors training exactly: the prompt's own
    blocks get their TRUE codes (zero-slot readout) injected as history for
    codelm; every NEW block is generated using codelm's FORECAST (not a
    true code, since the block doesn't exist yet) injected at the slot —
    the same genuine held-out prediction qcutelm_vlt6's generate() does via
    decode(codepred(codelm(...))). No KV cache (recomputes the full
    tokenizer-tier pass every step; codelm's short-sequence pass is cheap
    enough to also just recompute each block boundary)."""
    cfg = model.cfg
    K = cfg.K
    D = cfg.d_model
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    assert prompt_bytes.size(1) % K == 0 and prompt_bytes.size(1) >= K, "prompt length must be a positive multiple of K"
    B = prompt_bytes.size(0)
    n_prompt_blocks = prompt_bytes.size(1) // K

    def encode_true_codes(byte_ids: torch.Tensor) -> torch.Tensor:
        """byte_ids: [B, n*K] -> [B, n, dq] TRUE codes — Pass 1 role only
        (self.byte_emb/run_blocks/code_pre), recomputed fresh from scratch
        every call (no KV cache, consistent with this file's existing
        simplicity tradeoff elsewhere)."""
        n = byte_ids.size(1) // K
        be = model.byte_emb(byte_ids).view(B, n, K, D)
        zs = model.nocode_slot(B, n, device, be.dtype)
        hh = model.run_blocks(model.build_interleaved(be, zs))
        sh = hh.view(B, n, K + 1, D)[:, :, K, :]
        return model.quantize(model.code_pre(sh))

    all_bytes = prompt_bytes
    z_hist = encode_true_codes(all_bytes)                                     # prompt's own true codes

    dec_byte_embed = model.dec_byte_emb(prompt_bytes).view(B, n_prompt_blocks, K, D)
    embed_seq = model.build_interleaved(dec_byte_embed, model.z_proj(z_hist))  # prompt, decode-role embeddings
    pred_soft_full, _ = model.codelm(z_hist)
    slot_embed = model.z_proj(pred_soft_full[:, -1, :]).unsqueeze(1)          # forecast for the FIRST new block
    embed_seq = torch.cat([embed_seq, slot_embed], dim=1)

    out_bytes = [prompt_bytes]
    cur_block_bytes = []
    n_generated = 0
    while n_generated < n_new_bytes:
        h = model.run_dec_blocks(embed_seq)
        logits = model.dec_head(h[:, -1, :])
        next_byte = logits.argmax(-1)
        out_bytes.append(next_byte.unsqueeze(1))
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
        embed_seq = torch.cat([embed_seq, model.dec_byte_emb(next_byte).unsqueeze(1)], dim=1)
        cur_block_bytes.append(next_byte)
        n_generated += 1
        if len(cur_block_bytes) == K:
            z_hist = encode_true_codes(all_bytes)                            # TRUE codes, all blocks so far (Pass 1 role)
            pred_soft_full, _ = model.codelm(z_hist)
            slot_embed_new = model.z_proj(pred_soft_full[:, -1, :]).unsqueeze(1)   # forecast for the NEXT block
            embed_seq = torch.cat([embed_seq, slot_embed_new], dim=1)
            cur_block_bytes = []
    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def _bytes_repr(t: torch.Tensor) -> str:
    return repr(bytes([int(x) & 0xff for x in t.tolist()]).decode("latin1"))


def qualitative_gen(model: InterleavedSymmetricLM, data: torch.Tensor, prompt_len: int, n_new_bytes: int, device: str, log, step: int) -> None:
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
def eval_model(model: InterleavedSymmetricLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        accum.setdefault("total_loss", []).append(loss.item())
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["code_loss"] / math.log(2)             # forecast-conditioned pass — comparable to bytelm/bpelm's bpb
    result["no_code_bpb"] = result["nocode_loss"] / math.log(2)     # same-weights baseline, no code at all
    return result


def train(model: InterleavedSymmetricLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    init_head_bias_to_unigram(model, train_data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vlt8", dynamic_ncols=True)
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

        postfix = dict(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}",
            bpb=f"{metrics['code_loss'].item()/math.log(2):.4f}",
            no_code_bpb=f"{metrics['nocode_loss'].item()/math.log(2):.4f}",
            code_acc=f"{metrics['code_conditioned_acc'].item()*100:.2f}%",
            no_code_acc=f"{metrics['no_code_acc'].item()*100:.2f}%",
            code_match_loss=f"{metrics['code_match_loss'].item():.4f}",
        )
        if "aux_recon_loss" in metrics:
            postfix["aux_recon_acc"] = f"{metrics['aux_recon_acc'].item()*100:.2f}%"
        if "encode_match_loss" in metrics:
            postfix["encode_match_loss"] = f"{metrics['encode_match_loss'].item():.4f}"
        pbar.set_postfix(**postfix)

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])

        if args.gen_every > 0 and (step % args.gen_every == 0 or step == args.steps):
            qualitative_gen(model, val_data, args.gen_prompt_len, args.gen_new_bytes, device, log, step)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Symmetric narrow tokenizer + separate wide codelm, block-aligned windowed attention (fork of qcute.qcutelm_vlt7)", parents=[pre])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--dq", type=int, default=6)
    p.add_argument("--quant_type", type=str, default="ifsq", choices=["bsq", "fsq", "ifsq"])
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=-1)
    p.add_argument("--lm_d_model", type=int, default=256)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_n_layers", type=int, default=3)
    p.add_argument("--lm_mlp_mult", type=int, default=4)
    p.add_argument("--code_match_weight", type=float, default=1.0)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--trainable_slot_embed", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--shared_tokenizer_phases", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--aux_recon_weight", type=float, default=0.0)
    p.add_argument("--encode_match_weight", type=float, default=0.0)

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
    cfg = Config(
        K=args.K, context_len=args.context_len, dq=args.dq, quant_type=args.quant_type,
        fsq_levels=args.fsq_levels, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, mlp_mult=args.mlp_mult, attn_window=args.attn_window, lm_d_model=args.lm_d_model,
        lm_n_heads=args.lm_n_heads, lm_n_layers=args.lm_n_layers, lm_mlp_mult=args.lm_mlp_mult,
        code_match_weight=args.code_match_weight, rope_base=args.rope_base,
        trainable_slot_embed=args.trainable_slot_embed, shared_tokenizer_phases=args.shared_tokenizer_phases,
        aux_recon_weight=args.aux_recon_weight, encode_match_weight=args.encode_match_weight,
    )
    model = InterleavedSymmetricLM(cfg).to(device)
    n_tokenizer = sum(p_.numel() for n_, p_ in model.named_parameters() if not n_.startswith("codelm"))
    n_codelm = sum(p_.numel() for n_, p_ in model.named_parameters() if n_.startswith("codelm"))
    n_params = n_tokenizer + n_codelm

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcutelm_vlt8_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"K={cfg.K} context_len={cfg.context_len} dq={cfg.dq} quant_type={cfg.quant_type} "
        f"params={n_params/1e6:.3f}M (tokenizer={n_tokenizer/1e6:.3f}M codelm={n_codelm/1e6:.3f}M) device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
