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
    # BSQ only: default training path is tightly coupled — encoder -> z_t ->
    # LM -> predicted latent -> BSQ-quantized -> decoder -> bytes_{t+1}, i.e.
    # the decoder's primary reconstruction target is the LM's prediction, not
    # the encoder's own code. aux_recon additionally decodes the encoder's
    # z_t directly back to bytes_t (bypassing the LM) as an auxiliary loss —
    # set False to disable it and train on the tightly-coupled path alone.
    aux_recon: bool = True
    # BSQ only: regress the quantizer from BSQ to plain LFQ (Lookup-Free
    # Quantization, Yu et al. 2023) by dropping BSQ's L2-normalize-onto-the-
    # hypersphere step — LFQ just signs the raw projection (hypercube corners
    # {-1,+1}^dq, unconstrained scale) instead of signing a unit vector and
    # rescaling by 1/sqrt(dq) (hypersphere corners, ||z_hat||=1). Targets
    # (sign bits) are identical either way; only z_hat's geometry changes.
    lfq: bool = False
    maskgit_T: int = 4  # decoder inference refinement steps, see ChunkDecoder/maskgit_decode
    # Encoder/decoder cross-K-position mixer (see MixerBlock): "attention"
    # (full non-causal self-attention over the K positions) or "conv" (single
    # non-causal 1D conv, kernel_size=2K-1 with symmetric same-padding, so
    # every position's window covers the whole chunk — see FullConvMixer for
    # why kernel_size=K, tried first, was wrong). mixer_mlp adds a post-mixer
    # MLP block (standard transformer-style); set False to test the mixer alone.
    mixer: str = "attention"
    mixer_mlp: bool = True
    # LM input: compositional per-dim embedding (see FactorizedCodeEmbedding)
    # instead of a linear projection of the continuous z. Takes the discrete
    # `targets` as input instead of `z_hat`; output format is unchanged
    # (still the same per-dim logits used throughout this file).
    lm_factorized_input: bool = False

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


def bsq_quantize(v: torch.Tensor, dq: int, lfq: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """BSQ quantization math (normalize + sign STE), factored out of the
    encoder's BSQ module so the LM's own output head can reuse it directly
    on raw dq-dim vectors — same quantization boundary, two different
    producers (encoder's projected features, or the LM's predicted latent).

    lfq=True regresses this to plain LFQ: skip the L2-normalize-to-unit-
    sphere step, sign the raw projection directly (hypercube corners
    {-1,+1}^dq instead of BSQ's hypersphere corners ||z_hat||=1). Sign bits
    (targets) are unaffected by normalization, so they're identical either
    way — only z_hat's geometry/scale changes.

    `targets` is used elsewhere as a BCE/CE loss *label* (e.g. pred_loss),
    so it must never carry gradient back into whatever produced `v`. The
    `>` comparison already breaks the autograd graph on its own (PyTorch
    doesn't attach a grad_fn to comparison ops), so this .detach() is
    currently a no-op — kept explicit anyway so that stays true even if
    this computation is ever changed to something differentiable (e.g. a
    soft/temperature-based comparison), which would otherwise silently
    leak a second gradient path into the encoder through the loss target."""
    if lfq:
        z_hat = v + (torch.sign(v) - v).detach()  # STE, no normalize/rescale
        targets = (v > 0).float().detach()
        return z_hat, targets
    v_unit = F.normalize(v, dim=-1)
    z_hat = (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)  # STE
    targets = (v_unit > 0).float().detach()
    return z_hat, targets


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def bsq_entropy_reg(v: torch.Tensor) -> torch.Tensor:
    """LFQ/BSQ-style entropy regularization (Yu et al. 2023 §3.2, MAGVIT-v2;
    also BSQ 2024's closed-form version), missing from this project's BSQ
    module until now — see docs/status.md's literature-contrast entry for
    why this specifically was flagged as a real gap: both papers' actual
    published recipes lean on this term to keep code usage spread out,
    and its absence here plausibly explains the qualitative-generation
    collapse into a fixed repeating byte pattern (docs/status.md).

    Two opposing pressures on the per-bit Bernoulli probabilities
    p=sigmoid(v): (1) minimize each *example's* bit entropy — push
    predictions toward confident/decisive corners, matching the hard
    quantization boundary; (2) maximize the *batch-averaged* bit-usage
    entropy — spread which corners get used across examples, directly
    countering collapse onto one dominant code. Returns
    E_batch[H(p)] - H(E_batch[p]), the standard combined term (minimize
    this: pulls per-example entropy down, batch-average entropy up)."""
    probs = torch.sigmoid(v)  # [..., dq]
    per_example = bernoulli_entropy(probs).sum(-1).mean()
    batch_avg = probs.reshape(-1, probs.size(-1)).mean(0)
    batch = bernoulli_entropy(batch_avg).sum()
    return per_example - batch


def bsq_sample(out: torch.Tensor, dq: int, lfq: bool, temperature: float) -> torch.Tensor:
    """Stochastic alternative to bsq_quantize's deterministic hard sign, used
    only at generate() time (training always uses the hard STE sign — this
    never touches training). `out` is the LM's raw, pre-quantization output;
    treat it as per-bit logits, exactly what pred_loss's BCE already trains
    it to be (bsq_quantize's targets are `v_unit > 0`, and normalizing by a
    positive L2 norm never flips sign, so `sign(v) == sign(v_unit)` always —
    the raw output's sign already carries the same information the trained
    loss cares about). Independently Bernoulli-samples each bit from
    sigmoid(out/temperature) instead of always taking the argmax (sign) bit.
    temperature=1.0 samples from the trained distribution as-is; lower
    sharpens toward the deterministic mode, higher flattens toward a coin
    flip per bit. Motivation: qualitative generation with the deterministic
    path was observed to collapse into a fixed repeating cycle regardless of
    the prompt (see docs/status.md) — sampling is one way to break that."""
    probs = torch.sigmoid(out / temperature)
    bits = torch.bernoulli(probs)
    scale = 1.0 if lfq else 1.0 / math.sqrt(dq)
    return (2 * bits - 1) * scale


class BSQ(nn.Module):
    """Binary spherical quantization. Targets are sign bits in {0, 1}."""

    def __init__(self, d_in: int, dq: int, lfq: bool = False):
        super().__init__()
        self.dq = dq
        self.lfq = lfq
        self.proj = nn.Linear(d_in, dq)

    def forward(self, u: torch.Tensor):
        return bsq_quantize(self.proj(u), self.dq, self.lfq)

    @staticmethod
    def bits_to_z(bits: torch.Tensor, dq: int) -> torch.Tensor:
        return (2 * bits - 1) / math.sqrt(dq)


def make_bottleneck(cfg: Config) -> nn.Module:
    if cfg.bottleneck == "fsq":
        return FSQ(cfg.d_enc, cfg.dq, cfg.L)
    if cfg.bottleneck == "bsq":
        return BSQ(cfg.d_enc, cfg.dq, cfg.lfq)
    raise ValueError(f"unknown bottleneck: {cfg.bottleneck}")


# ---------------------------------------------------------------------------
# Encoder / decoder — plain MLPs over one chunk, non-streaming (see module
# docstring). Deliberately non-causal within the chunk: the LM (autoregressive
# across chunks) is what owns causality here, not the encoder/decoder — a
# causal-TCN encoder was tried and reverted (see docs/status.md) since the
# chunk is always fully observed before encoding, so hiding future bytes from
# earlier positions bought nothing and actively fought a later decay-weighted
# pooling step that wanted the *opposite* (early positions to be information-
# rich, not causally starved).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-K-position mixers, shared by ChunkEncoder and ChunkDecoder
# ---------------------------------------------------------------------------
#
# An earlier version of ChunkDecoder computed each of the K positions from
# z/its own byte-or-MASK embedding *alone*, via a plain per-position
# Linear/GELU/Linear MLP — since nn.Linear broadcasts over all but the last
# dimension, that was *zero* cross-position mixing, despite the docstring
# claiming otherwise (a real bug, not a missed optimization — see
# docs/status.md). Every mixer below actually mixes across the K positions;
# there is no "point-wise MLP, no mixing" option left anywhere in this file.


class FullSelfAttention(nn.Module):
    """Full (non-causal) self-attention over a short fixed-length sequence —
    the K chunk positions are always fully present (some real, some MASK)
    at once; there's no autoregressive order to respect within a chunk,
    unlike LatentLM's causal attention over time."""

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
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FullConvMixer(nn.Module):
    """Cheaper alternative to FullSelfAttention: a single non-causal 1D
    conv, kernel_size=2K-1 with *symmetric* padding=K-1 on each side (K is
    tiny, e.g. 4, so this is still cheap). This is the kernel size that
    actually matters: with kernel_size=K (tried first, wrong) and an
    uneven pad split, only one output position ends up seeing all K real
    inputs — the rest get an inconsistent, position-dependent *partial*
    window (e.g. at K=4, the last position would only see the last two
    real inputs, missing the first two entirely). kernel_size=2K-1 with
    symmetric same-padding guarantees every output position's window spans
    the *entire* real input range regardless of its position — matching
    FullSelfAttention's actual full-coverage property via a wide conv
    instead of a softmax, not an accidentally-partial one. Not causal on
    purpose — see docs/status.md's "encoder reverted to plain MLP" entry
    for why causal masking within a fully-observed chunk is the wrong idea
    (that reasoning applies here too, not just to the earlier attempt)."""

    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2 * K - 1, padding=K - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, K, D] -> [N, K, D]
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


def make_mixer(cfg: Config, d_model: int) -> nn.Module:
    if cfg.mixer == "attention":
        return FullSelfAttention(d_model, cfg.n_heads)
    if cfg.mixer == "conv":
        return FullConvMixer(d_model, cfg.K)
    raise ValueError(f"unknown mixer: {cfg.mixer}")


class MixerBlock(nn.Module):
    """Pre-norm residual block: mixer (attention or conv, see make_mixer)
    over the K positions, optionally followed by a per-position MLP block
    (standard transformer-style) — cfg.mixer_mlp=False tests the mixer
    alone. Shared by ChunkEncoder and ChunkDecoder so both use the exact
    same cross-position-mixing machinery."""

    def __init__(self, cfg: Config, d_model: int, d_hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = make_mixer(cfg, d_model)
        self.use_mlp = cfg.mixer_mlp
        if self.use_mlp:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln1(x))
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        return x


class ChunkEncoder(nn.Module):
    """byte_emb + pos_emb per position -> MixerBlock (cross-K mixing) ->
    flatten -> Linear -> bottleneck. Same block structure as ChunkDecoder
    below, mirrored (encoder: bytes -> z; decoder: z -> bytes)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab, cfg.d_byte)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.K, cfg.d_byte))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.block = MixerBlock(cfg, cfg.d_byte, cfg.d_enc)
        self.ln_f = nn.LayerNorm(cfg.K * cfg.d_byte)
        self.out_proj = nn.Linear(cfg.K * cfg.d_byte, cfg.d_enc)
        self.bottleneck = make_bottleneck(cfg)

    def forward(self, chunk: torch.Tensor):
        # chunk: [N, K] long -> z_hat: [N, dq], targets: [N, dq]
        N = chunk.size(0)
        h = self.byte_emb(chunk) + self.pos_emb.unsqueeze(0)  # [N, K, d_byte]
        h = self.block(h)
        u = self.out_proj(self.ln_f(h.reshape(N, -1)))
        return self.bottleneck(u)


class ChunkDecoder(nn.Module):
    """MaskGIT-style (handover §1.4b): given z (context) plus a byte chunk
    with some positions replaced by a MASK token, predict the *masked*
    positions' real bytes from the *unmasked* ones + z — the *point* of
    MaskGIT is exactly this cross-position conditioning, which requires the
    K positions to actually interact (see the mixer section's docstring
    above for why the old version didn't). Same MixerBlock as ChunkEncoder.

    MASK id = cfg.vocab (256); the byte embedding table is sized vocab+1
    to hold it. Output is always a plain [N, K, vocab] logits tensor over
    real bytes only — MASK is never a decode target, only an input state."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.vocab
        self.byte_emb = nn.Embedding(cfg.vocab + 1, cfg.d_byte)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.K, cfg.d_byte))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.z_proj = nn.Linear(cfg.dq, cfg.d_byte)
        self.block = MixerBlock(cfg, cfg.d_byte, cfg.d_dec)
        self.ln_f = nn.LayerNorm(cfg.d_byte)
        self.head = nn.Linear(cfg.d_byte, cfg.vocab)

    def forward(self, z: torch.Tensor, x_masked: torch.Tensor) -> torch.Tensor:
        # z: [N, dq], x_masked: [N, K] long (mask_id at masked positions) -> logits [N, K, vocab]
        h = self.byte_emb(x_masked) + self.pos_emb.unsqueeze(0) + self.z_proj(z).unsqueeze(1)
        h = self.block(h)
        return self.head(self.ln_f(h))


def maskgit_mask(x: torch.Tensor, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine mask-rate schedule (Chang et al. 2022, MaskGIT; handover §1.4b):
    sample r~U(0,1) per example, mask_rate=cos(pi/2*r), independently
    Bernoulli-mask each of the K positions at that rate. Forces at least one
    masked position per example — K is only 4 here, so an all-unmasked draw
    (no positions to supervise) is a real, non-negligible chance otherwise.
    x: [N, K] real byte ids -> (x_masked: [N, K], mask: [N, K] bool)."""
    N, K = x.shape
    r = torch.rand(N, device=x.device)
    mask_rate = torch.cos(math.pi / 2 * r).clamp(0, 1)  # [N]
    mask = torch.bernoulli(mask_rate.unsqueeze(1).expand(N, K)).bool()
    none_masked = ~mask.any(dim=1)
    if none_masked.any():
        force_idx = torch.randint(0, K, (int(none_masked.sum()),), device=x.device)
        mask[none_masked, force_idx] = True
    x_masked = torch.where(mask, torch.full_like(x, mask_id), x)
    return x_masked, mask


@torch.no_grad()
def maskgit_decode(decoder: ChunkDecoder, z: torch.Tensor, T: int) -> torch.Tensor:
    """T-step confidence-based MaskGIT inference (handover §1.4b): start
    fully masked, each step decode logits for the still-masked positions,
    commit the highest-confidence predictions per the cosine schedule's
    target masked-count for that step, remask the rest, repeat. Final step
    always reveals everything remaining (cos(pi/2*1)=0). z: [N, dq] ->
    bytes: [N, K] long."""
    cfg = decoder.cfg
    N, K = z.size(0), cfg.K
    x = torch.full((N, K), decoder.mask_id, dtype=torch.long, device=z.device)
    masked = torch.ones((N, K), dtype=torch.bool, device=z.device)
    for i in range(T):
        logits = decoder(z, x)  # [N, K, vocab]
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)  # [N, K]
        conf = conf.masked_fill(~masked, -1.0)  # already-revealed positions can't be re-picked

        target_masked = round(K * math.cos(math.pi / 2 * (i + 1) / T)) if i < T - 1 else 0
        n_masked_now = masked.sum(dim=1)  # [N]
        n_reveal = (n_masked_now - target_masked).clamp(min=0)
        for n in range(N):
            k = int(n_reveal[n].item())
            if k <= 0:
                continue
            topk = torch.topk(conf[n], k).indices
            x[n, topk] = pred[n, topk]
            masked[n, topk] = False
    return x


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
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.SiLU(), nn.Linear(4 * cfg.d_model, cfg.d_model)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class FactorizedCodeEmbedding(nn.Module):
    """Compositional (PQ-style) embedding for a dq-dim discrete code
    (targets: [..., dq] long, each entry in [0, levels_per_dim)) — one
    embedding vector per (dimension, level) pair, summed across dimensions.
    Generalizes to any code *combination* with zero extra training, unlike
    a per-whole-code vocab table (see docs/status.md's vocab-coverage
    discussion): every individual (dim, level) pair gets trained whenever
    it occurs in *any* code, so an unseen combination of already-seen
    per-dim values still gets a meaningful embedding, composed from pieces
    that were each trained. levels_per_dim=2 for BSQ/LFQ (bits), cfg.L for
    FSQ (levels)."""

    def __init__(self, dq: int, levels_per_dim: int, d_model: int):
        super().__init__()
        self.dq = dq
        self.table = nn.Parameter(torch.zeros(dq, levels_per_dim, d_model))
        nn.init.normal_(self.table, std=0.02)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        # targets: [..., dq], long (FSQ level indices) or float 0./1. (BSQ/LFQ
        # bits, from bsq_quantize) -> [..., d_model]
        dq_idx = torch.arange(self.dq, device=targets.device)
        selected = self.table[dq_idx, targets.long()]  # [..., dq, d_model]
        return selected.sum(dim=-2)


class LatentLM(nn.Module):
    """vocab_size=None, factorized_input=False (default): continuous-latent
    mode, as used throughout this file — input is a linear projection of
    the continuous z, output is FSQ's per-dim categorical logits or BSQ's
    raw dq-dim latent.

    vocab_size=<int>: discrete-vocabulary mode (see build_code_vocab/
    train_vocab_lm below) — input is a plain token embedding, output is a
    weight-tied categorical softmax head over the vocab, exactly like
    qcute.bpelm/bytelm. Used once a tokenizer (encoder+decoder) has been
    pretrained and frozen: with the code space fixed, treating each
    distinct code that actually occurs as one vocabulary entry (like a BPE
    token) is now cheap and well-defined, unlike trying to build an
    embedding table over BSQ's full implicit codebook (2^dq) up front.

    factorized_input=True (mutually exclusive with vocab_size): input is
    FactorizedCodeEmbedding on the discrete per-dim code (targets, long),
    output stays the same per-dim format as the continuous-latent default
    above — the "closer to decoder format" alternative to a vocab softmax,
    with no OOV/UNK problem since it's compositional, not a lookup table."""

    def __init__(self, cfg: Config, vocab_size: int | None = None, factorized_input: bool = False):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.factorized_input = factorized_input
        if vocab_size is not None:
            self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        elif factorized_input:
            levels_per_dim = cfg.L if cfg.bottleneck == "fsq" else 2
            self.code_emb = FactorizedCodeEmbedding(cfg.dq, levels_per_dim, cfg.d_model)
        else:
            self.in_proj = nn.Linear(cfg.dq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if vocab_size is not None:
            self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # weight-tied, like bpelm/bytelm
        elif cfg.bottleneck == "fsq":
            self.head = nn.Linear(cfg.d_model, cfg.dq * cfg.L)
        else:
            self.head = nn.Linear(cfg.d_model, cfg.dq)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # vocab/factorized mode: z is [B, T, ...] long (ids or per-dim
        # codes). continuous mode: [B, T, dq] float.
        B, T = z.shape[0], z.shape[1]
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_base, z.device)
        if self.vocab_size is not None:
            x = self.tok_emb(z)
        elif self.factorized_input:
            x = self.code_emb(z)
        else:
            x = self.in_proj(z)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.ln_f(x))
        if self.vocab_size is None and self.cfg.bottleneck == "fsq":
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
        self.lm = LatentLM(cfg, factorized_input=cfg.lm_factorized_input)

    def forward(self, byte_chunks: torch.Tensor):
        # byte_chunks: [B, T, K] long
        B, T, K = byte_chunks.shape
        z_hat, targets = self.encoder(byte_chunks.reshape(B * T, K))
        z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)
        # LM input: discrete per-dim `targets` (FactorizedCodeEmbedding) if
        # cfg.lm_factorized_input, else the default continuous `z_hat`.
        lm_in = targets if self.cfg.lm_factorized_input else z_hat

        if self.cfg.bottleneck == "bsq":
            return self._forward_bsq_tightly_coupled(byte_chunks, z_hat, targets, lm_in, B, T, K)

        # FSQ: unchanged, loosely-coupled Option A — each chunk's own code is
        # decoded independently of the LM (see module docstring; BSQ's
        # tightly-coupled path below hasn't been extended to FSQ yet).
        # Decoder is MaskGIT-style (see ChunkDecoder/maskgit_mask): loss is
        # CE at the masked positions only, not all K.
        flat_bytes = byte_chunks.reshape(B * T, K)
        x_masked, mask = maskgit_mask(flat_bytes, self.decoder.mask_id)
        rec_logits = self.decoder(z_hat.reshape(B * T, -1), x_masked)  # [B*T, K, vocab]
        rec_loss = F.cross_entropy(rec_logits[mask], flat_bytes[mask])
        recon_acc = (rec_logits.argmax(-1)[mask] == flat_bytes[mask]).float().mean()

        pred_logits = self.lm(lm_in)[:, :-1]        # predict z_{t+1} from z_{<=t}
        pred_targets = targets[:, 1:]
        pred_loss = F.cross_entropy(pred_logits.reshape(-1, self.cfg.L), pred_targets.reshape(-1))
        latent_acc = (pred_logits.argmax(-1) == pred_targets).float().mean()

        loss = rec_loss + pred_loss

        # BPB (handover §1.7 for the exact-ELBO case; no longer exact now that
        # the decoder is MaskGIT-style — rec_loss is mean CE over only the
        # *masked* positions sampled this batch, not all K, so bpb_rec is a
        # heuristic proxy, not a tight ELBO. The doc calls this out directly
        # (§1.4b): MaskGIT's masked CE is "heuristic but strong empirically";
        # only the time-free-masked-diffusion variant (not implemented here)
        # gives a proper ELBO -> exact bpb. pred_loss is nats/dim, so *dq
        # gives nats/chunk before dividing by the K bytes that chunk represents.
        bpb_rec = rec_loss / math.log(2)
        bpb_pred = (pred_loss * self.cfg.dq) / (K * math.log(2))
        bpb_total = bpb_rec + bpb_pred  # full ELBO

        return loss, {
            "rec_loss": rec_loss,
            "pred_loss": pred_loss,
            "recon_acc": recon_acc,
            "latent_acc": latent_acc,
            "bpb_total": bpb_total,
            "bpb_lm_only": bpb_pred,
        }

    def _forward_bsq_tightly_coupled(self, byte_chunks, z_hat, targets, lm_in, B, T, K):
        """Default path: encoder -> z_t -> LM -> predicted latent -> BSQ-quantized
        -> decoder -> bytes_{t+1}, graded against the *true* bytes_{t+1} — the
        decoder's primary target is the LM's prediction, not the encoder's own
        code (unlike FSQ's forward() above, and unlike this same model's old
        behavior). See docs/architecture.md for the full rationale.

        cfg.aux_recon additionally decodes the encoder's own z_t straight back
        to bytes_t (bypassing the LM) as an optional auxiliary regularizer —
        excluded from the reported bpb (it's not part of the generative path:
        qualitative_generate()/generate() never decode the encoder's code
        directly, only the LM's prediction of it)."""
        dq = self.cfg.dq

        v_pred = self.lm(lm_in)[:, :-1]                    # [B, T-1, dq]: predicts step t+1, pre-quantization
        z_pred, pred_bits = bsq_quantize(v_pred, dq, self.cfg.lfq)
        pred_targets = targets[:, 1:]

        # Code-level supervision: keeps the LM's raw output calibrated toward
        # the true next code's sign even before the decoder has learned to
        # turn that code into anything meaningful.
        pred_loss = F.binary_cross_entropy_with_logits(v_pred, pred_targets)
        latent_acc = (pred_bits == pred_targets).float().mean()

        # Primary loss: decode the LM's *predicted* next latent, grade against
        # the true next chunk's bytes. Decoder is MaskGIT-style (see
        # ChunkDecoder/maskgit_mask): loss is CE at the masked positions only.
        true_next_bytes = byte_chunks[:, 1:]
        flat_next_bytes = true_next_bytes.reshape(-1, K)
        x_masked, mask = maskgit_mask(flat_next_bytes, self.decoder.mask_id)
        rec_logits = self.decoder(z_pred.reshape(-1, dq), x_masked)  # [B*(T-1), K, vocab]
        rec_loss = F.cross_entropy(rec_logits[mask], flat_next_bytes[mask])
        recon_acc = (rec_logits.argmax(-1)[mask] == flat_next_bytes[mask]).float().mean()

        # Entropy regularization (see bsq_entropy_reg): raw value only, not
        # folded into `loss` here — like uncertainty weighting, whether/how
        # to weight it is a training-loop decision, not model architecture
        # (see main()'s training loop). Computed on v_pred (the LM's raw
        # output) since that's where the diagnosed generation collapse
        # actually lives, not on the encoder's own output.
        entropy_reg = bsq_entropy_reg(v_pred)

        loss = rec_loss + pred_loss
        metrics = {
            "rec_loss": rec_loss, "pred_loss": pred_loss, "recon_acc": recon_acc, "latent_acc": latent_acc,
            "entropy_reg": entropy_reg,
        }

        if self.cfg.aux_recon:
            flat_bytes = byte_chunks.reshape(-1, K)
            aux_x_masked, aux_mask = maskgit_mask(flat_bytes, self.decoder.mask_id)
            aux_logits = self.decoder(z_hat.reshape(B * T, dq), aux_x_masked)  # [B*T, K, vocab]
            aux_rec_loss = F.cross_entropy(aux_logits[aux_mask], flat_bytes[aux_mask])
            aux_recon_acc = (aux_logits.argmax(-1)[aux_mask] == flat_bytes[aux_mask]).float().mean()
            loss = loss + aux_rec_loss
            metrics["aux_rec_loss"] = aux_rec_loss
            metrics["aux_recon_acc"] = aux_recon_acc

        # bpb_rec is now a heuristic proxy (masked-CE, not a tight ELBO) —
        # see the FSQ forward() path's comment above for the same caveat.
        bpb_rec = rec_loss / math.log(2)
        bpb_pred = (pred_loss * dq) / (K * math.log(2))
        metrics["bpb_total"] = bpb_rec + bpb_pred
        metrics["bpb_lm_only"] = bpb_pred

        return loss, metrics

    @torch.no_grad()
    def generate(
        self, prompt_chunks: torch.Tensor, n_chunks: int, temperature: float = 1.0, bsq_sample_generation: bool = False,
    ) -> torch.Tensor:
        """FSQ: Option A (pure latent autoregression, handover §2.1) — the
        LM's sampled code is fed back directly, no re-encoding of decoded
        bytes. BSQ: tightly-coupled — the LM's raw output *is* the next
        latent (matching training); it's BSQ-quantized the same way the
        encoder quantizes, then decoded. `temperature` always affects FSQ's
        categorical sampling; for BSQ it only matters when
        `bsq_sample_generation=True` (see bsq_sample) — by default BSQ stays
        deterministic (sign), same as at training time."""
        self.eval()
        cfg = self.cfg
        B, T0, K = prompt_chunks.shape
        z_hat, targets = self.encoder(prompt_chunks.reshape(B * T0, K))
        z_history = z_hat.reshape(B, T0, -1)
        lm_history = targets.reshape(B, T0, -1) if cfg.lm_factorized_input else z_history
        out_chunks = [prompt_chunks]

        for _ in range(n_chunks):
            out = self.lm(lm_history)[:, -1]  # last position's prediction
            if cfg.bottleneck == "fsq":
                probs = F.softmax(out / temperature, dim=-1)              # [B, dq, L]
                levels = torch.multinomial(probs.reshape(-1, cfg.L), 1).reshape(B, cfg.dq)
                z_next = FSQ.levels_to_z(levels, cfg.L)
                next_targets = levels
            elif bsq_sample_generation:
                z_next = bsq_sample(out, cfg.dq, cfg.lfq, temperature)
                next_targets = (z_next > 0).long()
            else:
                z_next, next_targets_f = bsq_quantize(out, cfg.dq, cfg.lfq)  # LM output is a latent, quantized here
                next_targets = next_targets_f.long()

            next_chunk = maskgit_decode(self.decoder, z_next, cfg.maskgit_T)   # [B, K]
            out_chunks.append(next_chunk.unsqueeze(1))
            z_history = torch.cat([z_history, z_next.unsqueeze(1)], dim=1)
            if cfg.lm_factorized_input:
                lm_history = torch.cat([lm_history, next_targets.unsqueeze(1)], dim=1)
            else:
                lm_history = z_history

        return torch.cat(out_chunks, dim=1)


@torch.no_grad()
def score_continuation_bpb(model: QCuteLM, full_chunks: torch.Tensor, n_prompt_chunks: int, device: str) -> float:
    """Bpb restricted to the continuation region (chunks[n_prompt_chunks:]).
    full_chunks: [1, T, K], T = n_prompt_chunks + n_continuation_chunks.

    FSQ: decoder reconstruction CE for those chunks (from the encoder's own
    code) + LM next-code CE for predictions whose *target* chunk lies in the
    continuation — matches forward()'s loosely-coupled FSQ path.

    BSQ: matches forward()'s tightly-coupled path — decode the LM's
    *predicted* next latent (not the encoder's own code) and grade against
    the true continuation bytes; pred nats from the same code-level BCE."""
    model.eval()
    cfg = model.cfg
    B, T, K = full_chunks.shape
    z_hat, targets = model.encoder(full_chunks.reshape(B * T, K))
    z_hat, targets = z_hat.reshape(B, T, -1), targets.reshape(B, T, -1)
    lm_in = targets if cfg.lm_factorized_input else z_hat
    start = max(0, n_prompt_chunks - 1)  # first prediction whose target is a continuation chunk

    if cfg.bottleneck == "bsq":
        v_pred = model.lm(lm_in)[:, :-1]
        z_pred, pred_bits = bsq_quantize(v_pred, cfg.dq, cfg.lfq)
        pred_targets = targets[:, 1:]
        true_next_bytes = full_chunks[:, 1:]

        cont_z_pred = z_pred[:, start:]
        cont_true_bytes = true_next_bytes[:, start:].reshape(-1, K)
        cont_x_masked, cont_mask = maskgit_mask(cont_true_bytes, model.decoder.mask_id)
        rec_logits = model.decoder(cont_z_pred.reshape(-1, cfg.dq), cont_x_masked)
        rec_nats = F.cross_entropy(rec_logits[cont_mask], cont_true_bytes[cont_mask])

        cont_pred_logits, cont_pred_targets = v_pred[:, start:], pred_targets[:, start:]
        pred_nats = F.binary_cross_entropy_with_logits(cont_pred_logits, cont_pred_targets)
    else:
        flat_bytes = full_chunks.reshape(B * T, K)
        x_masked, mask = maskgit_mask(flat_bytes, model.decoder.mask_id)
        rec_logits = model.decoder(z_hat.reshape(B * T, -1), x_masked).reshape(B, T, K, cfg.vocab)
        mask = mask.reshape(B, T, K)
        cont_rec_logits = rec_logits[:, n_prompt_chunks:]
        cont_rec_targets = full_chunks[:, n_prompt_chunks:]
        cont_mask = mask[:, n_prompt_chunks:]
        rec_nats = F.cross_entropy(cont_rec_logits[cont_mask], cont_rec_targets[cont_mask])

        pred_logits = model.lm(lm_in)[:, :-1]
        pred_targets = targets[:, 1:]
        cont_pred_logits, cont_pred_targets = pred_logits[:, start:], pred_targets[:, start:]
        pred_nats = F.cross_entropy(cont_pred_logits.reshape(-1, cfg.L), cont_pred_targets.reshape(-1))

    bpb_rec = rec_nats / math.log(2)
    bpb_pred = (pred_nats * cfg.dq) / (K * math.log(2))
    model.train()
    return (bpb_rec + bpb_pred).item()


def qualitative_generate(
    model: QCuteLM, prompt_chunks: torch.Tensor, n_gen_chunks: int,
    ground_truth_chunks: torch.Tensor | None, device: str, log=print,
    temperature: float = 1.0, bsq_sample_generation: bool = False,
) -> None:
    """Generate a continuation from a prompt (dataset-drawn or user-supplied)
    and, if a real ground-truth continuation is available (dataset-drawn),
    show it alongside the model's guess plus the model's bpb on the truth —
    a qualitative complement to the aggregate val_bpb number."""
    out = model.generate(prompt_chunks, n_gen_chunks, temperature=temperature, bsq_sample_generation=bsq_sample_generation)
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


@torch.no_grad()
def eval_ae_recon_acc(model: "QCuteLM", data_iter, n_batches: int) -> float:
    """Encoder+decoder only, bypassing the LM entirely: bytes -> encoder ->
    z_hat -> decoder -> bytes. Used by pretrain_autoencoder's stopping
    criterion (and its own progress logging) so it measures the actual
    quantity the threshold is about, not a noisy single training batch."""
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        batch = next(data_iter)
        B, T, K = batch.shape
        flat_bytes = batch.reshape(B * T, K)
        z_hat, _ = model.encoder(flat_bytes)
        x_masked, mask = maskgit_mask(flat_bytes, model.decoder.mask_id)
        logits = model.decoder(z_hat, x_masked)
        correct += (logits.argmax(-1)[mask] == flat_bytes[mask]).float().sum().item()
        total += mask.sum().item()
    model.train()
    return correct / total


def pretrain_autoencoder(model: "QCuteLM", train_data: torch.Tensor, val_iter, args, log) -> float:
    """Optional encoder+decoder-only warm-start, run before joint training:
    train on plain reconstruction (bytes -> encoder -> z_hat -> decoder ->
    bytes, LM untouched) until val recon_acc clears --pretrain_target_acc or
    --pretrain_steps is hit, whichever first. Own optimizer/LR, separate from
    the joint-training one below (a fresh AdamW over encoder+decoder params
    only, deliberately not touching the LM's weights).

    Motivation (see docs/status.md's decoder-bottleneck findings): the joint
    forward pass makes the decoder learn to decode two different code
    distributions from scratch simultaneously (the encoder's true z_hat via
    aux_recon, and the LM's noisy/evolving z_pred) while the LM is also still
    learning. Giving the decoder a head start on the *true*-code mapping
    alone, before the LM enters the picture, is a less elegant lever than an
    architecture change but much cheaper to try. Returns the final recon_acc
    reached (for logging/deciding whether the threshold was hit)."""
    cfg = model.cfg
    device = next(model.parameters()).device
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.AdamW(ae_params, lr=args.pretrain_lr, betas=(0.9, 0.95), weight_decay=0.1)
    train_iter = batch_iter(train_data, args.batch_size, args.seq_chunks, cfg.K, device)

    model.train()
    recon_acc = 0.0
    pbar = tqdm(range(1, args.pretrain_steps + 1), desc="pretrain_ae", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.pretrain_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(train_iter)
        B, T, K = batch.shape
        flat_bytes = batch.reshape(B * T, K)
        z_hat, _ = model.encoder(flat_bytes)
        x_masked, mask = maskgit_mask(flat_bytes, model.decoder.mask_id)
        logits = model.decoder(z_hat, x_masked)
        loss = F.cross_entropy(logits[mask], flat_bytes[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae_params, args.grad_clip)
        opt.step()
        train_acc = (logits.argmax(-1)[mask] == flat_bytes[mask]).float().mean()
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", train_recon_acc=f"{train_acc.item()*100:.2f}%")
        if step % args.pretrain_eval_every == 0 or step == args.pretrain_steps:
            recon_acc = eval_ae_recon_acc(model, val_iter, args.eval_batches)
            log(f"{pbar}  val_recon_acc {recon_acc*100:.2f}%", step=step,
                pretrain_train_recon_acc=train_acc.item(), pretrain_recon_acc=recon_acc)
            if recon_acc >= args.pretrain_target_acc:
                log(f"pretrain_ae target reached: recon_acc {recon_acc*100:.2f}% >= "
                    f"{args.pretrain_target_acc*100:.1f}% at step {step}", step=step, pretrain_recon_acc=recon_acc)
                break
    else:
        log(f"pretrain_ae max steps ({args.pretrain_steps}) reached without hitting target, "
            f"final recon_acc {recon_acc*100:.2f}%", pretrain_recon_acc=recon_acc)
    return recon_acc


@torch.no_grad()
def build_code_vocab(model: "QCuteLM", data: torch.Tensor, K: int, device: str, batch_size: int = 1024):
    """Encode the whole corpus with the (frozen) encoder and collect the
    distinct codes that actually occur — a vocabulary built from this
    project's own learned tokenizer, the same way BPE builds one from merge
    rules, instead of the full combinatorial codebook (2^dq for BSQ, 8^dq
    for FSQ — both far too large to embed directly, see LatentLM's
    docstring/architecture.md). Dedup key is `targets` (the exact discrete
    code — integer levels for FSQ, bits for BSQ/LFQ), not `z_hat` (float,
    unsafe to hash directly even though it's STE-exact in practice).
    Returns (code_table: [V, dq] float, code_to_id: dict[tuple, int])."""
    model.eval()
    n = (len(data) // K) * K
    chunks = data[:n].reshape(-1, K).to(device)
    code_to_id: dict[tuple, int] = {}
    z_rows = []
    for i in range(0, chunks.size(0), batch_size):
        z_hat, targets = model.encoder(chunks[i : i + batch_size])
        targets_cpu, z_cpu = targets.cpu(), z_hat.cpu()
        for j in range(targets_cpu.size(0)):
            key = tuple(targets_cpu[j].tolist())
            if key not in code_to_id:
                code_to_id[key] = len(code_to_id)
                z_rows.append(z_cpu[j])
    model.train()
    return torch.stack(z_rows, dim=0), code_to_id


@torch.no_grad()
def encode_to_vocab_ids(model: "QCuteLM", data: torch.Tensor, K: int, code_to_id: dict, device: str, batch_size: int = 1024) -> torch.Tensor:
    """Map each K-byte chunk of data to its vocab id via code_to_id (built by
    build_code_vocab, from train data only — like a tokenizer's vocab).
    Codes not seen during vocab-building (e.g. val-only chunks) map to a
    reserved UNK id = len(code_to_id)."""
    model.eval()
    unk_id = len(code_to_id)
    n = (len(data) // K) * K
    chunks = data[:n].reshape(-1, K).to(device)
    ids = []
    for i in range(0, chunks.size(0), batch_size):
        _, targets = model.encoder(chunks[i : i + batch_size])
        for row in targets.cpu().tolist():
            ids.append(code_to_id.get(tuple(row), unk_id))
    model.train()
    return torch.tensor(ids, dtype=torch.long)


def vocab_id_batch_iter(ids: torch.Tensor, batch_size: int, context: int, device: str):
    seq_len = context + 1
    n = (len(ids) - 1) // seq_len
    while True:
        starts = torch.randint(0, n, (batch_size,))
        batch = torch.stack([ids[i * seq_len : (i + 1) * seq_len] for i in starts])
        yield batch.to(device)


@torch.no_grad()
def vocab_lm_generate(
    lm: LatentLM, decoder: ChunkDecoder, code_table_full: torch.Tensor,
    prompt_ids: torch.Tensor, n_new: int, temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressively sample n_new vocab ids from lm, then decode them
    back to bytes via code_table_full (row per vocab id, including the UNK
    row) + the frozen MaskGIT decoder. prompt_ids: [B, T0] long -> bytes:
    [B, n_new, K] long."""
    lm.eval()
    tokens = prompt_ids.clone()
    for _ in range(n_new):
        logits = lm(tokens)[:, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_id], dim=1)
    lm.train()
    new_ids = tokens[:, prompt_ids.size(1):]
    B, T = new_ids.shape
    z = code_table_full[new_ids.reshape(-1)]
    bytes_out = maskgit_decode(decoder, z, decoder.cfg.maskgit_T)
    return bytes_out.reshape(B, T, -1)


def train_vocab_lm(
    cfg: Config, decoder: ChunkDecoder, code_table: torch.Tensor,
    train_ids: torch.Tensor, val_ids: torch.Tensor, args, log, run_name: str, device: str,
) -> LatentLM:
    """Trains a fresh, standalone LatentLM in discrete-vocabulary mode over
    the frozen tokenizer's empirically-built vocab — see build_code_vocab
    and LatentLM's docstring. This is the "dumber, simpler" path: no
    tightly-coupled joint co-adaptation, no continuous latent regression —
    just a plain categorical causal-LM over a fixed vocab, exactly like
    qcute.bpelm's architecture, except the tokenizer is this project's own
    learned encoder/decoder instead of sentencepiece BPE."""
    vocab_size = code_table.size(0) + 1  # +1 for the UNK id (see encode_to_vocab_ids)
    code_table_full = torch.cat([code_table, torch.zeros(1, cfg.dq)], dim=0).to(device)
    lm = LatentLM(cfg, vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in lm.parameters())
    log(f"vocab_lm: vocab_size={vocab_size} (incl. UNK) params={n_params/1e6:.2f}M K={cfg.K}")

    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
    context = args.seq_chunks
    train_iter = vocab_id_batch_iter(train_ids, args.batch_size, context, device)
    val_iter = vocab_id_batch_iter(val_ids, args.batch_size, context, device)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)
    K = cfg.K

    lm.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_vocab_lm", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(train_iter)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = lm(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        acc = (logits.argmax(-1) == targets).float().mean()
        bpb = loss / (K * math.log(2))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), args.grad_clip)
        opt.step()
        pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", token_acc=f"{acc.item()*100:.2f}%", bpb=f"{bpb.item():.4f}")
        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), token_acc=acc.item(), bpb=bpb.item())
        if step % args.eval_every == 0 or step == args.steps:
            lm.eval()
            with torch.no_grad():
                vbatch = next(val_iter)
                vin, vtgt = vbatch[:, :-1], vbatch[:, 1:]
                vlogits = lm(vin)
                vloss = F.cross_entropy(vlogits.reshape(-1, vocab_size), vtgt.reshape(-1))
                vacc = (vlogits.argmax(-1) == vtgt).float().mean()
                vbpb = vloss / (K * math.log(2))
            log(
                f"step {step:5d}  val_token_acc {vacc.item()*100:.2f}%  val_bpb {vbpb.item():.4f}",
                step=step, val_token_acc=vacc.item(), val_bpb=vbpb.item(),
            )
            checkpointer.step(
                {"lm": lm.state_dict(), "code_table": code_table, "vocab_size": vocab_size,
                 "cfg": asdict(cfg), "step": step, "val_bpb": vbpb.item()},
                vbpb.item(),
            )
            # Qualitative sample every eval — actual (readable-or-not) text,
            # not just the bpb number, so training progress is visible even
            # while bpb is still an imperfect proxy (see MaskGIT's heuristic-
            # loss caveat elsewhere in this file).
            prompt_len = min(8, vin.size(1))
            prompt_ids = vin[:1, :prompt_len]
            with torch.no_grad():
                prompt_chunks = maskgit_decode(decoder, code_table_full[prompt_ids[0]], decoder.cfg.maskgit_T)
            prompt_bytes = bytes(prompt_chunks.reshape(-1).tolist())
            gen_chunks = vocab_lm_generate(lm, decoder, code_table_full, prompt_ids, n_new=16)
            gen_bytes = bytes(gen_chunks[0].reshape(-1).tolist())
            log(f"qual_prompt (decoded from val ids): {prompt_bytes!r}", step=step)
            log(f"qual_generated:                     {gen_bytes!r}", step=step)
            lm.train()
    return lm


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

    qualitative_generate(
        model, prompt_chunks, n_gen_chunks, ground_truth_chunks, device, log=log,
        temperature=args.qual_temperature, bsq_sample_generation=args.bsq_sample_generation,
    )


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
    p.add_argument("--grad_clip", type=float, default=1.0)
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
        "--pretrain_ae", action="store_true",
        help="before joint training, train encoder+decoder alone (LM untouched) on plain reconstruction "
             "until --pretrain_target_acc or --pretrain_steps, whichever first; see pretrain_autoencoder()"
    )
    p.add_argument("--pretrain_steps", type=int, default=3000, help="max steps for --pretrain_ae")
    p.add_argument("--pretrain_target_acc", type=float, default=0.95, help="stop --pretrain_ae early once val recon_acc clears this")
    p.add_argument("--pretrain_lr", type=float, default=1e-3, help="separate LR for the --pretrain_ae phase (its own AdamW instance)")
    p.add_argument("--pretrain_eval_every", type=int, default=100)
    p.add_argument(
        "--pretrain_checkpoint_path", type=Path, default=None,
        help="where --pretrain_ae saves the encoder+decoder-only checkpoint; default checkpoints/<run_name>/pretrain_ae.pt"
    )
    p.add_argument(
        "--init_encoder_decoder_from", type=Path, default=None,
        help="warm-start encoder+decoder weights (LM stays freshly initialized) from a checkpoint saved by --pretrain_ae "
             "in an earlier run, instead of re-running pretraining here"
    )
    p.add_argument(
        "--freeze_after_pretrain", action="store_true",
        help="requires --pretrain_ae: freeze encoder+decoder after pretraining, build a discrete vocabulary from the "
             "frozen tokenizer's actual codes, and train a plain categorical LM (embedding table + softmax head) over "
             "it instead of continuous-latent joint training; see train_vocab_lm()"
    )
    p.add_argument(
        "--disable_aux_recon", action="store_true",
        help="BSQ only: disable the auxiliary encoder-latent->decoder loss, training on the tightly-coupled LM->decoder path alone"
    )
    p.add_argument(
        "--disable_pred_loss", action="store_true",
        help="BSQ only: drop the explicit code-level BCE supervision (pred_loss) entirely, training on rec_loss "
             "(+aux_rec_loss if enabled) alone — the LM still gets gradient via rec_loss's STE path through "
             "bsq_quantize, just without a direct 'match the true next code' constraint; pred_loss/latent_acc "
             "are still computed and logged, just excluded from backward"
    )
    p.add_argument(
        "--lfq", action="store_true",
        help="BSQ only: regress the quantizer to plain LFQ (sign the raw projection, hypercube corners) "
             "instead of BSQ (L2-normalize onto the hypersphere first, then sign)"
    )
    p.add_argument(
        "--mixer", choices=["attention", "conv"], default="attention",
        help="encoder/decoder cross-K-position mixer (see MixerBlock): full non-causal self-attention, "
             "or a single non-causal 1D conv (kernel_size=K, cheaper, no softmax)"
    )
    p.add_argument(
        "--disable_mixer_mlp", action="store_true",
        help="drop the post-mixer MLP block in ChunkEncoder/ChunkDecoder, testing the mixer (attention/tcn) alone"
    )
    p.add_argument(
        "--lm_factorized_input", action="store_true",
        help="LM input: compositional per-dim (PQ-style) embedding on the discrete code instead of a linear "
             "projection of continuous z (see FactorizedCodeEmbedding); output format unchanged"
    )
    p.add_argument(
        "--uncertainty_weighting", action="store_true",
        help="BSQ only: Kendall & Gal (2018)-style learned homoscedastic weighting on rec_loss/aux_rec_loss/entropy_reg "
             "(one trainable log_var scalar per loss, in the training loop, not the model) instead of a fixed 1x sum"
    )
    p.add_argument(
        "--entropy_reg_weight", type=float, default=0.0,
        help="BSQ only: LFQ/BSQ-style entropy regularization weight on the LM's predicted latent (bsq_entropy_reg), "
             "0 disables; ignored (superseded by its own learned log_var) if --uncertainty_weighting is also on"
    )
    p.add_argument(
        "--qual_gen_bytes", type=int, default=0,
        help="if >0, after training/eval generate this many bytes qualitatively and log prompt/generated/(ground truth)"
    )
    p.add_argument("--qual_source", choices=["train", "val", "user"], default="val")
    p.add_argument("--qual_prompt_bytes", type=int, default=64, help="prompt length when --qual_source is train/val (rounded down to a multiple of K)")
    p.add_argument("--qual_user_text", type=str, default=None, help="prompt text when --qual_source user (utf-8 encoded, padded to a multiple of K)")
    p.add_argument("--qual_temperature", type=float, default=1.0, help="FSQ always; BSQ only if --bsq_sample_generation")
    p.add_argument(
        "--bsq_sample_generation", action="store_true",
        help="BSQ only: Bernoulli-sample each latent bit from sigmoid(logit/temperature) at generate() time "
             "instead of the deterministic hard sign used at training time; see bsq_sample()"
    )

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in config_vars.items() if k in known})
    args = p.parse_args()
    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")
    if args.qual_gen_bytes > 0 and args.qual_source == "user" and not args.qual_user_text:
        p.error("--qual_source user requires --qual_user_text")
    if args.freeze_after_pretrain and not args.pretrain_ae:
        p.error("--freeze_after_pretrain requires --pretrain_ae")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        cfg = Config(**ckpt["cfg"])
        model = QCuteLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
    else:
        cfg = build_config(
            args.bottleneck, args.dq, K=args.K, aux_recon=not args.disable_aux_recon, lfq=args.lfq,
            mixer=args.mixer, mixer_mlp=not args.disable_mixer_mlp, lm_factorized_input=args.lm_factorized_input,
        )
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

    if args.init_encoder_decoder_from is not None:
        ae_ckpt = torch.load(args.init_encoder_decoder_from, map_location=device)
        model.encoder.load_state_dict(ae_ckpt["encoder"])
        model.decoder.load_state_dict(ae_ckpt["decoder"])
        log(f"encoder/decoder warm-started from {args.init_encoder_decoder_from} "
            f"(pretrain recon_acc {ae_ckpt.get('recon_acc', float('nan'))*100:.2f}%)")

    if args.pretrain_ae:
        pretrain_recon_acc = pretrain_autoencoder(model, train_data, val_iter, args, log)
        pretrain_ckpt_path = args.pretrain_checkpoint_path or (args.checkpoint_dir / run_name / "pretrain_ae.pt")
        pretrain_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
             "cfg": asdict(cfg), "recon_acc": pretrain_recon_acc},
            pretrain_ckpt_path,
        )
        log(f"pretrain_ae checkpoint saved to {pretrain_ckpt_path}")

    if args.freeze_after_pretrain:
        # The "dumber, simpler" alternative to tightly-coupled joint
        # training: freeze the just-pretrained encoder+decoder (the
        # "vocab" checkpoint saved above) so the LM phase can't erase it
        # the way plain --pretrain_ae did (docs/status.md), then hand the
        # LM a real discrete vocabulary + embedding table + categorical
        # softmax head over the frozen tokenizer's actual codes — matching
        # how FSQ/LFQ/BSQ papers' downstream priors are trained (frozen
        # tokenizer first, then a separate categorical LM), and matching
        # qcute.bpelm's architecture, just with a learned tokenizer instead
        # of BPE. Replaces the rest of main()'s continuous joint-training
        # loop entirely for this run.
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False
        log("encoder+decoder frozen after pretraining")

        code_table, code_to_id = build_code_vocab(model, train_data, cfg.K, device)
        log(f"vocab built from train data: {len(code_to_id)} distinct codes "
            f"(of {(len(train_data)//cfg.K)} chunks)")
        train_ids = encode_to_vocab_ids(model, train_data, cfg.K, code_to_id, device)
        val_ids = encode_to_vocab_ids(model, val_data, cfg.K, code_to_id, device)
        unk_frac = (val_ids == len(code_to_id)).float().mean().item()
        log(f"val UNK rate (codes unseen in train): {unk_frac*100:.2f}%")

        train_vocab_lm(cfg, model.decoder, code_table, train_ids, val_ids, args, log, run_name, device)
        return

    # Uncertainty weighting (Kendall & Gal 2018-style learned homoscedastic
    # weighting on the two CE losses, rec_loss and aux_rec_loss) is a
    # training-time loss-combination choice, not part of the model — the
    # log_var parameters live here, not on QCuteLM, and model.forward()'s
    # returned `loss` (an unweighted sum) is only used when this is off;
    # when on, the actual backward loss is recomputed below from the raw
    # per-term losses in `metrics`. pred_loss (BCE, different loss family
    # and scale) is left unweighted either way. Motivated by the diagnosed
    # gradient-scale imbalance (docs/status.md: aux_rec_loss's gradient
    # norm dwarfing rec_loss/pred_loss under LFQ) — lets training learn to
    # down-weight whichever loss is intrinsically larger/harder instead of
    # us guessing a fixed coefficient.
    uw = args.uncertainty_weighting and cfg.bottleneck == "bsq"
    entropy_reg_on = args.entropy_reg_weight > 0 and cfg.bottleneck == "bsq"
    extra_params = []
    if uw:
        log_var_rec = nn.Parameter(torch.zeros((), device=device))
        extra_params.append(log_var_rec)
        if cfg.aux_recon:
            log_var_aux = nn.Parameter(torch.zeros((), device=device))
            extra_params.append(log_var_aux)
        if entropy_reg_on:
            log_var_entropy = nn.Parameter(torch.zeros((), device=device))
            extra_params.append(log_var_entropy)

    opt = torch.optim.AdamW(list(model.parameters()) + extra_params, lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=0.1)
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
        if uw:
            loss = torch.exp(-log_var_rec) * metrics["rec_loss"] + log_var_rec
            if not args.disable_pred_loss:
                loss = loss + metrics["pred_loss"]
            if "aux_rec_loss" in metrics:
                loss = loss + torch.exp(-log_var_aux) * metrics["aux_rec_loss"] + log_var_aux
        elif args.disable_pred_loss:
            # Ablation: no direct code-level supervision at all — pred_loss's
            # STE gradient path (rec_loss -> bsq_quantize -> v_pred) still
            # trains the LM, just without the explicit "match the true next
            # code's bits" constraint; pred_loss/latent_acc are still
            # computed and logged below, just excluded from backward.
            loss = metrics["rec_loss"]
            if "aux_rec_loss" in metrics:
                loss = loss + metrics["aux_rec_loss"]
        if entropy_reg_on:
            if uw:
                # same learned-log_var treatment as rec/aux above, no extra
                # fixed multiplier — --entropy_reg_weight is only the fixed-
                # coefficient path below (uw off).
                loss = loss + torch.exp(-log_var_entropy) * metrics["entropy_reg"] + log_var_entropy
            else:
                loss = loss + args.entropy_reg_weight * metrics["entropy_reg"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + extra_params, args.grad_clip)
        opt.step()
        postfix = {
            "lr": f"{lr:.2e}", "loss": f"{loss.item():.4f}", "rec": f"{metrics['rec_loss'].item():.4f}",
            "pred": f"{metrics['pred_loss'].item():.4f}", "recon_acc": f"{metrics['recon_acc'].item()*100:.2f}%",
            "latent_acc": f"{metrics['latent_acc'].item()*100:.2f}%",
            "bpb": f"{metrics['bpb_total'].item():.4f}", "bpb_lm_only": f"{metrics['bpb_lm_only'].item():.4f}",
        }
        if "aux_rec_loss" in metrics:
            postfix["aux_rec"] = f"{metrics['aux_rec_loss'].item():.4f}"
            postfix["aux_recon_acc"] = f"{metrics['aux_recon_acc'].item()*100:.2f}%"
        if entropy_reg_on:
            postfix["entropy_reg"] = f"{metrics['entropy_reg'].item():.4f}"
        if uw:
            postfix["log_var_rec"] = f"{log_var_rec.item():.3f}"
            if "aux_rec_loss" in metrics:
                postfix["log_var_aux"] = f"{log_var_aux.item():.3f}"
            if entropy_reg_on:
                postfix["log_var_entropy"] = f"{log_var_entropy.item():.3f}"
        pbar.set_postfix(postfix)
        if step % args.log_every == 0:
            extra_log = {}
            if uw:
                extra_log["log_var_rec"] = log_var_rec.item()
                if "aux_rec_loss" in metrics:
                    extra_log["log_var_aux"] = log_var_aux.item()
                if entropy_reg_on:
                    extra_log["log_var_entropy"] = log_var_entropy.item()
            log(f"{pbar}", step=step, lr=lr, **{k: v.item() for k, v in metrics.items()}, **extra_log)
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
