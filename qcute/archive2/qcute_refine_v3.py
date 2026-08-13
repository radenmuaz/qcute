"""qcute.qcute_refine_v3 — CLONE of qcute_refine_v2.py (same recursive NTP
encoder tower, same DecoderLevel/tok_loss reconstruction path — see below,
unchanged), PLUS ONE structural addition: EncoderLevel FUSION
(Config.fuse_encoder_levels, default True).

WHY (the finding this fixes): in v2, `val_bpb` — the ONLY metric every
comparison table in docs/status.md is built from, and what checkpointer.
step() uses to pick best.pt — is computed exclusively from `byte_loss`
(EncoderLevel[0]'s own NTP loss), which is produced by a purely bottom-up
sweep with ZERO access to the coarser code, cross-attention, or the
DecoderLevel/tok_loss path (whose own h reads are DETACHED, by design —
see below). Cross-attention could only ever move `tok_loss`/`pair0_tok_
acc`, a metric no baseline in this project is ever compared against.
Session's KV-contribution probes (docs/kv_contribution.md) already showed
cross-attention genuinely helps `tok_loss` — that benefit was real, just
structurally invisible to every val_bpb-based comparison this whole
session made, and plausibly explains why the full v2 architecture lost to
a trivial 1-layer bytelm diagnostic (docs/status.md) despite the
hierarchy's added complexity.

THE FIX — EncoderLevel fusion (v3's only real change, see EncoderLevel.
_fuse and RefineLM.forward's own "v3 PASS 2" comment for the full
mechanism): RefineLM.forward runs the SAME bottom-up sweep as v2 first
(PASS 1 — required regardless, since level i+1's own input IS level i's
own output code, so level i+1 structurally cannot exist before level i
has finished; "fuse before running the encoder" is circular for a level
fusing from ITSELF in one pass). Once PASS 1 has produced every level's
hidden state, levels 0..n_active-2 are re-run (PASS 2, same weights, same
input) — this time with a NEW cross-attention step (EncoderLevel._fuse)
BEFORE their own self.blocks, attending to the level-above's own PASS-1
hidden state (same jagged-staircase causal mask as DecoderLevel already
uses, factored into the shared jagged_causal_mask_and_positions
function). PASS 2's ntp_loss/h REPLACES PASS 1's for that level — so
byte_loss (and val_bpb, and checkpointing) now genuinely depends on, and
can learn to use, the coarser code. Q is the level's own embedding
directly (no separately-learned q-embedding — "encoder reads this
directly" per the session ask); KV is the level-above's hidden state via
one new small projection (fuse_kv_proj), detached before use — same
"don't let this reshape the OTHER level's own hidden state" principle
DecoderLevel's own cross-attention reads already follow, so only the
FUSING level's own weights (fuse_cross, its own embed/blocks) get shaped
by this, not the level being attended to. Cost: roughly one extra
self-attention-block-sized forward pass per fused level (levels below the
top only) — level 0 is usually the shallowest (tier_n_layers=1), so
modest relative to the whole model.

Config.fuse_encoder_levels=False reproduces v2's forward exactly (PASS 1
only) — a genuine, same-file A/B control for isolating fusion's own
effect from any other v3-vs-v2 difference.

EVERYTHING BELOW THIS POINT (the encoder tower's own per-level shape,
CodeEmbed/code_embed_mode, DecoderLevel/tok_loss path, byte_ntp_weight)
is UNCHANGED from qcute_refine_v2.py — same recursive NTP encoder tower
as qcute_refine.py, with the block-local joint-chain-MTP Detokenizer
replaced by a cross-attention decoder that REUSES the encoder tower's own
already-computed hidden states instead of running any new self-attention
trunk of its own.

ENCODER TOWER: unchanged from qcute_refine.py — EncoderLevel[i] embeds its
own input sequence x^(i) (bytes at i=0, else c_{i-1}), runs a causal
transformer (own weights, tier_n_layers[i] deep), keeps an always-on
unconditioned NTP loss on its own next input element (own head, own
target — "so that targets are stable"), and every K[i] positions BSQ-
quantizes into its own emitted code c^(i), which becomes level i+1's
entire input. EncoderLevel[i].forward now returns its post-ln_f hidden
state h^(i) as a first-class output (qcute_refine.py already computed
this and discarded it; v2 is what actually uses it).

TOKENIZER (decoder), one per ADJACENT level pair (i, i+1), i = 0..N-2 —
NOT one per level like qcute_refine.py's Detokenizer_i (there is no
decoder above the top level; nothing coarser exists to cross-attend to):

    Q  = h^(i)    ("previous level['s] code LM" — EncoderLevel[i]'s own
                    hidden states, the finer sequence being decoded)
    KV = h^(i+1)  ("current level['s] code LM" — EncoderLevel[i+1]'s own
                    hidden states; EncoderLevel[i+1]'s OWN INPUT is
                    exactly c^(i), so its hidden state at code-block index
                    b is already, by construction, "as of" real time
                    (b+1)*K[i] — no new computation needed to get this)

Both h^(i) and h^(i+1) are DETACHED before use — this decoder's loss must
not reshape either EncoderLevel's own hidden state (which stays trained
purely by its own unconditioned NTP loss; two objectives competing over
the same hidden state is exactly the moving-target failure mode this
whole design avoids elsewhere).

A single cross-attention TRANSFORMER BLOCK (cross-attn sublayer + MLP
sublayer, each pre-norm + residual, mirroring this file's own causal
`Block`'s shape) combines Q and KV. Causal safety: query position t may
only attend to KV block b once b is FULLY complete, i.e. b < (t+1)//K[i]
(the same "past, already-resolved block" rule qcute_refine.py's module
docstring worked through) — enforced via an explicit boolean attention
mask, not RoPE (Q and KV live at different granularities/lengths, so a
shared rotary basis doesn't apply). A single learned "null" KV slot is
prepended and always visible, so early positions (before any KV block is
complete) still get a well-defined attention distribution instead of an
all-masked row — the same "zero-KV is load-bearing, not just a
regularizer" mechanism documented for qcute_refine's own variable-length-
tokenizer sibling forks.

DECODE HEAD: predicts x^(i)'s own NEXT token — single position, not a
joint K-block prediction (the old Detokenizer's whole "multi-token
prediction" mechanism is gone, not merely cheapened). Defaults to a
single plain nn.Linear (`tok_head_mode="linear"`, independent per-bit
logits — cheap, mirrors bytelm's own parallel-Linear-head style);
`tok_head_mode="chain"` swaps in the exact chain-rule BitPredictHead for
comparison, at the ~200-1800x per-call cost qcute_refine.py's own session
diagnosis measured.

Generation is unaffected: only EncoderLevel[0]'s own NTP head is
generative (a genuinely new byte requires nothing but causal history;
DecoderLevel's own KV side needs an ALREADY-COMPLETE code block, so it
can decode existing children but never propose new ones) —
generate_no_cache/generate_kv_cache/validate_generation are copied
unchanged from qcute_refine.py.

No shared imports with qcute_refine.py or any qcutelm_vlt* fork (self-
contained-module convention, matching how qcutelm_vlt7->vlt8->...
evolved) — everything duplicated.

    uv run python -m qcute.qcute_refine_v3 --config configs/qcute_refine_v3_rope.py
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
    Ks: tuple[int, ...] = (2, 2, 2)
    dqs: tuple[int, ...] = (8, 8, 8)
    tier_d_models: tuple[int, ...] = (96, 96, 96)
    tier_n_layers: tuple[int, ...] = (1, 1, 1)
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int | tuple[int, ...] = 128   # single int: broadcast to every level (backward-
                                    # compatible default). Per-level tuple (length n_levels): lets the
                                    # TOP level get its OWN, genuinely-sub-full window instead of always
                                    # inheriting whatever value the finer levels use — e.g. with a shared
                                    # scalar, a top level whose own sequence length happens to equal the
                                    # window falls back to dense (T>window is False, see
                                    # CausalSelfAttention) purely by coincidence, not by choice. -1 in
                                    # either form means dense for that level.
    rope_base: float = 10000.0
    bit_chain_n_heads: int = 2
    bit_chain_gamma: float = 1.0
    bit_chain_fixed_kernel: bool = True
    bit_head_class: str = "attn"  # which BitPredictHead* implementation backs every "chain"-style head in
                                    # the model (byte level's own bits-mode NTP/decode head, and any
                                    # level's code_head_mode/tok_head_mode=="chain"): "attn" (default,
                                    # original) = BitPredictHeadAttn, self-attention over the bit chain.
                                    # "conv" = BitPredictHeadConv, causal Conv1d instead. "ssm" =
                                    # BitPredictHeadSSM, linear-decay recurrence instead. All three are
                                    # drop-in equivalent in API (verified fixed/loop-consistent) — see
                                    # each class's own docstring for the tradeoffs.
    bit_conv_kernel_size: int | None = None  # None (default) = dq (full receptive field) — see
                                    # BitPredictHeadConv's own docstring for why windowing isn't obviously
                                    # the right inductive bias here. Only used when bit_head_class=="conv".
    bit_conv_impl: str = "matmul"  # "matmul" (default, session reparam) or "conv1d" (original) — same
                                    # op (fixed causal window, weights shared across positions), different
                                    # dispatch; see BitPredictHeadConv's own docstring. Only used when
                                    # bit_head_class=="conv".
    bit_inner_downsample: int = 1  # 1 (default) = no downsampling, identical to pre-flag behavior (no
                                    # extra op/params). >1 (2, 4, ...) projects the incoming hidden vector
                                    # down to d_model//bit_inner_downsample once, then runs every chain op
                                    # inside whichever BitPredictHead* class (embeds/attn/conv/ssm-state/
                                    # head) at that smaller width instead of the full d_model — applies to
                                    # ALL THREE bit_head_class variants uniformly. Session: "flag to
                                    # downsample bit predict embeds or inner". Must evenly divide d_model
                                    # (and, for bit_head_class=="attn", the resulting inner dim must be
                                    # divisible by bit_chain_n_heads too).
    bit_ssm_d_state: int | None = None       # None (default) = d_model. Only used when bit_head_class=="ssm".
    code_ntp_weight: float = 1.0  # scales levels>0's own NTP loss (level 0's byte_loss scaled separately,
                                    # see byte_ntp_weight below). ==0.0 SKIPS those levels' ntp_head forward
                                    # entirely (real speed lever — see qcute_refine.py's own session diagnosis).
    byte_ntp_weight: float = 1.0  # scales level 0's own NTP loss (byte_loss). Default 1.0 = original
                                    # behavior (unscaled, always included). ==0.0 isolates the decoder tower's
                                    # own tok_loss as the sole training signal (alongside code_ntp_weight=0.0)
                                    # — an "unconstrained diagnostic" for what the cross-attention decoder path
                                    # alone can reach, with no competing gradient from the encoder-side NTP
                                    # heads. Unlike code_ntp_weight==0.0, does NOT skip the level-0 NTP head's
                                    # forward pass (h/byte_acc are still needed downstream/for logging even
                                    # when byte_loss itself doesn't enter the total loss).
    quant_type: str = "bsq"       # "bsq" (default) or "identity" (ceiling-baseline diagnostic, training-
                                    # only, sound only alongside code_ntp_weight=tok_weight=0.0) — see
                                    # qcute_refine.py's Config docstring for the full rationale, unchanged.
    code_embed_mode: str = "linear"  # how a level's own dq-dim BSQ (or identity-quantized) code gets mapped
                                    # to a D-dim representation wherever it's consumed as a raw code (NOT
                                    # level 0's byte-bits embed, which byte_repr already covers separately).
                                    # "linear" (default/original): single nn.Linear(in_dq, D). "mlp": small
                                    # 2-layer nonlinear map (Linear->GELU->Linear), same shape, strictly more
                                    # expressive, negligible extra cost (in_dq is tiny, e.g. 8). "pq_table":
                                    # treats the code as an index into a 2**in_dq-row nn.Embedding table
                                    # instead of a linear combination of its +-1 components — exact, since
                                    # BSQ's own forward value truly is one of 2**in_dq discrete hypersphere
                                    # corners (see bsq_quantize); only valid with quant_type=="bsq" and
                                    # in_dq<=16 (table row count), asserted in CodeEmbed. Session hypothesis:
                                    # "dq is starved" — a linear map over an 8-dim +-1 vector can only express
                                    # 8 additive directions; an arbitrary function of a genuinely small state
                                    # space (256 codes at dq=8) may need a table or nonlinearity to exploit,
                                    # not a hyperplane. See CodeEmbed's own docstring for the straight-through
                                    # gradient trick pq_table needs to keep training the code's own producer
                                    # (code_pre) — a naive hard lookup would otherwise sever that gradient path
                                    # entirely, since nothing else currently trains code_pre.
    fuse_encoder_levels: bool = True   # v3's own core mechanism (see module docstring). True (default in
                                    # this file): every level below the top gets a PASS 2 (RefineLM.forward)
                                    # that cross-attends to the level-above's own hidden state BEFORE its own
                                    # self.blocks, and THAT run's ntp_loss/h replaces PASS 1's for byte_loss/
                                    # val_bpb/checkpointing purposes — fixes v2's blind spot where byte_loss
                                    # (the ONLY metric ever compared against baselines) had zero access to the
                                    # coarser code, cross-attention, or the decoder tower's own tok_loss
                                    # signal (h reads there are detached too — encoder and decoder were fully
                                    # separate graphs in v2, sharing only forward values). False: reproduces
                                    # v2's forward exactly (PASS 1 only, no fusion) — a genuine A/B control
                                    # for isolating fusion's own effect, same file/weights-shape otherwise.
    byte_repr: str = "bits"       # LEVEL 0 ONLY. "bits" (default/original): byte_to_bits 8-dim
                                    # projection + BitPredictHead chain NTP head (both EncoderLevel_0 and
                                    # DecoderLevel_0). "embed": traditional nn.Embedding(vocab, D) lookup
                                    # + plain nn.Linear(D, vocab) NTP head, 256-way cross-entropy — exactly
                                    # bytelm.py's own convention. Both real, kept options — see
                                    # EncoderLevel's own docstring.
    code_head_mode: str = "chain"  # LEVELS>0 ONLY (encoder side). "chain" (default/original):
                                    # BitPredictHead, exact chain-rule cross-bit conditioning.
                                    # "independent": single plain nn.Linear(D, in_dq), independent
                                    # per-bit logits, no BitPredictHead — every earlier BSQ fork's own
                                    # default before the Fetch-style chain head was introduced.
    # --- DecoderLevel (cross-attention decoder) ---
    tok_d_model: int = 96          # shared cross-attention working width — h^(i)/h^(i+1) (which may have
                                    # different tier_d_models[i]/[i+1]) are each linearly projected into
                                    # this common space before cross-attending.
    tok_n_heads: int = 4
    tok_mlp_mult: int = 4
    tok_head_mode: str = "linear"  # "linear" (default): single plain nn.Linear, independent per-bit
                                    # logits — cheap. "chain": exact chain-rule BitPredictHead instead
                                    # (~200-1800x more expensive per call, per qcute_refine.py's own
                                    # measured diagnosis) — opt-in comparison, not the default.
    tok_weight: float = 1.0        # scales the summed DecoderLevel losses. ==0.0 SKIPS every
                                    # DecoderLevel's forward entirely (not just zero-weights it).
    cross_attn_rope: bool = True   # DEFAULT ON — "decoder must be timestep aware." Applies RoPE to the
                                    # cross-attention Q/K: Q gets its own raw-byte-time position (0..L-1);
                                    # each K slot gets the null slot's own fixed reference position (0) or,
                                    # for a real code block, the raw-byte-time position it becomes fully
                                    # causally resolved at ((b+1)*K[level]-1) — gives the model actual
                                    # relative-distance information instead of just the boolean allowed/
                                    # blocked mask. False restores the original no-positional-info
                                    # cross-attention (real option, not removed).
    decoder_own_trunk: bool = False   # DEFAULT OFF — the original design keeps this off: DecoderLevel
                                    # reuses EncoderLevel[level]/[level+1]'s own already-computed hidden
                                    # states (h_prev/h_curr) as Q/KV sources, zero redundant trunk compute.
                                    # True: DecoderLevel instead builds its OWN separate-weight copies of
                                    # those two trunks (via two private EncoderLevel instances,
                                    # compute_ntp=False, their own emitted codes discarded) and runs raw
                                    # sequences (byte ids / codes) through them itself — the "own trunk"
                                    # design discussed this session, ~+61% params / ~+57% FLOPs on the
                                    # decoder side (session estimate). Mutually exclusive with
                                    # decoder_kv_pass_through — own_trunk takes priority if both are set.
    decoder_kv_pass_through: bool = False   # DEFAULT OFF. True: KV comes directly from a fresh
                                    # Linear(dqs[level], tok_d_model) projection of the level's own raw
                                    # emitted code c_i, instead of EncoderLevel[level+1]'s hidden state —
                                    # decouples DecoderLevel[level] from needing EncoderLevel[level+1]'s
                                    # own trunk to have finished at all (session discussion: "is it
                                    # possible to not reuse h but direct embedding or code proj" — this is
                                    # the KV-side version; trades away level+1's own cross-code-block
                                    # contextualization for removing that sequential dependency). No
                                    # effect if decoder_own_trunk is also True.
    decoder_q_pass_through: bool = False    # DEFAULT OFF. True: Q comes directly from a fresh embedding
                                    # of this level's own raw input seq_repr (nn.Embedding(vocab,
                                    # tok_d_model) if use_byte_softmax, else Linear(in_dq, tok_d_model)) —
                                    # the Q-side counterpart to decoder_kv_pass_through, session
                                    # discussion's other half ("is it possible to not reuse h... direct
                                    # embedding"). Combined with decoder_kv_pass_through=True, this strips
                                    # ALL contextualization from both sides of the cross-attention — pure
                                    # raw-token/raw-code embeddings in, nothing else — deliberately a
                                    # worst-case/floor probe: "see limits of decoder" (how well can
                                    # cross-attention alone do with zero causal self-attention context on
                                    # either side?). No effect if decoder_own_trunk is also True.
    layer_warmup_steps: tuple[int, ...] = ()   # LAYERWISE CURRICULUM (queued ablation, not yet run):
                                    # length must be n_levels-1 (one entry per level-activation gap,
                                    # like `tokenizers` itself) or empty (=all-zeros, i.e. every level
                                    # active from step 0 — the default, backward-compatible behavior).
                                    # layer_warmup_steps[i] = how many steps level i trains ALONE (with
                                    # levels >i entirely absent from the forward pass — not just zero-
                                    # weighted, genuinely not run) before level i+1 turns on. Reason:
                                    # let the lower LM's own BSQ codes become stable before handing them
                                    # to the level above as ITS training target — feeding a still-
                                    # collapsing/shifting code upward immediately trains the upper level
                                    # on a moving target, the same instability class this file's "always-
                                    # on direct NTP loss, own head own target" design otherwise guards
                                    # against within a single level. See RefineLM.n_active_levels/
                                    # activation_steps and train()'s per-stage param groups below.
    vocab: int = 256


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


MAX_PQ_TABLE_DQ = 16   # 2**16 = 65536 rows — the ceiling code_embed_mode=="pq_table" allows


class CodeEmbed(nn.Module):
    """Maps a level's own dq-dim BSQ (or identity-quantized) code to a
    D-dim representation, wherever a raw code is consumed directly (an
    EncoderLevel's own input at level>0, or a DecoderLevel's
    kv_pass_through/q_pass_through raw-code embed at any level — NOT
    level 0's byte-bits embed, a different concern already covered by
    Config.byte_repr). See Config.code_embed_mode for the "linear"/"mlp"/
    "pq_table" mode descriptions and the motivating hypothesis.

    pq_table's forward value is an exact table lookup (the code truly is
    one of 2**in_dq discrete corners when quant_type=="bsq" — see
    bsq_quantize), but a naive lookup is non-differentiable in the code
    itself, which would sever the ONLY gradient path currently training
    the code's own producer (EncoderLevel.code_pre — nothing else reads
    its output). Fixed with the same straight-through idiom bsq_quantize
    itself already uses: a continuous nn.Linear `proxy` of the code
    carries the backward gradient, while the table's actual row is what's
    used forward — `proxy + (hard - proxy).detach()` is exactly `hard` on
    the forward pass and exactly `d(proxy)/d(code)` on the backward pass."""

    def __init__(self, cfg: "Config", in_dq: int, D: int):
        super().__init__()
        self.mode = cfg.code_embed_mode
        self.in_dq = in_dq
        if self.mode == "linear":
            self.proj = nn.Linear(in_dq, D)
        elif self.mode == "mlp":
            self.proj = nn.Sequential(nn.Linear(in_dq, D), nn.GELU(), nn.Linear(D, D))
        elif self.mode == "pq_table":
            assert cfg.quant_type == "bsq", (
                "code_embed_mode='pq_table' requires quant_type='bsq' — 'identity' codes are "
                "continuous, unbounded values with no discrete corners to index."
            )
            assert in_dq <= MAX_PQ_TABLE_DQ, (
                f"code_embed_mode='pq_table': in_dq={in_dq} would need a 2**{in_dq}-row table "
                f"— keep in_dq<={MAX_PQ_TABLE_DQ}."
            )
            self.table = nn.Embedding(2 ** in_dq, D)
            self.proxy = nn.Linear(in_dq, D)
            self.register_buffer("_powers", (2 ** torch.arange(in_dq)).long(), persistent=False)
        else:
            raise ValueError(f"unknown code_embed_mode {self.mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("linear", "mlp"):
            return self.proj(x)
        idx = ((x > 0).long() * self._powers).sum(-1)
        hard = self.table(idx)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_at(pos_id: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.tensor([[float(pos_id)]], device=device) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    """Same as rope_cos_sin, but for an arbitrary (non-contiguous) set of
    position ids rather than a fixed 0..seq_len-1 range — needed for
    DecoderLevel's cross-attention RoPE, where Q lives at raw-byte-time
    resolution (0..L-1) and K lives at code-block resolution (each block
    tagged with the raw-byte-time position it becomes fully causally
    resolved at), so the two sides can't share a single contiguous
    range. position_ids: [T] long. -> (cos, sin), each [T, head_dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def jagged_causal_mask_and_positions(L: int, n_blocks: int, K: int, kv_window: int | None, device: torch.device):
    """v3: factored out of DecoderLevel's own mask/rope-position construction (unchanged math, see
    DecoderLevel.forward's own "JAGGED STAIRCASE" comment for the full derivation) so EncoderLevel's
    new fusion cross-attention (v3's core change — see module docstring) can share the exact same,
    already-verified causal geometry instead of re-deriving it. Both attend from a length-L Q sequence
    at some level's own raw/finer-time resolution to a coarser level's own n_blocks code-block KV
    sequence (plus a prepended, always-visible null slot). Returns (disallow [L, 1+n_blocks] bool,
    True=blocked — nn.MultiheadAttention/CrossBlock convention; k_pos [1+n_blocks] long, each KV
    slot's own raw-time position for cross_attn_rope, null slot pinned to 0)."""
    t_idx = torch.arange(L, device=device).unsqueeze(1)
    b_idx = torch.arange(n_blocks, device=device).unsqueeze(0)
    n_complete = (t_idx + 1) // K
    visible = b_idx < n_complete
    if kv_window is not None:
        visible = visible & (b_idx >= n_complete - kv_window)
    null_col = torch.ones(L, 1, dtype=torch.bool, device=device)
    visible = torch.cat([null_col, visible], dim=1)
    disallow = ~visible
    block_pos = (torch.arange(n_blocks, device=device) + 1) * K - 1
    null_pos = block_pos.new_zeros(1)
    k_pos = torch.cat([null_pos, block_pos])
    return disallow, k_pos


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
        self._warned_dense_fallback = False
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
            if self.window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={self.window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window — falling back to DENSE attention for this layer. "
                      f"Only warns once per layer instance.")
                self._warned_dense_fallback = True
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

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        B, _, D = x_new.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        y = F.scaled_dot_product_attention(q, new_k, new_v, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(B, 1, D)), new_k, new_v


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

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None):
        attn_out, new_k, new_v = self.attn.forward_step(self.ln1(x_new), cos_new, sin_new, cache_k, cache_v)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_k, new_v


class CrossBlock(nn.Module):
    """Single cross-attention transformer block: cross-attn sublayer (Q
    from one sequence, K/V from another) + MLP sublayer, each pre-norm +
    residual — same shape as this file's own causal `Block`, with
    self-attention swapped for cross-attention. RoPE is OPTIONAL (on by
    default via Config.cross_attn_rope — "decoder must be timestep
    aware"): Q and KV live at different granularities/lengths, so they
    can't share one contiguous rotary range the way self-attention does,
    but each side CAN still get its own explicit position tag —
    DecoderLevel computes and passes those in (rope_q for Q's raw-byte-
    time positions, rope_k for each KV slot's own raw-byte-time position:
    the null slot at 0, each real code block at the raw position it
    becomes fully causally resolved). Without this, the only signal the
    cross-attention had for "how far back in real time is this code
    block" was the boolean visibility mask (allowed/blocked, no
    distance) — RoPE gives it the actual relative distance."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        # manual QKV + F.scaled_dot_product_attention instead of nn.MultiheadAttention — see
        # BitPredictHead's own comment for the MPS NaN-gradient finding this avoids.
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor,
                rope_q: tuple[torch.Tensor, torch.Tensor] | None = None,
                rope_k: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """attn_mask: bool [Lq, Lkv], True = BLOCKED (DecoderLevel's own "disallow" convention,
        matching nn.MultiheadAttention's convention) — inverted internally since
        F.scaled_dot_product_attention's boolean convention is the opposite (True = may attend).
        rope_q/rope_k: None (default off path) or (cos, sin) each [T, head_dim] — see class docstring."""
        qn, kvn = self.ln_q(q), self.ln_kv(kv)
        B, Lq, D = qn.shape
        Lkv = kvn.shape[1]
        H, hd = self.n_heads, self.head_dim
        qh = self.q_proj(qn).reshape(B, Lq, H, hd).transpose(1, 2)
        kvp = self.kv_proj(kvn).reshape(B, Lkv, 2, H, hd).permute(2, 0, 3, 1, 4)
        kh, vh = kvp[0], kvp[1]
        if rope_q is not None:
            qh = apply_rope(qh, *rope_q)
        if rope_k is not None:
            kh = apply_rope(kh, *rope_k)
        sdpa_mask = ~attn_mask if attn_mask is not None else None
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=sdpa_mask)
        attn_out = self.out_proj(y.transpose(1, 2).reshape(B, Lq, D))
        q = q + attn_out
        q = q + self.mlp(self.ln2(q))
        return q


def byte_to_bits(byte_ids: torch.Tensor) -> torch.Tensor:
    """[*] long byte ids (0..255) -> [*, 8] float, each bit in {-1,+1}/sqrt(8) — LSB-first, deterministic,
    no learned parameters. A byte losslessly IS its own 8-bit BSQ-shaped code."""
    bits = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    return (2 * bits - 1) / math.sqrt(8)


def bits_to_byte(bits: torch.Tensor) -> torch.Tensor:
    b = (bits > 0).long()
    powers = (2 ** torch.arange(8, device=bits.device))
    return (b * powers).sum(-1)


class BitPredictHeadAttn(nn.Module):
    """Predicts `dq` chained bits from a hidden vector via Fetch-style causal
    self-attention over the bit sequence — the exact chain-rule
    factorization of the joint dq-bit distribution (ported from
    qcutelm_vlt11.py via qcute_refine.py). Used for EncoderLevel's own
    unconditioned NTP head (always), and optionally
    (tok_head_mode="chain") for DecoderLevel's own decode head."""

    def __init__(self, d_model: int, dq: int, n_heads: int = 2, gamma: float = 1.0, fixed_kernel: bool = True, downsample: int = 1):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        assert d_inner % n_heads == 0, f"inner dim={d_inner} (d_model={d_model}/downsample={downsample}) not divisible by n_heads={n_heads}"
        self.dq = dq
        self.gamma = gamma
        self.fixed_kernel = fixed_kernel
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        # downsample>1: work internally at d_inner=d_model//downsample instead of d_model — a cheap
        # in_proj down to d_inner up front, then every chain op (embeds/qkv/out_proj/head) runs at the
        # smaller width. downsample=1 (default): in_proj is None, identical to pre-flag behavior, no
        # extra op/params. Session: "flag to downsample bit predict embeds or inner".
        self.in_proj = nn.Linear(d_model, d_inner) if downsample > 1 else None
        self.head = nn.Linear(d_inner, 1)
        self.bit_pos_emb = nn.Embedding(dq, d_inner)
        self.bit_val_emb = nn.Embedding(2, d_inner)
        # manual QKV + F.scaled_dot_product_attention instead of nn.MultiheadAttention — session found
        # nn.MultiheadAttention's MPS backward produces NaN gradients at d_model=256 (confirmed: identical
        # run stable on CPU, NaN only on MPS, isolated via named_parameters() to exactly this submodule's
        # out_proj.weight.grad) despite being fine at the earlier d_model=96 configs' scale. Every other
        # attention op in this codebase (CausalSelfAttention, CrossBlock) already uses manual SDPA and has
        # been stable all session — this makes BitPredictHead consistent with that, not a new mechanism.
        self.qkv_proj = nn.Linear(d_inner, 3 * d_inner)
        self.out_proj = nn.Linear(d_inner, d_inner)
        causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _mha(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x: [N, T, D]. attn_mask: None, or additive float [T, T] (SDPA accepts the same -inf/0
        convention causal_mask is already built in, no conversion needed). -> [N, T, D]."""
        N, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv_proj(x).reshape(N, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(y.transpose(1, 2).reshape(N, T, D))

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        """h: [N, D]. true_bits: [N, dq] float in {-1,+1}-ish (teacher-forcing) or None (greedy chain
        decode at inference). -> raw_logits [N, dq]."""
        if self.in_proj is not None:
            h = self.in_proj(h)
        if self.fixed_kernel and true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)
        zero_vec = val_embeds.new_zeros(N, 1, D)
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)
        pos = self.bit_pos_emb.weight.unsqueeze(0)
        h_scale = h.new_ones(1, self.dq, 1)
        if self.dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        x = h_scale * h.unsqueeze(1) + shifted + pos
        attn_out = self._mha(x, attn_mask=self.causal_mask)
        fetched = h.unsqueeze(1) + attn_out
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        chain_vecs = [h + self.bit_pos_emb.weight[0]]
        logits_list = []
        for j in range(self.dq):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self._mha(x, attn_mask=None)
            fetched = h + attn_out[:, -1, :]
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                chain_vecs.append(self.gamma * h + self.bit_val_emb(bit_val) + self.bit_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


class BitPredictHeadConv(nn.Module):
    """Same chain-rule job as BitPredictHeadAttn, but a causal 1D
    convolution over the bit-embedding sequence instead of self-attention
    (session: "not use self attention but simply series of linears...").
    kernel_size defaults to `dq` (full receptive field over the whole bit
    chain) — cheap and lossless at this scale since dq stays small in
    every current config (DecoderLevel predicts single tokens, not joint
    MTP blocks), and windowing would impose a locality prior that's
    arguably WRONG for LSB-first bit orderings specifically (bit 0 has no
    particular reason to correlate more with bit 1 than with bit 7, unlike
    real token sequences where nearby positions genuinely correlate more —
    session conclusion, not an assumption).

    Deliberately has NO bit_pos_emb (unlike Attn) — adding an absolute
    position embedding would break translation-invariance, which is
    exactly the property that makes "shared Linear read of a fixed-size
    sliding/causal window" equivalent to a real conv in the first place
    (same weights reused at every relative offset, not indexed by
    absolute position).

    conv_impl selects the implementation of that "shared Linear read of a
    fixed-size window" — both are the SAME operation (same weight-sharing
    property above), just different ops: "conv1d" (original) calls
    nn.Conv1d directly; "matmul" (session: "reparam to use nn.linear
    instead of conv1d") flattens the window into one [K*D] vector and
    applies a plain nn.Linear(K*D, D) instead — mathematically the same
    class of computation (fixed window, weights shared across positions),
    just dispatched as a matmul instead of the conv op. Session
    benchmark (scripts/bench_bit_heads.py) found nn.Conv1d has real
    per-call overhead in the sequential decode loop (worst case ~3900x
    slower than a plain independent nn.Linear head at dq=16, vs.
    "matmul"'s expected much-flatter overhead) — kept BOTH as a flag
    rather than replacing, matching the rest of this file's convention."""

    def __init__(self, d_model: int, dq: int, kernel_size: int | None = None, gamma: float = 1.0, conv_impl: str = "matmul", downsample: int = 1):
        super().__init__()
        assert conv_impl in ("conv1d", "matmul")
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        self.dq = dq
        self.gamma = gamma
        self.conv_impl = conv_impl
        self.kernel_size = kernel_size if kernel_size is not None else dq
        # see BitPredictHeadAttn's own in_proj comment — same downsample flag, same "identity at 1x" property
        self.in_proj = nn.Linear(d_model, d_inner) if downsample > 1 else None
        self.head = nn.Linear(d_inner, 1)
        self.bit_val_emb = nn.Embedding(2, d_inner)
        if conv_impl == "conv1d":
            self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=self.kernel_size, bias=True)
        else:
            self.proj = nn.Linear(self.kernel_size * d_inner, d_inner, bias=True)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _window_read(self, x_windows: torch.Tensor) -> torch.Tensor:
        """x_windows: [N, D, dq, K] (oldest->newest along last dim) ->
        [N, dq, D]. Dispatches on conv_impl; same op either way."""
        if self.conv_impl == "conv1d":
            N, D, dq, K = x_windows.shape
            x_t = x_windows.permute(0, 2, 1, 3).reshape(N * dq, D, K)
            return self.conv(x_t).squeeze(-1).reshape(N, dq, D)
        N, D, dq, K = x_windows.shape
        flat = x_windows.permute(0, 2, 3, 1).reshape(N, dq, K * D)   # [N, dq, K*D]
        return self.proj(flat)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        K = self.kernel_size
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)                          # [N, dq, D]
        zero_vec = val_embeds.new_zeros(N, 1, D)
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)   # position j holds bit j-1's embed
        x_t = shifted.transpose(1, 2)                                    # [N, D, dq]
        x_padded = F.pad(x_t, (K - 1, 0))                                 # causal: pad LEFT only
        x_windows = x_padded.unfold(2, K, 1)                              # [N, D, dq, K], oldest->newest
        conv_out = self._window_read(x_windows)                          # [N, dq, D]
        h_scale = h.new_ones(1, self.dq, 1)
        if self.dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        fetched = h_scale * h.unsqueeze(1) + conv_out
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        past: list[torch.Tensor] = []   # decided bits' embeddings so far, oldest first
        logits_list = []
        for j in range(self.dq):
            window = past[-self.kernel_size:]
            pad_len = self.kernel_size - len(window)
            seq = [h.new_zeros(N, D)] * pad_len + window
            if self.conv_impl == "conv1d":
                x = torch.stack(seq, dim=2)              # [N, D, kernel_size]
                conv_out = self.conv(x).squeeze(-1)      # [N, D]
            else:
                x = torch.cat(seq, dim=-1)                # [N, kernel_size*D], oldest->newest
                conv_out = self.proj(x)                    # [N, D]
            h_scale_j = 1.0 if j == 0 else self.gamma
            fetched = h_scale_j * h + conv_out
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                past.append(self.bit_val_emb(bit_val))
        return torch.stack(logits_list, dim=1)


class BitPredictHeadSSM(nn.Module):
    """Same chain-rule job again, via a linear-decay recurrence instead
    of attention or convolution (session follow-up: "can ssm decay 0" —
    yes, see below). s_j = alpha * s_{j-1} + bit_embed_{j-1}, alpha a
    LEARNED per-channel decay. Unrolled: s_j = sum_{k<j} alpha^(j-1-k) *
    bit_embed_k — a lower-triangular decay-weighted sum, computed as ONE
    batched matmul (no sequential loop) during teacher-forced training —
    this is the parallelization property a nonlinear/gated recurrence
    (a real GRU) would NOT have; forcing the update to stay linear in
    s_{j-1} is exactly what buys it.

    alpha=0 is a fully-supported special case, not a degenerate/undefined
    one: the k=j-1 (immediately-preceding-bit) term always has exponent 0
    (alpha^0=1 regardless of alpha, including alpha=0 — 0**0 is 1 both
    mathematically in this context and in torch's own float pow), while
    every k<j-1 term (exponent>=1) vanishes. So alpha=0 means "condition
    only on the immediately preceding bit" — exactly BitPredictHeadConv's
    own kernel_size=1 case — cleanly, with no division/log anywhere in
    the formula to blow up at that limit. alpha is parametrized via
    sigmoid so it can approach but never algebraically equal exactly 0;
    in practice that's not a meaningful difference (sigmoid(-8) < 1e-3)."""

    def __init__(self, d_model: int, dq: int, d_state: int | None = None, gamma: float = 1.0, downsample: int = 1):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        self.dq = dq
        self.gamma = gamma
        # d_state's own None-default now tracks d_inner, not d_model — downsampling shrinks the default
        # recurrent state width too, consistent with the other two heads' in_proj. Still independently
        # overridable via d_state, same as before.
        self.d_state = d_state if d_state is not None else d_inner
        # see BitPredictHeadAttn's own in_proj comment — same downsample flag, same "identity at 1x" property
        self.in_proj = nn.Linear(d_model, d_inner) if downsample > 1 else None
        self.head = nn.Linear(d_inner, 1)
        self.bit_val_emb = nn.Embedding(2, self.d_state)
        self.state_proj = nn.Linear(self.d_state, d_inner)
        self.decay_logit = nn.Parameter(torch.zeros(self.d_state))   # sigmoid(0)=0.5 init

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)   # [d_state], in (0,1) — see class docstring re: alpha=0

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        dq = self.dq
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)                    # [N, dq, d_state]
        alpha = self._alpha()                                      # [d_state]

        idx = torch.arange(dq, device=h.device)
        offsets = idx.unsqueeze(1) - 1 - idx.unsqueeze(0)           # [dq, dq]: j-1-k
        valid = offsets >= 0                                        # True iff k < j
        offsets_clamped = offsets.clamp(min=0).float()
        decay = (alpha.view(1, 1, -1) ** offsets_clamped.unsqueeze(-1)) * valid.unsqueeze(-1).float()   # [dq,dq,d_state]

        s = torch.einsum("jkc,nkc->njc", decay, val_embeds)         # [N, dq, d_state]
        state_contrib = self.state_proj(s)                          # [N, dq, D]
        h_scale = h.new_ones(1, dq, 1)
        if dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        fetched = h_scale * h.unsqueeze(1) + state_contrib
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        alpha = self._alpha()
        s = h.new_zeros(N, self.d_state)
        logits_list = []
        for j in range(self.dq):
            state_contrib = self.state_proj(s)
            h_scale_j = 1.0 if j == 0 else self.gamma
            fetched = h_scale_j * h + state_contrib
            logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                s = alpha * s + self.bit_val_emb(bit_val)
        return torch.stack(logits_list, dim=1)


def chain_bce_loss(raw_logits: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
    """Sum over the bit dim (nats per predicted unit), then mean over
    everything else — matches qcutelm_vlt11/qcute_refine's own convention."""
    return F.binary_cross_entropy_with_logits(raw_logits, (true_bits > 0).float(), reduction="none").sum(-1).mean()


def build_bit_head(cfg: Config, d_model: int, dq: int) -> nn.Module:
    """Dispatches to whichever BitPredictHead* implementation
    Config.bit_head_class selects — the single place every "chain"-style
    head (byte level's own bits-mode head, and any code_head_mode/
    tok_head_mode=="chain" head) gets built, so switching architectures
    is one flag, not per-call-site edits."""
    if cfg.bit_head_class == "attn":
        return BitPredictHeadAttn(d_model, dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "conv":
        return BitPredictHeadConv(d_model, dq, kernel_size=cfg.bit_conv_kernel_size, gamma=cfg.bit_chain_gamma, conv_impl=cfg.bit_conv_impl, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "ssm":
        return BitPredictHeadSSM(d_model, dq, d_state=cfg.bit_ssm_d_state, gamma=cfg.bit_chain_gamma, downsample=cfg.bit_inner_downsample)
    else:
        raise ValueError(f"unknown bit_head_class {cfg.bit_head_class!r}")


class EncoderLevel(nn.Module):
    """One level of the recursive NTP tower: embeds its own input
    sequence, runs a small causal transformer, always-on direct NTP loss
    on its own next input element (own head, own target), and every K-th
    position BSQ-quantizes into this level's own emitted code.

    Two independent flags, both kept as real options (not one replacing
    the other — "do not remove that, put as flag"):

    Config.byte_repr (level 0 only): "bits" (original/default) — the
    byte_to_bits 8-dim projection + BitPredictHead chain NTP head, same
    as qcute_refine.py. "embed": TRADITIONAL representation instead — a
    real nn.Embedding(vocab, D) lookup table and a plain nn.Linear(D,
    vocab) NTP head, trained with ordinary 256-way cross-entropy, exactly
    bytelm.py's own convention (session: "byte level use traditional
    embedding table and 256-way softmax").

    Config.code_head_mode (level>0 only): "chain" (original/default) —
    BitPredictHead, exact chain-rule cross-bit conditioning. "independent"
    — a single plain nn.Linear(D, in_dq), INDEPENDENT per-bit logits, no
    BitPredictHead at all (session: "code level use independent linear
    heads like original bsq" — every earlier BSQ fork's own default
    before the Fetch-style chain head was introduced). Both still use the
    same per-bit BCE loss (`chain_bce_loss` — the name is legacy; the
    function itself is just "sum BCE over the bit dim, mean over the
    rest," agnostic to whether the logits came from a chained or
    independent head)."""

    def __init__(self, cfg: Config, level: int, in_dq: int, window: int | None,
                 fuse_d_model: int | None = None, fuse_kv_window: int | None = None):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.cfg = cfg
        self.is_byte_level = level == 0
        D = cfg.tier_d_models[level]
        if self.is_byte_level:
            assert cfg.byte_repr in ("bits", "embed")
            if cfg.byte_repr == "embed":
                self.byte_embed = nn.Embedding(cfg.vocab, D)
                self.ntp_head = nn.Linear(D, cfg.vocab)
            else:
                self.embed = nn.Linear(in_dq, D)
                self.ntp_head = build_bit_head(cfg, D, in_dq)
        else:
            assert cfg.code_head_mode in ("chain", "independent")
            self.embed = CodeEmbed(cfg, in_dq, D)
            if cfg.code_head_mode == "independent":
                self.ntp_head = nn.Linear(D, in_dq)
            else:
                self.ntp_head = build_bit_head(cfg, D, in_dq)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, window) for _ in range(cfg.tier_n_layers[level])])
        self.ln_f = nn.LayerNorm(D)
        self.code_pre = nn.Linear(D, cfg.dqs[level])

        # v3 FUSION (module docstring has the full rationale): if this level has a level ABOVE it
        # (fuse_d_model is not None, i.e. level < n_levels-1), it can optionally cross-attend to that
        # level's own hidden state BEFORE running its own self.blocks — the mechanism that lets
        # byte_loss (and any other level's own NTP loss) actually depend on, and be shaped by, the
        # coarser code, unlike v2 where cross-attention only ever touched the separate, detached
        # DecoderLevel/tok_loss path. Q = x directly (already D-dim, this level's own embedding —
        # "encoder reads this directly, no separate embed" per the session ask), no extra q_proj
        # needed. KV = the level-above's hidden state, projected D_{level+1} -> D via fuse_kv_proj
        # (only place a new "own" weight is learned, unavoidable when D's differ). Same jagged
        # causal mask as DecoderLevel (see jagged_causal_mask_and_positions) — position t may only
        # see level-above blocks that completed strictly before t, so no label leakage.
        self.fuse_d_model = fuse_d_model
        self.fuse_kv_window = fuse_kv_window
        if fuse_d_model is not None:
            self.fuse_cross = CrossBlock(D, cfg.n_heads, cfg.mlp_mult)
            self.fuse_kv_proj = nn.Linear(fuse_d_model, D)
            self.fuse_null_kv = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.normal_(self.fuse_null_kv, std=0.02)

    def _fuse(self, x: torch.Tensor, fuse_kv: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] this level's own pre-self-attention embedding. fuse_kv: [B, n_blocks, D_above]
        the level-above's own hidden state (already, by construction, "as of" the raw-time position
        each block completes — see jagged_causal_mask_and_positions). Returns x, fused."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        B, L, D = x.shape
        n_blocks = fuse_kv.size(1)
        device = x.device
        kv = self.fuse_kv_proj(fuse_kv)
        null = self.fuse_null_kv.expand(B, 1, D)
        kv = torch.cat([null, kv], dim=1)
        disallow, k_pos = jagged_causal_mask_and_positions(L, n_blocks, K, self.fuse_kv_window, device)
        rope_q = rope_k = None
        if cfg.cross_attn_rope:
            head_dim = D // cfg.n_heads
            q_pos = torch.arange(L, device=device)
            rope_q = rope_cos_sin_for_positions(q_pos, head_dim, cfg.rope_base, device)
            rope_k = rope_cos_sin_for_positions(k_pos, head_dim, cfg.rope_base, device)
        return self.fuse_cross(x, kv, attn_mask=disallow, rope_q=rope_q, rope_k=rope_k)

    def forward(self, seq_repr: torch.Tensor, compute_ntp: bool = True,
                fuse_kv: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """seq_repr: level 0 gets raw byte ids [B, L] (long) regardless of
        byte_repr — this class converts internally via byte_to_bits when
        byte_repr=="bits", so RefineLM.forward stays mode-agnostic. level
        i>0 gets its own continuous code [B, L, in_dq] (float).
        compute_ntp=False SKIPS the ntp_head call entirely (real speed
        lever — level 0 must never pass False). fuse_kv: v3 ONLY — None
        (default) reproduces v2's forward exactly (this is what
        RefineLM.forward's own PASS 1 uses, bottom-up, to produce every
        level's code); the level-above's own hidden state (PASS 2, see
        RefineLM.forward) fuses it in via cross-attention before
        self.blocks runs, so THIS call's ntp_loss/h genuinely depend on
        it. Returns (c_i [B, n_blocks, dqs[level]], ntp_loss, ntp_acc,
        h [B, L, D])."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        D = cfg.tier_d_models[self.level]

        if self.is_byte_level and cfg.byte_repr == "embed":
            x = self.byte_embed(seq_repr)
            B, L = seq_repr.shape
        elif self.is_byte_level:
            bits = byte_to_bits(seq_repr)
            x = self.embed(bits)
            B, L, _ = bits.shape
        else:
            x = self.embed(seq_repr)
            B, L, _ = seq_repr.shape
        n_blocks = L // K

        if fuse_kv is not None:
            assert self.fuse_d_model is not None, "fuse_kv passed but this EncoderLevel has no fuse module (no level above it?)"
            x = self._fuse(x, fuse_kv)

        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        h = self.ln_f(x)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            if self.is_byte_level and cfg.byte_repr == "embed":
                target = seq_repr[:, 1:].reshape(-1)
                logits = self.ntp_head(h_flat)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif self.is_byte_level:
                true_flat = byte_to_bits(seq_repr[:, 1:]).reshape(-1, self.in_dq)
                raw = self.ntp_head(h_flat, true_flat)
                ntp_loss = chain_bce_loss(raw, true_flat)
                with torch.no_grad():
                    ntp_acc = ((raw > 0) == (true_flat > 0)).float().mean()
            else:
                true_flat = seq_repr[:, 1:, :].reshape(-1, self.in_dq)
                raw = self.ntp_head(h_flat, true_flat) if cfg.code_head_mode == "chain" else self.ntp_head(h_flat)
                ntp_loss = chain_bce_loss(raw, true_flat)
                with torch.no_grad():
                    ntp_acc = ((raw > 0) == (true_flat > 0)).float().mean()
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        h_blocks = h.view(B, n_blocks, K, D)
        pre_q = self.code_pre(h_blocks[:, :, K - 1, :])
        if cfg.quant_type == "bsq":
            c_i = bsq_quantize(pre_q, cfg.dqs[self.level])
        elif cfg.quant_type == "identity":
            c_i = pre_q
        else:
            raise ValueError(f"unknown quant_type {cfg.quant_type!r}")
        return c_i, ntp_loss, ntp_acc, h


class DecoderLevel(nn.Module):
    """Decodes EncoderLevel[level]'s own input x^(level) by cross-
    attending from EncoderLevel[level]'s own hidden states (Q, "previous
    level['s] code LM") to EncoderLevel[level+1]'s own hidden states (KV,
    "current level['s] code LM" — EncoderLevel[level+1]'s OWN INPUT is
    exactly c^(level), so its hidden state IS the thing to attend to; no
    separate trunk is built here at all). See module docstring for the
    full causal-mask/null-KV/detach rationale."""

    def __init__(self, cfg: Config, level: int, in_dq: int, kv_window: int | None = None):
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.K = cfg.Ks[level]
        self.cfg = cfg
        # KV visibility capped to the most recent `kv_window` completed blocks, matching
        # EncoderLevel[level+1]'s OWN attn_window (same units: level+1's own block/position scale) — before
        # this, the cross-attention mask only enforced causality (block b visible once complete) with no
        # cap on how far BACK it could reach, unbounded regardless of what window the encoder itself used.
        # None (attn_window[level+1] == -1, dense) preserves that original unbounded reach.
        self.kv_window = kv_window
        # only level 0 with byte_repr=="embed" gets the special vocab-softmax path; level 0 with
        # byte_repr=="bits" behaves exactly like any other level (in_dq=8, tok_head_mode picks the head) —
        # matches EncoderLevel's own byte_repr flag, "do not remove that, put as flag"
        self.use_byte_softmax = level == 0 and cfg.byte_repr == "embed"
        self.own_trunk = cfg.decoder_own_trunk
        self.kv_pass_through = cfg.decoder_kv_pass_through and not self.own_trunk
        self.q_pass_through = cfg.decoder_q_pass_through and not self.own_trunk
        D = cfg.tok_d_model

        if self.own_trunk:
            # own, separate-weight copies of EncoderLevel[level]/[level+1]'s own trunk shape — dense
            # attention (window=None) for simplicity, not necessarily matching the encoder's own window
            # choice; compute_ntp=False always (own ntp_head/code_pre exist but are never used/trained —
            # unused byproducts of reusing the EncoderLevel class directly, same "acceptable minor waste"
            # pattern as other unused-byproduct cases in this file).
            self.own_main = EncoderLevel(cfg, level, in_dq, window=None)
            self.own_side = EncoderLevel(cfg, level + 1, cfg.dqs[level], window=None)
            self.q_proj = nn.Linear(cfg.tier_d_models[level], D)
            self.kv_proj = nn.Linear(cfg.tier_d_models[level + 1], D)
        else:
            if self.q_pass_through:
                # direct embedding straight to tok_d_model width, no h_prev/trunk involved at all.
                # level 0's in_dq is fixed byte bits (byte_repr=="bits"), not a BSQ code —
                # code_embed_mode doesn't apply there, same exclusion as EncoderLevel's own byte-level embed.
                if self.use_byte_softmax:
                    self.q_embed = nn.Embedding(cfg.vocab, D)
                elif self.level == 0:
                    self.q_embed = nn.Linear(in_dq, D)
                else:
                    self.q_embed = CodeEmbed(cfg, in_dq, D)
            else:
                self.q_proj = nn.Linear(cfg.tier_d_models[level], D)
            if self.kv_pass_through:
                # kv_input here is always this level's own EMITTED code c^(level) (a genuine BSQ/
                # identity-quantized code regardless of self.level, including level 0) — code_embed_mode
                # always applies.
                self.code_proj = CodeEmbed(cfg, cfg.dqs[level], D)
            else:
                self.kv_proj = nn.Linear(cfg.tier_d_models[level + 1], D)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.null_kv, std=0.02)
        self.cross_block = CrossBlock(D, cfg.tok_n_heads, cfg.tok_mlp_mult)
        if self.use_byte_softmax:
            self.head = nn.Linear(D, cfg.vocab)
        elif cfg.tok_head_mode == "linear":
            self.head = nn.Linear(D, in_dq)
        elif cfg.tok_head_mode == "chain":
            self.head = build_bit_head(cfg, D, in_dq)
        else:
            raise ValueError(f"unknown tok_head_mode {cfg.tok_head_mode!r}")

    def forward(self, main_input: torch.Tensor, kv_input: torch.Tensor, seq_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """main_input/kv_input meaning depends on this decoder's own mode
        (chosen once at construction from cfg, RefineLM.forward supplies
        the matching arguments — see its own dispatch):
          - default (reuse): main_input=h_prev, kv_input=h_curr — both
            EncoderLevel's own already-computed (detached) hidden states.
          - own_trunk: main_input=raw seq_repr for this level, kv_input=
            raw code c_i — this decoder runs its OWN private encoder
            copies over them to produce h_prev/h_curr itself.
          - q_pass_through: main_input=raw seq_repr for this level
            (embedded directly, no h_prev/trunk involved at all).
          - kv_pass_through: main_input=h_prev (still reused unless
            q_pass_through is ALSO set), kv_input=raw code c_i (projected
            directly, no trunk at all on the KV side).
        seq_repr: level `level`'s own true input (decode target) — raw
        byte ids [B, L] (long) at level 0, else its own continuous code
        [B, L, in_dq] (float). Returns (loss, acc)."""
        cfg = self.cfg
        K = self.K
        D = cfg.tok_d_model

        if self.own_trunk:
            _, _, _, h_prev = self.own_main(main_input, compute_ntp=False)
            _, _, _, h_curr = self.own_side(kv_input, compute_ntp=False)
            q = self.q_proj(h_prev)
            kv = self.kv_proj(h_curr)
        else:
            if self.q_pass_through:
                # main_input is raw seq_repr here — level 0 + byte_repr=="bits" still arrives as raw byte
                # ids (long), same convention EncoderLevel itself uses, so convert the same way it does.
                if self.use_byte_softmax:
                    q = self.q_embed(main_input)
                else:
                    q_in = byte_to_bits(main_input) if self.level == 0 else main_input
                    q = self.q_embed(q_in)
            else:
                q = self.q_proj(main_input)   # main_input = h_prev (reuse mode)
            kv = self.code_proj(kv_input) if self.kv_pass_through else self.kv_proj(kv_input)

        B, L, _ = q.shape
        n_blocks = kv_input.size(1)   # same regardless of mode: h_curr's own length == c_i's own length
        device = q.device

        null = self.null_kv.expand(B, 1, D)
        kv = torch.cat([null, kv], dim=1)   # [B, 1+n_blocks, D] — null slot always visible

        # JAGGED STAIRCASE mask (session-verified, K=4 worked example): visible block count steps +1
        # every K raw positions, but the step boundary lands at t=(b+1)*K-1, NOT at a K-multiple —
        # block b's code depends on EncoderLevel's hidden state at that block's LAST position
        # (h_blocks[:, :, K-1, :] in EncoderLevel.forward), so it isn't causally available until the
        # byte at that position has been seen. Concretely for K=4 (decoder position t predicts byte
        # t+1, via the h_dec[:, :-1] vs seq_repr[:, 1:] shift below): position t=2 (predicts byte 3)
        # correctly sees NO blocks — block 0 depends on byte 3 itself, the very thing being
        # predicted, so exposing it would leak the label. Position t=3 (predicts byte 4, unrelated
        # to block 0) correctly sees block 0 — fully determined by already-seen bytes 0-3, no
        # leakage. Verified empirically: visible blocks per t for K=4, n_blocks=4 are
        # t=0,1,2->[] t=3..6->[0] t=7..10->[0,1] t=11..14->[0,1,2] t=15->[0,1,2,3] — a step of +1
        # every K positions, offset by K-1, matching exactly when EncoderLevel actually computes
        # each block's code (and matching cross_attn_rope's own block_pos=(b+1)*K-1, so the position
        # encoding and the visibility mask agree on "when" each block conceptually exists). v3: this
        # geometry is now factored into jagged_causal_mask_and_positions (module-level function) so
        # EncoderLevel's new fusion cross-attention can share it — see that function's own docstring.
        disallow, k_pos = jagged_causal_mask_and_positions(L, n_blocks, K, self.kv_window, device)

        rope_q = rope_k = None
        if cfg.cross_attn_rope:
            head_dim = D // cfg.tok_n_heads
            q_pos = torch.arange(L, device=device)                            # Q's own raw-byte-time positions
            rope_q = rope_cos_sin_for_positions(q_pos, head_dim, cfg.rope_base, device)
            rope_k = rope_cos_sin_for_positions(k_pos, head_dim, cfg.rope_base, device)

        h_dec = self.cross_block(q, kv, attn_mask=disallow, rope_q=rope_q, rope_k=rope_k)   # [B, L, D]

        h_flat = h_dec[:, :-1, :].reshape(-1, D)
        if self.use_byte_softmax:
            target = seq_repr[:, 1:].reshape(-1)
            logits = self.head(h_flat)
            loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
        else:
            true_seq = byte_to_bits(seq_repr) if self.level == 0 else seq_repr   # level 0 + byte_repr=="bits"
            true_flat = true_seq[:, 1:, :].reshape(-1, self.in_dq)
            if cfg.tok_head_mode == "chain":
                raw = self.head(h_flat, true_flat)
            else:
                raw = self.head(h_flat)
            loss = chain_bce_loss(raw, true_flat)
            with torch.no_grad():
                acc = ((raw > 0) == (true_flat > 0)).float().mean()
        return loss, acc


class RefineLM(nn.Module):
    """N-level recursive NTP tower (N EncoderLevels) + N-1 DecoderLevels,
    one per adjacent level pair, each a cross-attention decoder reusing
    the tower's own already-computed hidden states — see module docstring."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        assert len(cfg.tier_n_layers) == self.n_levels
        assert cfg.tok_d_model % cfg.tok_n_heads == 0

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens
        self.code_seq_lens = [seq_lens[i] // cfg.Ks[i] for i in range(self.n_levels)]

        for i, d in enumerate(cfg.tier_d_models):
            assert d % cfg.n_heads == 0, f"tier_d_models[{i}] ({d}) must be divisible by n_heads ({cfg.n_heads})"

        # resolve attn_window into one value per level — single int broadcasts (backward compatible),
        # a tuple must have length n_levels (lets e.g. the top level get its own, smaller, genuinely-
        # sub-full window instead of always inheriting whatever the finer levels use — see Config's
        # own docstring for the "coincidental dense fallback" problem this fixes).
        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels, f"attn_window tuple must have length n_levels={self.n_levels}, got {len(raw_windows)}"
        windows = [None if w == -1 else w for w in raw_windows]
        self.windows = windows
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] ({window}) must divide level {i}'s sequence length ({L}), or be >= it"

        in_dqs = [8] + list(cfg.dqs[:-1])
        self.in_dqs = in_dqs
        # v3: every level below the top gets fuse_d_model/fuse_kv_window set (the level-above's own
        # D/kv_window, same values DecoderLevel already receives for the same pair) — top level gets
        # None (nothing coarser exists to fuse from).
        self.encoders = nn.ModuleList([
            EncoderLevel(
                cfg, i, in_dqs[i], windows[i],
                fuse_d_model=cfg.tier_d_models[i + 1] if i < self.n_levels - 1 else None,
                fuse_kv_window=windows[i + 1] if i < self.n_levels - 1 else None,
            )
            for i in range(self.n_levels)
        ])
        self.decoders = nn.ModuleList([DecoderLevel(cfg, i, in_dqs[i], kv_window=windows[i + 1]) for i in range(self.n_levels - 1)])

        lw = cfg.layer_warmup_steps if cfg.layer_warmup_steps else (0,) * (self.n_levels - 1)
        assert len(lw) == self.n_levels - 1, (
            f"layer_warmup_steps must have length n_levels-1={self.n_levels - 1} (or be empty for "
            f"'no curriculum'), got {len(lw)}: {lw}"
        )
        self.layer_warmup_steps = lw
        activation_steps = [0]
        for w in lw:
            activation_steps.append(activation_steps[-1] + w)
        self.activation_steps = activation_steps   # length n_levels; activation_steps[0] == 0 always

    def n_active_levels(self, step: int | None) -> int:
        """step=None (eval's default, and every call site before this
        feature existed): all levels active, matching prior behavior
        exactly. Otherwise: level 0 is always active; level i (i>=1)
        becomes active once `step >= self.activation_steps[i]` — the
        layerwise curriculum (see Config.layer_warmup_steps)."""
        if step is None or not any(self.layer_warmup_steps):
            return self.n_levels
        n = 1
        for i in range(1, self.n_levels):
            if step >= self.activation_steps[i]:
                n += 1
            else:
                break
        return n

    def forward(self, byte_ids: torch.Tensor, n_active: int | None = None) -> tuple[torch.Tensor, dict]:
        """n_active: precomputed by the CALLER (via self.n_active_levels(step), in plain eager
        Python, never inside a torch.compile'd region) rather than taking raw `step` here directly.
        This is what lets --compile coexist with Config.layer_warmup_steps: dynamo would guard on
        `step`'s exact value if it reached this function (recompiling almost every training step,
        since step changes every call) — but n_active only takes a handful of distinct values over
        an entire run (one per curriculum stage), so guarding on IT instead means at most
        n_levels-1 recompiles total, each one a genuinely necessary graph change (a new level
        activating really is a different compute graph), not a per-step cost. None (default,
        matching every call site before Config.layer_warmup_steps existed) = all levels active."""
        cfg = self.cfg
        if n_active is None:
            n_active = self.n_levels
        seq_repr = byte_ids   # level 0's own input is now raw byte ids (long) — traditional embedding
                                # table, not the bit-vector projection
        ntp_losses, ntp_accs = [], []
        h_list, x_list = [], []
        byte_loss = byte_acc = None

        for i in range(n_active):
            compute_ntp = i == 0 or cfg.code_ntp_weight > 0
            c_i, ntp_loss, ntp_acc, h_i = self.encoders[i](seq_repr, compute_ntp=compute_ntp)
            ntp_losses.append(ntp_loss)
            ntp_accs.append(ntp_acc)
            if i == 0:
                byte_loss, byte_acc = ntp_loss, ntp_acc
            h_list.append(h_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        # v3 PASS 2 (module docstring has the full rationale): PASS 1 above is the same bottom-up
        # sweep as v2, unchanged — needed regardless, since level i+1 structurally can't exist before
        # level i has produced its own code c_i (level i+1's own input IS c_i), so fusing "before the
        # encoder" is circular for a level fusing FROM ITSELF in one pass. Once PASS 1 has finished,
        # every level's own hidden state DOES exist, so levels 0..n_active-2 can be re-run — same
        # weights, same input x_list[i] — this time cross-attending to h_list[i+1] BEFORE their own
        # self.blocks (EncoderLevel._fuse). This refined ntp_loss/acc REPLACES PASS 1's for exactly
        # those levels; the TOP active level (nothing above it) keeps its PASS 1 result unchanged.
        # h_list[i+1] is DETACHED here — same "don't reshape the other level's own hidden state"
        # principle v2's DecoderLevel already follows for its own cross-attention reads (see its own
        # comment) — so this only lets byte_loss/level i's own NTP loss learn to USE the coarser
        # code (real gradient into level i's OWN fuse_cross/embed/blocks), without also making level
        # i's own loss reshape level i+1's weights, which would be a second, competing objective over
        # the same hidden state (the exact "moving target" failure mode this file's own encoder-side
        # NTP design otherwise avoids). c_i (what feeds level i+1) is NOT recomputed here — always
        # PASS 1's, so there's no infinite regress up the tower.
        if cfg.fuse_encoder_levels:
            for i in range(n_active - 1):
                c_i2, ntp_loss2, ntp_acc2, h_i2 = self.encoders[i](
                    x_list[i], compute_ntp=True, fuse_kv=h_list[i + 1].detach()
                )
                ntp_losses[i] = ntp_loss2
                ntp_accs[i] = ntp_acc2
                h_list[i] = h_i2   # PASS 2's h is strictly the better representation — decoder consumes it too
                if i == 0:
                    byte_loss, byte_acc = ntp_loss2, ntp_acc2

        compute_tok = cfg.tok_weight > 0
        tok_losses, tok_accs = [], []
        for i in range(n_active - 1):
            if compute_tok:
                decoder = self.decoders[i]
                # x_list[i+1] == c_i (level i's own emitted code — captured as x_list's NEXT entry by
                # construction above), always detached before crossing into a decoder — same "decoder loss
                # must not reshape encoder" principle the default h_list[...].detach() already follows.
                need_raw_main = decoder.own_trunk or decoder.q_pass_through
                need_raw_kv = decoder.own_trunk or decoder.kv_pass_through
                main_arg = x_list[i].detach() if need_raw_main else h_list[i].detach()
                kv_arg = x_list[i + 1].detach() if need_raw_kv else h_list[i + 1].detach()
                tl, ta = decoder(main_arg, kv_arg, x_list[i])
            else:
                tl, ta = h_list[i].new_zeros(()), h_list[i].new_zeros(())
            tok_losses.append(tl)
            tok_accs.append(ta)

        ntp_total = torch.stack(ntp_losses).sum()
        tok_total = torch.stack(tok_losses).sum() if tok_losses else byte_loss.new_zeros(())
        code_ntp_total = torch.stack(ntp_losses[1:]).sum() if len(ntp_losses) > 1 else byte_loss.new_zeros(())
        loss = cfg.byte_ntp_weight * byte_loss + cfg.code_ntp_weight * code_ntp_total + cfg.tok_weight * tok_total
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "ntp_loss_total": ntp_total, "tok_loss_total": tok_total,
            "n_active_levels": byte_loss.new_tensor(float(n_active)),   # tensor, not plain int — every
                                                                          # metrics value gets .item()'d
                                                                          # downstream (eval_model/train)
            **{f"level{i}_ntp_loss": l for i, l in enumerate(ntp_losses)},
            **{f"level{i}_ntp_acc": a for i, a in enumerate(ntp_accs)},
            **{f"pair{i}_tok_loss": l for i, l in enumerate(tok_losses)},
            **{f"pair{i}_tok_acc": a for i, a in enumerate(tok_accs)},
        }
        return loss, metrics


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor) -> torch.Tensor:
    if model.cfg.byte_repr == "embed":
        logits = model.encoders[0].ntp_head(h_last)   # plain 256-way Linear — greedy argmax
        return logits.argmax(-1)
    logits = model.encoders[0].ntp_head(h_last, true_bits=None)   # chain mode greedy-decodes internally
    return bits_to_byte(logits)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Reference (slow, obviously-correct) byte-by-byte generation:
    recomputes EncoderLevel_0 from scratch over the WHOLE sequence every
    new byte. Only EncoderLevel_0's own NTP head is generative (see module
    docstring) — DecoderLevel never participates."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encoders[0]
    cfg = model.cfg
    D = cfg.tier_d_models[0]

    for _ in range(n_new_bytes):
        L = all_bytes.size(1)
        x = enc0.byte_embed(all_bytes) if cfg.byte_repr == "embed" else enc0.embed(byte_to_bits(all_bytes))
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in enc0.blocks:
            x = block(x, cos, sin)
        h = enc0.ln_f(x)
        next_byte = _sample_next_byte(model, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """KV-cache-efficient generation, identical mechanism to qcute_refine.py's
    own generate_kv_cache — dense-attention only for level 0 specifically
    (the only level generation ever touches; other levels' own windows
    don't matter here)."""
    cfg = model.cfg
    assert model.windows[0] is None, "generate_kv_cache only supports dense attention at level 0 (attn_window[0] must be -1) — see docstring"
    enc0 = model.encoders[0]
    D = cfg.tier_d_models[0]
    n_layers = len(enc0.blocks)
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)

    cache_k: list[torch.Tensor | None] = [None] * n_layers
    cache_v: list[torch.Tensor | None] = [None] * n_layers

    def step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = (enc0.byte_embed(byte_id) if cfg.byte_repr == "embed" else enc0.embed(byte_to_bits(byte_id))).unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(enc0.blocks):
            x, cache_k[li], cache_v[li] = block.forward_step(x, cos_new, sin_new, cache_k[li], cache_v[li])
        return enc0.ln_f(x).squeeze(1)

    L0 = prompt_bytes.size(1)
    last_h = None
    for pos in range(L0):
        last_h = step(prompt_bytes[:, pos], pos)

    out_bytes = [prompt_bytes]
    for i in range(n_new_bytes):
        next_byte = _sample_next_byte(model, last_h)
        out_bytes.append(next_byte.unsqueeze(1))
        last_h = step(next_byte, L0 + i)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


def validate_generation(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


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
def eval_model(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str, step: int | None = None) -> dict:
    model.eval()
    # n_active computed HERE, once, in plain eager Python — never inside the (possibly compiled)
    # model call itself. See RefineLM.forward's own docstring for why this matters for --compile.
    n_active = model.n_active_levels(step)
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx, n_active=n_active)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    return result


def build_param_groups(model: RefineLM) -> list[dict]:
    """One param group per activation STAGE (stage 0 = encoders[0] alone;
    stage i>=1 = encoders[i] + tokenizers[i-1], the pair that turns on
    together once level i activates — see Config.layer_warmup_steps) —
    lets train() give each stage its own reset warmup schedule. With no
    curriculum (layer_warmup_steps empty), every stage activates at step 0
    and this is behaviorally identical to one global param group."""
    groups = [{"params": list(model.encoders[0].parameters()), "stage": 0}]
    for i in range(1, model.n_levels):
        params = list(model.encoders[i].parameters()) + list(model.decoders[i - 1].parameters())
        groups.append({"params": params, "stage": i})
    return groups


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v3", dynamic_ncols=True)
    for step in pbar:
        # each stage gets its OWN warmup, reset to 0 at that stage's own activation_steps[stage] — the
        # SAME shared lr_at/lr_at_warmup_constant_cosine functions every other run uses, just called with
        # a per-stage relative step instead of the global one. Stages not yet active get lr=0 (their
        # params also aren't in this step's forward graph at all, so grad is None -> AdamW skips them
        # regardless; lr=0 here is belt-and-suspenders, not load-bearing).
        for g in opt.param_groups:
            rel_step = step - model.activation_steps[g["stage"]]
            if rel_step < 0:
                lr = 0.0
            elif args.cosine_decay:
                lr = lr_at_warmup_constant_cosine(rel_step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
            else:
                lr = lr_at(rel_step, args.warmup_steps, args.lr_peak)
            g["lr"] = lr
        lr = opt.param_groups[0]["lr"]   # stage-0 lr, for logging/postfix — always active, always representative

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        # n_active computed HERE, once per step, in plain eager Python — never inside the (possibly
        # compiled) model call itself. See RefineLM.forward's own docstring: this is what lets
        # --compile coexist with Config.layer_warmup_steps — dynamo guards on n_active's value
        # (stable for long stretches, changes only at curriculum stage boundaries) instead of
        # step's (changes every single call, which is what made whole-model compile recompile
        # almost every step before this).
        n_active = model.n_active_levels(step)
        loss, metrics = model(ctx, n_active=n_active)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
            tok_loss=f"{metrics['tok_loss_total'].item():.4f}",
        )

        if step % args.log_every == 0:
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(), bpb=train_bpb, byte_acc=metrics["byte_acc"].item())

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device, step=step)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            log(f"{pbar}  {val_str}", step=step, **{f"val_{k}": v for k, v in val.items()})
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def _broadcast_int_tuple(s, n: int) -> tuple[int, ...]:
    t = _parse_int_tuple(s)
    if len(t) == 1:
        return t * n
    assert len(t) == n, f"expected 1 (broadcast) or {n} values, got {len(t)}: {t}"
    return t


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Recursive NTP tower + cross-attention DecoderLevel + EncoderLevel fusion (qcute_refine_v3)", parents=[pre])
    p.add_argument("--dqs", type=_parse_int_tuple, default=(8, 8, 8))
    p.add_argument("--Ks", default=(2, 2, 2))
    p.add_argument("--tier_n_layers", default=(1, 1, 1))
    p.add_argument("--tier_d_models", type=_parse_int_tuple, default=(96, 96, 96))
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=128)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--bit_head_class", type=str, default="attn", choices=["attn", "conv", "ssm"])
    p.add_argument("--bit_conv_kernel_size", type=int, default=None)
    p.add_argument("--bit_conv_impl", type=str, default="matmul", choices=["conv1d", "matmul"])
    p.add_argument("--bit_inner_downsample", type=int, default=1)
    p.add_argument("--bit_ssm_d_state", type=int, default=None)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "identity"])
    p.add_argument("--code_embed_mode", type=str, default="linear", choices=["linear", "mlp", "pq_table"])
    p.add_argument("--fuse_encoder_levels", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--byte_repr", type=str, default="bits", choices=["bits", "embed"])
    p.add_argument("--code_head_mode", type=str, default="chain", choices=["chain", "independent"])
    p.add_argument("--tok_d_model", type=int, default=96)
    p.add_argument("--tok_n_heads", type=int, default=4)
    p.add_argument("--tok_mlp_mult", type=int, default=4)
    p.add_argument("--tok_head_mode", type=str, default="linear", choices=["linear", "chain"])
    p.add_argument("--tok_weight", type=float, default=1.0)
    p.add_argument("--cross_attn_rope", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--decoder_own_trunk", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decoder_kv_pass_through", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decoder_q_pass_through", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--layer_warmup_steps", type=lambda s: () if s in ("", "()") else _parse_int_tuple(s), default=())

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
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--compile", type=lambda x: x.lower() != "false", default=False,
                    help="torch.compile() the WHOLE RefineLM (true single-graph compile, not a "
                         "per-submodule loop). Default False. Works fine WITH Config.layer_warmup_steps "
                         "now (no assert/restriction) — the original version of this flag made "
                         "RefineLM.forward take the raw training-step int directly and branch on its "
                         "exact value inside n_active_levels(); dynamo guards on that exact int and "
                         "recompiled the whole graph almost every step (measured: net SLOWER than "
                         "eager, confirmed via TORCH_LOGS=recompiles), EVEN for configs that never set "
                         "layer_warmup_steps at all. Fixed properly (not worked around) by moving the "
                         "step->n_active computation OUT of the compiled call entirely: "
                         "train()/eval_model() now call self.n_active_levels(step) themselves, in "
                         "plain eager Python, and pass the resulting n_active int into "
                         "RefineLM.forward(byte_ids, n_active=...) instead of raw step — dynamo then "
                         "guards on n_active, which only takes a handful of distinct values across an "
                         "entire run (one per curriculum stage transition) instead of one per step, so "
                         "at most n_levels-1 recompiles happen total, each one a genuinely necessary "
                         "graph change (a new level activating really is a different compute graph). "
                         "Verified via TORCH_LOGS=recompiles on a real config: 1 total recompile-related "
                         "log line across 40 real training steps (was ~1 per step before the fix). A "
                         "training-script-level runtime flag, not an architecture choice, so it's plain "
                         "CLI/args, not a Config field.")

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.dqs = _parse_int_tuple(args.dqs)
    n_levels = len(args.dqs)
    args.Ks = _broadcast_int_tuple(args.Ks, n_levels)
    args.tier_n_layers = _broadcast_int_tuple(args.tier_n_layers, n_levels)
    args.tier_d_models = _parse_int_tuple(args.tier_d_models)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, dqs=args.dqs, tier_d_models=args.tier_d_models, tier_n_layers=args.tier_n_layers,
        context_len=args.context_len, n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window,
        rope_base=args.rope_base, bit_chain_n_heads=args.bit_chain_n_heads, bit_chain_gamma=args.bit_chain_gamma,
        bit_chain_fixed_kernel=args.bit_chain_fixed_kernel, bit_head_class=args.bit_head_class,
        bit_conv_kernel_size=args.bit_conv_kernel_size, bit_conv_impl=args.bit_conv_impl,
        bit_inner_downsample=args.bit_inner_downsample, bit_ssm_d_state=args.bit_ssm_d_state,
        code_ntp_weight=args.code_ntp_weight, byte_ntp_weight=args.byte_ntp_weight,
        quant_type=args.quant_type, code_embed_mode=args.code_embed_mode, fuse_encoder_levels=args.fuse_encoder_levels,
        byte_repr=args.byte_repr, code_head_mode=args.code_head_mode,
        tok_d_model=args.tok_d_model, tok_n_heads=args.tok_n_heads,
        tok_mlp_mult=args.tok_mlp_mult, tok_head_mode=args.tok_head_mode, tok_weight=args.tok_weight,
        cross_attn_rope=args.cross_attn_rope, decoder_own_trunk=args.decoder_own_trunk,
        decoder_kv_pass_through=args.decoder_kv_pass_through, decoder_q_pass_through=args.decoder_q_pass_through,
        layer_warmup_steps=args.layer_warmup_steps,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        # true whole-model compile, works fine WITH Config.layer_warmup_steps — see --compile's
        # own help text above (train()/eval_model() pass n_active, not raw step, into the model).
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v3_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} tier_n_layers={cfg.tier_n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} tok_head_mode={cfg.tok_head_mode} "
        f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
