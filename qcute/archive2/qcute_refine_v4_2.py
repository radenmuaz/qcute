"""qcute.qcute_refine_v4_2 — CLONE of qcute_refine_v4_1.py, further
UNIFIED (see the v4.1 section below for what carries over unchanged) PLUS
CONCAT-ONLY FUSION (session ask: "make v4.2 use by default only concat
mode, remove any cross attn stuff"). `Config.fuse_position` and the
separate `CrossBlock` cross-attention module it used to select between
("pre"/"post"/"both") are GONE from this file entirely — v4.2 fuses via
"concat" ONLY (see v4's own Config.fuse_position docstring, preserved
below, for what "concat" originally meant as one option among several):
the level-above's hidden state (projected via `fuse_kv_proj`, null-
prepended if `fuse_use_null_kv`) is appended to the tail of every
self.blocks layer's own K/V, one joint attention call per layer, no
separate cross-attention weights at all. Unconditional whenever
`fuse_encoder_levels=True` — no flag, matching this whole file's own
"no flag, run that scheme by default" convention (v4.1's `share_levellm`/
`share_code_head` before it). Generalizes to any level count unchanged
(verified this session via 3-level `validate_generation`) — every
non-top level fuses from the level above it, and `generate_kv_cache`
only ever needs a fused cache for level 0 (the only level ever sampled
from), regardless of `n_levels`.

Also (session ask, same message): `dq` is a single shared scalar
(minimum 8) and the embed/ntp_head/code_pre are the exact same objects
across EVERY level, byte level included — no separate byte_embed/256-way
softmax path. Independent-per-bit BCE head, not exact chain-rule
factorization — see the `dq`/loss-cropping notes further down for the
real "this is an upper bound, not exact cross-entropy" caveat this
implies for `val_bpb`.

---

qcute.qcute_refine_v4_1 — CLONE of qcute_refine_v4.py, PLUS EXTREME
WEIGHT SHARING (session ask: "clone v4 to v4.1 for extreme weight
sharing, only one levellm (can adjust num layer and dim) but shared
across byte and code levels"). New `Config.share_levellm` (default True):
every level's `LevelLM.self.blocks`/`ln_f`/fusion modules
(`fuse_kv_proj`/`fuse_null_kv`/fusion cross-attention, later REMOVED by
v4.2 above) are the SAME nn.Module objects as level 0's own — genuinely
tied weights (same Parameters, gradients from every level accumulate
into them), not separate same-shaped copies. Requires uniform
`tier_d_models`/`tier_n_layers` across every level (a shared trunk can't
have a different width/depth per level).

"even though shared, the k and attn window for each level can be
different" (session ask) — `Ks`/`attn_window` STAY genuinely per-level.
This required one real mechanical change from v4: `window` moved from a
CausalSelfAttention CONSTRUCTOR argument (baked into the module, one
value forever) to a FORWARD-time argument (`CausalSelfAttention.forward`/
`Block.forward` now both take `window` as a parameter) — the only way a
single shared module can still serve levels with different windows.

"this single levellm generates for itself its own bsq code for level 1
and above and ntp them, and in decoding phase reuse them" — unchanged
from v4's own PASS 1/PASS 2 structure (`RefineLM._encode`): the shared
trunk still runs once per level (its own forward pass, own NTP loss/code
emission), fusion still cross-attends level i to level i+1's own hidden
state — only the WEIGHTS doing this work are now shared, not the
computational structure itself.

"for first impl the levellm has both byte head and byte embed table and
bsq head and bsq linear map, though bsq head and bsq linear map can be
different each level by default, can be set weight sharing for ablation"
— `Config.share_code_head` (default False, independent of
`share_levellm`): ties every CODE level's (1..n_levels-1) own `embed`
(CodeEmbed)/`ntp_head`/`code_pre` ("bsq linear map," the D->dq
pre-quantization projection) to level 1's own. Off by default — each code
level keeps its own BSQ head/embed even though the big trunk is shared.
Byte level 0's own `byte_embed`/`ntp_head` are never affected (there's
only one byte-typed level; "sharing" it is meaningless).

First-impl scope (session: "for first impl"): `share_levellm=True`
requires `byte_repr="embed"` and `code_head_mode="independent"` (asserted
in `LevelLM.__init__`) — `BitPredictHead`'s chain-mode heads aren't wired
into the shared-trunk path yet, a follow-up not this session's scope.

Everything below (until noted) is v4's own original module docstring,
unchanged — v4.1 does not alter PASS 1/PASS 2, fusion, generation, or any
of v4's own session history; it only adds the sharing mechanism above on
top.

---

qcute.qcute_refine_v4 — CLONE of qcute_refine_v3.py, MINUS DecoderLevel
entirely. v3 added LevelLM fusion (Config.fuse_encoder_levels,
default True — see below) on top of v2's original DecoderLevel/tok_loss
cross-attention path; v4 removes DecoderLevel altogether, since fusion and
DecoderLevel turned out to do the literal same job with the same input
requirements (both predict a level's own next input, given that level's
own sequence history, optionally conditioned on the coarser code) — see
the session discussion this file's own history is built from. DecoderLevel
never contributed to `byte_loss`/`val_bpb` even in v3 (its reads stayed
detached), so removing it costs nothing measurable and saves the compute
of an entire extra CrossBlock + its own embeddings every step.

v4 ALSO fixes a gap v3 left open: v3's generate_no_cache/generate_kv_cache
were copied unchanged from v2 and never touched fusion at all — training
used it, generation didn't. v4's generation (both no-cache and KV-cache
variants) is fusion-aware and validated against each other via
validate_generation — see generate_kv_cache's own docstring for the
hierarchical caching scheme this required (a CLEAN cache per level, plus
one FUSED cache for level 0 specifically, the only level ever sampled
from).

WHY (the finding this fixes): in v2, `val_bpb` — the ONLY metric every
comparison table in docs/status.md is built from, and what checkpointer.
step() uses to pick best.pt — is computed exclusively from `byte_loss`
(LevelLM[0]'s own NTP loss), which is produced by a purely bottom-up
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

THE FIX — LevelLM fusion (v3's only real change, see LevelLM.
_fuse and RefineLM.forward's own "v3 PASS 2" comment for the full
mechanism): RefineLM.forward runs the SAME bottom-up sweep as v2 first
(PASS 1 — required regardless, since level i+1's own input IS level i's
own output code, so level i+1 structurally cannot exist before level i
has finished; "fuse before running the encoder" is circular for a level
fusing from ITSELF in one pass). Once PASS 1 has produced every level's
hidden state, levels 0..n_active-2 are re-run (PASS 2, same weights, same
input) — this time with a NEW cross-attention step (LevelLM._fuse)
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

Config.fuse_encoder_levels=False disables fusion — with no DecoderLevel
either, this makes every level fully independent except for the forward
code-chain (c_i feeds level i+1) and the code_ntp_weight gradient channel
— a genuine floor/control for isolating fusion's own effect.

EVERYTHING BELOW THIS POINT (the encoder tower's own per-level shape,
CodeEmbed/code_embed_mode, byte_ntp_weight) is UNCHANGED from
qcute_refine_v3.py/qcute_refine_v2.py — same recursive NTP encoder tower
as qcute_refine.py.

ENCODER TOWER: unchanged from qcute_refine.py — LevelLM[i] embeds its
own input sequence x^(i) (bytes at i=0, else c_{i-1}), runs a causal
transformer (own weights, tier_n_layers[i] deep), keeps an always-on
unconditioned NTP loss on its own next input element (own head, own
target — "so that targets are stable"), and every K[i] positions BSQ-
quantizes into its own emitted code c^(i), which becomes level i+1's
entire input. LevelLM[i].forward now returns its post-ln_f hidden
state h^(i) as a first-class output (qcute_refine.py already computed
this and discarded it; v2 is what actually uses it).

FUSION (LevelLM._fuse), one per ADJACENT level pair (i, i+1),
i = 0..N-2 — there is no fusion above the top level; nothing coarser
exists to cross-attend to. Inserted BEFORE level i's own self.blocks runs
(not after, as v2/v3's DecoderLevel was), so level i's own self-attention
— and hence its own ntp_loss — operates on a representation that has
already cross-attended to the level above:

    Q  = level i's own pre-self-attention embedding (its own input x^(i),
         embedded — NOT a separately-learned embedding; "encoder reads
         this directly")
    KV = h^(i+1), LevelLM[i+1]'s own hidden state from PASS 1 (see
         RefineLM._encode) — LevelLM[i+1]'s OWN INPUT is exactly
         c^(i), so its hidden state at code-block index b is already, by
         construction, "as of" real time (b+1)*K[i] — no new computation
         needed to get this

h^(i+1) is DETACHED before use — level i's fusion must not reshape level
(i+1)'s own hidden state (which stays trained purely by its own
unconditioned NTP loss; two objectives competing over the same hidden
state is exactly the moving-target failure mode this whole design avoids
elsewhere) — only level i's OWN weights (fuse_cross, its own embed/
blocks) get gradient from this.

A single cross-attention TRANSFORMER BLOCK (`fuse_cross`, a `CrossBlock` —
cross-attn sublayer + MLP sublayer, each pre-norm + residual, mirroring
this file's own causal `Block`'s shape) combines Q and KV. Causal safety:
query position t may only attend to KV block b once b is FULLY complete,
i.e. b < (t+1)//K[i] (the same "past, already-resolved block" rule
qcute_refine.py's module docstring worked through) — enforced via an
explicit boolean attention mask (jagged_causal_mask_and_positions),
not RoPE (Q and KV live at different granularities/lengths, so a shared
rotary basis doesn't apply — cross_attn_rope instead tags each side with
its own explicit raw-time position). A single learned "null" KV slot is
prepended and always visible, so early positions (before any KV block is
complete) still get a well-defined attention distribution instead of an
all-masked row.

No separate decode head: the level's OWN existing ntp_head (already
built for its unconditioned NTP loss) now also serves as the fused/
conditioned prediction — there's nothing else to build, since fusion
changes what self.blocks sees as input, not what happens after.

Generation properly uses fusion (v4's own fix — see generate_kv_cache's
own docstring for the hierarchical caching scheme this needed): only
LevelLM[0]'s own NTP head is ever sampled from (a genuinely new byte
requires nothing but causal history; fusion's own KV side needs an
ALREADY-COMPLETE code block, so it can condition on existing history but
never propose new blocks ahead of time).

No shared imports with qcute_refine.py or any qcutelm_vlt* fork (self-
contained-module convention, matching how qcutelm_vlt7->vlt8->...
evolved) — everything duplicated.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_pq.py
"""
import argparse
import gzip
import json
import math
import time
import warnings
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
    """v4.2 — CLONE of qcute_refine_v4_1.py, trimmed and made UNCONDITIONAL (session: "make that
    4.2 and no flag, run that scheme by default"). v4.1's `share_levellm`/`share_code_head` flags
    are GONE — v4.2 always shares everything: one trunk (self.blocks/ln_f/fuse_*) AND one head/
    embed/code_pre, across EVERY level INCLUDING level 0 (v4.1 kept byte level 0 structurally
    separate — its own byte_embed/256-way softmax ntp_head; v4.1's share_code_head only ever tied
    the CODE levels to each other). Session ask: "define a single bsq dq, and share head across
    all levels including byte". Consequences, both real simplifications:

    - `dqs`/`tier_d_models`/`tier_n_layers` (per-level tuples) collapse to single `dq`/`d_model`/
      `n_layers` values — there is only ONE shared width/depth/code-dimension now, a tuple never
      made sense once every level is forced uniform anyway ("trim other things").
    - `byte_repr`/`code_head_mode` are GONE — level 0 no longer gets a special nn.Embedding(vocab,
      D)+256-way-softmax path; EVERY level (byte included) uses the exact same `CodeEmbed(dq, D)`
      embed + `nn.Linear(D, dq)` ntp_head, trained with the same per-bit BCE loss
      (`chain_bce_loss`) code levels already used. Byte level's own dq-bit representation is
      produced by `byte_to_dqbits`/decoded by `dqbits_to_byte` (see their own docstrings) — dq
      minimum 8 (session: "dq minimum 8") since a byte needs 8 bits to be representable at all;
      `dq > 8` is allowed (more shared capacity for code levels) but byte SAMPLING only ever reads
      the first 8 bits back out — "for sampling bytes, just crop byte 9 and more, let bits valued
      0-255 reserved for raw bytes." At `dq==8` (the default) this is a lossless no-op identical
      to `byte_to_bits`/`bits_to_byte`; the cropping only matters once `dq>8` is actually used.
    - BitPredictHead ("chain" head mode) is now unreachable dead code in this file (kept, unused,
      not deleted this session — removing ~300 lines of otherwise-correct code wasn't worth the
      edit risk for a "first impl, trim other things" pass) — v4.2 always uses the plain
      independent-per-bit linear head, matching v4.1's own first-impl scope requirement anyway."""

    Ks: tuple[int, ...] = (2, 2, 2)
    dq: int = 8   # SINGLE shared BSQ code width, every level (byte included) — must be >= 8 (a
                    # byte needs 8 bits). See class docstring for the dq>8/cropping behavior.
    d_model: int = 96    # SINGLE shared width — every level's self.blocks/embed/head operate here.
    n_layers: int = 1    # SINGLE shared depth — every level's self.blocks has this many layers.
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
                                    # the model (byte level's own bits-mode NTP head, and any level's
                                    # code_head_mode=="chain"): "attn" (default,
                                    # original) = BitPredictHeadAttn, self-attention over the bit chain.
                                    # "conv" = BitPredictHeadConv, causal Conv1d instead. "ssm" =
                                    # BitPredictHeadSSM, linear-decay recurrence instead. "hsoftmax" =
                                    # BitPredictHeadHSoftmax, classic hierarchical softmax over the same
                                    # dq-depth binary tree — session: "find something that satisfy chain
                                    # probs validity and cheap and same repr power as large softmax
                                    # head." Unlike attn/conv/ssm (which reuse ONE classifying direction
                                    # per bit POSITION, shared across every prefix reaching that
                                    # position — the diagnosed bottleneck), hsoftmax gives every one of
                                    # the 2**dq-1 tree NODES its own private weight vector, so it has the
                                    # same degrees of freedom as a dense softmax-256 (255*D vs 256*D
                                    # params) while only ever touching the dq=8 nodes on the true
                                    # root-to-leaf path per example (same FLOPs as "independent"'s
                                    # nn.Linear(D,dq), ~32x cheaper than dense softmax-256). "conv_dilated"
                                    # = BitPredictHeadConvDilated, a WaveNet-style dilated depthwise/dense
                                    # conv STACK instead of "conv"'s single big kernel — session: "do this
                                    # stacked small kernel... check ar gen conv code, then train this."
                                    # Params/FLOPs scale ~L*dilation_base vs. "conv"'s dq (real savings at
                                    # larger dq, modest at dq=8); PURELY LINEAR across layers (no
                                    # activation), a genuine expressivity cost, not a free win — see its
                                    # own docstring. All five are drop-in equivalent in API (verified
                                    # fixed/loop-consistent) — see each class's own docstring for the
                                    # tradeoffs.
    bit_conv_kernel_size: int | None = None  # None (default) = dq (full receptive field) — see
                                    # BitPredictHeadConv's own docstring for why windowing isn't obviously
                                    # the right inductive bias here. Only used when bit_head_class=="conv".
    bit_conv_impl: str = "matmul"  # "matmul" (default, session reparam) or "conv1d" (original) — same
                                    # op (fixed causal window, weights shared across positions), different
                                    # dispatch; see BitPredictHeadConv's own docstring. "depthwise" (new)
                                    # = per-channel K-tap filters, no cross-channel mixing — 171x/228x
                                    # fewer params/FLOPs at full width (session: "consider making
                                    # bitpredictconv more efficient... maybe try group conv or
                                    # depthwise"). Only used when bit_head_class=="conv".
    conv_dilated_base: int = 2   # dilation_base for bit_head_class=="conv_dilated" (BitPredictHeadConvDilated)
                                    # — layer l has kernel_size=conv_dilated_base, dilation=
                                    # conv_dilated_base**l, L=ceil(log_base(dq)) layers. Only used when
                                    # bit_head_class=="conv_dilated".
    conv_dilated_mode: str = "depthwise"   # "depthwise" (default) or "dense" (session: "finish impl conv
                                    # dilated dense") — see BitPredictHeadConvDilated's own docstring.
                                    # Only used when bit_head_class=="conv_dilated".
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
    bit_downsample_h: bool = False   # False (default): h stays at full d_model in BitPredictHeadAttn/SSM
                                    # regardless of bit_inner_downsample — only the embed/attention/state
                                    # machinery runs at the smaller d_inner width (session: "can the
                                    # downsample flag only be applied on embeds, h maintains full dim").
                                    # True: restores the ORIGINAL (pre-this-session) behavior — h ALSO
                                    # projected down to d_inner via in_proj — for A/B testing whether
                                    # downsampling h itself (not just the embeds) causes a real quality
                                    # loss (session: "queue more experiments to test this hypothesis, repr
                                    # loss because of downsample h"). Only affects bit_head_class in
                                    # ("attn", "ssm") — "conv"/"conv_dilated" still use add (not concat) so
                                    # can't decouple yet; "hsoftmax" fundamentally can't decouple at all
                                    # (dot-product mechanism) — see docs/kv_contribution.md §18.
    bit_per_position_head: bool = True   # True (default): dq SEPARATE per-bit-position weight rows in
                                    # BitPredictHeadAttn/SSM (session: "for each self attn and ssm, allow
                                    # indp heads for each timestep different head, on by default"). False:
                                    # a single SHARED head instead (v4's original design). Only affects
                                    # bit_head_class in ("attn", "ssm").
    code_head_mode: str = "independent"   # RE-ENABLED (session: "later re enable flag to use
                                    # bitpredict heads, default indp now"). "independent" (default,
                                    # matches v4.2's whole "trim other things" direction): plain
                                    # nn.Linear(D, dq), independent per-bit logits — no true_bits
                                    # conditioning needed at all, cheapest. "chain": one of the
                                    # BitPredictHead* classes (Config.bit_head_class picks Attn/
                                    # Conv/SSM), exact autoregressive chain-rule factorization over
                                    # the dq bits, teacher-forced at train time (_forward_fixed) —
                                    # same shared object across every level (byte included), same as
                                    # "independent"'s ntp_head. Unlike v4/v4.1, this is now the ONLY
                                    # place BitPredictHead is reachable in this file. "word" (new) =
                                    # BitPredictHeadWordPredict — decomposes dq bits into
                                    # dq//word_bits WORDS, each a genuine 2**word_bits-way softmax
                                    # (session: "design another head, wordpredict... useful for dq
                                    # more than 8") — a middle ground between "chain"'s per-bit
                                    # BitPredictHead* family and a single flat V=2**dq softmax.
                                    # word_bits==dq degenerates to exactly that flat softmax (verified
                                    # numerically identical to byte_head_256way's own nn.Linear).
    word_bits: int = 8   # only used when code_head_mode=="word". Must evenly divide dq. Default 8 =
                                    # one word per byte-worth of bits (matches dq's own default=8,
                                    # so n_words=1 — the degenerate/flat-softmax case — unless dq is
                                    # raised). Try 4 or 2 for genuine multi-word chains.
    word_d_embed: int | None = None   # explicit override for BitPredictHeadWordPredict's per-word
                                    # embedding width. None (default): derived from word_embed_downsample.
    word_embed_downsample: int = 1   # d_embed = d_model // word_embed_downsample when word_d_embed
                                    # is None (session: "if d_embed not given use embed_downsample
                                    # param to down sample x vs d"). Only used when code_head_mode==
                                    # "word" and word_d_embed is None.
    byte_head_256way: bool = False   # ABLATION flag (session: "another hack is make ablation use
                                    # regular byte 256-way head, put as flag"). False (default):
                                    # level 0 uses the SAME shared dq-bit embed/head as every code
                                    # level (v4.2's whole point). True: level 0 gets its OWN,
                                    # UNSHARED nn.Embedding(vocab, D) + nn.Linear(D, vocab), trained
                                    # with ordinary 256-way cross-entropy — exactly v4's own
                                    # byte_repr="embed" mode, matching bytelm.py's convention. This
                                    # opts level 0 OUT of the head-sharing pool entirely — code
                                    # levels (1+) then share their OWN dq-bit embed/head among
                                    # THEMSELVES (owned by level 1, not level 0) instead. self.blocks/
                                    # ln_f/fuse_* (the TRUNK) stay shared across EVERY level
                                    # regardless of this flag — only tests whether the exact-softmax/
                                    # unshared byte head specifically resolves the instability
                                    # documented in docs/kv_contribution.md §11, isolated from
                                    # trunk-sharing (which stays on either way).
    byte_softmax_head_only: bool = False   # ABLATION flag, narrower than byte_head_256way (session:
                                    # "no byte embedding, assume byte as bits 0 to bits 255, only
                                    # head is softmax"). False (default): unaffected. True: level 0
                                    # KEEPS the shared dq-bit CodeEmbed for its input AND the shared
                                    # code_pre (the BSQ projection feeding level 1) — only its OUTPUT
                                    # ntp_head becomes its own unshared nn.Linear(D, vocab), trained
                                    # with ordinary 256-way cross-entropy instead of the shared dq-bit
                                    # chain_bce_loss. Isolates whether it's specifically the shared
                                    # dq-bit OUTPUT head (byte 256-way vs. code's ~8-bit target both
                                    # forced through one nn.Linear/BitPredictHead) that's unstable,
                                    # independent of whether the INPUT embedding is also shared —
                                    # byte_head_256way (above) unshares embed+head+code_pre together
                                    # and can't distinguish which of the three actually matters;
                                    # mutually exclusive with byte_head_256way (asserted below).
    byte_head_factored: bool = False   # ABLATION flag, same "narrow" shape as byte_softmax_head_only
                                    # (embed/code_pre stay the shared dq-bit ones; only the byte-level
                                    # READOUT is private) but the private readout is a
                                    # FactoredSoftmaxHead instead of a dense nn.Linear(D,vocab) —
                                    # session: "use structured matrix but to replace dense linear map
                                    # to 2**n way output softmax... some loss in repr ok for params
                                    # saving." Computes logits as an OUTER SUM of two small
                                    # projections (D->v1, D->v2, v1*v2==vocab) instead of one dense
                                    # D->vocab matrix — params drop from vocab*D to D*(v1+v2) (8x
                                    # fewer at vocab=256,D=256,v1=v2=16), FLOPs drop correspondingly
                                    # (~7.9x). Still a genuine softmax (over a STRUCTURED logit
                                    # vector) — no chain-rule/teacher-forcing needed, unlike the
                                    # BitPredictHead* family. Representational cost: the vocab-way
                                    # logit vector, reshaped [v1,v2], is constrained to row-effect +
                                    # column-effect (additively separable in the two-factor index) —
                                    # can't express genuine "outcome i needs to interact with outcome
                                    # j" cross terms a dense head could. Mutually exclusive with
                                    # byte_head_256way/byte_softmax_head_only/quant_type=="simplex".
    byte_head_lowrank: bool = False   # ABLATION flag, same "narrow" shape as byte_head_factored, but
                                    # the private readout is LowRankSoftmaxHead (the classic "softmax
                                    # bottleneck," Yang et al. 2018: Linear(D,rank) -> Linear(rank,
                                    # vocab)) instead of the outer-sum FactoredSoftmaxHead — session:
                                    # "how good is factoredsoftmax vs just low rank... analyze rank."
                                    # Strictly MORE expressive than byte_head_factored at matched
                                    # rank/param budget (every FactoredSoftmaxHead-representable logit
                                    # matrix is also representable by LowRankSoftmaxHead at rank
                                    # v1+v2, since the former is a zero-free-parameter special case of
                                    # the latter — see LowRankSoftmaxHead's own docstring) — the
                                    # theoretically safer of the two structured/cheap alternatives to
                                    # a dense byte_softmax_head_only. Mutually exclusive with
                                    # byte_head_256way/byte_softmax_head_only/byte_head_factored/
                                    # quant_type=="simplex".
    byte_head_rank: int | None = None   # rank for byte_head_lowrank. None (default) = d_model // 4
                                    # (session's own suggested "4x downscale"). Only used when
                                    # byte_head_lowrank=True.
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
    untie_levels: bool = False    # session: "by untie, make it like v4, different head, embed, lm
                                    # transformer each level" -- v4.2's OWN defining feature is
                                    # extreme, UNCONDITIONAL sharing (module docstring: "one trunk
                                    # ... AND one head/embed/code_pre, across EVERY level INCLUDING
                                    # level 0") -- RefineLM.__init__ always passes shared=encoders[0]/
                                    # shared_head=encoders[0] (or the code_bits!=vocab split-pool
                                    # variant) for every level i>0, with no flag to turn it off
                                    # (v4.1's own share_levellm=False option was REMOVED when this
                                    # file was cloned from v4.1, per its own module docstring's
                                    # "v4.2 always shares everything" section). True here reverts
                                    # to v4's ORIGINAL per-level separation: every level builds a
                                    # completely FRESH self.blocks/ln_f/fuse_* trunk (shared=None)
                                    # and a fresh embed/head/code_pre pool (shared_head=None) of its
                                    # own -- no aliasing across levels at all, for ANY quant_type,
                                    # not just quant_type=="simplex" (LevelLM.__init__'s existing
                                    # `shared_head is not None -> borrow, else -> build fresh` branch
                                    # already handles shared_head=None correctly for every mode, no
                                    # new code path needed there). Orthogonal to and composable with
                                    # `simplex_untie_head`: this controls whether levels share pools
                                    # WITH EACH OTHER; simplex_untie_head controls whether, WITHIN one
                                    # pool, the embed and NTP classifier are tied to each other.
                                    # Setting both True is the fullest "like v4" separation --
                                    # params scale roughly with n_levels instead of staying flat.
    fuse_mode: str = "concat"     # session: "rerun with this: post cross attn" -- "concat" (default,
                                    # v4.2's own unconditional mechanism, see module docstring): no
                                    # separate cross-attention weights, the level-above's own hidden
                                    # state is appended to self.blocks' own K/V. "cross_attn_post":
                                    # reintroduces v4's original CrossBlock-based fusion (see that
                                    # class's own docstring) -- a genuinely separate, separately-
                                    # weighted cross-attention sublayer run AFTER self.blocks/ln_f
                                    # produce this level's own clean hidden state (Q), reading the
                                    # level-above's own hidden state (K/V). Only meaningful for a
                                    # level with `fuse_d_model is not None` (i.e. not the top level).
    untie_fusion_pass: bool = False   # session: "make the each level-pass separate weights, as if
                                    # there is 3 level lm... by free weight this means like a fresh
                                    # level lm own embed, transformer, linear head" -- True gives
                                    # PASS 2 (this level's own FUSED forward call, RefineLM._encode's
                                    # second sweep) a COMPLETELY separate identity from PASS 1: own
                                    # `embed_pass2`, own `blocks_pass2`/`ln_f_pass2` trunk, own
                                    # `simplex_head_pass2` if simplex_untie_head is also set -- not
                                    # just a separate trunk. Combined with `untie_levels`, a 2-level
                                    # config effectively trains THREE independent LMs: level 0's own
                                    # unconditional pass (PASS 1), level 1's own unconditional pass
                                    # (PASS 1, the top level, never fuses regardless), and level 0's
                                    # fused pass (PASS 2) as a third, separately-weighted model that
                                    # only additionally reads the level-above via cross-attention
                                    # (`fuse_mode`). Currently only implemented for byte-level
                                    # `quant_type=="simplex"` (LevelLM.__init__ asserts this) -- the
                                    # only case this session's own diagnostic configs exercise.
    quant_type: str = "bsq"       # TWO real modes: "bsq" (default) and "simplex" (new this session —
                                    # session: "generalize with flag to mode where every level is softmax
                                    # head 256 way... instead of sign and ste, do gumbel softmax ste...
                                    # basically no grid assumption that bsq carries... maintain 2 modes
                                    # now: bsq, and simplex"). Plus "identity" (ceiling-baseline
                                    # diagnostic, training-only, sound only alongside code_ntp_weight=
                                    # tok_weight=0.0 — see qcute_refine.py's Config docstring for the full
                                    # rationale, unchanged) — not a third first-class mode, a debugging aid.
                                    # BSQ's dq independent sign-bits form an implicit hypercube GRID
                                    # (2**dq corners, with a bit-factorized structure that code_head_mode=
                                    # "independent"/"chain" both lean on). "simplex" drops that structure
                                    # entirely: every level's code is a flat, unstructured V=2**code_bits-
                                    # way CATEGORY (a point on the probability simplex, no bit
                                    # factorization at all — hence the name), produced via
                                    # `gumbel_quantize` (torch's built-in Gumbel-Softmax + hard straight-
                                    # through, replacing bsq_quantize's sign()+STE) and predicted via a
                                    # genuine V-way softmax classifier (cross-entropy, replacing
                                    # chain_bce_loss) at EVERY level uniformly, byte included — the same
                                    # exact-softmax mechanism byte_head_256way/byte_softmax_head_only use
                                    # for level 0 alone, now uniform across the whole tower. Intuition
                                    # (session): letting the model learn its own best discrete byte-code/
                                    # downsampling scheme fully end-to-end, unconstrained by BSQ's grid,
                                    # rather than imposing hypercube structure on what a "code" can be.
                                    # Uses NO separate `code_pre`/`ntp_head` modules at all — see
                                    # `code_bits` and `LevelLM.__init__`'s own "simplex" branch for why.
    code_bits: int = 8            # Only used when quant_type=="simplex": CODE levels' (1+) codebook size
                                    # V = 2**code_bits, one shared scalar across every code level (matching
                                    # this file's single-shared-scalar convention — dq/d_model/n_layers
                                    # above). UNLIKE `dq` (must be >=8 to represent a raw byte via
                                    # independent bits), `code_bits` has no such floor — levels>0 emit a
                                    # genuinely separate learned categorical code, not a byte-bit encoding,
                                    # so code_bits<8 (a smaller, more heavily compressed alphabet) is a
                                    # valid, intended use (session: "this mode can generalize to n<8").
                                    # Byte level 0 ALWAYS uses its own fixed vocab=256 (2**8) table — a raw
                                    # byte genuinely has 8 bits of information, that doesn't shrink or grow
                                    # with this flag. At the default code_bits==8, V==vocab==256 exactly,
                                    # so level 0's table and every code level's table happen to be the SAME
                                    # SHAPE and get fully unified into one pool (true "every level softmax
                                    # 256-way", byte included) — LevelLM.__init__/RefineLM.__init__ handle
                                    # this by pool-shape, not a special case: whenever code_bits!=8, byte
                                    # keeps its own private vocab=256 pool and code levels 1+ share a
                                    # SEPARATE V-sized pool among themselves (owned by level 1), same
                                    # split pattern `byte_head_256way` already uses for the same reason —
                                    # you cannot alias two differently-shaped embedding tables. Warns
                                    # (doesn't error) above 8: e.g. code_bits=16 means every code level's
                                    # classifier is a dense 65536-way softmax — expensive, not unsound.
    simplex_untie_head: bool = False   # Only used when quant_type=="simplex" (session: "easier if you can
                                    # make 4.2 untie weight mode") — classic weight-tying ablation (Press &
                                    # Wolf 2017's own "tied vs untied" question), applied to the ONE place
                                    # this file's "simplex" mode literally ties weights: the NTP
                                    # classification readout (`F.linear(h, self.embed.weight)` in forward()'s
                                    # loss computation and _sample_next_byte) shares its weight matrix with
                                    # the INPUT embedding table by default. False (default): unchanged,
                                    # weight-tied, exactly as originally built. True: each level gets its
                                    # own PRIVATE `nn.Linear(D, V)` classifier (`LevelLM.simplex_head`),
                                    # mirroring the exact same is_byte_level/shared_head/else sharing
                                    # structure `self.embed` itself already uses — so this is orthogonal to,
                                    # not a replacement for, the existing byte/code-level sharing scheme:
                                    # untying only separates "read a token" from "classify the next one,"
                                    # it does not change WHICH levels share a pool with each other.
                                    # Deliberately does NOT touch the OTHER weight-tie this mode has —
                                    # `code_embed.weight`, used to produce the code a level hands UPWARD
                                    # (forward()'s own c_i computation, `maybe_emit_code` in
                                    # generate_kv_cache) — a conceptually different job (encoding a
                                    # compressed representation, not classifying a known-vocabulary token),
                                    # not what "weight tying" classically refers to.
    gumbel_tau: float = 1.0       # Softmax temperature used by the quant_type=="simplex" quantizer
                                    # (`gumbel_quantize`) regardless of `use_gumbel_noise`. Lower =
                                    # peakier/closer to a true hard argmax; higher = softer/smoother.
                                    # 1.0 is the standard starting default in the literature.
    use_gumbel_noise: bool = False  # Only used when quant_type=="simplex". False (default, session:
                                    # "is it ok to have no gumbel, just default argmax and ste like
                                    # bsq did... because gumbel is expensive"): deterministic softmax
                                    # + hard-argmax straight-through — `soft + (hard - soft).detach()`,
                                    # the EXACT same idiom `bsq_quantize` already uses for its own
                                    # sign()+STE, just categorical (softmax/argmax) instead of
                                    # per-dim (sign) — no random sampling, cheapest option. True:
                                    # genuine Gumbel-Softmax (`F.gumbel_softmax`, samples fresh Gumbel
                                    # noise every call before the hard argmax+STE) — the textbook
                                    # stochastic relaxation, adds real per-step cost (noise sampling +
                                    # anneal-sensitive) for exploration `bsq_quantize` never needed
                                    # either (BSQ has no stochastic variant in this file at all).
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
    # v4.2 (session: "make v4.2 use by default only concat mode, remove any cross attn stuff"):
    # `Config.fuse_position` ("pre"/"post"/"both") and the separate `CrossBlock` cross-attention
    # module it selected between are GONE — v4.2 fuses via "concat" ONLY, unconditionally,
    # whenever `fuse_encoder_levels=True`. No separate cross-attention weights exist in this file
    # at all: the level-above's hidden state (projected to this level's D via `fuse_kv_proj`,
    # null-prepended the same way if `fuse_use_null_kv`) is appended to the TAIL of every
    # self.blocks layer's own K/V, and each layer does ONE joint attention call over [local
    # windowed K/V ; fused tail] — see CausalSelfAttention._fuse_kv_proj/LevelLM._prep_concat.
    # Each layer derives its OWN K/V for the fused tail via ITS OWN qkv weights (same weights it
    # uses for local tokens) — no new parameters beyond `fuse_kv_proj`/`fuse_null_kv`. Generalizes
    # to any level count unchanged from v4/v4.1 (every non-top level fuses from the level above it,
    # `RefineLM._encode`'s own PASS 2 loop; `generate_kv_cache` only ever needs a fused cache for
    # level 0 specifically — the only level ever sampled from — regardless of `n_levels`, verified
    # this session via 3-level `validate_generation`).
    fuse_use_null_kv: bool = True  # whether the concat fusion tail gets a learned "null" KV slot
                                    # (self.fuse_null_kv), always visible, prepended to the real KV rows.
                                    # True (default): matches every fusion config trained so far. Exists
                                    # because CROSS-attention to a coarser, possibly-not-yet-resolved
                                    # sequence can have genuinely ZERO real KV rows available (early
                                    # positions, before any code block completes) — unlike self-attention,
                                    # which always has at least the diagonal (itself) as a key, so it never
                                    # needs an analogous fallback. False: no null slot at all — verified
                                    # (session empirical test, F.scaled_dot_product_attention with
                                    # zero-length or fully-masked KV, forward AND backward) this does NOT
                                    # crash or NaN on this backend; cross-attention output is simply a
                                    # well-defined ZERO for positions with no real KV yet, i.e. a hard
                                    # no-op fallback instead of a learned non-trivial bias. Genuine
                                    # ablation, not just a correctness toggle — tests whether the null
                                    # slot's own cheap learned capacity (implicated in the k32_narrow
                                    # fusion-contribution probe's finding that ~90% of fusion's benefit was
                                    # capacity/structure, not content — see docs/kv_contribution.md §7) is
                                    # itself doing real work, or whether the model does just as well
                                    # falling back to nothing at all for those positions.
    fusion_ntp_weight: float = 1.0  # NEW (session ask: "make it modular, each lm can act
                                    # independently and still get good bpb ... generalize to any
                                    # level"). Scales the SUM of every fusing level's PASS 2
                                    # (fused/cross-attended) NTP loss, ADDED to the loss alongside
                                    # PASS 1's own — not a replacement. Before this: PASS 2 silently
                                    # OVERWROTE PASS 1's loss for every fusing level (0..n_active-2),
                                    # so only the fused view ever trained, and no level's own
                                    # self-attention-only weights were ever pushed toward standalone
                                    # competence (see docs/kv_contribution.md's "null_kv as a
                                    # training-time regularizer, not a content channel" finding —
                                    # motivated this directly). Now: for every level i, PASS 1's own
                                    # loss (byte_ntp_weight for i==0, code_ntp_weight for i>0) AND,
                                    # if i fuses (i < n_active-1 and fuse_encoder_levels), PASS 2's
                                    # loss (this weight) BOTH enter the total loss — 3 distinct NTP
                                    # losses for the common 2-level case (level0 PASS1 byte, level1
                                    # PASS1 code, level0 PASS2 fused byte), generalizing to 2*n_active-1
                                    # for n_active levels (n_active PASS1 terms + n_active-1 PASS2
                                    # terms). `byte_loss`/`val_bpb`/checkpointing still track ONLY
                                    # level 0's PASS 2 value specifically (the "real"/longest-path
                                    # metric, unchanged from before) — this weight controls how much
                                    # PASS 2 across ALL fusing levels contributes to the GRADIENT,
                                    # not what gets reported/compared against baselines.
    # --- v4: NO DecoderLevel at all (see module docstring) — LevelLM fusion is the only
    # cross-attention mechanism, and it directly conditions the metric that matters (byte_loss),
    # unlike v2/v3's DecoderLevel/tok_loss which never could (detached, separate scalar). Every
    # tok_d_model/tok_n_heads/tok_mlp_mult/tok_head_mode/tok_weight/decoder_own_trunk/
    # decoder_kv_pass_through/decoder_q_pass_through field from v2/v3 is GONE — meaningless without
    # DecoderLevel. Use qcute_refine_v3.py if you need those for comparison/reproducibility.
    cross_attn_rope: bool = True   # DEFAULT ON — applies RoPE to LevelLM._fuse's own cross-attention
                                    # Q/K (the only cross-attention left in this file): Q gets its own
                                    # raw-time position (0..L-1); each KV slot gets the null slot's own
                                    # fixed reference position (0) or, for a real code block, the raw-time
                                    # position it becomes fully causally resolved at ((b+1)*K[level]-1) —
                                    # gives the model actual relative-distance information instead of just
                                    # the boolean allowed/blocked mask. False restores no-positional-info
                                    # cross-attention (real option, not removed) — session finding
                                    # (docs/status.md): False actually WON in the v2/DecoderLevel setting
                                    # (2.5645 vs 2.6310) — untested whether that holds for fusion too.
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

    def __post_init__(self):
        assert self.dq >= 8, f"Config.dq must be >= 8 (a byte needs 8 bits to be representable), got {self.dq}"
        assert sum([self.byte_head_256way, self.byte_softmax_head_only, self.byte_head_factored, self.byte_head_lowrank]) <= 1, \
            "byte_head_256way, byte_softmax_head_only, byte_head_factored, and byte_head_lowrank are mutually exclusive ablations"
        if self.byte_head_rank is None:
            self.byte_head_rank = self.d_model // 4
        if self.quant_type == "simplex":
            assert not (self.byte_head_256way or self.byte_softmax_head_only or self.byte_head_factored or self.byte_head_lowrank), \
                "quant_type='simplex' already makes every level (byte included) an exact softmax " \
                "classifier over its own shared embedding table — byte_head_256way/" \
                "byte_softmax_head_only/byte_head_factored/byte_head_lowrank are redundant with " \
                "(and structurally incompatible with) it"
            if self.code_bits > 8:
                warnings.warn(
                    f"Config.code_bits={self.code_bits} > 8 under quant_type='simplex': every code "
                    f"level's softmax classifier is 2**{self.code_bits}={2**self.code_bits}-way — "
                    f"this is sound but the softmax gets large fast; not an error, just a heads-up.",
                    stacklevel=2,
                )


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


def gumbel_quantize(logits: torch.Tensor, tau: float, use_gumbel_noise: bool = False) -> torch.Tensor:
    """quant_type=="simplex"'s own quantizer — replaces bsq_quantize's sign()+STE (hypercube corner
    rounding) with a categorical straight-through estimator (a point on the probability SIMPLEX,
    hence the mode's name — no hypercube/bit-factorization structure at all).

    use_gumbel_noise=False (default, cheap): plain softmax + hard-argmax STE — `soft +
    (hard - soft).detach()` — forward value is the true argmax one-hot, backward gradient flows
    through `soft` as if no rounding happened at all. Deterministic, no sampling — the EXACT same
    "hard forward, soft backward" idiom bsq_quantize already uses for sign()+STE, just categorical.

    use_gumbel_noise=True: genuine Gumbel-Softmax — samples fresh Gumbel noise before the hard
    argmax+STE, the textbook stochastic relaxation. Real per-step cost (noise sampling) the default
    doesn't pay; bsq_quantize itself has no stochastic variant either, so the deterministic default
    is the more direct analogue of what BSQ already does.

    BUG FIX (session: crashed with `torch.AcceleratorError: scatter: index -1 is out of bounds` on
    MPS after ~800 training steps): originally called `F.gumbel_softmax(logits, tau=tau, hard=True,
    dim=-1)` directly, whose OWN internal Gumbel sampling (`-log(-log(u))`, `u ~ Uniform(0,1)`) has
    no epsilon clamp — over enough steps x a 256-way softmax x however many codes per batch, `u`
    eventually underflows to exactly `0.0` or `1.0` in float32, sending `-log(-log(u))` to `±inf`;
    two `inf`s colliding in the same softmax row produce `NaN`, and `NaN.argmax()` returned `-1` on
    MPS specifically, which `F.one_hot` then can't scatter into a size-256 dim. Fixed by sampling
    Gumbel noise manually with `u` clamped away from both `0` and `1` before the double-log — the
    standard numerically-safe form, torch's own built-in just doesn't apply it."""
    if use_gumbel_noise:
        eps = torch.finfo(logits.dtype).tiny   # smallest positive normal float — clamps u off {0,1}
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(u))
        soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    else:
        soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(soft.argmax(-1), num_classes=logits.shape[-1]).to(soft.dtype)
    return soft + (hard - soft).detach()


MAX_PQ_TABLE_DQ = 16   # 2**16 = 65536 rows — the ceiling code_embed_mode=="pq_table" allows


class CodeEmbed(nn.Module):
    """Maps a level's own dq-dim BSQ (or identity-quantized) code to a
    D-dim representation, wherever a raw code is consumed directly (an
    LevelLM's own input at level>0, or a DecoderLevel's
    kv_pass_through/q_pass_through raw-code embed at any level — NOT
    level 0's byte-bits embed, a different concern already covered by
    Config.byte_repr). See Config.code_embed_mode for the "linear"/"mlp"/
    "pq_table" mode descriptions and the motivating hypothesis.

    pq_table's forward value is an exact table lookup (the code truly is
    one of 2**in_dq discrete corners when quant_type=="bsq" — see
    bsq_quantize), but a naive lookup is non-differentiable in the code
    itself, which would sever the ONLY gradient path currently training
    the code's own producer (LevelLM.code_pre — nothing else reads
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


def jagged_causal_mask_and_positions(L: int, n_blocks: int, K: int, kv_window: int | None, device: torch.device,
                                       include_null: bool = True):
    """v3: factored out of DecoderLevel's own mask/rope-position construction (unchanged math, see
    DecoderLevel.forward's own "JAGGED STAIRCASE" comment for the full derivation) so LevelLM's
    new fusion cross-attention (v3's core change — see module docstring) can share the exact same,
    already-verified causal geometry instead of re-deriving it. Both attend from a length-L Q sequence
    at some level's own raw/finer-time resolution to a coarser level's own n_blocks code-block KV
    sequence (plus, if include_null, a prepended, always-visible null slot — see Config.
    fuse_use_null_kv). Returns (disallow [L, (1 if include_null else 0)+n_blocks] bool,
    True=blocked — nn.MultiheadAttention/CrossBlock convention; k_pos [(1 if include_null else 0)+
    n_blocks] long, each KV slot's own raw-time position for cross_attn_rope, null slot pinned to 0).
    include_null=False with n_blocks=0 (or every real block masked for a given query row) genuinely
    hands F.scaled_dot_product_attention a zero-length or fully-masked KV — verified this session
    (forward AND backward) to degrade cleanly to a well-defined zero, no crash/NaN, on this backend."""
    t_idx = torch.arange(L, device=device).unsqueeze(1)
    b_idx = torch.arange(n_blocks, device=device).unsqueeze(0)
    n_complete = (t_idx + 1) // K
    visible = b_idx < n_complete
    if kv_window is not None:
        visible = visible & (b_idx >= n_complete - kv_window)
    block_pos = (torch.arange(n_blocks, device=device) + 1) * K - 1
    if include_null:
        null_col = torch.ones(L, 1, dtype=torch.bool, device=device)
        visible = torch.cat([null_col, visible], dim=1)
        null_pos = block_pos.new_zeros(1)
        k_pos = torch.cat([null_pos, block_pos])
    else:
        k_pos = block_pos
    disallow = ~visible
    return disallow, k_pos


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    """`fuse_kv`/`fuse_disallow`/`fuse_rope_k` (all default None): v4.2's only fusion mechanism
    (concat, unconditional — see module docstring) — a fixed, already-D-dim, already
    null-prepended [B, Nf, D] coarser-level KV tail, appended to this layer's OWN local K/V so one
    joint SDPA call covers both. Every layer derives its OWN K/V view of the SAME fixed fuse_kv
    tensor via `_fuse_kv_proj` (this layer's own self.qkv weights, not a separate cross-attention
    module) — Q for the fused positions is never computed (not needed, those aren't real query
    positions). `fuse_disallow` [T, Nf] bool (True=blocked, jagged_causal_mask_and_positions'
    convention) gates which fused rows each local query position may see."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._warned_dense_fallback = False
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _fuse_kv_proj(self, fuse_kv: torch.Tensor, fuse_rope_k) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nf, D = fuse_kv.shape
        H, hd = self.n_heads, self.head_dim
        qkv_f = self.qkv(fuse_kv).reshape(B, Nf, 3, H, hd).permute(2, 0, 3, 1, 4)
        fk, fv = qkv_f[1], qkv_f[2]   # discard the (unused/never-queried) q slice
        if fuse_rope_k is not None:
            fk = apply_rope(fk, *fuse_rope_k)
        return fk, fv

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None,
                fuse_kv: torch.Tensor | None = None, fuse_disallow: torch.Tensor | None = None,
                fuse_rope_k=None) -> torch.Tensor:
        """v4.1: `window` is now a FORWARD-time argument, not baked into the module at construction
        — required for extreme weight sharing (Config.share_levellm): the SAME CausalSelfAttention
        object is called at every level, and levels can have different K/attn_window (session ask:
        "even though shared, the k and attn window for each level can be different") even though
        they share every weight. Caller (LevelLM.forward, via RefineLM) passes each level's own
        window in, matching v4's original per-instance `self.window` value exactly when NOT shared."""
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None and T % window == 0 and T > window:
            y = self._forward_chunked(q, k, v, window, fuse_kv, fuse_disallow, fuse_rope_k)
        else:
            if window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window — falling back to DENSE attention for this layer. "
                      f"Only warns once per layer instance.")
                self._warned_dense_fallback = True
            if fuse_kv is not None:
                fk, fv = self._fuse_kv_proj(fuse_kv, fuse_rope_k)
                k_full = torch.cat([k, fk], dim=2)
                v_full = torch.cat([v, fv], dim=2)
                local_allow = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                attn_mask = torch.cat([local_allow, ~fuse_disallow], dim=1)   # [T, T+Nf]
                y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                          fuse_kv: torch.Tensor | None = None, fuse_disallow: torch.Tensor | None = None,
                          fuse_rope_k=None) -> torch.Tensor:
        B, H, T, hd = q.shape
        W = window
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

        if fuse_kv is not None:
            # SAME fixed fuse_kv tail for every chunk (each chunk just sees a different, jaggedly-
            # masked SUBSET of it, per its own absolute t range) — this layer's own K/V view of it,
            # via _fuse_kv_proj, computed ONCE and broadcast across chunks (cheap: Nf is small).
            fk, fv = self._fuse_kv_proj(fuse_kv, fuse_rope_k)   # [B, H, Nf, hd]
            Nf = fk.size(2)
            fk_b = fk.unsqueeze(2).expand(B, H, n_chunks, Nf, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Nf, hd)
            fv_b = fv.unsqueeze(2).expand(B, H, n_chunks, Nf, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Nf, hd)
            kb = torch.cat([kb, fk_b], dim=2)
            vb = torch.cat([vb, fv_b], dim=2)
            # fuse_disallow is [T, Nf] built at this layer's own absolute-t resolution (T=n_chunks*W) —
            # chunk c's queries are exactly the contiguous t-slice [c*W, (c+1)*W), so a plain .view
            # slices it correctly per chunk, no re-derivation needed.
            fuse_allow_chunked = (~fuse_disallow).view(n_chunks, W, Nf)
            fuse_mask_batched = fuse_allow_chunked.unsqueeze(0).expand(B, n_chunks, W, Nf).reshape(B * n_chunks, 1, W, Nf)
            mask_batched = torch.cat([mask_batched, fuse_mask_batched], dim=-1)

        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None,
                      fuse_kv: torch.Tensor | None = None, fuse_rope_k=None):
        """fuse_kv/fuse_rope_k: "concat" mode's generation-time counterpart to forward's own —
        NOT part of the cached local self-attention K/V (new_k/new_v, returned/cached, never
        include it — fuse_kv_rows grows every step, so it's cheaply recomputed fresh each call
        instead of cached). No explicit visibility mask needed here — same reasoning
        generate_kv_cache's own module docstring already gives for "pre"/"post"/"both": every row
        present in fuse_kv_rows at this step is, by construction, already causally valid."""
        B, _, D = x_new.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x_new).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        new_k = k if cache_k is None else torch.cat([cache_k, k], dim=2)
        new_v = v if cache_v is None else torch.cat([cache_v, v], dim=2)
        if fuse_kv is not None:
            fk, fv = self._fuse_kv_proj(fuse_kv, fuse_rope_k)
            k_attn, v_attn = torch.cat([new_k, fk], dim=2), torch.cat([new_v, fv], dim=2)
        else:
            k_attn, v_attn = new_k, new_v
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(B, 1, D)), new_k, new_v


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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None,
                fuse_kv: torch.Tensor | None = None, fuse_disallow: torch.Tensor | None = None,
                fuse_rope_k=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, window, fuse_kv, fuse_disallow, fuse_rope_k)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_step(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                      cache_k: torch.Tensor | None, cache_v: torch.Tensor | None,
                      fuse_kv: torch.Tensor | None = None, fuse_rope_k=None):
        attn_out, new_k, new_v = self.attn.forward_step(self.ln1(x_new), cos_new, sin_new, cache_k, cache_v,
                                                           fuse_kv, fuse_rope_k)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_k, new_v


class CrossBlock(nn.Module):
    """Single cross-attention transformer block: cross-attn sublayer (Q from one sequence, K/V
    from another) + MLP sublayer, each pre-norm + residual — same shape as this file's own causal
    `Block`, with self-attention swapped for cross-attention. Ported near-verbatim from
    qcute_refine_v4.py's own `CrossBlock` (session: "rerun with this: post cross attn" — v4.2's
    own concat-only fusion has no separate cross-attention weights at all, see the module
    docstring; this reintroduces v4's original mechanism as an alternative `fuse_mode`). RoPE is
    optional (Config.cross_attn_rope) — Q and KV live at different granularities/lengths, so they
    can't share one contiguous rotary range the way self-attention does, but each side can still
    get its own explicit position tag via rope_q/rope_k."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
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
        """attn_mask: bool [Lq, Lkv], True = BLOCKED — inverted internally since
        F.scaled_dot_product_attention's boolean convention is the opposite (True = may attend)."""
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


def byte_to_dqbits(byte_ids: torch.Tensor, dq: int) -> torch.Tensor:
    """[*] long byte ids (0..255) -> [*, dq] float, each bit in {-1,+1}/sqrt(dq) — LSB-first,
    deterministic, no learned parameters. v4.2: byte level shares the exact same dq-dim
    representation every code level uses (Config.dq, single shared value, minimum 8 — see
    Config's own docstring). dq==8 (the default): a byte losslessly IS its own 8-bit BSQ-shaped
    code, identical to v4's own byte_to_bits. dq>8: bits 8..dq-1 are zero-padded — reserved extra
    shared-representation capacity a byte's own identity never needs (only code levels benefit
    from it); see dqbits_to_byte for the matching crop on the decode side."""
    assert dq >= 8
    bits8 = ((byte_ids.unsqueeze(-1) >> torch.arange(8, device=byte_ids.device)) & 1).float()
    if dq > 8:
        pad = torch.zeros(*byte_ids.shape, dq - 8, device=byte_ids.device, dtype=bits8.dtype)
        bits = torch.cat([bits8, pad], dim=-1)
    else:
        bits = bits8
    return (2 * bits - 1) / math.sqrt(dq)


def dqbits_to_byte(bits: torch.Tensor) -> torch.Tensor:
    """[*, dq] -> [*] byte ids (0..255) — CROPS to the first 8 bits regardless of dq (session:
    "for sampling bytes, just crop byte 9 and more, let bits valued 0-255 reserved for raw
    bytes"). Any bits beyond the 8th are ignored for byte reconstruction — extra shared-
    representation capacity, not part of a byte's own identity. dq==8: identical to v4's own
    bits_to_byte (no cropping possible/needed, already exactly 8 bits)."""
    b = (bits[..., :8] > 0).long()
    powers = (2 ** torch.arange(8, device=bits.device))
    return (b * powers).sum(-1)


class BitPredictHeadAttn(nn.Module):
    """Predicts `dq` chained bits from a hidden vector via Fetch-style causal
    self-attention over the bit sequence — the exact chain-rule
    factorization of the joint dq-bit distribution (ported from
    qcutelm_vlt11.py via qcute_refine.py). Used for LevelLM's own NTP
    head, in both its unconditioned (PASS 1) and fused (PASS 2) calls.

    REVERTED to v4's original design (session: "for attn, comment current impl, revert to v4") —
    full QKV self-attention + out_proj, a single SHARED head (not per-position), after the session's
    own revamp (concat/per-position/Q-K-only/bos_val_emb — see SUPERSEDED block below) was found to
    REGRESS empirically (`attn_id4_pq` on the revamped head: 3.5659 best_val_bpb, WORSE than the
    original's 3.2067 — docs/kv_contribution.md §16/§17). One deliberate difference from v4's exact
    original, needed to satisfy the session's other standing ask ("can the downsample flag only be
    applied on embeds, h maintains full dim... to support same dim h"): v4's original mixed `h`
    directly into the attention's own INPUT (`x = h_scale*h + shifted + pos`), which forces `h` and
    the bit-embeds to share one dimension — incompatible with decoupling `h` from `d_inner`. Here,
    `h` only enters at the FINAL fetch step (via concat, not add), so attention's own Q/K/V
    machinery runs purely on bit-embeds/position at `d_inner` while `h` stays at full `d_model`
    throughout, same principle already applied to BitPredictHeadSSM."""

    # ============================================================================================
    # SUPERSEDED (session revamp, later found to regress vs. v4's original — kept as a reference
    # block, never called): per-position head (`nn.Linear(2*d_inner, dq)`, einsum-read), CONCAT
    # `h`+attn_out, trainable `bos_val_emb`, Q/K-only attention (no V/out_proj, weighted-sums the
    # RAW bit-embeds). Full writeup: docs/kv_contribution.md §14 (the revamp) and §16/§17 (the
    # regression that prompted this revert).
    #
    # def __init__(self, d_model, dq, n_heads=2, gamma=1.0, fixed_kernel=True, downsample=1):
    #     super().__init__()
    #     d_inner = d_model // downsample
    #     self.head_dim = d_inner // n_heads
    #     self.head = nn.Linear(d_model + d_inner, dq)   # per-position, concat-sized
    #     self.bit_pos_emb = nn.Embedding(dq, d_inner)
    #     self.bit_val_emb = nn.Embedding(2, d_inner)
    #     self.bos_val_emb = nn.Parameter(torch.zeros(d_inner))
    #     self.q_proj = nn.Linear(d_inner, d_inner)
    #     self.k_proj = nn.Linear(d_inner, d_inner)
    #     causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
    #     self.register_buffer("causal_mask", causal_mask, persistent=False)
    #     h_scale = torch.cat([torch.ones(1), torch.full((max(dq - 1, 0),), gamma)]).view(1, dq, 1)
    #     self.register_buffer("h_scale", h_scale, persistent=False)
    #
    # def _mha(self, x, values, attn_mask):   # Q/K-only, weighted-sums RAW values (no V/out_proj)
    #     N, T, D = x.shape
    #     H, hd = self.n_heads, self.head_dim
    #     q = self.q_proj(x).view(N, T, H, hd).transpose(1, 2)
    #     k = self.k_proj(x).view(N, T, H, hd).transpose(1, 2)
    #     v = values.view(N, T, H, hd).transpose(1, 2)
    #     y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    #     return y.transpose(1, 2).reshape(N, T, D)
    #
    # def _forward_fixed(self, h, true_bits):
    #     N, _ = h.shape
    #     bit_ids = (true_bits > 0).long()
    #     val_embeds = self.bit_val_emb(bit_ids)
    #     zero_vec = self.bos_val_emb.view(1, 1, -1).expand(N, 1, -1)
    #     shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)
    #     pos = self.bit_pos_emb.weight.unsqueeze(0)
    #     x = shifted + pos
    #     attn_out = self._mha(x, values=shifted, attn_mask=self.causal_mask)
    #     h_scaled = self.h_scale * h.unsqueeze(1)
    #     fetched = torch.cat([h_scaled, attn_out], dim=-1)
    #     return torch.einsum("njd,jd->nj", fetched, self.head.weight) + self.head.bias.unsqueeze(0)
    #
    # def _forward_loop(self, h, true_bits):
    #     N, _ = h.shape
    #     raw_vecs = [self.bos_val_emb.view(1, -1).expand(N, -1)]
    #     logits_list = []
    #     for j in range(self.dq):
    #         raw = torch.stack(raw_vecs, dim=1)
    #         x = raw + self.bit_pos_emb.weight[:j + 1].unsqueeze(0)
    #         attn_out = self._mha(x, values=raw, attn_mask=None)[:, -1, :]
    #         h_scale_j = 1.0 if j == 0 else self.gamma
    #         fetched = torch.cat([h_scale_j * h, attn_out], dim=-1)
    #         logit_j = F.linear(fetched, self.head.weight[j:j + 1], self.head.bias[j:j + 1]).squeeze(-1)
    #         logits_list.append(logit_j)
    #         if j < self.dq - 1:
    #             bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
    #             raw_vecs.append(self.bit_val_emb(bit_val))
    #     return torch.stack(logits_list, dim=1)
    # ============================================================================================

    def __init__(self, d_model: int, dq: int, n_heads: int = 2, gamma: float = 1.0, fixed_kernel: bool = True, downsample: int = 1, downsample_h: bool = False, per_position_head: bool = True):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        assert d_inner % n_heads == 0, f"inner dim={d_inner} (d_model={d_model}/downsample={downsample}) not divisible by n_heads={n_heads}"
        self.dq = dq
        self.gamma = gamma
        self.fixed_kernel = fixed_kernel
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        # downsample_h (session: "queue more experiments to test this hypothesis, repr loss because
        # of downsample h") — False (default): h stays full d_model, as implemented above/in class
        # docstring. True: restores the ORIGINAL (pre-this-session) behavior — h ALSO projected down
        # to d_inner via in_proj before entering the concat — so the two can be A/B'd directly at the
        # same downsample ratio to isolate whether downsampling h itself (not just the embeds) causes
        # a real quality loss.
        self.downsample_h = downsample_h
        self.in_proj = nn.Linear(d_model, d_inner) if downsample_h and downsample > 1 else None
        # v4 original: full QKV self-attention + out_proj (manual SDPA, not nn.MultiheadAttention —
        # session found the latter's MPS backward produces NaN gradients at d_model=256).
        self.qkv_proj = nn.Linear(d_inner, 3 * d_inner)
        self.out_proj = nn.Linear(d_inner, d_inner)
        self.bit_pos_emb = nn.Embedding(dq, d_inner)
        self.bit_val_emb = nn.Embedding(2, d_inner)
        # per_position_head (session: "for each self attn and ssm, allow indp heads for each
        # timestep different head, on by default") — True (default): dq SEPARATE weight rows, one
        # per bit position (the session's own earlier revamp idea, read via einsum in
        # _forward_fixed, per-row slice in _forward_loop — same pattern BitPredictHeadSSM already
        # uses unconditionally). False: v4's original SINGLE SHARED head, applied identically at
        # every position via broadcasting. Independent of the full-QKV-vs-Q/K-only question (this
        # class always uses full QKV+out_proj now, see class docstring) and of downsample_h above —
        # three orthogonal axes, each toggleable on its own.
        self.per_position_head = per_position_head
        h_dim = d_inner if (downsample_h and downsample > 1) else d_model
        self.head = nn.Linear(h_dim + d_inner, dq if per_position_head else 1)
        causal_mask = torch.triu(torch.full((dq, dq), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        h_scale = torch.cat([torch.ones(1), torch.full((max(dq - 1, 0),), gamma)]).view(1, dq, 1)
        self.register_buffer("h_scale", h_scale, persistent=False)

    def _mha(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """v4 original: full QKV self-attention + out_proj. x: [N, T, d_inner] (bit-embeds/pos
        ONLY — h no longer mixed in here, unlike v4's exact original, so d_inner never needs to
        match h's own dimension). -> [N, T, d_inner]."""
        N, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv_proj(x).reshape(N, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(y.transpose(1, 2).reshape(N, T, D))

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        """h: [N, D] — stays full d_model unless downsample_h=True (see __init__). true_bits:
        [N, dq] float in {-1,+1}-ish (teacher-forcing) or None (greedy chain decode). -> raw_logits
        [N, dq]."""
        if self.in_proj is not None:
            h = self.in_proj(h)
        if self.fixed_kernel and true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, _ = h.shape
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)                      # [N, dq, d_inner]
        zero_vec = val_embeds.new_zeros(N, 1, val_embeds.shape[-1])
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)   # position j holds bit j-1's embed
        pos = self.bit_pos_emb.weight.unsqueeze(0)
        x = shifted + pos                                            # attention input: embeds/pos ONLY
        attn_out = self._mha(x, attn_mask=self.causal_mask)          # [N, dq, d_inner]
        h_scaled = self.h_scale * h.unsqueeze(1)                     # [N, dq, d_model] (broadcast)
        fetched = torch.cat([h_scaled, attn_out], dim=-1)            # CONCAT — supports h at full d_model
        if self.per_position_head:
            # per-position: dq separate weight rows, einsum avoids the [N,dq,dq]-then-diagonal waste
            return torch.einsum("njd,jd->nj", fetched, self.head.weight) + self.head.bias.unsqueeze(0)
        return self.head(fetched).squeeze(-1)                        # shared head, broadcasts over dq

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, _ = h.shape
        chain_vecs = [self.bit_pos_emb.weight[0].unsqueeze(0).expand(N, -1)]   # position 0: pos-embed only, no h/bit content yet
        logits_list = []
        for j in range(self.dq):
            x = torch.stack(chain_vecs, dim=1)
            attn_out = self._mha(x, attn_mask=None)[:, -1, :]
            h_scale_j = 1.0 if j == 0 else self.gamma
            fetched = torch.cat([h_scale_j * h, attn_out], dim=-1)     # CONCAT
            if self.per_position_head:
                logit_j = F.linear(fetched, self.head.weight[j:j + 1], self.head.bias[j:j + 1]).squeeze(-1)
            else:
                logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                chain_vecs.append(self.bit_val_emb(bit_val) + self.bit_pos_emb.weight[j + 1])
        return torch.stack(logits_list, dim=1)


class BitPredictHeadConv(nn.Module):
    """Same chain-rule job as BitPredictHeadAttn, but a causal 1D
    convolution over the bit-embedding sequence instead of self-attention
    (session: "not use self attention but simply series of linears...").
    kernel_size defaults to `dq` (full receptive field over the whole bit
    chain) — cheap and lossless at this scale since dq stays small in
    every current config (this architecture predicts single tokens, not
    joint MTP blocks), and windowing would impose a locality prior that's
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
    fixed-size window": "conv1d" (original) calls nn.Conv1d directly;
    "matmul" (session: "reparam to use nn.linear instead of conv1d")
    flattens the window into one [K*D] vector and applies a plain
    nn.Linear(K*D, D) instead — mathematically the same class of
    computation (fixed window, weights shared across positions), just
    dispatched as a matmul instead of the conv op. Session benchmark
    (scripts/bench_bit_heads.py) found nn.Conv1d has real per-call
    overhead in the sequential decode loop (worst case ~3900x slower than
    a plain independent nn.Linear head at dq=16, vs. "matmul"'s expected
    much-flatter overhead) — kept BOTH as a flag rather than replacing,
    matching the rest of this file's convention.

    "depthwise" (session: "make bitpredictconv more efficient... maybe
    try group conv or depthwise") — both "conv1d" and "matmul" are FULLY
    DENSE across channels: every one of the `d_inner` output channels
    reads from every one of the `d_inner` input channels at every one of
    the `K` window positions, costing `K*d_inner^2` params/FLOPs — the
    actual "huge compute" (at `d_inner=256,K=dq=8`: 524,544 params,
    ~8.4M FLOPs/example in `_forward_fixed`). "depthwise" gives each
    channel its OWN private `K`-tap filter (no cross-channel mixing at
    all — output channel c depends only on input channel c's own K-window
    history), matching `nn.Conv1d(..., groups=d_inner)`'s classic
    depthwise-separable-conv structure, but implemented as a plain einsum
    (not `nn.Conv1d`) to keep the same loop-overhead-avoidance property
    "matmul" was built for. Params/FLOPs drop by a full factor of
    `d_inner` (256x at this scale: 2,304 params, ~32K FLOPs) — the
    honest cost is losing all cross-channel mixing within the window read
    (channel c's bit-history influence on channel c' happens only
    indirectly, via `self.head`'s own read of the concatenated output,
    not within the window-conv step itself). A true intermediate (grouped
    conv, `1<groups<d_inner`) is a documented future generalization, not
    implemented here — depthwise is the extreme, simplest, and cheapest
    point on that spectrum."""

    def __init__(self, d_model: int, dq: int, kernel_size: int | None = None, gamma: float = 1.0, conv_impl: str = "matmul", downsample: int = 1):
        super().__init__()
        assert conv_impl in ("conv1d", "matmul", "depthwise")
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
        elif conv_impl == "matmul":
            self.proj = nn.Linear(self.kernel_size * d_inner, d_inner, bias=True)
        else:   # depthwise
            self.dw_weight = nn.Parameter(torch.empty(d_inner, self.kernel_size))
            nn.init.kaiming_uniform_(self.dw_weight, a=5 ** 0.5)   # matches nn.Conv1d's own default init scheme
            self.dw_bias = nn.Parameter(torch.zeros(d_inner))

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _window_read(self, x_windows: torch.Tensor) -> torch.Tensor:
        """x_windows: [N, D, dq, K] (oldest->newest along last dim) ->
        [N, dq, D]. Dispatches on conv_impl."""
        if self.conv_impl == "conv1d":
            N, D, dq, K = x_windows.shape
            x_t = x_windows.permute(0, 2, 1, 3).reshape(N * dq, D, K)
            return self.conv(x_t).squeeze(-1).reshape(N, dq, D)
        if self.conv_impl == "depthwise":
            # per-channel K-tap filter, no cross-channel mixing — einsum keeps this loop-overhead-
            # free like "matmul" (never touches nn.Conv1d, matching that impl's own rationale).
            return torch.einsum("ndjk,dk->njd", x_windows, self.dw_weight) + self.dw_bias.view(1, 1, -1)
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
            elif self.conv_impl == "depthwise":
                x = torch.stack(seq, dim=2)              # [N, D, kernel_size]
                conv_out = torch.einsum("ndk,dk->nd", x, self.dw_weight) + self.dw_bias.view(1, -1)
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


class BitPredictHeadConvDilated(nn.Module):
    """Dilated depthwise-separable causal conv STACK (WaveNet-style
    receptive-field-doubling) — session: "do this stacked small kernel,
    then check memory usage vs single large conv." PURELY LINEAR, no
    activation between the stacked layers (session: "i mean for memory
    and param save even though linear") — composing linear filters stays
    linear, so this is representationally a SUBSET of what
    BitPredictHeadConv's single full-width kernel can express (a real
    expressivity cost, not a free win — see that class's own docstring
    for the full analysis); this class exists purely to test the
    params/FLOPs side of the tradeoff, independent of expressivity.

    `dilation_base` (default 2): layer `l` (0-indexed) has kernel_size=
    `dilation_base` and dilation=`dilation_base**l`. Layers stack until
    the cumulative receptive field reaches `dq` — `L=ceil(log_b(dq))`
    layers, `L*dilation_base` total taps/channel vs. a single K=dq
    layer's `dq` taps (real savings whenever `dq` isn't already tiny —
    see class-level session analysis in docs/kv_contribution.md §17).

    `mode` (new): "depthwise" (default — per-channel, no cross-channel
    mixing, as described above) or "dense" (session: "finish impl conv
    dilated dense" — full cross-channel mixing at every layer, the
    dilated-stack analogue of `BitPredictHeadConv`'s own dense "matmul"
    impl, just split across `L` small-kernel layers instead of one big
    `K=dq` kernel). Params scale `L*d_inner^2*dilation_base` (dense) vs.
    `L*d_inner*dilation_base` (depthwise) — dense is still cheaper than a
    single big dense kernel (`L*dilation_base` "tap-layers" vs. `dq`, same
    ratio as depthwise's own savings) but nowhere near depthwise's own
    per-channel reduction; it exists to test whether cross-channel mixing
    (lost entirely in "depthwise") is worth restoring at dilated-stack
    scale specifically.

    `_forward_loop` (session: "check ar gen conv code, then train this" — generation support added
    after an initial training-only version) reuses the SAME `_dilated_stack` helper `_forward_fixed`
    does, called on the growing bit-history each step (recomputed from scratch every step, no
    WaveNet-style FIFO cache) — simple and correct at this `dq` scale, the same tradeoff
    `BitPredictHeadConv`'s own `_forward_loop` already makes. Verified fixed/loop-consistent
    (`torch.allclose`, exact/near-exact match) for both `mode`s, plus a standalone greedy-decode
    smoke test and downsample sanity check.

    Uses plain tensors (`unfold`+`einsum`), never `nn.Conv1d`, in BOTH forward paths — session
    diagnostic ("not dilated depthwise, but dilated full dense kernel") isolated `nn.Conv1d` itself
    as the cause of an earlier ~300x slowdown vs. single-layer `depthwise` (298ms/fwd vs. 0.96ms),
    independent of grouping (confirmed by testing `groups=1`/dense at each layer too — 14x faster
    than `groups=d_inner`, but still ~22x slower than a pure-tensor-op single depthwise layer) — the
    same issue `BitPredictHeadConv`'s own "matmul"/"depthwise" impls were already built to avoid.
    `mode="dense"`'s own `_dilated_stack` einsum verified to exactly reproduce a real
    `nn.Conv1d(groups=1)` stack (weights copied over, `torch.allclose` exact) — confirms the einsum
    correctly implements standard dense dilated-conv semantics, just without `nn.Conv1d`'s overhead."""

    def __init__(self, d_model: int, dq: int, dilation_base: int = 2, gamma: float = 1.0, downsample: int = 1, mode: str = "depthwise"):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        assert dilation_base >= 2
        assert mode in ("depthwise", "dense")
        d_inner = d_model // downsample
        self.dq = dq
        self.gamma = gamma
        self.dilation_base = dilation_base
        self.mode = mode
        self.in_proj = nn.Linear(d_model, d_inner) if downsample > 1 else None
        self.head = nn.Linear(d_inner, 1)
        self.bit_val_emb = nn.Embedding(2, d_inner)

        dilations, rf, l = [], 1, 0
        while rf < dq:
            dilations.append(dilation_base ** l)
            rf += (dilation_base - 1) * dilation_base ** l
            l += 1
        self.dilations = dilations
        self.receptive_field = rf   # >= dq (may overshoot on the last layer)
        # PLAIN TENSORS, not nn.Conv1d — see class docstring. "depthwise": one [d_inner,
        # dilation_base] weight per layer (no cross-channel mixing). "dense": one
        # [d_inner_out, d_inner_in, dilation_base] weight per layer (full cross-channel mixing,
        # the dilated-stack analogue of BitPredictHeadConv's own dense "matmul" impl).
        if mode == "depthwise":
            self.dw_weights = nn.ParameterList([
                nn.Parameter(torch.empty(d_inner, dilation_base)) for _ in dilations
            ])
        else:
            self.dw_weights = nn.ParameterList([
                nn.Parameter(torch.empty(d_inner, d_inner, dilation_base)) for _ in dilations
            ])
        for w in self.dw_weights:
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.dw_biases = nn.ParameterList([nn.Parameter(torch.zeros(d_inner)) for _ in dilations])

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _dilated_stack(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, D, T] (any T, causally padded internally per layer) -> [N, D, T] — the shared
        multi-layer dilated read used by both _forward_fixed (T=dq, one call) and _forward_loop
        (T=j+1, recomputed from scratch each step j — simple and correct at this dq scale, same
        "just recompute the window read on the growing history" tradeoff BitPredictHeadConv's own
        _forward_loop already makes, no WaveNet-style FIFO cache needed)."""
        for weight, bias, dilation in zip(self.dw_weights, self.dw_biases, self.dilations):
            pad = dilation * (self.dilation_base - 1)                    # causal: left-pad only
            x_padded = F.pad(x, (pad, 0))
            span = pad + 1                                                # = dilation*(K-1)+1
            windows = x_padded.unfold(2, span, 1)                         # [N, D_in, T, span]
            windows = windows[..., ::dilation]                            # [N, D_in, T, K] — dilated taps
            if self.mode == "depthwise":
                x = torch.einsum("ndtk,dk->ndt", windows, weight) + bias.view(1, -1, 1)
            else:   # dense — full cross-channel mixing, weight: [D_out, D_in, K]
                x = torch.einsum("nctk,dck->ndt", windows, weight) + bias.view(1, -1, 1)
            # output length stays T either way
        return x

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        N, D = h.shape
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)                          # [N, dq, D]
        zero_vec = val_embeds.new_zeros(N, 1, D)
        shifted = torch.cat([zero_vec, val_embeds[:, :-1, :]], dim=1)   # position j holds bit j-1's embed
        x = self._dilated_stack(shifted.transpose(1, 2))                # [N, D, dq]
        conv_out = x.transpose(1, 2)                                     # [N, dq, D]
        h_scale = h.new_ones(1, self.dq, 1)
        if self.dq > 1:
            h_scale = torch.cat([h_scale[:, :1, :], h_scale[:, 1:, :] * self.gamma], dim=1)
        fetched = h_scale * h.unsqueeze(1) + conv_out
        return self.head(fetched).squeeze(-1)

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N, D = h.shape
        past: list[torch.Tensor] = []   # decided bits' raw embeddings so far, oldest first
        logits_list = []
        for j in range(self.dq):
            seq = past if past else [h.new_zeros(N, D)]   # position 0: single zero "no history" step
            x = torch.stack(seq, dim=2)                     # [N, D, len(seq)]
            conv_out = self._dilated_stack(x)[:, :, -1]      # [N, D] — last position's own output
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

    Per-position head (session: "let each bit timestep use different head... similar to independent
    mode, but has state") — `self.head` is `nn.Linear(2*d_inner, dq)`, dq SEPARATE weight rows, one
    dedicated to each bit position, instead of one `nn.Linear(d_inner, 1)` every position used to
    share. Position j reads its own row's logit via `einsum("njd,jd->nj", fetched, self.head.
    weight)` (session: "use einsum" — replaces an earlier version that computed the full `[N,dq,dq]`
    matrix via `self.head(fetched)` then `torch.diagonal`'d it, an `n`x compute/memory waste
    computing off-diagonal entries never used; `_forward_loop`'s own `self.head.weight[j:j+1]` slice
    was already this cheap). Same "each bit gets its own weights" structure `code_head_mode=
    "independent"` already has, but UNLIKE independent mode, `state_contrib` (the decayed cumulative
    sum below) still carries genuine cross-bit information forward — the two axes (shared-vs-private
    WEIGHTS, stateless-vs-stateful CONDITIONING) are orthogonal, and this class sits at "private
    weights, stateful conditioning," a point the codebase didn't have before.

    CONCAT, not add (session: "make concat mode default... h_t always concat not add with current
    embed") — `fetched` used to be `h_scale*h + state_contrib` (summed, same dim throughout); now
    it's `torch.cat([h_scale*h, state_contrib], dim=-1)` — the head reads `h` and the recurrent
    state as two SEPARATE halves of a `2*d_inner`-dim vector instead of one pre-summed `d_inner`-dim
    blend, giving it strictly more information (summing is a lossy special case concat can still
    learn to approximate, never the other way around). `self.head`'s input dim doubles accordingly.

    Trainable BOS state (session: "consider a trainable bos token init zero at dq 0") — position 0
    has no preceding bit, so its `state_contrib` used to just be `state_proj(zeros)` (whatever
    `state_proj`'s own bias happens to be, an accident of that layer's init, not a deliberately
    chosen "no state yet" representation). Now `self.bos_state` (`nn.Parameter`, zero-initialized)
    stands in for `state_contrib` at position 0 directly — starts equivalent to the old zero-state
    behavior but free to move away from it during training, decoupled from `state_proj`'s own
    weights.

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

    def __init__(self, d_model: int, dq: int, d_state: int | None = None, gamma: float = 1.0, downsample: int = 1, downsample_h: bool = False, per_position_head: bool = True):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        self.dq = dq
        self.gamma = gamma
        # d_state's own None-default now tracks d_inner, not d_model — downsampling shrinks the default
        # recurrent state width too, consistent with the other two heads' in_proj. Still independently
        # overridable via d_state, same as before.
        self.d_state = d_state if d_state is not None else d_inner
        # downsample_h (session: "queue more experiments to test this hypothesis, repr loss because
        # of downsample h") — False (default): h stays full d_model throughout, no in_proj (session:
        # "can the downsample flag only be applied on embeds, h maintains full dim"). True: restores
        # the ORIGINAL (pre-this-session) behavior — h ALSO projected down to d_inner via in_proj —
        # so the two can be A/B'd at the same downsample ratio to isolate whether downsampling h
        # itself (not just the state machinery) causes a real quality loss. Feasible here (unlike
        # BitPredictHeadHSoftmax) because h only ever reaches this class via CONCAT below, which
        # doesn't require matching dims — see BitPredictHeadAttn's own docstring for the full
        # reasoning (same change applied there).
        self.downsample_h = downsample_h
        self.in_proj = nn.Linear(d_model, d_inner) if downsample_h and downsample > 1 else None
        # per_position_head (session: "for each self attn and ssm, allow indp heads for each
        # timestep different head, on by default") — True (default): dq SEPARATE weight rows, one
        # per bit position (session: "let each bit timestep use different head... h_t always concat
        # not add with current embed"), read via einsum in _forward_fixed / per-row slice in
        # _forward_loop. False: a single SHARED head instead (v4's original design), applied
        # identically at every position via broadcasting.
        self.per_position_head = per_position_head
        h_dim = d_inner if (downsample_h and downsample > 1) else d_model
        self.head = nn.Linear(h_dim + d_inner, dq if per_position_head else 1)
        self.bit_val_emb = nn.Embedding(2, self.d_state)
        self.state_proj = nn.Linear(self.d_state, d_inner)
        # trainable BOS state (session: "consider a trainable bos token init zero at dq 0") —
        # stands in for state_contrib at position 0 (no preceding bit exists yet); zero-init so
        # training starts equivalent to the old "state_proj(zeros)" behavior, free to move from it.
        self.bos_state = nn.Parameter(torch.zeros(d_inner))
        self.decay_logit = nn.Parameter(torch.zeros(self.d_state))   # sigmoid(0)=0.5 init
        # _forward_fixed's own decay-exponent grid/validity mask: pure functions of dq (unlike
        # `decay` itself, which depends on `alpha` — a LEARNED param that changes every step, so
        # THAT part can't be precomputed) — rebuilt via arange/outer-subtract on every single
        # forward call before this fix, wastefully, since dq never changes after construction.
        idx = torch.arange(dq)
        offsets = idx.unsqueeze(1) - 1 - idx.unsqueeze(0)             # [dq, dq]: j-1-k
        self.register_buffer("_valid", (offsets >= 0), persistent=False)
        self.register_buffer("_offsets_clamped", offsets.clamp(min=0).float(), persistent=False)
        h_scale = torch.cat([torch.ones(1), torch.full((max(dq - 1, 0),), gamma)]).view(1, dq, 1)
        self.register_buffer("h_scale", h_scale, persistent=False)

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)   # [d_state], in (0,1) — see class docstring re: alpha=0

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        bit_ids = (true_bits > 0).long()
        val_embeds = self.bit_val_emb(bit_ids)                    # [N, dq, d_state]
        alpha = self._alpha()                                      # [d_state]

        decay = (alpha.view(1, 1, -1) ** self._offsets_clamped.unsqueeze(-1)) * self._valid.unsqueeze(-1).float()   # [dq,dq,d_state]

        s = torch.einsum("jkc,nkc->njc", decay, val_embeds)         # [N, dq, d_state]
        state_contrib = self.state_proj(s)                          # [N, dq, d_inner]
        # position 0 gets the trainable BOS state instead of state_proj(zeros) — s[:,0,:] is
        # already all-zero (the _valid mask has no k<0 terms for j=0), so this is a pure override,
        # not a correction of a nonzero leak.
        bos = self.bos_state.view(1, 1, -1).expand(state_contrib.shape[0], 1, -1)
        state_contrib = torch.cat([bos, state_contrib[:, 1:, :]], dim=1)
        h_scaled = self.h_scale * h.unsqueeze(1)                     # [N, dq, D] (broadcast; D = d_model or d_inner)
        fetched = torch.cat([h_scaled, state_contrib], dim=-1)       # [N, dq, D+d_inner] — CONCAT, not add
        if self.per_position_head:
            # einsum, not self.head(fetched)+diagonal: position j only ever needs its OWN weight row
            # (self.head.weight[j]) dotted with its OWN fetched vector — computing the full [N,dq,dq]
            # matrix and discarding every off-diagonal entry wasted an `n`x factor of compute/memory
            # (session: "use einsum").
            return torch.einsum("njd,jd->nj", fetched, self.head.weight) + self.head.bias.unsqueeze(0)
        return self.head(fetched).squeeze(-1)                        # shared head, broadcasts over dq

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N = h.shape[0]
        alpha = self._alpha()
        s = h.new_zeros(N, self.d_state)
        logits_list = []
        for j in range(self.dq):
            state_contrib = self.bos_state.view(1, -1).expand(N, -1) if j == 0 else self.state_proj(s)
            h_scale_j = 1.0 if j == 0 else self.gamma
            fetched = torch.cat([h_scale_j * h, state_contrib], dim=-1)   # [N, D+d_inner] — CONCAT
            if self.per_position_head:
                # position j's own dedicated weight row (self.head.weight[j])/bias — already
                # O(d_inner) per step, no diagonal-select waste to fix here (only _forward_fixed had that).
                logit_j = F.linear(fetched, self.head.weight[j:j + 1], self.head.bias[j:j + 1]).squeeze(-1)
            else:
                logit_j = self.head(fetched).squeeze(-1)
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                s = alpha * s + self.bit_val_emb(bit_val)
        return torch.stack(logits_list, dim=1)


class BitPredictHeadWordPredict(nn.Module):
    """Decomposes the dq-bit code into a chain of WORDS (`word_bits` bits each, a genuine
    `2**word_bits`-way softmax per word) instead of individual bits — session: "design another
    head, wordpredict, which decompose to word like 8 bit, 4 bit, useful for dq more than 8." A
    middle ground between `BitPredictHeadHSoftmax`'s per-BIT binary tree (`dq` sequential steps,
    each a cheap 2-way decision, `O(V)` params) and a single flat `V=2**dq`-way softmax (1 step,
    `O(V)` params AND FLOPs) — `n_words=dq//word_bits` sequential steps, each a genuine
    `2**word_bits`-way softmax classifier with NO position-sharing bottleneck within a word
    (unlike attn/conv/ssm's own per-bit-position bottleneck — every outcome within a word gets its
    own row of the word classifier, exactly like a small dense softmax).

    Returns a LIST of `n_words` per-WORD logit tensors `[N, 2**word_bits]` each — a genuinely
    different shape than every other BitPredictHead* (`[N, dq]` per-bit logits for
    `chain_bce_loss`), since word classification is multi-class cross-entropy per word, not BCE.
    Wired into the pipeline via `code_head_mode="word"` (new), with its own loss (`self.loss`,
    below) and its own `LevelLM.forward`/generation dispatch — see those call sites.

    Conditioning (session: "past chain prob conditioning make simpler but more expensive" — plain
    concatenation of ALL previous words' own embeddings, growing linearly, instead of a
    fixed-size recurrent/attention state): word `i`'s classifier reads `cat([h, embed(word_0), ...,
    embed(word_{i-1})])`, dimension `d_model + i*d_embed` — genuinely SIMPLER than attn/ssm (no
    recurrence/attention machinery at all, just concatenation) but genuinely MORE EXPENSIVE per
    step (each word's own classifier weight matrix is strictly larger than the last).

    `_forward_fixed` (session: "find way to parallel launch kernel, maybe pad" — teacher-forcing
    means every word is already known upfront, so all `n_words` steps' own context vectors can be
    built in parallel instead of sequentially) PADS every word's context to the same `max_dim` (the
    LAST word's own width — the largest), zero-filling the unused tail, and reads all `n_words`
    classifiers via ONE batched einsum against a single `[n_words, word_vocab, max_dim]` weight
    tensor (padding columns naturally contribute 0 since their input is 0 — no separate masking
    needed) — one kernel launch instead of `n_words` separate `nn.Linear` calls. `_forward_loop`
    (autoregressive/generation) stays genuinely sequential (future words depend on GREEDILY decided
    previous ones, can't be precomputed), reusing the same weight tensor via a sliced `F.linear`.

    `word_bits` must evenly divide `dq`. `word_bits==dq` (`n_words=1`) DEGENERATES to a single flat
    `V=2**dq`-way softmax classifier — no concat/embedding machinery at all, structurally IDENTICAL
    to `byte_head_256way`/`byte_softmax_head_only`'s own plain `nn.Linear(D,vocab)` (verified via
    direct numerical comparison) — the "compatibility" the session asked for.

    `d_embed`: explicit override for each previous word's embedding width. If not given, derived
    from `embed_downsample` (session: "if d_embed not given use embed_downsample param to down
    sample x vs d") — `d_embed = d_model // embed_downsample`, matching every other
    BitPredictHead*'s own downsample convention."""

    def __init__(self, d_model: int, dq: int, word_bits: int = 8, d_embed: int | None = None, embed_downsample: int = 1):
        super().__init__()
        assert dq % word_bits == 0, f"word_bits={word_bits} must evenly divide dq={dq}"
        self.dq = dq
        self.d_model = d_model
        self.word_bits = word_bits
        self.n_words = dq // word_bits
        self.word_vocab = 2 ** word_bits
        if d_embed is None:
            assert d_model % embed_downsample == 0, f"d_model={d_model} not divisible by embed_downsample={embed_downsample}"
            d_embed = d_model // embed_downsample
        self.d_embed = d_embed
        # max_dim: the LAST word's own context width (d_model + (n_words-1)*d_embed) — every
        # earlier word's own weight slice is padded up to this with dead (never-trained, since
        # their input is always 0) columns, letting all n_words be read via one batched einsum.
        self.max_dim = d_model + (self.n_words - 1) * d_embed
        self.word_weight = nn.Parameter(torch.empty(self.n_words, self.word_vocab, self.max_dim))
        self.word_bias = nn.Parameter(torch.zeros(self.n_words, self.word_vocab))
        nn.init.kaiming_uniform_(self.word_weight, a=5 ** 0.5)
        self.word_embed = None if self.n_words == 1 else nn.Embedding(self.word_vocab, d_embed)
        # LSB-first within a word, matching dqbits_to_byte's own convention (powers = 2**arange(K)).
        powers = 2 ** torch.arange(word_bits)
        self.register_buffer("_word_powers", powers, persistent=False)

    def _bits_to_words(self, true_bits: torch.Tensor) -> torch.Tensor:
        """[N, dq] float in {-1,+1}-ish -> [N, n_words] long word indices in [0, word_vocab)."""
        N = true_bits.shape[0]
        bit_ids = (true_bits > 0).long().view(N, self.n_words, self.word_bits)
        return (bit_ids * self._word_powers).sum(-1)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> list[torch.Tensor]:
        """h: [N, d_model]. true_bits: [N, dq] (teacher-forcing) or None (greedy). -> list of
        n_words tensors, each [N, 2**word_bits] — per-word logits, NOT per-bit [N, dq]."""
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> list[torch.Tensor]:
        N = h.shape[0]
        if self.n_words == 1:
            return [F.linear(h, self.word_weight[0, :, :self.d_model], self.word_bias[0])]
        word_ints = self._bits_to_words(true_bits)              # [N, n_words]
        word_embeds = self.word_embed(word_ints)                 # [N, n_words, d_embed]
        ctx_all = h.new_zeros(N, self.n_words, self.max_dim)
        ctx_all[:, :, :self.d_model] = h.unsqueeze(1)
        for i in range(1, self.n_words):
            # word i's context needs embeds of words 0..i-1 — a python loop here only builds the
            # padded CONTEXT tensor (cheap indexing/concat), not the expensive classifier matmul
            # itself, which happens once, batched, below.
            flat = word_embeds[:, :i, :].reshape(N, i * self.d_embed)
            ctx_all[:, i, self.d_model:self.d_model + i * self.d_embed] = flat
        logits_all = torch.einsum("nid,ivd->niv", ctx_all, self.word_weight) + self.word_bias.unsqueeze(0)
        return [logits_all[:, i, :] for i in range(self.n_words)]

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> list[torch.Tensor]:
        N = h.shape[0]
        word_ints_true = self._bits_to_words(true_bits) if true_bits is not None else None
        logits_list = []
        ctx = h
        for i in range(self.n_words):
            active_dim = self.d_model + i * self.d_embed
            logits_i = F.linear(ctx, self.word_weight[i, :, :active_dim], self.word_bias[i])
            logits_list.append(logits_i)
            if i < self.n_words - 1:
                word_i = word_ints_true[:, i] if word_ints_true is not None else logits_i.argmax(-1)
                ctx = torch.cat([ctx, self.word_embed(word_i)], dim=-1)
        return logits_list

    def loss(self, logits_list: list[torch.Tensor], true_bits: torch.Tensor) -> torch.Tensor:
        """Sum of per-word cross-entropies, matching chain_bce_loss's own "sum over units, mean
        over batch" convention."""
        word_ints = self._bits_to_words(true_bits)   # [N, n_words]
        per_word = torch.stack([
            F.cross_entropy(logits_list[i], word_ints[:, i], reduction="none")
            for i in range(self.n_words)
        ], dim=1)   # [N, n_words]
        return per_word.sum(-1).mean()

    def logits_to_word_ints(self, logits_list: list[torch.Tensor]) -> torch.Tensor:
        """argmax each word's own logits -> [N, n_words] long word indices (for generation/eval
        accuracy, mirroring how every other head's raw_logits get argmax'd)."""
        return torch.stack([logits_list[i].argmax(-1) for i in range(self.n_words)], dim=1)

    def word_ints_to_bits(self, word_ints: torch.Tensor) -> torch.Tensor:
        """[N, n_words] long -> [N, dq] float in {-1,+1} — inverse of _bits_to_words, LSB-first
        within each word, for handing off to the same dq-bit representation every other head uses
        (byte reconstruction / code hand-off to the level above)."""
        bits = ((word_ints.unsqueeze(-1) >> torch.arange(self.word_bits, device=word_ints.device)) & 1)
        return (2 * bits.reshape(word_ints.shape[0], self.dq).float() - 1)


class BitPredictHeadHSoftmax(nn.Module):
    """Classic hierarchical softmax (Morin & Bengio 2005) over the same
    dq-depth binary tree every other BitPredictHead* factorizes — session:
    "find something that satisfy chain probs validity and cheap and same
    repr power as large softmax head." Diagnosed problem with attn/conv/ssm
    (even pre-revamp): each of the dq bit POSITIONS gets exactly one
    classifying direction, SHARED across every one of the 2**j prefixes
    that can reach position j — the state/attention machinery has to
    funnel "which prefix am I" through a single fixed hyperplane at each
    depth, a severe bottleneck softmax-256 (256 independently-oriented
    hyperplanes, one per OUTCOME) never has.

    Fix: give every one of the `2**dq - 1` tree NODES (not positions) its
    own private weight vector, addressed by a running node index that
    descends the tree as bits are decided (standard binary-heap indexing:
    root=0, left child=2n+1, right child=2n+2). `p(byte|h) =
    prod_j p(bit_j | node_idx_j, h)`, each factor a genuine Bernoulli via
    `sigmoid(h @ node_weight[node_idx_j] + node_bias[node_idx_j])` — same
    chain-rule validity proof as every other BitPredictHead*, but now
    `node_weight` has `(2**dq-1)*D` params (~= softmax-256's own `256*D`,
    same order of degrees of freedom) while only the dq=8 nodes on each
    example's own true path are ever gathered/read, so FLOPs/example stay
    at `dq*D` — the same class `code_head_mode="independent"`'s plain
    `nn.Linear(D,dq)` costs, NOT dense softmax-256's `256*D`.

    No state/attention/decay machinery at all (deliberately) — unlike
    attn/conv/ssm, the "which prefix" signal lives entirely in WHICH
    node_weight row gets read, not in a recurrent state blended with `h`,
    so there's nothing to accumulate and no h_scale/gamma decay needed:
    `h` reaches every position at full, undiminished strength, always."""

    def __init__(self, d_model: int, dq: int, downsample: int = 1):
        super().__init__()
        assert d_model % downsample == 0, f"d_model={d_model} not divisible by downsample={downsample}"
        d_inner = d_model // downsample
        self.dq = dq
        self.n_nodes = 2 ** dq - 1
        self.in_proj = nn.Linear(d_model, d_inner) if downsample > 1 else None
        self.node_weight = nn.Embedding(self.n_nodes, d_inner)
        self.node_bias = nn.Parameter(torch.zeros(self.n_nodes))
        # _forward_fixed's own prefix->node-index machinery: node_idx_j = (2**j - 1) + prefix_int_j,
        # prefix_int_j = sum_{k<j} bit_k * 2**(j-1-k) (bit_0 is the prefix's MSB) — derivable in
        # closed form from the same 2*n+1+bit recursion _forward_loop uses step-by-step, but here
        # precomputed as one [dq,dq] lower-triangular power matrix so the whole batch's node indices
        # for EVERY position come from a single matmul, no python-level sequential loop needed.
        P = torch.zeros(dq, dq)
        for j in range(1, dq):
            for k in range(j):
                P[j, k] = 2 ** (j - 1 - k)
        self.register_buffer("_prefix_powers", P, persistent=False)   # [dq, dq]
        level_offset = torch.tensor([2 ** j - 1 for j in range(dq)], dtype=torch.long)
        self.register_buffer("_level_offset", level_offset, persistent=False)   # [dq]

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_proj is not None:
            h = self.in_proj(h)
        if true_bits is not None:
            return self._forward_fixed(h, true_bits)
        return self._forward_loop(h, true_bits)

    def _forward_fixed(self, h: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
        bit_ids = (true_bits > 0).float()                                  # [N, dq]
        prefix_int = (bit_ids @ self._prefix_powers.T).long()               # [N, dq]
        node_idx = self._level_offset.unsqueeze(0) + prefix_int             # [N, dq], in [0, n_nodes)
        node_w = self.node_weight(node_idx)                                 # [N, dq, D]
        node_b = self.node_bias[node_idx]                                   # [N, dq]
        return torch.einsum("nd,njd->nj", h, node_w) + node_b

    def _forward_loop(self, h: torch.Tensor, true_bits: torch.Tensor | None) -> torch.Tensor:
        N = h.shape[0]
        node_idx = h.new_zeros(N, dtype=torch.long)   # root
        logits_list = []
        for j in range(self.dq):
            node_w = self.node_weight(node_idx)          # [N, D]
            node_b = self.node_bias[node_idx]             # [N]
            logit_j = (h * node_w).sum(-1) + node_b
            logits_list.append(logit_j)
            if j < self.dq - 1:
                bit_val = (true_bits[:, j] > 0).long() if true_bits is not None else (logit_j > 0).long()
                node_idx = 2 * node_idx + 1 + bit_val
        return torch.stack(logits_list, dim=1)


def _balanced_factors(V: int) -> tuple[int, int]:
    """Largest v1<=sqrt(V) that evenly divides V, paired with v2=V//v1 — the
    params/FLOPs-minimizing balanced split for FactoredSoftmaxHead (fixed
    D, minimizing v1+v2 subject to v1*v2==V is achieved at the split
    closest to sqrt(V), by AM-GM)."""
    v1 = int(V ** 0.5)
    while v1 > 1 and V % v1 != 0:
        v1 -= 1
    return v1, V // v1


class FactoredSoftmaxHead(nn.Module):
    """Kronecker-factored (2-stage) replacement for a dense `nn.Linear(D,
    V)` classifier — session: "use structured matrix but to replace dense
    linear map to 2**n way output softmax... some loss in repr ok for
    params saving." Computes the V-way logit vector as an OUTER SUM of two
    small projections instead of one dense D->V matrix:

        logits[i,j] = f1(h)[i] + f2(h)[j]      (i in [0,v1), j in [0,v2))

    reshaped to [V]. This is trivially a valid softmax classifier — it's
    an ordinary softmax over a STRUCTURED logit vector, no chain-rule/
    teacher-forcing machinery needed at all (unlike BitPredictHeadHSoftmax
    or any BitPredictHead* — this head is parallel/one-shot, not
    autoregressive over bits). Params drop from `V*D` to `D*(v1+v2)`,
    minimized (by AM-GM) at the balanced split v1=v2=sqrt(V) — an ~8x
    reduction at V=256,D=256 (v1=v2=16): `256*256=65,536` -> `256*32=
    8,192`. FLOPs drop correspondingly (`2*D*V` -> `2*D*(v1+v2)+V`, since
    the two small matmuls dominate and the outer-sum broadcast itself is
    just `V` additions, no multiplies).

    Cost: the reshaped [v1,v2] logit matrix is constrained to row-effect
    + column-effect (additively separable) — it CANNOT express a genuine
    "outcome i needs outcome j specifically" interaction term a dense
    D->V map could. A real, honest representational cut, not free lunch —
    this is the shallowest (2-stage) member of the same structured-matrix
    family real butterfly/Monarch matrices generalize to log(V)/sqrt(V)
    stages; deeper factorizations would push params/FLOPs down further at
    real implementation-complexity cost, not attempted here."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        v1, v2 = _balanced_factors(vocab)
        self.v1, self.v2, self.vocab = v1, v2, vocab
        self.f1 = nn.Linear(d_model, v1)
        self.f2 = nn.Linear(d_model, v2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., D] -> logits [..., vocab]."""
        l1 = self.f1(h)   # [..., v1]
        l2 = self.f2(h)   # [..., v2]
        logits = l1.unsqueeze(-1) + l2.unsqueeze(-2)   # [..., v1, v2]
        return logits.reshape(*h.shape[:-1], self.vocab)


class LowRankSoftmaxHead(nn.Module):
    """The classic "softmax bottleneck" (Yang et al. 2018) — a rank-r
    approximation of a dense `nn.Linear(D,V)`, via `h -> Linear(D,r) ->
    Linear(r,V)`. Session ask: "how good is factoredsoftmax vs just low
    rank... analyze rank." Unlike FactoredSoftmaxHead's outer-sum (which
    additionally forces every class's direction into a rigid, zero-free-
    parameter `w1_i+w2_j` additive template), this keeps a genuine free
    r-dim coefficient vector PER CLASS (a full row of the r->V matrix) —
    strictly more expressive than FactoredSoftmaxHead at matched rank/
    param budget (every outer-sum-representable logit matrix is also a
    rank-(v1+v2) low-rank-representable one, but not vice versa). The
    honest cost is exactly the rank-r bottleneck the cited paper studies:
    classes are restricted to a shared r-dim subspace of R^D, whatever
    else is free within it."""

    def __init__(self, d_model: int, vocab: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, vocab)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(h))


def chain_bce_loss(raw_logits: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
    """Sum over the bit dim (nats per predicted unit), then mean over
    everything else — matches qcutelm_vlt11/qcute_refine's own convention."""
    return F.binary_cross_entropy_with_logits(raw_logits, (true_bits > 0).float(), reduction="none").sum(-1).mean()


def build_bit_head(cfg: Config, d_model: int, dq: int) -> nn.Module:
    """Dispatches to whichever BitPredictHead* implementation
    Config.bit_head_class selects — the single place every "chain"-style
    head (byte level's own bits-mode head, and any level's
    code_head_mode=="chain" head) gets built, so switching architectures
    is one flag, not per-call-site edits."""
    if cfg.bit_head_class == "attn":
        return BitPredictHeadAttn(d_model, dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel, downsample=cfg.bit_inner_downsample, downsample_h=cfg.bit_downsample_h, per_position_head=cfg.bit_per_position_head)
    elif cfg.bit_head_class == "conv":
        return BitPredictHeadConv(d_model, dq, kernel_size=cfg.bit_conv_kernel_size, gamma=cfg.bit_chain_gamma, conv_impl=cfg.bit_conv_impl, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "ssm":
        return BitPredictHeadSSM(d_model, dq, d_state=cfg.bit_ssm_d_state, gamma=cfg.bit_chain_gamma, downsample=cfg.bit_inner_downsample, downsample_h=cfg.bit_downsample_h, per_position_head=cfg.bit_per_position_head)
    elif cfg.bit_head_class == "hsoftmax":
        return BitPredictHeadHSoftmax(d_model, dq, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "conv_dilated":
        return BitPredictHeadConvDilated(d_model, dq, dilation_base=cfg.conv_dilated_base, gamma=cfg.bit_chain_gamma, downsample=cfg.bit_inner_downsample, mode=cfg.conv_dilated_mode)
    else:
        raise ValueError(f"unknown bit_head_class {cfg.bit_head_class!r}")


class LevelLM(nn.Module):
    """One level of the recursive NTP tower. v4.2: EVERY level — byte level 0 included — uses the
    exact same shape of embed (`CodeEmbed(dq, D)`) + NTP head (`nn.Linear(D, dq)`, independent
    per-bit logits, `chain_bce_loss`) + `code_pre` (D -> dq pre-quantization projection), and
    (per `shared`, below) the literal SAME weight objects across every level, no exceptions. The
    only thing that still distinguishes level 0 from a code level is `is_byte_level`'s use in
    `forward`: converting raw byte ids to/from the shared dq-bit representation via
    `byte_to_dqbits`/`dqbits_to_byte` — everything downstream of that conversion is identical
    code for every level."""

    def __init__(self, cfg: Config, level: int, window: int | None,
                 fuse_d_model: int | None = None, fuse_kv_window: int | None = None,
                 shared: "LevelLM | None" = None, shared_head: "LevelLM | None" = None):
        """`shared`: None (only for level 0, which always constructs its own TRUNK fresh — see
        RefineLM.__init__) or level 0's own already-constructed LevelLM, whose self.blocks/ln_f/
        fuse_* get REUSED here (same nn.Module objects — PyTorch shares the underlying Parameters
        automatically; gradients from every level accumulate into them). The trunk is ALWAYS
        shared across every level regardless of `Config.byte_head_256way` — that flag only affects
        the embed/head/code_pre split below.

        `shared_head`: who owns the embed/ntp_head/code_pre this level uses. By default (`Config.
        byte_head_256way=False`) this is level 0 for every other level (same as `shared`'s trunk-
        ownership — one pool, byte included). Under `byte_head_256way=True`, level 0 builds its
        OWN separate embed/head (see below) and is never anyone's `shared_head`; code levels (1+)
        instead share a SEPARATE pool owned by level 1 — see RefineLM.__init__ for the two
        different ownership assignments this produces.

        `window` stays genuinely per-level regardless of either sharing scheme — passed at FORWARD
        time, never baked into construction (see CausalSelfAttention's own docstring for why)."""
        super().__init__()
        self.level = level
        self.cfg = cfg
        self.window = window
        self.is_byte_level = level == 0
        D = cfg.d_model
        dq = cfg.dq

        if cfg.quant_type == "simplex":
            # quant_type=="simplex" (session: "generalize... every level is softmax head 256 way...
            # no grid assumption that bsq carries... maintain 2 modes now: bsq, and simplex") — a
            # SINGLE nn.Embedding table per pool, used BOTH as the input embedding (gather for byte
            # ids, matmul-by-one-hot for a level's own code — see forward()) AND, weight-tied, as
            # the V-way softmax classifier for next-token prediction (F.linear(h, self.embed.weight))
            # — no separate ntp_head module at all ("do not use bsq linear map, but uses shared
            # embedding table for all level").
            #
            # `code_embed`: which table produces the CODE this level hands UPWARD (forward()'s own
            # code_pre-equivalent step) — usually the SAME object as `self.embed` (self-evident for
            # every code level: there's only one alphabet in play). The one exception is byte level
            # 0 when code_bits != 8 (V != vocab): level 0's OWN table is fixed at vocab=256 (a raw
            # byte's next-BYTE prediction always needs exactly that), but the CODE it hands to level
            # 1 must live in level 1's V-sized alphabet instead — two genuinely different tables for
            # two genuinely different jobs. Level 0 is always built FIRST, so it can't borrow level
            # 1's table yet at this point in construction — RefineLM.__init__ patches `encoders[0].
            # code_embed = encoders[1].embed` right after building level 1, same "assign an already-
            # built submodule onto another module" sharing idiom `shared`/`shared_head` already use
            # elsewhere in this file.
            V = 2 ** cfg.code_bits
            if self.is_byte_level:
                # Level 0 is always constructed FIRST (RefineLM.__init__), so it never borrows —
                # it always builds/owns its own vocab=256 table directly, regardless of code_bits:
                # a raw byte is fundamentally 8 bits, that doesn't shrink/grow with this flag.
                self.embed = nn.Embedding(cfg.vocab, D)
                if V == cfg.vocab:
                    self.code_embed = self.embed   # uniform pool: same table serves both jobs
                elif cfg.untie_levels:
                    self.code_embed = nn.Embedding(V, D)   # untie_levels: own fresh table, never
                                                              # aliased to level 1's (see Config.
                                                              # untie_levels' own docstring — "no
                                                              # aliasing across levels at all")
                # else: left unset here — RefineLM.__init__ patches it in once level 1 exists.
            elif shared_head is not None:
                # code_bits==8 (V==vocab): borrows level 0's OWN table — full uniform sharing,
                # "every level softmax 256-way" literally. code_bits!=8: borrows level 1's own
                # SEPARATE V-sized table instead — see RefineLM.__init__'s pool-split logic below.
                self.embed = shared_head.embed
                self.code_embed = self.embed
            else:
                # only reached by level 1 itself when code_bits!=8 (V!=vocab, tables can't alias) —
                # level 1 owns a fresh, separate V-sized pool for every code level to share.
                self.embed = nn.Embedding(V, D)
                self.code_embed = self.embed

            self.simplex_head = None
            if cfg.simplex_untie_head:
                # mirrors the EXACT same is_byte_level/shared_head/else structure above, one
                # private nn.Linear per pool instead of aliasing self.embed's own weight matrix —
                # see Config.simplex_untie_head's own docstring for what this does and doesn't
                # untie (only the NTP classifier, never code_embed's own upward-code weight-tie).
                head_V = cfg.vocab if self.is_byte_level else V
                if shared_head is not None and shared_head.simplex_head is not None:
                    self.simplex_head = shared_head.simplex_head
                else:
                    self.simplex_head = nn.Linear(D, head_V)
        elif self.is_byte_level and cfg.byte_head_256way:
            # ABLATION (session: "make ablation use regular byte 256-way head, put as flag") —
            # level 0 opts OUT of the shared dq-bit embed/head pool entirely: its own, unshared,
            # exact 256-way softmax path, exactly v4's own byte_repr="embed" mode/bytelm.py's own
            # convention. code_pre still emits a dq-dim code upward regardless (that's a separate
            # concern from what representation level 0 uses for its OWN next-byte prediction).
            self.byte_embed = nn.Embedding(cfg.vocab, D)
            self.ntp_head = nn.Linear(D, cfg.vocab)
            self.code_pre = nn.Linear(D, dq)
        elif shared_head is not None:
            self.embed = shared_head.embed
            self.ntp_head = shared_head.ntp_head
            self.code_pre = shared_head.code_pre
        else:
            assert cfg.code_head_mode in ("independent", "chain", "word")
            self.embed = CodeEmbed(cfg, dq, D)
            if cfg.code_head_mode == "independent":
                self.ntp_head = nn.Linear(D, dq)
            elif cfg.code_head_mode == "word":
                self.ntp_head = BitPredictHeadWordPredict(D, dq, word_bits=cfg.word_bits, d_embed=cfg.word_d_embed, embed_downsample=cfg.word_embed_downsample)
            else:
                self.ntp_head = build_bit_head(cfg, D, dq)
            self.code_pre = nn.Linear(D, dq)

        if self.is_byte_level and cfg.byte_softmax_head_only:
            # ABLATION (session: "no byte embedding, assume byte as bits 0 to bits 255, only head
            # is softmax") — narrower than byte_head_256way above: self.embed/self.code_pre stay
            # the SHARED dq-bit ones assigned above (level 0's input representation and its
            # code-to-level-1 handoff are UNCHANGED, still shared with every code level). Only the
            # byte-level READOUT gets its own private module, stored under a SEPARATE attribute
            # name (not overwriting self.ntp_head) — code levels 1+ still alias `shared_head.
            # ntp_head` from level 0 during THEIR OWN construction (in RefineLM.__init__, built
            # after level 0), so self.ntp_head here must stay the real shared dq-bit head for that
            # aliasing to remain correct; self.byte_softmax_head is what forward() actually reads
            # for byte-level prediction under this flag.
            self.byte_softmax_head = nn.Linear(D, cfg.vocab)

        if self.is_byte_level and cfg.byte_head_factored:
            # ABLATION, same narrow shape as byte_softmax_head_only above (embed/code_pre stay
            # shared) — only the readout differs: FactoredSoftmaxHead (outer-sum, structured/
            # cheap) instead of a dense nn.Linear(D,vocab). See Config.byte_head_factored's own
            # docstring for the params/FLOPs/rank tradeoff.
            self.byte_factored_head = FactoredSoftmaxHead(D, cfg.vocab)

        if self.is_byte_level and cfg.byte_head_lowrank:
            # ABLATION, same narrow shape again — readout is LowRankSoftmaxHead (the classic
            # softmax bottleneck) instead of dense or outer-sum. See Config.byte_head_lowrank's
            # own docstring for why this is expected to strictly beat byte_head_factored at
            # matched rank/param budget.
            self.byte_lowrank_head = LowRankSoftmaxHead(D, cfg.vocab, cfg.byte_head_rank)

        if shared is not None:
            self.blocks = shared.blocks
            self.ln_f = shared.ln_f
        else:
            self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(D)

        # v3 FUSION (module docstring has the full rationale): if this level has a level ABOVE it
        # (fuse_d_model is not None, i.e. level < n_levels-1), it can optionally cross-attend to that
        # level's own hidden state BEFORE running its own self.blocks — the mechanism that lets
        # byte_loss (and any other level's own NTP loss) actually depend on, and be shaped by, the
        # coarser code, unlike v2 where cross-attention only ever touched the separate, detached
        # DecoderLevel/tok_loss path. Q = x directly (already D-dim, this level's own embedding —
        # "encoder reads this directly, no separate embed" per the session ask), no extra q_proj
        # needed. KV = the level-above's hidden state, projected D_{level+1} -> D via fuse_kv_proj
        # (only place a new "own" weight is learned, unavoidable when D's differ — v4.2: D is
        # uniform everywhere, so fuse_kv_proj is really D -> D, kept as a real learned layer
        # anyway rather than an identity special-case). Same jagged causal mask as DecoderLevel
        # (see jagged_causal_mask_and_positions) — position t may only see level-above blocks that
        # completed strictly before t, so no label leakage.
        self.fuse_d_model = fuse_d_model
        self.fuse_kv_window = fuse_kv_window
        if fuse_d_model is not None:
            if shared is not None and getattr(shared, "fuse_d_model", None) is not None:
                # the fusion module is part of the shared trunk too — SAME weights regardless of
                # which adjacent level-pair is fusing (extreme sharing extended to fusion, not
                # just self.blocks/ln_f/embed/head).
                self.fuse_kv_proj = shared.fuse_kv_proj
                if cfg.fuse_use_null_kv:
                    self.fuse_null_kv = shared.fuse_null_kv
            else:
                self.fuse_kv_proj = nn.Linear(fuse_d_model, D)
                if cfg.fuse_use_null_kv:
                    self.fuse_null_kv = nn.Parameter(torch.zeros(1, 1, D))
                    nn.init.normal_(self.fuse_null_kv, std=0.02)
                # concat-only (v4.2): no separate cross-attention module at all — self.blocks' own
                # CausalSelfAttention layers derive K/V for the fused tail directly from their own
                # qkv weights (see _prep_concat).

        self.cross_fuse = None
        self.embed_pass2 = self.blocks_pass2 = self.ln_f_pass2 = self.simplex_head_pass2 = None
        if fuse_d_model is not None:
            # session: "rerun with this: post cross attn" -- v4.2's own concat-only fusion (above)
            # has NO separate cross-attention weights at all (module docstring: the level-above's
            # own hidden state gets read via self.blocks' own qkv). `fuse_mode="cross_attn_post"`
            # reintroduces v4's original `CrossBlock`-based fusion (see that class's own docstring,
            # ported near-verbatim) -- a genuine, separately-weighted cross-attention sublayer run
            # AFTER self.blocks/ln_f produce this level's own clean hidden state, Q=that hidden
            # state, K/V=the level-above's own (projected, null-prepended) hidden state. One
            # CrossBlock instance per level, not one per self.blocks layer (unlike concat, which
            # threads fuse_kv through every layer).
            if cfg.fuse_mode == "cross_attn_post":
                self.cross_fuse = CrossBlock(D, cfg.n_heads, cfg.mlp_mult)
            if cfg.untie_fusion_pass:
                # session: "define another config make the each level-pass separate weights, as
                # if there is 3 level lm... by free weight this means like a fresh level lm own
                # embed, transformer, linear head" -- PASS 2 (this level's own FUSED forward call)
                # gets a COMPLETELY separate identity from PASS 1: own embed, own self.blocks/ln_f
                # trunk, own classifier -- not just a separate trunk. Scoped to the byte-level
                # `quant_type=="simplex"` case (the only path either of this session's two new
                # diagnostic configs actually exercises) -- asserted below rather than silently
                # falling back for any other combination.
                assert self.is_byte_level and cfg.quant_type == "simplex", (
                    "untie_fusion_pass is only implemented for byte-level quant_type='simplex' "
                    "(the case this session's diagnostic configs actually use)"
                )
                self.embed_pass2 = nn.Embedding(cfg.vocab, D)
                self.blocks_pass2 = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
                self.ln_f_pass2 = nn.LayerNorm(D)
                if cfg.simplex_untie_head:
                    self.simplex_head_pass2 = nn.Linear(D, cfg.vocab)

    def _fuse_cross(self, x: torch.Tensor, fuse_kv: torch.Tensor) -> torch.Tensor:
        """`fuse_mode=="cross_attn_post"`'s own fusion step -- ported from qcute_refine_v4.py's own
        LevelLM._fuse (same computation, ONE CrossBlock instead of choosing between fuse_cross_pre/
        fuse_cross_post since this file only ever supports the "post" position). x: [B, L, D] this
        level's own POST-self.blocks hidden state (unlike v4's _fuse, called on the pre-self-
        attention embedding for "pre"/"both" -- "post" always reads the already-self-attended
        state). fuse_kv: [B, n_blocks, D_above] the level-above's own hidden state."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        B, L, D = x.shape
        n_blocks = fuse_kv.size(1)
        device = x.device
        kv = self.fuse_kv_proj(fuse_kv)
        if cfg.fuse_use_null_kv:
            null = self.fuse_null_kv.expand(B, 1, D)
            kv = torch.cat([null, kv], dim=1)
        disallow, k_pos = jagged_causal_mask_and_positions(L, n_blocks, K, self.fuse_kv_window, device,
                                                              include_null=cfg.fuse_use_null_kv)
        rope_q = rope_k = None
        if cfg.cross_attn_rope:
            head_dim = D // cfg.n_heads
            q_pos = torch.arange(L, device=device)
            rope_q = rope_cos_sin_for_positions(q_pos, head_dim, cfg.rope_base, device)
            rope_k = rope_cos_sin_for_positions(k_pos, head_dim, cfg.rope_base, device)
        return self.cross_fuse(x, kv, attn_mask=disallow, rope_q=rope_q, rope_k=rope_k)

    def _prep_concat(self, x: torch.Tensor, fuse_kv: torch.Tensor):
        """Concat fusion's own prep, done ONCE and shared by every self.blocks layer — projects+
        null-prepends fuse_kv and builds the jagged visibility mask/rope-k positions once, since
        L/n_blocks/K are the same for every layer at this level. Returns (kv [B, Nf, D], disallow
        [L, Nf] bool True=blocked, rope_k or None) — passed straight into every Block.forward this
        level runs (see CausalSelfAttention's own fuse_kv/fuse_disallow/fuse_rope_k docstring for
        what each layer then does with it)."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        B, L, D = x.shape
        n_blocks = fuse_kv.size(1)
        device = x.device
        kv = self.fuse_kv_proj(fuse_kv)
        if cfg.fuse_use_null_kv:
            null = self.fuse_null_kv.expand(B, 1, D)
            kv = torch.cat([null, kv], dim=1)
        disallow, k_pos = jagged_causal_mask_and_positions(L, n_blocks, K, self.fuse_kv_window, device,
                                                              include_null=cfg.fuse_use_null_kv)
        rope_k = None
        if cfg.cross_attn_rope:
            head_dim = D // cfg.n_heads
            rope_k = rope_cos_sin_for_positions(k_pos, head_dim, cfg.rope_base, device)
        return kv, disallow, rope_k

    def simplex_logits(self, h: torch.Tensor, use_pass2: bool = False) -> torch.Tensor:
        """quant_type=="simplex" ONLY -- the one dispatch point for "classify h into the V-way
        next-token distribution," so Config.simplex_untie_head has exactly one place to change
        behavior (forward()'s own loss computation and _sample_next_byte both call this instead
        of duplicating the tied/untied branch). use_pass2 (session: "make the each level-pass
        separate weights"): read PASS 2's own private embed_pass2/simplex_head_pass2 instead of
        PASS 1's — only ever True when untie_fusion_pass is set."""
        if use_pass2:
            if self.simplex_head_pass2 is not None:
                return self.simplex_head_pass2(h)
            return F.linear(h, self.embed_pass2.weight)
        if self.simplex_head is not None:
            return self.simplex_head(h)
        return F.linear(h, self.embed.weight)

    def forward(self, seq_repr: torch.Tensor, compute_ntp: bool = True,
                fuse_kv: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """seq_repr: level 0 gets raw byte ids [B, L] (long) — converted internally to the shared
        dq-bit representation via `byte_to_dqbits` (v4.2: unconditional, no more byte_repr modes).
        level i>0 gets its own continuous dq-dim code [B, L, dq] (float) directly. compute_ntp=False
        SKIPS the ntp_head call entirely (real speed lever — level 0 must never pass False). fuse_kv:
        None (default) reproduces PASS 1 exactly (RefineLM._encode's own bottom-up sweep, producing
        every level's code); the level-above's own hidden state (PASS 2) fuses it in via concat
        (v4.2's only fusion mechanism — see module docstring), so THIS call's ntp_loss/h genuinely
        depend on it. Returns (c_i [B, n_blocks, dq], ntp_loss, ntp_acc, h [B, L, D])."""
        cfg = self.cfg
        K = cfg.Ks[self.level]
        D = cfg.d_model
        dq = cfg.dq

        # session: "make the each level-pass separate weights, as if there is 3 level lm... by
        # free weight this means like a fresh level lm own embed, transformer, linear head" --
        # PASS 2 (fuse_kv is not None) reads through embed_pass2/blocks_pass2/ln_f_pass2 instead
        # of the PASS-1 identity when untie_fusion_pass is set (else identical to before: same
        # embed/blocks/ln_f regardless of pass). Only byte-level quant_type=="simplex" ever
        # constructs *_pass2 (asserted in __init__), which is exactly the only case fuse_kv is
        # non-None for level 0 (the top level never receives fuse_kv) in this session's configs.
        use_pass2 = fuse_kv is not None and cfg.untie_fusion_pass
        embed = self.embed_pass2 if use_pass2 else self.embed
        blocks = self.blocks_pass2 if use_pass2 else self.blocks
        ln_f = self.ln_f_pass2 if use_pass2 else self.ln_f

        if cfg.quant_type == "simplex" and self.is_byte_level:
            x = embed(seq_repr)   # gather: seq_repr is [B, L] long raw byte ids
            B, L = seq_repr.shape
        elif cfg.quant_type == "simplex":
            x_in = seq_repr             # [B, L, V] float, one-hot-ish (hard-STE forward value)
            B, L, _ = x_in.shape
            x = x_in @ self.embed.weight   # matmul, NOT gather — a hard index lookup would sever
                                             # the Gumbel-Softmax STE gradient path entirely, since
                                             # x_in isn't a plain integer index here, it CARRIES
                                             # gradient information through its soft backward value
        elif self.is_byte_level and cfg.byte_head_256way:
            x = self.byte_embed(seq_repr)
            B, L = seq_repr.shape
        elif self.is_byte_level:
            x_in = byte_to_dqbits(seq_repr, dq)
            B, L, _ = x_in.shape
            x = self.embed(x_in)
        else:
            x_in = seq_repr
            B, L, _ = x_in.shape
            x = self.embed(x_in)
        n_blocks = L // K

        if fuse_kv is not None:
            assert self.fuse_d_model is not None, "fuse_kv passed but this LevelLM has no fuse module (no level above it?)"

        # concat mode threads fuse_kv through every self.blocks layer's own K/V; cross_attn_post
        # mode runs self.blocks/ln_f CLEAN (no fuse_kv at all) and instead applies a separate
        # CrossBlock AFTER, below.
        concat_kv = concat_disallow = concat_rope_k = None
        if fuse_kv is not None and cfg.fuse_mode == "concat":
            concat_kv, concat_disallow, concat_rope_k = self._prep_concat(x, fuse_kv)

        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in blocks:
            x = block(x, cos, sin, self.window, fuse_kv=concat_kv, fuse_disallow=concat_disallow, fuse_rope_k=concat_rope_k)

        h = ln_f(x)
        if fuse_kv is not None and cfg.fuse_mode == "cross_attn_post":
            h = self._fuse_cross(h, fuse_kv)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            if cfg.quant_type == "simplex":
                # weight-tied V-way softmax classifier — SAME table as the input embed above, no
                # separate ntp_head module. Target: byte level reads raw ids directly (already a
                # class index); code levels recover the index via argmax of the (hard-STE, already
                # one-hot in its forward value) code, exactly mirroring byte_head_256way's own
                # target-derivation, generalized to every level.
                target = (seq_repr[:, 1:].reshape(-1) if self.is_byte_level
                          else seq_repr[:, 1:, :].argmax(-1).reshape(-1))
                logits = self.simplex_logits(h_flat, use_pass2=use_pass2)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif self.is_byte_level and cfg.byte_head_256way:
                # ablation path: exact 256-way softmax, unshared — same computation v4's own
                # byte_repr="embed" mode used, nothing dq-bit/chain related here at all.
                target = seq_repr[:, 1:].reshape(-1)
                logits = self.ntp_head(h_flat)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif self.is_byte_level and cfg.byte_softmax_head_only:
                # narrower ablation: input embedding (x above) and code_pre still went through the
                # SHARED dq-bit path — only the readout used here is private (self.byte_softmax_head,
                # NOT self.ntp_head, which stays the real shared dq-bit head other levels alias).
                target = seq_repr[:, 1:].reshape(-1)
                logits = self.byte_softmax_head(h_flat)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif self.is_byte_level and cfg.byte_head_factored:
                # narrower ablation, same shape as byte_softmax_head_only above — readout is
                # self.byte_factored_head (outer-sum) instead of a dense head.
                target = seq_repr[:, 1:].reshape(-1)
                logits = self.byte_factored_head(h_flat)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif self.is_byte_level and cfg.byte_head_lowrank:
                # narrower ablation, same shape again — readout is self.byte_lowrank_head (the
                # classic softmax bottleneck) instead of a dense or outer-sum head.
                target = seq_repr[:, 1:].reshape(-1)
                logits = self.byte_lowrank_head(h_flat)
                ntp_loss = F.cross_entropy(logits, target)
                with torch.no_grad():
                    ntp_acc = (logits.argmax(-1) == target).float().mean()
            elif cfg.code_head_mode == "word":
                if self.is_byte_level:
                    true_flat_full = byte_to_dqbits(seq_repr[:, 1:], dq).reshape(-1, dq)
                else:
                    true_flat_full = seq_repr[:, 1:, :].reshape(-1, dq)
                raw_list = self.ntp_head(h_flat, true_flat_full)
                ntp_loss = self.ntp_head.loss(raw_list, true_flat_full)
                with torch.no_grad():
                    pred_bits = self.ntp_head.word_ints_to_bits(self.ntp_head.logits_to_word_ints(raw_list))
                    if self.is_byte_level:
                        pred_byte = dqbits_to_byte(pred_bits)
                        true_byte = seq_repr[:, 1:].reshape(-1)
                        ntp_acc = (pred_byte == true_byte).float().mean()
                    else:
                        ntp_acc = ((pred_bits > 0) == (true_flat_full > 0)).float().mean()
            else:
                if self.is_byte_level:
                    true_flat_full = byte_to_dqbits(seq_repr[:, 1:], dq).reshape(-1, dq)
                else:
                    true_flat_full = seq_repr[:, 1:, :].reshape(-1, dq)
                # chain heads need the FULL dq-dim true_bits as teacher-forcing input (the shared head's
                # own fixed dq output shape, needed for code levels regardless of byte-level cropping
                # below) — crop happens AFTER the head runs, only for byte-level loss/acc, never for
                # what conditions the chain.
                raw = self.ntp_head(h_flat, true_flat_full) if cfg.code_head_mode == "chain" else self.ntp_head(h_flat)
                true_flat = true_flat_full
                if self.is_byte_level and dq > 8:
                    # CROP to the first 8 bits for the LOSS too, not just dqbits_to_byte's own
                    # sampling crop — bits 8..dq-1 are a constant pad (byte_to_dqbits), so without
                    # this, byte_loss/val_bpb would include BCE terms on a trivially-learnable
                    # constant target, inflating the "nats-over-dq-bits" value away from genuine
                    # bits-per-BYTE once the padding is learned to near-zero cost either way, but
                    # polluting early-training signal and the metric's own units regardless.
                    raw, true_flat = raw[:, :8], true_flat[:, :8]
                ntp_loss = chain_bce_loss(raw, true_flat)
                with torch.no_grad():
                    if self.is_byte_level:
                        # TRUE byte accuracy (crops to the first 8 bits, session: "for sampling bytes,
                        # just crop byte 9 and more") — not per-bit accuracy, a genuinely stronger/more
                        # interpretable metric matching what every baseline's own byte_acc measures.
                        pred_byte = dqbits_to_byte(raw)
                        true_byte = seq_repr[:, 1:].reshape(-1)
                        ntp_acc = (pred_byte == true_byte).float().mean()
                    else:
                        ntp_acc = ((raw > 0) == (true_flat > 0)).float().mean()
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        # v4: slice to the floor(L/K)*K-length COMPLETE-block prefix before viewing — training's L is
        # always exactly divisible by K (RefineLM.__init__ asserts it), so this is a no-op there, but
        # generation's L grows one byte at a time and is NOT always divisible; a trailing partial
        # block simply doesn't produce a code yet (correct: it isn't causally resolved regardless).
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        if cfg.quant_type == "simplex":
            # weight-tied, no separate code_pre module — uses `code_embed` (usually == `embed`;
            # differs only for byte level 0 when code_bits != vocab, see LevelLM.__init__).
            pre_q = F.linear(h_blocks[:, :, K - 1, :], self.code_embed.weight)
            c_i = gumbel_quantize(pre_q, cfg.gumbel_tau, cfg.use_gumbel_noise)
        elif cfg.quant_type == "bsq":
            pre_q = self.code_pre(h_blocks[:, :, K - 1, :])
            c_i = bsq_quantize(pre_q, dq)
        elif cfg.quant_type == "identity":
            pre_q = self.code_pre(h_blocks[:, :, K - 1, :])
            c_i = pre_q
        else:
            raise ValueError(f"unknown quant_type {cfg.quant_type!r}")
        return c_i, ntp_loss, ntp_acc, h


class RefineLM(nn.Module):
    """N-level recursive NTP tower (N LevelLMs), each level below the
    top fusing the level-above's own hidden state into its own self-
    attention (LevelLM._fuse) — see module docstring. No DecoderLevel
    at all in v4."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert cfg.d_model % cfg.n_heads == 0, f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens
        self.code_seq_lens = [seq_lens[i] // cfg.Ks[i] for i in range(self.n_levels)]

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

        # v4.2 EXTREME WEIGHT SHARING, UNCONDITIONAL (session: "make that 4.2 and no flag, run
        # that scheme by default"). Level 0 is built FIRST (fresh trunk always; fresh head/embed/
        # code_pre too UNLESS byte_head_256way), every other level REUSES the trunk (see LevelLM.
        # __init__'s own `shared` docstring) — the trunk is ALWAYS one shared pool, byte included,
        # regardless of byte_head_256way. `window` stays genuinely per-level regardless
        # (forward-time arg, never baked into construction).
        #
        # Head/embed/code_pre ownership (session: "make ablation use regular byte 256-way head,
        # put as flag") is a SEPARATE pool assignment from the trunk's: default (byte_head_256way=
        # False) — level 0 owns it, every level 1+ borrows, same single pool as the trunk. True —
        # level 0 builds its own unshared 256-way head (LevelLM.__init__ handles this internally,
        # no shared_head needed for level 0); level 1 becomes the NEW owner for the CODE levels'
        # own separate dq-bit pool, and levels 2+ borrow from level 1 instead of level 0.
        encoders: list[LevelLM] = []
        for i in range(self.n_levels):
            fuse_d_model = cfg.d_model if i < self.n_levels - 1 else None
            fuse_kv_window = windows[i + 1] if i < self.n_levels - 1 else None
            if cfg.untie_levels:
                # session: "make it like v4, different head, embed, lm transformer each level" —
                # no aliasing at all, every level builds everything fresh (see Config.untie_levels'
                # own docstring). Overrides every sharing scheme below, including the split-pool one.
                shared_head = None
            elif cfg.byte_head_256way or (cfg.quant_type == "simplex" and cfg.code_bits != 8):
                # split pool: byte (level 0) owns/keeps its own private table; level 1 owns a
                # SEPARATE pool every code level (2+) borrows from instead — same pattern for both
                # triggers, since both boil down to "byte's table and code levels' table can't be
                # the same object" (different shape/semantics for byte_head_256way, different SIZE
                # for quant_type=="simplex" with code_bits!=8).
                shared_head = encoders[1] if i > 1 else None
            else:
                shared_head = encoders[0] if i > 0 else None
            encoders.append(LevelLM(
                cfg, i, windows[i],
                fuse_d_model=fuse_d_model, fuse_kv_window=fuse_kv_window,
                shared=None if cfg.untie_levels else (encoders[0] if i > 0 else None),
                shared_head=shared_head,
            ))
            if i == 1 and cfg.quant_type == "simplex" and cfg.code_bits != 8 and not cfg.untie_levels:
                # level 0's own code_embed couldn't be wired at construction time (level 1 didn't
                # exist yet) — patch it in now that it does. See LevelLM.__init__'s own comment.
                # Skipped under untie_levels: level 0 already built its OWN fresh code_embed table
                # above (no aliasing across levels at all under this flag).
                encoders[0].code_embed = encoders[1].embed
        self.encoders = nn.ModuleList(encoders)

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

    def _encode(self, byte_ids: torch.Tensor, n_active: int, compute_ntp: bool = True) -> tuple[list, list, list, list, list, list]:
        """Shared by forward() (training/eval) AND generation (generate_no_cache calls this
        directly; generate_kv_cache implements an incremental-cache-equivalent of the same two-pass
        logic — see its own docstring) — the ONE place PASS 1 + PASS 2 is implemented, so training
        and generation can never diverge on what "the model's own prediction" means. Runs PASS 1
        (bottom-up sweep, exactly v2's own forward — required regardless, since level i+1's own
        input IS level i's own output code, so level i+1 cannot exist before level i finishes: fusing
        "before the encoder" is circular for a level fusing from ITSELF in one pass), then PASS 2
        (v3's fusion — re-runs levels 0..n_active-2, same weights, same input, this time cross-
        attending to the level-above's own PASS-1 hidden state via LevelLM._fuse BEFORE
        self.blocks runs). PASS 2's ntp_loss/acc/h REPLACES PASS 1's for every fused level; the TOP
        active level (nothing above it) keeps its PASS 1 result. fuse_kv is DETACHED — only the
        FUSING level's own weights (fuse_cross, its own embed/blocks) get gradient from this; the
        level being read from is not reshaped (same principle v2's DecoderLevel used to follow for
        its own cross-attention reads). c_i (what feeds level i+1) is NEVER recomputed in PASS 2 —
        always PASS 1's — so there's no infinite regress up the tower.

        compute_ntp=False (generation's own use — h/c_i are always computed regardless, this only
        gates the loss branch) skips every level's ntp_loss computation entirely — needed because
        generation calls this with a growing, possibly very short byte sequence (as few as 1 byte),
        and computing cross-entropy/BCE loss against an empty or near-empty target would be at best
        wasted work, at worst a crash on a genuinely empty batch (e.g. before any code block has had
        a chance to complete). h_blocks/code_pre/quantize all degrade gracefully to empty tensors on
        a too-short input (verified: `.view` to a 0-sized shape, and elementwise ops on 0-sized
        tensors, are both well-defined) — so c_i/h stay correct and generation-safe even at the very
        start of a sequence, with compute_ntp=False keeping the ntp_loss branch (the ONLY part that
        isn't safe on empty input) fully out of the graph.

        Returns (ntp_losses_pass1, ntp_accs_pass1, ntp_losses_pass2, ntp_accs_pass2, h_list,
        x_list). `*_pass1` (length n_active): every level's OWN, never-overwritten, no-fusion
        NTP loss/acc — modularity goal (session ask: "make it modular, each lm can act
        independently and still get good bpb") means these stay part of the gradient for every
        level, not just a log-only diagnostic. `*_pass2` (length n_active, `None` at any index
        that doesn't fuse — i.e. index n_active-1, the top active level, and every index if
        `fuse_encoder_levels=False`): the fused/cross-attended loss/acc for every level that HAS
        a level above it to fuse from — generalizes to any level count, not just level 0 (a
        3+-level config already fuses every non-top level, unchanged from before). PASS 2 no
        longer overwrites PASS 1 — both are real, differentiable, and both enter forward()'s own
        loss. h_list still ends up holding the fused hidden state wherever fusion ran (needed
        downstream for generation/further levels) — only the LOSS bookkeeping changed, not which
        hidden state feeds what."""
        cfg = self.cfg
        seq_repr = byte_ids
        ntp_losses_pass1, ntp_accs_pass1, h_list, x_list = [], [], [], []

        for i in range(n_active):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, ntp_loss, ntp_acc, h_i = self.encoders[i](seq_repr, compute_ntp=want_ntp)
            ntp_losses_pass1.append(ntp_loss)
            ntp_accs_pass1.append(ntp_acc)
            h_list.append(h_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        ntp_losses_pass2: list = [None] * n_active
        ntp_accs_pass2: list = [None] * n_active
        if cfg.fuse_encoder_levels:
            for i in range(n_active - 1):
                c_i2, ntp_loss2, ntp_acc2, h_i2 = self.encoders[i](
                    x_list[i], compute_ntp=compute_ntp, fuse_kv=h_list[i + 1].detach()
                )
                ntp_losses_pass2[i] = ntp_loss2
                ntp_accs_pass2[i] = ntp_acc2
                h_list[i] = h_i2

        return ntp_losses_pass1, ntp_accs_pass1, ntp_losses_pass2, ntp_accs_pass2, h_list, x_list

    def forward(self, byte_ids: torch.Tensor, n_active: int | None = None) -> tuple[torch.Tensor, dict]:
        """n_active: precomputed by the CALLER (via self.n_active_levels(step), in plain eager
        Python, never inside a torch.compile'd region) rather than taking raw `step` here directly —
        see _encode's own docstring for what actually happens; this method is now just loss
        aggregation on top of it. None (default) = all levels active."""
        cfg = self.cfg
        if n_active is None:
            n_active = self.n_levels

        ntp_losses_pass1, ntp_accs_pass1, ntp_losses_pass2, ntp_accs_pass2, h_list, x_list = self._encode(byte_ids, n_active)

        # "real" / longest-path metric, unchanged from before this session's change: level 0's PASS 2
        # (fused) value if it fused, else its PASS 1 value (fuse_encoder_levels=False, or a curriculum
        # stage where level 0 has nothing above it yet). val_bpb/checkpointing key off THIS, not the
        # sum of all three loss terms below.
        byte_loss = ntp_losses_pass2[0] if ntp_losses_pass2[0] is not None else ntp_losses_pass1[0]
        byte_acc = ntp_accs_pass2[0] if ntp_accs_pass2[0] is not None else ntp_accs_pass1[0]

        # PASS 1 term: every level's own, standalone (no-fusion) NTP loss — modularity goal, each
        # LevelLM pushed toward being independently competent, not just useful as fusion's input.
        pass1_code_total = (torch.stack(ntp_losses_pass1[1:]).sum() if n_active > 1
                             else byte_loss.new_zeros(()))
        pass1_total = cfg.byte_ntp_weight * ntp_losses_pass1[0] + cfg.code_ntp_weight * pass1_code_total

        # PASS 2 term: every FUSING level's cross-attended loss, summed — generalizes beyond level 0
        # to any level count (levels 0..n_active-2 all fuse from the level above them).
        pass2_terms = [l for l in ntp_losses_pass2 if l is not None]
        pass2_total = (cfg.fusion_ntp_weight * torch.stack(pass2_terms).sum() if pass2_terms
                        else byte_loss.new_zeros(()))

        loss = pass1_total + pass2_total
        ntp_total = torch.stack(ntp_losses_pass1 + pass2_terms).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "pass1_total": pass1_total, "pass2_total": pass2_total, "ntp_loss_total": ntp_total,
            "n_active_levels": byte_loss.new_tensor(float(n_active)),   # tensor, not plain int — every
                                                                          # metrics value gets .item()'d
                                                                          # downstream (eval_model/train)
            # every level's OWN (no-fusion) loss/acc — level0 = "unconditional_pass1" from
            # scripts/probe_v4_fusion_contribution.py, now tracked live every step for every level,
            # not just a post-hoc probe against a finished checkpoint. Top level's pass1 IS its
            # regular value (never fuses) — same number under both names, kept for a uniform column.
            **{f"level{i}_ntp_loss_pass1": l for i, l in enumerate(ntp_losses_pass1)},
            **{f"level{i}_ntp_acc_pass1": a for i, a in enumerate(ntp_accs_pass1)},
            # every FUSING level's cross-attended loss/acc (absent — no key — at non-fusing indices).
            **{f"level{i}_ntp_loss_pass2": l for i, l in enumerate(ntp_losses_pass2) if l is not None},
            **{f"level{i}_ntp_acc_pass2": a for i, a in enumerate(ntp_accs_pass2) if a is not None},
        }
        return loss, metrics


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor, from_pass2: bool = False) -> torch.Tensor:
    """from_pass2 (session: "make the each level-pass separate weights"): True when h_last came
    from RefineLM._encode's own PASS 2 (fused) sweep rather than PASS 1 -- callers driven by the
    normal generate_no_cache/generate_kv_cache path pass whatever _encode/maybe_emit_code actually
    produced (fused, whenever fuse_encoder_levels=True and n_levels>=2); a dedicated
    level-0-unconditional generation path passes False. Only matters when
    Config.untie_fusion_pass is set — PASS 2 then has its own private classifier, so classifying
    its own output with PASS 1's weights would be wrong."""
    enc0 = model.encoders[0]
    if model.cfg.quant_type == "simplex":
        return enc0.simplex_logits(h_last, use_pass2=from_pass2 and model.cfg.untie_fusion_pass).argmax(-1)
    if model.cfg.byte_head_256way:
        return enc0.ntp_head(h_last).argmax(-1)   # [B, vocab] exact softmax logits, unshared head
    if model.cfg.byte_softmax_head_only:
        return enc0.byte_softmax_head(h_last).argmax(-1)   # own private vocab head; embed/code_pre
                                                              # still the shared dq-bit ones elsewhere
    if model.cfg.byte_head_factored:
        return enc0.byte_factored_head(h_last).argmax(-1)   # own private outer-sum head
    if model.cfg.byte_head_lowrank:
        return enc0.byte_lowrank_head(h_last).argmax(-1)   # own private low-rank-bottleneck head
    if model.cfg.code_head_mode == "word":
        logits_list = enc0.ntp_head(h_last)   # list of n_words tensors, greedy (true_bits=None)
        word_ints = enc0.ntp_head.logits_to_word_ints(logits_list)
        raw = enc0.ntp_head.word_ints_to_bits(word_ints)
        return dqbits_to_byte(raw)
    raw = enc0.ntp_head(h_last)   # [B, dq] independent-bit logits, shared head
    return dqbits_to_byte(raw)   # crops to the first 8 bits — see dqbits_to_byte's own docstring


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Reference (slow, obviously-correct) byte-by-byte generation:
    recomputes the WHOLE encoder stack (every active level, PASS 1 + PASS 2
    fusion) from scratch every new byte, via RefineLM._encode — the SAME
    method forward() uses for training/eval, so generation can never
    silently diverge from what val_bpb actually measures. This is the gap
    v3 left open (its generate_no_cache/generate_kv_cache only ever touched
    LevelLM[0] directly, bypassing fusion entirely) — fixed here by
    routing through _encode instead of reimplementing level 0's forward
    pass inline."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    n_active = model.n_levels

    for _ in range(n_new_bytes):
        _, _, _, _, h_list, _ = model._encode(all_bytes, n_active, compute_ntp=False)
        from_pass2 = model.cfg.fuse_encoder_levels and n_active >= 2
        next_byte = _sample_next_byte(model, h_list[0][:, -1, :], from_pass2=from_pass2)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)

    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """KV-cache-efficient generation, fusion-aware — v4's own addition (v2/v3
    never needed this: generation there never touched cross-attention at
    all, since DecoderLevel wasn't part of the generative path either).

    Maintains, per level i (0..n_levels-1), a CLEAN self-attention cache:
    produces level i's own hidden state at each of ITS OWN positions, used
    (a) to emit codes feeding level i+1 once every Ks[i] new arrivals, and
    (b) — for i>=1 — as the KV source for level (i-1)'s fusion. PLUS one
    FUSED self-attention cache, for level 0 ONLY: the actual sampling path.

    Only level 0 needs a fused cache, regardless of how many levels exist:
    RefineLM._encode's own PASS 2 reads h_list[i+1] BEFORE that level could
    ever be overwritten by ITS OWN PASS 2 (see _encode's docstring) — so
    fuse_kv always sources from the level-above's CLEAN pass, never a fused
    one — and only level 0's own ntp_head is ever sampled from. So level
    i>0's own fused pass, even if it existed, would influence nothing
    generation reads.

    Causality note: unlike training's batched jagged mask, no explicit mask
    is applied here — a block's projected KV row is only ever APPENDED to
    the running fuse-KV cache once that block has genuinely, causally
    resolved (see `maybe_emit_code`), so every row present at any given step
    is automatically valid to attend to in full — attending to "everything
    currently cached" is exactly equivalent to the batched mask evaluated at
    that same query position (verified: a block's `resolved_pos` — its own
    last raw-time position — is computed identically to `k_pos` in
    jagged_causal_mask_and_positions, and the row is appended right after
    that position's clean_step, before the SAME position's fused_step reads
    it — matching the training-time mask's own boundary condition, where a
    block resolved exactly at query position t IS already visible to t)."""
    cfg = model.cfg
    assert model.windows[0] is None, "generate_kv_cache only supports dense attention at level 0 (attn_window[0] must be -1) — see docstring"
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.size(0)
    n_levels = model.n_levels
    Ks = cfg.Ks
    fuse_on = cfg.fuse_encoder_levels and n_levels > 1

    clean_cache_k = [[None] * len(model.encoders[i].blocks) for i in range(n_levels)]
    clean_cache_v = [[None] * len(model.encoders[i].blocks) for i in range(n_levels)]
    clean_pos = [0] * n_levels     # next rope position to use, per level's OWN sequence
    pending = [0] * n_levels       # positions arrived since level i's last complete block
    block_idx = [0] * n_levels     # complete blocks emitted so far, per level

    fused_cache_k: list[torch.Tensor | None] = [None] * len(model.encoders[0].blocks)
    fused_cache_v: list[torch.Tensor | None] = [None] * len(model.encoders[0].blocks)

    fuse_kv_rows = None   # [B, n_kv, D0] running cache (null row + one per completed level-0 block)
    fuse_k_pos = None     # [n_kv] each row's own raw-time position, for cross_attn_rope

    def clean_step(level: int, token: torch.Tensor) -> torch.Tensor:
        """Advance level `level`'s CLEAN self-attention by one of its own input positions.
        token: [B] long (level 0) or [B, dq] float (level>0). Returns h_new [B, D]."""
        enc = model.encoders[level]
        D = cfg.d_model
        if cfg.quant_type == "simplex" and level == 0:
            x = enc.embed(token).unsqueeze(1)          # gather: token is [B] long byte id
        elif cfg.quant_type == "simplex":
            x = (token @ enc.embed.weight).unsqueeze(1)   # matmul: token is [B, V] one-hot code
        elif level == 0 and cfg.byte_head_256way:
            x = enc.byte_embed(token).unsqueeze(1)
        elif level == 0:
            x = enc.embed(byte_to_dqbits(token, cfg.dq)).unsqueeze(1)
        else:
            x = enc.embed(token).unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(clean_pos[level], head_dim, cfg.rope_base, device)
        for li, block in enumerate(enc.blocks):
            x, clean_cache_k[level][li], clean_cache_v[level][li] = block.forward_step(
                x, cos_new, sin_new, clean_cache_k[level][li], clean_cache_v[level][li]
            )
        clean_pos[level] += 1
        return enc.ln_f(x).squeeze(1)

    def maybe_emit_code(level: int, h_new: torch.Tensor) -> None:
        """After a clean_step at `level`, check whether a block just completed there; if so,
        quantize it into a code, feed it upward (recursing into level+1), and — level 0 only —
        append the new fuse-KV row generation's sampling path will read."""
        nonlocal fuse_kv_rows, fuse_k_pos
        if level + 1 >= n_levels:
            return   # top level's own code is never consumed by anything -- nothing to do
        pending[level] += 1
        if pending[level] < Ks[level]:
            return
        pending[level] = 0
        enc = model.encoders[level]
        if cfg.quant_type == "simplex":
            pre_q = F.linear(h_new, enc.code_embed.weight)
            c_new = gumbel_quantize(pre_q, cfg.gumbel_tau, cfg.use_gumbel_noise)
        else:
            pre_q = enc.code_pre(h_new)
            c_new = bsq_quantize(pre_q, cfg.dq) if cfg.quant_type == "bsq" else pre_q
        block_idx[level] += 1
        resolved_pos = block_idx[level] * Ks[level] - 1   # matches jagged_causal_mask_and_positions'
                                                             # own block_pos=(b+1)*K-1 exactly
        h_above = clean_step(level + 1, c_new)
        maybe_emit_code(level + 1, h_above)
        if level == 0 and fuse_on:
            new_row = model.encoders[0].fuse_kv_proj(h_above).unsqueeze(1)   # [B, 1, D0]
            fuse_kv_rows = torch.cat([fuse_kv_rows, new_row], dim=1)
            fuse_k_pos = torch.cat([fuse_k_pos, torch.tensor([resolved_pos], device=device)])

    def fused_step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        """Advance level 0's FUSED pass by one byte — the actual sampling path. v4.2: concat-only —
        fuse_kv_rows IS already this level's own D-dim, null-prepended KV (identical to what
        training's own _prep_concat computes each forward), fed straight into every block's
        forward_step. No explicit visibility mask needed — see the module docstring's own
        reasoning (every row present is, by construction, already causally valid at this step)."""
        enc0 = model.encoders[0]
        D = cfg.d_model
        if cfg.quant_type == "simplex":
            x = enc0.embed(byte_id).unsqueeze(1)
        elif cfg.byte_head_256way:
            x = enc0.byte_embed(byte_id).unsqueeze(1)
        else:
            x = enc0.embed(byte_to_dqbits(byte_id, cfg.dq)).unsqueeze(1)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        concat_kv = concat_rope_k = None
        if fuse_on:
            concat_kv = fuse_kv_rows
            if cfg.cross_attn_rope:
                concat_rope_k = rope_cos_sin_for_positions(fuse_k_pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(enc0.blocks):
            x, fused_cache_k[li], fused_cache_v[li] = block.forward_step(
                x, cos_new, sin_new, fused_cache_k[li], fused_cache_v[li],
                fuse_kv=concat_kv, fuse_rope_k=concat_rope_k,
            )
        return enc0.ln_f(x).squeeze(1)

    if fuse_on:
        D0 = cfg.d_model
        if cfg.fuse_use_null_kv:
            fuse_kv_rows = model.encoders[0].fuse_null_kv.expand(B, 1, D0).clone()
            fuse_k_pos = torch.zeros(1, dtype=torch.long, device=device)
        else:
            # no null slot: starts genuinely empty (0 KV rows) — verified this session that
            # fuse_cross/SDPA handles a zero-length KV cleanly (well-defined zero output, no
            # crash/NaN) — matches training's own jagged_causal_mask_and_positions(include_null=False)
            # behavior for the same early-position case.
            fuse_kv_rows = torch.zeros(B, 0, D0, device=device)
            fuse_k_pos = torch.zeros(0, dtype=torch.long, device=device)

    L0 = prompt_bytes.size(1)
    last_fused_h = None
    for pos in range(L0):
        h_clean = clean_step(0, prompt_bytes[:, pos])
        maybe_emit_code(0, h_clean)
        last_fused_h = fused_step(prompt_bytes[:, pos], pos)

    out_bytes = [prompt_bytes]
    for i in range(n_new_bytes):
        next_byte = _sample_next_byte(model, last_fused_h, from_pass2=fuse_on)
        out_bytes.append(next_byte.unsqueeze(1))
        pos = L0 + i
        h_clean = clean_step(0, next_byte)
        maybe_emit_code(0, h_clean)
        last_fused_h = fused_step(next_byte, pos)

    if was_training:
        model.train()
    return torch.cat(out_bytes, dim=1)[0]


@torch.no_grad()
def generate_level0_uncond(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Session: "for level 0 uncond and level 0 pass 2 cross attn" -- level 0's own PASS-1-ONLY
    generation, ignoring every level above it and any fusion entirely, even when
    fuse_encoder_levels=True (unlike generate_no_cache, which always reproduces whatever
    RefineLM._encode's own PASS 2 sweep does). Calls model.encoders[0] directly with fuse_kv=None
    every step -- exactly PASS 1's own computation for level 0, matching _encode's own PASS 1
    sweep at i=0. Same "recompute everything, obviously correct" tradeoff generate_no_cache
    itself makes; no KV cache."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encoders[0]
    for _ in range(n_new_bytes):
        _, _, _, h = enc0(all_bytes, compute_ntp=False)   # fuse_kv=None always -- PASS 1 only
        next_byte = _sample_next_byte(model, h[:, -1, :], from_pass2=False)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


def qualitative_generate(model: "RefineLM", prompt_bytes: torch.Tensor, gen_len: int,
                          ground_truth: torch.Tensor | None, device: str, log=print, label: str = "") -> None:
    """Same role as qcute.bytelm's own qualitative_generate -- greedy AR
    continuation from a dataset-drawn prompt, logged alongside ground truth
    -- but called from INSIDE train()'s own eval round here (session: "plug
    in ar generation code in train code eval round... goal is to get
    generation close to train sample else conclude degenerate arch"),
    not just once after training finishes like bytelm's version. Uses
    generate_no_cache (not generate_kv_cache): the same "obviously correct,
    windowed-attn-safe" reference every other qual-gen script in this
    session already standardized on. Also generates level 0's own
    UNCONDITIONAL (PASS-1-only, no fusion) continuation for the SAME
    prompt via generate_level0_uncond (session: "for level 0 uncond and
    level 0 pass 2 cross attn") -- a direct within-model comparison of
    "with fusion" vs. "without," not just against an external baseline.
    label: prefixed to every log line (e.g. "train"/"val") so both
    regions' output is distinguishable when logged back to back."""
    prefix = f"qual_{label}_" if label else "qual_"
    out = generate_no_cache(model, prompt_bytes, gen_len, device)
    gen_bytes = bytes(out[prompt_bytes.numel():].tolist())
    out_uncond = generate_level0_uncond(model, prompt_bytes, gen_len, device)
    gen_bytes_uncond = bytes(out_uncond[prompt_bytes.numel():].tolist())
    log(f"{prefix}prompt:            {bytes(prompt_bytes.tolist())!r}")
    log(f"{prefix}generated_pass2:   {gen_bytes!r}")
    log(f"{prefix}generated_uncond:  {gen_bytes_uncond!r}")
    if ground_truth is not None:
        log(f"{prefix}ground_truth:      {bytes(ground_truth.tolist())!r}")


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
def _add_per_level_bpb(result: dict) -> dict:
    """Shared by eval_model (val) and train() (train) — see eval_model's own comment for the
    "level i>0's bpb isn't really bits-PER-BYTE" caveat (context_len mismatch across levels),
    tracked anyway per session request. Mutates and returns `result`."""
    for k in list(result.keys()):
        if k.endswith("_ntp_loss_pass1") or k.endswith("_ntp_loss_pass2"):
            result[k.replace("_ntp_loss_", "_bpb_")] = result[k] / math.log(2)
    return result


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
    result["bpb"] = result["byte_loss"] / math.log(2)   # "real"/longest-path metric — level 0's PASS 2
                                                          # (fused) value if fused, else PASS 1's.
    # per-level pass1/pass2 bpb, generalized to every level tracked in metrics — CAVEAT: only level 0's
    # is genuinely bits-PER-BYTE (context_len raw bytes); level i>0's own "bpb" here is nats/log(2) of
    # a CODE-token cross-entropy over a context_len/Ks[i-1] shorter sequence, not directly comparable
    # across levels or to any byte-level baseline — tracked anyway per session request ("track all
    # train bpb val bpb even though the context len is mismatch"), just don't read it as true bpb.
    return _add_per_level_bpb(result)


def build_param_groups(model: RefineLM) -> list[dict]:
    """One param group per activation STAGE (stage 0 = encoders[0]'s own params;
    stage i>=1 = whatever of encoders[i]'s params haven't already been claimed by an earlier
    stage — see Config.layer_warmup_steps; v4 has no DecoderLevel to bundle in alongside it,
    unlike v2/v3) — lets train() give each stage its own reset warmup schedule. With no
    curriculum (layer_warmup_steps empty), every stage activates at step 0 and this is
    behaviorally identical to one global param group.

    v4.1 DEDUPLICATES by Parameter identity (`id(p)`) — under `Config.share_levellm`/
    `share_code_head`, encoders[i>0].parameters() genuinely OVERLAPS encoders[0]'s own (same
    nn.Module objects — see LevelLM.__init__'s shared_trunk/shared_code_head), and
    torch.optim.Optimizer raises if the same Parameter object appears in two groups. Since
    encoders[0] is always processed first, every SHARED weight lands in stage 0 (correct: shared
    weights are active — and need gradient — from the very first active level onward, exactly
    like stage 0's own params); each later stage's group then only contains whatever it did NOT
    share (typically just that level's own embed/ntp_head/code_pre when share_code_head=False).
    v4 (share_levellm=False) never has overlap in the first place, so this is a no-op there —
    identical grouping to before."""
    groups = []
    seen: set[int] = set()
    for i in range(model.n_levels):
        params = []
        for p in model.encoders[i].parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        groups.append({"params": params, "stage": i})
    return groups


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v4", dynamic_ncols=True)
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
        )

        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = _add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb   # keep the official name/value (== byte_loss/log(2))
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device, step=step)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.qual_gen_bytes > 0:
                # session: "every eval round, generate both train prompt and val prompt" --
                # args.qual_source is no longer a choice between the two, both always run.
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                          window[args.qual_prompt_bytes:], device, log=log, label=label)


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

    p = argparse.ArgumentParser(description="Recursive NTP tower + LevelLM fusion only, no DecoderLevel (qcute_refine_v4)", parents=[pre])
    p.add_argument("--dq", type=int, default=8)
    p.add_argument("--Ks", default=(2, 2, 2))
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=128)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--bit_chain_n_heads", type=int, default=2)
    p.add_argument("--bit_chain_gamma", type=float, default=1.0)
    p.add_argument("--bit_chain_fixed_kernel", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--bit_head_class", type=str, default="attn", choices=["attn", "conv", "ssm", "hsoftmax", "conv_dilated"])
    p.add_argument("--bit_conv_kernel_size", type=int, default=None)
    p.add_argument("--bit_conv_impl", type=str, default="matmul", choices=["conv1d", "matmul", "depthwise"])
    p.add_argument("--bit_inner_downsample", type=int, default=1)
    p.add_argument("--bit_downsample_h", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--bit_per_position_head", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--bit_ssm_d_state", type=int, default=None)
    p.add_argument("--conv_dilated_base", type=int, default=2)
    p.add_argument("--conv_dilated_mode", type=str, default="depthwise", choices=["depthwise", "dense"])
    p.add_argument("--code_head_mode", type=str, default="independent", choices=["independent", "chain", "word"])
    p.add_argument("--word_bits", type=int, default=8)
    p.add_argument("--word_d_embed", type=int, default=None)
    p.add_argument("--word_embed_downsample", type=int, default=1)
    p.add_argument("--byte_head_256way", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--byte_softmax_head_only", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--byte_head_factored", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--byte_head_lowrank", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--byte_head_rank", type=int, default=None)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--fusion_ntp_weight", type=float, default=1.0)
    p.add_argument("--untie_levels", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--simplex_untie_head", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--fuse_mode", type=str, default="concat", choices=["concat", "cross_attn_post"])
    p.add_argument("--untie_fusion_pass", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "identity", "simplex"])
    p.add_argument("--code_bits", type=int, default=8)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--use_gumbel_noise", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--code_embed_mode", type=str, default="linear", choices=["linear", "mlp", "pq_table"])
    p.add_argument("--fuse_encoder_levels", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--fuse_use_null_kv", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--cross_attn_rope", type=lambda x: x.lower() != "false", default=True)
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
    p.add_argument("--qual_gen_bytes", type=int, default=0,
                    help="if >0, AR-generate this many bytes EVERY eval round (not just post-training, unlike qcute.bytelm's own --qual_gen_bytes) and log prompt/generated/ground_truth -- a diagnostic to watch generation quality progress live during training")
    p.add_argument("--qual_source", choices=["train", "val"], default="train",
                    help="which region to draw the qual-gen prompt from -- default train (not bytelm's val default): the diagnostic question here is 'can it even reproduce train', not generalization")
    p.add_argument("--qual_prompt_bytes", type=int, default=64)

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
    # v4.2: n_levels inferred from Ks's own length directly (dq/d_model/n_layers are single
    # shared scalars now, no longer a per-level tuple to infer level count from).
    args.Ks = _parse_int_tuple(args.Ks)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, dq=args.dq, d_model=args.d_model, n_layers=args.n_layers,
        context_len=args.context_len, n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window,
        rope_base=args.rope_base, bit_chain_n_heads=args.bit_chain_n_heads, bit_chain_gamma=args.bit_chain_gamma,
        bit_chain_fixed_kernel=args.bit_chain_fixed_kernel, bit_head_class=args.bit_head_class,
        bit_conv_kernel_size=args.bit_conv_kernel_size, bit_conv_impl=args.bit_conv_impl,
        bit_inner_downsample=args.bit_inner_downsample, bit_ssm_d_state=args.bit_ssm_d_state,
        bit_downsample_h=args.bit_downsample_h, bit_per_position_head=args.bit_per_position_head,
        conv_dilated_base=args.conv_dilated_base, conv_dilated_mode=args.conv_dilated_mode,
        code_head_mode=args.code_head_mode, byte_head_256way=args.byte_head_256way,
        word_bits=args.word_bits, word_d_embed=args.word_d_embed, word_embed_downsample=args.word_embed_downsample,
        byte_softmax_head_only=args.byte_softmax_head_only,
        byte_head_factored=args.byte_head_factored, byte_head_lowrank=args.byte_head_lowrank,
        byte_head_rank=args.byte_head_rank,
        code_ntp_weight=args.code_ntp_weight, byte_ntp_weight=args.byte_ntp_weight,
        fusion_ntp_weight=args.fusion_ntp_weight,
        untie_levels=args.untie_levels, simplex_untie_head=args.simplex_untie_head,
        fuse_mode=args.fuse_mode, untie_fusion_pass=args.untie_fusion_pass,
        quant_type=args.quant_type, code_bits=args.code_bits, gumbel_tau=args.gumbel_tau,
        use_gumbel_noise=args.use_gumbel_noise,
        code_embed_mode=args.code_embed_mode, fuse_encoder_levels=args.fuse_encoder_levels,
        fuse_use_null_kv=args.fuse_use_null_kv,
        cross_attn_rope=args.cross_attn_rope,
        layer_warmup_steps=args.layer_warmup_steps,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        # true whole-model compile, works fine WITH Config.layer_warmup_steps — see --compile's
        # own help text above (train()/eval_model() pass n_active, not raw step, into the model).
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} — tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} dq={cfg.dq} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} fuse_encoder_levels={cfg.fuse_encoder_levels} "
        f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
