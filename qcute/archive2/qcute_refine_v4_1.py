"""qcute.qcute_refine_v4_1 — CLONE of qcute_refine_v4.py, PLUS EXTREME
WEIGHT SHARING (session ask: "clone v4 to v4.1 for extreme weight
sharing, only one levellm (can adjust num layer and dim) but shared
across byte and code levels"). New `Config.share_levellm` (default True):
every level's `LevelLM.self.blocks`/`ln_f`/fusion modules
(`fuse_kv_proj`/`fuse_null_kv`/`fuse_cross_pre`/`fuse_cross_post`) are the
SAME nn.Module objects as level 0's own — genuinely tied weights (same
Parameters, gradients from every level accumulate into them), not
separate same-shaped copies. Requires uniform `tier_d_models`/
`tier_n_layers` across every level (a shared trunk can't have a different
width/depth per level).

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
    share_levellm: bool = True    # v4.1's own reason to exist (session ask: "extreme weight
                                    # sharing, only one levellm... shared across byte and code
                                    # levels"). True (default): every level's self.blocks/ln_f/
                                    # fuse_* (fuse_kv_proj/fuse_null_kv/fuse_cross_pre/
                                    # fuse_cross_post) are the SAME nn.Module objects as level 0's
                                    # own — genuinely tied weights, not just same-shaped separate
                                    # copies (see LevelLM.__init__'s own shared_trunk docstring).
                                    # Requires uniform tier_d_models AND tier_n_layers across every
                                    # level (asserted in RefineLM.__init__) — a shared trunk can't
                                    # have a different width/depth per level by construction. `Ks`/
                                    # `attn_window` stay genuinely per-level regardless (session:
                                    # "even though shared, the k and attn window for each level can
                                    # be different") — window became a FORWARD-time argument to
                                    # CausalSelfAttention this file (v4 baked it into the module at
                                    # construction; that's incompatible with one shared module
                                    # serving levels with different windows). False: reproduces
                                    # qcute_refine_v4.py's own per-level-independent-weights
                                    # behavior exactly (this file's own version of "no sharing").
    share_code_head: bool = False  # ablation, off by default ("bsq head and bsq linear map...
                                    # different each level by default... can be set weight sharing
                                    # for ablation" — session ask). Only affects CODE levels
                                    # (1..n_levels-1) — byte level 0's own byte_embed/ntp_head are
                                    # never shared (there's only one byte-typed level). True: every
                                    # code level's own `embed` (CodeEmbed)/`ntp_head`/`code_pre`
                                    # (the "bsq linear map", D -> dq pre-quantization projection)
                                    # get tied to level 1's own (the first code level) — requires
                                    # uniform dqs across every code level (asserted). Independent of
                                    # share_levellm — the shared TRUNK (self.blocks) and the
                                    # per-level BSQ head/embed are two separate sharing questions,
                                    # by design (session: kept them as two distinct knobs).
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
    fuse_position: str = "pre"    # WHERE _fuse's cross-attention sits relative to self.blocks, for every
                                    # level that fuses. "pre" (default, original v3/v4 behavior): fuse THEN
                                    # self.blocks — every raw-embedded position gets cross-level context
                                    # BEFORE positions exchange information with each other via self-
                                    # attention, so self-attention can then propagate one position's fused
                                    # context to other positions during mixing. "post": self.blocks THEN
                                    # fuse — positions first mix purely among themselves (no cross-level
                                    # info at all yet), and only the FINAL per-position representation gets
                                    # to look at the coarser code, with no further mixing afterward (a
                                    # position's fused pickup stays local to it). Both are equally causally
                                    # sound (self-attention's own mask and fuse's own jagged mask are
                                    # independent constraints on different axes — neither's correctness
                                    # depends on which ran first) but compute genuinely different functions,
                                    # not equivalent ones — a real architectural variant, not a bugfix
                                    # relationship. Session ask: "add flag ... pre or post cross attn".
                                    # "both" (NEW): runs BOTH — a separate `fuse_cross_pre` CrossBlock before
                                    # self.blocks AND a separate `fuse_cross_post` CrossBlock after. Two
                                    # independent sets of cross-attention weights (real extra params, not a
                                    # config-only toggle) — session ask: "allow levellm does decoding with
                                    # pre and post cross attn, add more param". "concat" (NEW): no separate
                                    # CrossBlock at ALL. Instead, the level-above's hidden state (projected
                                    # to this level's D via the same `fuse_kv_proj`, null-prepended the same
                                    # way if `fuse_use_null_kv`) is appended to the TAIL of every
                                    # self.blocks layer's own K/V, and each layer does ONE joint attention
                                    # call over [local windowed K/V ; fused tail] instead of two sequential
                                    # attention operations — see CausalSelfAttention._fuse_kv_proj/
                                    # LevelLM._prep_concat. Each layer derives its OWN K/V for the fused
                                    # tail via ITS OWN qkv weights (same weights it uses for local tokens,
                                    # applied to the fixed projected coarser state) — no new cross-attention
                                    # parameters at all (cheaper than "pre"/"post"/"both", reuses existing
                                    # self-attention capacity instead of adding a parallel mechanism).
                                    # Visibility uses the same jagged_causal_mask_and_positions geometry as
                                    # CrossBlock's own mask, just merged into self-attention's own mask
                                    # instead of a separate SDPA call. Session ask: "if user disable both pre
                                    # and post, a special self attn with concat higher level kv at behind".
    fuse_use_null_kv: bool = True  # whether _fuse's cross-attention gets a learned "null" KV slot
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
    byte_repr: str = "bits"       # LEVEL 0 ONLY. "bits" (default/original): byte_to_bits 8-dim
                                    # projection + BitPredictHead chain NTP head. "embed": traditional
                                    # nn.Embedding(vocab, D) lookup + plain nn.Linear(D, vocab) NTP head,
                                    # 256-way cross-entropy — exactly bytelm.py's own convention. Both real,
                                    # kept options — see LevelLM's own docstring.
    code_head_mode: str = "chain"  # LEVELS>0 ONLY (encoder side). "chain" (default/original):
                                    # BitPredictHead, exact chain-rule cross-bit conditioning.
                                    # "independent": single plain nn.Linear(D, in_dq), independent
                                    # per-bit logits, no BitPredictHead — every earlier BSQ fork's own
                                    # default before the Fetch-style chain head was introduced.
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


def bsq_quantize(v: torch.Tensor, dq: int) -> torch.Tensor:
    v_unit = F.normalize(v, dim=-1)
    return (v_unit + (torch.sign(v_unit) - v_unit).detach()) / math.sqrt(dq)


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
    """`fuse_kv`/`fuse_disallow`/`fuse_rope_k` (all default None): ONLY used by LevelLM's "concat"
    fuse_position (see Config.fuse_position's own docstring) — a fixed, already-D-dim, already
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
    qcutelm_vlt11.py via qcute_refine.py). Used for LevelLM's own NTP
    head, in both its unconditioned (PASS 1) and fused (PASS 2) calls."""

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
    head (byte level's own bits-mode head, and any level's
    code_head_mode=="chain" head) gets built, so switching architectures
    is one flag, not per-call-site edits."""
    if cfg.bit_head_class == "attn":
        return BitPredictHeadAttn(d_model, dq, cfg.bit_chain_n_heads, cfg.bit_chain_gamma, cfg.bit_chain_fixed_kernel, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "conv":
        return BitPredictHeadConv(d_model, dq, kernel_size=cfg.bit_conv_kernel_size, gamma=cfg.bit_chain_gamma, conv_impl=cfg.bit_conv_impl, downsample=cfg.bit_inner_downsample)
    elif cfg.bit_head_class == "ssm":
        return BitPredictHeadSSM(d_model, dq, d_state=cfg.bit_ssm_d_state, gamma=cfg.bit_chain_gamma, downsample=cfg.bit_inner_downsample)
    else:
        raise ValueError(f"unknown bit_head_class {cfg.bit_head_class!r}")


class LevelLM(nn.Module):
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
                 fuse_d_model: int | None = None, fuse_kv_window: int | None = None,
                 shared_trunk: "LevelLM | None" = None, shared_code_head: "LevelLM | None" = None):
        """v4.1 EXTREME WEIGHT SHARING (session ask: "only one levellm... shared across byte and
        code levels... even though shared, the k and attn window for each level can be different"):

        `shared_trunk`: None (default — v4-identical, this level gets its OWN self.blocks/ln_f/
        fuse_* weights) or another already-constructed LevelLM (always level 0's own, by
        convention — see RefineLM.__init__) whose self.blocks/ln_f/fuse_kv_proj/fuse_null_kv/
        fuse_cross_pre/fuse_cross_post get REUSED (same nn.Module objects assigned here, not
        copied — PyTorch shares the underlying Parameters automatically, gradients from every
        level accumulate into the same weights). Requires uniform tier_d_models/tier_n_layers
        across every level (asserted) — `window` stays genuinely per-level regardless (passed at
        FORWARD time now, not baked into CausalSelfAttention's own construction — see that
        class's own docstring).

        `shared_code_head`: None (default — "bsq head and bsq linear map... different each level
        by default") or another code-level LevelLM whose embed/ntp_head/code_pre get reused the
        same way — Config.share_code_head's ablation, requires uniform in_dq/dqs across the code
        levels that share (asserted). Byte level (level 0) is never affected — there's only one
        byte-typed level, "sharing" it is meaningless.

        First-impl scope (session: "for first impl"): byte_repr="embed" and code_head_mode=
        "independent" only when share_levellm is on — BitPredictHead's chain-mode heads aren't
        wired into the shared-trunk path yet (asserted below), richer combinations are a
        follow-up, not this session's scope."""
        super().__init__()
        self.level = level
        self.in_dq = in_dq
        self.cfg = cfg
        self.window = window
        self.is_byte_level = level == 0
        D = cfg.tier_d_models[level]
        if shared_trunk is not None:
            assert cfg.byte_repr == "embed" and cfg.code_head_mode == "independent", (
                "v4.1 share_levellm's first impl only supports byte_repr='embed'/code_head_mode="
                "'independent' — BitPredictHead chain-mode heads aren't wired into the shared path yet"
            )
        if self.is_byte_level:
            assert cfg.byte_repr in ("bits", "embed")
            if cfg.byte_repr == "embed":
                self.byte_embed = nn.Embedding(cfg.vocab, D)
                self.ntp_head = nn.Linear(D, cfg.vocab)
            else:
                self.embed = nn.Linear(in_dq, D)
                self.ntp_head = build_bit_head(cfg, D, in_dq)
        elif shared_code_head is not None:
            # Config.share_code_head=True ablation: reuse ANOTHER code level's own embed/head/
            # code_pre weights outright — same objects, genuinely tied, not just same-shaped.
            assert in_dq == shared_code_head.in_dq and cfg.dqs[level] == cfg.dqs[shared_code_head.level], (
                "share_code_head requires uniform in_dq/dqs across every code level that shares"
            )
            self.embed = shared_code_head.embed
            self.ntp_head = shared_code_head.ntp_head
        else:
            assert cfg.code_head_mode in ("chain", "independent")
            self.embed = CodeEmbed(cfg, in_dq, D)
            if cfg.code_head_mode == "independent":
                self.ntp_head = nn.Linear(D, in_dq)
            else:
                self.ntp_head = build_bit_head(cfg, D, in_dq)

        if shared_trunk is not None:
            assert D == cfg.tier_d_models[shared_trunk.level], "share_levellm requires uniform tier_d_models"
            self.blocks = shared_trunk.blocks
            self.ln_f = shared_trunk.ln_f
        else:
            self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.tier_n_layers[level])])
            self.ln_f = nn.LayerNorm(D)

        if shared_code_head is not None:
            self.code_pre = shared_code_head.code_pre
        else:
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
            if shared_trunk is not None and getattr(shared_trunk, "fuse_d_model", None) is not None:
                # v4.1: the fusion module is part of the shared trunk too — SAME weights regardless
                # of which adjacent level-pair is fusing (extreme sharing's own logic extended to
                # fusion, not just self.blocks/ln_f). Requires shared_trunk's own fuse_d_model to
                # match (asserted implicitly by tier_d_models uniformity already required above,
                # since fuse_d_model IS the level-above's D, itself uniform under share_levellm).
                self.fuse_kv_proj = shared_trunk.fuse_kv_proj
                if cfg.fuse_use_null_kv:
                    self.fuse_null_kv = shared_trunk.fuse_null_kv
                if cfg.fuse_position in ("pre", "both"):
                    self.fuse_cross_pre = shared_trunk.fuse_cross_pre
                if cfg.fuse_position in ("post", "both"):
                    self.fuse_cross_post = shared_trunk.fuse_cross_post
            else:
                self.fuse_kv_proj = nn.Linear(fuse_d_model, D)
                if cfg.fuse_use_null_kv:
                    self.fuse_null_kv = nn.Parameter(torch.zeros(1, 1, D))
                    nn.init.normal_(self.fuse_null_kv, std=0.02)
                # "pre"/"post"/"both" each need their OWN CrossBlock (a "both" level gets two —
                # genuinely more params, see Config.fuse_position's own docstring). "concat" needs
                # NEITHER — self.blocks' own CausalSelfAttention layers derive K/V for the fused tail
                # directly from their own qkv weights (see _prep_concat), cheaper than the others.
                if cfg.fuse_position in ("pre", "both"):
                    self.fuse_cross_pre = CrossBlock(D, cfg.n_heads, cfg.mlp_mult)
                if cfg.fuse_position in ("post", "both"):
                    self.fuse_cross_post = CrossBlock(D, cfg.n_heads, cfg.mlp_mult)

    def _fuse(self, x: torch.Tensor, fuse_kv: torch.Tensor, cross_block: "CrossBlock") -> torch.Tensor:
        """x: [B, L, D] this level's own pre-self-attention embedding. fuse_kv: [B, n_blocks, D_above]
        the level-above's own hidden state (already, by construction, "as of" the raw-time position
        each block completes — see jagged_causal_mask_and_positions). cross_block: self.fuse_cross_pre
        or self.fuse_cross_post — which of the (up to two, under "both") CrossBlocks to run. Returns
        x, fused."""
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
        return cross_block(x, kv, attn_mask=disallow, rope_q=rope_q, rope_k=rope_k)

    def _prep_concat(self, x: torch.Tensor, fuse_kv: torch.Tensor):
        """"concat" fuse_position's own prep, done ONCE and shared by every self.blocks layer
        (unlike "pre"/"post"'s own CrossBlock, which owns its own weights per call) — projects+
        null-prepends fuse_kv (identical to _fuse's own kv construction) and builds the jagged
        visibility mask/rope-k positions once, since L/n_blocks/K are the same for every layer at
        this level. Returns (kv [B, Nf, D], disallow [L, Nf] bool True=blocked, rope_k or None) —
        passed straight into every Block.forward this level runs (see CausalSelfAttention's own
        fuse_kv/fuse_disallow/fuse_rope_k docstring for what each layer then does with it)."""
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
            assert self.fuse_d_model is not None, "fuse_kv passed but this LevelLM has no fuse module (no level above it?)"
            assert cfg.fuse_position in ("pre", "post", "both", "concat"), f"unknown fuse_position {cfg.fuse_position!r}"

        if fuse_kv is not None and cfg.fuse_position in ("pre", "both"):
            x = self._fuse(x, fuse_kv, self.fuse_cross_pre)

        concat_kv = concat_disallow = concat_rope_k = None
        if fuse_kv is not None and cfg.fuse_position == "concat":
            concat_kv, concat_disallow, concat_rope_k = self._prep_concat(x, fuse_kv)

        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, self.window, fuse_kv=concat_kv, fuse_disallow=concat_disallow, fuse_rope_k=concat_rope_k)

        if fuse_kv is not None and cfg.fuse_position in ("post", "both"):
            x = self._fuse(x, fuse_kv, self.fuse_cross_post)

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

        # v4: slice to the floor(L/K)*K-length COMPLETE-block prefix before viewing — training's L is
        # always exactly divisible by K (RefineLM.__init__ asserts it), so this is a no-op there, but
        # generation's L grows one byte at a time and is NOT always divisible; a trailing partial
        # block simply doesn't produce a code yet (correct: it isn't causally resolved regardless).
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        pre_q = self.code_pre(h_blocks[:, :, K - 1, :])
        if cfg.quant_type == "bsq":
            c_i = bsq_quantize(pre_q, cfg.dqs[self.level])
        elif cfg.quant_type == "identity":
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
        assert len(cfg.dqs) == self.n_levels
        assert len(cfg.tier_d_models) == self.n_levels
        assert len(cfg.tier_n_layers) == self.n_levels

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

        if cfg.share_levellm:
            assert len(set(cfg.tier_d_models)) == 1, "share_levellm requires uniform tier_d_models across every level"
            assert len(set(cfg.tier_n_layers)) == 1, "share_levellm requires uniform tier_n_layers across every level"
        if cfg.share_code_head and self.n_levels > 2:
            assert len(set(cfg.dqs[1:])) == 1, "share_code_head requires uniform dqs across every code level (levels 1+)"

        # v4.1 EXTREME WEIGHT SHARING (Config.share_levellm, default True — this file's whole
        # point): level 0 is built FIRST and, when sharing is on, every other level's self.blocks/
        # ln_f/fuse_* get REUSED from it (see LevelLM.__init__'s own shared_trunk docstring) — same
        # weight objects at every level, `window` stays genuinely per-level regardless (forward-time
        # arg now, not baked into construction). Config.share_code_head (default False, an
        # ablation) similarly ties every CODE level's (1..n_levels-1) own embed/ntp_head/code_pre to
        # the FIRST code level's — "bsq head and bsq linear map... different each level by default...
        # can be set weight sharing for ablation" (session ask). Build order matters: level 0, then
        # level 1 (owns the shared code head if share_code_head), then the rest.
        encoders: list[LevelLM] = []
        for i in range(self.n_levels):
            fuse_d_model = cfg.tier_d_models[i + 1] if i < self.n_levels - 1 else None
            fuse_kv_window = windows[i + 1] if i < self.n_levels - 1 else None
            shared_trunk = encoders[0] if (cfg.share_levellm and i > 0) else None
            shared_code_head = encoders[1] if (cfg.share_code_head and i > 1) else None
            encoders.append(LevelLM(
                cfg, i, in_dqs[i], windows[i],
                fuse_d_model=fuse_d_model, fuse_kv_window=fuse_kv_window,
                shared_trunk=shared_trunk, shared_code_head=shared_code_head,
            ))
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


def _sample_next_byte(model: "RefineLM", h_last: torch.Tensor) -> torch.Tensor:
    if model.cfg.byte_repr == "embed":
        logits = model.encoders[0].ntp_head(h_last)   # plain 256-way Linear — greedy argmax
        return logits.argmax(-1)
    logits = model.encoders[0].ntp_head(h_last, true_bits=None)   # chain mode greedy-decodes internally
    return bits_to_byte(logits)


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
        next_byte = _sample_next_byte(model, h_list[0][:, -1, :])
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
        token: [B] long (level 0) or [B, in_dq] float (level>0). Returns h_new [B, D_level]."""
        enc = model.encoders[level]
        D = cfg.tier_d_models[level]
        if level == 0 and cfg.byte_repr == "embed":
            x = enc.byte_embed(token).unsqueeze(1)
        elif level == 0:
            x = enc.embed(byte_to_bits(token)).unsqueeze(1)
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
        pre_q = enc.code_pre(h_new)
        c_new = bsq_quantize(pre_q, cfg.dqs[level]) if cfg.quant_type == "bsq" else pre_q
        block_idx[level] += 1
        resolved_pos = block_idx[level] * Ks[level] - 1   # matches jagged_causal_mask_and_positions'
                                                             # own block_pos=(b+1)*K-1 exactly
        h_above = clean_step(level + 1, c_new)
        maybe_emit_code(level + 1, h_above)
        if level == 0 and fuse_on:
            new_row = model.encoders[0].fuse_kv_proj(h_above).unsqueeze(1)   # [B, 1, D0]
            fuse_kv_rows = torch.cat([fuse_kv_rows, new_row], dim=1)
            fuse_k_pos = torch.cat([fuse_k_pos, torch.tensor([resolved_pos], device=device)])

    def apply_fuse(x: torch.Tensor, pos: int, cross_block) -> torch.Tensor:
        """The fuse_cross step itself, factored out so fused_step can call it before/after
        self.blocks, and (under "both") both — matches LevelLM.forward's own pre/post/both
        dispatch exactly. cross_block: enc0.fuse_cross_pre or enc0.fuse_cross_post."""
        enc0 = model.encoders[0]
        D = cfg.tier_d_models[0]
        head_dim = D // cfg.n_heads
        rope_q = rope_k = None
        if cfg.cross_attn_rope:
            rope_q = rope_cos_sin_for_positions(torch.tensor([pos], device=device), head_dim, cfg.rope_base, device)
            rope_k = rope_cos_sin_for_positions(fuse_k_pos, head_dim, cfg.rope_base, device)
        return cross_block(x, fuse_kv_rows, attn_mask=None, rope_q=rope_q, rope_k=rope_k)

    def fused_step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        """Advance level 0's FUSED pass by one byte — the actual sampling path."""
        enc0 = model.encoders[0]
        D = cfg.tier_d_models[0]
        x = (enc0.byte_embed(byte_id) if cfg.byte_repr == "embed" else enc0.embed(byte_to_bits(byte_id))).unsqueeze(1)
        if fuse_on and cfg.fuse_position in ("pre", "both"):
            x = apply_fuse(x, pos, enc0.fuse_cross_pre)
        head_dim = D // cfg.n_heads
        cos_new, sin_new = rope_cos_sin_at(pos, head_dim, cfg.rope_base, device)
        # "concat" mode: fuse_kv_rows IS already this level's own D-dim, null-prepended KV
        # (identical to what training's own _prep_concat computes each forward) — no extra
        # projection needed here, just feed it straight into every block's forward_step. No
        # explicit visibility mask needed either — see the module docstring's own reasoning
        # (every row present is, by construction, already causally valid at this step).
        concat_kv = concat_rope_k = None
        if fuse_on and cfg.fuse_position == "concat":
            concat_kv = fuse_kv_rows
            if cfg.cross_attn_rope:
                concat_rope_k = rope_cos_sin_for_positions(fuse_k_pos, head_dim, cfg.rope_base, device)
        for li, block in enumerate(enc0.blocks):
            x, fused_cache_k[li], fused_cache_v[li] = block.forward_step(
                x, cos_new, sin_new, fused_cache_k[li], fused_cache_v[li],
                fuse_kv=concat_kv, fuse_rope_k=concat_rope_k,
            )
        if fuse_on and cfg.fuse_position in ("post", "both"):
            x = apply_fuse(x, pos, enc0.fuse_cross_post)
        return enc0.ln_f(x).squeeze(1)

    if fuse_on:
        D0 = cfg.tier_d_models[0]
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
        next_byte = _sample_next_byte(model, last_fused_h)
        out_bytes.append(next_byte.unsqueeze(1))
        pos = L0 + i
        h_clean = clean_step(0, next_byte)
        maybe_emit_code(0, h_clean)
        last_fused_h = fused_step(next_byte, pos)

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
    p.add_argument("--fusion_ntp_weight", type=float, default=1.0)
    p.add_argument("--quant_type", type=str, default="bsq", choices=["bsq", "identity"])
    p.add_argument("--code_embed_mode", type=str, default="linear", choices=["linear", "mlp", "pq_table"])
    p.add_argument("--fuse_encoder_levels", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--fuse_position", type=str, default="pre", choices=["pre", "post", "both", "concat"])
    p.add_argument("--fuse_use_null_kv", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--share_levellm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--share_code_head", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--byte_repr", type=str, default="bits", choices=["bits", "embed"])
    p.add_argument("--code_head_mode", type=str, default="chain", choices=["chain", "independent"])
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
        fusion_ntp_weight=args.fusion_ntp_weight,
        quant_type=args.quant_type, code_embed_mode=args.code_embed_mode, fuse_encoder_levels=args.fuse_encoder_levels,
        fuse_position=args.fuse_position, fuse_use_null_kv=args.fuse_use_null_kv,
        share_levellm=args.share_levellm, share_code_head=args.share_code_head,
        byte_repr=args.byte_repr, code_head_mode=args.code_head_mode,
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
    log(f"Ks={cfg.Ks} dqs={cfg.dqs} tier_d_models={cfg.tier_d_models} tier_n_layers={cfg.tier_n_layers} "
        f"seq_lens={model.seq_lens} context_len={cfg.context_len} fuse_encoder_levels={cfg.fuse_encoder_levels} "
        f"params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
