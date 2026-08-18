"""qcute_v5_concat: now the default v5 concat module (promoted from qcute_v5_concat_modes.py, which
itself forked from this file to add multi-mode decode loss -- see that section below; the
pre-multi-mode version is kept as qcute_v5_concat_no_modes.py). Originally promoted from
qcute_v5_concat_skip, forked from qcute_v5_concat_fixblock.py, adding one more pruning on top of
the fixblock changes (decode_bos removed, block 0's first K-1 targets excluded from decode's loss --
see qcute_v5_concat_fixblock.py's own docstring for that).

**_skip's own addition**: once a block's code exists, that block's raw bytes are redundant in the
decode buffer -- the code is supposed to be a sufficient summary of the block, so keeping both the
raw bytes AND the code as separately-attendable buffer entries wastes context/compute. A raw byte
key (buffer category 0) is now visible ONLY to queries still inside that byte's own (self-track,
K0-sized) block; once a query has moved past that block, only the block's CODE stays visible
(`code_pos<=query_pos`, already enforced elsewhere, unaffected by this). Pure function of (query
true_pos, key true_pos, K0) -- static/shape-only, added as one more boolean term alongside the
existing causal/window masks in `_merged_layout`/`_merged_decode_forward` (both the dense and
chunked paths), no new runtime cost for training. `generate_true_kv_cache` gets the REAL payoff:
once a block's code is appended, that block's raw byte K/V entries are evicted from the cache
immediately (not just once they fall outside the window) -- the running byte K/V footprint shrinks
from O(current position) to O(K0) at any time, on top of the encode side's own windowed cache. This
eviction is safe for KV-cache use because it's deterministic and never needs to be undone: a query
row's computed value never depends on what happens after it, and a byte's visibility from ANY given
query is decided purely by (that query's position, that byte's block, K0) -- never by how much more
sequence follows, so nothing computed before an eviction ever needs revisiting after it.

qcute_v5_concat.py's own docstring (chronological merged-interleave packing) still applies
otherwise -- every track's codes are placed at their true time position (a code lands physically
right after the last byte of the block it summarizes), merged into ONE physically time-ordered
buffer per level's decode. Because buffer order now IS time order:
  - causal masking is a plain buffer-index comparison (i>=j) -- no same-position tie-break needed,
    since a code always sorts strictly after the byte that produced it, so it's automatically
    invisible to that byte for free (previously handled by an explicit same_pos_code_excluded mask
    term).
  - windowed/banded attention slices CONTIGUOUS buffer ranges directly (see _merged_layout,
    _merged_decode_forward) -- no runtime argsort (unlike qcute_v5_concat_slow.py's banded path,
    which grouped-then-corrected and needed an argsort to restore time-adjacency before chunking).
    The one-time index/address construction is still a sort, but it depends only on shape
    (L, per-track (K, window)), never on data, so it's built once and cached (self._merged_cache),
    identically for training (fixed L every step) and generate_kv_cache's fixed-size FIFO window
    (same L reused every generation step).
  - every level's window is per-track (attn_window's existing per-level decode_window convention,
    unchanged) -- NOT a single global scalar. A byte key uses the self/track-0 window; a level-j
    code key uses that track's own window; each key's visibility is `query_true_pos - key_true_pos
    < that key's own window`, independently per track (matching qcute_v5_stack.py's attn_window
    semantics, no factor-of-2 fudge unlike the old "prepend" mask).
Single- and multi-track decode are now ONE mechanism (no more separate selfcode/dense/banded code
paths) -- a single track is just the T=1 case. See docs/status.md for the design discussion this
implements. qcute_v5_concat_slow.py is kept as the O(L^2) dense reference this is checked against
(scripts/test_v5_concat.py). Adds Config.quant_type: "softmax" (default, unchanged categorical
code_head + gumbel/argmax) or "bsq" (binary spherical quantization, Config.bsq_bits-wide sign code,
straight-through).

Every encode_lm/decode_lm is its own independent weight instance -- weight-sharing logic
(share_level_weights) has been pruned entirely; see qcute_v5_concat_ws.py for that variant, kept
as a reference.

**Multi-mode decode loss (promoted from the qcute_v5_concat_modes.py fork)** --
qcute_v5_stack.py's staged cross-attention decode gets a loss at every conditioning depth (self
only, self+track1, self+track1+track2, ...) for free, as a byproduct of its sequential per-track
stages (decode_stage_extra_losses). This module's decode has no such byproduct on its own -- it's
ONE flat self-attention pass over a merged buffer holding every track's codes at once, so there's
no natural intermediate readout to tap. Config.multi_mode_impl selects how (or whether) to get
those same per-depth losses:
  - "off" (default): one loss per level (the full/deepest mode only), zero overhead -- bit-exact
    with the pre-multi-mode behavior (qcute_v5_concat_no_modes.py, kept as that no-op reference,
    checked in scripts/test_v5_concat_modes.py).
  - "multipass": naive reference -- calls _merged_decode_forward once per mode m=1..T (T = number
    of available tracks), each with tracks[:m], literally re-running the whole per-level decode
    forward T times. Correct by construction (identical to calling max_decode_sources=m T times),
    used only to verify "single_pass" below.
  - "single_pass": every mode m=1..T gets its OWN independent merged buffer (its own
    _merged_layout(L, tracks_meta[:m], device) segment, built from tracks[:m] exactly as a
    standalone max_decode_sources=m call would), concatenated along the sequence dim with the
    other modes' segments and run through one shared self-attention pass per layer, using a
    block-diagonal mask (zero cross-segment attention -- no mode's keys/queries ever see another
    mode's entries). This is mathematically identical to "multipass" (T independent calls), just
    batched into fewer kernel launches; a duplicate-query-reading-a-shared-backbone design was
    tried first and rejected -- it doesn't preserve exactness at same-timestep code ties across
    modes, since a shallower mode's own hidden state at a tied position needs to be computed from
    scratch, not read off the (already more-contaminated) shared backbone.
    Supports both the dense and the chunked/banded (SWA) attention path:
      - dense: _merged_layout_multimode builds the block-diagonal structure once (cached per
        (L, tracks_meta, device)) and _merged_decode_forward_multimode runs it directly.
      - chunked: _merged_layout extended with forced_sc/forced_n_chunks/forced_n_prev_chunks
        (forcing a chunk grid larger than a segment's own natural need is always safe, pure
        padding via the same true_pos=-1e9/window=0 sentinel scheme already used for Lp padding;
        forcing smaller is asserted against). _merged_layout_multimode_chunked finds the shared
        grid every mode's segment can be padded onto, and _merged_decode_forward_multimode_chunked
        batches all T modes into one SDPA call per layer -- the chunked path's own
        (B, n_chunks)-flattened-batch trick extended with an added "mode" axis.
        _merged_decode_forward_multimode auto-dispatches between the two based on whether any
        segment needs chunking.
      - _merged_decode_forward_multimode_looped (option 1): trivial per-mode reference -- T calls
        to the existing, unmodified _merged_decode_forward -- used to verify the chunked batched
        path (option 2) directly, independent of "multipass"'s own LevelLM.forward plumbing.
    Verified exact (scripts/test_v5_concat_modes.py) against "multipass" across Ks=(1,), (4,1),
    (2,2,1), both dense (attn_window=-1) and chunked (real finite window) -- all-mode logits/
    losses match under quant_type="softmax" code_sample_mode="ste" (the only combination that's
    deterministic enough for an exact per-mode comparison).
Per-mode losses for m<T are collected the same way qcute_v5_stack.py's decode_stage_extra_losses
are -- summed into decode_stage_extra_total, weighted by mean(decode_ntp_weight), added to the
total loss -- so RefineLM.forward barely changes; it already had this field's plumbing designed to
be shared with qcute_v5_stack.py's per-level metrics dict shape.

    uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_3.py
    uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_modes_ks221.py
"""
import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
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
    Ks: tuple[int, ...] = (32, 32)
    d_model: int = 256
    n_layers: int = 2
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    # scalar, or per-level tuple; each level entry is scalar or (encode_window, decode_window),
    # where decode_window is scalar or a per-source tuple [self, +1, ..., top]; -1 = unbounded, 0 = drop that source
    attn_window: int | tuple[int, ...] = 32
    rope_base: float = 10000.0
    byte_ntp_weight: float = 1.0
    code_ntp_weight: float = 1.0
    decode_ntp_weight: float = 1.0
    gumbel_tau: float = 1.0
    vocab: int = 256
    code_extract_mode: str = "last_h"
    code_head_tied: bool = False
    ntp_head_tied: bool = False    # softmax only, level1+ ONLY: False (default) gives the decode
                                    # NTP softmax head (ntp_loss_acc/sample_next) its own separate
                                    # Linear(D,V), untied from stage_lm.embed.weight. True restores
                                    # classic weight tying. BSQ/FSQ are unaffected -- they already
                                    # use a dedicated code_predict head unconditionally, never
                                    # embed.weight. Level0 (byte-level) is ALSO unaffected: every
                                    # LevelLM's _ntp_loss_acc has a separate is_byte_level branch
                                    # that always uses that stage's own embed.weight directly,
                                    # bypassing self.quant.ntp_loss_acc entirely, regardless of
                                    # this flag or quant_type -- this flag only ever changes level1+
                                    # (non-byte, code-predicting) levels' NTP head.
    quant_type: str = "softmax"   # "softmax" (categorical code_head + gumbel/argmax), "bsq", or "fsq"
    bsq_bits: int = 4             # code width in bits when quant_type="bsq"
    bsq_lfq: bool = False          # True regresses BSQ to plain LFQ (no L2-normalize before sign,
                                    # no 1/sqrt(dq) rescale -- hypercube corners, not hypersphere).
                                    # Default False = unchanged BSQ behavior.
    entropy_reg_weight: float = 0.0   # bsq/softmax only (see bsq_entropy_reg/softmax_entropy_reg):
                                       # weight on the code-usage entropy regularization term,
                                       # summed over levels and added to the total loss. 0.0
                                       # (default, matches archived qcutelm.py's own default) = off.
                                       # No-op for quant_type="fsq" (QuantScheme.entropy_reg
                                       # defaults to None, not implemented for FSQ).
    fsq_dq: int = 6                # code width in dims when quant_type="fsq" (archived qcutelm.py default)
    fsq_levels: int = 8            # levels per dim when quant_type="fsq" (archived qcutelm.py default)
    fsq_bound: str = "sigmoid"     # "sigmoid" (default -- iFSQ, 2*sigmoid(1.6*v)-1, archived
                                    # qcutelm_vlt6.py's default) or "tanh" (original FSQ, Mentzer
                                    # et al. 2023) -- the bounding nonlinearity fsq_quantize applies
                                    # before rounding
    code_sample_mode: str = "ste"   # "ste" (deterministic hard forward, plain softmax/sign(),
                                     # straight-through backward -- default), "sample" (stochastic
                                     # hard forward -- Gumbel noise for softmax, Bernoulli(sigmoid)
                                     # for bsq -- still straight-through backward), or "soft" (no
                                     # hard forward at all: plain Gumbel-Softmax relaxation for
                                     # softmax, raw normalized vector for bsq -- Jang et al. 2016)
    multi_mode_impl: str = "off"    # "off" (default, unchanged qcute_v5_concat.py behavior),
                                     # "multipass" (naive T-separate-forward-calls reference), or
                                     # "single_pass" (one shared pass, query-copy trick -- dense
                                     # attention path only, see module docstring)


def gumbel_quantize(logits: torch.Tensor, tau: float, mode: str = "ste") -> torch.Tensor:
    if mode in ("sample", "soft"):
        eps = torch.finfo(logits.dtype).tiny
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(u))
        soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    else:
        soft = F.softmax(logits / tau, dim=-1)
    if mode == "soft":
        return soft
    hard = F.one_hot(soft.argmax(-1), num_classes=logits.shape[-1]).to(soft.dtype)
    return soft + (hard - soft).detach()


def bsq_quantize(v: torch.Tensor, dq: int, mode: str = "ste", lfq: bool = False) -> torch.Tensor:
    """lfq=True regresses BSQ to plain LFQ: skips the L2-normalize-to-unit-sphere step, signing the
    raw projection directly (hypercube corners {-1,+1}^dq instead of BSQ's hypersphere corners
    ||z_hat||=1) and skips the 1/sqrt(dq) rescale (only meaningful once normalized) -- ported from
    archived qcute/archive/qcutelm.py's bsq_quantize. Sign bits (used by to_ids/CodeEmbed/
    ntp_loss_acc's targets) are unaffected by normalization either way, only z_hat's geometry/scale
    changes. Default (lfq=False) is unchanged BSQ behavior."""
    v_eff = v if lfq else F.normalize(v, dim=-1)
    scale = 1.0 if lfq else 1.0 / math.sqrt(dq)
    if mode == "soft":
        return v_eff * scale
    if mode == "sample":
        probs = torch.sigmoid(v_eff)
        u = torch.rand_like(v_eff)
        hard = torch.where(u < probs, torch.ones_like(v_eff), -torch.ones_like(v_eff))
    else:
        hard = torch.sign(v_eff)
    return (v_eff + (hard - v_eff).detach()) * scale


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def bsq_entropy_reg(v: torch.Tensor) -> torch.Tensor:
    """LFQ/BSQ-style entropy regularization (Yu et al. 2023 SS3.2, MAGVIT-v2; also BSQ 2024's
    closed-form version), ported from archived qcute/archive/qcutelm.py -- both papers' recipes
    lean on this to keep code usage spread out; without it BSQ is prone to collapsing onto a
    dominant code. Two opposing pressures on the per-bit Bernoulli probabilities p=sigmoid(v):
    (1) minimize each *example's* bit entropy -- push predictions toward confident/decisive
    corners, matching the hard quantization boundary; (2) maximize the *batch-averaged* bit-usage
    entropy -- spread which corners get used across examples, directly countering collapse onto
    one dominant code. Returns E_batch[H(p)] - H(E_batch[p]) (minimize this: pulls per-example
    entropy down, batch-average entropy up). Applied to the RAW pre-quantization projection (v,
    before normalize/sign), matching the archived usage."""
    probs = torch.sigmoid(v)
    per_example = bernoulli_entropy(probs).sum(-1).mean()
    batch_avg = probs.reshape(-1, probs.size(-1)).mean(0)
    batch = bernoulli_entropy(batch_avg).sum()
    return per_example - batch


def softmax_entropy_reg(logits: torch.Tensor) -> torch.Tensor:
    """Same E_batch[H(p)] - H(E_batch[p]) structure as bsq_entropy_reg, generalized to a single
    V-way categorical code instead of dq independent Bernoulli bits -- exact categorical entropy
    here (one joint distribution), not a per-bit marginal proxy. Minimize this: pulls each
    example's softmax toward a confident/decisive one-hot (matching the hard
    argmax/gumbel-one-hot boundary) while pushing the batch-averaged category usage toward
    uniform, countering collapse onto a dominant code."""
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    per_example = -(probs * logp).sum(-1).mean()
    batch_avg = probs.reshape(-1, probs.size(-1)).mean(0)
    batch = -(batch_avg * batch_avg.clamp_min(1e-9).log()).sum()
    return per_example - batch


def fsq_quantize(v: torch.Tensor, L: int, mode: str = "ste", bound: str = "sigmoid") -> torch.Tensor:
    """Finite scalar quantization, ported from archived qcute/archive/qcutelm*.py. `bound` selects
    the squashing nonlinearity applied to v's dq dims before rounding to one of L integer levels
    (half_l=(L-1)/2 either way), STE backward -- same idiom as bsq_quantize/gumbel_quantize above:
      - "sigmoid" (default): iFSQ, half_l*(2*sigmoid(1.6*v)-1) -- archived qcutelm_vlt6.py's
        default variant.
      - "tanh": original FSQ (Mentzer et al. 2023), half_l*tanh(v).
    "sample" adds uniform dequantization-style dithering noise before rounding (FSQ has no natural
    stochastic analogue to Gumbel/Bernoulli sampling, since there's no softmax/sigmoid probability
    to sample from -- this is the closest equivalent, letting the rounding boundary itself vary).
    "soft" skips rounding/STE entirely, returning the bounded continuous value. Output normalized
    to roughly [-1, 1] (divided by half_l) to match bsq_quantize's unit-ish output scale, so
    code_sample_mode's STE/proxy idioms stay comparable in magnitude across quant_type."""
    half_l = (L - 1) / 2
    z_bounded = half_l * (torch.tanh(v) if bound == "tanh" else (2 * torch.sigmoid(1.6 * v) - 1))
    if mode == "soft":
        return z_bounded / half_l
    if mode == "sample":
        noise = torch.rand_like(z_bounded) - 0.5
        z_rounded = torch.round(z_bounded + noise).clamp(-half_l, half_l)
    else:
        z_rounded = torch.round(z_bounded)
    z_hat = z_bounded + (z_rounded - z_bounded).detach()
    return z_hat / half_l


MAX_PQ_TABLE_DQ = 16   # 2**16 = 65536 rows -- ceiling bsq_bits this table lookup allows


class CodeEmbed(nn.Module):
    """Maps a bsq code (one of 2**dq discrete hypersphere corners) to a D-dim vector via an exact
    table lookup, not a linear combination of its +-1 components (default/only mode for bsq's code
    consumption -- ported from qcute_refine_v2's pq_table, see docs/archive2/kv_contribution.md).
    A naive lookup is non-differentiable in the code, severing the only gradient path back to the
    code's own producer; fixed with the same straight-through idiom bsq_quantize uses: a continuous
    `proxy` linear carries the backward gradient, the table row is what's used forward."""

    def __init__(self, dq: int, D: int):
        super().__init__()
        assert dq <= MAX_PQ_TABLE_DQ, f"CodeEmbed: dq={dq} would need a 2**{dq}-row table -- keep dq<={MAX_PQ_TABLE_DQ}."
        self.table = nn.Embedding(2 ** dq, D)
        self.proxy = nn.Linear(dq, D)
        self.register_buffer("_powers", (2 ** torch.arange(dq)).long(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = ((x > 0).long() * self._powers).sum(-1)
        hard = self.table(idx)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


class FSQEmbed(nn.Module):
    """Maps an FSQ code (dq per-dim level indices in [0, L)) to a D-dim vector via a per-dim
    embedding table summed across dims -- compositional (PQ-style), generalizes to any of the
    L**dq combinations for free, unlike a per-whole-code table (which for FSQ's typical L=8,
    dq=6 would need 8**6=262144 rows -- ported from archived qcute/archive/qcutelm.py's
    FactorizedCodeEmbedding, generalizing CodeEmbed's binary/BSQ-only table lookup above to L
    levels). x is fsq_quantize's normalized [-1,1] output; STE via the same proxy idiom CodeEmbed
    uses, since the per-dim level lookup is non-differentiable in x."""

    def __init__(self, dq: int, L: int, D: int):
        super().__init__()
        self.dq, self.L = dq, L
        self.table = nn.Parameter(torch.zeros(dq, L, D))
        nn.init.normal_(self.table, std=0.02)
        self.proxy = nn.Linear(dq, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bound = (self.L - 1) / 2
        levels = (x * bound + bound).round().long().clamp(0, self.L - 1)
        dq_idx = torch.arange(self.dq, device=x.device)
        hard = self.table[dq_idx, levels].sum(dim=-2)
        proxy = self.proxy(x)
        return proxy + (hard - proxy).detach()


class QuantScheme:
    """Uniform interface for code quantization/embedding/prediction, so LevelLM and generation code
    dispatch through one polymorphic instance (self.quant / stage_lm.quant) instead of branching on
    cfg.quant_type at every call site. SoftmaxQuant/BSQQuant below are the only two implementations;
    Config.quant_type selects which one gets built, once, in make_quant."""

    def init_modules(self, D: int, V: int, code_head_tied: bool) -> tuple:
        """-> (code_head, code_embed, code_predict) nn.Module|None, assigned onto a fresh LevelLM."""
        raise NotImplementedError

    def quantize(self, pre_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_ids(self, source_c: torch.Tensor) -> torch.Tensor:
        """Display-only: packs a bsq bit-vector into an int; argmax for softmax's one-hot."""
        raise NotImplementedError

    def embed_for_decode(self, stage_lm: "LevelLM", source_c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ntp_loss_acc(self, stage_lm: "LevelLM", h_query: torch.Tensor,
                      target_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed_input(self, stage_lm: "LevelLM", seq_repr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_next(self, stage_lm: "LevelLM", h_query: torch.Tensor, vocab: int) -> torch.Tensor:
        raise NotImplementedError

    def entropy_reg(self, pre_q: torch.Tensor) -> torch.Tensor | None:
        """Optional code-usage entropy regularization term computed from the raw pre-quantization
        projection (before quantize()'s normalize/round/sign). None (default -- no term) for
        schemes that don't have one; only BSQQuant overrides this."""
        return None


class SoftmaxQuant(QuantScheme):
    def __init__(self, tau: float, mode: str = "ste", ntp_head_tied: bool = False):
        self.tau = tau
        self.mode = mode
        self.ntp_head_tied = ntp_head_tied

    def init_modules(self, D, V, code_head_tied):
        code_head = None if code_head_tied else nn.Linear(D, V, bias=False)
        if code_head is not None:
            nn.init.normal_(code_head.weight, std=0.02)
        # ntp_head reuses the code_predict slot (unused by softmax otherwise) -- the decode NTP
        # softmax head, untied from embed.weight unless ntp_head_tied=True.
        ntp_head = None if self.ntp_head_tied else nn.Linear(D, V, bias=False)
        if ntp_head is not None:
            nn.init.normal_(ntp_head.weight, std=0.02)
        return code_head, None, ntp_head

    def quantize(self, pre_q):
        return gumbel_quantize(pre_q, self.tau, self.mode)

    def to_ids(self, source_c):
        return source_c.argmax(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return source_c @ stage_lm.embed.weight

    def _ntp_logits(self, stage_lm, h_query):
        if self.ntp_head_tied:
            return F.linear(h_query, stage_lm.embed.weight)
        return stage_lm.code_predict(h_query)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target = target_repr.argmax(-1).reshape(-1)
        logits = self._ntp_logits(stage_lm, h_query)
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return seq_repr @ stage_lm.embed.weight

    def sample_next(self, stage_lm, h_query, vocab):
        logits = self._ntp_logits(stage_lm, h_query)
        next_id = logits.argmax(-1)
        return F.one_hot(next_id, num_classes=vocab).to(h_query.dtype)

    def entropy_reg(self, pre_q):
        return softmax_entropy_reg(pre_q)


class BSQQuant(QuantScheme):
    def __init__(self, bsq_bits: int, mode: str = "ste", lfq: bool = False):
        self.bsq_bits = bsq_bits
        self.mode = mode
        self.lfq = lfq

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.bsq_bits, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = CodeEmbed(self.bsq_bits, D)
        code_predict = nn.Linear(D, self.bsq_bits, bias=False)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        return bsq_quantize(pre_q, self.bsq_bits, self.mode, self.lfq)

    def to_ids(self, source_c):
        bits = (source_c > 0).long()
        weights = 2 ** torch.arange(bits.shape[-1], device=bits.device)
        return (bits * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        target_bits = (target_repr.reshape(-1, self.bsq_bits) > 0).float()
        pred = stage_lm.code_predict(h_query)
        loss = F.binary_cross_entropy_with_logits(pred, target_bits)
        with torch.no_grad():
            acc = ((pred > 0).float() == target_bits).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        pred = stage_lm.code_predict(h_query)
        return bsq_quantize(pred, self.bsq_bits, self.mode, self.lfq)

    def entropy_reg(self, pre_q):
        return bsq_entropy_reg(pre_q)


class FSQQuant(QuantScheme):
    def __init__(self, dq: int, L: int, mode: str = "ste", bound: str = "sigmoid"):
        self.dq, self.L, self.mode, self.bound = dq, L, mode, bound

    def init_modules(self, D, V, code_head_tied):
        code_head = nn.Linear(D, self.dq, bias=False)
        nn.init.normal_(code_head.weight, std=0.02)
        code_embed = FSQEmbed(self.dq, self.L, D)
        code_predict = nn.Linear(D, self.dq * self.L)
        nn.init.normal_(code_predict.weight, std=0.02)
        return code_head, code_embed, code_predict

    def quantize(self, pre_q):
        return fsq_quantize(pre_q, self.L, self.mode, self.bound)

    def to_ids(self, source_c):
        bound = (self.L - 1) / 2
        levels = (source_c * bound + bound).round().long().clamp(0, self.L - 1)
        weights = self.L ** torch.arange(self.dq, device=levels.device)
        return (levels * weights).sum(-1)

    def embed_for_decode(self, stage_lm, source_c):
        return stage_lm.code_embed(source_c)

    def ntp_loss_acc(self, stage_lm, h_query, target_repr):
        bound = (self.L - 1) / 2
        target_levels = (target_repr.reshape(-1, self.dq) * bound + bound).round().long().clamp(0, self.L - 1)
        pred = stage_lm.code_predict(h_query).reshape(-1, self.dq, self.L)
        loss = F.cross_entropy(pred.reshape(-1, self.L), target_levels.reshape(-1))
        with torch.no_grad():
            acc = (pred.argmax(-1) == target_levels).float().mean()
        return loss, acc

    def embed_input(self, stage_lm, seq_repr):
        return stage_lm.code_embed(seq_repr)

    def sample_next(self, stage_lm, h_query, vocab):
        bound = (self.L - 1) / 2
        pred = stage_lm.code_predict(h_query).reshape(*h_query.shape[:-1], self.dq, self.L)
        if self.mode == "sample":
            probs = F.softmax(pred, dim=-1)
            levels = torch.multinomial(probs.reshape(-1, self.L), 1).reshape(pred.shape[:-1])
        else:
            levels = pred.argmax(-1)
        return (levels.float() - bound) / bound


def make_quant(cfg: "Config") -> QuantScheme:
    if cfg.quant_type == "bsq":
        return BSQQuant(cfg.bsq_bits, cfg.code_sample_mode, cfg.bsq_lfq)
    if cfg.quant_type == "fsq":
        return FSQQuant(cfg.fsq_dq, cfg.fsq_levels, cfg.code_sample_mode, cfg.fsq_bound)
    return SoftmaxQuant(cfg.gumbel_tau, cfg.code_sample_mode, cfg.ntp_head_tied)


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def chunked_windowed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    """Causal attention restricted to (i - j < window) for a PLAIN (non-interleaved) sequence,
    computed via chunking (gather current + previous chunk, O(T*window)) instead of materializing
    a dense T x T mask (O(T^2)). Produces bit-identical results to the dense
    causal & (i-j<window) mask -- see docs/status.md's windowed-attention efficiency review.

    Fast path requires T % window == 0 (always true during training, where context_len is built
    to divide evenly -- see RefineLM.__init__); falls back to the dense mask (correct for any T,
    same as the original qcute_v5_concat.py) otherwise, which no-cache generation's ragged, growing
    T can hit.
    """
    B, H, T, hd = q.shape
    w = window
    device = q.device
    if T <= w:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if T % w != 0:
        pos = torch.arange(T, device=device)
        ti, tj = pos.unsqueeze(1), pos.unsqueeze(0)
        attn_mask = ((tj <= ti) & (ti - tj < w)).view(1, 1, T, T)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    n_chunks = T // w
    qb = q.view(B, H, n_chunks, w, hd)
    kb = k.view(B, H, n_chunks, w, hd)
    vb = v.view(B, H, n_chunks, w, hd)
    pad_k = torch.zeros(B, H, 1, w, hd, device=device, dtype=k.dtype)
    pad_v = torch.zeros(B, H, 1, w, hd, device=device, dtype=v.dtype)
    k_ext = torch.cat([pad_k, kb], dim=2)
    v_ext = torch.cat([pad_v, vb], dim=2)

    idx = torch.arange(n_chunks, device=device).view(n_chunks, 1) + torch.arange(2, device=device).view(1, 2)
    k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)
    v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)

    pos = torch.arange(T, device=device)
    pos_b = pos.view(n_chunks, w)
    pad_pos = torch.full((1, w), -10 ** 9, device=device, dtype=pos.dtype)
    pos_ext = torch.cat([pad_pos, pos_b], dim=0)
    pos_win = pos_ext[idx].reshape(n_chunks, 2 * w)

    ti = pos_b.unsqueeze(-1)
    tj = pos_win.unsqueeze(1)
    allow = (tj <= ti) & (ti - tj < w)
    mask_flat = allow.view(1, n_chunks, 1, w, 2 * w).expand(B, n_chunks, 1, w, 2 * w).reshape(B * n_chunks, 1, w, 2 * w)

    qb_flat = qb.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, w, hd)
    k_win_flat = k_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)
    v_win_flat = v_win.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)

    y = F.scaled_dot_product_attention(qb_flat, k_win_flat, v_win_flat, attn_mask=mask_flat)
    return y.view(B, n_chunks, H, w, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None:
            y = chunked_windowed_attention(q, k, v, window)
        else:
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        a = self.attn(self.ln1(x), cos, sin, window)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


_warned_thin_window: set[tuple[int, ...]] = set()


def _warn_thin_window(tracks: list[tuple[torch.Tensor, int, int | None]], window: int, min_codes: int = 2) -> None:
    """Cumulative code coverage a banded decode window actually buys, across all tracks (levels):
    codes_in_window[j] ~= window // K_j, summed. Below min_codes the window is thinner than the
    minimum useful conditioning (2 codes = the self-code LM-continuation floor used everywhere
    else in this codebase) -- almost certainly starving the model of the coarser-level context the
    multi-track decode exists to provide. Not a correctness check (attention is exact for whatever
    window is set); this is due diligence the caller owns, same spirit as sample_context's warning."""
    key = tuple(K for _, K, _ in tracks) + (window,)
    if key in _warned_thin_window:
        return
    total_codes = sum(max(0, window // K) for _, K, _ in tracks)
    if total_codes < min_codes:
        _warned_thin_window.add(key)
        Ks_str = ",".join(str(K) for _, K, _ in tracks)
        print(f"WARNING: banded decode window={window} covers only ~{total_codes} cumulative code(s) "
              f"across tracks Ks=({Ks_str}) -- below min_codes={min_codes}. Coarser-level context is "
              f"likely starved; consider a larger attn_window decode_window for these levels.")


class LevelLM(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.quant = make_quant(cfg)
        D = cfg.d_model
        V = cfg.vocab
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.code_head, self.code_embed, self.code_predict = self.quant.init_modules(D, V, cfg.code_head_tied)
        self.code_query = self.code_out = self.query_embed = None
        if cfg.code_extract_mode == "light_query_attn":
            self.code_query = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.code_query, std=0.02)
            self.code_out = nn.Linear(D, D, bias=False)
        elif cfg.code_extract_mode == "query_embed":
            self.query_embed = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.query_embed, std=0.02)
        self._merged_cache: dict = {}   # (L, tracks_meta, device) -> structural tensors, see _merged_layout
        self._merged_multimode_cache: dict = {}   # (L, tracks_meta, device) -> extended-mask tensors, see _merged_layout_multimode
        self._merged_multimode_chunked_cache: dict = {}   # (L, tracks_meta, device) -> shared chunk-grid tensors, see _merged_layout_multimode_chunked

    def _classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.code_head(pooled) if self.code_head is not None else F.linear(pooled, self.embed.weight)

    def _query_embed_pool(self, x0: torch.Tensor, K: int, n_blocks: int, window: int | None) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device

        x0_blocks = x0.view(B, n_blocks, K, D)
        q_tok = self.query_embed.view(1, 1, 1, D).expand(B, n_blocks, 1, D)
        xe = torch.cat([x0_blocks, q_tok], dim=2).view(B, n_blocks * (K + 1), D)
        Le = n_blocks * (K + 1)

        slot = torch.arange(K + 1, device=device).repeat(n_blocks)
        is_query = slot == K
        block_of = torch.arange(n_blocks, device=device).repeat_interleave(K + 1)
        within_block_pos = torch.where(is_query, torch.full_like(slot, K - 1), slot)
        real_pos = block_of * K + within_block_pos

        cos, sin = rope_cos_sin_for_positions(real_pos, hd, cfg.rope_base, device)

        win = window if window is not None else L
        ti = real_pos.unsqueeze(1)
        tj = real_pos.unsqueeze(0)
        causal = tj <= ti
        windowed = (ti - tj) < win
        query_only_sees_own_block = (~is_query).unsqueeze(0) | (block_of.unsqueeze(1) == block_of.unsqueeze(0))
        real_never_sees_query = ~(is_query.unsqueeze(0) & (~is_query).unsqueeze(1))
        allow = causal & windowed & query_only_sees_own_block & real_never_sees_query
        attn_mask = allow.view(1, 1, Le, Le)

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_blocks = he.view(B, n_blocks, K + 1, D)
        return he_blocks[:, :, K, :]

    def _merged_layout(self, L: int, tracks_meta: tuple[tuple[int, int | None], ...],
                        device: torch.device, forced_sc: int | None = None,
                        forced_n_chunks: int | None = None, forced_n_prev_chunks: int | None = None) -> dict:
        """Chronological merged-interleave layout: every track's codes land at their true time
        position (right after the last byte of the block they summarize), merged with the raw byte
        stream into ONE physically time-ordered buffer -- so causal masking is a plain buffer-index
        comparison (a tied code always sorts strictly after the byte that produced it, so it's
        automatically invisible to that byte, no same-position exclusion term needed) and windowed
        attention can slice CONTIGUOUS buffer ranges with no runtime sort. This depends only on
        shape (L, each track's (K, window)), never on data, so it's built once per signature and
        cached (self._merged_cache) -- identical cost whether called every training step (fixed L)
        or every generate_kv_cache step (same fixed FIFO-window L reused every step).

        forced_sc/forced_n_chunks/forced_n_prev_chunks: only used by
        _merged_layout_multimode_chunked (multi_mode_impl='single_pass' under a finite window) --
        override the naturally-computed chunk grid so every mode's own segment can be padded onto
        ONE shared (sc, n_chunks, n_prev_chunks) grid and batched together. Forcing a value larger
        than what this (L, tracks_meta) would naturally need is always safe (extra chunks/lookback
        are pure padding, masked out the same sentinel way Lp-padding already is below); forcing
        smaller is not (asserted against) -- it would silently narrow the real window."""
        key = (L, tracks_meta, str(device), forced_sc, forced_n_chunks, forced_n_prev_chunks)
        cached = self._merged_cache.get(key)
        if cached is not None:
            return cached
        T = len(tracks_meta)
        byte_true_pos = torch.arange(L, device=device)
        byte_category = torch.zeros(L, dtype=torch.long, device=device)

        code_true_pos_parts, code_category_parts, code_window_parts, n_blocks_list = [], [], [], []
        for j, (K, window) in enumerate(tracks_meta):
            n_blocks = L // K
            n_blocks_list.append(n_blocks)
            # No true_pos=-1 bos sentinel here (fixblock): block 0 has no real code before it, so
            # it simply isn't represented in the code stream at all -- only the n_blocks real codes.
            tp = (torch.arange(n_blocks, device=device) + 1) * K - 1
            code_true_pos_parts.append(tp)
            code_category_parts.append(torch.full((n_blocks,), j + 1, dtype=torch.long, device=device))
            wv = float(window) if window is not None else float(L)
            code_window_parts.append(torch.full((n_blocks,), wv, device=device))
        code_true_pos = (torch.cat(code_true_pos_parts) if T > 0
                          else torch.empty(0, dtype=torch.long, device=device))
        code_category = (torch.cat(code_category_parts) if T > 0
                          else torch.empty(0, dtype=torch.long, device=device))
        code_window = torch.cat(code_window_parts) if T > 0 else torch.empty(0, device=device)

        total_true_pos = torch.cat([byte_true_pos, code_true_pos])
        total_category = torch.cat([byte_category, code_category])
        # byte (category 0) always sorts before any code (category >=1) at the same true_pos --
        # this single sort key is what makes the same-position exclusion automatic (see docstring).
        sort_key = (total_true_pos + 1) * (T + 2) + total_category
        perm = torch.argsort(sort_key, stable=True)

        w0 = tracks_meta[0][1] if tracks_meta[0][1] is not None else L
        byte_window = torch.full((L,), float(w0), device=device)
        total_window = torch.cat([byte_window, code_window])

        true_pos_sorted = total_true_pos[perm]
        window_of_slot = total_window[perm]
        category_sorted = total_category[perm]
        K0 = tracks_meta[0][0] if T > 0 else None
        Le = L + code_true_pos.shape[0]
        # NTP query extraction: byte t's training/generation query is NOT byte t's own buffer slot
        # -- it's the LAST buffer entry sharing true_pos==t (itself, or a tied code if one
        # completes exactly there), matching qcute_v5_concat_slow.py's query_seq mechanism (a
        # just-completed code's own state is what predicts the immediately-following byte, per
        # docs/qcute_refine_v4_4_1_v4_5_1_math.md's LM-continuation). true_pos_sorted is sorted and
        # every byte value 0..L-1 is present (bytes alone already cover the full range), so the
        # last buffer index with true_pos<=t equals the last one with true_pos==t exactly --
        # searchsorted gives this in one op, no scatter needed.
        extract_pos = torch.searchsorted(true_pos_sorted, torch.arange(L, device=device), right=True) - 1
        struct = dict(perm=perm, extract_pos=extract_pos, true_pos_sorted=true_pos_sorted,
                      window_of_slot=window_of_slot, category_sorted=category_sorted, K0=K0,
                      Le=Le, n_blocks_list=n_blocks_list)

        finite_windows = [w for _, w in tracks_meta if w is not None]
        if finite_windows or forced_sc is not None:
            sc_natural = max(1, min(min(finite_windows), Le)) if finite_windows else Le
            sc = forced_sc if forced_sc is not None else sc_natural
            assert forced_sc is None or forced_sc <= sc_natural, (
                f"_merged_layout: forced_sc={forced_sc} > this segment's own natural sc={sc_natural} "
                f"-- forcing a LARGER chunk size than natural would silently narrow the effective "
                f"window (fewer, coarser chunks); only forcing SMALLER (finer) is safe.")
            n_chunks_natural = -(-Le // sc)
            n_chunks = forced_n_chunks if forced_n_chunks is not None else n_chunks_natural
            assert n_chunks >= n_chunks_natural, (
                f"_merged_layout: forced_n_chunks={forced_n_chunks} < natural requirement "
                f"{n_chunks_natural} -- would drop real buffer entries.")
            Lp = n_chunks * sc
            pad_len = Lp - Le
            W_max = max((w if w is not None else Le) for _, w in tracks_meta) if tracks_meta else 0
            # Chunk lookback must be counted in BUFFER-INDEX units, not true_pos units: each
            # true_pos value can hold up to (1 byte + one code per track) buffer entries when
            # tracks' blocks tie there, so a window of W_max true_pos units can require looking
            # back up to W_max*density buffer slots, not W_max (bug found 2026-08-17, see
            # docs/chunked_decode_window_bug.md -- undercounting this silently narrowed the
            # effective attention window whenever chunking triggered, most severely at small K
            # where every true_pos gets a byte+code pair).
            density = len(tracks_meta) + 1
            n_prev_natural = min(max(1, -(-(W_max * density) // sc)), max(0, n_chunks - 1))
            n_prev_chunks = forced_n_prev_chunks if forced_n_prev_chunks is not None else n_prev_natural
            assert n_prev_chunks >= n_prev_natural, (
                f"_merged_layout: forced_n_prev_chunks={forced_n_prev_chunks} < natural requirement "
                f"{n_prev_natural} -- would silently narrow the effective window.")

            true_pos_p, window_p = true_pos_sorted, window_of_slot
            cat_p = category_sorted
            if pad_len > 0:
                true_pos_p = F.pad(true_pos_p, (0, pad_len), value=-10 ** 9)
                window_p = F.pad(window_p, (0, pad_len), value=0.0)
                cat_p = F.pad(cat_p, (0, pad_len), value=0)
            pos_b = true_pos_p.view(n_chunks, sc)
            win_b = window_p.view(n_chunks, sc)
            cat_b = cat_p.view(n_chunks, sc)
            pad_pos = torch.full((n_prev_chunks, sc), -10 ** 9, device=device, dtype=pos_b.dtype)
            pad_win = torch.zeros((n_prev_chunks, sc), device=device, dtype=win_b.dtype)
            pad_cat = torch.zeros((n_prev_chunks, sc), device=device, dtype=cat_b.dtype)
            pos_ext = torch.cat([pad_pos, pos_b], dim=0)
            win_ext = torch.cat([pad_win, win_b], dim=0)
            cat_ext = torch.cat([pad_cat, cat_b], dim=0)
            idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
                   + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
            Kc = (n_prev_chunks + 1) * sc
            pos_win = pos_ext[idx].reshape(n_chunks, Kc)
            win_win = win_ext[idx].reshape(n_chunks, Kc)
            cat_win = cat_ext[idx].reshape(n_chunks, Kc)

            ti = pos_b.unsqueeze(-1)
            tj = pos_win.unsqueeze(1)
            # Own chunk occupies the LAST sc columns of the gathered Kc range (idx's own-chunk
            # offset is n_prev_chunks); prior chunks are entirely in the past (always causal) and
            # entries within the flattened Kc range are already in strict buffer order, so a plain
            # local column<=row comparison over the own-chunk slice reproduces buffer-index
            # causality -- no true_pos tie-break needed (see _merged_layout's docstring).
            local_row = torch.arange(sc, device=device).view(1, sc, 1)
            local_col = torch.arange(Kc, device=device).view(1, 1, Kc) - n_prev_chunks * sc
            causal = local_col <= local_row
            windowed = (ti - tj) < win_win.unsqueeze(1)
            # _skip: a raw byte key (category==0) is visible only to queries still inside that
            # byte's own (self-track, K0-sized) block -- once a query has moved past the block that
            # produced it, only that block's CODE stays visible (code_pos<=query_pos, already
            # enforced by causal+windowed above -- codes are never restricted by this term). Pure
            # function of (query true_pos, key true_pos, K0) -- static/shape-only, same as every
            # other mask term here.
            is_byte_key = (cat_win == 0).unsqueeze(1)
            same_block = (ti // K0) == (tj // K0)
            byte_ok = (~is_byte_key) | same_block
            allow = causal & windowed & byte_ok
            struct.update(sc=sc, n_chunks=n_chunks, Lp=Lp, pad_len=pad_len,
                          n_prev_chunks=n_prev_chunks, idx=idx, Kc=Kc,
                          chunk_mask=allow.view(1, n_chunks, 1, sc, Kc))
        self._merged_cache[key] = struct
        return struct

    def _merged_decode_forward(self, x0: torch.Tensor, tracks: list[tuple[torch.Tensor, int, int | None]],
                                extra_query: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Chronological merged-interleave decode -- handles 1..N tracks uniformly (no separate
        single-track code path). Builds the merged buffer via a pure gather using _merged_layout's
        cached permutation (no sort at forward-call time), runs dense or windowed/banded attention
        over it, and extracts each byte t's NTP-training query via the cached extract_pos: the
        LAST buffer entry sharing true_pos==t, which is a just-completed code's state when one
        ties there (matching qcute_v5_concat_slow.py's query_seq mechanism -- a code's own state
        is what predicts the immediately-following byte) or the byte's own state otherwise.
        extra_query=True additionally returns the buffer's chronologically LAST hidden state as
        query_last -- the same idea applied at the sequence's current end, for generation -- for
        ANY number of tracks, not just the old single-track case."""
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        tracks_meta = tuple((K, window) for _, K, window in tracks)
        struct = self._merged_layout(L, tracks_meta, device)
        perm, extract_pos, Le = struct["perm"], struct["extract_pos"], struct["Le"]

        code_parts = []
        for j, (code_kv, K, _window) in enumerate(tracks):
            n_blocks = struct["n_blocks_list"][j]
            code_parts.append(code_kv[:, :n_blocks, :])
        all_code = torch.cat(code_parts, dim=1) if code_parts else x0.new_zeros(B, 0, D)
        unordered = torch.cat([x0, all_code], dim=1)
        combined = unordered[:, perm, :]

        finite_windows = [w for _, w in tracks_meta if w is not None]
        use_chunked = bool(finite_windows) and "sc" in struct and Le > struct["sc"]

        if not use_chunked:
            cos, sin = rope_cos_sin_for_positions(struct["true_pos_sorted"].clamp(min=0), hd, cfg.rope_base, device)
            T = len(tracks_meta)
            # _skip needs an explicit mask whenever any track exists (T>0), even with no window
            # configured -- the byte-same-block restriction below is a semantic pruning, not a
            # windowing device, so it applies regardless of `window`. Only the genuinely trivial
            # T==0 case (no codes at all, plain self-attention over bytes) keeps the is_causal fast
            # path, since there's nothing for any byte to be redundant with there.
            fully_causal = not finite_windows and T == 0
            attn_mask = None
            if not fully_causal:
                ti = struct["true_pos_sorted"].unsqueeze(1)
                tj = struct["true_pos_sorted"].unsqueeze(0)
                buf_i = torch.arange(Le, device=device).unsqueeze(1)
                buf_j = torch.arange(Le, device=device).unsqueeze(0)
                causal = buf_j <= buf_i
                if finite_windows:
                    allow = causal & ((ti - tj) < struct["window_of_slot"].unsqueeze(0))
                else:
                    allow = causal
                if T > 0:
                    K0 = struct["K0"]
                    is_byte_key = (struct["category_sorted"] == 0).unsqueeze(0)
                    same_block = (ti // K0) == (tj // K0)
                    allow = allow & ((~is_byte_key) | same_block)
                attn_mask = allow.view(1, 1, Le, Le)

            xe = combined
            for block in self.blocks:
                xn = block.ln1(xe)
                qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                if fully_causal:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
                xe = xe + a
                xe = xe + block.mlp(block.ln2(xe))
            he = self.ln_f(xe)
        else:
            sc, n_chunks, Lp, pad_len = struct["sc"], struct["n_chunks"], struct["Lp"], struct["pad_len"]
            n_prev_chunks, idx, Kc = struct["n_prev_chunks"], struct["idx"], struct["Kc"]
            _warn_thin_window(tracks, sc)
            xe = combined
            true_pos_p = struct["true_pos_sorted"]
            if pad_len > 0:
                xe = F.pad(xe, (0, 0, 0, pad_len))
                true_pos_p = F.pad(true_pos_p, (0, pad_len), value=0)
            cos, sin = rope_cos_sin_for_positions(true_pos_p.clamp(min=0), hd, cfg.rope_base, device)
            for block in self.blocks:
                xn = block.ln1(xe)
                qkv = block.attn.qkv(xn).reshape(B, Lp, 3, H, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

                qb = q.view(B, H, n_chunks, sc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, sc, hd)
                kb_flat = k.view(B, H, n_chunks, sc, hd)
                vb_flat = v.view(B, H, n_chunks, sc, hd)
                pad_k = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=k.dtype)
                pad_v = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=v.dtype)
                k_ext = torch.cat([pad_k, kb_flat], dim=2)
                v_ext = torch.cat([pad_v, vb_flat], dim=2)
                k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)
                v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)

                mask_batched = struct["chunk_mask"].expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
                yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
                y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

                a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
                xe = xe + a
                xe = xe + block.mlp(block.ln2(xe))
            he = self.ln_f(xe)[:, :Le, :]

        h_out = he[:, extract_pos, :]
        query_last = he[:, -1, :] if extra_query else None
        return h_out, query_last

    def _merged_layout_multimode(self, L: int, tracks_meta: tuple[tuple[int, int | None], ...],
                                  device: torch.device) -> dict:
        """Cached (per (L, tracks_meta, device), same convention as _merged_layout) block-diagonal
        layout for multi_mode_impl='single_pass': T=len(tracks_meta) independent segments, one per
        mode m=1..T (segment m == the SAME buffer _merged_layout(L, tracks_meta[:m], device) would
        build on its own), concatenated along the sequence dim with a block-diagonal attention mask
        -- zero cross-segment visibility, each segment's own causal/window/_skip mask reused
        VERBATIM from _merged_decode_forward's dense path. Because segments never attend to each
        other, this is mathematically identical to T separate _merged_decode_forward calls (the
        'multipass' reference) -- same total buffer size/compute, just one shared pass through
        self.blocks instead of T separate Python-level re-entries. Dense-only (each segment must
        itself stay off the chunked path -- checked by the caller)."""
        key = (L, tracks_meta, str(device))
        cached = self._merged_multimode_cache.get(key)
        if cached is not None:
            return cached
        T = len(tracks_meta)
        seg_structs = [self._merged_layout(L, tracks_meta[:m], device) for m in range(1, T + 1)]

        offsets, off = [], 0
        for s in seg_structs:
            offsets.append(off)
            off += s["Le"]
        Lm = off

        allow = torch.zeros(Lm, Lm, dtype=torch.bool, device=device)
        true_pos_parts = []
        for o, s in zip(offsets, seg_structs):
            Le_m = s["Le"]
            ti = s["true_pos_sorted"].unsqueeze(1)
            tj = s["true_pos_sorted"].unsqueeze(0)
            buf_i = torch.arange(Le_m, device=device).unsqueeze(1)
            buf_j = torch.arange(Le_m, device=device).unsqueeze(0)
            causal = buf_j <= buf_i
            allow_m = causal & ((ti - tj) < s["window_of_slot"].unsqueeze(0))
            K0 = s["K0"]
            is_byte_key = (s["category_sorted"] == 0).unsqueeze(0)
            same_block = (ti // K0) == (tj // K0)
            allow_m = allow_m & ((~is_byte_key) | same_block)
            allow[o:o + Le_m, o:o + Le_m] = allow_m
            true_pos_parts.append(s["true_pos_sorted"])

        multi = dict(Lm=Lm, allow=allow, total_true_pos=torch.cat(true_pos_parts), offsets=offsets,
                     Le_list=[s["Le"] for s in seg_structs], extract_pos_list=[s["extract_pos"] for s in seg_structs])
        self._merged_multimode_cache[key] = multi
        return multi

    def _merged_decode_forward_multimode(self, x0: torch.Tensor,
                                          tracks: list[tuple[torch.Tensor, int, int | None]]
                                          ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (h_full, h_modes): h_full is IDENTICAL to _merged_decode_forward's own h_out
        (the deepest/T-track mode); h_modes[m-1] (m=1..T-1) is the same hidden states
        _merged_decode_forward(x0, tracks[:m]) would produce on its own -- see
        _merged_layout_multimode's docstring for why this is exact, not approximate.

        Dense (this method) is used when every segment stays off the chunked path; if any segment
        would need chunking, dispatches to _merged_decode_forward_multimode_chunked (a genuinely
        batched chunked implementation) instead -- see that method's docstring. Both are exact
        (verified against _merged_decode_forward_multimode_looped, scripts/test_v5_concat_modes.py)."""
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        T = len(tracks)
        if T <= 1:
            h_out, _ = self._merged_decode_forward(x0, tracks, extra_query=False)
            return h_out, []

        tracks_meta_full = tuple((K, window) for _, K, window in tracks)
        any_chunked = False
        for m in range(1, T + 1):
            tracks_meta_m = tuple((K, window) for _, K, window in tracks[:m])
            struct_m = self._merged_layout(L, tracks_meta_m, device)
            finite_windows_m = [w for _, w in tracks_meta_m if w is not None]
            if bool(finite_windows_m) and "sc" in struct_m and struct_m["Le"] > struct_m["sc"]:
                any_chunked = True
                break
        if any_chunked:
            return self._merged_decode_forward_multimode_chunked(x0, tracks)

        multi = self._merged_layout_multimode(L, tracks_meta_full, device)
        Lm, allow, total_true_pos = multi["Lm"], multi["allow"], multi["total_true_pos"]
        offsets, Le_list, extract_pos_list = multi["offsets"], multi["Le_list"], multi["extract_pos_list"]

        combined_parts = []
        for m in range(1, T + 1):
            tracks_m = tracks[:m]
            tracks_meta_m = tuple((K, window) for _, K, window in tracks_m)
            struct_m = self._merged_layout(L, tracks_meta_m, device)
            code_parts = []
            for j, (code_kv, K, _window) in enumerate(tracks_m):
                n_blocks = struct_m["n_blocks_list"][j]
                code_parts.append(code_kv[:, :n_blocks, :])
            all_code = torch.cat(code_parts, dim=1) if code_parts else x0.new_zeros(B, 0, D)
            unordered = torch.cat([x0, all_code], dim=1)
            combined_parts.append(unordered[:, struct_m["perm"], :])

        xe = torch.cat(combined_parts, dim=1)
        cos, sin = rope_cos_sin_for_positions(total_true_pos.clamp(min=0), hd, cfg.rope_base, device)
        attn_mask = allow.view(1, 1, Lm, Lm)

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Lm, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Lm, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))
        he = self.ln_f(xe)

        h_list = []
        for o, Le_m, ep in zip(offsets, Le_list, extract_pos_list):
            h_list.append(he[:, o:o + Le_m, :][:, ep, :])
        return h_list[-1], h_list[:-1]

    def _merged_decode_forward_multimode_looped(self, x0: torch.Tensor,
                                                  tracks: list[tuple[torch.Tensor, int, int | None]]
                                                  ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Option 1 (simplest SWA-compatible path): literally calls the existing, unmodified
        _merged_decode_forward once per mode m=1..T (dense OR chunked, whichever that mode's own
        window config needs -- _merged_decode_forward already supports both). Same compute as
        multi_mode_impl='multipass' (not a single-kernel win), but implemented here so it can serve
        as the correctness reference for _merged_decode_forward_multimode_chunked's batched version,
        and as an always-available fallback."""
        T = len(tracks)
        if T <= 1:
            h_out, _ = self._merged_decode_forward(x0, tracks, extra_query=False)
            return h_out, []
        h_list = [self._merged_decode_forward(x0, tracks[:m], extra_query=False)[0] for m in range(1, T + 1)]
        return h_list[-1], h_list[:-1]

    def _merged_layout_multimode_chunked(self, L: int, tracks_meta: tuple[tuple[int, int | None], ...],
                                          device: torch.device) -> dict:
        """Option 2's structural precompute: pads every mode's own (sc, n_chunks, n_prev_chunks)
        chunk grid (see _merged_layout's forced_* params) onto ONE shared grid so all T modes can
        run as a single batched SDPA call per layer -- the chunked-path counterpart of
        _merged_layout_multimode's dense block-diagonal design. sc_shared is the FINEST (smallest)
        chunk size any mode naturally needs (forcing a smaller-than-natural sc is always safe --
        more, thinner chunks still cover the same causal/window/_skip visibility, just less
        efficiently); n_chunks/n_prev_chunks are then the largest any mode needs at that shared sc.
        Once sc/n_chunks/n_prev_chunks are identical across modes, the gather index `idx`
        (a pure function of (n_chunks, n_prev_chunks), never of data) is ALSO identical across
        modes -- only chunk_mask (which depends on each mode's own true_pos/category/window
        content) differs per mode and needs its own stack. Cached per (L, tracks_meta, device)."""
        key = (L, tracks_meta, str(device))
        cached = self._merged_multimode_chunked_cache.get(key)
        if cached is not None:
            return cached
        T = len(tracks_meta)

        def natural_sc(s: dict) -> int:
            return s["sc"] if "sc" in s else s["Le"]

        natural_structs = [self._merged_layout(L, tracks_meta[:m], device) for m in range(1, T + 1)]
        sc_shared = max(1, min(natural_sc(s) for s in natural_structs))

        at_shared_sc = [self._merged_layout(L, tracks_meta[:m], device, forced_sc=sc_shared) for m in range(1, T + 1)]
        n_chunks_max = max(s["n_chunks"] for s in at_shared_sc)
        n_prev_max = max(s["n_prev_chunks"] for s in at_shared_sc)

        structs = [self._merged_layout(L, tracks_meta[:m], device, forced_sc=sc_shared,
                                        forced_n_chunks=n_chunks_max, forced_n_prev_chunks=n_prev_max)
                   for m in range(1, T + 1)]
        chunk_masks = torch.stack([s["chunk_mask"].view(n_chunks_max, sc_shared, (n_prev_max + 1) * sc_shared)
                                    for s in structs], dim=0)   # (T, n_chunks, sc, Kc)
        idx = structs[0]["idx"]   # identical across modes once sc/n_chunks/n_prev_chunks are shared

        multi = dict(sc=sc_shared, n_chunks=n_chunks_max, n_prev_chunks=n_prev_max,
                     Kc=(n_prev_max + 1) * sc_shared, Lp=n_chunks_max * sc_shared,
                     chunk_masks=chunk_masks, idx=idx,
                     Le_list=[s["Le"] for s in structs], extract_pos_list=[s["extract_pos"] for s in structs],
                     true_pos_sorted_list=[s["true_pos_sorted"] for s in structs],
                     perm_list=[s["perm"] for s in structs], n_blocks_lists=[s["n_blocks_list"] for s in structs])
        self._merged_multimode_chunked_cache[key] = multi
        return multi

    def _merged_decode_forward_multimode_chunked(self, x0: torch.Tensor,
                                                   tracks: list[tuple[torch.Tensor, int, int | None]]
                                                   ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Option 2: genuinely batched SWA-compatible multi-mode decode -- extends the existing
        single-mode chunked path's own (B, n_chunks) -> flattened-batch trick with a 'mode' axis,
        using _merged_layout_multimode_chunked's shared chunk grid. Every mode's own segment is
        independent (block-diagonal across modes, same as the dense multimode path -- modes never
        attend to each other, only within their own padded buffer), so this is exact, not
        approximate -- verified against _merged_decode_forward_multimode_looped."""
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        T = len(tracks)
        tracks_meta_full = tuple((K, window) for _, K, window in tracks)
        multi = self._merged_layout_multimode_chunked(L, tracks_meta_full, device)
        sc, n_chunks, n_prev_chunks, Kc, Lp = multi["sc"], multi["n_chunks"], multi["n_prev_chunks"], multi["Kc"], multi["Lp"]
        chunk_masks, idx = multi["chunk_masks"], multi["idx"]
        Le_list, extract_pos_list = multi["Le_list"], multi["extract_pos_list"]
        true_pos_list, perm_list, n_blocks_lists = multi["true_pos_sorted_list"], multi["perm_list"], multi["n_blocks_lists"]

        xe_parts, true_pos_p_parts = [], []
        for m in range(1, T + 1):
            tracks_m = tracks[:m]
            perm_m, n_blocks_list_m = perm_list[m - 1], n_blocks_lists[m - 1]
            code_parts = []
            for j, (code_kv, K, _window) in enumerate(tracks_m):
                n_blocks = n_blocks_list_m[j]
                code_parts.append(code_kv[:, :n_blocks, :])
            all_code = torch.cat(code_parts, dim=1) if code_parts else x0.new_zeros(B, 0, D)
            unordered = torch.cat([x0, all_code], dim=1)
            combined_m = unordered[:, perm_m, :]
            Le_m = Le_list[m - 1]
            pad_len_m = Lp - Le_m
            if pad_len_m > 0:
                combined_m = F.pad(combined_m, (0, 0, 0, pad_len_m))
            xe_parts.append(combined_m)
            tp = true_pos_list[m - 1]
            if pad_len_m > 0:
                tp = F.pad(tp, (0, pad_len_m), value=0)
            true_pos_p_parts.append(tp)

        xe = torch.stack(xe_parts, dim=1)          # (B, T, Lp, D)
        true_pos_p = torch.stack(true_pos_p_parts, dim=0)   # (T, Lp)
        cos_flat, sin_flat = rope_cos_sin_for_positions(true_pos_p.clamp(min=0).reshape(-1), hd, cfg.rope_base, device)
        cos = cos_flat.view(1, T, 1, Lp, hd)
        sin = sin_flat.view(1, T, 1, Lp, hd)

        pad_k = torch.zeros(B, T, H, n_prev_chunks, sc, hd, device=device)
        pad_v = torch.zeros(B, T, H, n_prev_chunks, sc, hd, device=device)
        mask_batched = (chunk_masks.view(1, T, n_chunks, 1, sc, Kc).expand(B, T, n_chunks, 1, sc, Kc)
                         .reshape(B * T * n_chunks, 1, sc, Kc))

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, T, Lp, 3, H, hd).permute(3, 0, 1, 4, 2, 5)   # (3,B,T,H,Lp,hd)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            q_c = q.view(B, T, H, n_chunks, sc, hd)
            k_c = k.view(B, T, H, n_chunks, sc, hd)
            v_c = v.view(B, T, H, n_chunks, sc, hd)
            k_ext = torch.cat([pad_k.to(k.dtype), k_c], dim=3)
            v_ext = torch.cat([pad_v.to(v.dtype), v_c], dim=3)
            k_win = k_ext[:, :, :, idx].reshape(B, T, H, n_chunks, Kc, hd)
            v_win = v_ext[:, :, :, idx].reshape(B, T, H, n_chunks, Kc, hd)

            qb = q_c.permute(0, 1, 3, 2, 4, 5).reshape(B * T * n_chunks, H, sc, hd)
            kb = k_win.permute(0, 1, 3, 2, 4, 5).reshape(B * T * n_chunks, H, Kc, hd)
            vb = v_win.permute(0, 1, 3, 2, 4, 5).reshape(B * T * n_chunks, H, Kc, hd)

            yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
            y = yb.view(B, T, n_chunks, H, sc, hd).permute(0, 1, 3, 2, 4, 5).reshape(B, T, H, Lp, hd)
            y = y.permute(0, 1, 3, 2, 4).reshape(B, T, Lp, D)

            a = block.attn.out(y)
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))
        he = self.ln_f(xe)   # (B, T, Lp, D)

        h_list = []
        for m in range(1, T + 1):
            Le_m, ep = Le_list[m - 1], extract_pos_list[m - 1]
            h_list.append(he[:, m - 1, :Le_m, :][:, ep, :])
        return h_list[-1], h_list[:-1]

    def _ntp_loss_acc(self, h_query: torch.Tensor, target_repr: torch.Tensor, is_byte_level: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if is_byte_level:
            target = target_repr.reshape(-1)
            logits = F.linear(h_query, self.embed.weight)
            loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean()
            return loss, acc
        return self.quant.ntp_loss_acc(self, h_query, target_repr)

    def forward(self, seq_repr: torch.Tensor, level: int, window: int | None, compute_ntp: bool = True,
                decode_tracks: list[tuple[torch.Tensor, int, int | None]] | None = None,
                extra_query: bool = False, multi_mode: str = "off"
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
                           torch.Tensor | None, list, list, torch.Tensor | None]:
        cfg = self.cfg
        K = cfg.Ks[level]
        D = cfg.d_model
        is_byte_level = level == 0

        if is_byte_level:
            x = self.embed(seq_repr)
            B, L = seq_repr.shape
        else:
            x = self.quant.embed_input(self, seq_repr)
            B, L, _ = seq_repr.shape

        x0 = x
        head_dim = D // cfg.n_heads

        query_last = None
        h_modes: list[torch.Tensor] = []
        if decode_tracks is not None:
            assert len(decode_tracks) >= 1
            if multi_mode != "off" and len(decode_tracks) > 1:
                if multi_mode == "single_pass":
                    h, h_modes = self._merged_decode_forward_multimode(x0, decode_tracks)
                elif multi_mode == "multipass":
                    h, query_last = self._merged_decode_forward(x0, decode_tracks, extra_query=extra_query)
                    h_modes = [self._merged_decode_forward(x0, decode_tracks[:m], extra_query=False)[0]
                               for m in range(1, len(decode_tracks))]
                else:
                    raise ValueError(f"unknown multi_mode {multi_mode!r}")
            else:
                h, query_last = self._merged_decode_forward(x0, decode_tracks, extra_query=extra_query)
        else:
            cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
            for block in self.blocks:
                x = block(x, cos, sin, window)
            h = self.ln_f(x)

        if compute_ntp:
            if decode_tracks is not None:
                # fixblock: targets 1..K-1 are still inside block 0, which has no real code before
                # it (decode_bos removed) -- excluded from decode's own loss entirely, not just
                # given a fake anchor. Only targets K..L-1 (conditioned on a real, completed code)
                # count here; see RefineLM.forward's byte_loss_full for the spliced comparison
                # metric that adds encode's own loss back in for this excluded range.
                h_flat = h[:, K - 1:-1, :].reshape(-1, D)
                ntp_loss, ntp_acc = self._ntp_loss_acc(h_flat, seq_repr[:, K:], is_byte_level)
                mode_losses, mode_accs = [], []
                for h_m in h_modes:
                    h_m_flat = h_m[:, K - 1:-1, :].reshape(-1, D)
                    l_m, a_m = self._ntp_loss_acc(h_m_flat, seq_repr[:, K:], is_byte_level)
                    mode_losses.append(l_m)
                    mode_accs.append(a_m)
            else:
                h_flat = h[:, :-1, :].reshape(-1, D)
                ntp_loss, ntp_acc = self._ntp_loss_acc(h_flat, seq_repr[:, 1:], is_byte_level)
                mode_losses, mode_accs = [], []
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())
            mode_losses, mode_accs = [], []

        n_blocks = L // K
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        if cfg.code_extract_mode == "last_h":
            pooled = h_blocks[:, :, K - 1, :]
        elif cfg.code_extract_mode == "softmax_pool":
            q_implicit = h_blocks[:, :, K - 1, :]
            scores = (h_blocks * q_implicit.unsqueeze(2)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
        elif cfg.code_extract_mode == "light_query_attn":
            scores = (h_blocks * self.code_query.view(1, 1, 1, D)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
            pooled = self.code_out(pooled)
        elif cfg.code_extract_mode == "query_embed":
            pooled = self._query_embed_pool(x0, K, n_blocks, window)
        else:
            raise ValueError(f"unknown code_extract_mode {cfg.code_extract_mode!r}")
        pre_q = self._classify(pooled)
        c_i = self.quant.quantize(pre_q)
        entropy_reg = self.quant.entropy_reg(pre_q)

        return c_i, ntp_loss, ntp_acc, h, query_last, None, mode_losses, mode_accs, entropy_reg


class RefineLM(nn.Module):
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

        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels, f"attn_window tuple must have length n_levels={self.n_levels}, got {len(raw_windows)}"
        windows: list[int | None] = []
        decode_windows: list[list[int | None]] = []
        for i, w in enumerate(raw_windows):
            n_sources = self.n_levels - i
            if isinstance(w, (tuple, list)):
                assert len(w) == 2, f"attn_window[{i}] must be a scalar or an (encode_window, decode_window) 2-tuple, got {w!r}"
                ew, dw = w
            else:
                ew = dw = w
            windows.append(None if ew == -1 else ew)
            if isinstance(dw, (tuple, list)):
                assert len(dw) == n_sources, (
                    f"attn_window[{i}]'s decode_window must be a scalar (broadcast) or a tuple of "
                    f"length n_levels-{i}={n_sources} (one per decode source: self, +1, ..., top), got {dw!r}")
                decode_windows.append([None if x == -1 else x for x in dw])
            else:
                decode_windows.append([None if dw == -1 else dw] * n_sources)
        self.windows = windows
        self.decode_windows = decode_windows
        # Effective-codes summary (see _warn_thin_window's formula): for each level's decode, how
        # much cumulative coarser-level history its configured window(s) actually buy, in code
        # units. Printed once at construction so this is visible before a step is ever run, not
        # just as a late warning if it turns out to be thin.
        for i, dwlist in enumerate(decode_windows):
            cum_K, per_track, invisible_srcs = 1, [], []
            for src_offset, dwindow in enumerate(dwlist):
                cum_K *= cfg.Ks[i + src_offset]
                if dwindow is None:
                    per_track.append(f"K={cum_K}:full")
                else:
                    n_codes = dwindow // cum_K
                    per_track.append(f"K={cum_K}:{n_codes}codes")
                    if dwindow != 0 and n_codes == 0:
                        invisible_srcs.append(cum_K)
            print(f"decode effective codes level{i}: " + ", ".join(per_track))
            if invisible_srcs:
                print(f"WARNING: level{i} decode_window is too small for cumulative K in "
                      f"{invisible_srcs} -- 0 codes fit, that source is completely invisible to "
                      f"this level's decode (not just thin). Increase attn_window[{i}]'s "
                      f"decode_window or drop that source.")
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] encode window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
        for i, dwlist in enumerate(decode_windows):
            L = seq_lens[i]
            for src_offset, dwindow in enumerate(dwlist):
                if dwindow is not None and dwindow != 0:
                    assert L % dwindow == 0 or L <= dwindow, (
                        f"attn_window[{i}]'s decode_window[{src_offset}] ({dwindow}) must divide "
                        f"level {i}'s sequence length ({L}), or be >= it")

        self.encode_lms = nn.ModuleList([LevelLM(cfg) for _ in range(self.n_levels)])
        self.decode_lms = nn.ModuleList([LevelLM(cfg) for _ in range(self.n_levels)])

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True, max_decode_sources: int | None = None,
             want_next_query: bool = False):
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list = [], [], [], [], []
        encode_entropy_regs: list = []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, loss_i, acc_i, h_i, _, _, _, _, entropy_reg_i = self.encode_lms[i](seq_repr, level=i, window=self.windows[i], compute_ntp=want_ntp)
            encode_losses.append(loss_i)
            encode_accs.append(acc_i)
            h_list.append(h_i)
            c_list.append(c_i)
            x_list.append(seq_repr)
            encode_entropy_regs.append(entropy_reg_i)
            seq_repr = c_i

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        decode_derived_c: dict[int, torch.Tensor] = {}
        h_out = list(h_list)
        next_query: list[torch.Tensor | None] = [None] * self.n_levels
        query_seq_out: list[torch.Tensor | None] = [None] * self.n_levels
        decode_stage_extra_losses: list = []
        for i in reversed(range(self.n_levels)):
            L_i = x_list[i].shape[1]
            tracks: list[tuple[torch.Tensor, int, int | None]] = []
            cum_K = 1
            for j in range(i, self.n_levels):
                cum_K *= cfg.Ks[j]
                window = self.decode_windows[i][j - i]
                if window == 0:
                    continue
                # Training always calls with L_i an exact multiple of cum_K (context_len is built
                # to divide evenly at every level, see __init__), so this floor check is a no-op
                # there. It only bites during generation, where L_i grows one byte at a time and is
                # rarely block-aligned -- stop adding tracks (keep whichever finer ones already
                # collected) rather than discarding everything, so e.g. a self track can still be
                # used even when a coarser track isn't affordable yet. _merged_layout's own
                # per-track block-count (L//K, floor) is floor-based too, so a not-yet-complete
                # trailing block is simply excluded from the buffer, never fabricated.
                if L_i // cum_K < 1:
                    break
                source_c = decode_derived_c[j] if (j > i and j in decode_derived_c) else c_list[j]
                dec_lm = self.decode_lms[i]
                code_embeds = dec_lm.quant.embed_for_decode(dec_lm, source_c)
                tracks.append((code_embeds, cum_K, window))
            if not tracks:
                continue
            # No "2+ blocks" floor here -- see qcute_v5_concat_fixblock.py's _run for why (fixblock's
            # own K-1 target exclusion already handles "no code yet" precisely, at target
            # granularity, a strictly finer floor than requiring 2 whole blocks).
            full_tracks = tracks
            if max_decode_sources is not None:
                full_tracks = full_tracks[:max_decode_sources]
            # multi_mode only applies to the un-truncated (max_decode_sources=None) call -- an
            # explicit max_decode_sources override (e.g. generate_self_only_cond) already IS a
            # single specific mode, nothing to expand further.
            multi_mode = cfg.multi_mode_impl if max_decode_sources is None else "off"
            c_i2, loss_i2, acc_i2, h_i2, query_last_i, query_seq_i, mode_losses_i, mode_accs_i, _entropy_reg_i2 = self.decode_lms[i](
                x_list[i], level=i, window=self.windows[i], compute_ntp=compute_ntp,
                decode_tracks=full_tracks, extra_query=(want_next_query and i == 0), multi_mode=multi_mode)
            decode_losses[i] = loss_i2
            decode_accs[i] = acc_i2
            # (level, mode_index, loss, acc) -- mode_index 1..T-1 (T = len(full_tracks), the final
            # mode T's own loss/acc is decode_losses[i]/decode_accs[i] above, not duplicated here).
            for m_idx, (l_m, a_m) in enumerate(zip(mode_losses_i, mode_accs_i), start=1):
                decode_stage_extra_losses.append((i, m_idx, l_m, a_m))
            if h_i2.shape[1] < L_i:
                # _merged_decode_forward always returns exactly L_i byte positions now (h_out is
                # gathered via extract_pos, one slot per input byte) -- this branch is dead in the
                # new decode path, kept only because generation can still route through the
                # encode-only fallback above (decode_tracks is None) for very short/ragged prefixes.
                h_i2 = torch.cat([h_i2, h_list[i][:, h_i2.shape[1]:, :]], dim=1)
            h_out[i] = h_i2
            next_query[i] = query_last_i
            query_seq_out[i] = query_seq_i
            if max_decode_sources is None:
                decode_derived_c[i] = c_i2

        return (encode_losses, encode_accs, decode_losses, decode_accs, h_out, c_list,
                next_query, decode_derived_c, query_seq_out, h_list[0], decode_stage_extra_losses,
                encode_entropy_regs)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        (encode_losses, encode_accs, decode_losses, decode_accs, h_list, c_list,
         _next_query, _decode_derived_c, _query_seq, h0_encode,
         decode_stage_extra_losses, encode_entropy_regs) = self._run(byte_ids)

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        # byte_loss_full: fixblock's own decode loss (byte_loss above, when decode ran) only covers
        # targets K0..L-1 -- targets 1..K0-1 (still inside block 0, no real code exists yet) are
        # excluded entirely rather than trained code-free. Splice encode's own loss (already
        # code-free by construction, same computation qcute_v5_concat.py's bos-anchored decode loss
        # effectively degenerated to for this range) back in for that range, weighted by position
        # count, so this stays comparable to qcute_v5_concat.py's val_bpb and to full-sequence
        # baselines (bytelm etc). Only meaningful once decode actually ran for level0.
        K0 = cfg.Ks[0]
        if decode_losses[0] is not None and K0 > 1:
            D0 = h0_encode.shape[-1]
            h0_partial = h0_encode[:, :K0 - 1, :].reshape(-1, D0)
            tgt0_partial = byte_ids[:, 1:K0].reshape(-1)
            logits0 = F.linear(h0_partial, self.encode_lms[0].embed.weight)
            enc_partial_loss = F.cross_entropy(logits0, tgt0_partial)
            n_enc, n_dec = K0 - 1, byte_ids.shape[1] - K0
            byte_loss_full = (enc_partial_loss * n_enc + decode_losses[0] * n_dec) / (n_enc + n_dec)
        else:
            byte_loss_full = byte_loss

        encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                              else byte_loss.new_zeros(()))
        encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

        decode_terms = [l for l in decode_losses if l is not None]
        decode_total = (cfg.decode_ntp_weight * torch.stack(decode_terms).sum() if decode_terms
                         else byte_loss.new_zeros(()))

        # decode_stage_extra_total: shallower-mode losses (self-only, self+track1, ...), mirroring
        # qcute_v5_stack.py's own decode_stage_extra_losses/-total for its staged cross-attention
        # intermediate stages -- only non-empty when Config.multi_mode_impl != "off" and a level has
        # more than one available track. Each entry is (level, mode_idx, loss, acc); mode_idx=1 is
        # self-only, mode_idx=2 is self+track1, etc. (mode T -- the deepest/full mode -- is
        # decode_losses[level]/decode_accs[level] above, not duplicated here).
        extra_loss_terms = [l for (_i, _m, l, _a) in decode_stage_extra_losses]
        decode_stage_extra_total = (cfg.decode_ntp_weight * torch.stack(extra_loss_terms).sum()
                                     if extra_loss_terms else byte_loss.new_zeros(()))

        # entropy_reg (BSQQuant only, see bsq_entropy_reg): sum across levels, weighted by
        # cfg.entropy_reg_weight (default 0.0 -- opt-in, matches archived qcutelm.py's default).
        entropy_reg_terms = [r for r in encode_entropy_regs if r is not None]
        entropy_reg_total = (torch.stack(entropy_reg_terms).sum() if entropy_reg_terms
                              else byte_loss.new_zeros(()))

        loss = encode_total + decode_total + decode_stage_extra_total + cfg.entropy_reg_weight * entropy_reg_total
        ntp_total = torch.stack(encode_losses + decode_terms + extra_loss_terms).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_loss_full": byte_loss_full, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_stage_extra_total": decode_stage_extra_total, "ntp_loss_total": ntp_total,
            "entropy_reg_total": entropy_reg_total,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
            **{f"level{lvl}_mode{m}_ntp_loss_decode": l for (lvl, m, l, _a) in decode_stage_extra_losses},
            **{f"level{lvl}_mode{m}_ntp_acc_decode": a for (lvl, m, _l, a) in decode_stage_extra_losses},
            **{f"level{i}_entropy_reg": r for i, r in enumerate(encode_entropy_regs) if r is not None},
        }

        return loss, metrics


def _sample_next_byte(embed_weight: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h_last, embed_weight)
    return logits.argmax(-1)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                       max_decode_sources: int | None = None) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    K0 = model.cfg.Ks[0]
    all_bytes = prompt_bytes
    for _ in range(n_new_bytes):
        L = all_bytes.shape[1]
        # No padding: _run/_merged_decode_forward are floor-tolerant now, so feeding the true,
        # growing byte sequence gives exactly the same decode-conditioned (or, below a level's
        # minimum block count, encode-only) representation training would compute at this position
        # -- never a fabricated trailing byte. want_next_query only matters (and is only honored)
        # on a K0-aligned prefix, where the merged-decode buffer's chronologically last slot can
        # be returned as a genuine bare-code extra query; elsewhere it's a no-op and
        # next_query[0] stays None.
        block_aligned = L % K0 == 0
        _, _, _, _, h_list, _, next_query, _decode_derived_c, _query_seq, _, _, _ = model._run(
            all_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        # next_query[0]: _merged_decode_forward's extra_query -- the buffer's chronologically
        # last hidden state, which automatically incorporates any code that just became available
        # (see its docstring). h_list[0][:, -1, :]: the standard byte-slot next-token
        # representation used when no track has a complete block yet (encode-only fallback) --
        # both are real, trained-for slots, matching what check_gen_consistency compares against.
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, query)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                       max_decode_sources: int | None = None) -> torch.Tensor:
    """FIFO-windowed generation: truncate to the trailing cfg.context_len bytes before every step,
    exactly mirroring training's own windowing (sample_context truncates to context_len the same
    way, and RoPE positions are always relative to whatever window is fed in, never absolute from
    generation's start -- so this isn't a new mechanism, it's training's existing window applied at
    generation time). Not a per-layer K/V tensor cache -- still recomputes the windowed forward each
    step -- but bounds that recompute to O(context_len) instead of O(current total length), which is
    what actually matters once generation runs long. New byte pushed on, oldest byte falls off once
    the window is full, same push-and-drop as any fixed-size FIFO.

    Only guaranteed to match generate_no_cache while prompt_len + n_new_bytes <= context_len (same
    caveat as qcute.bytelm's own generate_kv_cache/validate_generation pair) -- beyond that point
    generate_no_cache keeps growing its unbounded context while this one keeps sliding, so the two
    are expected to diverge by design, not by bug. See validate_generation.
    """
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    K0 = model.cfg.Ks[0]
    context_len = model.cfg.context_len
    all_bytes = prompt_bytes
    for _ in range(n_new_bytes):
        window_bytes = all_bytes[:, -context_len:]   # FIFO: only the trailing context_len bytes ever matter
        L = window_bytes.shape[1]
        block_aligned = L % K0 == 0
        _, _, _, _, h_list, _, next_query, _decode_derived_c, _query_seq, _, _, _ = model._run(
            window_bytes, compute_ntp=False, max_decode_sources=max_decode_sources,
            want_next_query=block_aligned)
        query = next_query[0] if next_query[0] is not None else h_list[0][:, -1, :]
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, query)
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


def validate_generation(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> bool:
    """Only meaningful while prompt_len + n_new_bytes <= model.cfg.context_len -- see
    generate_kv_cache's docstring for why the two are only guaranteed to agree inside that bound."""
    out_a = generate_no_cache(model, prompt_bytes, n_new_bytes, device)
    out_b = generate_kv_cache(model, prompt_bytes, n_new_bytes, device)
    assert torch.equal(out_a, out_b), (
        f"generate_no_cache and generate_kv_cache diverged:\n"
        f"  no_cache = {out_a.tolist()}\n"
        f"  kv_cache = {out_b.tolist()}"
    )
    return True


@torch.no_grad()
def generate_true_kv_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Genuine incremental per-layer K/V cache -- unlike generate_kv_cache (FIFO window
    re-truncation: recomputes the full windowed forward from scratch every step), this persists
    attention K/V tensors across steps and only does O(1) new-token/new-code work per step.
    Exploits RoPE's shift-invariance to use absolute, never-reset positions (see
    docs/kv_cache_design.md) instead of window-relative ones.

    Unlike qcute_v5.py's staged cross-attention decode (which needs a separate boundary-query
    stream since the strict code_pos<query_pos mask excludes a block's own just-completed code from
    predicting that block's own first element), the merged-buffer design here needs none of that:
    causal masking is plain buffer-INDEX order (buf_j<=buf_i), and a track's code entry is simply
    inserted into the SAME stream, chronologically right after the byte that completed its block --
    so it's ordinary single-stream causal self-attention with two kinds of tokens (bytes, codes)
    sharing one path, and the buffer's chronologically last entry (code if one was just appended
    this step, else the byte) is always the right query for the next byte, no special-casing needed.

    _skip's own addition: `evict_block_bytes` removes a block's raw byte K/V entries the moment
    that block's code is appended (not waiting for window-based eviction) -- the running byte K/V
    footprint stays O(K) instead of O(current position). No extra masking needed here (unlike the
    dense/training forward): eviction happens strictly before any later query could otherwise see
    those bytes, so physical removal alone enforces the same-block-only visibility rule for free.

    Scoped to n_levels==1 (single level, one self track) and code_extract_mode='last_h' -- general
    multi-track/multi-level caching is future work, see docs/kv_cache_design.md.
    """
    assert model.n_levels == 1, "generate_true_kv_cache: only n_levels==1 (single-level) supported so far"
    cfg = model.cfg
    assert cfg.code_extract_mode == "last_h", "generate_true_kv_cache: only code_extract_mode='last_h' supported so far"
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    B = prompt_bytes.shape[0]
    K = cfg.Ks[0]
    D = cfg.d_model
    H = cfg.n_heads
    hd = D // H
    enc = model.encode_lms[0]
    dec = model.decode_lms[0]
    enc_window = model.windows[0]
    dec_window = model.decode_windows[0][0]
    n_layers = len(enc.blocks)

    enc_k: list = [None] * n_layers
    enc_v: list = [None] * n_layers
    dec_k: list = [None] * n_layers
    dec_v: list = [None] * n_layers
    buf_true_pos = torch.zeros(0, dtype=torch.long, device=device)
    buf_is_code = torch.zeros(0, dtype=torch.bool, device=device)

    def encode_step(byte_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = enc.embed(byte_id).unsqueeze(1)
        q_pos = torch.tensor([pos], device=device)
        cos, sin = rope_cos_sin_for_positions(q_pos, hd, cfg.rope_base, device)
        for li, block in enumerate(enc.blocks):
            xn = block.ln1(x)
            qkv = block.attn.qkv(xn).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
            qn, kn, vn = qkv[0], qkv[1], qkv[2]
            qn = apply_rope(qn, cos, sin)
            kn = apply_rope(kn, cos, sin)
            enc_k[li] = kn if enc_k[li] is None else torch.cat([enc_k[li], kn], dim=2)
            enc_v[li] = vn if enc_v[li] is None else torch.cat([enc_v[li], vn], dim=2)
            if enc_window is not None and enc_k[li].shape[2] > enc_window:
                enc_k[li] = enc_k[li][:, :, -enc_window:]
                enc_v[li] = enc_v[li][:, :, -enc_window:]
            y = F.scaled_dot_product_attention(qn, enc_k[li], enc_v[li])
            x = x + block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
            x = x + block.mlp(block.ln2(x))
        return enc.ln_f(x)

    def dec_append(x_embed: torch.Tensor, true_pos: int, is_code: bool) -> torch.Tensor:
        nonlocal buf_true_pos, buf_is_code
        rope_pos = torch.tensor([max(true_pos, 0)], device=device)
        cos, sin = rope_cos_sin_for_positions(rope_pos, hd, cfg.rope_base, device)
        x = x_embed
        for li, block in enumerate(dec.blocks):
            xn = block.ln1(x)
            qkv = block.attn.qkv(xn).reshape(B, 1, 3, H, hd).permute(2, 0, 3, 1, 4)
            qn, kn, vn = qkv[0], qkv[1], qkv[2]
            qn = apply_rope(qn, cos, sin)
            kn = apply_rope(kn, cos, sin)
            if dec_k[li] is None:
                k_cat, v_cat, mask = kn, vn, None
            else:
                k_cat = torch.cat([dec_k[li], kn], dim=2)
                v_cat = torch.cat([dec_v[li], vn], dim=2)
                mask = None
                if dec_window is not None:
                    reach = true_pos - buf_true_pos
                    allow = torch.cat([reach < dec_window, torch.ones(1, dtype=torch.bool, device=device)])
                    mask = allow.view(1, 1, 1, -1)
            y = F.scaled_dot_product_attention(qn, k_cat, v_cat, attn_mask=mask)
            x = x + block.attn.out(y.transpose(1, 2).reshape(B, 1, D))
            x = x + block.mlp(block.ln2(x))
            dec_k[li], dec_v[li] = k_cat, v_cat
        buf_true_pos = torch.cat([buf_true_pos, torch.tensor([true_pos], device=device)])
        buf_is_code = torch.cat([buf_is_code, torch.tensor([is_code], device=device)])
        if dec_window is not None:
            keep = (true_pos - buf_true_pos) < dec_window
            if not keep.all():
                for li in range(n_layers):
                    dec_k[li] = dec_k[li][:, :, keep]
                    dec_v[li] = dec_v[li][:, :, keep]
                buf_true_pos = buf_true_pos[keep]
                buf_is_code = buf_is_code[keep]
        return dec.ln_f(x)

    def evict_block_bytes(block_start: int, block_end: int) -> None:
        # _skip: once a block's code has just been appended, that block's raw byte K/V entries are
        # permanently redundant (no query from this point on is allowed to see them, only the
        # code) -- evict them now instead of waiting for window-based eviction. Safe/deterministic:
        # nothing computed before this point ever depends on what happens after it.
        nonlocal buf_true_pos, buf_is_code
        evict = (~buf_is_code) & (buf_true_pos >= block_start) & (buf_true_pos <= block_end)
        if evict.any():
            keep = ~evict
            for li in range(n_layers):
                dec_k[li] = dec_k[li][:, :, keep]
                dec_v[li] = dec_v[li][:, :, keep]
            buf_true_pos = buf_true_pos[keep]
            buf_is_code = buf_is_code[keep]

    all_bytes = prompt_bytes
    L0 = all_bytes.shape[1]
    # No bos append (fixblock): block 0 has no real code before it, so nothing is prepended -- the
    # buffer simply starts at byte 0.
    for pos in range(L0 + n_new_bytes - 1):
        byte_id = all_bytes[:, pos]
        h_enc = encode_step(byte_id, pos)
        x_byte = dec.embed(byte_id).unsqueeze(1)
        out_this_pos = dec_append(x_byte, pos, is_code=False)
        if (pos + 1) % K == 0:
            pooled = h_enc.squeeze(1)
            pre_q = enc._classify(pooled)
            c_new = enc.quant.quantize(pre_q).unsqueeze(1)
            code_embed = dec.quant.embed_for_decode(dec, c_new)
            out_this_pos = dec_append(code_embed, pos, is_code=True)
            evict_block_bytes(pos - K + 1, pos)
        if pos < K - 1:
            # fixblock: byte at pos+1 is still inside block 0 (no real code exists before it yet)
            # until pos reaches K-1 (where code 0 just became available above) -- fall back to pure
            # encode output for those positions instead of sampling from an uncoditioned decode row.
            out_this_pos = h_enc
        if pos >= L0 - 1:
            next_byte = _sample_next_byte(dec.embed.weight, out_this_pos.squeeze(1))
            all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_self_only_cond(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    return generate_no_cache(model, prompt_bytes, n_new_bytes, device, max_decode_sources=1)


@torch.no_grad()
def generate_encode_only(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encode_lms[0]
    for _ in range(n_new_bytes):
        _, _, _, h, _, _, _, _, _ = enc0(all_bytes, level=0, window=model.windows[0], compute_ntp=False)
        next_byte = _sample_next_byte(enc0.embed.weight, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_level1_codes(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_codes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    enc0, enc1 = model.encode_lms[0], model.encode_lms[1]
    codes, _, _, _, _, _, _, _, _ = enc0(prompt_bytes, level=0, window=model.windows[0], compute_ntp=False)
    n_prompt_codes = codes.shape[1]
    for _ in range(n_new_codes):
        _, _, _, h1, _, _, _, _, _ = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
        next_code = enc1.quant.sample_next(enc1, h1[:, -1, :], model.cfg.vocab)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return enc1.quant.to_ids(codes[0, n_prompt_codes:])


@torch.no_grad()
def level1_ground_truth_codes(model: "RefineLM", full_bytes: torch.Tensor, prompt_len: int, device: str) -> torch.Tensor:
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    enc0 = model.encode_lms[0]
    K0 = model.cfg.Ks[0]
    c0, _, _, _, _, _, _, _, _ = enc0(full_bytes, level=0, window=model.windows[0], compute_ntp=False)
    ids = enc0.quant.to_ids(c0[0])
    n_prompt_codes = prompt_len // K0
    return ids[n_prompt_codes:]


def _annotate_bytes_with_codes(byte_ids: torch.Tensor, code_ids: torch.Tensor, K: int) -> str:
    bytes_list = byte_ids.tolist()
    codes_list = code_ids.tolist()
    parts = []
    for b in range(0, len(bytes_list), K):
        block = "".join(repr(bytes([x]))[2:-1] for x in bytes_list[b:b + K])
        code_idx = b // K
        c = codes_list[code_idx] if code_idx < len(codes_list) else "?"
        parts.append(f"{block}{{{c}}}")
    return "".join(parts)


@torch.no_grad()
def _decode_source_codes(model: "RefineLM", full_bytes: torch.Tensor, device: str, level: int = -1) -> torch.Tensor:
    was_training = model.training
    model.eval()
    seq_repr = full_bytes.to(device)
    if seq_repr.dim() == 1:
        seq_repr = seq_repr.unsqueeze(0)
    c_list = []
    for i in range(model.n_levels):
        c_i, _, _, _, _, _, _, _, _ = model.encode_lms[i](seq_repr, level=i, window=model.windows[i], compute_ntp=False)
        c_list.append(c_i)
        seq_repr = c_i
    source_c = c_list[level]
    if was_training:
        model.train()
    return model.encode_lms[level].quant.to_ids(source_c)[0]


@torch.no_grad()
def check_gen_consistency(model: "RefineLM", full_bytes: torch.Tensor, device: str,
                           prompt_len: int = 32, tol: float = 1e-3, log=print, label: str = "") -> int:
    """Cheap, reusable correctness check: the incremental no-cache generation code path and the
    one-shot teacher-forced training forward pass MUST produce identical logits when both are fed
    the same ground-truth bytes -- any gap is a genuine generation-vs-training bug (this caught
    two real ones: the windowed-attention dense-fallback and the can_chunk track-dropping bug, see
    docs/status.md), not exposure bias or model quality. Returns the mismatch count (0 = pass).
    """
    was_training = model.training
    model.eval()
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    L_total = full_bytes.shape[1]
    embed0 = model.decode_lms[0].embed.weight
    K0 = model.cfg.Ks[0]

    _, _, _, _, h_list_tf, _, _, _, query_seq_tf, _, _, _ = model._run(
        full_bytes, compute_ntp=False, max_decode_sources=None, want_next_query=False)
    # query_seq_tf[0] is always None with the chronological merged-interleave decode --
    # _merged_decode_forward already extracts each byte's NTP-training query correctly via
    # extract_pos (the last buffer entry sharing that true_pos, a tied code's state when one
    # completes there -- see _merged_layout's docstring), so h_list_tf[0] alone is always the
    # right reference tensor. query_seq stays part of the _run contract only for shape parity
    # with qcute_v5_concat_slow.py's older, separately-tracked single-track mechanism.
    using_query_seq = query_seq_tf[0] is not None
    query_ref_tf = query_seq_tf[0] if using_query_seq else h_list_tf[0]
    logits_tf_all = F.linear(query_ref_tf[0], embed0)

    n_mismatch = 0
    for t in range(prompt_len, L_total - 1):
        ref_idx = t - K0 if using_query_seq else t - 1
        if ref_idx < 0 or ref_idx >= logits_tf_all.shape[0]:
            continue
        padded = full_bytes[:, :t]
        _, _, _, _, h_list_gen, _, next_query_gen, _, _, _, _, _ = model._run(
            padded, compute_ntp=False, max_decode_sources=None, want_next_query=True)
        query_gen = next_query_gen[0] if next_query_gen[0] is not None else h_list_gen[0][:, t - 1, :]
        logits_gen = F.linear(query_gen[0], embed0)
        if (logits_gen - logits_tf_all[ref_idx]).abs().max().item() >= tol:
            n_mismatch += 1
    if was_training:
        model.train()
    prefix = f"gen_consistency_{label}" if label else "gen_consistency"
    log(f"{prefix}: {n_mismatch}/{L_total - 1 - prompt_len} timesteps mismatched "
        f"(generation vs teacher-forced logits on ground-truth input)")
    return n_mismatch


def qualitative_generate(model: "RefineLM", prompt_bytes: torch.Tensor, gen_len: int,
                          ground_truth: torch.Tensor | None, device: str, log=print, label: str = "") -> None:
    prefix = f"qual_{label}_" if label else "qual_"
    out_uncond = generate_encode_only(model, prompt_bytes, gen_len, device)
    gen_bytes_uncond = bytes(out_uncond[prompt_bytes.numel():].tolist())
    log(f"{prefix}prompt:              {bytes(prompt_bytes.tolist())!r}")
    if ground_truth is not None:
        log(f"{prefix}ground_truth:        {bytes(ground_truth.tolist())!r}")
    log(f"{prefix}level0_uncond:       {gen_bytes_uncond!r}")
    # Every possible level0 conditioning depth, m=1 (self-only) .. n_levels (self+every coarser
    # track, == the old "cond_full"/max_decode_sources=None) -- generate_no_cache's own tracks
    # loop stops early (fewer than n_levels tracks) if a coarser block isn't affordable yet within
    # gen_len, so m beyond what's actually reachable just re-runs the deepest available mode; still
    # correct, just redundant with the previous m's output in that case.
    out_cond_full = None
    for m in range(1, model.n_levels + 1):
        out_m = generate_no_cache(model, prompt_bytes, gen_len, device, max_decode_sources=m)
        gen_bytes_m = bytes(out_m[prompt_bytes.numel():].tolist())
        tag = "full" if m == model.n_levels else str(m)
        log(f"{prefix}level0_mode{tag}:      {gen_bytes_m!r}")
        if m == model.n_levels:
            out_cond_full = out_m
    decode_K = 1
    for k in model.cfg.Ks:
        decode_K *= k
    # code_ids_full = _decode_source_codes(model, out_cond_full, device, level=-1)
    # n_prompt_codes = prompt_bytes.numel() // decode_K
    # gen_code_ids = code_ids_full[n_prompt_codes:]
    # annotated = _annotate_bytes_with_codes(out_cond_full[prompt_bytes.numel():], gen_code_ids, decode_K)
    # log(f"{prefix}level0_cond_full_codes: {annotated}  <{gen_code_ids.tolist()}>")
    if model.n_levels > 1:
        K0 = model.cfg.Ks[0]
        n_new_codes = gen_len // K0
        if n_new_codes > 0:
            level1_gen = generate_level1_codes(model, prompt_bytes, n_new_codes, device)
            log(f"{prefix}level1_gen:          {level1_gen.tolist()}")
            if ground_truth is not None:
                full_bytes = torch.cat([prompt_bytes.reshape(-1), ground_truth.reshape(-1)])
                level1_gt = level1_ground_truth_codes(model, full_bytes, prompt_bytes.numel(), device)
                log(f"{prefix}level1_gt:           {level1_gt.tolist()}")


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


_warned_short_data: set[int] = set()


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    if len(data) < context_len and id(data) not in _warned_short_data:
        _warned_short_data.add(id(data))
        print(f"WARNING: sample_context data ({len(data)} bytes) is shorter than context_len "
              f"({context_len}) -- every batch from this split is silently truncated to {len(data)} "
              f"bytes (e.g. a val split under a large context_len). Not an error by itself (LevelLM's "
              f"single-track NTP target now matches whatever length actually comes out), but the "
              f"resulting bpb/loss numbers reflect a shorter context than configured.")
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
    for k in list(result.keys()):
        if k.endswith("_ntp_loss_encode") or k.endswith("_ntp_loss_decode"):
            result[k.replace("_ntp_loss_", "_bpb_")] = result[k] / math.log(2)
    return result


def eval_model(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    result["bpb_full"] = result["byte_loss_full"] / math.log(2)
    return _add_per_level_bpb(result)


@torch.no_grad()
def eval_model_full(model: RefineLM, data: torch.Tensor, batch_size: int, device: str) -> dict:
    """Deterministic full-val-set pass: non-overlapping context_len windows, walked in
    fixed chronological order starting at byte 0 (never random), each byte scored exactly
    once -- unlike eval_model's random-with-replacement sampling, this is reproducible and
    exhaustive (up to a <context_len remainder at the end, dropped)."""
    model.eval()
    context_len = model.cfg.context_len
    n_windows = len(data) // context_len
    accum: dict[str, float] = {}
    total_n = 0
    for start in range(0, n_windows, batch_size):
        idxs = range(start, min(start + batch_size, n_windows))
        ctx = torch.stack([data[i * context_len:(i + 1) * context_len] for i in idxs]).to(device)
        _, metrics = model(ctx)
        bsz = ctx.size(0)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v.item() * bsz
        total_n += bsz
    model.train()
    result = {k: v / total_n for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    result["bpb_full"] = result["byte_loss_full"] / math.log(2)
    return _add_per_level_bpb(result)


def build_param_groups(model: RefineLM) -> list[dict]:
    seen: set[int] = set()
    params = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return [{"params": params}]


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
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

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
        )
        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = _add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.qual_gen_bytes > 0:
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                          window[args.qual_prompt_bytes:], device, log=log, label=label)
                    check_gen_consistency(model, window, device, prompt_len=args.qual_prompt_bytes,
                                           log=log, label=label)


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Chronological merged-interleave decode, independent per-level weights", parents=[pre])
    p.add_argument("--Ks", default=(32, 32))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", default=32)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--decode_ntp_weight", type=float, default=1.0)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--code_sample_mode", type=str, default="ste", choices=["ste", "sample", "soft"])
    p.add_argument("--code_extract_mode", type=str, default="last_h",
                    choices=["last_h", "softmax_pool", "light_query_attn", "query_embed"])
    p.add_argument("--code_head_tied", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--quant_type", type=str, default="softmax", choices=["softmax", "bsq", "fsq"])
    p.add_argument("--bsq_bits", type=int, default=4)
    p.add_argument("--bsq_lfq", action="store_true")
    p.add_argument("--entropy_reg_weight", type=float, default=0.0)
    p.add_argument("--ntp_head_tied", action="store_true")
    p.add_argument("--fsq_dq", type=int, default=6)
    p.add_argument("--fsq_levels", type=int, default=8)
    p.add_argument("--fsq_bound", type=str, default="sigmoid", choices=["sigmoid", "tanh"])
    p.add_argument("--multi_mode_impl", type=str, default="off", choices=["off", "multipass", "single_pass"])

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--qual_gen_bytes", type=int, default=0)
    p.add_argument("--qual_prompt_bytes", type=int, default=64)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--compile", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--eval_only", action="store_true", help="skip training; load --checkpoint_path and run a full-val-set eval pass")
    p.add_argument("--checkpoint_path", type=Path, default=None, help="required with --eval_only")

    _cli_excluded_config_fields: set[str] = set()
    _missing_from_cli = ({f.name for f in dataclass_fields(Config)} - {a.dest for a in p._actions}
                          - _cli_excluded_config_fields)
    assert not _missing_from_cli, (
        f"Config field(s) {_missing_from_cli} have no matching --arg registered in main()'s argparse "
        f"setup -- a config .py file setting them would be silently ignored. Add a p.add_argument(...) "
        f"for each, and pass it through to Config(...) below.")

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.Ks = _parse_int_tuple(args.Ks)

    if args.eval_only and args.checkpoint_path is None:
        p.error("--eval_only requires --checkpoint_path")

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers, context_len=args.context_len,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window, rope_base=args.rope_base,
        byte_ntp_weight=args.byte_ntp_weight, code_ntp_weight=args.code_ntp_weight,
        decode_ntp_weight=args.decode_ntp_weight, gumbel_tau=args.gumbel_tau,
        code_extract_mode=args.code_extract_mode, code_head_tied=args.code_head_tied,
        vocab=args.vocab, quant_type=args.quant_type, bsq_bits=args.bsq_bits, bsq_lfq=args.bsq_lfq,
        entropy_reg_weight=args.entropy_reg_weight, ntp_head_tied=args.ntp_head_tied,
        fsq_dq=args.fsq_dq, fsq_levels=args.fsq_levels, fsq_bound=args.fsq_bound,
        code_sample_mode=args.code_sample_mode, multi_mode_impl=args.multi_mode_impl,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_4_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} d_model={cfg.d_model} n_layers={cfg.n_layers} seq_lens={model.seq_lens} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    if args.eval_only:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        val = eval_model_full(model, val_data, args.batch_size, device)
        val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
        log(f"eval_only_full_valset  {val_str}", **{f"val_{k}": v for k, v in val.items()})
        return

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
