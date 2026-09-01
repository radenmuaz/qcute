"""Encoder-decoder SummFormer -- successor to model_summformer_v2.py's SummTransformerV2 (frozen
per-lineage copies in lm/, image_gen/, image_classification/). No backward compatibility with those
configs; this is a clean redesign. Copies (not imports -- this file has no cross-directory
dependency, self-contained like every other lineage's summformer.py) the RoPE/attention primitives
(Attn, MLP, Block, chunked_windowed_attention, windowed_cross_attention, sdpa_with_sink,
causal_mask, rope_cos_sin_for_positions, cross_entropy) verbatim from that lineage -- only the
orchestration above the block level changes.

Why the old design was replaced (see chat/docs/status_tpu.md 2026-08-30):
  - SummTransformerV2's `source_index=-1` fuse-stage pooling always re-pooled from the FULL current
    trunk sequence at every stage, independently -- strides never compounded across stages, so
    stacking more fuse stages at a small stride never meaningfully extended receptive field
    (confirmed empirically AND by re-deriving from model_summformer_v1.py, which DOES chain: each
    stage pools from the PREVIOUS stage's already-downsampled code output, cum_K = product of each
    stage's stride, giving real multiplicative reach growth). That v1 mechanism is restored here as
    Encoder's chained pooling.
  - Cross-attention was bolted onto specific `dst` trunk depths as an all-or-nothing extra
    self-attn+MLP block (FuseStageV2), not a per-layer sublayer -- Decoder here is a standard
    self-attn -> cross-attn -> MLP stack every layer, cross-attn optional per layer.
  - Encoder-decoder was never a first-class option -- SummTransformerV2 only supported one
    self-referential trunk (encoder implicitly = decoder's own early layers). That's now the
    default topology's SPECIAL CASE (n_shared_layers > 0, Encoder has 0 of its own additional
    layers), not a separate code path: Decoder's cross-attention always reads from Encoder's output
    list, whether Encoder shares layers with Decoder's own trunk or is a fully separate stack (e.g.
    a different modality/vocab, still causal -- this codebase's windowed attention has no
    non-causal mode, encoder included).

Receptive field note (do not re-derive incorrectly, see chat): chaining compounds MULTIPLICATIVELY
in the number of CHAINED encoder stages via cum_stride = product(stage strides) -- NOT in
n_layers, and NOT automatically "K**n_layers" just from having many decoder layers. A stage's own
code_window, mapped back to absolute input positions, spans `code_window * cum_stride_at_that_stage`
-- so cum_stride growth is what compounds, bounded by how many stages are actually chained.

TODO (unimplemented, this file is currently a byte-identical copy of summformer_slow.py -- see
chat 2026-08-31/09-01 for the full derivation): query-sparse Encoder chain pooling, to save compute
at chain stages whose full output is never actually consumed.

PROBLEM. Encoder.__call__'s chain loop (see Encoder class below) does, per stage i:
    stage_h = cur_h[:, stage.stride-1::stage.stride, :][:, :n_blocks, :]   # subsample FIRST
    stage_h = self.transformers[i](stage_h, local_pos)                     # then attend among survivors
This is already cheap WITHIN stage i's own transformer call (BlockStack only ever processes the
already-subsampled n_blocks_i positions, never the full pre-subsample length -- confirmed by
reading BlockStack._forward_impl, it just loops over whatever length it's handed, no hidden
full-length work). The real waste is ACROSS stages: stage i computes ALL n_blocks_i outputs, but if
stage i is not read by the Decoder's cross-attention (i.e. not in that CrossAttnSpec's
`encoder_output` set) and only feeds stage i+1 forward, then stage i+1's OWN subsample immediately
discards all but 1/stride_{i+1} of what stage i just computed. That (stride_{i+1}-1)/stride_{i+1}
fraction of stage i's query/attention-score/out-projection/MLP compute is provably never read by
anything -- proven via a forward-pass argument (self-attention at query position j depends only on
q_j and the keys/values in j's own window, never on whether some OTHER position's output was also
computed), NOT by a numerical equivalence test (that test was run and DISCONFIRMED: a
"gather-raw-window-then-attend" rectangular-attention variant, prototyped and then deleted this
session, is NOT numerically equivalent to this file's subsample-then-square-attend pooling --
0.14%-7.3% relative diff at a single stage in isolation, depending on weight scale -- see chat).
This TODO is explicitly a compatibility-BREAKING optimization: an existing checkpoint trained on
plain summformer.py/summformer_slow.py will not reproduce the same outputs after this change (same
param shapes, different learned function) -- confirmed acceptable per user instruction
("just break compatibility") once the narrower fully-compatible version (skip only the
provably-dead OUTPUT computation for known-discarded positions, keeping the exact same window/mask
math) was scoped out as strictly more conservative but harder to wire generically through the
Decoder's CrossAttnSpec-driven `encoder_output` set.

FIX. For each chain stage i NOT needed in full downstream (see "which stages" below), replace
subsample-then-attend with a single strided-window ("causal strided-conv-with-attention-weights")
op that computes queries ONLY at the positions that survive onward, with keys/values gathered from
a LOCAL RAW window (not the full stage input, and not just other survivors):
    query_idx  = arange(stride-1, n_blocks*stride, stride)                       # static, from stride
    offsets    = arange(window-1, -1, -1)                                        # static, from window
    gather_idx = clip(query_idx[:, None] - offsets[None, :], 0, L-1)             # static -- (n_blocks, window)
    valid      = (query_idx[:, None] - offsets[None, :]) >= 0                    # static
    x_q        = take(cur_h, query_idx, axis=1)              # (B, n_blocks, D)          -- gather BEFORE projecting
    kv_raw     = take(cur_h, gather_idx, axis=1)              # (B, n_blocks, window, D)  -- gather BEFORE projecting
    q          = wq(x_q); k, v = split(wkv(kv_raw))            # project only the gathered positions, not all of L
    out        = attend(q, k, v, mask=valid)                   # (B, n_blocks, D) -- ALREADY next stage's input shape
Gathering raw positions BEFORE projecting through wq/wkv (rather than projecting the whole stage
input then gathering afterward) is a pure compute-cost fix with ZERO numerical effect (wq/wkv are
per-position nnx.Linear, no cross-position mixing, so gather-then-project == project-then-gather
for the selected positions, up to floating-point reassociation) -- this is what makes the op cost
`n_blocks * window` instead of `O(L)`; the earlier deleted rectangular-attention prototype got this
part wrong (projected wkv over the FULL L then gathered, confirmed via GFLOPs benchmark to be the
dominant cost: ~3.08 of 6.60 GFLOPs in query_hourglass_tiny_2.py). `stride` and `window` are exactly
ChainStageConfig's existing fields -- no new config semantics needed, just consuming both together
in one op instead of sequentially (slice, then separately window-attend on the slice).

WHICH STAGES benefit (savings are NOT uniform across a chain): a stage feeding the Decoder's
cross-attention (its index appears as some CrossAttnSpec.encoder_output) needs ALL n_blocks_i
positions regardless -- zero savings there, must stay as full self-attention over all n_blocks_i
survivors (either the current subsample-then-attend, or this same op with query_idx = arange(n_blocks_i),
i.e. no cut). A stage NOT cross-attended, whose only consumer is the next chain stage, can cut its
OWN query count down to n_blocks_i / stride_{i+1} (single-hop: push the next stage's stride into
this stage's own output) -- see chat's cumulative-savings derivation: with uniform stride s and
depth >= 2, this yields a flat, DEPTH-INDEPENDENT cumulative saving of exactly (s-1)/s on the
affected sub-chain (50% at s=2, 75% at s=4), confirmed both analytically (geometric-series partial
sums are self-similar) and by direct enumeration for depth=8. A more aggressive fully-cascaded
version (collapse an entire run of consecutive non-cross-attended stages to directly target the
count needed by the nearest downstream consumer, skipping the intermediate stages' own
representation-building self-attention entirely) yields much larger savings (~85x on the collapsed
sub-chain in the query_hourglass_tiny_2_last4.py numeric example) but is a bigger architectural
change -- not scoped here, implement single-hop first.

CONCRETE EXAMPLE (query_hourglass_tiny_2_last4.py: T=150528, stride=4, 8 stages,
CROSS_STAGES=(4,5,6,7)): stages 0-3 are the only ones eligible (not in CROSS_STAGES); single-hop
fix cuts their combined chain-attention cost from ~51.2M to ~12.8M (D^2-weighted proxy units) --
overall chain-attention total drops from ~63.9M to ~25.5M, i.e. ~2.5x cheaper chain compute (~60%
reduction), stages 4-7 unaffected (cross-attended, no savings possible). Rough Amdahl estimate for
TOTAL model GFLOPs (chain-attention is roughly half of total per the query_hourglass_tiny_2 GFLOPs
breakdown): ~1.3-1.4x total speedup. Verify the real number with
summformer_jax/image_classification/scripts/bench_last4_fast_vs_slow.py's cost_analysis() method
once implemented, rather than trusting this estimate.

IMPLEMENTATION NOTES.
  - `stride`/`window` per stage, and which stages are cross-attended (derive from the Decoder's
    `cross: tuple[CrossAttnSpec]` passed alongside the Encoder, or thread an explicit
    `cross_stages: frozenset[int]` into Encoder.__init__/__call__ -- Encoder currently has no idea
    which of its own stages the Decoder reads, this needs wiring through).
  - `query_idx`/`gather_idx`/`valid` are pure functions of static config ints (context_len, chain
    strides up to stage i, that stage's window) -- precompute ONCE at Encoder.__init__ (mirroring
    how `_resolve_windows`/`cum_strides`/`max_blocks` are already precomputed there), not inside
    __call__, and definitely not from traced/dynamic shapes.
  - Level 1 (the Embedder, upstream of the whole Encoder chain) is explicitly OUT OF SCOPE for this
    trick in the lm/ lineage: NTP loss needs a valid per-position output at every raw position, so
    Embedder must stay dense/square. This constraint doesn't apply to image_classification (no
    per-position loss, only a pooled query head) -- worth checking separately whether Embedder-level
    savings are worth adding there too, but keep that as a distinct change from this Encoder-chain fix.
  - Numerical check for whatever gets implemented: it will NOT match summformer_slow.py bit-for-bit
    (that's expected and accepted, see PROBLEM above) -- the correctness bar instead is (a) shapes
    match summformer_slow.py's Encoder output list exactly (same (stage_h_trunk, pos_abs,
    cum_stride) structure, same dtypes), (b) forward pass produces finite, non-NaN outputs across a
    real config, (c) at the degenerate stride=1 case (no compression at all), this scheme MUST
    reduce to bit-identical output vs summformer_slow.py (query_idx becomes arange(L), gather_idx
    becomes a plain causal window -- a genuine regression check that the gather machinery itself has
    no bug, independent of the stride>1 architecture-change question).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

# ----------------------------------------------------------------------------
# RoPE + attention primitives (from model_summformer.py)
# ----------------------------------------------------------------------------

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}


def rope_cos_sin_for_positions(position_ids: jnp.ndarray, head_dim: int, base: float):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    if cos.ndim == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


def sdpa_with_sink(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, attn_mask: jnp.ndarray,
                    use_sink: bool = True, use_flash: bool = False) -> jnp.ndarray:
    """use_flash=True (only takes effect when use_sink=False -- the sink token has no equivalent
    in jax.nn.dot_product_attention's mask-only API) dispatches to JAX's fused/flash-capable
    kernel instead of the manual einsum+softmax+einsum -- opt-in, default False, no behavior
    change for any existing config (see chat 2026-08-30: tried on request; this codebase's
    attention calls are mostly small/already-windowed so flash's actual benefit -- avoiding O(T^2)
    materialization -- rarely applies here, this is measuring that directly rather than assuming)."""
    scale = 1.0 / math.sqrt(k.shape[-1])
    if not use_sink:
        if use_flash:
            qT, kT, vT = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
            y = jax.nn.dot_product_attention(qT, kT, vT, mask=attn_mask, scale=scale)
            return y.transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        scores = jnp.where(attn_mask, scores.astype(jnp.float32), -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bhkd->bhqd", attn, v)

    B, H, _, hd = q.shape
    sink_k = jnp.zeros((B, H, 1, hd), dtype=k.dtype)
    sink_v = jnp.zeros((B, H, 1, hd), dtype=v.dtype)
    k2 = jnp.concatenate([sink_k, k], axis=2)
    v2 = jnp.concatenate([sink_v, v], axis=2)
    sink_col = jnp.ones(attn_mask.shape[:-1] + (1,), dtype=bool)
    mask2 = jnp.concatenate([sink_col, attn_mask], axis=-1)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k2) * scale
    scores = jnp.where(mask2, scores, -jnp.inf)
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v2.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v2)


def causal_mask(query_pos: jnp.ndarray, key_pos: jnp.ndarray, window) -> jnp.ndarray:
    allow = key_pos.reshape(1, -1) <= query_pos.reshape(-1, 1)
    if window is not None:
        allow = allow & ((query_pos.reshape(-1, 1) - key_pos.reshape(1, -1)) < window)
    return allow.reshape(1, 1, *allow.shape)


def chunked_windowed_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, window: int,
                                use_sink: bool = True, use_flash: bool = False) -> jnp.ndarray:
    """Real O(T*window) windowed self-attention -- ported from
    qcute/qcute_lagcodec/qcute_lagcodec_common.py's chunked_windowed_attention (torch), JAX'd and
    extended with the zero-KV sink. Reshapes into `window`-sized blocks; each block attends only
    to itself + the immediately preceding block (2*window keys, not T keys) -- exactly matches
    causal_mask(..., window)'s semantics (verified: for a query at local position li in its chunk,
    a same-chunk key is always within window by construction; a previous-chunk key at local
    position lj is within window iff li < lj, which is exactly what causal_mask's `i-j < window`
    condition gives), not an approximation. Falls back to dense sdpa_with_sink when T<=window or
    T%window!=0, same as the reference."""
    B, H, T, hd = q.shape
    w = window
    if T <= w:
        mask = causal_mask(jnp.arange(T), jnp.arange(T), None)
        return sdpa_with_sink(q, k, v, mask, use_sink, use_flash)
    if T % w != 0:
        mask = causal_mask(jnp.arange(T), jnp.arange(T), w)
        return sdpa_with_sink(q, k, v, mask, use_sink, use_flash)

    n_chunks = T // w
    qb = q.reshape(B, H, n_chunks, w, hd)
    kb = k.reshape(B, H, n_chunks, w, hd)
    vb = v.reshape(B, H, n_chunks, w, hd)
    pad_k = jnp.zeros((B, H, 1, w, hd), dtype=k.dtype)
    pad_v = jnp.zeros((B, H, 1, w, hd), dtype=v.dtype)
    k_ext = jnp.concatenate([pad_k, kb], axis=2)
    v_ext = jnp.concatenate([pad_v, vb], axis=2)

    idx = jnp.arange(n_chunks).reshape(n_chunks, 1) + jnp.arange(2).reshape(1, 2)
    k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)
    v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, 2 * w, hd)

    pos = jnp.arange(T)
    pos_b = pos.reshape(n_chunks, w)
    pad_pos = jnp.full((1, w), -10 ** 9, dtype=pos.dtype)
    pos_ext = jnp.concatenate([pad_pos, pos_b], axis=0)
    pos_win = pos_ext[idx].reshape(n_chunks, 2 * w)

    ti = pos_b[:, :, None]
    tj = pos_win[:, None, :]
    allow = (tj <= ti) & (ti - tj < w)  # (n_chunks, w, 2*w)
    mask_flat = jnp.broadcast_to(allow[None, :, None], (B, n_chunks, 1, w, 2 * w)).reshape(B * n_chunks, 1, w, 2 * w)

    qb_flat = qb.transpose(0, 2, 1, 3, 4).reshape(B * n_chunks, H, w, hd)
    k_win_flat = k_win.transpose(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)
    v_win_flat = v_win.transpose(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * w, hd)

    y = sdpa_with_sink(qb_flat, k_win_flat, v_win_flat, mask_flat, use_sink, use_flash)
    return y.reshape(B, n_chunks, H, w, hd).transpose(0, 2, 1, 3, 4).reshape(B, H, T, hd)


def windowed_cross_attention(q, k, v, seq_pos, code_pos_abs, stride: int, window: int,
                              use_sink: bool = True) -> jnp.ndarray:
    """Real O(T*window/stride) windowed CROSS-attention (trunk queries -> code-LM keys), instead
    of forward_cross's O(T*S) dense-plus-mask -- q_len=T and kv_len=S=T/stride differ, so this
    can't reuse chunked_windowed_attention's same-length block-reshape trick. Approach: for each
    trunk query position i, gather the `n_gather` code positions immediately at-or-before
    floor((i+1)/stride)-1 (code_pos_abs is monotonic, spaced `stride` apart), then apply
    causal_mask's EXACT SAME predicate (code_pos_abs[j] <= i and i - code_pos_abs[j] < window)
    restricted to that small gathered slice -- correct by construction as long as n_gather is wide
    enough to contain every entry the dense mask would allow (n_gather = ceil(window/stride) + 1,
    computed by the caller from stride/window, sized so the oldest gathered code position is always
    at or before the window cutoff). Degenerate case: window < stride means many queries may see
    ZERO valid code keys (same as the dense-masked path would give for the same window value --
    not specific to this implementation, an inherent conflict between window and stride)."""
    B, H, T, hd = q.shape
    _, _, S, _ = k.shape
    n_gather = min(S, max(1, -(-window // stride) + 1))  # ceil(window/stride) + 1, capped at S

    j_max = jnp.floor_divide(seq_pos + 1, stride) - 1  # (T,) most recent code index <= this query
    offsets = jnp.arange(n_gather - 1, -1, -1)
    gather_idx = j_max[:, None] - offsets[None, :]  # (T, n_gather)
    valid = (gather_idx >= 0) & (gather_idx < S)
    clipped = jnp.clip(gather_idx, 0, S - 1)

    k_g = jnp.take(k, clipped, axis=2)  # (B, H, T, n_gather, hd)
    v_g = jnp.take(v, clipped, axis=2)
    gathered_pos = code_pos_abs[clipped]  # (T, n_gather)
    allow = valid & (gathered_pos <= seq_pos[:, None]) & ((seq_pos[:, None] - gathered_pos) < window)
    mask = allow.reshape(1, 1, T, n_gather)

    scale = 1.0 / math.sqrt(hd)
    if use_sink:
        sink = jnp.zeros((B, H, T, 1, hd), dtype=k.dtype)
        k_g = jnp.concatenate([sink, k_g], axis=3)
        v_g = jnp.concatenate([sink, v_g], axis=3)
        sink_col = jnp.ones((1, 1, T, 1), dtype=bool)
        mask = jnp.concatenate([sink_col, mask], axis=-1)

    scores = jnp.einsum("bhtd,bhtnd->bhtn", q, k_g) * scale
    scores = jnp.where(mask, scores.astype(jnp.float32), -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1).astype(v_g.dtype)
    return jnp.einsum("bhtn,bhtnd->bhtd", attn, v_g)


class Attn(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, scale_layers: int, dtype, param_dtype, *,
                 rngs: nnx.Rngs, use_sink: bool = True, use_flash: bool = False):
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.attn_dim = n_heads * self.head_dim
        self.use_sink = use_sink
        self.use_flash = use_flash
        init = nnx.initializers.normal(stddev=0.02)
        out_init = nnx.initializers.normal(stddev=0.02 * (2 * scale_layers) ** -0.5)
        self.wq = nnx.Linear(d_model, self.attn_dim, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        # wk/wv fused into one Linear (always applied to the SAME input tensor in every call site
        # -- _qkv, forward_cross, forward_cross_windowed all compute k,v from one shared x) -- one
        # kernel dispatch instead of two, everywhere in the model (see chat 2026-08-30: many small
        # attention calls' dispatch overhead, not matmul FLOPs, was confirmed the actual step-time
        # bottleneck at this model scale).
        self.wkv = nnx.Linear(d_model, 2 * self.attn_dim, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.out = nnx.Linear(self.attn_dim, d_model, use_bias=True, kernel_init=out_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def _qkv(self, x, B, T):
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        kv = self.wkv(x).reshape(B, T, 2, H, hd)
        k = kv[:, :, 0].transpose(0, 2, 1, 3)
        v = kv[:, :, 1].transpose(0, 2, 1, 3)
        return q, k, v

    def forward(self, x, cos, sin, attn_mask, pos_method: str) -> jnp.ndarray:
        B, T, D = x.shape
        q, k, v = self._qkv(x, B, T)
        if pos_method == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = sdpa_with_sink(q, k, v, attn_mask, self.use_sink, self.use_flash)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

    def prime_static_cache(self, x, cos, sin, cap: int, window, pos_method: str):
        """Prefill: dense/chunked-windowed forward pass over the whole prompt (bit-identical to
        forward/forward_windowed), plus builds the INITIAL fixed-size circular KV cache (size
        `cap`, a static Python int -- window if bounded, else the layer's own cache_cap = the
        model's context_len, decided by the caller) from the prompt's last `cap` positions. `cap`
        is always >= window when window is not None (the sliding window never needs more than
        `window` slots); for window=None it equals context_len so the cache can hold the whole
        sequence without ever wrapping -- still a static shape, just sized to the known max."""
        B, T, D = x.shape
        q, k, v = self._qkv(x, B, T)
        if pos_method == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None:
            y = chunked_windowed_attention(q, k, v, window, self.use_sink)
        else:
            mask = causal_mask(jnp.arange(T), jnp.arange(T), None)
            y = sdpa_with_sink(q, k, v, mask, self.use_sink)
        out = self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

        # Seed the circular buffer via scatter into slot=(absolute_position % cap), matching
        # forward_incremental_static's decode-time addressing exactly -- NOT sequential placement
        # into slots [0..n_keep-1]. These only coincide when T % cap == 0; otherwise sequential
        # placement silently misaligns which slot holds which position, causing decode-time writes
        # to evict the wrong (still-valid) entry instead of the truly-oldest one. Unwritten slots
        # (only possible when T < cap) default to a sentinel position (-10**9), which causal_mask's
        # `(query - key) < window` clause always rejects (never needs a separate valid mask).
        H, hd = self.n_heads, self.head_dim
        n_keep = min(T, cap)
        keep_positions = jnp.arange(T - n_keep, T, dtype=jnp.int32)
        slots = keep_positions % cap
        k_buf = jnp.zeros((B, H, cap, hd), dtype=k.dtype).at[:, :, slots, :].set(k[:, :, -n_keep:])
        v_buf = jnp.zeros((B, H, cap, hd), dtype=v.dtype).at[:, :, slots, :].set(v[:, :, -n_keep:])
        pos_buf = jnp.full((cap,), -10 ** 9, dtype=jnp.int32).at[slots].set(keep_positions)
        write_pos = jnp.array(T, dtype=jnp.int32)  # total tokens written so far (not mod cap)
        return out, (k_buf, v_buf, pos_buf, write_pos)

    def forward_incremental_static(self, x_new, cos_new, sin_new, cache, cap: int, pos_method: str):
        """Decode: ONE new token (Tn=1), fixed-shape circular-buffer update -- no concatenate, no
        growing shapes, so a jax.jit wrapping this (and everything built on it) compiles once and
        is reused for every generation step, not retraced per step."""
        B, Tn, D = x_new.shape
        assert Tn == 1, "forward_incremental_static is decode-only (Tn=1)"
        q, k, v = self._qkv(x_new, B, Tn)
        k_buf, v_buf, pos_buf, write_pos = cache
        new_abs_pos = write_pos  # scalar traced int32, the new token's absolute position
        if pos_method == "rope":
            q = apply_rope(q, cos_new, sin_new)
            k = apply_rope(k, cos_new, sin_new)
        idx = write_pos % cap
        k_buf = jax.lax.dynamic_update_slice_in_dim(k_buf, k, idx, axis=2)
        v_buf = jax.lax.dynamic_update_slice_in_dim(v_buf, v, idx, axis=2)
        pos_buf = jax.lax.dynamic_update_slice_in_dim(pos_buf, new_abs_pos[None], idx, axis=0)

        query_pos = jnp.broadcast_to(new_abs_pos, (1,))
        allow = (pos_buf.reshape(1, -1) <= query_pos.reshape(-1, 1))
        allow = allow & ((query_pos.reshape(-1, 1) - pos_buf.reshape(1, -1)) < cap)
        mask = allow.reshape(1, 1, *allow.shape)

        y = sdpa_with_sink(q, k_buf, v_buf, mask, self.use_sink)
        out = self.out(y.transpose(0, 2, 1, 3).reshape(B, Tn, self.attn_dim))
        return out, (k_buf, v_buf, pos_buf, write_pos + 1)

    def forward_incremental(self, x_new, cos_new, sin_new, cache, window, pos_method: str):
        """Ported from summformer_jax/lm/model_summformer.py's Attn.forward_incremental (proven
        correct there via check_kv_cache_consistency). window=None means unbounded (cache never
        truncated) -- same causal_mask semantics as the dense path, just incremental."""
        B, Tn, D = x_new.shape
        q, k, v = self._qkv(x_new, B, Tn)
        if pos_method == "rope":
            q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        if cache is None:
            k_all, v_all, S_prev = k, v, 0
        else:
            k_prev, v_prev = cache
            k_all, v_all = jnp.concatenate([k_prev, k], axis=2), jnp.concatenate([v_prev, v], axis=2)
            S_prev = k_prev.shape[2]
        S = k_all.shape[2]
        new_pos = jnp.arange(S_prev, S_prev + Tn)
        key_pos = jnp.arange(S)
        mask = causal_mask(new_pos, key_pos, window)
        y = sdpa_with_sink(q, k_all, v_all, mask, self.use_sink)
        out = self.out(y.transpose(0, 2, 1, 3).reshape(B, Tn, self.attn_dim))
        if window is not None and S > window:
            k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
        return out, (k_all, v_all)

    def forward_windowed(self, x, cos, sin, window: int, pos_method: str) -> jnp.ndarray:
        """Real O(T*window) self-attention via chunked_windowed_attention, not dense-plus-mask."""
        B, T, D = x.shape
        q, k, v = self._qkv(x, B, T)
        if pos_method == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = chunked_windowed_attention(q, k, v, window, self.use_sink, self.use_flash)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

    def forward_cross(self, x_q, x_kv, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method: str) -> jnp.ndarray:
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x_q).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        kv = self.wkv(x_kv).reshape(B, S, 2, H, hd)
        k = kv[:, :, 0].transpose(0, 2, 1, 3)
        v = kv[:, :, 1].transpose(0, 2, 1, 3)
        if pos_method == "rope":
            q = apply_rope(q, cos_q, sin_q)
            k = apply_rope(k, cos_k, sin_k)
        y = sdpa_with_sink(q, k, v, attn_mask, self.use_sink, self.use_flash)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

    def forward_cross_windowed(self, x_q, x_kv, cos_q, sin_q, cos_k, sin_k, seq_pos, code_pos_abs,
                                stride: int, window: int, pos_method: str) -> jnp.ndarray:
        """Real O(T*window/stride) windowed cross-attention -- training-forward only (see
        windowed_cross_attention's docstring; not used by the incremental/fully-static generation
        paths, which stay on the dense forward_cross/__call__ path unchanged)."""
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x_q).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        kv = self.wkv(x_kv).reshape(B, S, 2, H, hd)
        k = kv[:, :, 0].transpose(0, 2, 1, 3)
        v = kv[:, :, 1].transpose(0, 2, 1, 3)
        if pos_method == "rope":
            q = apply_rope(q, cos_q, sin_q)
            k = apply_rope(k, cos_k, sin_k)
        y = windowed_cross_attention(q, k, v, seq_pos, code_pos_abs, stride, window, self.use_sink)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))


class MLP(nnx.Module):
    def __init__(self, d_model: int, mlp_mult: int, scale_layers: int, dtype, param_dtype, *, rngs: nnx.Rngs):
        hidden = mlp_mult * d_model
        init = nnx.initializers.normal(stddev=0.02)
        proj_init = nnx.initializers.normal(stddev=0.02 * (2 * scale_layers) ** -0.5)
        self.fc = nnx.Linear(d_model, hidden, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.proj = nnx.Linear(hidden, d_model, use_bias=True, kernel_init=proj_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.proj(jax.nn.gelu(self.fc(x), approximate=True))


class Block(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, scale_layers: int, dtype, param_dtype, *,
                 rngs: nnx.Rngs, use_sink: bool = True, use_flash: bool = False):
        self.ln1 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink, use_flash=use_flash)
        self.ln2 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.mlp = MLP(d_model, mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs)

    def __call__(self, x, cos, sin, attn_mask, pos_method: str, window: int | None = None) -> jnp.ndarray:
        """window (if given) uses the real chunked O(T*window) attention path instead of
        dense-plus-mask; attn_mask is then ignored (must be None in that case -- caller's choice,
        not decided here)."""
        if window is not None:
            x = x + self.attn.forward_windowed(self.ln1(x), cos, sin, window, pos_method)
        else:
            x = x + self.attn.forward(self.ln1(x), cos, sin, attn_mask, pos_method)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental(self, x_new, cos_new, sin_new, cache, window, pos_method: str):
        attn_out, new_cache = self.attn.forward_incremental(self.ln1(x_new), cos_new, sin_new, cache, window, pos_method)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_cache

    def prime_static_cache(self, x, cos, sin, cap: int, window, pos_method: str):
        attn_out, cache = self.attn.prime_static_cache(self.ln1(x), cos, sin, cap, window, pos_method)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, cache

    def forward_incremental_static(self, x_new, cos_new, sin_new, cache, cap: int, pos_method: str):
        attn_out, new_cache = self.attn.forward_incremental_static(self.ln1(x_new), cos_new, sin_new, cache, cap, pos_method)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_cache


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return nll.mean()




def _norm_window(w):
    return None if w in (-1, None) else w


def _auto_window(max_len: int) -> int:
    """Default auto-derive formula for window=-1: literally max_len -- the longest sequence this
    attention surface will ever see (context_len for Embedder/Decoder/Encoder.layers self-attn,
    from the DATALOADER's own seq_len, not a separately hand-set config value; max_blocks[i] for
    chain stage i's own transformer; cum_stride for a cross-attn spec). This is dense-equivalent
    (window >= T means chunked_windowed_attention's own T<=w fallback kicks in, same computation
    as literal unbounded) -- confirmed intentional (see chat 2026-08-30): -1 is a SAFE correct
    default sized to the actual known sequence length, not a compute-saving heuristic -- set an
    explicit smaller int per window when savings are wanted. force_dense=True is then a redundant
    but harmless alias for the same effective behavior (kept for semantic clarity)."""
    return max(1, max_len)


def _zero_attn_cache(Bsz: int, n_heads: int, head_dim: int, cap: int, dtype):
    """Empty fixed-shape circular KV cache, same format Attn.prime_static_cache produces --
    unwritten slots default to a sentinel position (-10**9) that causal_mask's `< window` clause
    always rejects, exactly like a freshly-primed cache with T < cap."""
    k0 = jnp.zeros((Bsz, n_heads, cap, head_dim), dtype=dtype)
    v0 = jnp.zeros((Bsz, n_heads, cap, head_dim), dtype=dtype)
    pos0 = jnp.full((cap,), -10 ** 9, dtype=jnp.int32)
    return (k0, v0, pos0, jnp.array(0, dtype=jnp.int32))


# ----------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------

@dataclass
class StackConfig:
    """Architecture-only config for one plain self-attn+MLP block stack (SharedTrunk, or an
    Encoder's/Decoder's own layers). No task fields (vocab_size, num_classes, ...) live here."""
    n_layers: int = 0
    d_model: int = 256
    n_heads: int = 4
    mlp_mult: int = 4
    window: int | tuple = -1  # -1 = auto-derive (see _auto_window); per-layer if a tuple
    force_dense: bool = False  # -1/omitted window means LITERAL unbounded instead of auto-derived
    pos_method: str = "rope"
    rope_base: float = 10000.0
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    use_sink: bool = True  # NOT safe to default False globally (see chat 2026-08-30): the zero-KV
    # sink guarantees every attention row has at least one valid key. QueryClassifierHead's cross-
    # attn is safe without it only because its query sits AFTER the full encoder sequence (every
    # code position is unconditionally <= that query position) -- a genuine causal AR decoder
    # (e.g. the LM lineage) has early positions with structurally ZERO valid cross-attn keys for a
    # given stage before that stage's first code token exists, which without the sink produces an
    # all -inf softmax row -> NaN (confirmed empirically: every position NaN'd). Set use_sink=False
    # explicitly per-config only where the topology guarantees no empty-key rows are possible.
    use_flash: bool = False  # only takes effect where use_sink=False (see sdpa_with_sink docstring)
    use_remat: bool = False  # gradient checkpointing on BlockStack's forward (see BlockStack.__call__)


@dataclass
class ChainStageConfig:
    """One pooling stage inside an Encoder, followed by its OWN small causal transformer (a
    BlockStack, same class as everything else in this file), CHAINED off the previous stage's
    output (v1-style: this stage's transformer output feeds the next stage's pooling). The first
    stage's source is the Encoder's input (post Encoder.layers, or the raw input if Encoder has no
    layers of its own)."""
    stride: int
    n_layers: int = 1
    d_model: int | None = None  # None -> trunk d_model
    n_heads: int | None = None  # None -> trunk n_heads
    window: int | tuple = -1
    force_dense: bool = False


@dataclass
class CrossAttnSpec:
    """Decoder layer `dst` (0-indexed among decoder layers) cross-attends to Encoder output
    `encoder_output`. window=-1 (default) auto-derives to that encoder output's OWN cum_stride --
    a HEURISTIC upper-bound choice (the window every query is GUARANTEED to see its nearest code
    token with), NOT a proven necessary minimum. Confirmed (2026-08-30, see
    scripts/check_connectivity.py and chat) that cross-attn window can be as small as 1 -- even the
    absolute minimum -- WITHOUT hurting receptive field, PROVIDED self-attention (in the Decoder
    and in each Encoder chain stage) has adequate window. Mechanism: a tiny cross-attn window only
    delivers real signal to whichever decoder positions happen to land exactly on a stage's code
    grid; self-attention in the FOLLOWING decoder layers then relays that to neighboring positions
    -- the grid spacing guarantees some position always aligns, so self-attention (not
    cross-attention) is the parameter that actually controls receptive field. Concretely: with
    self-attn window sized adequately (~7-10+ for an 8-stage/stride-2/L=256 config, see
    scripts/check_chain_receptive_field.py), cross-attn window=1 reached every tested position;
    with self-attn UNDERSIZED (e.g. window=stride exactly), no cross-attn window fixes it (cross
    was tested down to literal unbounded/force_dense in that regime and still failed) -- confirms
    self-attn, not cross-attn, is the load-bearing knob. force_dense bypasses auto-derivation
    entirely for literal unbounded, cheap only when that encoder output's own sequence (n_blocks)
    is short. PRACTICAL ADVICE: don't rely on -1 for either self-attn or cross-attn in real
    configs -- set both explicitly, and verify your specific (L, stride, n_stages) config with
    scripts/check_connectivity.py (exact, deterministic, no weights needed) before trusting any
    window choice; the safe minimum is NOT a simple closed-form function of stride/cum_stride, and
    marginal windows are weight-magnitude-fragile (confirmed: window=stride+1 self-attn flips
    between connected/disconnected across random seeds/init scales at the SAME architecture --
    real connectivity existed per check_connectivity.py, but signal strength was too fragile to
    trust)."""
    dst: int
    encoder_output: int
    window: int | tuple = -1
    force_dense: bool = False


# ----------------------------------------------------------------------------
# Embedder
# ----------------------------------------------------------------------------

def _resolve_windows(window_cfg, n_layers: int, max_len: int, force_dense: bool = False, label: str = "") -> list:
    raw = list(window_cfg) if isinstance(window_cfg, tuple) else [window_cfg] * n_layers
    assert len(raw) == n_layers
    out = []
    for i, w in enumerate(raw):
        if w not in (-1, None):
            out.append(w)
        elif force_dense:
            out.append(None)
        else:
            derived = _auto_window(max_len)
            tag = f" [{label}]" if label else ""
            print(f"[summformer] auto-derived self-attn window{tag} layer {i}: {derived} "
                  f"(= max_len={max_len}, dataloader seq_len; window=-1 and force_dense=False)")
            out.append(derived)
    return out


class BlockStack(nnx.Module):
    """Plain causal self-attn+MLP stack -- internal building block used by Embedder's optional
    layers, Encoder's own (non-cross-attending) layers, and each chain stage's own transformer.
    Not user-facing on its own. `max_len`: the longest sequence THIS stack will ever process --
    used only to auto-derive window=-1 entries (see _auto_window); has no effect on explicitly-set
    windows. `label`: cosmetic, shown in the auto-derive print."""
    def __init__(self, cfg: StackConfig, max_len: int, *, label: str = "", rngs: nnx.Rngs):
        self.cfg = cfg
        self.windows = _resolve_windows(cfg.window, cfg.n_layers, max_len, cfg.force_dense, label)
        self.blocks = nnx.List([
            Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, max(1, cfg.n_layers), cfg.compute_dtype,
                  cfg.param_dtype, rngs=rngs, use_sink=cfg.use_sink, use_flash=cfg.use_flash)
            for _ in range(cfg.n_layers)
        ])

    def _forward_impl(self, x: jnp.ndarray, seq_pos: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        cos, sin = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        for i, block in enumerate(self.blocks):
            w = self.windows[i]
            if w is not None:
                x = block(x, cos, sin, None, cfg.pos_method, window=w)
            else:
                mask = causal_mask(seq_pos, seq_pos, None)
                x = block(x, cos, sin, mask, cfg.pos_method)
        return x

    def __call__(self, x: jnp.ndarray, seq_pos: jnp.ndarray) -> jnp.ndarray:
        """cfg.use_remat=True (default False, opt-in) wraps the whole stack's forward in
        jax.checkpoint (via nnx.remat) -- recomputes activations in the backward pass instead of
        storing them, trading (mostly idle, see chat 2026-08-30: measured MFU ~6%) TensorCore
        cycles for HBM. Added specifically for the Embedder (processes the full raw sequence at
        small d_model -- the single largest activation tensor in the model) and Encoder chain
        stages (layers_cfg.use_remat propagates to every stage_cfg uniformly, same pattern as
        use_sink/use_flash)."""
        if self.cfg.use_remat:
            return nnx.remat(BlockStack._forward_impl)(self, x, seq_pos)
        return self._forward_impl(x, seq_pos)

    def cache_caps(self, max_len: int) -> list:
        """Fixed cache capacity per block: window if bounded, else max_len (the largest this
        stack will ever need to remember -- caller-supplied since BlockStack itself doesn't know
        its own max sequence length)."""
        return [w if w is not None else max_len for w in self.windows]

    def prime_static_cache(self, x: jnp.ndarray, seq_pos: jnp.ndarray, caps: list):
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        cos, sin = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block.prime_static_cache(x, cos, sin, caps[i], self.windows[i], cfg.pos_method)
            caches.append(cache)
        return x, caches

    def forward_incremental_static(self, x_new: jnp.ndarray, pos_scalar, caches: list, caps: list):
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        pos = jnp.broadcast_to(jnp.asarray(pos_scalar), (1,))
        cos, sin = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        new_caches = []
        for i, block in enumerate(self.blocks):
            x_new, cache = block.forward_incremental_static(x_new, cos, sin, caches[i], caps[i], cfg.pos_method)
            new_caches.append(cache)
        return x_new, new_caches


# ----------------------------------------------------------------------------
# Embedder: raw input -> continuous stream. Owns the input_map (embed table / linear / MLP /
# custom nnx.Module) AND the optional causal self-attn+MLP layers on top -- this replaces the old
# separate Embedder+SharedTrunk split. Encoder and Decoder below never see raw/discrete input,
# only the continuous (B,L,D) stream this produces.
# ----------------------------------------------------------------------------

class Embedder(nnx.Module):
    """input_map(raw_input) -> (B,L,D), then 0+ causal self-attn+MLP layers (cfg.n_layers).
    input_map defaults to an nnx.Embed lookup table (vocab_size given); pass a custom nnx.Module
    (nnx.Linear, an MLP, anything mapping raw_input -> (B,L,D)) via `input_map` for continuous or
    other-modality input instead. n_layers=0 -> pure embedding, no attention (e.g. for an Encoder
    that supplies its OWN layers separately).

    Weight sharing between an encoder and a decoder is plain Python aliasing: construct ONE
    Embedder and reuse the same instance in both places (SummFormer's `encoder_embedder=None`
    does exactly this by default) -- there is no separate "shared" flag."""
    def __init__(self, cfg: StackConfig, context_len: int, vocab_size: int | None = None,
                 input_map: nnx.Module | None = None, *, rngs: nnx.Rngs):
        assert (vocab_size is None) != (input_map is None), \
            "give exactly one of vocab_size (default nnx.Embed lookup) or input_map (custom module)"
        init = nnx.initializers.normal(stddev=0.02)
        self.context_len = context_len
        self.input_map = input_map if input_map is not None else nnx.Embed(
            vocab_size, cfg.d_model, embedding_init=init, dtype=cfg.compute_dtype, param_dtype=cfg.param_dtype, rngs=rngs)
        self.pos_method = cfg.pos_method
        self.wpe = (nnx.Embed(context_len, cfg.d_model, embedding_init=init, dtype=cfg.compute_dtype, param_dtype=cfg.param_dtype, rngs=rngs)
                    if cfg.pos_method == "learnable" else None)
        self.blocks = BlockStack(cfg, context_len, label="Embedder", rngs=rngs) if cfg.n_layers > 0 else None

    def __call__(self, raw_input: jnp.ndarray, seq_pos: jnp.ndarray) -> jnp.ndarray:
        x0 = self.input_map(raw_input)
        if self.pos_method == "learnable":
            x0 = x0 + self.wpe(seq_pos)[None]
        return self.blocks(x0, seq_pos) if self.blocks is not None else x0

    def cache_caps(self) -> list:
        return self.blocks.cache_caps(self.context_len) if self.blocks is not None else []

    def prime_static_cache(self, raw_input: jnp.ndarray, seq_pos: jnp.ndarray, caps: list):
        x0 = self.input_map(raw_input)
        if self.pos_method == "learnable":
            x0 = x0 + self.wpe(seq_pos)[None]
        if self.blocks is None:
            return x0, []
        return self.blocks.prime_static_cache(x0, seq_pos, caps)

    def forward_incremental_static(self, raw_new: jnp.ndarray, pos_scalar, caches: list, caps: list):
        x0 = self.input_map(raw_new)
        if self.pos_method == "learnable":
            x0 = x0 + self.wpe(jnp.broadcast_to(jnp.asarray(pos_scalar), (1,)))[None]
        if self.blocks is None:
            return x0, []
        return self.blocks.forward_incremental_static(x0, pos_scalar, caches, caps)


# ----------------------------------------------------------------------------
# Encoder: own layers (optional) + chained pooling producing a list of (code_h, code_pos_abs)
# ----------------------------------------------------------------------------

class Encoder(nnx.Module):
    def __init__(self, layers_cfg: StackConfig, chain: tuple, context_len: int, *,
                 output_d_model: int | None = None, rngs: nnx.Rngs):
        """output_d_model: dimension `out_proj` bridges each stage's output TO -- what the
        consuming Decoder's own d_model actually is. Defaults to layers_cfg.d_model (the encoder's
        own base dim) for backward compat, but MUST be set explicitly to the Decoder's d_model
        whenever they differ (e.g. an hourglass Encoder feeding a differently-sized Decoder) --
        confirmed a real bug otherwise (dot_general shape mismatch), see chat 2026-08-30."""
        self.layers = BlockStack(layers_cfg, context_len, label="Encoder.layers", rngs=rngs) if layers_cfg.n_layers > 0 else None
        self.chain_cfg = chain
        self.context_len = context_len
        D, param_dtype, dtype = layers_cfg.d_model, layers_cfg.param_dtype, layers_cfg.compute_dtype
        out_D = output_d_model if output_d_model is not None else D
        init = nnx.initializers.normal(stddev=0.02)

        in_projs, out_projs, transformers, ln_fs = [], [], [], []
        stage_dims, cum_strides, max_blocks = [], [], []
        cum = 1
        prev_dim = D  # first stage pools from Encoder.layers' output (or x_in), dimension D
        for stage in chain:
            sD = stage.d_model or D
            sH = stage.n_heads or layers_cfg.n_heads
            stage_dims.append(sD)
            cum *= stage.stride
            cum_strides.append(cum)
            max_blocks.append(max(1, context_len // cum))
            # in_proj bridges from the PREVIOUS stage's own dim (chaining correctly through the
            # hourglass), not always the encoder's base D -- confirmed a real bug otherwise.
            in_projs.append(
                nnx.Linear(prev_dim, sD, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                if sD != prev_dim else None)
            out_projs.append(
                nnx.Linear(sD, out_D, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                if sD != out_D else None)
            stage_cfg = StackConfig(n_layers=stage.n_layers, d_model=sD, n_heads=sH,
                                     mlp_mult=layers_cfg.mlp_mult, window=stage.window, force_dense=stage.force_dense,
                                     pos_method=layers_cfg.pos_method, rope_base=layers_cfg.rope_base,
                                     compute_dtype=dtype, param_dtype=param_dtype, use_sink=layers_cfg.use_sink,
                                     use_flash=layers_cfg.use_flash, use_remat=layers_cfg.use_remat)
            transformers.append(BlockStack(stage_cfg, max_blocks[-1], label=f"Encoder.chain[{len(transformers)}]", rngs=rngs))
            ln_fs.append(nnx.LayerNorm(sD, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs))
            prev_dim = sD
        self.stage_dims = stage_dims
        self.cum_strides = cum_strides
        self.max_blocks = max_blocks
        self.in_projs = nnx.List(in_projs)
        self.out_projs = nnx.List(out_projs)
        self.transformers = nnx.List(transformers)
        self.ln_fs = nnx.List(ln_fs)

    def __call__(self, x_in: jnp.ndarray, seq_pos: jnp.ndarray) -> list[tuple[jnp.ndarray, jnp.ndarray, int]]:
        """Returns a list of (stage_h_trunk_dim, pos_abs, cum_stride), one per chain stage, each
        already projected back to trunk d_model so Decoder's cross-attention needs no dim
        awareness. cum_stride is a static Python int (product of strides up to and including this
        stage) -- NOT re-derived from pos_abs at the call site, since pos_abs is a traced array
        under jit and int(traced_value) raises ConcretizationTypeError there (confirmed: this was
        a real bug, only surfaced once someone jitted the full model rather than running eager)."""
        h = self.layers(x_in, seq_pos) if self.layers is not None else x_in

        outputs = []
        cur_h = h  # first stage pools from Encoder.layers' output (or x_in if it has none)
        cum_stride = 1
        for i, stage in enumerate(self.chain_cfg):
            L = cur_h.shape[1]
            n_blocks = L // stage.stride
            if n_blocks < 1:
                break
            in_proj = self.in_projs[i]
            stage_h = cur_h[:, stage.stride - 1::stage.stride, :][:, :n_blocks, :]
            stage_h = in_proj(stage_h) if in_proj is not None else stage_h

            local_pos = jnp.arange(n_blocks)
            stage_h = self.ln_fs[i](self.transformers[i](stage_h, local_pos)).astype(stage_h.dtype)

            cum_stride *= stage.stride
            pos_abs = (jnp.arange(n_blocks) + 1) * cum_stride - 1

            out_proj = self.out_projs[i]
            stage_h_trunk = out_proj(stage_h) if out_proj is not None else stage_h
            outputs.append((stage_h_trunk, pos_abs, cum_stride))

            cur_h = stage_h  # CHAIN: next stage pools from THIS stage's own (downsampled) output
        return outputs

    def init_incremental_state(self, Bsz: int) -> dict:
        stage_states = []
        for i in range(len(self.chain_cfg)):
            sD, max_nb = self.stage_dims[i], self.max_blocks[i]
            pre_buf = jnp.zeros((Bsz, max_nb, sD), dtype=self.transformers[i].cfg.compute_dtype)
            stage_states.append({"blocks_done": 0, "pre_buf": pre_buf, "cache": None})
        return {"own_cache": None, "stage_states": stage_states}

    def step_incremental(self, state: dict, stage0_input_new: jnp.ndarray, own_pos: int) -> dict:
        """stage0_input_new: (B,1,D) -- new value on the Encoder's OWN input stream at position
        own_pos (0-indexed, a plain Python int -- the whole incremental generation loop runs as
        eager Python steps, not jax.lax.scan, so this is always a concrete value, never traced;
        see SummFormer.generate_kv_cache). If Encoder.layers is set, that value passes through it
        first (its own incremental self-attn cache); the result feeds stage 0's trigger check.
        cache=None on a stage/on Encoder.layers means 'never primed yet' -- primed on first use
        (a length-1 prime is bit-identical to the dense path's math for that one position, no
        separate zero-cache construction needed)."""
        own_cache, stage_states = state["own_cache"], list(state["stage_states"])
        if self.layers is not None:
            caps = self.layers.cache_caps(self.context_len)
            pos_arr = jnp.array([own_pos])
            if own_cache is None:
                h, own_cache = self.layers.prime_static_cache(stage0_input_new, pos_arr, caps)
            else:
                h, own_cache = self.layers.forward_incremental_static(stage0_input_new, own_pos, own_cache, caps)
        else:
            h = stage0_input_new

        cur_h, cur_pos = h, own_pos  # cur_pos: position of cur_h in ITS OWN (this level's) index space
        for i, stage in enumerate(self.chain_cfg):
            if (cur_pos + 1) % stage.stride != 0:
                break  # this stage (and everything chained after it) has nothing new to do
            st = dict(stage_states[i])
            old_nb = st["blocks_done"]
            in_proj = self.in_projs[i]
            stage_in = in_proj(cur_h) if in_proj is not None else cur_h  # (B,1,sD)

            caps_i = self.transformers[i].cache_caps(self.max_blocks[i])
            if st["cache"] is None:
                stage_out, new_cache = self.transformers[i].prime_static_cache(stage_in, jnp.array([old_nb]), caps_i)
            else:
                stage_out, new_cache = self.transformers[i].forward_incremental_static(stage_in, old_nb, st["cache"], caps_i)
            stage_out = self.ln_fs[i](stage_out).astype(stage_in.dtype)  # (B,1,sD)

            st["pre_buf"] = jax.lax.dynamic_update_slice_in_dim(st["pre_buf"], stage_out, old_nb, axis=1)
            st["cache"], st["blocks_done"] = new_cache, old_nb + 1
            stage_states[i] = st

            cur_h, cur_pos = stage_out, old_nb  # next stage's trigger uses THIS stage's own local index
        return {"own_cache": own_cache, "stage_states": stage_states}

    def cross_attn_kv(self, state: dict, i: int) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        """Current (possibly stale-since-last-fire) cross-attention K/V for chain output `i`,
        matching the dense forward's (stage_h_trunk, pos_abs, cum_stride) return shape. Unwritten
        slots get pos_abs=-10**9 (sentinel, same convention as _zero_attn_cache) so windowed
        cross-attn naturally excludes them via its own `< window` check; the dense (window=None)
        path additionally needs an explicit valid-mask (see DecoderLayer -- causal_mask alone
        would NOT reject a very-negative-but-technically-<=-query sentinel)."""
        st = state["stage_states"][i]
        out_proj = self.out_projs[i]
        stage_h_trunk = out_proj(st["pre_buf"]) if out_proj is not None else st["pre_buf"]
        max_nb, cum_stride = self.max_blocks[i], self.cum_strides[i]
        idx = jnp.arange(max_nb)
        pos_abs = jnp.where(idx < st["blocks_done"], (idx + 1) * cum_stride - 1, -10 ** 9)
        return stage_h_trunk, pos_abs, cum_stride


# ----------------------------------------------------------------------------
# Decoder: standard self-attn -> cross-attn -> MLP stack, cross-attn optional per layer
# ----------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
    def __init__(self, d_model, n_heads, mlp_mult, scale_layers, dtype, param_dtype, has_cross: bool,
                 *, rngs: nnx.Rngs, use_sink: bool = True, use_flash: bool = False):
        self.ln1 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.self_attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink, use_flash=use_flash)
        self.has_cross = has_cross
        if has_cross:
            self.ln_cross = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
            self.cross_attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink, use_flash=use_flash)
        self.ln2 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.mlp = MLP(d_model, mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs)

    def __call__(self, x, cos, sin, self_mask, self_window, pos_method: str,
                 cross_kv=None, cross_pos_abs=None, cross_stride=None, cross_window=None,
                 cross_cos_k=None, cross_sin_k=None, seq_pos=None):
        if self_window is not None:
            x = x + self.self_attn.forward_windowed(self.ln1(x), cos, sin, self_window, pos_method)
        else:
            x = x + self.self_attn.forward(self.ln1(x), cos, sin, self_mask, pos_method)

        if self.has_cross and cross_kv is not None:
            xn = self.ln_cross(x)
            if cross_window is not None:
                x = x + self.cross_attn.forward_cross_windowed(
                    xn, cross_kv, cos, sin, cross_cos_k, cross_sin_k, seq_pos, cross_pos_abs,
                    cross_stride, cross_window, pos_method)
            else:
                mask = causal_mask(seq_pos, cross_pos_abs, None)
                x = x + self.cross_attn.forward_cross(xn, cross_kv, cos, sin, cross_cos_k, cross_sin_k, mask, pos_method)

        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental_static(self, x_new, cos_new, sin_new, self_cache, self_cap, self_window,
                                    pos_method: str, cross_kv=None, cross_pos_abs=None, cross_stride=None,
                                    cross_window=None, cross_cos_k=None, cross_sin_k=None, query_pos=None):
        xn = self.ln1(x_new)
        if self_cache is None:
            attn_out, new_self_cache = self.self_attn.prime_static_cache(xn, cos_new, sin_new, self_cap, self_window, pos_method)
        else:
            attn_out, new_self_cache = self.self_attn.forward_incremental_static(xn, cos_new, sin_new, self_cache, self_cap, pos_method)
        x = x_new + attn_out

        if self.has_cross and cross_kv is not None:
            xn2 = self.ln_cross(x)
            if cross_window is not None:
                cross_out = self.cross_attn.forward_cross_windowed(
                    xn2, cross_kv, cos_new, sin_new, cross_cos_k, cross_sin_k, query_pos, cross_pos_abs,
                    cross_stride, cross_window, pos_method)
            else:
                # cross_pos_abs may be sentinel-padded (unwritten future slots, see
                # Encoder.cross_attn_kv) -- causal_mask alone accepts a very-negative sentinel
                # (always `<= query_pos`) since there's no window upper bound here, so AND in an
                # explicit valid mask (real positions are always >= 0).
                mask = causal_mask(query_pos, cross_pos_abs, None) & (cross_pos_abs >= 0).reshape(1, 1, 1, -1)
                cross_out = self.cross_attn.forward_cross(xn2, cross_kv, cos_new, sin_new, cross_cos_k, cross_sin_k, mask, pos_method)
            x = x + cross_out

        x = x + self.mlp(self.ln2(x))
        return x, new_self_cache


class Decoder(nnx.Module):
    def __init__(self, cfg: StackConfig, cross_specs: tuple, context_len: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.context_len = context_len
        self.windows = _resolve_windows(cfg.window, cfg.n_layers, context_len, cfg.force_dense, "Decoder")
        cross_by_layer = {s.dst: s for s in cross_specs}
        self.cross_by_layer = cross_by_layer
        self._printed_cross_windows = set()  # layer indices already auto-derive-printed (print once, not per call)
        self.layers = nnx.List([
            DecoderLayer(cfg.d_model, cfg.n_heads, cfg.mlp_mult, max(1, cfg.n_layers), cfg.compute_dtype,
                         cfg.param_dtype, has_cross=(i in cross_by_layer), rngs=rngs, use_sink=cfg.use_sink,
                         use_flash=cfg.use_flash)
            for i in range(cfg.n_layers)
        ])
        self.ln_f = nnx.LayerNorm(cfg.d_model, dtype=jnp.float32, param_dtype=cfg.param_dtype, rngs=rngs)

    def _resolve_cross_window(self, layer_idx: int, spec, cum_stride: int):
        """window=-1 (default) auto-derives to `cum_stride` (that encoder output's own compounded
        stride) -- the mathematically-minimum window for every decoder query to see at least its
        nearest code token (confirmed: window < stride gives many/most queries ZERO valid code
        keys, a real empirical failure, not theoretical). NOTE this minimum alone is NOT
        necessarily enough for good receptive field -- see CrossAttnSpec's own docstring for the
        confirmed counter-example. force_dense=True bypasses auto-derivation for literal
        unbounded. Printed once per layer (first call only), not every call."""
        w = spec.window
        if w not in (-1, None):
            return w
        if spec.force_dense:
            return None
        if layer_idx not in self._printed_cross_windows:
            print(f"[summformer] auto-derived cross-attn window (Decoder layer {layer_idx} -> "
                  f"encoder_output {spec.encoder_output}): {cum_stride} (= cum_stride, the MINIMUM "
                  f"valid value -- may not give good receptive field on its own, see CrossAttnSpec docstring)")
            self._printed_cross_windows.add(layer_idx)
        return cum_stride

    def __call__(self, x: jnp.ndarray, seq_pos: jnp.ndarray, encoder_outputs: list) -> jnp.ndarray:
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        cos, sin = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        # lazy: only materialize the O(T^2) dense mask if some layer actually needs it (all-windowed
        # configs never touch it -- computing it unconditionally was a real OOM at T=150528, ~22GB+
        # for the boolean array alone, confirmed 2026-08-30 on tpu5)
        self_mask = causal_mask(seq_pos, seq_pos, None) if any(w is None for w in self.windows) else None

        for i, layer in enumerate(self.layers):
            w = self.windows[i]
            spec = self.cross_by_layer.get(i)
            if spec is None or spec.encoder_output >= len(encoder_outputs):
                # spec.encoder_output not yet produced (e.g. sequence too short for that stage's
                # cum_stride to yield even one block) -- self-attn only for this layer, matching
                # what the dense forward would give if that stage's `n_blocks < 1` break fired.
                x = layer(x, cos, sin, self_mask, w, cfg.pos_method)
                continue
            code_h, code_pos_abs, cum_stride = encoder_outputs[spec.encoder_output]
            cross_window = self._resolve_cross_window(i, spec, cum_stride)
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
            x = layer(x, cos, sin, self_mask, w, cfg.pos_method,
                      cross_kv=code_h, cross_pos_abs=code_pos_abs, cross_stride=cum_stride,
                      cross_window=cross_window, cross_cos_k=cos_k, cross_sin_k=sin_k, seq_pos=seq_pos)
        return self.ln_f(x)

    def init_incremental_state(self, Bsz: int) -> list:
        return [None] * len(self.layers)  # per-layer self-attn cache; None = not yet primed

    def step_incremental(self, state: list, x_new: jnp.ndarray, pos: int,
                          encoder_state_provider=None) -> tuple[jnp.ndarray, list]:
        """encoder_state_provider(encoder_output_index) -> (code_h, pos_abs, cum_stride), called
        only for layers that actually cross-attend. Typically Encoder.cross_attn_kv bound to the
        live Encoder incremental state (Case A/C, growing); for a fully-precomputed/separate
        encoder (Case B) pass a plain closure returning the SAME static tensors every call --
        no Encoder incremental machinery needed there at all."""
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        pos_arr = jnp.array([pos])
        cos, sin = (rope_cos_sin_for_positions(pos_arr, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        new_state = list(state)
        for i, layer in enumerate(self.layers):
            w = self.windows[i]
            cap = w if w is not None else self.context_len
            spec = self.cross_by_layer.get(i)
            cross_kwargs = {}
            provided = encoder_state_provider(spec.encoder_output) if (spec is not None and encoder_state_provider is not None) else None
            if provided is not None:
                code_h, code_pos_abs, cum_stride = provided
                cross_window = self._resolve_cross_window(i, spec, cum_stride)
                cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
                cross_kwargs = dict(cross_kv=code_h, cross_pos_abs=code_pos_abs, cross_stride=cum_stride,
                                     cross_window=cross_window, cross_cos_k=cos_k, cross_sin_k=sin_k, query_pos=pos_arr)
            x_new, new_cache = layer.forward_incremental_static(x_new, cos, sin, state[i], cap, w, cfg.pos_method, **cross_kwargs)
            new_state[i] = new_cache
        return self.ln_f(x_new), new_state


# ----------------------------------------------------------------------------
# SummFormer: full encoder-decoder orchestration
# ----------------------------------------------------------------------------

class SummFormer(nnx.Module):
    """Composes pre-built Embedder/Encoder/Decoder instances -- caller constructs each piece and
    hands them in, rather than SummFormer building configs internally, so any combination
    (including custom Embedder input_maps/subclassed Encoder/Decoder) composes without touching
    this class.

    `encoder_embedder`: None (default) means the encoder consumes the SAME continuous stream
    `embedder` already produced for the decoder -- no separate call, no separate weights. This is
    today's SummTransformerV2 special case: `embedder` has n_layers>0 (the "shared early layers"),
    `encoder` has 0 own layers (pure chained pooling off that shared stream), `decoder` cross-attends
    to the encoder's chained outputs. For a genuinely separate encoder (own input/modality, or just
    weight-shared-but-different-input), pass a second Embedder instance -- the SAME instance again
    for weight sharing (plain aliasing), a DIFFERENT instance for fully independent weights -- and
    call with `encoder_input` set."""
    def __init__(self, embedder: Embedder, encoder: Encoder, decoder: Decoder,
                 encoder_embedder: Embedder | None = None):
        self.embedder = embedder
        self.encoder_embedder = encoder_embedder
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, decoder_input: jnp.ndarray, encoder_input: jnp.ndarray | None = None) -> jnp.ndarray:
        L = decoder_input.shape[1]
        seq_pos = jnp.arange(L)
        dec_embed = self.embedder(decoder_input, seq_pos)

        if encoder_input is not None:
            enc_embedder = self.encoder_embedder if self.encoder_embedder is not None else self.embedder
            Le = encoder_input.shape[1]
            enc_pos = jnp.arange(Le)
            enc_embed = enc_embedder(encoder_input, enc_pos)
        else:
            enc_pos = seq_pos
            enc_embed = dec_embed  # shared case: reuse the SAME continuous stream, no extra call

        encoder_outputs = self.encoder(enc_embed, enc_pos)
        return self.decoder(dec_embed, seq_pos, encoder_outputs)

    def init_incremental_state(self, Bsz: int, separate_encoder_input: jnp.ndarray | None = None) -> dict:
        """separate_encoder_input given (Case B/C -- encoder's input doesn't grow with decoder
        generation): the encoder runs ONCE, densely, right now -- its output is static for the
        rest of generation, no incremental Encoder machinery needed. None (Case A -- today's
        default, self-referential): the encoder chain grows token-by-token alongside the decoder,
        via Encoder.init_incremental_state/step_incremental."""
        dec_cache = self.decoder.init_incremental_state(Bsz)
        if separate_encoder_input is not None:
            enc_embedder = self.encoder_embedder if self.encoder_embedder is not None else self.embedder
            Le = separate_encoder_input.shape[1]
            enc_pos = jnp.arange(Le)
            enc_embed = enc_embedder(separate_encoder_input, enc_pos)
            encoder_outputs_static = self.encoder(enc_embed, enc_pos)
            return {"mode": "static", "embedder_cache": None, "decoder_cache": dec_cache,
                    "encoder_outputs_static": encoder_outputs_static}
        return {"mode": "chained", "embedder_cache": None, "decoder_cache": dec_cache,
                "encoder_state": self.encoder.init_incremental_state(Bsz)}

    def step_incremental(self, state: dict, token_new: jnp.ndarray, pos: int) -> tuple[jnp.ndarray, dict]:
        """token_new: (B,1) raw decoder input for this new position. In 'chained' mode this SAME
        value also feeds the encoder (one shared input stream, by construction). Returns
        (decoder_hidden_new (B,1,D), new_state)."""
        caps = self.embedder.cache_caps()
        if state["embedder_cache"] is None:
            dec_embed_new, new_embedder_cache = self.embedder.prime_static_cache(token_new, jnp.array([pos]), caps)
        else:
            dec_embed_new, new_embedder_cache = self.embedder.forward_incremental_static(token_new, pos, state["embedder_cache"], caps)

        if state["mode"] == "static":
            outputs = state["encoder_outputs_static"]
            provider = lambda i: outputs[i] if i < len(outputs) else None
            dec_h, new_dec_cache = self.decoder.step_incremental(state["decoder_cache"], dec_embed_new, pos, provider)
            return dec_h, {**state, "embedder_cache": new_embedder_cache, "decoder_cache": new_dec_cache}

        new_encoder_state = self.encoder.step_incremental(state["encoder_state"], dec_embed_new, pos)
        provider = lambda i: self.encoder.cross_attn_kv(new_encoder_state, i)
        dec_h, new_dec_cache = self.decoder.step_incremental(state["decoder_cache"], dec_embed_new, pos, provider)
        return dec_h, {**state, "embedder_cache": new_embedder_cache, "decoder_cache": new_dec_cache,
                       "encoder_state": new_encoder_state}


# ----------------------------------------------------------------------------
# ARHead: next-token-prediction task head on top of a SummFormer backbone
# ----------------------------------------------------------------------------

class ARHead(nnx.Module):
    """Vocab head (optionally weight-tied to the decoder embedder's input_map, if that's an
    nnx.Embed) + cross-entropy loss + both a full-recompute baseline (generate_no_cache --
    CORRECTNESS CHECK ONLY, see check_kv_cache_consistency) and the real incremental KV-cache
    decoder (generate_kv_cache -- the actual generation path)."""
    def __init__(self, model: SummFormer, vocab_size: int, weight_tie: bool = False, *, rngs: nnx.Rngs):
        self.model = model
        self.vocab_size = vocab_size
        self.weight_tie = weight_tie
        if weight_tie:
            assert isinstance(model.embedder.input_map, nnx.Embed), \
                "weight_tie requires embedder.input_map to be an nnx.Embed lookup table"
            self.head = None
        else:
            init = nnx.initializers.normal(stddev=0.02)
            D = model.decoder.cfg.d_model
            self.head = nnx.Linear(D, vocab_size, use_bias=False, kernel_init=init,
                                    dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def _head_weight(self) -> jnp.ndarray:
        return self.model.embedder.input_map.embedding.value if self.weight_tie else self.head.kernel.value.T

    def __call__(self, decoder_input: jnp.ndarray, encoder_input: jnp.ndarray | None = None) -> tuple:
        x = self.model(decoder_input, encoder_input)
        w = self._head_weight()
        logits = x[:, :-1, :] @ w.T
        targets = decoder_input[:, 1:]
        loss = cross_entropy(logits, targets)
        return loss, {"loss": loss, "bpb": loss / math.log(2)}

    def _logits_from_hidden(self, h: jnp.ndarray) -> jnp.ndarray:
        return h @ self._head_weight().T

    def generate_no_cache(self, prompt_tokens: jnp.ndarray, n_new_tokens: int,
                           key: jax.random.PRNGKey | None = None, temperature: float = 1.0) -> jnp.ndarray:
        """Full-recompute-per-step sampling -- CORRECTNESS BASELINE ONLY (see
        check_kv_cache_consistency), not the path to actually generate with."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        all_tokens = prompt_tokens
        for _ in range(n_new_tokens):
            h = self.model(all_tokens)
            logits = self._logits_from_hidden(h[:, -1, :])
            if key is None or temperature == 0:
                next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            else:
                key, sub = jax.random.split(key)
                next_token = jax.random.categorical(sub, logits / temperature, axis=-1)[:, None]
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
        return all_tokens[0]

    def generate_kv_cache(self, prompt_tokens: jnp.ndarray, n_new_tokens: int,
                           encoder_input: jnp.ndarray | None = None) -> jnp.ndarray:
        """Real incremental generation -- the actual generation path. Must match
        generate_no_cache's greedy trajectory exactly (see check_kv_cache_consistency)."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        Bsz, prompt_len = prompt_tokens.shape
        state = self.model.init_incremental_state(Bsz, separate_encoder_input=encoder_input)
        all_tokens = prompt_tokens
        h_last = None
        for pos in range(prompt_len):
            h_last, state = self.model.step_incremental(state, prompt_tokens[:, pos:pos + 1], pos)
        next_logits = self._logits_from_hidden(h_last[:, -1, :])
        for _ in range(n_new_tokens):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            h_last, state = self.model.step_incremental(state, next_token, all_tokens.shape[1] - 1)
            next_logits = self._logits_from_hidden(h_last[:, -1, :])
        return all_tokens[0]

    def check_kv_cache_consistency(self, seq_len: int, key: jax.random.PRNGKey, n_new_tokens: int = 8,
                                    encoder_input: jnp.ndarray | None = None) -> dict:
        prompt_len = max(1, seq_len - n_new_tokens)
        prompt = jax.random.randint(key, (1, prompt_len), 0, self.vocab_size)
        out_no_cache = self.generate_no_cache(prompt, n_new_tokens, key=None, temperature=0)
        out_kv = self.generate_kv_cache(prompt, n_new_tokens, encoder_input=encoder_input)
        match_rate = float(jnp.mean((out_no_cache == out_kv).astype(jnp.float32)))
        return {"match": bool(match_rate == 1.0), "match_rate": match_rate,
                "no_cache": out_no_cache, "kv_cache": out_kv}


# ----------------------------------------------------------------------------
# ClassifierHead: pooling task head on top of a SummFormer backbone
# ----------------------------------------------------------------------------

def topk_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, k: int) -> jnp.ndarray:
    topk_preds = jax.lax.top_k(logits, k)[1]
    hit = jnp.any(topk_preds == labels[:, None], axis=-1)
    return hit.astype(jnp.float32).mean()


class ClassifierHead(nnx.Module):
    """pooling: 'last' (single causal pass, last position) | 'mean' (mean-pool the causal output)
    | 'bidirectional' (forward + reversed-sequence causal passes, mean-pooled each then averaged
    -- this codebase's attention is causal-only, no true bidirectional mode; see chat)."""
    def __init__(self, model: SummFormer, num_classes: int, pooling: str = "last", *, rngs: nnx.Rngs):
        assert pooling in ("last", "mean", "bidirectional")
        self.model = model
        self.pooling = pooling
        D = model.decoder.cfg.d_model
        init = nnx.initializers.normal(stddev=0.02)
        self.head = nnx.Linear(D, num_classes, use_bias=True, kernel_init=init,
                                dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        h_fwd = self.model(token_ids)
        if self.pooling == "last":
            pooled = h_fwd[:, -1, :]
        elif self.pooling == "mean":
            pooled = h_fwd.mean(axis=1)
        else:
            h_bwd = self.model(token_ids[:, ::-1])
            pooled = (h_fwd.mean(axis=1) + h_bwd.mean(axis=1)) / 2
        return self.head(pooled)


class QueryClassifierHead(nnx.Module):
    """Alternative to ClassifierHead: only the Encoder sees the image (via its own `embedder`);
    the Decoder side is a small set of TRAINABLE query tokens (like a ViT [CLS] token / DETR
    object query) instead of image-derived embeddings -- cross-attends into the Encoder's chained
    outputs via the SAME causal Decoder/CrossAttnSpec machinery (no new attention code -- with
    only n_query positions, causal self-attention among them is trivial/harmless, and cross-attn
    to encoder outputs is unaffected by the decoder's own causal masking either way). Much cheaper
    than ClassifierHead: no T-length decoder self-attention at all -- cost is dominated by the
    Encoder + a handful of cheap n_query-length Decoder layers.

    IMPORTANT: query positions are placed AFTER the full image sequence (>= L) so causal masking
    (key_pos <= query_pos) doesn't wrongly exclude any encoder code position. This means a
    WINDOWED cross-attn spec would only see code positions within `window` of that large query
    position -- i.e. only the LAST few code tokens, not "all" of them. Use `force_dense=True` (or
    a window at least as large as the image length) on every CrossAttnSpec passed to `decoder`
    here for genuine full-coverage cross-attention -- cheap regardless, since n_query is tiny."""
    def __init__(self, embedder: Embedder, encoder: Encoder, decoder: Decoder, num_classes: int,
                 n_query: int = 1, *, rngs: nnx.Rngs):
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.n_query = n_query
        D = decoder.cfg.d_model
        init = nnx.initializers.normal(stddev=0.02)
        self.query = nnx.Embed(n_query, D, embedding_init=init, dtype=decoder.cfg.compute_dtype,
                                param_dtype=decoder.cfg.param_dtype, rngs=rngs)
        self.head = nnx.Linear(D, num_classes, use_bias=True, kernel_init=init,
                                dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        B, L = token_ids.shape
        seq_pos = jnp.arange(L)
        enc_embed = self.embedder(token_ids, seq_pos)
        encoder_outputs = self.encoder(enc_embed, seq_pos)

        q_ids = jnp.arange(self.n_query)
        q = jnp.broadcast_to(self.query(q_ids)[None], (B, self.n_query, self.query.embedding.value.shape[-1]))
        # positions AFTER the full image sequence -- causal_mask requires key_pos <= query_pos, so
        # this is what lets the queries see every encoder code position (all < L) rather than
        # being wrongly restricted by small/early query positions.
        q_pos = jnp.arange(L, L + self.n_query)
        h = self.decoder(q, q_pos, encoder_outputs)
        return self.head(h[:, -1, :])  # last query timestep, even with multiple queries
