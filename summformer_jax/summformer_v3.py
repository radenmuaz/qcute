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
                    use_sink: bool = True) -> jnp.ndarray:
    scale = 1.0 / math.sqrt(k.shape[-1])
    if not use_sink:
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
                                use_sink: bool = True) -> jnp.ndarray:
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
        return sdpa_with_sink(q, k, v, mask, use_sink)
    if T % w != 0:
        mask = causal_mask(jnp.arange(T), jnp.arange(T), w)
        return sdpa_with_sink(q, k, v, mask, use_sink)

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

    y = sdpa_with_sink(qb_flat, k_win_flat, v_win_flat, mask_flat, use_sink)
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
                 rngs: nnx.Rngs, use_sink: bool = True):
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.attn_dim = n_heads * self.head_dim
        self.use_sink = use_sink
        init = nnx.initializers.normal(stddev=0.02)
        out_init = nnx.initializers.normal(stddev=0.02 * (2 * scale_layers) ** -0.5)
        self.wq = nnx.Linear(d_model, self.attn_dim, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.wk = nnx.Linear(d_model, self.attn_dim, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.wv = nnx.Linear(d_model, self.attn_dim, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.out = nnx.Linear(self.attn_dim, d_model, use_bias=True, kernel_init=out_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def _qkv(self, x, B, T):
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        return q, k, v

    def forward(self, x, cos, sin, attn_mask, pos_method: str) -> jnp.ndarray:
        B, T, D = x.shape
        q, k, v = self._qkv(x, B, T)
        if pos_method == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = sdpa_with_sink(q, k, v, attn_mask, self.use_sink)
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
        y = chunked_windowed_attention(q, k, v, window, self.use_sink)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

    def forward_cross(self, x_q, x_kv, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method: str) -> jnp.ndarray:
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x_q).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = self.wk(x_kv).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        if pos_method == "rope":
            q = apply_rope(q, cos_q, sin_q)
            k = apply_rope(k, cos_k, sin_k)
        y = sdpa_with_sink(q, k, v, attn_mask, self.use_sink)
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
        k = self.wk(x_kv).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
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
                 rngs: nnx.Rngs, use_sink: bool = True):
        self.ln1 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink)
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
    window: int | tuple = -1  # -1 = unbounded; per-layer if a tuple, same convention as before
    pos_method: str = "rope"
    rope_base: float = 10000.0
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    use_sink: bool = True


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


@dataclass
class CrossAttnSpec:
    """Decoder layer `dst` (0-indexed among decoder layers) cross-attends to Encoder output
    `encoder_output`. window=-1/None -> dense (unbounded, only cheap when that encoder output's
    sequence is short -- see chat's stage-3-dense-vs-windowed tradeoff)."""
    dst: int
    encoder_output: int
    window: int | tuple = -1


# ----------------------------------------------------------------------------
# Embedder
# ----------------------------------------------------------------------------

def _resolve_windows(window_cfg, n_layers: int) -> list:
    if isinstance(window_cfg, tuple):
        assert len(window_cfg) == n_layers
        return [_norm_window(w) for w in window_cfg]
    return [_norm_window(window_cfg)] * n_layers


class BlockStack(nnx.Module):
    """Plain causal self-attn+MLP stack -- internal building block used by Embedder's optional
    layers, and by Encoder's own (non-cross-attending) layers. Not user-facing on its own. window
    follows the same -1=unbounded / tuple-per-layer convention as the old ConfigV2.main_window."""
    def __init__(self, cfg: StackConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.windows = _resolve_windows(cfg.window, cfg.n_layers)
        self.blocks = nnx.List([
            Block(cfg.d_model, cfg.n_heads, cfg.mlp_mult, max(1, cfg.n_layers), cfg.compute_dtype,
                  cfg.param_dtype, rngs=rngs, use_sink=cfg.use_sink)
            for _ in range(cfg.n_layers)
        ])

    def __call__(self, x: jnp.ndarray, seq_pos: jnp.ndarray) -> jnp.ndarray:
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
        self.input_map = input_map if input_map is not None else nnx.Embed(
            vocab_size, cfg.d_model, embedding_init=init, param_dtype=cfg.param_dtype, rngs=rngs)
        self.pos_method = cfg.pos_method
        self.wpe = (nnx.Embed(context_len, cfg.d_model, embedding_init=init, param_dtype=cfg.param_dtype, rngs=rngs)
                    if cfg.pos_method == "learnable" else None)
        self.blocks = BlockStack(cfg, rngs=rngs) if cfg.n_layers > 0 else None

    def __call__(self, raw_input: jnp.ndarray, seq_pos: jnp.ndarray) -> jnp.ndarray:
        x0 = self.input_map(raw_input)
        if self.pos_method == "learnable":
            x0 = x0 + self.wpe(seq_pos)[None]
        return self.blocks(x0, seq_pos) if self.blocks is not None else x0


# ----------------------------------------------------------------------------
# Encoder: own layers (optional) + chained pooling producing a list of (code_h, code_pos_abs)
# ----------------------------------------------------------------------------

class Encoder(nnx.Module):
    def __init__(self, layers_cfg: StackConfig, chain: tuple, *, rngs: nnx.Rngs):
        self.layers = BlockStack(layers_cfg, rngs=rngs) if layers_cfg.n_layers > 0 else None
        self.chain_cfg = chain
        D, param_dtype, dtype = layers_cfg.d_model, layers_cfg.param_dtype, layers_cfg.compute_dtype
        init = nnx.initializers.normal(stddev=0.02)

        in_projs, out_projs, transformers, ln_fs = [], [], [], []
        for stage in chain:
            sD = stage.d_model or D
            sH = stage.n_heads or layers_cfg.n_heads
            in_projs.append(
                nnx.Linear(D, sD, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                if sD != D else None)
            out_projs.append(
                nnx.Linear(sD, D, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
                if sD != D else None)
            stage_cfg = StackConfig(n_layers=stage.n_layers, d_model=sD, n_heads=sH,
                                     mlp_mult=layers_cfg.mlp_mult, window=stage.window,
                                     pos_method=layers_cfg.pos_method, rope_base=layers_cfg.rope_base,
                                     compute_dtype=dtype, param_dtype=param_dtype, use_sink=layers_cfg.use_sink)
            transformers.append(BlockStack(stage_cfg, rngs=rngs))
            ln_fs.append(nnx.LayerNorm(sD, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs))
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
            stage_h = self.ln_fs[i](self.transformers[i](stage_h, local_pos))

            cum_stride *= stage.stride
            pos_abs = (jnp.arange(n_blocks) + 1) * cum_stride - 1

            out_proj = self.out_projs[i]
            stage_h_trunk = out_proj(stage_h) if out_proj is not None else stage_h
            outputs.append((stage_h_trunk, pos_abs, cum_stride))

            cur_h = stage_h  # CHAIN: next stage pools from THIS stage's own (downsampled) output
        return outputs


# ----------------------------------------------------------------------------
# Decoder: standard self-attn -> cross-attn -> MLP stack, cross-attn optional per layer
# ----------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
    def __init__(self, d_model, n_heads, mlp_mult, scale_layers, dtype, param_dtype, has_cross: bool,
                 *, rngs: nnx.Rngs, use_sink: bool = True):
        self.ln1 = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.self_attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink)
        self.has_cross = has_cross
        if has_cross:
            self.ln_cross = nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
            self.cross_attn = Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink)
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


class Decoder(nnx.Module):
    def __init__(self, cfg: StackConfig, cross_specs: tuple, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.windows = _resolve_windows(cfg.window, cfg.n_layers)
        cross_by_layer = {s.dst: s for s in cross_specs}
        self.cross_by_layer = cross_by_layer
        self.layers = nnx.List([
            DecoderLayer(cfg.d_model, cfg.n_heads, cfg.mlp_mult, max(1, cfg.n_layers), cfg.compute_dtype,
                         cfg.param_dtype, has_cross=(i in cross_by_layer), rngs=rngs, use_sink=cfg.use_sink)
            for i in range(cfg.n_layers)
        ])
        self.ln_f = nnx.LayerNorm(cfg.d_model, dtype=jnp.float32, param_dtype=cfg.param_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, seq_pos: jnp.ndarray, encoder_outputs: list) -> jnp.ndarray:
        cfg = self.cfg
        hd = cfg.d_model // cfg.n_heads
        cos, sin = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        self_mask = causal_mask(seq_pos, seq_pos, None)

        for i, layer in enumerate(self.layers):
            w = self.windows[i]
            spec = self.cross_by_layer.get(i)
            if spec is None:
                x = layer(x, cos, sin, self_mask, w, cfg.pos_method)
                continue
            code_h, code_pos_abs, cum_stride = encoder_outputs[spec.encoder_output]
            cross_window = _norm_window(spec.window)
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
            x = layer(x, cos, sin, self_mask, w, cfg.pos_method,
                      cross_kv=code_h, cross_pos_abs=code_pos_abs, cross_stride=cum_stride,
                      cross_window=cross_window, cross_cos_k=cos_k, cross_sin_k=sin_k, seq_pos=seq_pos)
        return self.ln_f(x)


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
