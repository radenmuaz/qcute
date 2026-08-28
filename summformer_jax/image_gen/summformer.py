"""Self-contained copy of summformer_jax/lm/model_summformer_v2.py's architecture (ConfigV2,
SummTransformerV2, FuseStageV2) plus the shared primitives it depends on from
summformer_jax/lm/model_summformer.py (Attn, MLP, Block, causal_mask, cross_entropy,
rope_cos_sin_for_positions, ROPE_PRESETS) -- inlined here rather than imported across directories
so image_gen/ has no dependency on lm/'s files (which tpu1-4 are actively training against).

Chosen base is v2, not v1: v2 is the only variant with `main_window`/`fuse_stages` levers, which
is the whole point of the block-local design this folder exists for (see chat: small trunk window
+ fuse-stage cross-attention reconnection, ballpark-mirroring Fractal Generative Models'
arxiv.org/abs/2502.17437 recursive patch grid without literal patchify -- see
image_gen/configs/image256_fractal{2,3}level.py for the worked K-derivation).

Two additions beyond a plain copy, both explicitly scoped (see chat -- "true" cross-region
BATCHED parallel decode needs the code-LM to be free-running/pre-committed rather than pooled
post-hoc from the trunk, which is a bigger redesign NOT done here):
  - check_block_locality(): a receptive-field probe. Perturbs one distant input byte and checks
    whether it moves a given trunk position's logit -- verifies the disjoint-window-plus-
    reconnection STRUCTURE actually holds for a given config, rather than assuming it from the
    K/window arithmetic alone.
  - generate_no_cache(): straightforward full-recompute-per-step autoregressive sampling
    (correctness baseline, not a speed win) -- the current architecture's code is pooled from the
    trunk's own already-materialized output, so generation here is still sequential overall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
        assert Tn == 1, "forward_incremental_static is decode-only (Tn=1); use prime_static_cache for the prompt"
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


# ----------------------------------------------------------------------------
# FuseStageV2 + ConfigV2 + SummTransformerV2 (from model_summformer_v2.py)
# ----------------------------------------------------------------------------

class FuseStageV2(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int, scale_layers: int,
                 dtype, param_dtype, *, rngs: nnx.Rngs, use_sink: bool = True):
        self.ln1 = nnx.List([nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for _ in range(n_layers)])
        self.attn = nnx.List([Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink) for _ in range(n_layers)])
        self.ln2 = nnx.List([nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for _ in range(n_layers)])
        self.mlp = nnx.List([MLP(d_model, mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs) for _ in range(n_layers)])

    def __call__(self, x, code_kv_list, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method: str) -> jnp.ndarray:
        assert len(code_kv_list) == len(self.attn)
        for l in range(len(self.attn)):
            xn = self.ln1[l](x)
            coden = self.ln1[l](code_kv_list[l])
            x = x + self.attn[l].forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method)
            x = x + self.mlp[l](self.ln2[l](x))
        return x


@dataclass
class ConfigV2:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    mlp_mult: int = 4
    pos_method: str = "rope"
    rope_base: float = 10000.0
    rope_preset: str | None = None
    context_len: int = 1024
    main_window: int | tuple | None = None
    # (insert_after, stride, window, code_n_layers, source_index) -- trunk-dim code-LM (backward compat)
    # (insert_after, stride, window, code_n_layers, source_index, code_d_model, code_n_heads) -- own dim,
    #   projected to/from trunk dim at the fuse-stage boundary (Attn/FuseStageV2 stay trunk-dim only).
    fuse_stages: tuple = ()
    input_preset: int = 8
    vocab_size: int | None = 256      # byte-level RGB by default here (not GPT2-BPE 50304 like lm/'s)
    mtp_heads: int = 1
    mtp_weight: float = 1.0
    weight_tie: bool = False
    zero_kv_sink: bool = True
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32


def _resolve_main_windows(w, n_layers: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_layers, f"main_window tuple must have length n_layers={n_layers}, got {len(w)}"
        return tuple(w)
    return (w,) * n_layers


class SummTransformerV2(nnx.Module):
    def __init__(self, cfg: ConfigV2, *, rngs: nnx.Rngs):
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        assert cfg.pos_method in ("rope", "learnable", "base")
        self.cfg = cfg
        D = cfg.d_model
        self.head_dim = D // cfg.n_heads
        V = cfg.vocab_size if cfg.vocab_size is not None else 2 ** cfg.input_preset
        self.vocab = V
        assert D % cfg.n_heads == 0
        dtype, param_dtype = cfg.compute_dtype, cfg.param_dtype

        n_fuse_layers_total = sum(spec[3] * 2 for spec in cfg.fuse_stages)
        scale_layers = cfg.n_layers + n_fuse_layers_total

        init = nnx.initializers.normal(stddev=0.02)
        self.embed = nnx.Embed(V, D, embedding_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.wpe = (
            nnx.Embed(cfg.context_len, D, embedding_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            if cfg.pos_method == "learnable" else None
        )

        self.main_blocks = nnx.List(
            [Block(D, cfg.n_heads, cfg.mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=cfg.zero_kv_sink)
             for _ in range(cfg.n_layers)])
        self.main_windows = _resolve_main_windows(cfg.main_window, cfg.n_layers)
        self.ln_f = nnx.LayerNorm(D, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        # per-stage code-LM dim: defaults to trunk dim D (5-tuple spec, backward compat) or its own
        # (larger, typically) dim from a 7-tuple spec -- code-LM runs cheaply-infrequently (once per
        # K trunk positions) so it can afford to be wider/deeper than the trunk without dominating
        # cost. Attn/FuseStageV2 themselves stay trunk-dim only; code_in_proj/code_out_proj bridge
        # the boundary so no cross-attention machinery needs to support mixed dims directly.
        code_dims = [(spec[5] if len(spec) > 5 else D) for spec in cfg.fuse_stages]
        code_n_heads_list = [(spec[6] if len(spec) > 6 else cfg.n_heads) for spec in cfg.fuse_stages]
        for cd, ch in zip(code_dims, code_n_heads_list):
            assert cd % ch == 0, f"code_d_model={cd} must be divisible by code_n_heads={ch}"
        self.code_dims = code_dims
        self.code_head_dims = [cd // ch for cd, ch in zip(code_dims, code_n_heads_list)]
        # static upper bound on how many codes a stage can ever produce -- context_len//K, known at
        # construction time (context_len is fixed per config) -- used to size the fully-static
        # incremental-decode buffers (see _make_fully_static_incremental_stepper).
        self.max_n_blocks = [cfg.context_len // spec[1] for spec in cfg.fuse_stages]

        self.fuse_stages = nnx.List([
            FuseStageV2(D, cfg.n_heads, cfg.mlp_mult, spec[3], scale_layers, dtype, param_dtype,
                        rngs=rngs, use_sink=cfg.zero_kv_sink)
            for spec in cfg.fuse_stages
        ])
        self.code_lms = nnx.List([
            nnx.List([Block(cd, ch, cfg.mlp_mult, scale_layers, dtype, param_dtype,
                             rngs=rngs, use_sink=cfg.zero_kv_sink) for _ in range(spec[3])])
            for spec, cd, ch in zip(cfg.fuse_stages, code_dims, code_n_heads_list)
        ])
        self.code_ln_fs = nnx.List([
            nnx.LayerNorm(cd, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for cd in code_dims
        ])
        # projections at the trunk<->code-LM boundary -- identity-shaped (D->D) when code_d_model==D,
        # still allocated for a uniform code path (cheap, one Linear per stage either way).
        self.code_in_proj = nnx.List([
            nnx.Linear(D, cd, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            for cd in code_dims
        ])
        self.code_out_proj = nnx.List([
            nnx.Linear(cd, D, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            for cd in code_dims
        ])
        self.insertions: dict[int, list[int]] = {}
        for i, spec in enumerate(cfg.fuse_stages):
            self.insertions.setdefault(spec[0], []).append(i)

        self.head = (
            nnx.Linear(D, V, use_bias=False, kernel_init=init, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
            if not cfg.weight_tie else None
        )
        self.weight_tie = cfg.weight_tie
        self.extra_heads = nnx.List(
            [nnx.Linear(D, V, use_bias=False, kernel_init=init, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
             for _ in range(max(0, cfg.mtp_heads - 1))])

    def _head_weight(self) -> jnp.ndarray:
        return self.embed.embedding.value if self.weight_tie else self.head.kernel.value.T  # type: ignore[union-attr]

    def _run_code_lm(self, stage_i: int, code_h, cos, sin, mask) -> list:
        """code_h/outs are at this stage's OWN code_d_model (post code_in_proj); returned list is
        projected back to trunk dim (code_out_proj) so FuseStageV2's cross-attention -- trunk-dim
        only -- needs no changes for mixed dims."""
        outs = []
        for block in self.code_lms[stage_i]:
            code_h = block(code_h, cos, sin, mask, self.cfg.pos_method)
            normed = self.code_ln_fs[stage_i](code_h)
            outs.append(self.code_out_proj[stage_i](normed))
        return outs

    def _pool_and_fuse(self, stage_i: int, x, x0, layer_hist, seq_pos, cos_b, sin_b):
        cfg = self.cfg
        hd = self.head_dim
        code_hd = self.code_head_dims[stage_i]
        insert_after, stride, window, code_n_layers, source_index = cfg.fuse_stages[stage_i][:5]
        source = x0 if source_index == 0 else (x if source_index == -1 else layer_hist[source_index])

        L = source.shape[1]
        n_blocks = L // stride
        if n_blocks < 1:
            return x
        code_h = self.code_in_proj[stage_i](source[:, stride - 1::stride, :][:, :n_blocks, :])
        code_local_pos = jnp.arange(n_blocks)
        cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, code_hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        code_mask = causal_mask(code_local_pos, code_local_pos, None)
        h_code_list = self._run_code_lm(stage_i, code_h, cos_c, sin_c, code_mask)

        code_pos_abs = (jnp.arange(n_blocks) + 1) * stride - 1
        cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        fuse_mask = causal_mask(seq_pos, code_pos_abs, window)
        return self.fuse_stages[stage_i](x, h_code_list, cos_b, sin_b, cos_k, sin_k, fuse_mask, cfg.pos_method)

    def _cascade(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        B, L = token_ids.shape
        hd = self.head_dim
        pm = cfg.pos_method

        seq_pos = jnp.arange(L)
        cos_b, sin_b = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))

        x0 = self.embed(token_ids)
        if pm == "learnable":
            x0 = x0 + self.wpe(seq_pos)[None]

        x = x0
        layer_hist = [x0]
        for stage_i in self.insertions.get(0, []):
            x = self._pool_and_fuse(stage_i, x, x0, layer_hist, seq_pos, cos_b, sin_b)

        for i, block in enumerate(self.main_blocks):
            w = self.main_windows[i]
            if w is not None:
                x = block(x, cos_b, sin_b, None, pm, window=w)
            else:
                seq_mask = causal_mask(seq_pos, seq_pos, None)
                x = block(x, cos_b, sin_b, seq_mask, pm)
            layer_hist.append(x)
            for stage_i in self.insertions.get(i + 1, []):
                x = self._pool_and_fuse(stage_i, x, x0, layer_hist, seq_pos, cos_b, sin_b)

        return self.ln_f(x)

    def __call__(self, token_ids: jnp.ndarray) -> tuple:
        cfg = self.cfg
        L = token_ids.shape[1]
        x = self._cascade(token_ids)

        w = self._head_weight()
        logits = x[:, :-1, :] @ w.T
        targets = token_ids[:, 1:]
        loss = cross_entropy(logits, targets)

        mtp_losses, mtp_accs = [], []
        for i, head_i in enumerate(self.extra_heads):
            k = i + 2
            if L <= k:
                continue
            logits_i = head_i(x[:, :-k, :])
            targets_i = token_ids[:, k:]
            mtp_losses.append(cross_entropy(logits_i, targets_i))
            mtp_accs.append((jnp.argmax(logits_i, axis=-1) == targets_i).astype(jnp.float32).mean())

        total_loss = loss
        if mtp_losses:
            total_loss = total_loss + cfg.mtp_weight * jnp.mean(jnp.stack(mtp_losses))

        metrics = {
            "loss": total_loss, "final_loss": loss, "bpb": loss / math.log(2),
            **{f"mtp{i+2}_loss": l for i, l in enumerate(mtp_losses)},
            **{f"mtp{i+2}_acc": a for i, a in enumerate(mtp_accs)},
        }
        return total_loss, metrics

    def _forward_next_token_logits(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        x = self._cascade(token_ids)
        return x[:, -1, :] @ self._head_weight().T

    def generate_no_cache(self, prompt_tokens: jnp.ndarray, n_new_tokens: int,
                           key: jax.random.PRNGKey | None = None, temperature: float = 1.0) -> jnp.ndarray:
        """Full-recompute-per-step sampling -- correctness baseline, NOT a speed win. See module
        docstring: true cross-region parallel decode needs a free-running code-LM redesign not
        done here. temperature=0 (or key=None) -> greedy argmax."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        all_tokens = prompt_tokens
        for _ in range(n_new_tokens):
            logits = self._forward_next_token_logits(all_tokens)
            if key is None or temperature == 0:
                next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            else:
                key, subkey = jax.random.split(key)
                next_token = jax.random.categorical(subkey, logits / temperature, axis=-1)[:, None]
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
        return all_tokens[0]

    def _make_incremental_stepper(self, Bsz: int):
        """Real incremental-KV-cache stepper -- O(1)-ish new work per new token (O(window) for the
        trunk's self-attention, cross-attention against a code sequence that only grows every K
        tokens). Ported/adapted from summformer_jax/lm/model_summformer.py's stepper (v1, single
        n_fuse loop) to v2's structure: arbitrary insert_after points via self.insertions, and
        mixed-dim code-LMs via code_in_proj/code_out_proj (same projections _pool_and_fuse uses).

        Correctness (not just cost) constraint: must produce EXACTLY the same values as
        generate_no_cache/_cascade -- verified via check_kv_cache_consistency below, not assumed."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim
        pm = cfg.pos_method
        n_stages = len(cfg.fuse_stages)

        main_caches = [None] * cfg.n_layers
        # per-depth accumulated hidden-state history: index 0 = x0 (raw embeddings), index j =
        # main_blocks[j-1]'s output -- same contract as _cascade's layer_hist, just persisted
        # across calls instead of rebuilt each time. Only x0 and any depth actually referenced by
        # some fuse_stages[stage_i]'s source_index get accumulated (avoids paying O(T*D) storage
        # for every unused depth).
        referenced_depths = {0}
        for spec in cfg.fuse_stages:
            src = spec[4]
            if src == -1:
                referenced_depths.add(spec[0])  # -1 means "the depth this stage fires at"
            elif src != 0:
                referenced_depths.add(src)
        hist = {j: jnp.zeros((Bsz, 0, D), dtype=cfg.compute_dtype) for j in referenced_depths}
        stage_code_cache = [None] * n_stages
        stage_n_blocks_done = [0] * n_stages

        def pool_and_fuse_incremental(stage_i: int, x_new, new_pos, cos_new, sin_new, current_depth: int):
            insert_after, stride, window, code_n_layers, source_index = cfg.fuse_stages[stage_i][:5]
            code_hd = self.code_head_dims[stage_i]
            if source_index == 0:
                source_hist = hist[0]
            elif source_index == -1:
                source_hist = hist[current_depth]
            else:
                source_hist = hist[source_index]

            n_blocks = source_hist.shape[1] // stride
            if n_blocks > stage_n_blocks_done[stage_i]:
                code_h = self.code_in_proj[stage_i](source_hist[:, stride - 1::stride, :][:, :n_blocks, :])
                code_local_pos = jnp.arange(n_blocks)
                cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, code_hd, cfg.rope_base) if pm == "rope" else (None, None))
                code_mask = causal_mask(code_local_pos, code_local_pos, None)
                stage_code_cache[stage_i] = self._run_code_lm(stage_i, code_h, cos_c, sin_c, code_mask)
                stage_n_blocks_done[stage_i] = n_blocks

            h_code_list = stage_code_cache[stage_i]
            if h_code_list is None:
                return x_new
            n_blocks_now = h_code_list[0].shape[1]
            code_pos_abs = (jnp.arange(n_blocks_now) + 1) * stride - 1
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if pm == "rope" else (None, None))
            fuse_mask = causal_mask(new_pos, code_pos_abs, window)
            return self.fuse_stages[stage_i](x_new, h_code_list, cos_new, sin_new, cos_k, sin_k, fuse_mask, pm)

        def step(token_chunk: jnp.ndarray, start_pos: int) -> jnp.ndarray:
            Tn = token_chunk.shape[1]
            pos = jnp.arange(start_pos, start_pos + Tn)
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            x0_new = self.embed(token_chunk)
            if pm == "learnable":
                x0_new = x0_new + self.wpe(pos)[None]
            if 0 in hist:
                hist[0] = jnp.concatenate([hist[0], x0_new], axis=1)

            x = x0_new
            for stage_i in self.insertions.get(0, []):
                x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=0)

            for i, block in enumerate(self.main_blocks):
                w = self.main_windows[i]
                x, main_caches[i] = block.forward_incremental(x, cos_b, sin_b, main_caches[i], w, pm)
                depth = i + 1
                if depth in hist:
                    hist[depth] = jnp.concatenate([hist[depth], x], axis=1)
                for stage_i in self.insertions.get(depth, []):
                    x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=depth)

            x = self.ln_f(x)
            return x @ self._head_weight().T

        return step

    def generate_kv_cache(self, prompt_tokens: jnp.ndarray, n_new_tokens: int) -> jnp.ndarray:
        """Real incremental KV cache (trunk self-attention AND fuse-stage cross-attention both
        incremental -- cross-attention's code side is recomputed, not cached, but only when a new
        block boundary is crossed, same as the proven lm/ design). Must match generate_no_cache's
        greedy trajectory exactly -- see check_kv_cache_consistency."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        step = self._make_incremental_stepper(prompt_tokens.shape[0])

        all_tokens = prompt_tokens
        logits_all = step(all_tokens, 0)
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_tokens):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits_all = step(next_token, all_tokens.shape[1] - 1)
            next_logits = logits_all[:, -1, :]

        return all_tokens[0]

    def check_kv_cache_consistency(self, seq_len: int, key: jax.random.PRNGKey,
                                    n_checks: int = 3, prompt_len: int = 8, n_new_tokens: int = 24) -> dict:
        """Diagnostic: generate_no_cache vs generate_kv_cache MUST produce bit-exact identical
        greedy trajectories. Should always return match_rate == 1.0."""
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            key, subkey = jax.random.split(key)
            prompt = jax.random.randint(subkey, (pl,), 0, self.vocab)
            out_full = self.generate_no_cache(prompt, n_new_tokens)
            out_cache = self.generate_kv_cache(prompt, n_new_tokens)
            if jnp.array_equal(out_full, out_cache):
                n_match += 1
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}

    def _make_static_incremental_stepper(self, Bsz: int):
        """Trunk uses a genuinely static (fixed-shape) circular KV cache per layer
        (prime_static_cache/forward_incremental_static) -- reusable across a jax.jit trace, unlike
        _make_incremental_stepper's concatenate-based cache which changes shape every call for the
        first `window` steps. The fuse-stage/code-LM side is deliberately NOT given the same
        treatment here -- see the module-level note on why (not a small fixed window; its natural
        bound is context_len//K, and its recompute trigger is a Python-level shape-changing branch,
        not a `jax.lax.cond`-gated one) -- so a decode_step call still retraces whenever a
        fuse-stage's code buffer grows, even though the trunk math itself is now fixed-shape."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim
        pm = cfg.pos_method
        caps = [w if w is not None else cfg.context_len for w in self.main_windows]

        main_caches = [None] * cfg.n_layers
        referenced_depths = {0}
        for spec in cfg.fuse_stages:
            src = spec[4]
            if src == -1:
                referenced_depths.add(spec[0])
            elif src != 0:
                referenced_depths.add(src)
        hist = {j: jnp.zeros((Bsz, 0, D), dtype=cfg.compute_dtype) for j in referenced_depths}
        n_stages = len(cfg.fuse_stages)
        stage_code_cache = [None] * n_stages
        stage_n_blocks_done = [0] * n_stages

        def pool_and_fuse_incremental(stage_i, x_new, new_pos, cos_new, sin_new, current_depth):
            insert_after, stride, window, code_n_layers, source_index = cfg.fuse_stages[stage_i][:5]
            code_hd = self.code_head_dims[stage_i]
            source_hist = hist[0] if source_index == 0 else (hist[current_depth] if source_index == -1 else hist[source_index])
            n_blocks = source_hist.shape[1] // stride
            if n_blocks > stage_n_blocks_done[stage_i]:
                code_h = self.code_in_proj[stage_i](source_hist[:, stride - 1::stride, :][:, :n_blocks, :])
                code_local_pos = jnp.arange(n_blocks)
                cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, code_hd, cfg.rope_base) if pm == "rope" else (None, None))
                code_mask = causal_mask(code_local_pos, code_local_pos, None)
                stage_code_cache[stage_i] = self._run_code_lm(stage_i, code_h, cos_c, sin_c, code_mask)
                stage_n_blocks_done[stage_i] = n_blocks
            h_code_list = stage_code_cache[stage_i]
            if h_code_list is None:
                return x_new
            n_blocks_now = h_code_list[0].shape[1]
            code_pos_abs = (jnp.arange(n_blocks_now) + 1) * stride - 1
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if pm == "rope" else (None, None))
            fuse_mask = causal_mask(new_pos, code_pos_abs, window)
            return self.fuse_stages[stage_i](x_new, h_code_list, cos_new, sin_new, cos_k, sin_k, fuse_mask, pm)

        def embed_step(token_chunk, pos):
            x0 = self.embed(token_chunk)
            if pm == "learnable":
                x0 = x0 + self.wpe(pos)[None]
            if 0 in hist:
                hist[0] = jnp.concatenate([hist[0], x0], axis=1)
            return x0

        def prime(prompt_tokens):
            Tn = prompt_tokens.shape[1]
            pos = jnp.arange(0, Tn)
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            x = embed_step(prompt_tokens, pos)
            for stage_i in self.insertions.get(0, []):
                x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=0)
            for i, block in enumerate(self.main_blocks):
                x, main_caches[i] = block.prime_static_cache(x, cos_b, sin_b, caps[i], self.main_windows[i], pm)
                depth = i + 1
                if depth in hist:
                    hist[depth] = jnp.concatenate([hist[depth], x], axis=1)
                for stage_i in self.insertions.get(depth, []):
                    x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=depth)
            return self.ln_f(x) @ self._head_weight().T

        def decode_step(token, pos_scalar):
            pos = jnp.asarray([pos_scalar])
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            x = embed_step(token, pos)
            for stage_i in self.insertions.get(0, []):
                x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=0)
            for i, block in enumerate(self.main_blocks):
                x, main_caches[i] = block.forward_incremental_static(x, cos_b, sin_b, main_caches[i], caps[i], pm)
                depth = i + 1
                if depth in hist:
                    hist[depth] = jnp.concatenate([hist[depth], x], axis=1)
                for stage_i in self.insertions.get(depth, []):
                    x = pool_and_fuse_incremental(stage_i, x, pos, cos_b, sin_b, current_depth=depth)
            return self.ln_f(x) @ self._head_weight().T

        return prime, decode_step

    def generate_kv_cache_static(self, prompt_tokens: jnp.ndarray, n_new_tokens: int) -> jnp.ndarray:
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        prime, decode_step = self._make_static_incremental_stepper(prompt_tokens.shape[0])
        all_tokens = prompt_tokens
        logits_all = prime(all_tokens)
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_tokens):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits_all = decode_step(next_token, all_tokens.shape[1] - 1)
            next_logits = logits_all[:, -1, :]
        return all_tokens[0]

    def check_kv_cache_consistency_static(self, seq_len: int, key: jax.random.PRNGKey,
                                           n_checks: int = 3, prompt_len: int = 8, n_new_tokens: int = 24) -> dict:
        """Same contract as check_kv_cache_consistency, for the static-cache path."""
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            key, subkey = jax.random.split(key)
            prompt = jax.random.randint(subkey, (pl,), 0, self.vocab)
            out_full = self.generate_no_cache(prompt, n_new_tokens)
            out_cache = self.generate_kv_cache_static(prompt, n_new_tokens)
            if jnp.array_equal(out_full, out_cache):
                n_match += 1
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}

    def _make_fully_static_incremental_stepper(self, Bsz: int):
        """Extends _make_static_incremental_stepper to the fuse-stage/code-LM side: every buffer
        (trunk KV, pooling-source history, code-LM's own KV, code-LM output) is now fixed-shape,
        and the code recompute trigger is a jax.lax.cond (both branches same-shaped), not a
        Python-level branch -- so a decode call no longer retraces when a fuse-stage's code buffer
        grows. Sized to context_len (self.max_n_blocks = context_len // K per stage), a static
        bound known at construction time -- see ConfigV2/SummTransformerV2.max_n_blocks.

        Correctness discipline unchanged from every other stepper here: must match generate_no_cache
        bit-exactly, verified via check_kv_cache_consistency_fully_static, not assumed from the
        padding/masking design being "obviously" equivalent."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim
        pm = cfg.pos_method
        caps = [w if w is not None else cfg.context_len for w in self.main_windows]
        n_stages = len(cfg.fuse_stages)

        main_caches = [None] * cfg.n_layers
        referenced_depths = {0}
        for spec in cfg.fuse_stages:
            src = spec[4]
            if src == -1:
                referenced_depths.add(spec[0])
            elif src != 0:
                referenced_depths.add(src)
        # sequential (non-circular) static history buffers -- pooling needs the FULL causal past,
        # never evicted, so sized to context_len outright (the true max, no wraparound needed).
        hist_buf = {j: jnp.zeros((Bsz, cfg.context_len, D), dtype=cfg.compute_dtype) for j in referenced_depths}
        total_written = jnp.array(0, dtype=jnp.int32)  # shared: every depth gets exactly 1 new row/call

        # jax.lax.cond needs both branches to return the SAME pytree structure from the start --
        # None (used by the dynamic stepper's cache) doesn't work here; forward_incremental_static
        # has no None-handling path, so seed proper zero-initialized (k,v,pos,write_pos) tuples.
        def _zero_code_cache(stage_i):
            cd, ch = self.code_dims[stage_i], self.code_dims[stage_i] // self.code_head_dims[stage_i]
            hd_c = self.code_head_dims[stage_i]
            max_nb = self.max_n_blocks[stage_i]
            k0 = jnp.zeros((Bsz, ch, max_nb, hd_c), dtype=self.cfg.compute_dtype)
            v0 = jnp.zeros((Bsz, ch, max_nb, hd_c), dtype=self.cfg.compute_dtype)
            pos0 = jnp.full((max_nb,), -10 ** 9, dtype=jnp.int32)
            return (k0, v0, pos0, jnp.array(0, dtype=jnp.int32))

        code_lm_caches = [[_zero_code_cache(i) for _ in range(spec[3])] for i, spec in enumerate(cfg.fuse_stages)]
        h_code_out_bufs = [
            [jnp.zeros((Bsz, self.max_n_blocks[i], D), dtype=cfg.compute_dtype) for _ in range(cfg.fuse_stages[i][3])]
            for i in range(n_stages)
        ]
        n_blocks_done = [jnp.array(0, dtype=jnp.int32) for _ in range(n_stages)]

        def pool_and_fuse_static(stage_i, x_new, new_pos, cos_new, sin_new, current_depth,
                                  total_written_now):
            insert_after, stride, window, code_n_layers, source_index = cfg.fuse_stages[stage_i][:5]
            code_hd = self.code_head_dims[stage_i]
            max_nb = self.max_n_blocks[stage_i]
            source_buf = hist_buf[0] if source_index == 0 else (hist_buf[current_depth] if source_index == -1 else hist_buf[source_index])

            def do_update(carry):
                caches_i, out_bufs_i, nb_done_i = carry
                sample_pos = (nb_done_i + 1) * stride - 1
                sample = jax.lax.dynamic_slice_in_dim(source_buf, sample_pos, 1, axis=1)
                h = self.code_in_proj[stage_i](sample)
                code_local_pos = nb_done_i[None]
                cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, code_hd, cfg.rope_base) if pm == "rope" else (None, None))
                new_caches, new_out_bufs = [], []
                for l, block in enumerate(self.code_lms[stage_i]):
                    h, new_cache = block.forward_incremental_static(h, cos_c, sin_c, caches_i[l], max_nb, pm)
                    new_caches.append(new_cache)
                    projected = self.code_out_proj[stage_i](self.code_ln_fs[stage_i](h))
                    new_out_bufs.append(jax.lax.dynamic_update_slice_in_dim(out_bufs_i[l], projected, nb_done_i, axis=1))
                return new_caches, new_out_bufs, nb_done_i + 1

            def no_update(carry):
                return carry

            need_update = (total_written_now // stride) > n_blocks_done[stage_i]
            new_caches, new_out_bufs, new_nb_done = jax.lax.cond(
                need_update, do_update, no_update,
                (code_lm_caches[stage_i], h_code_out_bufs[stage_i], n_blocks_done[stage_i]),
            )
            code_lm_caches[stage_i] = new_caches
            h_code_out_bufs[stage_i] = new_out_bufs
            n_blocks_done[stage_i] = new_nb_done

            code_pos_abs = (jnp.arange(max_nb) + 1) * stride - 1
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if pm == "rope" else (None, None))
            valid = (jnp.arange(max_nb) < n_blocks_done[stage_i]).reshape(1, 1, 1, max_nb)
            fuse_mask = causal_mask(new_pos, code_pos_abs, window) & valid
            # dense _pool_and_fuse SKIPS the fuse-stage entirely (x unchanged) when no codes exist
            # yet (n_blocks<1) -- relying on the sink alone to avoid NaN under an all-invalid mask
            # is NOT equivalent: FuseStageV2's residual MLP still applies even when attention output
            # is "just sink", silently diverging from dense's true no-op. Match dense exactly via
            # lax.cond (both branches same-shape: (B,Tn,D)).
            return jax.lax.cond(
                n_blocks_done[stage_i] > 0,
                lambda: self.fuse_stages[stage_i](x_new, h_code_out_bufs[stage_i], cos_new, sin_new, cos_k, sin_k, fuse_mask, pm),
                lambda: x_new,
            )

        def embed_and_hist(token_chunk, pos, write_idx):
            x0 = self.embed(token_chunk)
            if pm == "learnable":
                x0 = x0 + self.wpe(pos)[None]
            if 0 in hist_buf:
                hist_buf[0] = jax.lax.dynamic_update_slice_in_dim(hist_buf[0], x0, write_idx, axis=1)
            return x0

        def prime(prompt_tokens):
            nonlocal total_written
            Tn = prompt_tokens.shape[1]
            pos = jnp.arange(0, Tn)
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            x = embed_and_hist(prompt_tokens, pos, 0)
            total_written = jnp.array(Tn, dtype=jnp.int32)
            for stage_i in self.insertions.get(0, []):
                x = pool_and_fuse_static(stage_i, x, pos, cos_b, sin_b, 0, total_written)
            for i, block in enumerate(self.main_blocks):
                x, main_caches[i] = block.prime_static_cache(x, cos_b, sin_b, caps[i], self.main_windows[i], pm)
                depth = i + 1
                if depth in hist_buf:
                    hist_buf[depth] = jax.lax.dynamic_update_slice_in_dim(hist_buf[depth], x, 0, axis=1)
                for stage_i in self.insertions.get(depth, []):
                    x = pool_and_fuse_static(stage_i, x, pos, cos_b, sin_b, depth, total_written)
            return self.ln_f(x) @ self._head_weight().T

        def decode_step(token, pos_scalar):
            nonlocal total_written
            pos = jnp.asarray([pos_scalar])
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            x = embed_and_hist(token, pos, total_written)
            total_written = total_written + 1
            for stage_i in self.insertions.get(0, []):
                x = pool_and_fuse_static(stage_i, x, pos, cos_b, sin_b, 0, total_written)
            for i, block in enumerate(self.main_blocks):
                x, main_caches[i] = block.forward_incremental_static(x, cos_b, sin_b, main_caches[i], caps[i], pm)
                depth = i + 1
                if depth in hist_buf:
                    hist_buf[depth] = jax.lax.dynamic_update_slice_in_dim(hist_buf[depth], x, pos_scalar, axis=1)
                for stage_i in self.insertions.get(depth, []):
                    x = pool_and_fuse_static(stage_i, x, pos, cos_b, sin_b, depth, total_written)
            return self.ln_f(x) @ self._head_weight().T

        return prime, decode_step

    def generate_kv_cache_fully_static(self, prompt_tokens: jnp.ndarray, n_new_tokens: int,
                                        key: jax.random.PRNGKey | None = None, temperature: float = 1.0) -> jnp.ndarray:
        """key=None (default) -> greedy argmax, matching every other generate_* method's default
        and preserving check_kv_cache_consistency_fully_static's bit-exact-vs-generate_no_cache
        guarantee (that check always uses greedy). Pass a key for true stochastic samples -- e.g.
        train.py's eval-time sample generation should use this, not the greedy default, since a
        genuine sample draw is the point of a qualitative eval image, not the single most-likely
        (and often repetitive/degenerate, especially early in training) continuation."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        prime, decode_step = self._make_fully_static_incremental_stepper(prompt_tokens.shape[0])
        all_tokens = prompt_tokens
        logits_all = prime(all_tokens)
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_tokens):
            if key is None:
                next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            else:
                key, subkey = jax.random.split(key)
                next_token = jax.random.categorical(subkey, next_logits / temperature, axis=-1)[:, None]
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits_all = decode_step(next_token, all_tokens.shape[1] - 1)
            next_logits = logits_all[:, -1, :]
        return all_tokens[0]

    def check_kv_cache_consistency_fully_static(self, seq_len: int, key: jax.random.PRNGKey,
                                                 n_checks: int = 3, prompt_len: int = 8, n_new_tokens: int = 24) -> dict:
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            key, subkey = jax.random.split(key)
            prompt = jax.random.randint(subkey, (pl,), 0, self.vocab)
            out_full = self.generate_no_cache(prompt, n_new_tokens)
            out_cache = self.generate_kv_cache_fully_static(prompt, n_new_tokens)
            if jnp.array_equal(out_full, out_cache):
                n_match += 1
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}


def check_block_locality(model: SummTransformerV2, rngs: nnx.Rngs, seq_len: int,
                          query_pos: int, probe_pos: int) -> dict:
    """Receptive-field probe: perturbs token at `probe_pos`, checks whether logits at
    `query_pos` change. Returns whether they moved, plus whether that's structurally EXPECTED
    given main_window/fuse_stages (probe_pos reachable directly within window, or indirectly via
    a fuse-stage whose insert_after/K makes it visible by query_pos's position in the cascade) --
    doesn't itself decide expected/actual, just reports the observed delta for you to compare
    against your own reading of the config."""
    V = model.vocab
    key = jax.random.PRNGKey(0)
    tokens = jax.random.randint(key, (1, seq_len), 0, V)

    logits_a = model._forward_next_token_logits(tokens[:, :query_pos + 1])
    tokens_b = tokens.at[0, probe_pos].set((tokens[0, probe_pos] + 1) % V)
    logits_b = model._forward_next_token_logits(tokens_b[:, :query_pos + 1])

    max_abs_diff = float(jnp.abs(logits_a - logits_b).max())
    return {
        "query_pos": query_pos, "probe_pos": probe_pos,
        "affected": max_abs_diff > 1e-5, "max_abs_diff": max_abs_diff,
    }
