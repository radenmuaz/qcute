"""Shared Equinox building blocks for the image_gen_cifar_jax model family: RMSNorm, SwiGLU,
GQA+RoPE attention (batched-training form and single-step KV-cached form), a residual block,
and checkpoint save/load helpers. Every run_*.py Equinox port imports from here instead of
redefining these -- kept numerically identical to the plain-dict-pytree run_*_v1.py versions
(no bias terms anywhere, same RMSNorm eps, same RoPE convention) so ports can be validated by
copying v1's array weights in and checking exact output match.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel_lib
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask_lib

_SPLASH_BLOCK = 128  # Pallas TPU lane width -- block_kv_compute must be a multiple of this.


def _splash_pad(x: jnp.ndarray, block: int) -> jnp.ndarray:
    """Pads x's seq axis (-2) to a multiple of block; causal-safe (padded kv sits past any real
    query, padded query rows get sliced off by the caller) -- needed for T<128 or non-128-multiple
    T (e.g. incremental-decode diagnostics), which splash_attention's lane-width constraint rejects."""
    T = x.shape[-2]
    pad = (-T) % block
    if pad:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, pad), (0, 0)])
    return x


def _splash_attn_kernel(n_heads: int, padded_T: int, causal: bool, window: int = None):
    """Not cached: caching a SplashAttentionKernel (holds jnp.array-converted MaskInfo, created
    during whichever trace first calls this) across separate jax traces (train vs eval, or a
    retrace) leaks a tracer from the first, now-closed trace -- confirmed 2026-09-08, all 4 TPU
    nodes crashed with UnexpectedTracerError from an lru_cache'd version of this function.

    window (chat 2026-09-15): None -- unbounded (CausalMask/FullMask, "flash" attention -- same
    Pallas kernel, just the standard dense-over-the-whole-sequence mask). int -- causal SLIDING
    WINDOW of that many positions back (LocalMask, natively supported by JAX's splash_attention
    library -- genuinely block-sparse, not a dense-then-masked matmul, so it's real O(T*window)
    compute/memory savings). For scaling the ENCODER's own self-attention to large images (e.g.
    256x256 -> 65536-position sequences at level0) where full O(T^2) unbounded attention becomes
    prohibitive."""
    if window is not None:
        mask = splash_mask_lib.MultiHeadMask(
            [splash_mask_lib.LocalMask((padded_T, padded_T), window_size=(window, 0), offset=0)
             for _ in range(n_heads)]
        )
    else:
        mask_cls = splash_mask_lib.CausalMask if causal else splash_mask_lib.FullMask
        mask = splash_mask_lib.MultiHeadMask(
            [mask_cls((padded_T, padded_T)) for _ in range(n_heads)]
        )
    block = min(_SPLASH_BLOCK, padded_T)
    block_sizes = splash_kernel_lib.BlockSizes(
        block_q=block, block_kv=block, block_kv_compute=block,
        block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
        block_q_dq=block, block_kv_dq=block,
    )  # backward blocks are required too -- training runs grad, not just forward.
    return splash_kernel_lib.make_splash_mha_single_device(mask=mask, block_sizes=block_sizes)


def splash_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, causal: bool, sm_scale: float,
                      window: int = None, sink: jnp.ndarray = None) -> jnp.ndarray:
    """q:(B,Hq,T,hd), k/v:(B,Hkv,T,hd), Hq%Hkv==0 (native GQA -- splash groups kv heads internally,
    no repeat_kv needed unlike Pallas flash_attention). Returns (B,Hq,T,hd). window: see
    _splash_attn_kernel's docstring -- None (default) is unchanged/unbounded behavior. sink
    (chat 2026-09-15): optional (Hq,) per-head attention-sink logit, splash_attention's NATIVE
    `sinks` kernel arg -- NOT batched (shared across B), so vmap must close over it rather than
    map it like q/k/v."""
    B, Hq, T, hd = q.shape
    q_p, k_p, v_p = _splash_pad(q, _SPLASH_BLOCK), _splash_pad(k, _SPLASH_BLOCK), _splash_pad(v, _SPLASH_BLOCK)
    kernel = _splash_attn_kernel(Hq, q_p.shape[-2], causal, window)
    if sink is not None:
        y = jax.vmap(lambda qq, kk, vv: kernel(qq, kk, vv, sinks=sink))(q_p * sm_scale, k_p, v_p)
    else:
        y = jax.vmap(kernel)(q_p * sm_scale, k_p, v_p)
    return y[:, :, :T, :]


class LagCrossMask(splash_mask_lib._ComputableMask):
    """code_pos=(kv_idx+1)*cum_K-1 <= query_pos=q_idx+lag_bytes -- both are affine in their raw
    array index in image_lagcodec.run_lagcodec's StackDecoder (pos_real=arange, code_pos=affine
    map of arange), so the mask needs no closure over real position arrays. real_kv_len bounds out
    padded kv rows -- unlike causal self-attn, this mask can't rely on padding being automatically
    excluded (it's not "future", just arithmetically out of range)."""

    cum_K: int
    lag_bytes: int
    real_kv_len: int

    def __init__(self, shape, cum_K: int, lag_bytes: int, real_kv_len: int, shard_count: int = 1):
        self.cum_K, self.lag_bytes, self.real_kv_len = cum_K, lag_bytes, real_kv_len

        def fn(q_ids, kv_ids):
            lag_ok = ((kv_ids + 1) * cum_K - 1) <= (q_ids + lag_bytes)
            return lag_ok & (kv_ids < real_kv_len)

        super().__init__(shape=shape, mask_function=fn, shard_count=shard_count)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.shape == other.shape and self.cum_K == other.cum_K
                and self.lag_bytes == other.lag_bytes and self.real_kv_len == other.real_kv_len)

    def __hash__(self):
        return hash((type(self), self.shape, self.cum_K, self.lag_bytes, self.real_kv_len))


def _splash_cross_kernel(n_heads: int, padded_T: int, padded_Tc: int, cum_K: int, lag_bytes: int, real_Tc: int):
    """Not cached -- see _splash_attn_kernel's docstring (same leaked-tracer hazard)."""
    mask = splash_mask_lib.MultiHeadMask(
        [LagCrossMask((padded_T, padded_Tc), cum_K, lag_bytes, real_Tc) for _ in range(n_heads)]
    )
    bq, bkv = min(_SPLASH_BLOCK, padded_T), min(_SPLASH_BLOCK, padded_Tc)
    block_sizes = splash_kernel_lib.BlockSizes(
        block_q=bq, block_kv=bkv, block_kv_compute=bkv,
        block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkv,
        block_q_dq=bq, block_kv_dq=bkv,
    )
    return splash_kernel_lib.make_splash_mha_single_device(mask=mask, block_sizes=block_sizes)


def splash_cross_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                            cum_K: int, lag_bytes: int, sm_scale: float) -> jnp.ndarray:
    """q:(B,Hq,T,hd), k/v:(B,Hkv,Tc,hd) -- rectangular splash attention with the lag-shifted cross
    mask (see LagCrossMask) instead of a materialized (B,H,T,Tc) einsum+where+softmax. Returns
    (B,Hq,T,hd)."""
    B, Hq, T, hd = q.shape
    Tc = k.shape[-2]
    q_p, k_p, v_p = _splash_pad(q, _SPLASH_BLOCK), _splash_pad(k, _SPLASH_BLOCK), _splash_pad(v, _SPLASH_BLOCK)
    kernel = _splash_cross_kernel(Hq, q_p.shape[-2], k_p.shape[-2], cum_K, lag_bytes, Tc)
    y = jax.vmap(kernel)(q_p * sm_scale, k_p, v_p)
    return y[:, :, :T, :]


def rmsnorm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x * weight


def apply_xsa(y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Exclusive Self-Attention (arXiv:2603.09078): removes the attention output's projection
    onto the query token's own value vector v (same position, GQA-repeated to match y's head
    count). The paper finds plain attention output is biased toward high cosine similarity with
    v_i -- a point-wise transform the FFN should own -- crowding out genuine contextual mixing.
    z = y - (y . v_hat) v_hat, v_hat = v / ||v||_2. y and v: (..., hd), same shape.

    chat 2026-09-13 -- eps must go INSIDE the sqrt (rsqrt(sum(v**2)+eps)), not added to the norm
    afterward (v/(norm+eps)): the latter guards the forward division but jnp.linalg.norm's own
    gradient is v/||v|| (from d(sqrt(x))/dx), a genuine 0/0 at v==0 -- NaN in the backward pass
    even though the forward value is finite. Confirmed root cause of the "zero"-init NaN-loss
    bug (2026-09-13): NaN landed only on qkv leaves, never out/q_norm/k_norm -- the tell that
    only v's own gradient path was corrupted, regardless of how q/k/v happen to be initialized.
    Any exactly-zero v vector anywhere (routine under ZerO's partial-identity zero rows) triggers
    it. rsqrt(sum(v**2)+eps) has no such singularity at v=0 (verified: grad is exactly 0 there,
    not NaN) -- do not revert to v/(norm(v)+eps), it silently reintroduces this."""
    v_hat = v * jax.lax.rsqrt(jnp.sum(v ** 2, axis=-1, keepdims=True) + 1e-8)
    return y - jnp.sum(y * v_hat, axis=-1, keepdims=True) * v_hat


class RMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float = eqx.field(static=True, default=1e-6)

    def __init__(self, dim: int):
        self.weight = jnp.ones((dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return rmsnorm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# ZerO init (Zhao et al. 2021, arXiv:2110.12661 "ZerO Initialization: Initializing Residual
# Networks with only Zeros and Ones") -- fully deterministic, no randomness. Square weight ->
# identity; contraction (out<in) -> partial identity; expansion (out>in) -> an orthonormal
# Hadamard matrix (via Sylvester's construction, scaled by 1/sqrt(m) so H H^T = I) sandwiched
# between partial identities, which is what the paper's dynamical-isometry argument needs --
# this is the standard orthonormal-Hadamard normalization, since the paper's own m-dependent
# scaling constant isn't independently reproducible from the paper text alone. The paper also
# explicitly zero-initializes the LAST layer of each residual branch so the branch starts as a
# no-op (x + 0 = x) -- ported here as `residual_out=True`.
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    m = 1
    while m < n:
        m *= 2
    return m


def _hadamard(m: int) -> jnp.ndarray:
    """Sylvester construction of an m x m Hadamard matrix -- m must be a power of 2."""
    H = jnp.array([[1.0]])
    while H.shape[0] < m:
        H = jnp.block([[H, H], [H, -H]])
    return H


def zero_init_matrix(shape: tuple) -> jnp.ndarray:
    """shape=(in_dim, out_dim), matching this file's x @ W convention."""
    p, q = shape
    if p >= q:
        return jnp.eye(p, q)
    m = _next_pow2(q)
    H = _hadamard(m) / jnp.sqrt(m)
    return H[:p, :q]


def init_matrix(key, shape: tuple, scheme: str, residual_out: bool = False, n_layers: int = None) -> jnp.ndarray:
    """scheme="llama": N(0, 0.02^2); residual_out scales by 1/sqrt(2*n_layers) (GPT-2/LLaMA
    residual-output convention). scheme="zero": ZerO init above; residual_out forces literal
    zeros regardless of shape (the paper's "zero-init the residual branch's last layer" rule)."""
    if scheme == "zero":
        return jnp.zeros(shape) if residual_out else zero_init_matrix(shape)
    assert scheme == "llama", f"unknown init_scheme {scheme!r}"
    std = 0.02 / math.sqrt(2 * n_layers) if (residual_out and n_layers) else 0.02
    return jax.random.normal(key, shape) * std


def init_vector(key, dim: int, scheme: str) -> jnp.ndarray:
    if scheme == "zero":
        return jnp.zeros((dim,))
    return jax.random.normal(key, (dim,)) * 0.02


class SwiGLU(eqx.Module):
    gate: jnp.ndarray
    up: jnp.ndarray
    down: jnp.ndarray

    def __init__(self, key, d_model: int, mlp_mult: int, n_layers: int = None, init_scheme: str = "llama"):
        """init_scheme="llama": input projections at std=0.02; residual-output projection (down)
        scaled by 1/sqrt(2*n_layers) (GPT-2/LLaMA convention). "zero": ZerO init (see above) --
        down is forced to literal zeros (residual branch starts as a no-op). n_layers=None keeps
        the old flat-0.02 llama behavior for call sites that don't pass it."""
        hidden = d_model * mlp_mult
        k1, k2, k3 = jax.random.split(key, 3)
        self.gate = init_matrix(k1, (d_model, hidden), init_scheme)
        self.up = init_matrix(k2, (d_model, hidden), init_scheme)
        self.down = init_matrix(k3, (hidden, d_model), init_scheme, residual_out=True, n_layers=n_layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jax.nn.silu(x @ self.gate) * (x @ self.up)) @ self.down


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rope_cos_sin_pos(pos, head_dim: int, base: float) -> tuple:
    """pos: scalar (single-step) or (T,) (chunked prefill) -- broadcasts either way via the
    trailing-axis expansion, giving (head_dim,) or (T,head_dim) respectively."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.asarray(pos, dtype=jnp.float32)
    freqs = pos[..., None] * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    # cos/sin are fp32 (precision); cast back to x's dtype after -- otherwise bf16 x silently
    # upcasts to fp32 via type promotion, breaking dynamic_update_slice's dtype-matching cache
    # writes and defeating the whole point of bf16 compute (confirmed 2026-09-10).
    return (x * cos[None, None] + rotate_half(x) * sin[None, None]).astype(x.dtype)


def apply_rope_single(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return (x * cos[None, :] + rotate_half(x) * sin[None, :]).astype(x.dtype)


class Attention(eqx.Module):
    qkv: jnp.ndarray
    out: jnp.ndarray
    q_norm: jnp.ndarray
    k_norm: jnp.ndarray
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    use_xsa: bool = eqx.field(static=True)
    use_qknorm: bool = eqx.field(static=True)
    window: int = eqx.field(static=True)
    sink: jnp.ndarray

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, rope_base: float, n_layers: int = None,
                 init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True, window: int = None,
                 use_sink: bool = False):
        """See SwiGLU.__init__ for the init_scheme rationale -- applies identically here, with
        `out` as the residual-output projection. use_xsa: see apply_xsa() above (arXiv:2603.09078)
        -- applied right after the attention call, before the out-projection; default off, opt-in.
        use_qknorm: per-head RMSNorm on q/k before RoPE+scores (stabilizes attention logit scale);
        can be disabled. window (chat 2026-09-15): None (default) -- unbounded causal attention,
        unchanged. int -- causal sliding window of that many positions back (splash_attention's
        native LocalMask, genuinely block-sparse) -- see splash_attention()'s docstring. Only
        __call__ (the dense/batched form) respects this; step/chunk_step (KV-cached decode) are
        unaffected -- windowing is an ENCODER-side (self.blocks, full self-attention) concern,
        not currently wired into the decoder's incremental generation path. use_sink (chat
        2026-09-15): learned per-head attention-sink logit (one scalar per query head, init 0),
        using splash_attention's NATIVE `sinks` kernel arg -- a bias folded directly into the
        softmax max/sum, NOT an extra K/V token (cheaper: no extra sequence position, no extra
        matmul work). See the gpt-oss/StreamingLLM attention-sink literature; useful alongside
        `window` so a sliding-window layer always has somewhere to route unneeded attention mass."""
        hd = d_model // n_heads
        k1, k2 = jax.random.split(key, 2)
        if init_scheme == "zero":
            # self.qkv = init_matrix(k1, (d_model, d_model + 2 * n_kv_heads * hd), scheme="llama")
            # self.out = init_matrix(k2, (d_model, d_model), "llama", residual_out=True, n_layers=n_layers)
            # q_part = jnp.zeros((d_model, d_model))
            q_part = init_matrix(k1, (d_model, d_model), init_scheme)
            k_part = init_matrix(k1, (d_model, n_kv_heads * hd), init_scheme)
            v_part = init_matrix(k1, (d_model, n_kv_heads * hd), init_scheme)
            self.qkv = jnp.concatenate([q_part, k_part, v_part], axis=-1)
            self.out = init_matrix(k2, (d_model, d_model), init_scheme, residual_out=True, n_layers=n_layers)
        else:
            self.qkv = init_matrix(k1, (d_model, d_model + 2 * n_kv_heads * hd), init_scheme)
            self.out = init_matrix(k2, (d_model, d_model), init_scheme, residual_out=True, n_layers=n_layers)
        self.q_norm = jnp.ones((hd,))
        self.k_norm = jnp.ones((hd,))
        self.n_heads, self.n_kv_heads, self.rope_base = n_heads, n_kv_heads, rope_base
        self.use_xsa = use_xsa
        self.use_qknorm = use_qknorm
        self.window = window
        self.sink = jnp.zeros((n_heads,)) if use_sink else None

    def __call__(self, x: jnp.ndarray, causal: bool = True) -> jnp.ndarray:
        """Batched training-time forward: x is (B,T,D). causal=False is full bidirectional
        attention (discrete-diffusion-style masked head). Uses the Pallas TPU splash_attention
        kernel (block-sparse, genuine O(T) memory, native GQA -- see splash_attention() above)."""
        B, T, D = x.shape
        hd = D // self.n_heads
        qkv = x @ self.qkv
        q, k, v = jnp.split(qkv, [D, D + self.n_kv_heads * hd], axis=-1)
        q = q.reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        if self.use_qknorm:
            q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin(T, hd, self.rope_base)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(hd)
        y = splash_attention(q, k, v, causal=causal, sm_scale=scale, window=self.window, sink=self.sink)  # (B,H,T,hd)
        if self.use_xsa:
            n_rep = self.n_heads // self.n_kv_heads
            v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
            y = apply_xsa(y, v_self)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out

    def step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, T_max: int) -> tuple:
        """Single-step KV-cached form: x_new is (Bc,D), cache_k/v are (Bc,n_kv_heads,T_max,hd)."""
        Bc, D = x_new.shape
        hd = D // self.n_heads
        qkv = x_new @ self.qkv
        q, k, v = jnp.split(qkv, [D, D + self.n_kv_heads * hd], axis=-1)
        q, k, v = q.reshape(Bc, self.n_heads, hd), k.reshape(Bc, self.n_kv_heads, hd), v.reshape(Bc, self.n_kv_heads, hd)
        if self.use_qknorm:
            q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin_pos(pos, hd, self.rope_base)
        q, k = apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
        cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :].astype(cache_k.dtype), (0, 0, pos, 0))
        cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :].astype(cache_v.dtype), (0, 0, pos, 0))
        n_rep = self.n_heads // self.n_kv_heads
        k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
        v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
        valid = jnp.arange(T_max) <= pos
        logits = jnp.where(valid[None, None, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bht,bhtd->bhd", attn, v_full)  # (Bc,H,hd)
        if self.use_xsa:
            v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
            y = apply_xsa(y, v_self)
        y = y.reshape(Bc, D)
        return y @ self.out, cache_k, cache_v

    def chunk_step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray,
                    pos_start, T_max: int) -> tuple:
        """Parallel-prefill form: x_new is (Bc,T,D), T new KNOWN positions [pos_start,
        pos_start+T) written into the cache in ONE batched forward pass (causal within the chunk,
        full attention back into cache[:pos_start] from earlier chunks/groups) -- for content
        that's entirely given upfront (e.g. StageLagDecoder's ctx codes), not autoregressively
        generated one token at a time via step(). Same cache layout/dtype as step(), so the two
        are freely interleaved (chat 2026-09-10, fixing reconstruct_kv_cache's lag=max slowness --
        it was calling step() once per ctx code even though none of them need generating)."""
        Bc, T, D = x_new.shape
        hd = D // self.n_heads
        qkv = x_new @ self.qkv
        q, k, v = jnp.split(qkv, [D, D + self.n_kv_heads * hd], axis=-1)
        q = q.reshape(Bc, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(Bc, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(Bc, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        if self.use_qknorm:
            q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        pos_ids = pos_start + jnp.arange(T)
        cos, sin = rope_cos_sin_pos(pos_ids, hd, self.rope_base)  # (T, hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # x:(Bc,H,T,hd), cos/sin:(T,hd)
        cache_k = jax.lax.dynamic_update_slice(cache_k, k.astype(cache_k.dtype), (0, 0, pos_start, 0))
        cache_v = jax.lax.dynamic_update_slice(cache_v, v.astype(cache_v.dtype), (0, 0, pos_start, 0))
        n_rep = self.n_heads // self.n_kv_heads
        k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
        v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k_full) * scale  # (Bc,H,T,T_max)
        valid = jnp.arange(T_max)[None, :] <= pos_ids[:, None]  # (T,T_max)
        logits = jnp.where(valid[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v_full)  # (Bc,H,T,hd)
        if self.use_xsa:
            v_self = jnp.repeat(v, n_rep, axis=1) if n_rep > 1 else v
            y = apply_xsa(y, v_self)
        y = y.transpose(0, 2, 1, 3).reshape(Bc, T, D)
        return y @ self.out, cache_k, cache_v


class Block(eqx.Module):
    norm1: RMSNorm
    attn: Attention
    norm2: RMSNorm
    mlp: SwiGLU

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float,
                 n_layers: int = None, init_scheme: str = "llama", use_xsa: bool = False, use_qknorm: bool = True,
                 window: int = None, use_sink: bool = False):
        k1, k2 = jax.random.split(key, 2)
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(k1, d_model, n_heads, n_kv_heads, rope_base, n_layers=n_layers,
                               init_scheme=init_scheme, use_xsa=use_xsa, use_qknorm=use_qknorm, window=window,
                               use_sink=use_sink)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k2, d_model, mlp_mult, n_layers=n_layers, init_scheme=init_scheme)

    def __call__(self, x: jnp.ndarray, causal: bool = True) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, T_max: int) -> tuple:
        attn_out, ck, cv = self.attn.step(self.norm1(x_new), cache_k, cache_v, pos, T_max)
        x = x_new + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, ck, cv

    def chunk_step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray,
                    pos_start, T_max: int) -> tuple:
        attn_out, ck, cv = self.attn.chunk_step(self.norm1(x_new), cache_k, cache_v, pos_start, T_max)
        x = x_new + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, ck, cv


# ---------------------------------------------------------------------------
# Checkpointing -- equinox.tree_serialise_leaves/tree_deserialise_leaves save only the array
# leaves (static fields aren't touched), so we need the *same* model structure (same static
# hyperparameters) already constructed before loading -- caller must rebuild an identically
# shaped model/optimizer-state skeleton first, exactly like eqx's own recommended pattern.
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, opt_state, step: int, epoch: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / "model.eqx", model)
    eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)
    with open(path / "meta.pkl", "wb") as f:
        pickle.dump({"step": step, "epoch": epoch}, f)


def load_checkpoint(path: Path, model_skeleton, opt_state_skeleton) -> tuple:
    model = eqx.tree_deserialise_leaves(path / "model.eqx", model_skeleton)
    opt_state = eqx.tree_deserialise_leaves(path / "opt_state.eqx", opt_state_skeleton)
    with open(path / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, opt_state, meta["step"], meta["epoch"]


# ---------------------------------------------------------------------------
# SinkGD -- "Gradient Multi-Normalization for Stateless and Scalable LLM Training"
# (arXiv:2502.06742). Drop-in optax replacement for AdamW: 2D linear-layer weight matrices get
# a STATELESS Sinkhorn row/column L2-normalization of their raw gradient instead of AdamW's
# first/second-moment EMAs (saves ~2x the parameter memory those states cost, for those
# params), scaled by a much smaller effective LR (paper default 0.05x) since the normalized
# gradient is already unit-scale; everything else (1D params -- norms/biases -- and embedding
# tables, which aren't "linear layer" weights) stays on plain AdamW. Because the Sinkhorn
# branch has no state at all, it tolerates a bigger base LR than AdamW would.
# ---------------------------------------------------------------------------

def sr_sinkhorn(g: jnp.ndarray, iterations: int = 2) -> jnp.ndarray:
    """Alternating row/column L2-normalization of a 2D gradient matrix (SR-Sinkhorn, the
    efficient variant from the paper): converges to a fixed point where every row and column
    has L2 norm sqrt(m)/sqrt(n) respectively, giving a consistent update scale independent of
    the raw gradient's magnitude -- no momentum/variance state needed to achieve that."""
    n, m = g.shape
    x = g
    for _ in range(iterations):
        row_norm = jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True)) + 1e-8
        x = jnp.sqrt(n) * x / row_norm
        col_norm = jnp.sqrt(jnp.sum(x ** 2, axis=0, keepdims=True)) + 1e-8
        x = jnp.sqrt(m) * x / col_norm
    return x


def _is_sinkhorn_param(path, leaf) -> bool:
    """2D "linear layer" weight matrices get the stateless Sinkhorn treatment; embedding/lookup
    tables (also 2D, but not a matmul weight) and everything 1D (norms) stay on AdamW."""
    name = jax.tree_util.keystr(path).lower()
    excluded = ("embed", "bootstrap")
    return getattr(leaf, "ndim", 0) == 2 and not any(k in name for k in excluded)


def sinkgd(learning_rate, linear_lr_scale: float = 0.05, sinkhorn_iters: int = 2,
           b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
    """SinkGD: sr_sinkhorn + scaled SGD-style update for 2D linear weights (stateless), plain
    optax.adamw for everything else. `learning_rate` may be a float or an optax schedule
    (step -> lr), same as optax.adamw -- both branches see the same schedule, the Sinkhorn
    branch just additionally scales by linear_lr_scale on top."""

    def sinkhorn_normalize(_):
        def init_fn(params):
            return optax.EmptyState()

        def update_fn(updates, state, params=None):
            new_updates = jax.tree_util.tree_map(lambda g: sr_sinkhorn(g, sinkhorn_iters), updates)
            return new_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    sinkhorn_branch = optax.chain(
        sinkhorn_normalize(sinkhorn_iters),
        optax.scale_by_learning_rate(learning_rate),
        optax.scale(linear_lr_scale),
    )
    adam_branch = optax.adamw(learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: "sink" if _is_sinkhorn_param(path, leaf) else "adam", params)

    return optax.multi_transform({"sink": sinkhorn_branch, "adam": adam_branch}, label_fn)


def warmup_const_schedule(peak_lr: float, warmup_steps: int):
    """Plain linear warmup then flat forever (no decay) -- pairs with SinkGD, which (being
    stateless) doesn't accumulate the momentum/variance drift that flat-LR AdamW runs into
    near a sharp minimum, so a decay phase is less necessary here."""
    def schedule(step):
        return jnp.minimum(1.0, (step + 1) / max(warmup_steps, 1)) * peak_lr
    return schedule


def make_lr_schedule(kind: str, peak_lr: float, warmup_steps: int, total_steps: int = None,
                      end_value: float = 0.0, decay_steps: int = None):
    """chat 2026-09-13 -- kind="const" (default): warmup_const_schedule above, unchanged.
    kind="cosine": linear warmup then cosine decay to end_value (default 0) over decay_steps
    (default total_steps - warmup_steps -- this phase's own epoch_count*steps_per_epoch, decay
    resets fresh each phase). decay_steps lets the min lr be reached BEFORE phase end (e.g. by a
    given epoch); optax clips its internal step count at decay_steps, so lr holds flat at
    end_value for the remainder of the phase once reached (chat 2026-09-14)."""
    if kind == "const":
        return warmup_const_schedule(peak_lr, warmup_steps)
    assert kind == "cosine", f"unknown lr_schedule {kind!r}"
    assert total_steps is not None and total_steps > warmup_steps, \
        f"cosine schedule needs total_steps > warmup_steps (this phase's epoch_count*steps_per_epoch) " \
        f"-- got total_steps={total_steps}, warmup_steps={warmup_steps}"
    ds = decay_steps if decay_steps is not None else (total_steps - warmup_steps)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=warmup_steps,
        decay_steps=ds, end_value=end_value)
