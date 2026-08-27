"""JAX/Flax NNX port of qcute/summformer/summformer.py, made GPT2-like (LayerNorm, plain GELU
MLP, plain MHA -- no RMSNorm/SwiGLU/GQA/QK-norm) and extended to 3 pos_methods matching
gpt2_jax/model_gpt.py's convention: "rope", "learnable" (GPT2-style absolute position embedding,
added once at the byte-embedding step), "base" (NoPE, no positional signal anywhere).

Kept unchanged from the original (architecture-level): the Ks-tuple hierarchical-summarization
cascade, the zero-KV-sink attention primitive, FuseStage cross-attention + shared-weight
refinement pass, mtp_heads, the causality argument (see the original's own docstring), and the
real incremental-KV-cache generation path (ported from qcute_zero's stepper via summformer.py).

For "learnable"/"base" pos_method, RoPE application inside every Attn call becomes a no-op;
"learnable" instead adds a GPT2-style position embedding once when the byte stream is formed
(x0 = embed(byte_ids) + wpe(byte_pos)) -- pooled/code streams and cross-attn carry positional
information implicitly through that already-embedded byte-level h, not via a second lookup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


# ----------------------------------------------------------------------------
# RoPE + attention primitives
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
    # x: [B, H, T, hd]. cos/sin: [T, hd] (shared positions) or [B, T, hd] (per-batch positions).
    if cos.ndim == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


def sdpa_with_sink(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, attn_mask: jnp.ndarray) -> jnp.ndarray:
    """Mandatory zero-value/zero-key sink -- every query row keeps >=1 valid key, so a sink-only
    row (nothing else visible) is a provably clean zero instead of a NaN/uniform-softmax edge case."""
    B, H, T, hd = q.shape
    sink_k = jnp.zeros((B, H, 1, hd), dtype=k.dtype)
    sink_v = jnp.zeros((B, H, 1, hd), dtype=v.dtype)
    k2 = jnp.concatenate([sink_k, k], axis=2)
    v2 = jnp.concatenate([sink_v, v], axis=2)
    sink_col = jnp.ones(attn_mask.shape[:-1] + (1,), dtype=bool)
    mask2 = jnp.concatenate([sink_col, attn_mask], axis=-1)

    scale = 1.0 / math.sqrt(hd)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k2) * scale
    scores = jnp.where(mask2, scores, -jnp.inf)
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v2.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v2)


def causal_mask(query_pos: jnp.ndarray, key_pos: jnp.ndarray, window) -> jnp.ndarray:
    allow = key_pos.reshape(1, -1) <= query_pos.reshape(-1, 1)
    if window is not None:
        allow = allow & ((query_pos.reshape(-1, 1) - key_pos.reshape(1, -1)) < window)
    return allow.reshape(1, 1, *allow.shape)


def resolve_fuse_window(w, n_fuse: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_fuse
        return tuple(w)
    return (w,) * n_fuse


class Attn(nnx.Module):
    """Plain multi-head attention (no GQA, no QK-norm) -- GPT2-style."""

    def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs):
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.attn_dim = n_heads * self.head_dim
        init = nnx.initializers.normal(stddev=0.02)
        self.wq = nnx.Linear(d_model, self.attn_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.wk = nnx.Linear(d_model, self.attn_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.wv = nnx.Linear(d_model, self.attn_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.out = nnx.Linear(self.attn_dim, d_model, use_bias=False, kernel_init=init, rngs=rngs)

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
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))

    def forward_incremental(self, x_new, cos_new, sin_new, cache, window, pos_method: str):
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
        y = sdpa_with_sink(q, k_all, v_all, mask)
        out = self.out(y.transpose(0, 2, 1, 3).reshape(B, Tn, self.attn_dim))
        if window is not None and S > window:
            k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
        return out, (k_all, v_all)

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
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim))


class MLP(nnx.Module):
    """Plain GPT2-style MLP: fc (d -> mlp_mult*d) + GELU + proj (mlp_mult*d -> d)."""

    def __init__(self, d_model: int, mlp_mult: int, *, rngs: nnx.Rngs):
        hidden = mlp_mult * d_model
        init = nnx.initializers.normal(stddev=0.02)
        self.fc = nnx.Linear(d_model, hidden, use_bias=True, kernel_init=init, rngs=rngs)
        self.proj = nnx.Linear(hidden, d_model, use_bias=True, kernel_init=init, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.proj(jax.nn.gelu(self.fc(x), approximate=True))


class Block(nnx.Module):
    """Self-attention + MLP, pre-LayerNorm (GPT2-style). Shared (same weights) across the
    byte-level pass and the post-cross-attn refinement pass, same as the original."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = Attn(d_model, n_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp = MLP(d_model, mlp_mult, rngs=rngs)

    def __call__(self, x, cos, sin, attn_mask, pos_method: str) -> jnp.ndarray:
        x = x + self.attn.forward(self.ln1(x), cos, sin, attn_mask, pos_method)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental(self, x_new, cos_new, sin_new, cache, window, pos_method: str):
        attn_out, new_cache = self.attn.forward_incremental(self.ln1(x_new), cos_new, sin_new, cache, window, pos_method)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_cache


class FuseStage(nnx.Module):
    """Cross-attention + MLP, one instance per periodic-fusion stage, own weights throughout."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int, *, rngs: nnx.Rngs):
        self.ln1 = nnx.List([nnx.LayerNorm(d_model, rngs=rngs) for _ in range(n_layers)])
        self.attn = nnx.List([Attn(d_model, n_heads, rngs=rngs) for _ in range(n_layers)])
        self.ln2 = nnx.List([nnx.LayerNorm(d_model, rngs=rngs) for _ in range(n_layers)])
        self.mlp = nnx.List([MLP(d_model, mlp_mult, rngs=rngs) for _ in range(n_layers)])
        self.ln_out = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, code_kv, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method: str) -> jnp.ndarray:
        for l in range(len(self.attn)):
            xn = self.ln1[l](x)
            coden = self.ln1[l](code_kv)
            x = x + self.attn[l].forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method)
            x = x + self.mlp[l](self.ln2[l](x))
        return x

    def readout(self, x: jnp.ndarray, embed_weight: jnp.ndarray) -> jnp.ndarray:
        return self.ln_out(x) @ embed_weight.T


# ----------------------------------------------------------------------------
# Config + model
# ----------------------------------------------------------------------------

@dataclass
class Config:
    Ks: tuple = (32, 32, 1)          # cumulative periods; last entry unused (kept for tuple-length convention)
    d_model: int = 1024
    n_layers: int = 2                 # shared "block regular", reused for every level
    fuse_n_layers: int | None = None  # defaults to n_layers if unset
    n_heads: int = 16
    mlp_mult: int = 4
    pos_method: str = "rope"          # "rope" | "learnable" | "base" (NoPE)
    rope_base: float = 10000.0
    rope_preset: str | None = None    # "llama2"/"llama3"/"qwen3" overrides rope_base
    context_len: int = 1024
    attn_window: int | None = None    # None = max (unbounded)
    fuse_window: int | tuple | None = None
    input_preset: int = 8             # byte alphabet bits -- vocab = 2**input_preset, used only if vocab_size is unset
    vocab_size: int | None = 50304    # GPT2-BPE vocab (50257, padded to 50304), matching gpt2_jax's own Model exactly;
                                       # None falls back to the byte alphabet (2**input_preset)
    mtp_heads: int = 1                # extra byte-ahead heads (1 = disabled)
    mtp_weight: float = 1.0
    weight_tie: bool = False
    share_lm: bool = False
    share_fuse: bool = False


class SummTransformer(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        assert cfg.pos_method in ("rope", "learnable", "base")
        self.cfg = cfg
        D = cfg.d_model
        self.head_dim = D // cfg.n_heads
        V = cfg.vocab_size if cfg.vocab_size is not None else 2 ** cfg.input_preset
        self.vocab = V
        self.n_fuse = len(cfg.Ks) - 1
        assert D % cfg.n_heads == 0

        init = nnx.initializers.normal(stddev=0.02)
        self.embed = nnx.Embed(V, D, embedding_init=init, rngs=rngs)
        self.wpe = nnx.Embed(cfg.context_len, D, embedding_init=init, rngs=rngs) if cfg.pos_method == "learnable" else None

        n_lms = self.n_fuse + 1
        if cfg.share_lm:
            first = nnx.List([Block(D, cfg.n_heads, cfg.mlp_mult, rngs=rngs) for _ in range(cfg.n_layers)])
            self.lms = nnx.List([first for _ in range(n_lms)])
            first_ln = nnx.LayerNorm(D, rngs=rngs)
            self.ln_fs = nnx.List([first_ln for _ in range(n_lms)])
        else:
            self.lms = nnx.List(
                [nnx.List([Block(D, cfg.n_heads, cfg.mlp_mult, rngs=rngs) for _ in range(cfg.n_layers)])
                 for _ in range(n_lms)])
            self.ln_fs = nnx.List([nnx.LayerNorm(D, rngs=rngs) for _ in range(n_lms)])

        self.head = nnx.Linear(D, V, use_bias=False, kernel_init=init, rngs=rngs)
        self.weight_tie = cfg.weight_tie

        fuse_layers = cfg.fuse_n_layers if cfg.fuse_n_layers is not None else cfg.n_layers
        if cfg.share_fuse:
            first_fs = FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, rngs=rngs)
            self.fuse_stages = nnx.List([first_fs for _ in range(self.n_fuse)])
        else:
            self.fuse_stages = nnx.List(
                [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, rngs=rngs) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        self.extra_heads = nnx.List(
            [nnx.Linear(D, V, use_bias=False, kernel_init=init, rngs=rngs) for _ in range(max(0, cfg.mtp_heads - 1))])

    def _head_weight(self) -> jnp.ndarray:
        return self.embed.embedding.value if self.weight_tie else self.head.kernel.value.T

    def _readout(self, x: jnp.ndarray) -> jnp.ndarray:
        w = self._head_weight()
        return x @ w.T

    def _run_blocks(self, level: int, x, cos, sin, attn_mask) -> jnp.ndarray:
        for block in self.lms[level]:
            x = block(x, cos, sin, attn_mask, self.cfg.pos_method)
        return self.ln_fs[level](x)

    def _cascade(self, byte_ids: jnp.ndarray) -> jnp.ndarray:
        """Full recompute. Returns the final byte-level query stream x_cross (B, L, D), post
        every active fuse stage's cross-attention + refinement pass."""
        cfg = self.cfg
        B, L = byte_ids.shape
        hd = self.head_dim
        pm = cfg.pos_method

        byte_pos = jnp.arange(L)
        cos_b, sin_b = (rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        if pm == "learnable":
            x0 = x0 + self.wpe(byte_pos)[None]
        h = self._run_blocks(0, x0, cos_b, sin_b, byte_mask)

        cur_h = h
        x_cross = h
        cum_K = 1
        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break

            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]

            code_local_pos = jnp.arange(n_blocks)
            cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(s + 1, code_h, cos_c, sin_c, code_mask)

            code_pos_abs = (jnp.arange(n_blocks) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if pm == "rope" else (None, None))

            x_cross = self.fuse_stages[s](x_cross, h_code, cos_b, sin_b, cos_k, sin_k, fuse_mask, pm)
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)
            cur_h = h_code

        return x_cross

    def __call__(self, byte_ids: jnp.ndarray) -> tuple:
        cfg = self.cfg
        V = self.vocab
        L = byte_ids.shape[1]
        x_cross = self._cascade(byte_ids)

        logits = self._readout(x_cross[:, :-1, :])
        targets = byte_ids[:, 1:]
        loss = cross_entropy(logits, targets)

        mtp_losses, mtp_accs = [], []
        for i, head_i in enumerate(self.extra_heads):
            k = i + 2
            if L <= k:
                continue
            logits_i = head_i(x_cross[:, :-k, :])
            targets_i = byte_ids[:, k:]
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

    def _forward_next_byte_logits(self, byte_ids: jnp.ndarray) -> jnp.ndarray:
        """Full recompute over the whole sequence so far, returns logits for the NEXT byte."""
        x_cross = self._cascade(byte_ids)
        return self._readout(x_cross[:, -1, :])

    def generate_no_cache(self, prompt_bytes: jnp.ndarray, n_new_bytes: int) -> jnp.ndarray:
        """Byte-by-byte, full recompute each step -- correctness reference for generate_kv_cache."""
        if prompt_bytes.ndim == 1:
            prompt_bytes = prompt_bytes[None]
        all_bytes = prompt_bytes
        for _ in range(n_new_bytes):
            logits = self._forward_next_byte_logits(all_bytes)
            next_byte = jnp.argmax(logits, axis=-1, keepdims=True)
            all_bytes = jnp.concatenate([all_bytes, next_byte], axis=1)
        return all_bytes[0]

    def _make_incremental_stepper(self, Bsz: int):
        """Factory for the real incremental-KV-cache stepper -- O(1) new attention work per new
        byte, vs generate_no_cache's full O(L) recompute."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim
        pm = cfg.pos_method

        byte_caches = [None] * cfg.n_layers
        refine_caches = [[None] * cfg.n_layers for _ in range(self.n_fuse)]
        state = {"h_hist": None}
        stage_h_hist = [jnp.zeros((Bsz, 0, D)) for _ in range(self.n_fuse)]
        x_in_backlog = [None] * self.n_fuse
        cum_Ks = []
        cum = 1
        for K_s in cfg.Ks[: self.n_fuse]:
            cum *= K_s
            cum_Ks.append(cum)

        def step(byte_chunk: jnp.ndarray, start_pos: int) -> jnp.ndarray:
            Tn = byte_chunk.shape[1]
            pos = jnp.arange(start_pos, start_pos + Tn)
            cos_b, sin_b = (rope_cos_sin_for_positions(pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
            h_new = self.embed(byte_chunk)
            if pm == "learnable":
                h_new = h_new + self.wpe(pos)[None]
            for l, block in enumerate(self.lms[0]):
                h_new, byte_caches[l] = block.forward_incremental(h_new, cos_b, sin_b, byte_caches[l], cfg.attn_window, pm)
            h_new = self.ln_fs[0](h_new)
            state["h_hist"] = h_new if state["h_hist"] is None else jnp.concatenate([state["h_hist"], h_new], axis=1)

            x_in = h_new
            cur_h_hist = state["h_hist"]
            logits_full = self._readout(x_in)  # fallback if n_fuse==0 or no stage active yet
            for s in range(self.n_fuse):
                K_s = cfg.Ks[s]
                n_blocks = cur_h_hist.shape[1] // K_s
                if n_blocks > stage_h_hist[s].shape[1]:
                    code_h = cur_h_hist[:, K_s - 1::K_s, :][:, :n_blocks, :]
                    code_local_pos = jnp.arange(n_blocks)
                    cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
                    code_mask = causal_mask(code_local_pos, code_local_pos, None)
                    stage_h_hist[s] = self._run_blocks(s + 1, code_h, cos_c, sin_c, code_mask)
                h_code = stage_h_hist[s]
                n_blocks_now = h_code.shape[1]

                if n_blocks_now < 1:
                    x_in_backlog[s] = x_in if x_in_backlog[s] is None else jnp.concatenate([x_in_backlog[s], x_in], axis=1)
                    break

                code_pos_abs = (jnp.arange(n_blocks_now) + 1) * cum_Ks[s] - 1
                window_s = self.fuse_windows[s]
                cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if pm == "rope" else (None, None))

                if refine_caches[s][0] is None:
                    x_q = x_in if x_in_backlog[s] is None else jnp.concatenate([x_in_backlog[s], x_in], axis=1)
                    x_in_backlog[s] = None
                else:
                    x_q = x_in
                q_len = x_q.shape[1]
                q_start = (start_pos + Tn) - q_len
                q_pos = jnp.arange(q_start, q_start + q_len)
                cos_q, sin_q = (rope_cos_sin_for_positions(q_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))
                fuse_mask = causal_mask(q_pos, code_pos_abs, window_s)

                x_cross = self.fuse_stages[s](x_q, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask, pm)
                for l, block in enumerate(self.lms[0]):
                    x_cross, refine_caches[s][l] = block.forward_incremental(
                        x_cross, cos_q, sin_q, refine_caches[s][l], cfg.attn_window, pm)
                x_cross = self.ln_fs[0](x_cross)
                logits_full = self.fuse_stages[s].readout(x_cross, self._head_weight())
                x_in = x_cross
                cur_h_hist = h_code
            return logits_full

        return step

    def generate_kv_cache(self, prompt_bytes: jnp.ndarray, n_new_bytes: int) -> jnp.ndarray:
        """Real incremental KV cache. Produces the exact same argmax trajectory as
        generate_no_cache (see check_kv_cache_consistency)."""
        if prompt_bytes.ndim == 1:
            prompt_bytes = prompt_bytes[None]
        step = self._make_incremental_stepper(prompt_bytes.shape[0])

        all_bytes = prompt_bytes
        logits_all = step(all_bytes, 0)  # prime the caches with the whole prompt
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_bytes):
            next_byte = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_bytes = jnp.concatenate([all_bytes, next_byte], axis=1)
            logits_all = step(next_byte, all_bytes.shape[1] - 1)
            next_logits = logits_all[:, -1, :]

        return all_bytes[0]

    def check_kv_cache_consistency(self, val_data: jnp.ndarray, key: jax.random.PRNGKey,
                                    n_checks: int = 3, prompt_len: int = 8, n_new_bytes: int = 24) -> dict:
        """Diagnostic: generate_no_cache vs generate_kv_cache MUST produce bit-exact identical
        greedy trajectories. Should always return match_rate == 1.0."""
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            key, subkey = jax.random.split(key)
            start = int(jax.random.randint(subkey, (), 0, max(1, val_data.shape[0] - pl - n_new_bytes)))
            prompt = val_data[start:start + pl]
            out_full = self.generate_no_cache(prompt, n_new_bytes)
            out_cache = self.generate_kv_cache(prompt, n_new_bytes)
            if jnp.array_equal(out_full, out_cache):
                n_match += 1
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return nll.mean()
