"""Shared Equinox building blocks for the image_gen_cifar_jax model family: RMSNorm, SwiGLU,
GQA+RoPE attention (batched-training form and single-step KV-cached form), a residual block,
and checkpoint save/load helpers. Every run_*.py Equinox port imports from here instead of
redefining these -- kept numerically identical to the plain-dict-pytree run_*_v1.py versions
(no bias terms anywhere, same RMSNorm eps, same RoPE convention) so ports can be validated by
copying v1's array weights in and checking exact output match.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp


def rmsnorm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x * weight


class RMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float = eqx.field(static=True, default=1e-6)

    def __init__(self, dim: int):
        self.weight = jnp.ones((dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(eqx.Module):
    gate: jnp.ndarray
    up: jnp.ndarray
    down: jnp.ndarray

    def __init__(self, key, d_model: int, mlp_mult: int):
        hidden = d_model * mlp_mult
        k1, k2, k3 = jax.random.split(key, 3)
        self.gate = jax.random.normal(k1, (d_model, hidden)) * 0.02
        self.up = jax.random.normal(k2, (d_model, hidden)) * 0.02
        self.down = jax.random.normal(k3, (hidden, d_model)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jax.nn.silu(x @ self.gate) * (x @ self.up)) @ self.down


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rope_cos_sin_pos(pos, head_dim: int, base: float) -> tuple:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = pos * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def apply_rope_single(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos[None, :] + rotate_half(x) * sin[None, :]


class Attention(eqx.Module):
    qkv: jnp.ndarray
    out: jnp.ndarray
    q_norm: jnp.ndarray
    k_norm: jnp.ndarray
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, rope_base: float):
        hd = d_model // n_heads
        k1, k2 = jax.random.split(key, 2)
        self.qkv = jax.random.normal(k1, (d_model, d_model + 2 * n_kv_heads * hd)) * 0.02
        self.out = jax.random.normal(k2, (d_model, d_model)) * 0.02
        self.q_norm = jnp.ones((hd,))
        self.k_norm = jnp.ones((hd,))
        self.n_heads, self.n_kv_heads, self.rope_base = n_heads, n_kv_heads, rope_base

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Batched, causal-masked training-time forward: x is (B,T,D)."""
        B, T, D = x.shape
        hd = D // self.n_heads
        qkv = x @ self.qkv
        q, k, v = jnp.split(qkv, [D, D + self.n_kv_heads * hd], axis=-1)
        q = q.reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin(T, hd, self.rope_base)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k, v = jnp.repeat(k, n_rep, axis=1), jnp.repeat(v, n_rep, axis=1)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return y @ self.out

    def step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, T_max: int) -> tuple:
        """Single-step KV-cached form: x_new is (Bc,D), cache_k/v are (Bc,n_kv_heads,T_max,hd)."""
        Bc, D = x_new.shape
        hd = D // self.n_heads
        qkv = x_new @ self.qkv
        q, k, v = jnp.split(qkv, [D, D + self.n_kv_heads * hd], axis=-1)
        q, k, v = q.reshape(Bc, self.n_heads, hd), k.reshape(Bc, self.n_kv_heads, hd), v.reshape(Bc, self.n_kv_heads, hd)
        q, k = rmsnorm(q, self.q_norm), rmsnorm(k, self.k_norm)
        cos, sin = rope_cos_sin_pos(pos, hd, self.rope_base)
        q, k = apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
        cache_k = jax.lax.dynamic_update_slice(cache_k, k[:, :, None, :], (0, 0, pos, 0))
        cache_v = jax.lax.dynamic_update_slice(cache_v, v[:, :, None, :], (0, 0, pos, 0))
        n_rep = self.n_heads // self.n_kv_heads
        k_full = jnp.repeat(cache_k, n_rep, axis=1) if n_rep > 1 else cache_k
        v_full = jnp.repeat(cache_v, n_rep, axis=1) if n_rep > 1 else cache_v
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhd,bhtd->bht", q, k_full) * scale
        valid = jnp.arange(T_max) <= pos
        logits = jnp.where(valid[None, None, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bht,bhtd->bhd", attn, v_full).reshape(Bc, D)
        return y @ self.out, cache_k, cache_v


class Block(eqx.Module):
    norm1: RMSNorm
    attn: Attention
    norm2: RMSNorm
    mlp: SwiGLU

    def __init__(self, key, d_model: int, n_heads: int, n_kv_heads: int, mlp_mult: int, rope_base: float):
        k1, k2 = jax.random.split(key, 2)
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(k1, d_model, n_heads, n_kv_heads, rope_base)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(k2, d_model, mlp_mult)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_new: jnp.ndarray, cache_k: jnp.ndarray, cache_v: jnp.ndarray, pos, T_max: int) -> tuple:
        attn_out, ck, cv = self.attn.step(self.norm1(x_new), cache_k, cache_v, pos, T_max)
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
