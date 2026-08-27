"""JAX/Flax NNX port of Cable/src/model_gpt.py (https://github.com/axiomlab/Cable), exact
architecture port restricted to 3 of that repo's `pos_method` options (the rest — cable/kcable/
alibi/fire/kerple/t5bias/rotali/sinusoidal — are not ported):

  - "rope"      -- pos_methods/rope.py: rotary embeddings applied to q/k in attention, no
                   positional signal added to the input embedding.
  - "learnable" -- GPT-2's own original absolute position embedding (a `wpe` table added to
                   the token embedding), plain (BASE_ATTENTION) attention otherwise.
  - "base"      -- NoPE: BASE_ATTENTION with no wpe and no rotary -- no positional signal
                   anywhere in the model.

Same LayerNorm+GELU nanoGPT block structure, weight-tied lm_head/wte, and NANOGPT_SCALE_INIT
residual-projection init scaling (std *= (2*n_layer)**-0.5) as the PyTorch original -- see that
file's own docstring-equivalent comments for design rationale, not repeated here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class ModelConfig:
    pos_method: str = "rope"  # one of "rope", "learnable", "base"
    block_size: int = 1024
    vocab_size: int = 50304  # padded up from 50257, like Cable's own train_gpt.py
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512


def _linear_init(std: float):
    return nnx.initializers.normal(stddev=std)


class MLP(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, kernel_init=_linear_init(0.02), rngs=rngs)
        self.c_proj = nnx.Linear(
            4 * config.n_embd, config.n_embd,
            kernel_init=_linear_init(0.02 * (2 * config.n_layer) ** -0.5), rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        return self.c_proj(x)


def _rope_cos_sin(seq_len: int, rotary_dim: int, base: float = 10000.0):
    # Matches Cable's rope.py use of rotary_embedding_torch (rotate_dim=64, half-dim frequency
    # table, standard RoPE) -- rotary_dim is the per-head dim actually rotated (<= head_dim).
    inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)              # [T, rotary_dim/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [T, rotary_dim]
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, cos, sin):
    # x: [B, H, T, rotary_dim]
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class CausalSelfAttention(nnx.Module):
    """Ports pos_methods/base.py (pos_method="base"/"learnable") and pos_methods/rope.py
    (pos_method="rope") into one module -- both share the identical qkv/output-projection
    shape and causal-softmax-attention math in the original; the only difference between them
    is whether q/k get rotated before the score matmul."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.pos_method = config.pos_method
        self.head_dim = config.n_embd // config.n_head
        self.rotary_dim = min(64, self.head_dim)  # matches Cable's RotaryEmbedding(dim=64)
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, kernel_init=_linear_init(0.02), rngs=rngs)
        self.c_proj = nnx.Linear(
            config.n_embd, config.n_embd,
            kernel_init=_linear_init(0.02 * (2 * config.n_layer) ** -0.5), rngs=rngs,
        )

    def __call__(self, x):
        B, T, C = x.shape
        H, hd = self.n_head, self.head_dim
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # [B, H, T, hd]
        k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        if self.pos_method == "rope":
            cos, sin = _rope_cos_sin(T, self.rotary_dim)
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q = jnp.concatenate([_apply_rope(q_rot, cos, sin), q_pass], axis=-1)
            k = jnp.concatenate([_apply_rope(k_rot, cos, sin), k_pass], axis=-1)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(hd)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(causal_mask[None, None], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class Block(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, embedding_init=_linear_init(0.02), rngs=rngs)
        self.wpe = (
            nnx.Embed(config.block_size, config.n_embd, embedding_init=_linear_init(0.02), rngs=rngs)
            if config.pos_method == "learnable" else None
        )
        self.h = nnx.List([Block(config, rngs=rngs) for _ in range(config.n_layer)])
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)
        # weight-tied lm_head: reuse wte's embedding matrix as the output projection kernel
        # directly in __call__ (no separate nnx.Linear/param -- see forward below), matching
        # `self.transformer.wte.weight = self.lm_head.weight` in the PyTorch original.

    def __call__(self, idx):
        # idx: [B, T] int32 -> logits [B, T, vocab]
        B, T = idx.shape
        x = self.wte(idx)
        if self.config.pos_method == "learnable":
            assert T <= self.config.block_size, (
                f"Cannot forward sequence of length {T} in learnable positional encoding, "
                f"block size is only {self.config.block_size}"
            )
            pos = jnp.arange(T)
            x = x + self.wpe(pos)[None]
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.embedding.value.T  # tied head
        return logits


def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return nll.mean()
