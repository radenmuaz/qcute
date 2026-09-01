"""
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

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as _tpu_flash_attention
    _HAS_FLASH_ATTENTION = True
except ImportError:
    _tpu_flash_attention = None
    _HAS_FLASH_ATTENTION = False


def chunked_windowed_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, window: int) -> jnp.ndarray:
    """Real O(T*window) windowed causal self-attention -- ported from summformer_jax/summformer.py's
    chunked_windowed_attention (no zero-KV sink, no flash option -- gpt2_windowed_jax's own
    windowed-attention ablation baseline, kept as a self-contained copy per this repo's
    flat-file-per-training-script convention rather than a cross-lineage import). Reshapes into
    `window`-sized blocks; each block attends only to itself + the immediately preceding block
    (2*window keys, not T keys). Falls back to dense causal sdpa when T<=window or T%window!=0."""
    B, H, T, hd = q.shape
    w = window
    scale = 1.0 / math.sqrt(hd)

    def _dense_sdpa(q, k, v, mask):
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        scores = jnp.where(mask, scores.astype(jnp.float32), -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bhkd->bhqd", attn, v)

    if T <= w or T % w != 0:
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]
        return _dense_sdpa(q, k, v, mask)

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

    y = _dense_sdpa(qb_flat, k_win_flat, v_win_flat, mask_flat)
    return y.reshape(B, n_chunks, H, w, hd).transpose(0, 2, 1, 3, 4).reshape(B, H, T, hd)


@dataclass
class ModelConfig:
    pos_method: str = "rope"  # one of "rope", "learnable", "base"
    block_size: int = 1024
    vocab_size: int = 50304  # padded up from 50257, like Cable's own train_gpt.py
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    use_flash_attention: bool = False
    # bounded local self-attention window (chunked_windowed_attention above), -1 = dense/unbounded
    # (default, original gpt2_jax behavior). Mutually exclusive with use_flash_attention in
    # practice -- when window > 0 the flash-attention kernel is bypassed, see __call__ below.
    window: int = -1
    # mixed precision: matmuls (Linear/Embed) compute in this dtype, params stored in
    # param_dtype (fp32) as the master copy; LayerNorm/softmax/loss stay fp32 for stability,
    # matching torch.autocast's own policy in Cable's original.
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    # gradient checkpointing: wraps the whole block stack's forward in jax.checkpoint (via
    # nnx.remat), recomputing activations in the backward pass instead of storing them --
    # same opt-in, whole-stack-wrapped pattern as summformer_jax's BlockStack.use_remat.
    use_remat: bool = False


def _linear_init(std: float):
    return nnx.initializers.normal(stddev=std)


class MLP(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embd, 4 * config.n_embd, kernel_init=_linear_init(0.02),
            dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embd, config.n_embd,
            kernel_init=_linear_init(0.02 * (2 * config.n_layer) ** -0.5),
            dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
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
    # x: [B, H, T, rotary_dim]. cos/sin computed fp32 (frequency precision matters); cast
    # down to x's dtype so this doesn't silently upcast bf16 q/k back to fp32.
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
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
        self.block_size = config.block_size
        self.use_flash_attention = config.use_flash_attention and _HAS_FLASH_ATTENTION
        self.window = config.window
        self.c_attn = nnx.Linear(
            config.n_embd, 3 * config.n_embd, kernel_init=_linear_init(0.02),
            dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_embd, config.n_embd,
            kernel_init=_linear_init(0.02 * (2 * config.n_layer) ** -0.5),
            dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
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

        if self.window > 0:
            # Bounded local attention -- bypasses flash-attention (no windowed Pallas kernel
            # wired up here, matches summformer_jax's own choice not to combine the two).
            y = chunked_windowed_attention(q, k, v, self.window)
        elif self.use_flash_attention:
            # Pallas TPU kernel wants bf16 q/k/v; output comes back in that dtype too.
            y = _tpu_flash_attention(
                q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                causal=True, sm_scale=1.0 / math.sqrt(hd),
            ).astype(x.dtype)
        else:
            # softmax in fp32 for stability, matching autocast's own policy; matmuls stay
            # in q/k/v's compute dtype (bf16).
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(hd)
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            scores = jnp.where(causal_mask[None, None], scores.astype(jnp.float32), -jnp.inf)
            attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
            y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)

    def forward_incremental(self, x_new, cache, start_pos: int):
        """Incremental KV-cache step: x_new is only the NEW chunk (usually 1 token), cache is the
        (k, v) accumulated so far (or None on the first/priming call). Always plain SDPA, never
        the flash-attention kernel -- that Pallas kernel needs kv_seq_len % 128 == 0 (see
        docs/status_tpu.md's bench_generation.py finding), which a growing single-token-at-a-time
        cache essentially never satisfies; production incremental decode kernels are a different
        thing entirely from a training-time flash kernel, out of scope here."""
        B, Tn, C = x_new.shape
        H, hd = self.n_head, self.head_dim
        qkv = self.c_attn(x_new)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, Tn, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tn, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tn, H, hd).transpose(0, 2, 1, 3)

        if self.pos_method == "rope":
            # start_pos may be a traced value under jit (decode step) -- build the full
            # cos/sin table up to block_size (static) and dynamic_slice into it, rather than
            # jnp.arange(start_pos + Tn) which needs a concrete Python int.
            cos_full, sin_full = _rope_cos_sin(self.block_size, self.rotary_dim)
            cos = jax.lax.dynamic_slice_in_dim(cos_full, start_pos, Tn, axis=0)
            sin = jax.lax.dynamic_slice_in_dim(sin_full, start_pos, Tn, axis=0)
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q = jnp.concatenate([_apply_rope(q_rot, cos, sin), q_pass], axis=-1)
            k = jnp.concatenate([_apply_rope(k_rot, cos, sin), k_pass], axis=-1)

        if cache is None:
            k_all, v_all = k, v
        else:
            k_prev, v_prev = cache
            k_all, v_all = jnp.concatenate([k_prev, k], axis=2), jnp.concatenate([v_prev, v], axis=2)
        S = k_all.shape[2]

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_all) / math.sqrt(hd)
        # start_pos may be a traced value under jit -- add it to a static jnp.arange(Tn)
        # instead of jnp.arange(start_pos, start_pos + Tn), which needs a concrete start.
        query_pos = (jnp.arange(Tn) + start_pos).reshape(-1, 1)
        key_pos = jnp.arange(S).reshape(1, -1)
        causal_mask = key_pos <= query_pos  # [Tn, S]
        scores = jnp.where(causal_mask[None, None], scores.astype(jnp.float32), -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1).astype(v_all.dtype)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v_all)
        y = y.transpose(0, 2, 1, 3).reshape(B, Tn, C)
        return self.c_proj(y), (k_all, v_all)


class Block(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        # LayerNorm stays fp32 compute (reduction precision matters), regardless of compute_dtype.
        self.ln_1 = nnx.LayerNorm(config.n_embd, dtype=jnp.float32, param_dtype=config.param_dtype, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, dtype=jnp.float32, param_dtype=config.param_dtype, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_incremental(self, x_new, cache, start_pos: int):
        attn_out, new_cache = self.attn.forward_incremental(self.ln_1(x_new), cache, start_pos)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln_2(x_new))
        return x_new, new_cache


class Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size, config.n_embd, embedding_init=_linear_init(0.02),
            dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
        )
        self.wpe = (
            nnx.Embed(
                config.block_size, config.n_embd, embedding_init=_linear_init(0.02),
                dtype=config.compute_dtype, param_dtype=config.param_dtype, rngs=rngs,
            )
            if config.pos_method == "learnable" else None
        )
        self.h = nnx.List([Block(config, rngs=rngs) for _ in range(config.n_layer)])
        # fp32 for the final norm + tied-head logits/loss -- numerically sensitive, cheap
        # relative to the rest of the model (one matmul), matches autocast's own policy.
        self.ln_f = nnx.LayerNorm(config.n_embd, dtype=jnp.float32, param_dtype=config.param_dtype, rngs=rngs)
        # weight-tied lm_head: reuse wte's embedding matrix as the output projection kernel
        # directly in __call__ (no separate nnx.Linear/param -- see forward below), matching
        # `self.transformer.wte.weight = self.lm_head.weight` in the PyTorch original.

    def _blocks_forward(self, x):
        for block in self.h:
            x = block(x)
        return x

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
        # use_remat=True (default False, opt-in) wraps the whole block stack's forward in
        # jax.checkpoint, recomputing activations in the backward pass instead of storing them,
        # trading TensorCore cycles for HBM. Same pattern as summformer_jax's BlockStack.use_remat
        # -- uses plain jax.checkpoint over a split/merge'd pure function, not nnx.remat directly
        # on the bound method, since that raised flax.errors.TraceContextError ("Cannot mutate
        # Param from a different trace level") once nested inside train_gpt.py's own
        # pmap+lax.scan (grad_accum) tracing (confirmed 2026-09-01, same failure as
        # summformer_jax's BlockStack).
        if self.config.use_remat:
            graphdef, state = nnx.split(self)

            def pure_blocks_forward(state, x):
                module = nnx.merge(graphdef, state)
                return module._blocks_forward(x)

            x = jax.checkpoint(pure_blocks_forward)(state, x)
        else:
            x = self._blocks_forward(x)
        x = self.ln_f(x)
        logits = x @ self.wte.embedding.value.T  # tied head
        return logits

    def _forward_next_token_logits(self, idx):
        """Full recompute over the whole sequence so far, returns logits for the NEXT token."""
        return self(idx)[:, -1, :]

    def generate_no_cache(self, prompt_tokens, n_new_tokens: int):
        """Token-by-token, full recompute each step -- correctness reference for
        generate_kv_cache. Same convention as summformer_jax's SummTransformer."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        all_tokens = prompt_tokens
        for _ in range(n_new_tokens):
            logits = self._forward_next_token_logits(all_tokens)
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
        return all_tokens[0]

    def _make_incremental_stepper(self, Bsz: int):
        """Factory for the real incremental-KV-cache stepper -- O(1) new attention work per new
        token (vs. generate_no_cache's full O(L) recompute), one cache per layer."""
        caches = [None] * self.config.n_layer

        def step(token_chunk, start_pos: int):
            Tn = token_chunk.shape[1]
            x = self.wte(token_chunk)
            if self.config.pos_method == "learnable":
                pos = jnp.arange(start_pos, start_pos + Tn)
                x = x + self.wpe(pos)[None]
            for l, block in enumerate(self.h):
                x, caches[l] = block.forward_incremental(x, caches[l], start_pos)
            x = self.ln_f(x)
            return x @ self.wte.embedding.value.T

        return step

    def generate_kv_cache(self, prompt_tokens, n_new_tokens: int):
        """Real incremental KV cache. Produces the exact same argmax trajectory as
        generate_no_cache (see check_kv_cache_consistency)."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        step = self._make_incremental_stepper(prompt_tokens.shape[0])

        all_tokens = prompt_tokens
        logits_all = step(all_tokens, 0)  # prime the cache with the whole prompt
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_tokens):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits_all = step(next_token, all_tokens.shape[1] - 1)
            next_logits = logits_all[:, -1, :]

        return all_tokens[0]

    def _prime_pure(self, prompt_tokens):
        """Pure prefill: not jitted (prompt length varies call to call). Returns
        (logits, caches) where caches is a tuple of (k, v) per layer -- a plain pytree,
        no closure mutation (see summformer_jax/image_gen's _prime_pure for the same fix)."""
        Tn = prompt_tokens.shape[1]
        x = self.wte(prompt_tokens)
        if self.config.pos_method == "learnable":
            x = x + self.wpe(jnp.arange(Tn))[None]
        caches = []
        for block in self.h:
            x, cache = block.forward_incremental(x, None, 0)
            caches.append(cache)
        x = self.ln_f(x)
        logits = x @ self.wte.embedding.value.T
        return logits, tuple(caches)

    def _decode_step_pure(self, token, pos_scalar, caches):
        """PURE single-token decode step: caches passed in and returned explicitly, no
        captured-list mutation -- safe to wrap in nnx.jit and reuse the compiled trace
        across every call (only cache content changes, not pytree structure/shapes)."""
        x = self.wte(token)
        if self.config.pos_method == "learnable":
            x = x + self.wpe(pos_scalar[None])[None]
        new_caches = []
        for l, block in enumerate(self.h):
            x, cache = block.forward_incremental(x, caches[l], pos_scalar)
            new_caches.append(cache)
        x = self.ln_f(x)
        logits = x @ self.wte.embedding.value.T
        return logits, tuple(new_caches)

    def generate_kv_cache_jit(self, prompt_tokens, n_new_tokens: int):
        """Same trajectory as generate_kv_cache, but the decode step is a pure function
        wrapped in nnx.jit (_jitted_decode_step_pure below) so the compiled trace is
        reused across all n_new_tokens calls instead of re-tracing/executing eagerly."""
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        logits, caches = self._prime_pure(prompt_tokens)
        all_tokens = prompt_tokens
        next_logits = logits[:, -1, :]
        for _ in range(n_new_tokens):
            next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            all_tokens = jnp.concatenate([all_tokens, next_token], axis=1)
            logits, caches = _jitted_decode_step_pure(self, next_token, all_tokens.shape[1] - 1, caches)
            next_logits = logits[:, -1, :]
        return all_tokens[0]

    def check_kv_cache_consistency(self, val_data, key, n_checks: int = 3, prompt_len: int = 8,
                                    n_new_tokens: int = 24) -> dict:
        """Diagnostic: generate_no_cache vs generate_kv_cache MUST produce bit-exact identical
        greedy trajectories. Should always return match_rate == 1.0."""
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            key, subkey = jax.random.split(key)
            start = int(jax.random.randint(subkey, (), 0, max(1, val_data.shape[0] - pl - n_new_tokens)))
            prompt = val_data[start:start + pl]
            out_full = self.generate_no_cache(prompt, n_new_tokens)
            out_cache = self.generate_kv_cache(prompt, n_new_tokens)
            if jnp.array_equal(out_full, out_cache):
                n_match += 1
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}


_jitted_decode_step_pure = nnx.jit(Model._decode_step_pure)


def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return nll.mean()
