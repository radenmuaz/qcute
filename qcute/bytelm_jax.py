"""qcute.bytelm_jax — pure-JAX reimplementation of qcute.bytelm_tpu's LLaMA-style byte-LM
(RMSNorm, SwiGLU, RoPE, bias-free, zero_kv_sink), same enwik8 dataset, same d512x16-shaped
model (~67M params) — written 2026-08-23 to check whether torch_xla's flash-attention *wrapper*
(which re-invokes `jax.jit` on the Pallas kernel from inside a torch custom op on every call, per
the traceback seen debugging that path) is itself the source of the ~25x per-step slowdown found
combining zero_kv_sink with flash-attention in bytelm_tpu.py, as opposed to the K/V/Q concat cost
being fundamentally that expensive at the XLA level regardless of framework. Whole train step
(forward+backward+optimizer) is a single `jax.jit`-compiled function here, compiled once and
reused every step — the natural JAX pattern, unlike torch_xla's per-step lazy-graph tracing.

No AdamW/optimizer library dependency (optax isn't installed on the TPU nodes used this session)
-- Adam implemented directly via jax.tree_util, ~15 lines, see `adam_update`.

Minimal on purpose: single next-byte head only (bandwidth-matched to bytelm_tpu's PRESETS["d512x16"],
which already uses mtp_heads=1), no checkpointing, no eval-split full-pass, no CLI config-file
loading — just enough to benchmark steady-state it/s for the same zero_kv_sink+flash-attention
combination bytelm_tpu.py struggled with, using the identical enwik8.gz byte data.

    python3 -m qcute.bytelm_jax [--no_zero_kv_sink] [--context N] [--steps N]
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

# ---- config (mirrors bytelm_tpu.py's PRESETS["d512x16"]) ----
VOCAB = 256
D_MODEL = 512
N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
MLP_MULT = 4
HIDDEN = MLP_MULT * D_MODEL
ROPE_BASE = 10000.0
BATCH_SIZE = 4
LR_PEAK = 1e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 10.0
STEPS = 50
LOG_EVERY = 5
DATA_PATH = Path("datasets/enwik8.gz")
ZERO_KV_SINK = True  # set by main() from --no_zero_kv_sink; module-level so attention() sees it


def load_enwik8(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def init_params(key: jax.Array) -> dict:
    def linear(key, d_in, d_out, scale=0.02):
        return scale * jax.random.normal(key, (d_in, d_out), dtype=jnp.float32)

    keys = jax.random.split(key, 4 + N_LAYERS * 6)
    ki = iter(keys)
    params = {
        "tok_emb": 0.02 * jax.random.normal(next(ki), (VOCAB, D_MODEL), dtype=jnp.float32),
        "ln_f_w": jnp.ones((D_MODEL,), dtype=jnp.float32),
        "head_w": 0.02 * jax.random.normal(next(ki), (D_MODEL, VOCAB), dtype=jnp.float32),
        "blocks": [],
    }
    resid_scale = 0.02 / (2 * N_LAYERS) ** 0.5
    for _ in range(N_LAYERS):
        block = {
            "ln1_w": jnp.ones((D_MODEL,), dtype=jnp.float32),
            "ln2_w": jnp.ones((D_MODEL,), dtype=jnp.float32),
            "qkv_w": linear(next(ki), D_MODEL, 3 * D_MODEL),
            "out_w": resid_scale * jax.random.normal(next(ki), (D_MODEL, D_MODEL), dtype=jnp.float32),
            "gate_w": linear(next(ki), D_MODEL, HIDDEN),
            "up_w": linear(next(ki), D_MODEL, HIDDEN),
            "down_w": resid_scale * jax.random.normal(next(ki), (HIDDEN, D_MODEL), dtype=jnp.float32),
        }
        params["blocks"].append(block)
    return params


def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x * w


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> tuple[jax.Array, jax.Array]:
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    # x: [B, H, T, head_dim], cos/sin: [T, head_dim] (kept fp32 for RoPE precision, but cast to
    # x's dtype before the multiply -- otherwise JAX's type promotion silently upcasts a bf16 x
    # back to fp32 here, quietly undoing forward()'s bf16 cast for the rest of attention).
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos[None, None] + rot * sin[None, None]


def swiglu(x: jax.Array, gate_w, up_w, down_w) -> jax.Array:
    return (jax.nn.silu(x @ gate_w) * (x @ up_w)) @ down_w


def attention(x: jax.Array, block: dict, cos, sin) -> jax.Array:
    B, T, D = x.shape
    qkv = (x @ block["qkv_w"]).reshape(B, T, 3, N_HEADS, HEAD_DIM)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = apply_rope(q.transpose(0, 2, 1, 3), cos, sin)  # [B, H, T, hd]
    k = apply_rope(k.transpose(0, 2, 1, 3), cos, sin)
    v = v.transpose(0, 2, 1, 3)

    if ZERO_KV_SINK:
        # zero_kv_sink: prepend one all-zero K/V row (always attendable), pad Q with one dummy
        # leading row to keep q_len==kv_len==T+1 (required for the Pallas kernel's causal=True)
        # -- same trick as bytelm_tpu.py's CausalSelfAttention.forward, see that file's comments
        # for the full derivation/verification. Caller must pick context so T+1 is block-aligned.
        zero = jnp.zeros((B, N_HEADS, 1, HEAD_DIM), dtype=k.dtype)
        k = jnp.concatenate([zero, k], axis=2)
        v = jnp.concatenate([zero, v], axis=2)
        q_padded = jnp.concatenate([zero.astype(q.dtype), q], axis=2)
        y = flash_attention(q_padded, k, v, causal=True, sm_scale=1.0 / (HEAD_DIM ** 0.5))
        y = y[:, :, 1:, :]  # drop the dummy leading row
    else:
        y = flash_attention(q, k, v, causal=True, sm_scale=1.0 / (HEAD_DIM ** 0.5))
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ block["out_w"]


def forward(params: dict, tokens: jax.Array) -> jax.Array:
    # bf16 compute (fp32 master params via adam_update, matching bytelm_tpu.py's autocast_ctx)
    # -- forward/backward matmuls in bf16, only the norm/softmax reductions stay fp32-precision
    # internally as usual. Added 2026-08-23 after finding fp32-vs-bf16 was an uncontrolled
    # confound in the first zero_kv_sink-vs-no-sink JAX benchmark (both ran identically slow in
    # plain fp32, which masked whether the sink itself mattered at all).
    params = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16), params)
    B, T = tokens.shape
    cos, sin = rope_cos_sin(T, HEAD_DIM, ROPE_BASE)
    x = params["tok_emb"][tokens]
    for block in params["blocks"]:
        x = x + attention(rms_norm(x, block["ln1_w"]), block, cos, sin)
        x = x + swiglu(rms_norm(x, block["ln2_w"]), block["gate_w"], block["up_w"], block["down_w"])
    x = rms_norm(x, params["ln_f_w"])
    return x @ params["head_w"]  # [B, T, VOCAB]


def loss_fn(params: dict, batch: jax.Array) -> jax.Array:
    # batch: [B, CONTEXT+1] -- inputs = batch[:, :-1], targets = batch[:, 1:]
    logits = forward(params, batch[:, :-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    targets = batch[:, 1:]
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll) / jnp.log(2.0)  # bits-per-byte


def adam_update(params, grads, m, v, step, lr, wd, b1=0.9, b2=0.95, eps=1e-8):
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * (g ** 2), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** step), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p), params, m_hat, v_hat
    )
    return new_params, m, v


@jax.jit
def train_step(params, m, v, step, batch, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    g_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    clip_coef = jnp.minimum(1.0, GRAD_CLIP / (g_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * clip_coef, grads)
    params, m, v = adam_update(params, grads, m, v, step, lr, WEIGHT_DECAY)
    return params, m, v, loss


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR_PEAK * step / max(1, WARMUP_STEPS)
    return LR_PEAK


def main():
    global CONTEXT, ZERO_KV_SINK, STEPS
    p = argparse.ArgumentParser()
    p.add_argument("--no_zero_kv_sink", action="store_true")
    p.add_argument("--context", type=int, default=8191)
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()
    ZERO_KV_SINK = not args.no_zero_kv_sink
    CONTEXT = args.context
    STEPS = args.steps

    data = load_enwik8(DATA_PATH)
    print(f"train_bytes={len(data)}  d_model={D_MODEL}  n_layers={N_LAYERS}  context={CONTEXT}"
          f"  batch_size={BATCH_SIZE}  zero_kv_sink={ZERO_KV_SINK}  flash_attention=True")

    key = jax.random.PRNGKey(0)
    params = init_params(key)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"params={n_params / 1e6:.1f}M")

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    seq_len = CONTEXT + 1  # +1 for the shift-by-one input/target split
    rng = np.random.default_rng(0)
    max_start = len(data) - seq_len - 1

    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        starts = rng.integers(0, max_start, size=BATCH_SIZE)
        batch_np = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int32)
        batch = jnp.asarray(batch_np)
        lr = lr_at(step)
        params, m, v, loss = train_step(params, m, v, step, batch, lr)
        if step == 1:
            loss.block_until_ready()
            print(f"first_step_compile_s {time.perf_counter() - t0:.1f}")
            t0 = time.perf_counter()
        if step % LOG_EVERY == 0:
            loss.block_until_ready()
            elapsed = time.perf_counter() - t0
            it_s = (step - 1) / elapsed if step > 1 else 0.0
            print(f"step {step:4d}  bpb {float(loss):.4f}  lr {lr:.2e}  it_s {it_s:.3f}"
                  f"  elapsed_s {elapsed:.1f}")


if __name__ == "__main__":
    main()
