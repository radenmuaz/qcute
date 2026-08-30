"""Receptive-field diagnostic for the Encoder's chained pooling + Decoder cross-attention.
Perturbs token 0 and checks whether the last decoder position's hidden state changes, sweeping
self-attn window (uniform across Embedder/chain stages/Decoder) and cross-attn at -1 (auto-derive
= cum_stride) across several random seeds. float64 required -- float32 buries the real signal in
noise for small windows (confirmed 2026-08-30: a "no clean threshold" conclusion drawn from float32
was an artifact, not real).

Findings from the L=256/8-stage/stride=2 sweep this was built for (see chat/docs/status_tpu.md):
  - Real minimum self-attn window for full chain-to-decoder connectivity is ~7-10 for this config,
    NOT dense/max_len and NOT the naive stride/cum_stride minimum (those are far too small -- the
    chain's own point-sampling pooling needs real self-attention slack to compound across stages).
  - One seed (0) showed a sandwiched-zero anomaly (w10=True, w11-12=False, w13+=True) --confirmed
    via a softmax-skip (uniform-average) monkeypatch ablation that this is a weight-value-dependent
    coincidental cancellation, NOT a structural/topological connectivity gap (4/5 other seeds are
    cleanly monotonic). Don't rely on a single seed's sweep to conclude "no clean threshold exists."

    uv run python summformer_jax/scripts/check_chain_receptive_field.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summformer_jax/

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from summformer import SummFormer, Embedder, Encoder, Decoder, StackConfig, ChainStageConfig, CrossAttnSpec


def test_full(window: int, seed: int, L: int, n_stages: int, stride: int, D: int = 16, H: int = 2, V: int = 32) -> float:
    emb_cfg = StackConfig(n_layers=1, d_model=D, n_heads=H, window=window, compute_dtype=jnp.float64, param_dtype=jnp.float64)
    embedder = Embedder(emb_cfg, context_len=L, vocab_size=V, rngs=nnx.Rngs(seed))
    chain = tuple(ChainStageConfig(stride=stride, n_layers=1, window=window) for _ in range(n_stages))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D, n_heads=H, compute_dtype=jnp.float64, param_dtype=jnp.float64),
                       chain, context_len=L, rngs=nnx.Rngs(seed + 1))
    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i) for i in range(n_stages))  # -1 auto-derive
    decoder = Decoder(StackConfig(n_layers=n_stages, d_model=D, n_heads=H, window=window, compute_dtype=jnp.float64, param_dtype=jnp.float64),
                       cross, context_len=L, rngs=nnx.Rngs(seed + 2))
    model = SummFormer(embedder, encoder, decoder, encoder_embedder=None)

    tok = jnp.zeros((1, L), dtype=jnp.int32)
    h_a = model(tok)
    tok_b = tok.at[0, 0].set(1)
    h_b = model(tok_b)
    return float(jnp.abs(h_a[:, -1, :] - h_b[:, -1, :]).max())


def sweep(L: int, n_stages: int, stride: int, windows: range, seeds: list[int]) -> None:
    print(f"--- L={L} n_stages={n_stages} stride={stride} (cum_stride={stride**n_stages}) ---")
    for seed in seeds:
        row = []
        for w in windows:
            old = sys.stdout
            sys.stdout = io.StringIO()
            diff = test_full(w, seed, L, n_stages, stride)
            sys.stdout = old
            row.append("T" if diff > 1e-12 else "F")
        print(f"seed={seed}: " + " ".join(f"w{w}={r}" for w, r in zip(windows, row)))


if __name__ == "__main__":
    sweep(L=256, n_stages=8, stride=2, windows=range(2, 15), seeds=[0, 10, 20, 30, 42])
