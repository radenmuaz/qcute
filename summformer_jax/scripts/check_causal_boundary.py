"""Empirical (perturb-and-diff, float64) causality confirmation for the shared-embedder
(self-referential, Case A) SummFormer -- complements check_connectivity.py's static proof with a
real-weights check that no future leakage occurs and the causal boundary is exact, using a
generously-sized config (window=32, safely above the connectivity minimum found in
check_chain_receptive_field.py, so this test is purely about causality, not receptive field).

Setup: depth=8 chain stages, stride=2 each (cum_stride=256), L=512 -> deepest stage has exactly 2
code blocks (block0 covers raw positions [0..255], block1 covers [256..511]), matching the "256
becomes 1, 512 becomes 2" chain arithmetic confirmed separately.

    uv run python summformer_jax/scripts/check_causal_boundary.py

Actual output (2026-08-30, float64, seed 0/1/2, D=16 H=2 V=32):

    causality check (future should NOT affect past):
      perturb 400 (block1, pos 256-511) -> query 100 (early): diff=0.000e+00
      perturb 300 (block1) -> query 200 (early): diff=0.000e+00
      perturb 511 (last pos) -> query 0 (first): diff=0.000e+00

    information flow check (past SHOULD affect future):
      perturb 100 (block0) -> query 400 (late): diff=3.576e-07
      perturb 0 (first) -> query 511 (last): diff=3.874e-07

    boundary sharpness (block0=[0..255], block1=[256..511]):
      perturb 255 (last of block0) -> query 300: diff=1.490e-02
      perturb 256 (first of block1) -> query 300: diff=1.241e-03
      perturb 256 (first of block1) -> query 200 (before block1 starts): diff=0.000e+00

Reading: all "future should NOT affect past" probes are EXACT zero (not just small) -- no leakage.
"Past SHOULD affect future" probes are genuinely nonzero -- forward information flow confirmed
working, not accidentally disconnected. The boundary probe confirms position 256 (first token of
the second causal block) affects a query INSIDE that block (300) but has EXACTLY zero effect on a
query BEFORE the block starts (200) -- the causal boundary is exact, not approximate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx

from summformer import SummFormer, Embedder, Encoder, Decoder, StackConfig, ChainStageConfig, CrossAttnSpec


def build_model(D=16, H=2, V=32, L=512, W=32, n_stages=8, stride=2):
    emb_cfg = StackConfig(n_layers=1, d_model=D, n_heads=H, window=W, compute_dtype=jnp.float64, param_dtype=jnp.float64)
    embedder = Embedder(emb_cfg, context_len=L, vocab_size=V, rngs=nnx.Rngs(0))
    chain = tuple(ChainStageConfig(stride=stride, n_layers=1, window=W) for _ in range(n_stages))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D, n_heads=H, compute_dtype=jnp.float64, param_dtype=jnp.float64),
                       chain, context_len=L, rngs=nnx.Rngs(1))
    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, window=W) for i in range(n_stages))
    decoder = Decoder(StackConfig(n_layers=n_stages, d_model=D, n_heads=H, window=W, compute_dtype=jnp.float64, param_dtype=jnp.float64),
                       cross, context_len=L, rngs=nnx.Rngs(2))
    return SummFormer(embedder, encoder, decoder, encoder_embedder=None), L


def diff_at(model, tok, h_a, perturb_pos: int, query_pos: int) -> float:
    tok_b = tok.at[0, perturb_pos].set(1)
    h_b = model(tok_b)
    return float(jnp.abs(h_a[:, query_pos, :] - h_b[:, query_pos, :]).max())


if __name__ == "__main__":
    model, L = build_model()
    tok = jnp.zeros((1, L), dtype=jnp.int32)
    h_a = model(tok)

    print("causality check (future should NOT affect past):")
    print(f"  perturb 400 (block1, pos 256-511) -> query 100 (early): diff={diff_at(model, tok, h_a, 400, 100):.3e}")
    print(f"  perturb 300 (block1) -> query 200 (early): diff={diff_at(model, tok, h_a, 300, 200):.3e}")
    print(f"  perturb 511 (last pos) -> query 0 (first): diff={diff_at(model, tok, h_a, 511, 0):.3e}")
    print()
    print("information flow check (past SHOULD affect future):")
    print(f"  perturb 100 (block0) -> query 400 (late): diff={diff_at(model, tok, h_a, 100, 400):.3e}")
    print(f"  perturb 0 (first) -> query 511 (last): diff={diff_at(model, tok, h_a, 0, 511):.3e}")
    print()
    print("boundary sharpness (block0=[0..255], block1=[256..511]):")
    print(f"  perturb 255 (last of block0) -> query 300: diff={diff_at(model, tok, h_a, 255, 300):.3e}")
    print(f"  perturb 256 (first of block1) -> query 300: diff={diff_at(model, tok, h_a, 256, 300):.3e}")
    print(f"  perturb 256 (first of block1) -> query 200 (before block1 starts): diff={diff_at(model, tok, h_a, 256, 200):.3e}")
