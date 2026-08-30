"""QueryClassifierHead + hourglass-widened chain -- a genuinely different design point from
tiny_vit_like_chained.py, built and verified 2026-08-30 (see chat/docs/status_tpu.md):

  - QueryClassifierHead: only the Encoder sees the image; classification is done by a SINGLE
    trainable query token that densely cross-attends (force_dense=True) into every chain stage's
    output, reusing the causal Decoder/CrossAttnSpec machinery with the query placed AFTER the
    full image sequence (so causal masking never excludes anything). No T-length decoder
    self-attention at all -- this removes the single biggest cost in the causal-decoder design
    (tiny_vit_like_chained.py measured 1427 GFLOPs; this config measured 5.94 GFLOPs, a 240x cut).
  - Hourglass width: d_model DOUBLES at each chain stage (16,32,64,128,256,512,512, capped) while
    the embedder (which must touch the full T=150528 sequence) stays thin (d_model=8). This keeps
    per-stage FLOPs roughly FLAT across depth (T shrinks 4x/stage at stride=4, D^2 grows 4x,
    they cancel) instead of the wide, uniform-D embedder/trunk that dominated the old design's
    cost. Real params land at 9.24M (~1.6x ViT-Ti's 5.7M, genuinely capacitated, not param-starved
    the way a naive small-D-everywhere shrink would be).
  - Receptive field: NOT required to cover the full image for this head (query is async --
    aggregates dense cross-attn from whatever stages exist, no single "last timestep" dependency
    chain the way the causal decoder has). Target heuristic used: deepest stage's cum_stride
    covers ~10-40% of the image. This config: cum_stride=4^7=16384, 10.88% coverage, 9 effective
    code tokens at the final stage -- confirmed via scripts/check_connectivity.py's
    check_query_connectivity (query_reachable=True, all 7 stages active).

Measured (2026-08-30): n_params=9,240,272 (9.24M), GFLOPs=5.94 (forward pass, T=150528).
ViT-Ti/16 reference: ~5.7M params, ~1.3 GFLOPs.

NOTE: no train.py wired to this config format yet (train_v2.py targets tiny_vit_like_chained.py's
ClassifierHead-based build_summformer; QueryClassifierHead needs its own loss/train loop since it
has a different __call__ signature and no MTP/generation machinery). This file exports a ready
build_summformer(rngs) factory for that future train script.

    uv run python summformer_jax/image_classification/smoke_test_new_arch.py  # (once adapted for QueryClassifierHead)
"""
import jax.numpy as jnp
from flax import nnx

from summformer import Embedder, Encoder, Decoder, QueryClassifierHead, StackConfig, ChainStageConfig, CrossAttnSpec

IMAGE_SIZE = 224
VOCAB_SIZE = 256  # byte-level RGB
CONTEXT_LEN = IMAGE_SIZE * IMAGE_SIZE * 3  # 150528

EMBED_D = 8
STRIDE = 4
N_STAGES = 7
STAGE_DIMS = [16, 32, 64, 128, 256, 512, 512]  # hourglass: doubles until capped at 512
SELF_WINDOW = 8  # validated minimum-safe margin for stride=4 chain (see README.md)
DECODER_D = 128
DECODER_MLP_MULT = 2
N_QUERY = 1
NUM_CLASSES = 1000

batch_size = 32  # per-device
base_lr = 5e-4
warmup_epochs = 5.0
weight_decay = 0.05
num_epochs = 100.0


def build_summformer(rngs: nnx.Rngs) -> QueryClassifierHead:
    emb_cfg = StackConfig(n_layers=1, d_model=EMBED_D, n_heads=1, window=SELF_WINDOW,
                           compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=CONTEXT_LEN, vocab_size=VOCAB_SIZE, rngs=rngs)

    chain = tuple(
        ChainStageConfig(stride=STRIDE, n_layers=1, d_model=STAGE_DIMS[i],
                          n_heads=max(1, STAGE_DIMS[i] // 64), window=SELF_WINDOW)
        for i in range(N_STAGES)
    )
    encoder = Encoder(StackConfig(n_layers=0, d_model=EMBED_D, n_heads=1, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32),
                       chain, context_len=CONTEXT_LEN, output_d_model=DECODER_D, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in range(N_STAGES))
    decoder_cfg = StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                               mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES, n_query=N_QUERY, rngs=rngs)
