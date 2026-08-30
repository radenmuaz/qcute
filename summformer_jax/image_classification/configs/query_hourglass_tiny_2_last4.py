"""Speedup variant of query_hourglass_tiny_2.py: cross-attends ONLY to the last 4 encoder chain
outputs (stages 4-7, n_blocks=147/36/9/2 at cum_stride=1024/4096/16384/65536) instead of all 8.
Stages 0-3 (n_blocks=37632/9408/2352/588) are skipped entirely from cross-attn -- their large K/V
projections (wk/wv over the full stage output, independent of n_query) were confirmed the dominant
cost in query_hourglass_tiny_2 (~3.08 of 6.60 GFLOPs, see chat 2026-08-30), even though windowing
their cross-attn barely helped (windowed_cross_attention still projects full K/V before gathering).
Rationale for dropping them outright rather than windowing: each stage in the chain already
self-attended over (and thus folded information from) everything below it before being pooled into
the next stage, so by stage 4 the upper layers already carry compressed signal originating from
stages 0-3 -- explicit direct cross-attn to the earliest, largest, priciest stages is redundant
with what chaining already propagates upward, not the only path to that information.

Same STRIDE=4/8-stage/hourglass-width shape as query_hourglass_tiny_2.py -- ONLY the cross-attn
source set differs (CROSS_STAGES). Encoder itself is unchanged (still computes all 8 stages, since
chaining requires stage i's output to produce stage i+1 -- only the DECODER's cross-attn selection
changes, which is where the K/V projection cost actually lives).

    uv run python summformer_jax/image_classification/train.py --config configs/query_hourglass_tiny_2_last4.py --shard-dir /dev/shm/imagenet_raw
"""
import jax.numpy as jnp
from flax import nnx

from summformer import Embedder, Encoder, Decoder, QueryClassifierHead, StackConfig, ChainStageConfig, CrossAttnSpec

IMAGE_SIZE = 224
VOCAB_SIZE = 256
CONTEXT_LEN = IMAGE_SIZE * IMAGE_SIZE * 3

EMBED_D = 8
STRIDE = 4
N_STAGES = 8
THIN_D, FAT_D = 32, 256
STAGE_DIMS = [THIN_D] * (N_STAGES // 2) + [FAT_D] * (N_STAGES // 2)  # 4 thin, 4 fat
SELF_WINDOW = 8
DECODER_D = 128
DECODER_MLP_MULT = 2
N_QUERY = 1
NUM_CLASSES = 1000
CROSS_STAGES = (4, 5, 6, 7)  # only cross-attend to the last 4 (coarsest) encoder outputs

batch_size = 32
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

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in CROSS_STAGES)
    decoder_cfg = StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                               mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES, n_query=N_QUERY, rngs=rngs)
