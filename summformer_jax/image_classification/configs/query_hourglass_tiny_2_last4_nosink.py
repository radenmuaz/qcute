"""Same as query_hourglass_tiny_2_last4.py but use_sink=False everywhere (embedder, Encoder's
per-stage transformers via layers_cfg.use_sink, decoder) -- testing whether dropping the
zero-KV attention sink (an extra concat + softmax column on EVERY attention call) reduces the
per-step dispatch overhead identified 2026-08-30 as the actual step-time bottleneck (measured
MFU ~6%, dt_ms roughly constant whether the model does 2.65 or 6.60 GFLOPs -- so cutting small
per-call fixed costs, not matmul FLOPs, is the lever that matters here).

    uv run python summformer_jax/image_classification/train.py --config configs/query_hourglass_tiny_2_last4_nosink.py --shard-dir /dev/shm/imagenet_raw
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
CROSS_STAGES = (4, 5, 6, 7)

batch_size = 32
base_lr = 5e-4
warmup_epochs = 5.0
weight_decay = 0.05
num_epochs = 100.0


def build_summformer(rngs: nnx.Rngs) -> QueryClassifierHead:
    emb_cfg = StackConfig(n_layers=1, d_model=EMBED_D, n_heads=1, window=SELF_WINDOW,
                           compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, use_sink=False)
    embedder = Embedder(emb_cfg, context_len=CONTEXT_LEN, vocab_size=VOCAB_SIZE, rngs=rngs)

    chain = tuple(
        ChainStageConfig(stride=STRIDE, n_layers=1, d_model=STAGE_DIMS[i],
                          n_heads=max(1, STAGE_DIMS[i] // 64), window=SELF_WINDOW)
        for i in range(N_STAGES)
    )
    encoder = Encoder(StackConfig(n_layers=0, d_model=EMBED_D, n_heads=1, compute_dtype=jnp.bfloat16,
                                   param_dtype=jnp.float32, use_sink=False),
                       chain, context_len=CONTEXT_LEN, output_d_model=DECODER_D, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in CROSS_STAGES)
    decoder_cfg = StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                               mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, use_sink=False)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES, n_query=N_QUERY, rngs=rngs)
