"""Speedup variant of query_hourglass_tiny_2.py: drops force_dense on the two EARLIEST cross-attn
specs (stage0: 37632 code tokens, stage1: 9408 code tokens) -- those two stages' K/V projections
dominate query_hourglass_tiny_2's FLOPs (dense cross-attn projects wk/wv over the FULL code
sequence at every stage even though only 1 query token consumes it). Uses windowed cross-attn
instead, window=-1 (auto-derives to that stage's own cum_stride -- 4 and 16 respectively), which
CrossAttnSpec's docstring confirms is the mathematically-minimum window guaranteeing every decoder
query sees its nearest code token (grid-spaced by construction) -- so coverage is unaffected, only
K/V projection cost drops from O(n_blocks) to O(n_query * n_gather) = O(1 * ceil(window/stride)+1).
Stages 2-7 (2352 tokens and below) stay force_dense -- already cheap, not worth the added complexity.

    uv run python summformer_jax/image_classification/train.py --config configs/query_hourglass_tiny_2_windowed.py --shard-dir /dev/shm/imagenet_raw
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
WINDOWED_STAGES = {0, 1}  # cross-attn windowed instead of force_dense for these encoder outputs

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

    cross = tuple(
        CrossAttnSpec(dst=i, encoder_output=i, force_dense=(i not in WINDOWED_STAGES))
        for i in range(N_STAGES)
    )
    decoder_cfg = StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                               mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES, n_query=N_QUERY, rngs=rngs)
