"""Same as query_hourglass_tiny_2_last4_nosink.py (use_sink=False) but ALSO use_flash=True --
dispatches sdpa_with_sink's no-sink branch to jax.nn.dot_product_attention (JAX's fused/flash-
capable kernel) instead of the manual einsum+softmax+einsum, everywhere use_sink=False already
applies (embedder, Encoder chain stages, decoder self-attn -- NOT decoder cross-attn, which is
still dense force_dense=True and uses sdpa_with_sink too, so flash DOES apply there as well since
use_sink=False is set on decoder_cfg). use_flash only takes effect when use_sink=False (see
sdpa_with_sink docstring in summformer.py) -- added 2026-08-30 specifically to test this,
opt-in/default-False, no effect on any other existing config.

Expected payoff is LOW by design here: flash attention's main benefit is avoiding O(T^2)
score-matrix materialization for long dense attention, but every attention call in this
architecture is already small (chunked self-attn: T=window=8, 2*w=16 keys; cross-attn: at most
S=147 keys at stage4) -- there's no large dense attention anywhere to speed up. Testing directly
rather than assuming, per explicit request 2026-08-30.

    uv run python summformer_jax/image_classification/train.py --config configs/query_hourglass_tiny_2_last4_flash.py --shard-dir /dev/shm/imagenet_raw
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
                           compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, use_sink=False, use_flash=True)
    embedder = Embedder(emb_cfg, context_len=CONTEXT_LEN, vocab_size=VOCAB_SIZE, rngs=rngs)

    chain = tuple(
        ChainStageConfig(stride=STRIDE, n_layers=1, d_model=STAGE_DIMS[i],
                          n_heads=max(1, STAGE_DIMS[i] // 64), window=SELF_WINDOW)
        for i in range(N_STAGES)
    )
    encoder = Encoder(StackConfig(n_layers=0, d_model=EMBED_D, n_heads=1, compute_dtype=jnp.bfloat16,
                                   param_dtype=jnp.float32, use_sink=False, use_flash=True),
                       chain, context_len=CONTEXT_LEN, output_d_model=DECODER_D, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in CROSS_STAGES)
    decoder_cfg = StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                               mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, use_sink=False, use_flash=True)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES, n_query=N_QUERY, rngs=rngs)
