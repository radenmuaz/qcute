"""New-architecture (summformer.py, chained Encoder) replacement for the archived
tiny_vit_like.py (summformer_old/pre_v3_configs/image_classification/) -- same target scale
(~ViT-Tiny param count, d_model=128, n_layers=16, byte-level 224x224x3 image, context_len=150528)
but with a genuinely chained 8-stage/stride-2 Encoder (cum_stride=256, matching the exact shape
this session's receptive-field investigation validated -- see summformer_jax/README.md) instead of
the archived config's per-layer non-chaining fuse_stages (confirmed a real bug: strides never
compounded, receptive field maxed out far short of the image).

window=32 (self-attn, every surface) is comfortably above the confirmed-safe minimum for this
exact stride=2/depth=8 shape (real minimum found ~7-10 via scripts/check_connectivity.py +
scripts/check_chain_receptive_field.py, not a weight-fragile margin). Cross-attn window is also
explicit (=32, not -1/auto-derive) per README's guidance -- self-attn is the load-bearing knob for
receptive field, cross-attn just needs to deliver signal somewhere self-attn can relay it from.

    (once a new-architecture image_classification train.py exists:)
    uv run python summformer_jax/image_classification/train_v2.py --config configs/tiny_vit_like_chained.py
"""
import jax.numpy as jnp
from flax import nnx

from summformer import Embedder, Encoder, Decoder, SummFormer, ClassifierHead, StackConfig, ChainStageConfig, CrossAttnSpec

IMAGE_SIZE = 224
VOCAB_SIZE = 256  # byte-level RGB
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 16       # decoder layers, matches archived config
N_STAGES = 8        # chain stages -- cum_stride=256, matches the validated 8-stage/stride-2 shape
STRIDE = 2
WINDOW = 32         # uniform self-attn window everywhere, safely above the ~7-10 confirmed minimum
CROSS_WINDOW = 32    # explicit, NOT -1/auto-derive
CONTEXT_LEN = IMAGE_SIZE * IMAGE_SIZE * 3
NUM_CLASSES = 1000
POOLING = "mean"     # "last" | "mean" | "bidirectional" -- see ClassifierHead docstring

batch_size = 32  # per-device
base_lr = 5e-4   # DeiT's own value @ batch=512, scale by batch/512 in train script
warmup_epochs = 5.0
weight_decay = 0.05
num_epochs = 100.0


def build_summformer(rngs: nnx.Rngs) -> ClassifierHead:
    emb_cfg = StackConfig(n_layers=2, d_model=D_MODEL, n_heads=N_HEADS, window=WINDOW,
                           compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=CONTEXT_LEN, vocab_size=VOCAB_SIZE, rngs=rngs)

    chain = tuple(ChainStageConfig(stride=STRIDE, n_layers=1, window=WINDOW) for _ in range(N_STAGES))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D_MODEL, n_heads=N_HEADS,
                                   compute_dtype=jnp.bfloat16, param_dtype=jnp.float32),
                       chain, context_len=CONTEXT_LEN, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i * 2, encoder_output=i, window=CROSS_WINDOW) for i in range(N_STAGES))
    decoder_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, window=WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    model = SummFormer(embedder, encoder, decoder)
    return ClassifierHead(model, num_classes=NUM_CLASSES, pooling=POOLING, rngs=rngs)
