"""Window-ablation sibling of thin512_win16_allfuse8_chained.py -- WINDOW=8 (near the confirmed
receptive-field minimum of ~7-10 for this stride=2/8-stage/L=1024 shape, see
scripts/check_connectivity.py sweep 2026-08-30: window=6,7,8 all still connected at this shape).
Same everything else (D_MODEL, N_LAYERS, N_STAGES, STRIDE, batch/optim settings) -- part of a
window sweep (8/12/16/32) to find the pareto-optimal self-attn window (effectively the KV cache
size) for this architecture: smaller window = cheaper/less memory but closer to the connectivity
margin, larger = more expensive but safer. use_sink left at default True -- confirmed 2026-08-30
that use_sink=False is UNSAFE for this causal-AR-decoder-with-cross-attn topology (early positions
structurally have zero valid cross-attn keys for a stage before its first code token exists ->
softmax over an all -inf row -> NaN, confirmed empirically). Only QueryClassifierHead's topology
(query positioned after the full sequence, guaranteeing all keys valid) is safe without the sink.
A self-attn-only/cross-attn-still-sinked split is a possible future ablation, not applied here.

    uv run python summformer_jax/lm/train.py --config configs/thin512_win8_allfuse8_chained.py --data-dir data/fineweb-edu-10B
"""
import jax.numpy as jnp
from flax import nnx

from summformer import Embedder, Encoder, Decoder, SummFormer, ARHead, StackConfig, ChainStageConfig, CrossAttnSpec

pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
vocab_size = 50257  # GPT-2 BPE

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
N_STAGES = 8
WINDOW = 8
STRIDE = 2
CROSS_WINDOW = 8
SEQUENCE_LENGTH = 1024
WEIGHT_TIE = False

batch_size = 16
total_batch_size = 524288


def build_summformer(rngs: nnx.Rngs) -> ARHead:
    emb_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, window=WINDOW,
                           pos_method=pos_method, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=SEQUENCE_LENGTH, vocab_size=vocab_size, rngs=rngs)

    chain = tuple(ChainStageConfig(stride=STRIDE, n_layers=1, window=WINDOW) for _ in range(N_STAGES))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D_MODEL, n_heads=N_HEADS, pos_method=pos_method,
                                   compute_dtype=jnp.bfloat16, param_dtype=jnp.float32),
                       chain, context_len=SEQUENCE_LENGTH, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, window=CROSS_WINDOW) for i in range(N_STAGES))
    decoder_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, window=WINDOW,
                               pos_method=pos_method, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=SEQUENCE_LENGTH, rngs=rngs)

    model = SummFormer(embedder, encoder, decoder)
    return ARHead(model, vocab_size=vocab_size, weight_tie=WEIGHT_TIE, rngs=rngs)
