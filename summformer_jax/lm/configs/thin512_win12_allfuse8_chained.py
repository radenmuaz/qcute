"""Window-ablation sibling of thin512_win16_allfuse8_chained.py -- WINDOW=12 (between the
confirmed receptive-field minimum ~7-10 and the baseline's generous 16), part of a window sweep
(8/12/16/32) to find the pareto-optimal self-attn window (effectively the KV cache size) for this
stride=2/8-stage/L=1024 shape. Cross-attn window matches self-attn window (not cumprod/-1
auto-derive) -- self-attn is the load-bearing parameter for receptive field, cross-attn just needs
to deliver signal somewhere self-attn can relay from (see CrossAttnSpec docstring in
summformer.py). use_sink left at default True (see thin512_win8_allfuse8_chained.py's docstring
for why use_sink=False is unsafe for this causal-AR-decoder-with-cross-attn topology).

    uv run python summformer_jax/lm/train.py --config configs/thin512_win12_allfuse8_chained.py --data-dir data/fineweb-edu-10B
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
WINDOW = 12
STRIDE = 2
CROSS_WINDOW = 12
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
