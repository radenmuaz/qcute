"""Ablation of l8d512_seqlen1024.py: chain stages' own self-attention RoPE positions are
`local_pos * cum_stride` (the stage's absolute original-sequence position) instead of the default
plain `local_pos` (0,1,2,... over the pooled sequence, oblivious to the stage's timescale) --
`ChainStageConfig.rope_scale_by_stride=True`, opt-in flag, no effect on any other config. See
summformer_jax/summformer.py's Encoder.__call__ and chat 2026-09-02.

    uv run python summformer_jax/lm/train.py --config summformer_jax/lm/configs/sweep_seqlen_small_1/l8d512_seqlen1024_ropescale.py
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
STRIDE = 2
SEQUENCE_LENGTH = 1024
WEIGHT_TIE = False
sequence_length = SEQUENCE_LENGTH  # lowercase alias so train.py's config loader picks it up

batch_size = 32
total_batch_size = 524288


def build_summformer(rngs: nnx.Rngs) -> ARHead:
    emb_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, force_dense=True,
                           use_sink=False, use_flash=True, use_real_flash=True,
                           pos_method=pos_method, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    embedder = Embedder(emb_cfg, context_len=SEQUENCE_LENGTH, vocab_size=vocab_size, rngs=rngs)

    chain = tuple(ChainStageConfig(stride=STRIDE, n_layers=1, force_dense=True, rope_scale_by_stride=True)
                  for _ in range(N_STAGES))
    encoder = Encoder(StackConfig(n_layers=0, d_model=D_MODEL, n_heads=N_HEADS, pos_method=pos_method,
                                   use_sink=False, use_flash=True, use_real_flash=True,
                                   compute_dtype=jnp.bfloat16, param_dtype=jnp.float32),
                       chain, context_len=SEQUENCE_LENGTH, rngs=rngs)

    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in range(N_STAGES))
    decoder_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, force_dense=True,
                               use_sink=True, use_remat=True, use_real_flash=True,
                               pos_method=pos_method, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=SEQUENCE_LENGTH, rngs=rngs)

    model = SummFormer(embedder, encoder, decoder)
    return ARHead(model, vocab_size=vocab_size, weight_tie=WEIGHT_TIE, rngs=rngs)
