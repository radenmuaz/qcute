"""New-architecture (summformer.py, chained Encoder) replacement for thin512_win16_allfuse8.py.

Old config's `fuse_stages = tuple(((-1,i),(2,16),(1,None,None,16)) for i in range(1,9))` used
source_index=-1 with NO chaining -- confirmed a real bug this session (docs/status_tpu.md
2026-08-30): every one of the 8 fuse stages independently re-pooled from the trunk's current state
at stride=2, so strides never compounded (max reach ~a few hundred positions out of 1024, not the
2**8 the "8 stages of stride 2" framing implied). This config fixes that by making the 8 stages a
genuine CHAIN (ChainStageConfig, cum_stride compounds: 2,4,8,...,256) -- same d_model/n_heads/
n_layers/window=16/stride=2/n_stages=8 as the original (same param count ballpark, same "window
ablation at window=16" intent), but now the strides actually compound instead of8 independently
re-pooling the same span.

window=16 (self-attn, EVERY surface: Embedder, each chain stage's own transformer, Decoder) was
already checked safe for this exact stride=2/depth=8 shape: confirmed via
scripts/check_connectivity.py + scripts/check_chain_receptive_field.py (2026-08-30) that window=16
gives robust, seed-independent full-chain connectivity for an 8-stage/stride=2 pyramid (real
minimum found ~7-10, window=16 has comfortable margin, not sitting at a weight-fragile boundary).
Cross-attn window is set explicitly (=16, matching self-attn, not left at -1/auto-derive) per
README.md's "don't rely on -1 in real configs" guidance -- self-attn is what actually controls
receptive field (confirmed: cross-attn window=1 sufficed when self-attn was adequate), so 16 here
is deliberately generous/cheap rather than minimal.

NOTE: no train.py wired to this StackConfig/ChainStageConfig/CrossAttnSpec-based config format
exists yet (train.py still targets the old ConfigV2 format) -- this file exports ready-built
config objects (`build_summformer(rngs)` factory) for a future train.py to consume, not flat
scalars for the old loader. See summformer_jax/lm/smoke_test_new_arch.py for the exact
Embedder/Encoder/Decoder/ARHead composition pattern this follows.

    (once a new-architecture train.py exists:)
    uv run python summformer_jax/lm/train_v2.py --config configs/thin512_win16_allfuse8_chained.py
"""
import jax.numpy as jnp
from flax import nnx

from summformer import Embedder, Encoder, Decoder, SummFormer, ARHead, StackConfig, ChainStageConfig, CrossAttnSpec

pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
vocab_size = 50257  # GPT-2 BPE

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8          # decoder layers, matches old n_layers
N_STAGES = 8           # chain stages, matches old "allfuse8" (one fuse point per layer)
WINDOW = 16            # uniform self-attn window everywhere, matches old main_window/code_window
STRIDE = 2             # per-stage stride, matches old's (stride,window)=(2,16) pair
CROSS_WINDOW = 16      # explicit, NOT -1/auto-derive (see module docstring)
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

    # one cross-attn spec per decoder layer, matching old's "fuse stage after every layer"
    cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, window=CROSS_WINDOW) for i in range(N_STAGES))
    decoder_cfg = StackConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, window=WINDOW,
                               pos_method=pos_method, compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = Decoder(decoder_cfg, cross, context_len=SEQUENCE_LENGTH, rngs=rngs)

    model = SummFormer(embedder, encoder, decoder)  # encoder_embedder=None -> shared/self-referential
    return ARHead(model, vocab_size=vocab_size, weight_tie=WEIGHT_TIE, rngs=rngs)
