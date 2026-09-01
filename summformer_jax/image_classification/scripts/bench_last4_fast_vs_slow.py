"""Bench summformer.py (rectangular skipped-queries pooling) vs summformer_slow.py (literal
subsample-then-square-attend pooling) at query_hourglass_tiny_2_last4.py's exact hyperparameters.
Reports HLO cost-analysis GFLOPs (forward pass only) and local wall-clock forward-pass timing.

    uv run python summformer_jax/image_classification/scripts/bench_last4_fast_vs_slow.py
"""
import importlib
import sys
import time

import jax
import jax.numpy as jnp
from flax import nnx

IMAGE_SIZE = 224
VOCAB_SIZE = 256
CONTEXT_LEN = IMAGE_SIZE * IMAGE_SIZE * 3
EMBED_D = 8
STRIDE = 4
N_STAGES = 8
THIN_D, FAT_D = 32, 256
STAGE_DIMS = [THIN_D] * (N_STAGES // 2) + [FAT_D] * (N_STAGES // 2)
SELF_WINDOW = 8
DECODER_D = 128
DECODER_MLP_MULT = 2
N_QUERY = 1
NUM_CLASSES = 1000
CROSS_STAGES = (4, 5, 6, 7)
BATCH_SIZE = 32


def build(mod, rngs):
    emb_cfg = mod.StackConfig(n_layers=1, d_model=EMBED_D, n_heads=1, window=SELF_WINDOW,
                               compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    embedder = mod.Embedder(emb_cfg, context_len=CONTEXT_LEN, vocab_size=VOCAB_SIZE, rngs=rngs)

    chain = tuple(
        mod.ChainStageConfig(stride=STRIDE, n_layers=1, d_model=STAGE_DIMS[i],
                              n_heads=max(1, STAGE_DIMS[i] // 64), window=SELF_WINDOW)
        for i in range(N_STAGES)
    )
    encoder = mod.Encoder(mod.StackConfig(n_layers=0, d_model=EMBED_D, n_heads=1,
                                           compute_dtype=jnp.bfloat16, param_dtype=jnp.float32),
                           chain, context_len=CONTEXT_LEN, output_d_model=DECODER_D, rngs=rngs)

    cross = tuple(mod.CrossAttnSpec(dst=i, encoder_output=i, force_dense=True) for i in CROSS_STAGES)
    decoder_cfg = mod.StackConfig(n_layers=N_STAGES, d_model=DECODER_D, n_heads=max(1, DECODER_D // 64),
                                   mlp_mult=DECODER_MLP_MULT, window=SELF_WINDOW,
                                   compute_dtype=jnp.bfloat16, param_dtype=jnp.float32)
    decoder = mod.Decoder(decoder_cfg, cross, context_len=CONTEXT_LEN, rngs=rngs)

    return mod.QueryClassifierHead(embedder, encoder, decoder, num_classes=NUM_CLASSES,
                                    n_query=N_QUERY, rngs=rngs)


def bench(mod_name: str):
    sys.path.insert(0, "summformer_jax")
    mod = importlib.import_module(mod_name)
    rngs = nnx.Rngs(0)
    model = build(mod, rngs)
    graphdef, state = nnx.split(model)

    tokens = jnp.zeros((BATCH_SIZE, CONTEXT_LEN), dtype=jnp.int32)

    def fwd(state, tokens):
        m = nnx.merge(graphdef, state)
        return m(tokens)

    jitted = jax.jit(fwd)
    lowered = jitted.lower(state, tokens)
    compiled = lowered.compile()

    cost = compiled.cost_analysis()
    gflops = cost.get("flops", float("nan")) / 1e9

    out = jitted(state, tokens)
    jax.block_until_ready(out)

    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        out = jitted(state, tokens)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    ms_per_call = (t1 - t0) / n * 1000

    del sys.path[0]
    for k in list(sys.modules):
        if k == mod_name:
            del sys.modules[k]

    return gflops, ms_per_call


if __name__ == "__main__":
    print(f"device: {jax.devices()}")
    print(f"config: query_hourglass_tiny_2_last4 (batch={BATCH_SIZE}, context_len={CONTEXT_LEN})")
    print()
    fast_gflops, fast_ms = bench("summformer")
    slow_gflops, slow_ms = bench("summformer_slow")

    print(f"{'':20s} {'GFLOPs (fwd)':>15s} {'ms/call':>12s}")
    print(f"{'summformer':20s} {fast_gflops:15.3f} {fast_ms:12.2f}")
    print(f"{'summformer_slow':20s} {slow_gflops:15.3f} {slow_ms:12.2f}")
    print()
    print(f"GFLOPs ratio (slow/fast): {slow_gflops / fast_gflops:.2f}x")
    print(f"wall-clock ratio (slow/fast): {slow_ms / fast_ms:.2f}x")
