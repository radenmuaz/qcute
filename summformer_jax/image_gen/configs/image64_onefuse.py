"""Degenerate: trunk + exactly one fuse-stage (the finer K=8 stage from image64_mixdim.py, coarser
K=64 stage dropped) -- isolates the cost of a single cross-attn/code-LM stage, to see how much
image64_mixdim.py's 2-stage cost scales per added stage.

    uv run python summformer_jax/image_gen/scripts/bench_generation_speed.py --config summformer_jax/image_gen/configs/image64_onefuse.py --n-tokens 8
"""
pos_method = "rope"
vocab_size = 256
d_model = 128
n_heads = 4
n_layers = 3
main_window = 8
context_len = 12288
fuse_stages = (
    ((-1, 1), (8, None), (2, 256, 4)),
)
mtp_heads = 24
batch_size = 2
