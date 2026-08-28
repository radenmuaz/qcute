"""Small, actually-runnable test config for train.py/inference consistency checks -- 64x64x3
RGB raster bytes (seq_len=12288), not the 256x256 fractal-ballpark configs (those need d_model/
layer counts too large to smoke-test quickly). Single fuse-stage, K=192 (=8x8 patch x 3 channels,
scaled-down analog of the 256-config's 16x16x3=768 patch), code length 12288/192=64.

    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image64_test.py
"""
pos_method = "rope"
vocab_size = 256
d_model = 64
n_heads = 4
n_layers = 4
main_window = 12
context_len = 12288
fuse_stages = (((0, 2), (192, None), (1,)),)
batch_size = 2
