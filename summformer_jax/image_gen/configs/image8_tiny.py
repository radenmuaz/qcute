"""Tiny debug config -- resolution=8 (seq_len=192) -- isolates correctness from memory scale.

    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image8_tiny.py --resolution 8 --steps 5
"""
pos_method = "rope"
vocab_size = 256
d_model = 32
n_heads = 4
n_layers = 2
main_window = 12
context_len = 192
fuse_stages = (((0, 1), (8, -1), (1,)),)
batch_size = 2
