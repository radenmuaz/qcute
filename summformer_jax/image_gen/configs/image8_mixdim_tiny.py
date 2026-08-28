"""Tiny mixed-dim debug config (resolution=8, seq_len=192) for fast inference/locality checks.

    uv run python summformer_jax/image_gen/inference.py --config summformer_jax/image_gen/configs/image8_mixdim_tiny.py --resolution 8 --prompt-len 180 --gen-tokens 12
"""
pos_method = "rope"
vocab_size = 256
d_model = 16
n_heads = 2
n_layers = 2
main_window = 8
context_len = 192
fuse_stages = ((1, 8, None, 1, 0, 32, 4),)
batch_size = 2
