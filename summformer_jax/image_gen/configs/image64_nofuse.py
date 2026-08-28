"""Degenerate: trunk only, fuse_stages=() -- isolates self-attn KV cache cost alone (no cross-attn,
no code-LM) from image64_mixdim.py, to see how much of its 674ms/token (TPU) / 1327ms/token (CPU)
generation cost is trunk self-attn vs. fuse-stage cross-attn/code-LM overhead.

    uv run python summformer_jax/image_gen/scripts/bench_generation_speed.py --config summformer_jax/image_gen/configs/image64_nofuse.py --n-tokens 8
"""
pos_method = "rope"
vocab_size = 256
d_model = 128
n_heads = 4
n_layers = 3
main_window = 8
context_len = 12288
fuse_stages = ()
mtp_heads = 24
batch_size = 2
