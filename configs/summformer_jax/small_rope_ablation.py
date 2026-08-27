"""summformer_jax.train_summformer config: ablation against configs/gpt2_jax/small_rope_default.py's
GPT2-small baseline. Same total_batch_size (524288) and grad_accum formula -- B=4,
total_batch_size=524288, seq_len=1024, n_devices=4 -> grad_accum_steps=524288//(4*1024*4)=32
(conservative per-device B, same reasoning as the medium ablation config).

d_model=768/n_heads=12 match GPT2-small exactly; n_layers=1 + Ks=(2,2,2) (n_fuse=3) chosen so
effective depth (~n_layers*(1+2*n_fuse)=7) and param count (~126.8M) land close to GPT2-small's
12 layers / ~123.6M params.

    uv run python summformer_jax/train_summformer.py --config configs/summformer_jax/small_rope_ablation.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "2,2,2"
d_model = 768
n_heads = 12
n_layers = 1
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
