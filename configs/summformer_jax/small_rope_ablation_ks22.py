"""summformer_jax.train_summformer config: third small ablation variant -- same n_fuse=2/n_layers=2
(effective_depth=10, identical to small_rope_ablation_ks44.py) but Ks=(2,2) instead of Ks=(4,4).
Isolates the per-stage pooling factor as the only changed variable (params are IDENTICAL to ks44's
148.2M -- the K value doesn't affect param count, only n_fuse/n_layers do); FLOPs ratio is slightly
higher here (0.912x vs ks44's 0.846x, finer pooling costs a bit more compute) but still under
baseline. Motivation: ks44 (Ks=4,4) finished worse than the gpt2-small baseline (22.58 vs 20.83
val PPL) -- testing whether Ks=4's coarser per-stage pooling was losing too much fine-grained
information vs. the gentler Ks=2 pooling, at matched effective_depth. See
docs/status_tpu.md's "ks22 vs ks44" note.

d_model=768/n_heads=12 match GPT2-small exactly, same as every other small ablation variant.

    uv run python summformer_jax/train_summformer.py --config configs/summformer_jax/small_rope_ablation_ks22.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "2,2"
d_model = 768
n_heads = 12
n_layers = 2
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
