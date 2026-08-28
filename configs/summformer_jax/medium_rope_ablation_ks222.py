"""summformer_jax.train_summformer config: medium counterpart to small_rope_ablation_ks22.py --
same n_fuse=3/n_layers=3 (effective_depth=21, identical to medium_rope_ablation_ks444.py) but
Ks=(2,2,2) instead of Ks=(4,4,4). Isolates the per-stage pooling factor as the only changed
variable vs. ks444 (params identical, ~367.6M -- the K value doesn't affect param count, only
n_fuse/n_layers do; FLOPs ratio should be slightly higher, finer pooling costs a bit more compute,
mirroring the small-scale ks22-vs-ks44 pattern). Motivation: same as ks22's -- testing whether
ks444's coarser per-stage pooling was losing too much fine-grained information vs. gentler
pooling, at matched effective_depth. See docs/status_tpu.md's "ks222 vs ks444" note.

d_model=1024/n_heads=16 match GPT2-medium exactly, same as every other medium ablation variant.
Launch AFTER medium_rope_ablation_ks444 finishes (same node, tpu5) -- not run concurrently with it.

    uv run python summformer_jax/train_summformer.py --config configs/summformer_jax/medium_rope_ablation_ks222.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
Ks = "2,2,2"
d_model = 1024
n_heads = 16
n_layers = 3
sequence_length = 1024
batch_size = 4
total_batch_size = 524288
