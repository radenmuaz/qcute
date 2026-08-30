"""summformer_jax.train_summformer_v2 config: v2 architecture (model_summformer_v2.py), small
scale, FLOPS-matched variant. Ks=(2,2) -- 2 fuse stages, same rate as small_rope_v2_parammatch.py,
but main_layers=9 (fuse-stages at insert_after=(4,9)) instead of 2.

Verified via direct construction+FLOPs (2026-08-28, tpu7), swept against gpt2-small
(123,689,472 params, 294,319,751,168 FLOPs): main_layers=9 gives 0.977x FLOPs (closest match
WITHOUT overshooting -- main_layers=10 gives 1.038x, which does overshoot), at the cost of
169,413,888 params (+37.0% -- NOT param-matched; matching both simultaneously isn't possible at
this fuse-stage count, see small_rope_v2_parammatch.py for the param-matched variant instead).

    uv run python summformer_jax/lm/train_summformer_v2.py --config configs/summformer_jax_v2/small_rope_v2_flopsmatch.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 768
n_heads = 12
n_layers = 9
sequence_length = 1024
fuse_stages = (((-1, 4), (2, -1), (1,)), ((-1, 9), (2, -1), (1,)))
batch_size = 4
total_batch_size = 524288
