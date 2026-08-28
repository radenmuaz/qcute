"""summformer_jax.train_summformer_v2 config: v2 architecture (model_summformer_v2.py), medium
scale, FLOPS-matched variant. Ks=(2,2,2,2) -- 4 fuse stages, same rate as
medium_rope_v2_parammatch.py, but main_layers=18 (fuse-stages at insert_after=(4,9,14,18)) instead
of 10.

Verified via direct construction+FLOPs (2026-08-28, tpu7), swept against gpt2-medium
(353,822,720 params, 833,534,689,280 FLOPs): main_layers=18 gives 0.976x FLOPs (closest match
WITHOUT overshooting -- main_layers=19 gives 1.012x, which does overshoot), at the cost of
430,534,656 params (+21.7% -- NOT param-matched, see medium_rope_v2_parammatch.py for that
variant).

    uv run python summformer_jax/lm/train_summformer_v2.py --config configs/summformer_jax_v2/medium_rope_v2_flopsmatch.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 1024
n_heads = 16
n_layers = 18
sequence_length = 1024
fuse_stages = ((4, 2, None, 1, -1), (9, 2, None, 1, -1), (14, 2, None, 1, -1), (18, 2, None, 1, -1))
batch_size = 4
total_batch_size = 524288
