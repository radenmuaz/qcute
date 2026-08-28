"""summformer_jax.train_summformer_v2 config: v2 architecture (model_summformer_v2.py), medium
scale, PARAM-matched variant. Ks=(2,2,2,2) -- 4 fuse stages. main_layers=10, fuse-stages at
insert_after=(2,5,8,10). code_n_layers=1 each, source_index=-1 (recursive).

Verified via direct construction+FLOPs (2026-08-28, tpu7), swept against gpt2-medium
(353,822,720 params, 833,534,689,280 FLOPs): main_layers=10 gives 329,764,864 params (-6.8%,
closest match WITHOUT overshooting -- main_layers=12 gives +0.3%, which does overshoot) at 0.684x
FLOPs (not FLOPs-matched -- see medium_rope_v2_flopsmatch.py for that variant).

    uv run python summformer_jax/train_summformer_v2.py --config configs/summformer_jax_v2/medium_rope_v2_parammatch.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 1024
n_heads = 16
n_layers = 10
sequence_length = 1024
fuse_stages = ((2, 2, None, 1, -1), (5, 2, None, 1, -1), (8, 2, None, 1, -1), (10, 2, None, 1, -1))
batch_size = 4
total_batch_size = 524288
