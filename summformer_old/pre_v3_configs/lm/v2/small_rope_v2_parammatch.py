"""summformer_jax.train_summformer_v2 config: v2 architecture (model_summformer_v2.py), small
scale, PARAM-matched variant. Ks=(2,2) -- 2 fuse stages. main_layers=2, fuse-stages at
insert_after=(1,2) (spread across the main stack, not stacked). code_n_layers=1 each,
source_index=-1 (recursive: each stage pools whatever `x` currently is).

Verified via direct construction+FLOPs (2026-08-28, tpu7), swept against gpt2-small
(123,689,472 params, 294,319,751,168 FLOPs): main_layers=2 gives 119,798,784 params (-3.1%,
closest match WITHOUT overshooting -- main_layers=3 gives +2.6%, which does overshoot) at 0.549x
FLOPs (not FLOPs-matched -- see small_rope_v2_flopsmatch.py for that variant, which does not
param-match in return: matching both simultaneously isn't possible at this fuse-stage count).

    uv run python summformer_jax/lm/train_summformer_v2.py --config configs/summformer_jax_v2/small_rope_v2_parammatch.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 768
n_heads = 12
n_layers = 2
sequence_length = 1024
fuse_stages = (((-1, 1), (2, -1), (1,)), ((-1, 2), (2, -1), (1,)))
batch_size = 4
total_batch_size = 524288
