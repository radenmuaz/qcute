"""summformer_jax.train.py config: window ablation vs. thin512_win2_allfuse8.py -- identical
(d_model=512, n_heads=8, n_layers=8, fuse stage after EVERY layer, code_n_layers=1,
source_index=-1, 131.4M params) except window=16 applied UNIFORMLY to every attention surface
(was 2):
  - main trunk self-attention (main_window)
  - each fuse stage's own code-LM self-attention (code_part's 4th element, code_window)
  - each fuse stage's trunk<->code-LM cross-attention (the (stride, window) pair)

Part of a 4-way window sweep {2,4,8,16} to see the effect of window width at maximal fuse-stage
density. Not swept/tuned beyond the window value itself.

    uv run python summformer_jax/lm/train.py --config configs/thin512_win16_allfuse8.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 512
n_heads = 8
n_layers = 8
sequence_length = 1024
main_window = 16
fuse_stages = tuple(
    ((-1, i), (2, 16), (1, None, None, 16)) for i in range(1, 9)
)
batch_size = 4
total_batch_size = 524288
