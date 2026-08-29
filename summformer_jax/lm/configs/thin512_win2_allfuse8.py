"""summformer_jax.train.py config: v2 architecture (summformer.py), stress test of the FuseStage
mechanism itself rather than a paper-matched ablation. d_model=512, n_layers=8 (down from the
first pass's n_layers=12 -- 171.3M params, +38.5% vs. gpt2-small's 123.7M -- to bring this closer
to gpt2-small param scale: 131.4M params, +6.2%), a fuse stage inserted after each of the 8 main
layers (8 total stages -- max insertion density this architecture supports at this depth),
code_n_layers=1 each, source_index=-1 (recursive, pools from the layer that just ran). Not
swept/tuned beyond the window value itself -- a single smart-guess config to see whether the
mechanism holds up (trains stably, doesn't blow up) under maximal fuse-stage density + narrow
window, not a scientific ablation.

window=2 applied UNIFORMLY to every attention surface in the model (part of a 4-way sweep
{2,4,8,16}, see thin512_win{4,8,16}_allfuse8.py):
  - main trunk self-attention (main_window)
  - each fuse stage's own code-LM self-attention (code_part's 4th element, code_window)
  - each fuse stage's trunk<->code-LM cross-attention (the (stride, window) pair)

    uv run python summformer_jax/lm/train.py --config configs/thin512_win2_allfuse8.py
"""
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
d_model = 512
n_heads = 8
n_layers = 8
sequence_length = 1024
main_window = 2
fuse_stages = tuple(
    ((-1, i), (2, 2), (1, None, None, 2)) for i in range(1, 9)
)
batch_size = 4
total_batch_size = 524288
