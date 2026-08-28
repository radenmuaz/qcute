"""SKETCH config, not runnable yet -- no image-byte dataset_dir exists in this repo (needs a
prep script that flattens 256x256x3 RGB images into raster-order uint8 byte sequences, analogous
to scripts/jax/generate_pathfinder.py but emitting actual pixel bytes, not maze labels).

Ballpark 2-level analog of Fractal Generative Models (arxiv 2502.17437)'s recursive patch grid,
built entirely from model_summformer_v2.py's existing levers (main_window + one fuse_stages
entry) -- zero code edits. Sequence length N = 256*256*3 = 196608 raster RGB bytes.

Level mapping (not exact, see caveats):
  - trunk (level 0): main_window=12 ~ a few pixels' worth of bytes -- local RGB/pixel texture.
    RGB-channel-last ordering is already the byte-level AR order, no extra level needed for it.
  - one fuse_stage, K=768 (=16x16 patch x 3 channels) -> code sequence length 196608/768=256,
    matching Fractal's own top-level g1 length exactly.

Caveat (see chat discussion): source_index=0 makes this a point-sampling summary (one raw byte
every 768 positions), not a true patch-aggregate -- cheapest to test, but a weak/naive baseline.
Faithful aggregation via window x depth alone would need insert_after*window >= 768 (~64 layers
at window=12), impractical at this depth; treat this config as the lower-bound ablation arm, not
the intended final design. Also unverified at any scale -- run an overfit10k-equivalent (small
image subset) smoke test before trusting the shape choices below.

    uv run python summformer_jax/lm/train_summformer_v2.py --config configs/summformer_jax_v2/image256_fractal2level.py
"""
pos_method = "rope"
dataset_dir = "data/image_bytes_256"  # TODO: not yet prepared, see module docstring
vocab_size = 256
d_model = 512
n_heads = 8
n_layers = 6
main_window = 12
sequence_length = 196608
fuse_stages = (((0, 3), (768, None), (2,)),)
batch_size = 1
total_batch_size = 196608
