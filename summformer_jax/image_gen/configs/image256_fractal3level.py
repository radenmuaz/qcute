"""SKETCH config, not runnable yet -- see image256_fractal2level.py's docstring for the same
missing-dataset caveat (applies identically here).

Ballpark 3-level analog of Fractal Generative Models (arxiv 2502.17437)'s recursive patch grid --
same N=196608 raster RGB bytes, same zero-code-edit approach (main_window + fuse_stages), but two
independent fuse-stages at two different strides instead of one, loosely mirroring Fractal's
4-level scheme's multi-scale spirit (this repo's fuse-stages pool independently from the trunk,
not a literal nested/recursive tree the way Fractal's g1->g2->g3->g4 chain does -- an honest
structural difference, not an oversight).

Level mapping (not exact, see caveats):
  - trunk (level 0): main_window=12, same local RGB/pixel granularity as the 2-level config.
  - fuse_stage 1 (inserted early, layer 2), K=48 (=4x4 subpatch x 3 channels) -> code length
    196608/48=4096, matching Fractal's g2 (16 subpatches/patch x 256 patches = 4096) exactly.
  - fuse_stage 2 (inserted late, layer 6), K=768 (=16x16 patch x 3 channels) -> code length
    196608/768=256, matching Fractal's g1 length exactly, same as the 2-level config's stage.

Same source_index=0 point-sampling caveat as the 2-level config applies to BOTH stages here --
this is still the naive/cheap baseline arm, not the intended final design. Also still unverified
at any scale -- overfit10k-equivalent smoke test first.

    uv run python summformer_jax/lm/train_summformer_v2.py --config configs/summformer_jax_v2/image256_fractal3level.py
"""
pos_method = "rope"
dataset_dir = "data/image_bytes_256"  # TODO: not yet prepared, see image256_fractal2level.py
vocab_size = 256
d_model = 512
n_heads = 8
n_layers = 8
main_window = 12
sequence_length = 196608
fuse_stages = ((2, 48, None, 1, 0), (6, 768, None, 2, 0))
batch_size = 1
total_batch_size = 196608
