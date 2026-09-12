"""Fork of cifar10_zorder_s4_sampler_p05_linears.py (chat 2026-09-12) -- SAME byte_group=3
(one pixel per position) and group_head_type="linears", but traversal="raster" instead of
"zorder" -- the control for isolating the traversal-order variable across the
linears/ar/diffusion x raster/zorder grid (this run differs from cifar10_zorder_s4_sampler_p05
in byte_group=3 too, so it isn't a pure zorder-only isolation against the original p05 --
compare against cifar10_zorder_s4_sampler_p05_linears for the raster-vs-zorder comparison
specifically). Same architecture/optimizer/cascade_rollout_prob/epochs_per_phase as p05
otherwise.

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_zorder_s4_sampler_p05_linears_raster.py
"""

run_name = "cifar10_zorder_s4_sampler_p05_linears_raster"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 65536 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5

byte_group = 3                         # level0: one pixel (R,G,B) per position
group_head_type = ("linears", "linears", "linears", "linears", "linears")
traversal = "raster"

# --- training ---
batch_size = 16
epochs_per_phase = 1000
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
