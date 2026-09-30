"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar_probe_labelreg1.py
"""
# Probe, not a real model: single level (level1 dropped from cifar_parfullcausal2), label_reg_weight
# turned ON (was 0.0 everywhere so far -- default_label_fn_jax/label_fn machinery already existed but
# was always inactive). Tests whether the encoder's code_head logits, once cross-entropy-supervised
# toward the real downsampled image's own byte-quantized value (not just free NTP self-supervision),
# actually make code_idx directly interpretable as a downsampled image -- i.e. whether
# scripts/harvest_level0_codes.py's code-as-RGB plot shows real structure instead of noise.
# n_phases = n_levels if top_level_trainable else n_levels-1 (run_lagcodec.py:3397) -- a single level
# needs top_level_trainable=True (strides[-1] != -1) just to get a training phase at all, which also
# means level0 keeps its real pardec decoder (bundled with top_level_trainable) -- so this also checks
# the aux loss doesn't wreck normal reconstruction while it's at it.

# --- model ---
img_size = 32

d_model = (256,)
n_layers = (4,)  # cheap encoder
n_heads = (2,)
n_kv_heads = (None,)
decoder_d_model = (1024,)
decoder_n_layers = (4,)
decoder_n_heads = (8,)
decoder_n_kv_heads = (8,)

code_vocab = (256,)
pq_chunks = (3,)
pq_dim = (64,)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.0  # was 1.0 -- testing whether entropy reg was competing against label_reg_weight
# (label_mse climbed steadily after an early drop: 54 at step200 -> 772 at step15000 while loss/acc
# kept improving normally -- entropy_weight pushes code_head toward a uniform marginal, which can
# directly fight label_reg_weight's push toward matching a specific, non-uniform pixel-value target)
label_reg_weight = 1.0  # was 0.0 -- the whole point of this probe

from image_lagcodec.run_lagcodec import rgb_label_fn_jax
label_fn = rgb_label_fn_jax  # was default_label_fn_jax (implicit) -- that one grayscales+bit-packs,
# degenerating to a red-only target for pq_chunks=3/code_vocab=256 (confirmed visually in
# samples_level0_step14000_codegrid.png); this keeps real per-channel RGB in the target

strides = (4,)
attn_window = (256,)  # force short local context -- encoder can't just globally copy the image,
# must genuinely aggregate into the code; also applies to the decoder's own self-attn (shared base)
decoder_ncodes = (16,)
ncodes_window = (4,)
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True
level_refine_window = 0
level_refine_passes = 1

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.9
quantize_drop = 0.9

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
token_dim = (64,)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(3e4),)  # single phase, no level1 -- shorter than parfullcausal2's combined 7e4
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 2000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(3e4)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
