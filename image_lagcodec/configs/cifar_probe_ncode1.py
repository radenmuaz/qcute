"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar_probe_ncode1.py
"""
# Fork of cifar_probe_labelreg1.py: decoder_ncodes=1 (was 16) -- each pardec group predicts a
# single code, removing the artificial floor on recursion granularity (decoder_ncodes=16 at
# stride=4 forces Kspan=16*4=64 raw bytes per group; the eventual recursive/shared-depth design
# needs to be able to bottom out at the single finest block, down to 1x1 pixel at stride=1 in a
# future config -- decoder_ncodes=1 is the decode-side prerequisite for that, independent of stride).
# ncodes_window set to an EXPLICIT max (=n_blocks=256 for level0 at stride=4 on 32x32 CIFAR), not -1
# -- with decoder_ncodes=1, n_groups=n_blocks=256 (16x more than the labelreg1 probe's 16), so -1's
# auto-clamp-to-n_groups behavior and an explicit 256 land on the same number here; explicit is
# deliberate given how memory-sensitive this config already is (see cost note below), not trusting
# the sentinel blindly.
#
# Memory: B2 = batch_size * n_groups. labelreg1's tested config was G=16,Wg=64,n_groups=16,
# batch_size=8 -> B2=128, B2*Wg=8192. This config: G=1,Wg=256,n_groups=256 -> B2=256*batch_size,
# B2*Wg=65536*batch_size -- at batch_size=1 that's already ~8x labelreg1's cost. remat_level=True
# (checkpoints the whole decoder stack across the B2-batched call) is load-bearing here, not
# optional. No existing mechanism chunks/scans over n_groups itself (stream_chunks only affects the
# window/mask, not the B2 batch size) -- if this OOMs even at batch_size=1, the next lever is
# lowering ncodes_window below true max, not batch_size (already at floor).
#
# CIFAR has 50k train images -- should be enough data to support learning at eventual 1-pixel
# (24-bit RGB) finest granularity without data starvation; this config doesn't change stride yet,
# just proves decoder_ncodes=1 works at the current stride=4 level0 before combining with stride=1.

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
entropy_weight = 0.0  # kept off, matches labelreg1's finding (was fighting label_reg_weight)
label_reg_weight = 1.0

from image_lagcodec.run_lagcodec import rgb_label_fn_jax
label_fn = rgb_label_fn_jax

strides = (4,)
attn_window = (256,)
decoder_ncodes = (1,)  # was 16 -- see module docstring
ncodes_window = (256,)  # was 4 -- explicit max (=n_blocks), not -1; see module docstring.
# Fallback if this OOMs even at batch_size=1 (next lever, since batch_size is already at floor):
# ncodes_window = (64,)  # matches the encoder's own attn_window=256 more loosely (Wg=64*G=64*1=64
# # blocks of decoder context vs the encoder's 256-byte-position window -- same order of magnitude,
# # not an exact FLOP/mem match since encoder self-attn and decoder pardec have different per-token
# # costs, but a reasonable first cut if 256 is too much)
# ncodes_window = (16,)  # more conservative, matches decoder_ncodes=16's old window*G=4*16=64 in
# # spirit at a smaller absolute size -- try this if 64 also OOMs
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True  # load-bearing here, not just a convenience -- see memory note above
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
batch_size = 1  # was 8 -- ~8x per-sample cost increase from G=1,Wg=256 (see memory note above);
# already at the floor, next OOM lever is ncodes_window, not this
val_batch_size = 4  # no grad, some headroom vs train batch_size
level_steps = (int(3e4),)
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
