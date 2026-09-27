"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/imagenet64_res1.py
"""
# ImageNet64 fork of cifar_res_full1.py (the fixed/updated singleton run_lagcodec_res.py config) --
# same architecture/loss settings, swapping in ImageNet64 data settings (img_size=64, dataset,
# data_root on tmpfs, multihost=True for tpu34's 2 hosts) in place of cifar_res_full1.py's tpu2/tpu1
# CIFAR-10 settings. strides=(4,4,4) still divides 64 exactly (4^3=64), so the level structure is
# unchanged from the CIFAR config. Replaces the old run_lagcodec.py (imagenet64_parfullcausal1)
# 2-level, non-singleton, gumbel-quantize_mode config on tpu34 -- that run's quantize_mode="gumbel"
# is the "proven good" reference this config's own quantize_mode="gumbel" carries forward.

# --- data ---
img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
multihost = True

# --- model ---
codelm_d_model = 512
codelm_n_layers = 4
codelm_n_heads = 2
codelm_n_kv_heads = None

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0

label_fn = "rgb_label_fn_jax"

strides = (4, 4, 4)
attn_window = 256
upsampler_ncodes = (16, 16, 16)
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 4
remat_level = True

additive_drop_loss = False
downsampler_d_model = 512
downsampler_n_layers = 4
downsampler_n_heads = 8
downsampler_n_kv_heads = 8
downsampler_window = 4
downsampler_remat = True
upsampler_d_model = 512
upsampler_n_layers = 4
upsampler_n_heads = 8
upsampler_n_kv_heads = 8
upsampler_window = 4
upsampler_remat = True
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
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 4       # per-host, matches imagenet64_parfullcausal1's batch_size=4 (larger 64px images)
val_batch_size = 4
level_epochs = (1, 1, 5)  # epoch-based (not level_steps) -- ImageNet64 is much bigger than CIFAR-10,
# matches imagenet64_parfullcausal1's epoch-based scheduling; last (top) level gets more epochs
seed = 0
train_subset_n = None
val_subset_n = 512
gen_eval_every_epoch = 0.25  # matches imagenet64_parfullcausal1
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
