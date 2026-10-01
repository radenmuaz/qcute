"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/imagenet64_res_4.py
"""
# --- data ---
img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
# multihost = True
multihost = False

# --- model ---
# remat_level = True
share_across_levels = False
# share_downsampler_upsampler_lm = True
remat = True
codelm_d_model = 256
codelm_n_layers = 2
codelm_n_heads = 2
codelm_n_kv_heads = 2

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 0.1
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 0.1

label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = (4, 4, 4, 4,)
# attn_window = 1024
attn_window = 4096
upsampler_ncodes = 1
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0


downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
# downsampler_remat = True

# upsampler_d_model = downsampler_d_model
# upsampler_n_layers = downsampler_n_layers
# upsampler_n_heads = downsampler_n_heads
# upsampler_n_kv_heads = downsampler_n_kv_heads
# upsampler_window = downsampler_window
# upsampler_remat = downsampler_remat

# downsampler_d_model = 1024
# downsampler_n_layers = 4
# downsampler_n_heads = 8
# downsampler_n_kv_heads = 8
# downsampler_window = 2
# downsampler_remat = True
upsampler_d_model = 1024
upsampler_n_layers = 4
upsampler_n_heads = 4
upsampler_n_kv_heads = 4
upsampler_window = 1
# upsampler_remat = True

use_codelm_bos = False
# codelm_bos_prob = 0.8
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.8
quantize_drop = 0.8

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
codelm_token_head = "ar"   # CodeLM NTP/free-run head: autoregressive digits
pardec_token_head = "ar"    # downsampler/upsampler digit head: autoregressive digits
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 2       # per DEVICE; tpu34 = 2 hosts x 4 devices -> global batch 64 (16 OOMs: 20.4G program vs 16G free)
val_batch_size = 2
# level_epochs = (1, 1, 5)
# Approximate step equivalents (tpu34 v4-16: batch_size 8 x 4 local devices x 2 hosts = global batch 64;
# ImageNet64 train = 1,281,167 imgs -> 1 epoch = 1,281,167 / 64 = 20,018 steps):
#   (1, 1, 5) epochs  ~=  (20018, 20018, 100090) steps  (140,126 total)
# To schedule by steps instead, comment out level_epochs above and uncomment (level_epochs and
# level_steps are mutually exclusive; steps-per-epoch scales with batch_size / n devices / n hosts):
level_steps = (0, 0, 0, 1_000_000)
seed = 0
train_subset_n = None
val_subset_n = 512
# gen_eval_every_epoch = 0.25   # ~= every 5,005 steps at global batch 64
gen_eval_every_step = 10_000   # uncomment (and comment gen_eval_every_epoch) to eval by steps
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
# lr_min_step = int(100e3)
# lr_min_step = int(200e3)
warmup_steps = 10_000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
