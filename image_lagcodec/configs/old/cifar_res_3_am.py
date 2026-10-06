"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/cifar_res_2.py
"""
# Fork of cifar_res_full1.py -- diagnostic ablation for the "generation too blurry" observation
# (cifar_res_full1's finished-run samples were structurally correct but blurry/regression-to-mean,
# unlike the OLD pre-refactor run_lagcodec.py/run_lagcodec_recursive.py's less-blurry output).
# Two changes from cifar_res_full1.py, isolating whether CodeLM's own free-run is "unaware" of
# what it's generating (no anchor/conditioning signal at position 0):
#   1. use_codelm_bos=True, codelm_bos_prob=1.0 (ALWAYS substitute the learned bos/anchor token at
#      position 0, no probabilistic drop back to real content -- "no drop" per the user's framing).
#   2. strides=(4,) -- ONE level only (single downsample-by-4/upsample-by-4 pair), removing the
#      3-level cascade entirely as a confound while isolating the blur's root cause.
# level_steps bumped to the full original 70k budget (single phase now, not split 10k/10k/50k)
# so this gets comparable total compute to cifar_res_full1.py's cascade.

# --- model ---
img_size = 32

# share_downsampler_upsampler_lm = True
share_across_levels = False
codelm_d_model = 512
codelm_n_layers = 2
codelm_n_heads = 2
codelm_n_kv_heads = 2

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 0.1
mse_weight = 10.0
entropy_weight = 0.0
label_reg_weight = 0.1

label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = (4,)*5
attn_window = 1024
upsampler_ncodes = 1
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0
remat_level = True

additive_drop_loss = False
downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
# downsampler_remat = True   # enable only if OOM

upsampler_d_model = 512
upsampler_n_layers = 2
upsampler_n_heads = 2
upsampler_n_kv_heads = 2
upsampler_window = 2
# upsampler_remat = True   # enable only if OOM

# use_codelm_bos = False
use_codelm_bos = True
codelm_bos_prob = 0.8
curriculum_mode = "no_freeze"
# quantize_mode = "gumbel"
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
batch_size = 16
val_batch_size = 8
level_steps = (0,)*4 + (int(100e3),)
# level_steps = (int(10e3),)*4 + (int(100e3),) 
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 10000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(100e3)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 10000
ckpt_keep = 1
