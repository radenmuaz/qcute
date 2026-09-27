"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/cifar_res_full1.py
"""
# --- model ---
img_size = 32

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
entropy_weight = 0.1
label_reg_weight = 1.0

label_fn = "rgb_label_fn_jax"
bos_rate_mode = "relative"
# bos_rate_mode = "absolute"
strides = (4, 4)              # single level only -- no cascade
attn_window = 1024
upsampler_ncodes = (16, 16)
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 4
remat_level = True

additive_drop_loss = False
downsampler_d_model = 1024
downsampler_n_layers = 4
downsampler_n_heads = 8
downsampler_n_kv_heads = 8
downsampler_window = 4
# downsampler_remat = True
upsampler_d_model = 1024
upsampler_n_layers = 4
upsampler_n_heads = 8
upsampler_n_kv_heads = 8
upsampler_window = 4
# upsampler_remat = True

use_codelm_bos = False   # ON for this ablation (was False in cifar_res_full1.py)
# use_codelm_bos = True   # ON for this ablation (was False in cifar_res_full1.py)
# codelm_bos_prob = 1.0   # always substitute -- no probabilistic drop back to real content
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
batch_size = 8
val_batch_size = 8
level_steps = (int(10e3),int(20e3))  # single phase, full original 70k budget (was split 10k/10k/50k across 3 levels)
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 4000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(20e3)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 4000
ckpt_keep = 1
