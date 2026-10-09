"""
uv run python3 -m image_lagcodec.run_lagcodec_res_pretrain --config image_lagcodec/configs/pretrain_in64_2_stride4.py

pretrain_in64_2_stride4
fsdp:
00:00:11] === starting level5 for 2.5 epochs (400000 steps) ===                                                                          
[00:09:41] level5: first train_step (incl. jit compile) took 570.4s                                                                      
level5:   0%|                            | 481/400000 [25:34<222:50:44,  2.01s/it  

dp:
[00:00:10] === starting level5 for 2.5 epochs (400000 steps) ===                                                                          
[00:07:09] level5: first train_step (incl. jit compile) took 418.7s  
level5:   0%|                            | 155/400000 [15:31<50:58:56,  2.18it/s 
"""
# --- data ---
img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
multihost = False
fsdp = False

level_gt_input_prob = 0.8
traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True
gen_eval_prompt = 2048
# remat_level = True
share_across_levels = True
# share_downsampler_upsampler_lm = True
# remat = True
# remat_codelm = True
# downsampler_remat = True
# codelm_d_model = 1024
# codelm_n_layers = 16

codelm_d_model = 1024
codelm_n_layers = 16
codelm_n_heads = 8
codelm_n_kv_heads = 8
mlp_mult = 4

code_vocab = 256
pq_chunks = 3
pq_dim = 128
# pq_dim = 64
rope_base = 10000.0

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0


label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = (4, 4, 4, 4, 4, 4)
# attn_window = 1024
attn_window = -1
attn_lookahead = 0


downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
downsampler_rollout = True
downsampler_rollout_prob = 0.8

use_codelm_bos = True
codelm_bos_prob = 0.8
curriculum_mode = "no_freeze"
quantize_mode = "zgr"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.8
quantize_drop = 0.2

init_scheme = "llama"
# use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
codelm_token_head = "ar"
pardec_token_head = "ar"
token_dim = 128
# token_dim = 64
token_n_heads = 2
# --- training ---
# batch_size = 8 # fsdp
# val_batch_size = 8

batch_size = 2
val_batch_size = 2
# Approximate step equivalents (tpu34 v4-16: batch_size 8 x 4 local devices x 2 hosts = global batch 64;
# ImageNet64 train = 1,281,167 imgs -> 1 epoch = 1,281,167 / 64 = 20,018 steps):
level_steps = (0,0,0,0,0, 400_000)
seed = 0
# train_subset_n = None
# val_subset_n = 512
# gen_eval_every_epoch = 0.25   # ~= every 5,005 steps at global batch 64
gen_eval_every_step = 10_000   # uncomment (and comment gen_eval_every_epoch) to eval by steps
epoch_verbose = False
# ctx_stop_gradient = "pseudo"


grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = level_steps[-1] // 2
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-2)

# --- logging ---
log_every = 1000
ckpt_every_step = 10_000
# ckpt_every_step = 10
ckpt_keep = 1
