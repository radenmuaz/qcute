"""
uv run python3 -m image_lagcodec.run_lagcodec_res_pretrain --config image_lagcodec/configs/pretrain_in64_1_tpu1.py
"""
# --- data ---
img_size = 256
dataset = "imagenet256_jxl"
data_root = "/dev/shm/imagenet256_jxl"
multihost = False
fsdp = True
# fsdp_mode = "intra_node_fsdp_inter_node_dp"

traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True
gen_eval_prompt = 2048
# remat_level = True
share_across_levels = False

codelm_d_model = 2048
codelm_n_layers = 16
codelm_n_heads = 16
codelm_n_kv_heads = 16
mlp_mult = 2

code_vocab = 256
pq_chunks = 3
pq_dim = 128
rope_base = 10000.0

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0

label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = None
attn_window = -1
attn_lookahead = 0
# remat = True
# context_source = "own_embed"
# use_codelm_bos = False
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
token_head_type = "ar_flat"
codelm_token_head = "ar_flat"
pardec_token_head = "ar"
token_dim = 128
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True
gen_eval_prompt = 512

# --- training ---
batch_size = 4
val_batch_size = 4
# level_steps = (20_000, 20_000, 20_000, 100_000)
level_steps = (int(2e5),)
seed = 0
train_subset_n = None
val_subset_n = None
# val_subset_n = 1000
gen_eval_every_step = 10000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = level_steps[0] // 2
warmup_steps = 10_000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-2)

wa_verbose = False

# --- logging ---
log_every = 100
ckpt_every_step = 10_000
# ckpt_every_step = int(2e5)
ckpt_keep = 1