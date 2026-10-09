"""
uv run python3 -m image_lagcodec.run_lagcodec_res_pretrain_film --config image_lagcodec/configs/pretrain_in64_4_t.py
"""
# --- data ---
# img_size = 256
# dataset = "imagenet256"
# data_root = "/dev/shm/imagenet256"

img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"

multihost = False
# fsdp = True
# fsdp_mode = "intra_node_fsdp_inter_node_dp"

traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True
gen_eval_prompt = 256*3
# remat = True
share_across_levels = False
# share_across_levels = True
codelm_d_model = 512
codelm_n_layers = 8
codelm_n_heads = 8
codelm_n_kv_heads = 4
mlp_mult = 8

layer_drop_prob = 0.2

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
attn_window = 1024
# attn_window = -1
attn_lookahead = 0
# remat = True
# context_source = "own_embed"
use_codelm_bos = False
# use_codelm_bos = True
# codelm_bos_prob = 0.8
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
# token_head_type = "ar"
# codelm_token_head = "ar"
token_head_type = "ar_flat"
codelm_token_head = "ar_flat"
# token_head_type = "linears"
# codelm_token_head = "linears"

token_mask_prob = 0.2
token_dim = 128
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True

# --- training ---
batch_size = 16
val_batch_size = 16
# level_steps = (20_000, 20_000, 20_000, 100_000)
# level_steps = (int(4e5),)
# level_steps = (int(4e5)//batch_size,)
level_steps = (int(4e5)//batch_size*2,)
seed = 0
train_subset_n = None
val_subset_n = None
# val_subset_n = 1000
gen_eval_every_step = level_steps[-1]//10
epoch_verbose = False
skip_gen = True

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = level_steps[-1]*9 // 10
warmup_steps = level_steps[-1] // 10
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# grad_clip = 1.0
# lr = 0.02
# lr_schedule = "cosine"
# lr_min = 1e-5
# lr_min_step = level_steps[-1]*9 // 10
# warmup_steps = level_steps[-1] // 10
# optimizer = "sinkgd"
# optimizer_kwargs = dict(linear_lr_scale=0.05, weight_decay=0.0, sinkhorn_iters=2)

wa_verbose = False

# wa_mode = "wma"
# wa_every_step = 10_000
# wa_stack_size = 3
# wa_wma_weights = (1.0, 1.0, 1.0)

# --- logging ---
log_every = 1000
ckpt_every_step = level_steps[-1] // 10
# ckpt_every_step = int(2e5)
ckpt_keep = 10

# class_conditional = False

class_conditional = True
class_num_classes = 1000
class_drop_prob: float = 0.2
# class_bos_order = "level_then_class"

# class_bos_order = "class_then_level"