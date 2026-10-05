"""Tiny local smoke test for checkpoint save/resume and upsampler training.

Run:
  uv run python3 -m image_lagcodec.run_lagcodec_res_denoise --config image_lagcodec/configs/local_10step.py --run_name local_10step
"""

img_size = 32
share_across_levels = False
codelm_d_model = 128
codelm_n_layers = 1
codelm_n_heads = 1
codelm_n_kv_heads = 1

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 2
rope_base = 10000.0

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0
label_fn = "rgb_label_fn_jax"
bos_rate_mode = "absolute"
strides = (4, 4, 4, 4, 4)
attn_window = 1024
upsampler_ncodes = 1
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0
remat = True

downsampler_d_model = 64
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
downsampler_rollout = True
downsampler_rollout_prob = 0.2
context_source = "own_embed"
upsampler_d_model = 128
upsampler_n_layers = 2
upsampler_n_heads = 1
upsampler_n_kv_heads = 1
upsampler_window = 1
upsampler_rollout = True
upsampler_rollout_prob = 0.2

curriculum_mode = "no_freeze"
quantize_mode = "zgr"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.5
level_refine_passes = 1
quantize_drop = 0.2

init_scheme = "llama"
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
codelm_token_head = "ar"
pardec_token_head = "ar"
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = False
gen_eval_all_levels = False
gen_eval_teacher_force_sanity = False

log_levelwise_metrics = True
log_levelwise_eval = True
log_levelwise_gen = False

batch_size = 2
val_batch_size = 2
level_steps = (20,)
seed = 0
train_subset_n = 32
val_subset_n = 16

gen_eval_every_step = 100000
log_every = 1
ckpt_every_step = 10
ckpt_keep = 2

epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 5
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=0)
