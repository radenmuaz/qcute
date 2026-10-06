"""
uv run python3 -m image_lagcodec.run_lagcodec_res_pretrain --config image_lagcodec/configs/cifar_5_pretrain.py
"""
# Encoder-only CodeLM/downsampler pretraining.
img_size = 32
encoder_only_pretrain = True
log_levelwise_metrics = True
fsdp = True  # shard model parameters and optimizer state over the available JAX device mesh
share_across_levels = True
codelm_d_model = 1024
codelm_n_layers = 16
codelm_n_heads = 8
codelm_n_kv_heads = 8
mlp_mult = 4

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
strides = (4,4,4,4,4)
attn_window = -1
attn_lookahead = 0
# remat = True

downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
downsampler_rollout = True
downsampler_rollout_prob = 0.8
# downsampler_remat = True   # enable only if OOM

# context_source = "codelm_upper"
context_source = "own_embed"

# use_codelm_bos = False
use_codelm_bos = True
codelm_bos_prob = 0.8
# curriculum_mode = "freeze"
curriculum_mode = "no_freeze"
# level_select_prob = (0.9, 0.8, 0.7, 0.6)  # length n_levels-1=4
# multires_entry_gt_drop = (0.0, 0.5, 0.5, 0.5, 0.5)  # length n_levels=5, index 0 unused
quantize_mode = "zgr"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.8
# pss_input_mode = "argmax"                # matches greedy generation
# pss_input_mode = "sample"              # gumbel-max, matches sampled generation
# pss_temperature = 1.0
# downsampler side: no effect unless downsampler_ncodes > 1 (one token per row has no token input)
# downsampler_ncodes = 2
# downsampler_pss_passes = -1
# downsampler_pss_prob = 1.0
quantize_drop = 0.2
init_scheme = "llama"
# use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
# no digit-level AR sampling at all, parallel/MTP-style heads instead
codelm_token_head = "ar"   # CodeLM NTP/free-run head: all digits in one parallel matmul
pardec_token_head = "ar"    # downsampler/upsampler digit head: all digits in one parallel matmul
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
level_steps = (0,)*4+ (int(1e6),)
seed = 0
train_subset_n = None
val_subset_n = 1000
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
# lr_min_step = int(100e3)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

wa_verbose = False

# wa_every_step = 100
# wa_mode = "ema"
# wa_ema_decay = 0.9

# wa_every_step = 1000
# wa_mode = "wma"
# wa_stack_size = 3
# wa_wma_weights = (1.0,1.0,1.0)

# wa_stack_size = 5
# wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 500
ckpt_every_step = 5000
# ckpt_every_step = int(2e5)
ckpt_keep = 1