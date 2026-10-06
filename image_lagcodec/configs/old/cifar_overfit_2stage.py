"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/cifar_res_4_overfit_reinmax_ctxdetach_linear.py
"""
# Fork of cifar_res_4_overfit_reinmax_s_ctxdetach_linear.py -- level_steps fixed back to
# (0,)*4+(100000,) (skip phases 1-4 entirely, straight to joint all-levels training, matching the
# rest of the cifar_res_4 family) instead of the _s variant's (1e3,)*4+(100000,) staged steps;
# "_s_" dropped from the name accordingly. Cleaned to remove digit-level AR sampling entirely
# (added 2026-10-02): codelm_token_head/pardec_token_head both "linear" instead of "ar" -- all
# pq_chunks digits of a code/token are predicted in ONE parallel matmul, both at train time
# (pardec_score) AND at generation time (pardec_generate/encoder_free_run's per-step sampling is
# still sequential ACROSS positions, but no longer sequential WITHIN a position's digits). This
# removes the train/inference mismatch a checkpoint_level_eval.py probe pointed at (digit-AR
# self-feeding via pardec_generate is never exercised during teacher-forced AR-head training) by
# construction, rather than training around it (see the _uprollout fork for that alternative) --
# isolates whether avoiding digit-AR sampling altogether (MTP-style parallel inference) fixes the
# severe generate-vs-teacher-force MSE gap on its own. downsampler_rollout/upsampler_rollout left at
# default False (incompatible with pardec_token_head="linear", would raise). ctx_stop_gradient +
# decoder_scheduled_sampling_prob (level-to-level, orthogonal to digit-level AR) kept as-is.
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

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0

label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = (4,4)
attn_window = 1024
upsampler_ncodes = 1
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0
# remat_level = True

downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
# downsampler_remat = True   # enable only if OOM

upsampler_d_model = 1024
upsampler_n_layers = 4
upsampler_n_heads = 2
upsampler_n_kv_heads = 2
upsampler_window = 2
# upsampler_remat = True   # enable only if OOM

use_codelm_bos = False
# use_codelm_bos = True
# codelm_bos_prob = 1.0
# curriculum_mode = "freeze"
curriculum_mode = "no_freeze"
# level_select_prob = (0.9, 0.8, 0.7, 0.6)  # length n_levels-1=4
# multires_entry_gt_drop = (0.0, 0.5, 0.5, 0.5, 0.5)  # length n_levels=5, index 0 unused
quantize_mode = "reinmax_limit"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.0
quantize_drop = 0.0
# ctx_stop_gradient = True
ctx_stop_gradient = "pseudo"
# decoder_scheduled_sampling_prob = 0.3

init_scheme = "llama"
# use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "linears"
# no digit-level AR sampling at all, parallel/MTP-style heads instead
codelm_token_head = "linear"   # CodeLM NTP/free-run head: all digits in one parallel matmul
pardec_token_head = "linear"    # downsampler/upsampler digit head: all digits in one parallel matmul
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True
gen_eval_all_levels = True
gen_eval_teacher_force_sanity = True


# --- training ---
batch_size = 4
val_batch_size = 8
# level_steps = (10_000,)*4 + (int(100e3),)
level_steps = (0, int(100e3))
seed = 0
train_subset_n = 1000
val_subset_n = 100
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
# lr_min_step = int(100e3)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=0)

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
ckpt_every_step = 10000
ckpt_keep = 1
