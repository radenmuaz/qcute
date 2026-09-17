"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/overfit1.py
"""

# --- model ---
img_size = 32
# d_model = (128, 128)
d_model = (256, 256)
n_layers = (2, 2)
n_heads = (2, 2)
n_kv_heads = (None, None)
# strides = (4, 4)
code_vocab = (256, 256)
pq_chunks = (3, 3)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.0

strides = (4, 4)
# strides = (2, 2)
decoder_ncodes = 1
# ncodes_window = 0
ncodes_window = 64
attn_lookahead = 0
decode_past = 0
# decode_future = 8
decode_future = 16
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
# gumbel_temperature = 1.0
gumbel_temperature = 0.1
gumbel_at_inference = False

mse_softmax_tau = 1.0
level_drop = 0.9
quantize_drop = 0.9
# feedback_p = 0.5
# feedback_p = 0.0


init_scheme = "llama"
use_xsa = True
use_attn_sink = True
precision = "fp32"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
pq_dim = (128, 128)
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 16
val_batch_size = 8
phase_steps = (int(5e2), int(1e5))
seed = 0
# warmup_steps = 2
# train_subset_n = None
train_subset_n = 100
# train_subset_n = 100
gen_eval_every_step = 1000
epoch_verbose = False

grad_clip = 10.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e4)
warmup_steps = 100
# lr_min_epoch = 50
# lr_min_epoch = 400
weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "none"

# wa_mode = "ema"
# wa_every_step = 100
# wa_ema_decay = 0.9
# wa_verbose = False

# wa_mode = "wma"
# wa_every_epoch = 1
# wa_stack_size = 5
# wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
ckpt_every_step = 1000
ckpt_keep = 1
