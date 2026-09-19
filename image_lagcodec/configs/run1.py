"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/run2.py
"""

# --- model ---
img_size = 32
d_model = (256, 256)
# d_model = (512, 512)
# n_layers = (2, 2)
n_layers = (4, 4)
# n_heads = (4, 4)
n_heads = (2, 2)
n_kv_heads = (None, None)
# strides = (4, 4)
# pq_chunks = (6, 6)
code_vocab = (64, 64)
pq_chunks = (8, 8)
pq_dim = (128, 128)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

# Kspan = decoder_ncodes × stride[level]  # = G × K
# Pp    = level_refine_window × Kspan # = level_refine_window × decoder_ncodes × stride[level]

strides = (4, 4)
decoder_ncodes = 4
ncodes_window = 4
attn_lookahead = 0
decode_past = 0
decode_future = 16
level_refine_window = 1
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_passes = 3
cycle_refine_passes = 1
cond_depth = (2, 1)
cond_drop = 0.5

additive_drop_loss = True
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
# quantize_mode = "argmax"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.9
quantize_drop = 0.9
# feedback_p = 0.5
# feedback_p = 0.0


init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "fp32"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e3), int(1e5))
seed = 0
# warmup_steps = 2
train_subset_n = None
val_subset_n = None
# train_subset_n = 100
# val_subset_n = 10
# train_subset_n = 100
gen_eval_every_step = 1000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e5)
warmup_steps = 1000
# lr_min_epoch = 50
# lr_min_epoch = 400
# weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = dict(
                        weight_decay=0,
                        # weight_decay=1e-5,
                        # b1=0.8, b2=0.9,
                        #  eps=1e-8,eps_root=0.0,
                        #  nesterov=False

)

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

'''
Quick intuition: Adam's b2 sets an effective averaging window of 1/(1-b2) steps for the second-moment estimate. Default b2=0.999 → ~1000-step window. The rule of thumb: that window should be well under your total step count, or the optimizer never leaves its warmup regime.


b2 window = 1/(1-b2), keep it « total steps
n=100 (short phase): b2≈0.95, b1≈0.85-0.9
n=1000: b2≈0.97-0.98
n=50k (long schedule): keep defaults b1=0.9, b2=0.999
'''

'''
audit your code tpu1, overfit logs/overfit1/ checkpoint, i ran and still bad generation
check config there in that folder
run on tpu1 use tpu
'''