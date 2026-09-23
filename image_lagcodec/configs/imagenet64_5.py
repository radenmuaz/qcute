"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_5.py
"""

# --- model ---
multihost = True
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
img_size = 64
d_model = (512, 512)
n_layers = (4, 4)
n_heads = (8, 8)
n_kv_heads = (None, None)
code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (128, 128)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (4, 4)
decoder_ncodes = 4
ncodes_window = 4
attn_lookahead = 0
attn_window = (64, 64)
decode_past = 4
decode_future = 4
level_refine_window = 1
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_passes = 1
cycle_refine_passes = 1
cond_depth = (2, 1)
cond_drop = 0.5
cond_window = (4, -1)

additive_drop_loss = False
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
# quantize_mode = "argmax"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.5
quantize_drop = 0.5
# feedback_p = 0.5
# feedback_p = 0.0


init_scheme = "llama"
# use_xsa = True
# use_sink = True
use_xsa = False
use_sink = False
precision = "bf16"
remat = False
remat_level = True
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 4
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = False

# --- training ---
batch_size = 8
val_batch_size = 4
level_epochs = (0, 2)
seed = 0
train_subset_n = None
val_subset_n = 512
gen_eval_every_epoch = 0.25
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 1000
# lr_min_epoch = 50
# lr_min_epoch = 400
# weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = dict(
                        weight_decay=1e-3,
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
ckpt_every_step = 2000
ckpt_keep = 1

