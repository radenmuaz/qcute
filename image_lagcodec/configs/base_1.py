"""
uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/mtp_ar_ar.py
"""


# --- model ---
img_size = 32
# d_model = (512, 512, 512, 512, )
# n_layers = (4, 4, 4, 4,)
# n_layers = (2, 2, 2, 2,)
n_heads = (4, 4, 4, 4,)
d_model = (256, 256, 256, 256, )
n_layers = (4, 4, 4, 4,)
# n_heads = (2, 2, 2, 2,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (16, 16, 16, 16,)
pq_chunks = (4, 4, 4, 4,)
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
mtp_weight = 0.1
decoder_ncodes = 4
weight_sharing = False
curriculum_mode = "no_freeze"
# quantize_mode = "argmax"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_prob = 0.8
# init_scheme = "zero"
init_scheme = "llama"
use_xsa = True

byte_group = 3
token_head_type = "ar"
token_dim = (128, 32, 32, 32,)
token_n_heads = 2
mtp_horizon = 2
mtp_mode = "ar"
traversal = "zorder"

# --- training ---
batch_size = 64
# epochs_per_phase = (20, 20, 100, )
epochs_per_phase = (200, 200, 200, )
warmup_steps = 100
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 5e-4
lr_schedule = "cosine"
weight_decay = 1e-3
optimizer = "adamw"
optimizer_kwargs = {}

# lr = 1e-2
# # lr = 1e-2
# weight_decay = 0
# optimizer = "sinkgd"
# optimizer_kwargs = {"sinkhorn_iters": 2, "weight_decay": 0}

# --- logging ---
log_every = 100
qual_gen_n = 8
gen_eval_every = 10
val_batch_size = 16