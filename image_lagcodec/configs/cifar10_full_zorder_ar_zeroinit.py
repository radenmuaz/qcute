run_name = "cifar10_full_zorder_ar_zeroinit"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512,)
n_layers = (2, 2, 2, 2,)
n_heads = (4, 4, 4, 4,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4)
mlp_mult = 2
rope_base = 10000.0
ntp_weight = 1.0
decoder_ncodes = 4
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5
init_scheme = "zero"
use_xsa = True
use_qknorm = True

byte_group = 3
token_head_type = "ar"
token_dim = (128, 32, 32, 32,)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 256
# epochs_per_phase = (20, 20, 100, )
epochs_per_phase = (200, 200, 200, )
warmup_steps = 100
grad_clip = 10.0
seed = 0
train_subset_n = None

# lr = 1e-3
# weight_decay = 1e-5
# optimizer = "adamw"
# optimizer_kwargs = {}

lr = 1e-1
# lr = 1e-2
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 2, "weight_decay": 0}

# --- logging ---
log_every = 100
qual_gen_n = 8
