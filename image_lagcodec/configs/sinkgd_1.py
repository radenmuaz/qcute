
# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, )
n_layers = (4, 4, 4, 4,)
n_heads = (4, 4, 4, 4,)
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
quantize_mode = "argmax"
cascade_rollout_prob = 0.5
init_scheme = "llama"
use_xsa = True

byte_group = 3
token_head_type = "linears"
token_dim = (256, 32, 32, 32,)
token_n_heads = 4
mtp_horizon = 8
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 128
epochs_per_phase = (200, 200, 200, )
warmup_steps = 100
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 5e-3
lr_schedule = "cosine"
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 2, "weight_decay": 0}

# --- logging ---
log_every = 100
gen_eval_every = 10
qual_gen_n = 8
