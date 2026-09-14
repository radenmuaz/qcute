"""uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/linears_1_wma.py"""


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
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_prob = 0.8
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
val_batch_size = 16
epochs_per_phase = (400, 400, 400, )
warmup_steps = 100
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 5e-4
lr_schedule = "cosine"
weight_decay = 1e-5
optimizer = "adamw"
optimizer_kwargs = {}

# --- weight averaging ---
wa_mode = "wma"
wa_every = 1000
wa_stack_size = 5
wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
gen_eval_every = 10
ckpt_every = 10
ckpt_keep = 1
qual_gen_n = 16
