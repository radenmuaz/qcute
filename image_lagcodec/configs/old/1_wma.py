"""uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/1_wma.py"""


# --- model ---
img_size = 32
d_model = (512, 512, 512, 512, )
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
weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_drop = 0.2
init_scheme = "llama"
use_xsa = False
pq_dim = (128, 64, 64, 64)

byte_group = 3
token_head_type = "ar"
token_dim = (256, 64, 64, 64,)
token_n_heads = 4
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 64
val_batch_size = 16
phase_epochs = (200, 200, 200, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 100
weight_decay = 1e-3
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "wma"
quantize_drop = 0.8
wa_every_epoch = 5
wa_stack_size = 5
wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
gen_eval_every_epoch = 100
ckpt_every_epoch = 10
ckpt_keep = 1
qual_gen_n = 16
