"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/run20.py
"""
# Deep (DEPTH=4, stride=2) analog of run17.py: "2 real dependent passes" -- decoder_ncodes = n_blocks/2 per
# level (n_blocks=[512,256,128,64] -> G=[256,128,64,32]), interleave_decode=True (the only mechanism with real
# cross-group/pass dependency -- pardec's parallel batching never makes separate groups depend on each other's
# real output regardless of G). Needs the 2026-09-22 up_stride>=G removal (all four G values exceed
# up_stride=2) to even run.

# --- model ---
img_size = 32
DEPTH = 4
d_model = (256,)*DEPTH
n_layers = (2,)*DEPTH
n_heads = (2,)*DEPTH
n_kv_heads = (None,)* DEPTH
code_vocab = (256,) *DEPTH
pq_chunks = (3,)* DEPTH
pq_dim = (128,) * DEPTH
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (2,)*DEPTH
decoder_ncodes = (256, 128, 64, 32)  # n_blocks/2 per level -- exactly 2 groups/passes
interleave_decode = True
cond_depth = (2, 2, 2, 1)

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
level_gt_drop = 0.8
quantize_drop = 0.8

init_scheme = "llama"
use_xsa = False
use_sink = False
precision = "bf16"

byte_group = 3
token_head_type = "ar"
token_dim = (128,)*DEPTH
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (5_000, 5_000, 5_000, 30_000)
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = 43_000
warmup_steps = int(1e3)
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
