"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar_parfullcausal2.py
"""
# Fork of cifar_parfullcausal1.py: fixes the non-square-patch issue. Under traversal="zorder", a
# group of decoder_ncodes consecutive blocks only forms a SQUARE patch when decoder_ncodes is a
# power of 4 (Z-order recursively subdivides into quadrants; any other run length spans a partial
# quadrant, i.e. a rectangle). cifar_parfullcausal1's decoder_ncodes=(32,8) are NOT powers of 4.
# Fixed here to the smallest power-of-4 at or above the n_blocks/8 throughput threshold (same rule
# cifar_parfullcausal1 itself used): level0 n_blocks=256 (threshold 32) -> 64=4^3; level1
# n_blocks=64 (threshold 8) -> 16=4^2.
#
# Everything else unchanged from cifar_parfullcausal1.py: ncodes_window=-1 (pardec's "causal
# unbounded" mode -- every group still an independent batched row, but its context window Wg is
# the FULL causal prefix instead of a small fixed slice), refine off (isolate ncodes_window=-1's
# own effect).

# --- model ---
img_size = 32

d_model = (256, 256)
n_layers = (4, 4)  # cheap encoder
n_heads = (2, 2)
n_kv_heads = (None, None)
decoder_d_model = (1024, 1024)
decoder_n_layers = (4, 4)
decoder_n_heads = (8, 8)
decoder_n_kv_heads = (8, 8)

code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (64, 64)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 1.0

# Kspan = decoder_ncodes × stride[level]  # = G × K
strides = (4, 4)
# attn_window = (1024,1024)
decoder_ncodes = (16, 16)  # was (32, 8) -- not powers of 4, gave rectangular (not square) patches
# under z-order traversal; see module docstring
ncodes_window = (4, 4)
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True
level_refine_window = 0  # refine disabled (level_refine_passes=1): isolate ncodes_window=-1's own effect
level_refine_passes = 1

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.9
quantize_drop = 0.9

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
token_dim = (64, 64)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(2e4), int(5e4))
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 2000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(5e4)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# wa_mode = "ema"
# wa_verbose = False
# wa_every_step = 200
# wa_ema_decay = 0.99

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
