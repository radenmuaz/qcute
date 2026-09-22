"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/d2_eager.py
"""
# Fork of run1.py (the original 2-level config): "fully AR eager, 1 ncode" -- decoder_ncodes=1 (finest possible groups), stream_chunks=0
# (default, no window-rounding wait). Pardec (not interleave_decode): groups stay independent/parallel-batched,
# this variant only tightens each group's own context window to per-group granularity (least wait possible).

# --- model ---
img_size = 32
d_model = (256, 256)
n_layers = (2, 2)
n_heads = (2, 2)
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
decoder_ncodes = 1  # was 4 in run1 -- finest granularity
ncodes_window = 4
attn_lookahead = 0
decode_past = 0
decode_future = 4
level_refine_window = 1
level_refine_gumbel = True
level_refine_temperature = 1.0
# level_refine_passes = 2
cycle_refine_passes = 1
cond_depth = (2, 1)
cond_drop = 0.5

additive_drop_loss = True
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
precision = "fp32"

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e3), int(1e5))
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 1000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e5)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-3)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 1000
ckpt_keep = 1
