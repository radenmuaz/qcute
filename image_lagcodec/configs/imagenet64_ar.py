"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_ar.py
"""
# Fork of imagenet64_5.py: full AR (interleave_decode=True, decoder_ncodes=1) -- the real fully-sequential
# single-growing-cache causal chain, cond_depth=(2,1) preserved. Direct successor to the crashed gen_sync-based
# imagenet64_8 attempt (1h41m/eval, near-random gen_byte_acc, abandoned 2026-09-22) -- interleave_decode is the
# fast (single real cache, no wave-dispatch), verified-correct replacement for that mechanism. decode_past/
# decode_future/level_refine_passes dropped (ignored under interleave_decode, same as dense_decode).

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
decoder_ncodes = 1  # was 4 in imagenet64_5 -- full AR eager, finest granularity
interleave_decode = True
attn_lookahead = 0
attn_window = (64, 64)
cond_depth = (2, 1)
cond_window = (4, -1)

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.5
quantize_drop = 0.5

init_scheme = "llama"
use_xsa = False
use_sink = False
precision = "bf16"
remat = False
remat_level = True

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 4
mtp_horizon = 1
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
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-3)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 2000
ckpt_keep = 1
