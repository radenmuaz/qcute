"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_6.py
"""

# --- model ---
# 8 levels of stride 2, one layer per encoder/decoder stack (see run5/6/7). Level j code covers 2^(j+1) px. Groups are
# 8x8 px tiles (64 px) wherever a code is smaller than that (decoder_ncodes = 32/2^j), z-order: an 8x8 grid of tiles.
DEPTH = 8
multihost = True
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
img_size = 64
d_model = (512,) * DEPTH
n_layers = (1,) * DEPTH
n_heads = (8,) * DEPTH
n_kv_heads = (None,) * DEPTH
code_vocab = (256,) * DEPTH
pq_chunks = (3,) * DEPTH
pq_dim = (128,) * DEPTH
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (2,) * DEPTH
decoder_ncodes = (32, 16, 8, 4, 2, 1, 1, 1)  # 8x8 px per group (one code from level 5 up)
# own-level history in groups (previous 8x8 tile + own); -1 (all previous) is cheap on the coarse levels
ncodes_window = (1, 1, 1, 1, -1, -1, -1, -1)
# coarse levels wait for a quarter of their parent codes (one 32x32 quadrant) before decoding it; fine levels stream
stream_chunks = (0, 0, 0, 0, 4, 4, 4, 4)
attn_lookahead = 0
attn_window = (256,) * DEPTH
decode_past = 0
decode_future = 4
# refine drafts = previous pass output of the groups behind (1 group = one 8x8 tile). z-order neighbour distances in tiles:
# left <= 3, above <= 6 except the two central seam lines; window 6 covers those, level 0 is near-deterministic -> 2
level_refine_window = (2, 6, 6, 6, 6, 6, 6, 6)
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_passes = (2,) * DEPTH
level_refine_drop = 0.5  # stop before each extra pass w.p. 0.5 -> 1..2 passes per step
level_refine_gt_drop = 0.5  # refine draft = own sample w.p. 0.5, else the real target
refine_remat = True
cycle_refine_passes = 1
# coarser-level codes as extra context: window = own tile + one tile behind (in coarser codes)
cond_depth = (2, 3, 3, 3, 3, 3, 2, 1)
cond_window = (32, 16, 8, 4, 4, 4, 2, -1)
cond_drop = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0)

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
remat = True
remat_level = False

byte_group = 3
token_head_type = "ar"
token_dim = (256,) * DEPTH
token_n_heads = 4
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = False

# --- training ---
batch_size = 4
val_batch_size = 2
level_epochs = (0,) * (DEPTH - 1) + (10,)
seed = 0
train_subset_n = None
val_subset_n = 512
gen_eval_every_epoch = 1
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

