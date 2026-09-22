"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/run13.py
"""

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
decoder_ncodes = 2  # fork of run12 (G=1) -- G=2 is the max interleave_decode allows here (needs up_stride>=G, up_stride=2)
ncodes_window = 16
attn_window = (256,)*DEPTH
attn_lookahead = 0
interleave_decode = True  # hardcoded cond_depth<=2 interleave; verified 2026-09-22 (see test), chunk_groups perf knob added same day
cond_depth = (2, 2, 2, 1)  # each level conditions on the next coarser level's own codes; top level has none coarser
# decode_future left off: interleave_decode ignores it (same as dense_decode)

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
level_gt_drop = 0.8
quantize_drop = 0.8
# mse_softmax_tau = 1.0
# feedback_p = 0.5
# feedback_p = 0.0


init_scheme = "llama"
# use_xsa = True
# use_sink = True
use_xsa = False
use_sink = False
# precision = "fp32"
precision = "bf16"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (128,)*DEPTH
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8  # interleave_decode is a single real growing cache, not gen_sync's padded-wave approach -- try full batch first
level_steps = (5_000, 5_000, 5_000, 30_000)  # staged: each phase adds one level (no_freeze)
seed = 0
# warmup_steps = 2
train_subset_n = None
val_subset_n = None
# train_subset_n = 100
# val_subset_n = 10
# train_subset_n = 100
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = 43_000  # near the end of all phases (45k total)
warmup_steps = int(1e3)
# lr_min_epoch = 50
# lr_min_epoch = 400
# weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = dict(
                        weight_decay=1e-5,
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
ckpt_every_step = 5000
ckpt_keep = 1
