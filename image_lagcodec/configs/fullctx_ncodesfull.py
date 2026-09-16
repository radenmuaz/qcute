"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/fullctx_ncodesfull.py

TRUE FULLCTX preset, OTHER extreme (chat 2026-09-15, renamed from fullctx_unbounded.py):
decoder_ncodes=256 (>= every level's own n_blocks -- level0=256, level1=64, level2=16) triggers
the single-group fast-fallback path at every level (mathematically identical to the original
non-pardec sequential decode, see decode_generate_pardec's fallback docstring) -- "one long slow
AR", the fully-sequential opposite of fullctx_ncodes1.py's "one seed one token" max-parallel
extreme. ncodes_window is moot here (nothing to look back at with only one group) -- set to 0 for
clarity.
"""


# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, )
# n_layers = (2, 2, 2, 2,)
n_layers = (4, 4, 4, 4,)
n_heads = (2, 2, 2, 2,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (256, 256, 256, 256,)
pq_chunks = (3, 3, 3, 3,)
mlp_mult = 2
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.1
mse_weight = 1.0
entropy_weight = 0.01

decoder_ncodes = 256
ncodes_window = 0   # moot -- single group at every level, nothing to look back at
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_prob = 0.5
quantize_drop = 0.5
init_scheme = "llama"
use_xsa = True
pq_dim = (128, 128, 128, 128)
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128, 128, 128)
# token_dim = (64, 64, 64, 64,)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 16
val_batch_size = 16
epochs_per_phase = (100, 100, 100, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
# warmup_steps = 2
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 80
# lr_min_epoch = 400
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "none"

# --- logging ---
log_every = 100
gen_eval_every = 10
ckpt_every = 10
ckpt_keep = 1
qual_gen_n = 16
