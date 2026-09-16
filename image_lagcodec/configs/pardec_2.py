"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/pardec_2.py

decoder_ncodes (G) vs n_groups (=B2/B) at n_positions=1024, K=stride[0]=4 -> n_blocks=256 fixed.
Wg=(N+1)*G:
  G=1  -> n_groups=256, window=0 (disjoint) Wg=1,  window=1 Wg=2
  G=4  -> n_groups=64,  window=0 Wg=4,  window=1 Wg=8
  G=16 -> n_groups=16,  window=0 Wg=16, window=1 Wg=32
  G=64 -> n_groups=4,   window=0 Wg=64, window=1 Wg=128
window=0 is v1's original disjoint-groups behavior (no lookback, Wg=G). window=1 adds one full
previous group's ctx as read-only context (Wg=2G), independent of n_groups/group index (unlike
-1 in pardec_1_unbounded.py) -- constant 2G for every group except group 0 (G zero-padded blocks).
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

decoder_ncodes = 16
ncodes_window = 1   # chat 2026-09-15: renamed from decoder_ncodes_overlap (was in ctx BLOCKS, now
# in whole PREVIOUS GROUPS -- 1 = see the 1 immediately-preceding group's ctx as extra context)
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
batch_size = 8
val_batch_size = 16
epochs_per_phase = (100, 100, 100, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
# warmup_steps = 2
# epochs_per_phase = (1000, 1000, 1000, )
# train_subset_n = 1000

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
gen_eval_every = 100
ckpt_every = 100
ckpt_keep = 1
qual_gen_n = 16
