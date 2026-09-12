"""Full CIFAR-10 (chat 2026-09-12) -- first run this session NOT using the 100-image overfit-
sanity subset (train_subset_n=None -> the real 50,000-image training set). Curriculum ENABLED
(phase-by-phase, not --no_curriculum): epochs_per_phase=(20,20,20,100) -- phases 1-3 (levels
0,1,2 alone) get 20 epochs each (~62,500 steps at batch_size=16 on the full set), phase 4 (all 4
levels jointly, no_freeze) gets 100 epochs (~312,500 steps). traversal="zorder", byte_group=3, token_head_type="ar"
everywhere -- tiny causal chain per-position, token_dim sized per-level (128 for level0's large
byte alphabet, 32 for levels1-3's small PQ-code alphabet). No true MTP (mtp_horizon=1).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_full_zorder_ar.py
"""

run_name = "cifar10_full_zorder_ar"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512, 512)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)
mlp_mult = 2
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5

byte_group = 3
token_head_type = "ar"
token_dim = (128, 32, 32, 32, 32)
token_n_heads = 4
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 16
epochs_per_phase = (20, 20, 20, 100)
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
grad_clip = 1.0
seed = 0
train_subset_n = None

# --- logging ---
log_every = 10
qual_gen_n = 8
