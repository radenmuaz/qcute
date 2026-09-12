"""No-curriculum baseline (chat 2026-09-12): skips the phase-by-phase curriculum entirely, trains
ALL 4 decoder levels jointly from step 1 (--no_curriculum, curriculum_mode="no_freeze" still
required). traversal="zorder", byte_group=3 (one pixel per position), token_head_type="ar"
everywhere -- tiny causal chain per-position, token_dim sized per-level (128 for level0's large
byte alphabet, 32 for levels1-3's small PQ-code alphabet, matching "linears"' own param budget --
see run_lagcodec_zorder.py's audit). No true MTP (mtp_horizon=1, disabled).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_zorder_nocurr_ar.py --no_curriculum true --epochs_per_phase 4000
"""

run_name = "cifar10_zorder_nocurr_ar"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)
mlp_mult = 4
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

no_curriculum = True

# --- training ---
batch_size = 16
epochs_per_phase = 4000
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
grad_clip = 1.0
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
