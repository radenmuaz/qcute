"""No-curriculum baseline (chat 2026-09-12): skips the phase-by-phase curriculum entirely, trains
ALL 4 decoder levels jointly from step 1 (--no_curriculum, curriculum_mode="no_freeze" still
required). traversal="zorder", byte_group=3 (one pixel per position), token_head_type="linears"
everywhere. TRUE MTP enabled: mtp_horizon=4 (max allowed -- capped at each level's own stride=4),
mtp_mode="parallel" (K=4 independent linear heads off one hidden state, no chaining across the
4 future steps -- the only mtp_mode implemented so far; "ar" mtp_mode raises NotImplementedError).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_zorder_nocurr_mtp_linears_parallel.py --no_curriculum true --epochs_per_phase 4000
"""

run_name = "cifar10_zorder_nocurr_mtp_linears_parallel"

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
token_head_type = "linears"
mtp_horizon = 4
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
