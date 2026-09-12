"""Fork of cifar10_curr_off_s4_gumbel1.py (chat 2026-09-11): gumbel_temperature=0.01 (much
sharper than 0.1/1.0) + epochs_per_phase bumped 1000->1500. User observation on gumbel1/gumbel01
runs: cascade samples at intermediate phases (e.g. phase2_epoch1000) degenerated toward a single
mode (e.g. collapsing toward one visually-similar output across the batch) -- suspected gumbel's
stochastic training incentivizes differently than plain argmax; testing the temperature extremes
(0.01 here, 2.0 in the gumbel2 sibling) with more epochs to see if either avoids the collapse.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_s4_gumbel001.py
"""

run_name = "cifar10_curr_off_s4_gumbel001"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 65536 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.01
gumbel_at_inference = False

# --- training ---
batch_size = 16
epochs_per_phase = 1500
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
# sinkgd (previous optimizer here, chat 2026-09-11) -- do not delete, comment/uncomment to swap:
# lr = 1e-2
# weight_decay = 0
# optimizer = "sinkgd"
# optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
