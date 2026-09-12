"""Fork of cifar10_curr_off_s4_pq8v8.py (chat 2026-09-11): quantize_mode="gumbel",
gumbel_temperature=0.1 -- adds gumbel-softmax quantization on top of the pq8v8 capacity change.
Gumbel is TRAINING-only by default (gumbel_at_inference=False). Everything else identical to
pq8v8.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_s4_pq8v8_gumbel01.py
"""

run_name = "cifar10_curr_off_s4_pq8v8_gumbel01"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (2, 2, 2, 2, -1)   # changed 4->2 (chat 2026-09-12)
code_vocab = (8, 8, 8, 8, 8)
pq_chunks = (8, 8, 8, 8, 8)   # eff_vocab: 16777216 per level (8 chunks of vocab=8)
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False

# --- training ---
batch_size = 16
epochs_per_phase = 1000
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
