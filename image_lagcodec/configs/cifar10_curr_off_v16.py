"""Fork of cifar10_curr_off.py (chat 2026-09-11): code_vocab=16,pq_chunks=4 UNIFORM at every
level (eff_vocab=16**4=65536 each) instead of the per-level (8,8,4,4)/(5,5,5,5) split -- second
variable to isolate alongside weight_sharing=False. Everything else identical.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_v16.py
"""

run_name = "cifar10_curr_off_v16"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4)   # eff_vocab: 65536 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False

# --- training ---
batch_size = 16
epochs_per_phase = 3000
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
