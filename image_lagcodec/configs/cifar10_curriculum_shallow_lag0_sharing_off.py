"""Curriculum-trained hierarchical codec (chat 2026-09-10), weight_sharing=False: every level
with a decoder gets two INDEPENDENT sets of weights (encoder role and decoder role never share
parameters) -- ablation baseline against cifar10_curriculum_shallow_lag0_sharing_on.py. Same
rolling curriculum schedule and architecture otherwise.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curriculum_shallow_lag0_sharing_off.py
"""

run_name = "cifar10_curriculum_shallow_lag0_sharing_off"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = 4
pq_chunks = 5   # effective vocab = code_vocab**pq_chunks = 4**5 = 1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False

# --- training ---
batch_size = 16
epochs_per_phase = 3000
lr = 1e-2
warmup_steps = 100
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
