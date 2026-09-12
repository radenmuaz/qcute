"""Fork of cifar10_curr_off_s4_pq2.py (chat 2026-09-11): same eff_vocab=256 per level, but as a
SINGLE chunk -- code_vocab=256,pq_chunks=1 (256**1=256) instead of pq2's two 16-way chunks
(16**2=256). Same code capacity, tests PQ chunking granularity as the one variable. Same
strides/d_model/weight_sharing=False/optimizer/epochs_per_phase as pq2.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_s4_pq1v256.py
"""

run_name = "cifar10_curr_off_s4_pq1v256"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (256, 256, 256, 256, 256)
pq_chunks = (1, 1, 1, 1, 1)   # eff_vocab: 256 per level (single chunk)
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"   # chat 2026-09-11 -- see cifar10_curr_off_s4.py's comment.

# --- training ---
batch_size = 16
epochs_per_phase = 1000   # capped 3000->1000 (chat 2026-09-11) -- see s4.py's comment
# (v16's phase1 collapsed around epoch 2100->2200).
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
