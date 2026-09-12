"""Fork of cifar10_curr_off_v16.py (chat 2026-09-11): v16 confirmed working (pull+inspect
samples) -- conclusion was codebook capacity, not weight_sharing, was the real bottleneck. This
fork changes the DEPTH/STRIDE shape instead: uniform strides=(4,4,4,4,-1) (4 real levels of
stride=4 each, not the hand-tuned (3,16,16,-1)) -- 5 levels total now, n_phases=4. Same
d_model/code_vocab/pq_chunks/weight_sharing=False/optimizer as v16, extended to 5 levels.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_s4.py
"""

run_name = "cifar10_curr_off_s4"

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
curriculum_mode = "no_freeze"   # chat 2026-09-11: levels never freeze, phase p trains ALL of
# levels[0..p-1] jointly -- was "freeze" (each phase only trains its own new level, everything
# below frozen), which showed cascade collapsing hard the moment a new phase started.

# --- training ---
batch_size = 16
epochs_per_phase = 1000   # capped 3000->2000->1000 (chat 2026-09-11): v16's phase1
# catastrophically collapsed between epoch 2100->2200 (loss 0.013->10.75, never recovered)
# despite being near-perfect just before; then s4 itself ALSO collapsed at 2000 -- cap lower.
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
