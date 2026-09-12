"""Fork of cifar10_curriculum_shallow_lag0_sharing_on_thin.py (chat 2026-09-11): weight_sharing
=False -- thin/fat both got stuck at IDENTICAL dec_acc/bpb regardless of model size (256 vs
256/512/512), suspicious signal pointing at weight_sharing itself (encoderlevel_i=decoderlevel_i,
same weights forced to serve both an unconditioned-NTP role and a ctx-conditioned decode role)
as a capacity bottleneck rather than raw width. Everything else identical to thin.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off.py
"""

run_name = "cifar10_curr_off"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (8, 8, 4, 4)
pq_chunks = (5, 5, 5, 5)   # eff_vocab: 32768, 32768, 1024, 1024
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
