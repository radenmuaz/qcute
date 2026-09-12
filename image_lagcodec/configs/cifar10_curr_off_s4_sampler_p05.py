"""Fork of cifar10_curr_off_s4.py (chat 2026-09-12), run via run_lagcodec_sampler (NOT
run_lagcodec_curriculum) -- cascade rollout sampler, cascade_rollout_prob=0.5: per training
step, a global coin flip picks between plain generalized-teacher-forcing (every active level's
decode() conditioned on its own real encoder code) and the chained cascade-simulated ctx (every
level below the topmost conditioned on a cheap non-autoregressive pseudo-code derived from the
level above's own decode logits -- see run_lagcodec_sampler.py's module docstring for the full
mechanism). gumbel OFF (quantize_mode="argmax", default) -- isolating the rollout sampler as the
only new variable vs the plain s4 baseline. curriculum_mode is force-checked "no_freeze". Same
architecture/optimizer/epochs_per_phase as s4.

uv run python3 -m image_lagcodec.run_lagcodec_sampler --config image_lagcodec/configs/cifar10_curr_off_s4_sampler_p05.py
"""

run_name = "cifar10_curr_off_s4_sampler_p05"

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
quantize_mode = "argmax"   # gumbel OFF -- isolate the rollout sampler as the only new variable
cascade_rollout_prob = 0.5

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
