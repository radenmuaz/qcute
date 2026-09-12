"""Fork of cifar10_curr_off_s4.py (chat 2026-09-11): quantize_mode="gumbel", temperature=1.0 --
tests whether giving the decoder a more diverse teacher-forced ctx/target distribution during
training (stochastic code sampling instead of fixed argmax) closes the cascade exposure-bias gap
seen in both freeze and no_freeze s4 runs (final CASCADE_acc ~0.003-0.006 either way). Gumbel is
TRAINING-only by default (gumbel_at_inference=False) -- eval/inference still uses plain argmax.
Everything else identical to s4 (no_freeze, 1000 epochs/phase, same architecture).

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curr_off_s4_gumbel1.py
"""

run_name = "cifar10_curr_off_s4_gumbel1"

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
gumbel_temperature = 1.0
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
