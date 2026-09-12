"""Fork of cifar10_curr_off_s4_sampler_p05.py (chat 2026-09-12) -- best result of the whole
session (final CASCADE_acc=0.379, only run to survive phase3->phase4 intact). Same cascade
rollout sampler mechanism (cascade_rollout_prob=0.5), same everything else, EXCEPT code_vocab
16->8 (pq4v8: pq_chunks=4, code_vocab=8, eff_vocab=4096 per level, down from 65536) -- testing
whether the smaller per-level codebook (matching the capacity used in the reinforce ablations)
helps or hurts the sampler's best-known config.

uv run python3 -m image_lagcodec.run_lagcodec_sampler --config image_lagcodec/configs/cifar10_curr_off_s4_sampler_p05_pq4v8.py
"""

run_name = "cifar10_curr_off_s4_sampler_p05_pq4v8"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (8, 8, 8, 8, 8)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 4096 per level
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
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
