"""Fork of cifar10_curr_off_s4_sampler_p10.py (chat 2026-09-12): switched module to
run_lagcodec_reinforce (no curriculum, dual-pass REINFORCE term). Also: code_vocab 16->8
(eff vocab 4096, down from 65536 -- capacity reduction per user request), d_model 256->512,
mlp_mult 4->2. reinforce_weight=0.1 (one of 4 parallel ablations: 0.1/0.2/0.5/1.0).

uv run python3 -m image_lagcodec.run_lagcodec_reinforce --config image_lagcodec/configs/cifar10_reinforce_w01.py
"""

run_name = "cifar10_reinforce_w01"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512, 512)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (8, 8, 8, 8, 8)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 4096 per level
mlp_mult = 2
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
reinforce_weight = 0.1

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
