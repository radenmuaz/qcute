"""Fork of cifar10_stack_s4_pq4v4.py (chat 2026-09-11): code_vocab=8 (not 4), pq_chunks=4 kept
-- 8**4=4096 eff_vocab per level (between pq4v4's 256 and s4's 65536). pq4v4 showed much
healthier codebook utilization than the single-mega-chunk pq1v256 (0.656 vs 0.032) but trailed
s4's raw accuracy at the same epoch -- this tests whether a bit more per-chunk vocab (8 instead
of 4, same 4-chunk split) recovers accuracy while keeping utilization healthy. Everything else
identical to pq4v4/s4.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_s4_pq4v8.py
"""

run_name = "cifar10_stack_s4_pq4v8"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (8, 8, 8, 8, 8)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 4096 per level (4 chunks of vocab=8)
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 11   # max -- see cifar10_stack_s4.py docstring
train_last_encoder = False

# --- training ---
batch_size = 16
n_devices = None
epochs = 200
lr = 5e-4
lr_schedule = "warmup_cosine"
warmup_steps = 1000
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
# sinkgd (previous optimizer here, chat 2026-09-11) -- do not delete, comment/uncomment to swap:
# lr = 0.01
# weight_decay = 0
# optimizer = "sinkgd"
# optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = None

# --- logging / eval ---
log_every = 1000
eval_every_epochs = 10
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
