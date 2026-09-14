"""Fork of cifar10_stack_s4.py (chat 2026-09-11): code_vocab=256,pq_chunks=1 (single-chunk,
byte-sized codebook) instead of s4's vocab=16/pq=4 (16**4=65536) -- same eff_vocab=256**1=65536
per level, tests PQ chunking granularity (one big byte-alphabet-sized chunk vs four 16-way
chunks) as the one variable. Everything else identical to s4.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_s4_pq1v256.py
"""

run_name = "cifar10_stack_s4_pq1v256"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (256, 256, 256, 256, 256)
pq_chunks = (1, 1, 1, 1, 1)   # eff_vocab: 65536 per level (single chunk)
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
