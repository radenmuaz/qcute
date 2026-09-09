"""Same as cifar10_stack_lag0.py, lag=16.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_lag16.py
"""

run_name = "cifar10_stack_lag16"

# --- model ---
img_size = 32
d_model = (512, 512, 512)
n_layers = (2, 2, 2)
n_heads = (4, 4, 4)
n_kv_heads = (None, None, None)
strides = (3, 4, 4)
code_vocab = 16
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 16

# --- training ---
batch_size = 64
n_devices = None
epochs = 200
lr = 0.01
lr_schedule = "warmup_cosine"
warmup_steps = 1000
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = None

# --- logging / eval ---
log_every = 1000
eval_every_epochs = 10
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
