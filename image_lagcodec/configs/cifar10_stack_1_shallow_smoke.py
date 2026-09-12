"""Fast smoke test forked from cifar10_stack_1_shallow.py -- train_subset_n=10, batch_size=2,
n_devices=1, epochs=1, eval_every_epochs=1: verifies training + the qual-gen/reconstruct_kv_cache
path (the bf16/fp32 dtype crash site) end-to-end in minutes, not the ~1h it took to hit epoch 10
on the real run.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_1_shallow_smoke.py
"""

run_name = "cifar10_stack_1_shallow_smoke"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = 16
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 0

# --- training ---
batch_size = 2
n_devices = 1
epochs = 1
lr = 0.01
lr_schedule = "warmup_const"
warmup_steps = 1
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 10

# --- logging / eval ---
log_every = 1
eval_every_epochs = 1
qual_gen_n = 2
qual_gen_greedy = True
qual_gen_temperature = 1.0
