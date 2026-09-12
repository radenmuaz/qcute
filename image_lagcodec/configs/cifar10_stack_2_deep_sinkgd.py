"""Fork of cifar10_stack_2_deep.py (chat 2026-09-11) -- re-enabling sinkgd after the adamw run
(256-dim, lr=5e-4, wd=1e-2) produced blurry/flat reconstructions despite moderate val_acc
(~0.50-0.53 at epoch~99), suspected mode-collapse. Same architecture, sinkgd restored.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_2_deep_sinkgd.py
"""

run_name = "cifar10_stack_2_deep_sinkgd"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None, None)
strides = (3, 4, 4, 4, 4, -1)
code_vocab = 4
pq_chunks = 5   # effective vocab = code_vocab**pq_chunks = 4**5 = 1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 0

# --- training ---
batch_size = 16
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
