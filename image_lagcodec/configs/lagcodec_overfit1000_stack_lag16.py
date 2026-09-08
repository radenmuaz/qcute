"""Same as lagcodec_overfit1000_stack_lag0.py, lag=16 (17 topmost-consulted-level codes visible
from the start, lag_bytes=(16+1)*12=204 bytes).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_stack_lag16.py
"""

run_name = "cifar_lagcodec_overfit1000_stack_lag16"

# --- model ---
img_size = 32
d_model = (256, 256, 256)
n_layers = (2, 2, 2)
n_heads = (4, 4, 4)
n_kv_heads = (None, None, None)
strides = (3, 4, 4)
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 16

# --- training ---
batch_size = 16
n_devices = None
epochs = 3000
lr = 1e-2
warmup_steps = 100
weight_decay = 1e-5
optimizer = "sinkgd"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging / eval ---
log_every = 200
eval_every_epochs = 1000
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
