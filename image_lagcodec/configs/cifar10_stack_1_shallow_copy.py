"""Ablation of cifar10_stack_1_shallow.py: kv_lm_mode="copy" -- the decoder's cross-attn KV for
each consulted level is produced by a FRESH, independent clone of that level's own transformer
(same depth/shape as the corresponding encoder level, own dedicated embedding table, own weights
-- no coupling to the encoder's objective, still gets the context-aware win over plain identity).
Everything else identical to the base shallow config.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_1_shallow_copy.py
"""

run_name = "cifar10_stack_1_shallow_copy"

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
kv_lm_mode = "copy"

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
