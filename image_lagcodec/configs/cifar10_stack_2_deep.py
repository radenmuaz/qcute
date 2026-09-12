"""6-level encoder, four intermediate levels, same total byte-span as cifar10_stack_1_shallow.py
(768 bytes = 32x8x3 raster strip, byte-count proxy for a 16x16x3 patch): strides=(3,4,4,4,4,-1)
-- level0 groups RGB (stride 3), levels 1-4 each group 4 codes from the level below (stride 4
each, cumulative 3*4*4*4*4=768 bytes/code), level5 is the discarded top (stride=-1 sentinel).
Decoder consults levels 0-4 (n_consulted=5).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_2_deep.py
"""

run_name = "cifar10_stack_2_deep"

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
lr = 5e-4
lr_schedule = "warmup_cosine"
warmup_steps = 1000
weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = {}
# previous sinkgd setup (kept for reference, not deleted):
# lr = 0.01
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
