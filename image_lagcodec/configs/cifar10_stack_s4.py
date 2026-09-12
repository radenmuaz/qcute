"""Fork of cifar10_stack_fair1024_b.py (chat 2026-09-11): full dataset (train_subset_n=None)
StackDecoder run at the SAME uniform stride=4 x4-levels shape as the curriculum s4 family
(strides=(4,4,4,4,-1), code_vocab=16/pq_chunks=4 uniform -> eff_vocab=65536 per level) -- no
capacity-splitting experiment here, goal is just a normal, adequately-provisioned fit (fair1024's
whole point was avoiding UNDER-capacity, so give every level full headroom). d_model/n_layers/
n_heads uniform across levels; lag=11 (max: lag_bytes=(lag+1)*prod(strides[:-1])=(11+1)*256=3072
=SEQ_LEN, whole image's codes visible before the byte decoder starts, per StackDecoder's lag
convention). Same train_last_encoder=False/optimizer/epochs as fair1024_b.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_s4.py
"""

run_name = "cifar10_stack_s4"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 65536 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 11   # max -- see docstring
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
