"""Fork of cifar10_stack_s4.py (chat 2026-09-11): code_vocab=4,pq_chunks=4 (4**4=256 eff_vocab
per level) instead of s4's vocab=16/pq=4 (65536) -- opposite extreme from pq1v256's single
256-way mega-chunk (which showed codebook collapse, util~0.03 vs s4's ~0.25): here MANY small
(4-way) chunks instead. Same eff_vocab as pq1v256 (256) but via 4 tiny chunks, testing whether
splitting into more/smaller chunks avoids the collapse pq1v256 showed. Everything else identical
to s4.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_s4_pq4v4.py
"""

run_name = "cifar10_stack_s4_pq4v4"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (4, 4, 4, 4, 4)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 256 per level (4 chunks of vocab=4)
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
