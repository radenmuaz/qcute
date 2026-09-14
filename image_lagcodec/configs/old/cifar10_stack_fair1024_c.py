"""Fair-capacity ablation C (chat 2026-09-11): same goal as cifar10_stack_fair1024_a.py
(cumulative eff_vocab across the 3 used levels ~= 1024, matching a single VQ/FSQ code's budget
for a 16x16 patch) but keeping code_vocab=4 at every level (matches the current/prior configs'
granularity) and varying only pq_chunks per level: eff_vocab 16, 4, 16 -> cumulative
16*4*16=1024. Same d_model/lag/train_last_encoder choices as candidate A -- only code_vocab/
pq_chunks differ, isolating that one variable between the two runs.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_fair1024_c.py
"""

run_name = "cifar10_stack_fair1024_c"

# --- model ---
img_size = 32
d_model = (32, 256, 256, 0)   # d_model[-1]=0: don't-care, topmost level never built
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (4, 4, 4, 4)      # level3 (topmost, index -1) don't-care: train_last_encoder=False
pq_chunks = (2, 1, 2, 2)       # eff_vocab: 16, 4, 16 -> cumulative 16*4*16=1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 3   # max -- see cifar10_stack_fair1024_a.py docstring
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
