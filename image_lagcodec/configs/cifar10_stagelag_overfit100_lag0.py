"""StageLagDecoder overfit sanity check (100-image subset), lag=0 -- own code only, block-
diagonal, degenerate G=1 case (now BOS-seeded, fully causal within/across groups -- verified
byte-exact reconstruct_full_recompute vs reconstruct_kv_cache at toy scale 2026-09-10). Same
encoder architecture as cifar10_stack_1_shallow.py for direct comparability.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stagelag_overfit100_lag0.py
"""

run_name = "cifar10_stagelag_overfit100_lag0"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = 4
pq_chunks = 5   # effective vocab = code_vocab**pq_chunks = 4**5 = 1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn_lag"
lag = 0

# --- training ---
batch_size = 16
n_devices = None
epochs = 3000
lr = 1e-2
lr_schedule = "warmup_const"
warmup_steps = 100
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 100

# --- logging / eval ---
log_every = 10   # steps_per_epoch=1 at this train_subset_n/batch_size, so log_every=200 made the
# "total" progress bar look frozen for long stretches (chat 2026-09-10) -- 10 gives frequent ticks.
eval_every_epochs = 500
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
