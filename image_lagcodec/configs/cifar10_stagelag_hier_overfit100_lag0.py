"""StageLagDecoder-hierarchy overfit sanity check (100-image subset), lag=0. decoder_type=
"self_attn_lag_hier": one StageLagDecoder per level 0..N-2 (distinct weights, not shared with
EncoderLevel -- "duplicate EncoderLevel to DecoderLevel", chat 2026-09-10), each conditioned on
codes[level] and predicting codes[level-1] (or bytes for level=0). At generation time, chains
top-down from the given top code (codes[N-2]) all the way to bytes -- see run_reconstruct's
self_attn_lag_hier branch, samples_epoch{N}_cascade.png. Same encoder architecture as
cifar10_stagelag_overfit100_lag0.py (single-level decoder) for direct comparability.

lag is defined at the TOP used level (level2 here) and propagates DOWN via hier_stage_lags (chat
2026-09-10, matches StackDecoder's own lag convention): lag=0 means "own top-level code only, no
extra top-level context" -- but that still propagates to a full 16-code window at level1 and a
256-code window at level0 (one top-level code's worth), NOT a degenerate/minimal window at every
level the way a naive uniform-lag=0 would have meant.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stagelag_hier_overfit100_lag0.py
"""

run_name = "cifar10_stagelag_hier_overfit100_lag0"

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
decoder_type = "self_attn_lag_hier"
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
