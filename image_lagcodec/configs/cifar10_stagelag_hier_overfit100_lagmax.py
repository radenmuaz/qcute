"""StageLagDecoder-hierarchy overfit sanity check (100-image subset), lag=max. Same
decoder_type="self_attn_lag_hier" chain as cifar10_stagelag_hier_overfit100_lag0.py. cfg.lag is
defined at the TOP used level (level2 here, matching StackDecoder's own lag convention -- see its
docstring) and propagates DOWN via hier_stage_lags (chat 2026-09-10): one top-level group spans
strides[i] as many codes at the level below, so every level sees the same real image span. Level2
has n_blocks=SEQ_LEN/(3*16*16)=4, so its true max is lag=3 (NOT 1023 -- that was the byte-level's
own max under the old, wrong "same numeric lag at every level" scheme, which would now propagate
into catastrophic over-padding at the lower levels). lag=3 propagates to exactly n_blocks at every
level (level1: 4*16=64, level0: 4*16*16=1024) -- one full, zero-padded group per level.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stagelag_hier_overfit100_lagmax.py
"""

run_name = "cifar10_stagelag_hier_overfit100_lagmax"

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
lag = 3   # top level's true max (n_blocks(level2)-1 = 4-1); propagates down via hier_stage_lags

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
