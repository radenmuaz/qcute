"""StageLagDecoder overfit sanity check (100-image subset), "deep" (6-level) encoder architecture
-- pairs with cifar10_stagelag_overfit100_lag0.py's "shallow" (4-level) one. lag=0 -- own code
only, block-diagonal, degenerate G=1 case, fully causal within/across groups. Also exercises the
new hierarchical cascade generation (HierEncoder.generate_lower_codes -- top/level4 code given
from the real image, levels 3/2/1/0 generated via each level's own trained NTP head reused
generatively, see chat 2026-09-10): saves both samples_epoch{N}_reconstruct.png (all codes real)
and samples_epoch{N}_cascade.png (only the top code real, everything below generated).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stagelag_overfit100_deep_lag0.py
"""

run_name = "cifar10_stagelag_overfit100_deep_lag0"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512, 512, 512)
n_layers = (2, 2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None, None)
strides = (3, 4, 4, 4, 4, -1)
code_vocab = 16
pq_chunks = 4
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
log_every = 200
eval_every_epochs = 500
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
