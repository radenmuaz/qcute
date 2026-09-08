"""decoder_type="self_attn_lag", lag=4 -- groups of 5 CONSECUTIVE top-level (level1) codes
prepended before their combined 5*K bytes; every byte in a group can causally see all 5 codes
(4 of them causally LATER than its own), block-diagonal across groups. n_blocks=1536 (strides=
(2,2)) is not divisible by 5 -- StageLagDecoder pads internally (verified toy-scale, see
run_lagcodec.py's StageLagDecoder.forward docstring), no special handling needed here.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_lag4.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_lag4"

# --- model ---
img_size = 32
d_model = (256,) * 2
n_layers = (2,) * 2
n_heads = (4,) * 2
n_kv_heads = (None,) * 2
strides = (2, 2)
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn_lag"
lag = 4

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
