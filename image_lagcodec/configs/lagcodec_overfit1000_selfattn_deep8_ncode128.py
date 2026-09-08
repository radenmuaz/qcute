"""Same as lagcodec_overfit1000_selfattn_deep8_ncode64.py, recon_ncode=(128,)*8.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_deep8_ncode128.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_deep8_ncode128"

# --- model ---
img_size = 32
d_model = (256,) * 8
n_layers = (2,) * 8
n_heads = (4,) * 8
n_kv_heads = (None,) * 8
strides = (2,) * 8
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn"
recon_ncode = (128,) * 8

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
start_level = 0
