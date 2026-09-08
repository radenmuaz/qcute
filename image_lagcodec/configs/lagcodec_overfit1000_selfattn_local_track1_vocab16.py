"""Same as lagcodec_overfit1000_selfattn_local_track1.py, code_vocab=16 (was 8) -- more per-block
capacity (16^4=65536 codes vs 8^4=4096), testing whether the speckle/salt-and-pepper error
pattern (audit: wrong_pixel_rate=0.81, near-random spatially, block-level byte precision only
~43% at code_vocab=8) improves with more code capacity per block.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_local_track1_vocab16.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_local_track1_vocab16"

# --- model ---
img_size = 32
d_model = (256,) * 3
n_layers = (2,) * 3
n_heads = (4,) * 3
n_kv_heads = (None,) * 3
strides = (2, 2, 1)
code_vocab = 16
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn_local_track1"

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
