"""decoder_type="self_attn_lag", lag=1535 (= n_blocks-1, the MAX lag for strides=(2,2)'s
n_blocks=1536) -- ONE single group spanning the WHOLE sequence: every code must be known before
reconstructing ANY byte, full non-causal whole-sequence context, single autoregressive byte pass
at the end. The other extreme from lag4/lag0/self_attn_local's block-diagonal isolation.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_lagmax.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_lagmax"

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
lag = 1535  # n_blocks(=3072/2=1536) - 1 -- max lag, one group = whole sequence

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
