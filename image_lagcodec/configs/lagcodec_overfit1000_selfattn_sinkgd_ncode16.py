"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="self_attn", optimizer="sinkgd"
lr=5e-4, recon_ncode=(16,16,16). Identical training to lagcodec_overfit1000_selfattn_sinkgd_ncode8.py
(recon_ncode is inference-time only) -- differs only in how the periodic reconstruction groups
sibling blocks into a shared local AR chain (groups of 16, here).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_sinkgd_ncode16.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_sinkgd_ncode16"

# --- model ---
img_size = 32
d_model = (256, 256, 256)
n_layers = (2, 2, 2)
n_heads = (4, 4, 4)
n_kv_heads = (None, None, None)
strides = (3, 4, 4)
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn"
recon_ncode = (16, 16, 16)
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 2000
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
