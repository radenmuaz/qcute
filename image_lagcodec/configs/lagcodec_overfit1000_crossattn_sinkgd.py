"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="cross_attn", optimizer="sinkgd"
(warmup then CONSTANT lr, no decay -- warmup_const_schedule -- SinkGD is stateless so a decay
phase is less necessary). Ablation pair with lagcodec_overfit1000_selfattn_sinkgd.py -- same
model/training hparams otherwise, so the only difference is decoder_type. Also a second axis vs
lagcodec_overfit1000_crossattn.py: adamw+lr=5e-4 there, sinkgd+lr=1e-3 here.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_crossattn_sinkgd.py
"""

run_name = "cifar_lagcodec_overfit1000_crossattn_sinkgd"

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
decoder_type = "cross_attn"
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 2000
lr = 1e-3
warmup_steps = 100
weight_decay = 1e-5
optimizer = "sinkgd"
optimizer_kwargs = {}
seed = 0
train_subset_n = 1000

# --- logging / eval ---
log_every = 20
eval_every_epochs = 20
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
