"""image_lagcodec/run_lagcodec.py -- SANITY CHECK, decoder_type="cross_attn" (the ORIGINAL
StageCrossAttnDecoder: self-attn + a SEPARATE cross-attention stack conditioning on level0's own
code), decoding level0 DIRECTLY -- start_level=0 (default), no multi-level staging, recon_ncode
left at default (1,1,1). Same training hparams as the self_attn sinkgd ablation configs (sinkgd
lr=1e-2, train_subset_n=100, epochs=2000) for a direct, apples-to-apples comparison against
StageSelfAttnDecoder now that both reconstruct_group implementations share the same (K+1)-step
KV-cache fix.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_crossattn_sinkgd_sanity.py
"""

run_name = "cifar_lagcodec_overfit1000_crossattn_sinkgd_sanity"

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
# recon_ncode left None -> defaults to (1,1,1); start_level left at CLI default 0

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
