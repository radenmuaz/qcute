"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="self_attn" (StageSelfAttnDecoder,
plain causal self-attn only -- this level's own code embedded as each block's start token instead
of a separate cross-attn stack. Ablation pair with lagcodec_overfit1000_crossattn.py -- same
model/training hparams otherwise, so the only difference is decoder_type).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn"

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
# dec_d_model/dec_n_layers/dec_n_heads/dec_n_kv_heads left None -> mirror d_model/n_layers/n_heads/n_kv_heads above

# --- training ---
batch_size = 16
n_devices = None
epochs = 1000
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-5
optimizer = "adamw"
optimizer_kwargs = {}
seed = 0
train_subset_n = 1000

# --- logging / eval ---
log_every = 20
eval_every_epochs = 20
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
