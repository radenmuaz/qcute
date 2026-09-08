"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="cross_attn" (StageCrossAttnDecoder,
self-attn + separate cross-attn-to-own-code stack -- ablation pair with lagcodec_overfit1000_selfattn.py).
Trains on the first 1000 CIFAR-10 images: train byte_bpb should collapse toward ~0 and
samples_epoch{N}_reconstruct.png should show near-perfect reconstruction once memorized.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_crossattn.py
"""

run_name = "cifar_lagcodec_overfit1000_crossattn"

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
