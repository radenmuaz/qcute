"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, decoder_type="self_attn", optimizer="sinkgd",
recon_ncode=(4,4,4). Identical training to lagcodec_overfit1000_selfattn_sinkgd.py (recon_ncode
is inference-time only) -- this run only differs in how the periodic reconstruction groups
sibling blocks (groups of 4, here, vs isolated singles there / pairs in _ncode2.py).

Motivation: a single code position's capacity is capped at code_vocab^pq_chunks = 8^4 = 4096
distinct values -- too small on its own. recon_ncode>1 lets later blocks in a merged AR chain
condition on already-decoded SIBLING blocks' codes too (not just the shared parent context),
drawing on combined information from multiple code positions instead of being bottlenecked by
any single one's 4096-value ceiling.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_sinkgd_ncode4.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_sinkgd_ncode4"

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
recon_ncode = (4, 4, 4)
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
