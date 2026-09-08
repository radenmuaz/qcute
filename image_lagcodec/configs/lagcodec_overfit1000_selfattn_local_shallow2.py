"""Redo of the shallow2 overfit ablation with decoder_type="self_attn_local" -- block-diagonal
self-attn (StackDecoderLocal-inspired): every block decodes independently (zero cross-block
target-byte visibility), relying on the fact that its own code is already a causal summary of
everything upstream (HierEncoder is causal). forward() and generation are the IDENTICAL
computation by construction (n_blocks folded into batch either way) -- no recon_ncode grouping
construct, no K+1 cache bug, no group-boundary cold starts possible by design.

2-level hierarchy, strides=(2,2) (vs deep8's 8 levels) -- n_units: L0=1536, L1=768.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_local_shallow2.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_local_shallow2"

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
decoder_type = "self_attn_local"

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
