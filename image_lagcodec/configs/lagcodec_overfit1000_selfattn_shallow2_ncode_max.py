"""Simplest possible shallow case: 2-level hierarchy, strides=(2,2) (vs deep8's 8 levels),
recon_ncode maxed to n_units_at(level) for every level -- single group per level == full
continuous causal chain, no group-boundary cold starts. No start_level override (defaults to
top level, full 2-level reconstruct_tree chain). Minimal failure surface -- if free-running gen
fails here too, the bug is in the decoder forward pass / chaining itself, not depth or grouping.

n_units per level (SEQ_LEN=3072, strides=(2,2)): L0=1536, L1=768 -- recon_ncode set to these.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_shallow2_ncode_max.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_shallow2_ncode_max"

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
decoder_type = "self_attn"
recon_ncode = (1536, 768)

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
