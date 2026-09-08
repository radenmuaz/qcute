"""Same as lagcodec_overfit1000_selfattn_deep8_ncode64.py, but recon_ncode maxed out to
n_units_at(level) for every level -- single group per level == full continuous causal chain,
no group-boundary cold starts at all. No start_level override (defaults to top level, full
8-level reconstruct_tree chain). If free-running gen still fails here, the bug is in the
decoder forward pass / multi-level chaining itself, not the recon_ncode grouping mechanism.

n_units per level (SEQ_LEN=3072, strides=(2,)*8): L0=1536,L1=768,L2=384,L3=192,L4=96,L5=48,
L6=24,L7=12 -- recon_ncode set to exactly these values.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_deep8_ncode_max.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_deep8_ncode_max"

# --- model ---
img_size = 32
d_model = (256,) * 8
n_layers = (2,) * 8
n_heads = (4,) * 8
n_kv_heads = (None,) * 8
strides = (2,) * 8
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn"
recon_ncode = (1536, 768, 384, 192, 96, 48, 24, 12)

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
