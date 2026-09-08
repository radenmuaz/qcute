"""image_lagcodec/run_lagcodec.py -- hierarchical causal encoder (PQ-quantized codes + per-
level non-circular NTP loss) + flat per-byte AR reconstruction decoder (interleaved RGB,
3072 bytes/image, own_code_min_lag=0 hardcoded). Task 1 only: reconstruction quality, no
free-running rollout yet.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_base.py
"""

run_name = "cifar_lagcodec_base"

# --- model ---
img_size = 32
d_model = (256, 256, 256)
n_layers = (2, 2, 2)
n_heads = (4, 4, 4)
n_kv_heads = (None, None, None)
strides = (3, 4, 4)   # short 3-level hierarchy; top level ends with 3072/(3*4*4)=64 codes
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0

# --- training ---
batch_size = 16          # per-device (3072-length sequences are much longer than before)
n_devices = None
epochs = 300
lr = 3e-4
warmup_steps = 1000
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
seed = 0

# --- logging / eval ---
log_every = 50
eval_every_epochs = 1
qual_gen_n = 4
qual_gen_greedy = True     # reconstruction is deterministic argmax by default (no sampling noise)
qual_gen_temperature = 1.0
