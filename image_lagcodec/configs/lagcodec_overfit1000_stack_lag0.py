"""decoder_type="stack" (generalized StackDecoder, 2026-09-08 rewrite: any depth via a uniform
self-attn+cross-attn+mlp block per level, BOS-prepended standard-shifted NTP, lag scheduling
knob). lag=0 -- baseline, each byte position sees only the 1 topmost-consulted-level code whose
span it falls within (no extra lookahead). strides=(3,4,4): level1=1024 codes (3 bytes each),
level2=256 codes (12 bytes each, topmost consulted -- level3's own output hard-excluded).
lag_bytes=(0+1)*12=12.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_stack_lag0.py
"""

run_name = "cifar_lagcodec_overfit1000_stack_lag0"

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
decoder_type = "stack"
lag = 0

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
