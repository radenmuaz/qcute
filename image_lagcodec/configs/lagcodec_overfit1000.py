"""image_lagcodec/run_lagcodec.py -- OVERFIT SANITY CHECK, not a real training run. Trains on
only the first 1000 CIFAR-10 images to verify the reconstruction path is correct: train byte_bpb
should collapse toward ~0 and samples_epoch{N}_reconstruct.png should show near-perfect
reconstruction vs ground truth once memorized.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000.py
"""

run_name = "cifar_lagcodec_overfit1000"

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
