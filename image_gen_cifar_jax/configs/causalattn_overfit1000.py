"""image_gen_cifar_jax/run_causalattn.py -- OVERFIT SANITY CHECK, not a real training run.
Trains on only the first 1000 CIFAR-10 images, capped at 50 epochs, to verify the
generation/save code path is correct: train bpb should collapse toward ~0 (memorization) and
samples_epoch{N}_traincompare.png should show generated images closely matching the ground
truth train images side by side. If it still looks like scanline noise even overfit this
hard, that's a real bug, not undertraining/architecture.

uv run python3 -m image_gen_cifar_jax.run_causalattn --config image_gen_cifar_jax/configs/causalattn_overfit1000.py
"""

run_name = "cifar_causalattn_overfit1000"

# --- model ---
d_model = 256
n_layers = 1
n_heads = 4
code_vocab = 16
pq_chunks = 4
decoder_mode = "seq"
col_group_size = 32  # MIMO -- full cross-column mixing
class_conditional = False
n_classes = 10

# --- training ---
batch_size = 25
n_devices = None
epochs = 5000
lr = 3e-4
warmup_steps = 50
seed = 0
train_subset_n = 1000

# --- logging / eval ---
log_every = 10
eval_every_epochs = 5
checkpoint_every_epochs = 10
qual_gen_n = 8
qual_gen_greedy = False
qual_gen_temperature = 0.05
