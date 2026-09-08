"""image_gen_cifar_jax/run_ar_clockwork.py -- FULL CIFAR-10 training (sequential/DeepSeek-MTP
RGB head), same scaled-up 6-level rise-then-fall sandwich (1,2,4,4,2,1) and mtp_dim=640 for
param parity (~27.4M) with the parallel/diffusion sibling configs. Serves as the comparison
point against the new diffusion-head run (both on the full 50k train split, no train_subset_n),
aiming to overfit as hard/fast as possible -- val is not the target and is expected to worsen.
1 epoch warmup -> 10 epochs flat at peak lr -> cosine decay for the remaining ~2989 epochs,
minimal (not zero) weight decay.

uv run python3 -m image_gen_cifar_jax.run_ar_clockwork --config image_gen_cifar_jax/configs/ar_clockwork_full_sandwich6_sequential.py
"""

run_name = "cifar_ar_clockwork_full_sandwich6_sequential"

# --- model ---
embed_dim = 256
d_model = (256, 320, 384, 384, 320, 256)
n_layers = (2, 2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None, None)
strides = (1, 2, 4, 4, 2, 1)
mlp_mult = 4
rope_base = 10000.0
class_conditional = False
n_classes = 10
row_weight = 1.0
ntp_weight = 1.0
head_type = "sequential"
mtp_dim = 640
mtp_n_heads = 2
mtp_mlp_mult = 4

# --- training ---
batch_size = 32
n_devices = None
epochs = 3000
lr = 5e-4
lr_decay = True
warmup_epochs = 1
constant_epochs = 10
min_lr_ratio = 0.01
weight_decay = 1e-5
warmup_steps = 1000      # unused when lr_decay=True, kept for CLI compat
seed = 0
# train_subset_n intentionally omitted -- full 50k train split

# --- logging / eval ---
log_every = 50
eval_every_epochs = 50
checkpoint_every_epochs = 50
qual_gen_n = 8
qual_gen_greedy = False
qual_gen_temperature = 0.05
