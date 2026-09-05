"""image_gen_cifar_jax/run_ar_clockwork.py -- OVERFIT SANITY CHECK (sequential/DeepSeek-MTP-
style RGB head), not a real training run. Trains on only the first 1000 CIFAR-10 images,
capped at 50 epochs, to verify the generation/save code path is correct: train bpb should
collapse toward ~0 (memorization) and samples_epoch{N}_traincompare.png should show generated
images closely matching the ground truth train images side by side.

uv run python3 -m image_gen_cifar_jax.run_ar_clockwork --config image_gen_cifar_jax/configs/ar_clockwork_overfit1000_sequential.py
"""

run_name = "cifar_ar_clockwork_overfit1000_sequential"

# --- model ---
embed_dim = 256
d_model = (256, 256, 128, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (1, 2, 4, 1)
mlp_mult = 4
rope_base = 10000.0
class_conditional = False
n_classes = 10
row_weight = 1.0
ntp_weight = 1.0
head_type = "sequential"
mtp_dim = 640        # param-parity with the parallel head (~6.9M vs ~6.29M) -- previous 64
# gave the sequential head only ~96K params, a 65x capacity gap vs parallel, which was
# starving it, not a bug in the chaining logic itself (already verified structurally correct)
mtp_n_heads = 2      # kept at 2 (not scaled with mtp_dim) so the extra capacity goes into the
# MLP/embedding width rather than inflating per-head attention capacity over a 3-token sequence
mtp_mlp_mult = 4

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
