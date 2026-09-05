"""image_gen_cifar_jax/run_ar_clockwork.py -- sandwich ClockworkRNN AR baseline.

Every field listed explicitly (even where it matches the code default) -- this run's
resolved_config.py and this file itself are both saved into the run dir, so nothing about
what actually ran is left implicit.

uv run python3 -m image_gen_cifar_jax.run_ar_clockwork --config image_gen_cifar_jax/configs/ar_clockwork_sandwich.py
"""

run_name = "cifar_ar_clockwork_sandwich_ntp"

# --- model ---
embed_dim = 256
d_model = (256, 256, 128, 256)   # sandwich: [fast input, ...filling..., fast collector]
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)  # None -> max(1, n_heads[i]//4) GQA-by-default
strides = (1, 2, 4, 1)            # strides[0]==1 (input), strides[-1]==1 (collector, forced)
mlp_mult = 4
rope_base = 10000.0
class_conditional = False
n_classes = 10
row_weight = 1.0         # main 32-ahead (full next-row) head loss weight
ntp_weight = 1.0         # NTP anchor head (next single pixel) loss weight -- aux only, unused at gen time

# --- training ---
batch_size = 32          # per-device
n_devices = None         # None -> all local devices
epochs = 300
lr = 3e-4
warmup_steps = 1000
seed = 0

# --- logging / eval ---
log_every = 50
eval_every_epochs = 1
qual_gen_n = 4
qual_gen_greedy = False   # False -> categorical sample at qual_gen_temperature; True -> argmax
qual_gen_temperature = 1.0
