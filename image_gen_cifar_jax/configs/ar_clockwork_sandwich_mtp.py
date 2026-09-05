"""image_gen_cifar_jax/run_ar_clockwork.py -- sandwich ClockworkRNN AR baseline with the
DeepSeek-MTP-style SEQUENTIAL RGB head (tiny causal decoder chains R->G->B per column via
real byte embeddings, instead of the baseline's 3 independent linear heads). Same trunk as
ar_clockwork_sandwich.py -- only head_type/mtp_* differ. Compare against that run's logs.

Every field listed explicitly (even where it matches the code default) -- this run's
resolved_config.py and this file itself are both saved into the run dir, so nothing about
what actually ran is left implicit.

uv run python3 -m image_gen_cifar_jax.run_ar_clockwork --config image_gen_cifar_jax/configs/ar_clockwork_sandwich_mtp.py
"""

run_name = "cifar_ar_clockwork_sandwich_mtp"

# --- model (trunk, identical to ar_clockwork_sandwich.py) ---
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

# --- RGB output head: sequential (DeepSeek-MTP-style R->G->B chain), tiny/fast ---
head_type = "sequential"
mtp_dim = 64             # sequential head's internal width
mtp_n_heads = 2          # plain MHA (no GQA, no KV cache -- 3-step sequence recomputed fresh)
mtp_mlp_mult = 4

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
