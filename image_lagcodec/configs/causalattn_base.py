"""image_gen_cifar_jax/run_causalattn.py -- causal hierarchical PQ-quantized encoder + NTP priors.

Every field listed explicitly (even where it matches the code default) -- this run's
resolved_config.py and this file itself are both saved into the run dir.

uv run python3 -m image_gen_cifar_jax.run_causalattn --config image_gen_cifar_jax/configs/causalattn_base.py
"""

run_name = "cifar_causalattn_base"

# --- model ---
d_model = 256
n_layers = 1
n_heads = 4
code_vocab = 16            # per-chunk PQ codebook width
pq_chunks = 4               # combinatorial capacity code_vocab**pq_chunks
decoder_mode = "seq"        # "seq" or "mtp"
col_group_size = 1          # 1=SISO, img_size=MIMO, else grouped
class_conditional = False
n_classes = 10

# --- training ---
batch_size = 32             # per-device
n_devices = None            # None -> all local devices
epochs = 300
lr = 3e-4
warmup_steps = 1000
seed = 0

# --- logging / eval ---
log_every = 50
eval_every_epochs = 1
qual_gen_n = 4
qual_gen_greedy = False     # False -> categorical sample at qual_gen_temperature; True -> argmax
qual_gen_temperature = 1.0
