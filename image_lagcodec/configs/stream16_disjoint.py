"""
KIV -- not launched yet (chat 2026-09-15). Written to illustrate the "disjoint chunks" point in
the spectrum, spot #2 of the axes table.
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/stream16_disjoint.py

DISJOINT preset: same fixed 16-code chunking as stream16_unbounded.py/stream16_window2.py, but
ncodes_window=0 -- zero cross-group context, every 16-chunk decodes fully independently of every
other chunk. Cheapest/fastest of the "fixed 16-chunk" family (no ctx window overhead at all), but
also least contextual -- a lower bound on quality vs stream16_window2.py (window=2) and
stream16_unbounded.py (window=-1) at the same grouping. streaming is moot here since window=0
means there's nothing to look back at either way.
"""


# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, )
n_layers = (4, 4, 4, 4,)
n_heads = (2, 2, 2, 2,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (256, 256, 256, 256,)
pq_chunks = (3, 3, 3, 3,)
mlp_mult = 2
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.1
mse_weight = 1.0
entropy_weight = 0.01

decoder_ncodes = 16
ncodes_window = 0   # disjoint: fixed 16-code chunks, zero cross-group context
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_prob = 0.5
quantize_drop = 0.5
init_scheme = "llama"
use_xsa = True
pq_dim = (128, 128, 128, 128)

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128, 128, 128)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 16
val_batch_size = 16
epochs_per_phase = (100, 100, 100, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 80
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "none"

# --- logging ---
log_every = 100
gen_eval_every = 100
ckpt_every = 100
ckpt_keep = 1
qual_gen_n = 16
