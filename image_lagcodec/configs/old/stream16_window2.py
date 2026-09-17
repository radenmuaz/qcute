"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/stream16_window2.py

STREAMING preset, bounded-lookback variant of stream16_unbounded.py (chat 2026-09-15): same fixed
16-code chunking (decoder_ncodes=16, as if new encoder codes keep arriving 16 at a time), but
ncodes_window=2 instead of -1 -- each new 16-chunk only attends to the 2 immediately-preceding
chunks' ctx codes (Wg=(2+1)*16=48 ctx blocks), not everything decoded so far. Cheaper/more local
than stream16_unbounded.py's unbounded lookback; tests whether a bounded window loses meaningful
quality vs unbounded. streaming stays default True (causal growing prefix) -- ncodes_window=2 is
a genuinely bounded case, not -1, so streaming=False (fullctx) doesn't apply here.
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
ncodes_window = 2   # bounded: fixed 16-code chunks, lookback into only the 2 preceding chunks
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_drop = 0.5
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
phase_epochs = (100, 100, 100, )
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
gen_eval_every_epoch = 100
ckpt_every_epoch = 100
ckpt_keep = 1
qual_gen_n = 16
