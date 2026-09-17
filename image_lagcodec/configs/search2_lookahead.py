"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/search2_lookahead.py

8-way random search off run_1.py (top_level_trainable base), chat 2026-09-17. Ablates attn_lookahead=2, decode_past=0, decode_future=0 on level0 (2^3 factorial across search1-8, this is run 2/8).
"""

# --- model ---
img_size = 32
d_model = (128, 128, 128, 128,)
n_layers = (1, 1, 1, 1)
n_heads = (2, 2, 2, 2)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, 4)   # was (4,4,4,-1) -- real last stride opts into top_level_trainable
code_vocab = (256, 256, 256, 256,)
pq_chunks = (3, 3, 3, 3,)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.1
mse_weight = 1.0
entropy_weight = 0.1


decoder_ncodes = (16, 16, 16, 4)   # level3 (top, new) only has 4 own codes -- avoid the clamp warning
ncodes_window = -1   # streaming: fixed 16-code chunks, unbounded lookback into past chunks
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 1.0
gumbel_at_inference = False
cascade_rollout_drop = 0.1
quantize_drop = 0.9
# mse_softmax_tau = 1.0
init_scheme = "llama"
use_xsa = True
use_attn_sink = True
precision = "fp32"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
pq_dim = (128, 128, 128, 128)
token_dim = (128, 128, 128, 128)
# token_dim = (64, 64, 64, 64,)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 16
val_batch_size = 16
phase_epochs = (5, 5, 100, 100)   # 4th entry added -- n_phases is now n_levels=4 (top level trained too)
warmup_steps = 1000
grad_clip = 10.0
seed = 0
# warmup_steps = 2
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 50
# lr_min_epoch = 400
weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = {}

# wa_mode = "none"

wa_mode = "ema"
wa_every_epoch = 1
wa_ema_decay = 0.9

# wa_mode = "wma"
# wa_every_epoch = 1
# wa_stack_size = 5
# wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
gen_eval_every_epoch = 10
ckpt_every_epoch = 100
ckpt_keep = 1

attn_lookahead = (2, 0, 0, 0)
decode_past = (0, 0, 0, 0)
decode_future = (0, 0, 0, 0)
