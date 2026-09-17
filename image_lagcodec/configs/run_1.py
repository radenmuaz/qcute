"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/run_1.py

chat 2026-09-17: fixed the "wasted top level" semantics bug -- the top level (last tuple entry)
used to be structurally unused: strides[-1]=-1 was a don't-care sentinel, has_decoder=(i<n-1)
skipped it, and phase_forward's encoder loop (capped at n_phases=n_levels-1) never reached it
either. Its d_model/n_heads/etc still allocated real params (they were deliberately starved to
d_model=1/n_heads=1 in the old version of this file specifically BECAUSE they were known-dead
weight) that never received gradient. Fix: strides[-1] is now a REAL stride (opts into
top_level_trainable=True in Config.__post_init__) -- this gives the top level a real decoder too,
and n_phases becomes n_levels (not n_levels-1), so phase_epochs (and every other per-phase tuple)
needs one MORE entry than before. d_model/n_heads on the top level restored to real values to
match. This fix is OPT-IN (gated on strides[-1] != -1) -- every other existing config keeps
strides[-1]=-1 and is completely unaffected.
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
