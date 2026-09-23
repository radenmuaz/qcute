"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/run12.py
"""

# --- model ---
img_size = 32
DEPTH = 4
d_model = (256,)*DEPTH
n_layers = (2,)*DEPTH
n_heads = (2,)*DEPTH
n_kv_heads = (None,)* DEPTH
code_vocab = (256,) *DEPTH
pq_chunks = (3,)* DEPTH
pq_dim = (128,) * DEPTH
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (2,)*DEPTH
decoder_ncodes = 1
ncodes_window = 16
attn_window = (256,)*DEPTH
attn_lookahead = 0
# Strict per-code causal decode WITH cond_depth support: decoder_ncodes=1 (one own-code + its target bytes per
# row) + gen_sync=True (decode_generate_pardec_sync, real cross-group byte history off a running buffer, never a
# same-step hallucination -- see the 2026-09-22 masking fix) + stream_chunks default 0 (chunk_groups=1, so
# _pardec_ctx_rows' own-ctx AND cond_depth extra-ctx windows both use the EXACT per-group boundary (g+1)*G, no
# chunk-rounding, no leak). dense_decode can't do this (decode_logits_and_target/decode_generate don't accept
# extra_ctx at all); this is the cond_depth-capable equivalent of it, at the cost of _pardec_ctx_rows' padded
# (not zero-copy) windows rather than a real growing cache.
interleave_decode = True  # hardcoded cond_depth<=2 interleave, verified 2026-09-22 (see test)
cond_depth = (2, 2, 2, 1)  # each level conditions on the next coarser level's own codes; top level has none coarser
# decode_future left off: interleave_decode ignores it (same as dense_decode)

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
level_gt_drop = 0.8
quantize_drop = 0.8
# mse_softmax_tau = 1.0
# feedback_p = 0.5
# feedback_p = 0.0


init_scheme = "llama"
# use_xsa = True
# use_sink = True
use_xsa = False
use_sink = False
# precision = "fp32"
precision = "bf16"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (128,)*DEPTH
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8  # interleave_decode is a single real growing cache, not gen_sync's padded-wave approach -- try full batch first
level_steps = (5_000, 5_000, 5_000, 30_000)  # staged: each phase adds one level (no_freeze)
seed = 0
# warmup_steps = 2
train_subset_n = None
val_subset_n = None
# train_subset_n = 100
# val_subset_n = 10
# train_subset_n = 100
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = 43_000  # near the end of all phases (45k total)
warmup_steps = int(1e3)
# lr_min_epoch = 50
# lr_min_epoch = 400
# weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = dict(
                        weight_decay=1e-5,
                        # b1=0.8, b2=0.9,
                        #  eps=1e-8,eps_root=0.0,
                        #  nesterov=False

)

wa_mode = "none"

# wa_mode = "ema"
# wa_every_step = 100
# wa_ema_decay = 0.9
# wa_verbose = False

# wa_mode = "wma"
# wa_every_epoch = 1
# wa_stack_size = 5
# wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1

'''
Quick intuition: Adam's b2 sets an effective averaging window of 1/(1-b2) steps for the second-moment estimate. Default b2=0.999 → ~1000-step window. The rule of thumb: that window should be well under your total step count, or the optimizer never leaves its warmup regime.


b2 window = 1/(1-b2), keep it « total steps
n=100 (short phase): b2≈0.95, b1≈0.85-0.9
n=1000: b2≈0.97-0.98
n=50k (long schedule): keep defaults b1=0.9, b2=0.999
'''

'''
audit your code tpu1, overfit logs/overfit1/ checkpoint, i ran and still bad generation
check config there in that folder
run on tpu1 use tpu
'''