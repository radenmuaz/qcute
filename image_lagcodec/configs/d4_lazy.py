"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/d4_lazy.py
"""
# Deep (DEPTH=4, stride=2) analog of d2_lazy.py: "full lazy, wait for the whole level" -- decoder_ncodes=1,
# stream_chunks=1. interleave_decode is ALWAYS one fully sequential causal chain regardless of stream_chunks
# (a no-op there, confirmed 2026-09-22), so this variant only exists under pardec -- interleave_decode is OFF
# here (unlike run12/run13), same DEPTH=4/stride=2 model shape otherwise. Groups stay independent/parallel-
# batched (pardec); only the context-window visibility is maximally lazy (waits for the whole level).

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
stream_chunks = 1  # 1 chunk = wait for the whole level before any group's window is used (fully offline)
ncodes_window = 16
attn_window = (256,)*DEPTH
attn_lookahead = 0
cond_depth = (2, 2, 2, 1)  # each level conditions on the next coarser level's own codes; top level has none coarser
# interleave_decode NOT set (pardec instead) -- this variant needs the independent/parallel-group property

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
level_gt_drop = 0.8
quantize_drop = 0.8

init_scheme = "llama"
use_xsa = False
use_sink = False
precision = "bf16"

byte_group = 3
token_head_type = "ar"
token_dim = (128,)*DEPTH
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (5_000, 5_000, 5_000, 30_000)
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 5000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = 43_000
warmup_steps = int(1e3)
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
