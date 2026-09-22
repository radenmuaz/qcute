"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/d2_chunk2.py
"""
# Fork of d2_eager.py (same decoder_ncodes=1): "2 chunks, independent across chunks, shared window within" --
# stream_chunks=2 (exactly 2 chunks spanning each level's groups). Chunk A and chunk B never see each other
# (still strictly causal); every group INSIDE one chunk shares that chunk's identical rounded window (the
# "interleaved"/shared-visibility rounding). Still pardec: groups remain independent/parallel-batched.

# --- model ---
img_size = 32
d_model = (256, 256)
n_layers = (4, 4)
n_heads = (2, 2)
n_kv_heads = (None, None)
code_vocab = (64, 64)
pq_chunks = (8, 8)
pq_dim = (128, 128)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (4, 4)
decoder_ncodes = 1
stream_chunks = 2  # exactly 2 chunks per level -- independent across the chunk boundary, shared window within
ncodes_window = 4
attn_lookahead = 0
decode_past = 0
decode_future = 4
level_refine_window = 1
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_passes = 2
cycle_refine_passes = 1
cond_depth = (2, 1)
cond_drop = 0.5

additive_drop_loss = True
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.9
quantize_drop = 0.9

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "fp32"

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e3), int(1e5))
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 1000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e5)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-3)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 1000
ckpt_keep = 1
