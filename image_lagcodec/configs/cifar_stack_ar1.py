"""
uv run python3 -m image_lagcodec.run_lagcodec_stack --config image_lagcodec/configs/cifar_stack_ar1.py
"""
# Fork of cifar_ar5_cd.py, pointed at the new cross-attention decoder (run_lagcodec_stack.py,
# pardec/interleave_decode's flat self-attn sequence replaced by self-attn + cond_depth cross-attn
# sublayers per layer). cond_depth=(2,1) kept as-is -- level0 cross-attends to its own code AND
# level1's code (2 sources), exercising the extra_ctx_code_soft path (only cond_depth=1 tested so
# far, via the stack fork's own cifar_ar2_bigger.py smoke run). interleave_decode/decoder_ncodes/
# ncodes_window/decode_past/level_refine_window are IGNORED under run_lagcodec_stack.py (kept here
# only because they're harmless leftovers from the fork source, not because they do anything).

# --- model ---
img_size = 32

# d_model = (512, 512)
# n_layers = (4, 4)
# n_heads = (2, 2)
# n_kv_heads = (None, None)

d_model = (512, 512)
n_layers = (8, 8)  # was (4, 4)
n_heads = (2, 2)
n_kv_heads = (None, None)

code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (256, 256)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 1.0

interleave_decode = True  # ignored by run_lagcodec_stack.py -- stack decoder is the only mechanism
attn_window = 1024
strides = (4, 4)
decoder_ncodes = 4  # ignored by run_lagcodec_stack.py
ncodes_window = 0  # ignored
attn_lookahead = 0
decode_past = 0  # ignored
level_refine_window = 1  # ignored
cond_depth = (2, 1)  # level0: own code + level1's code (2 cross-attn sources); level1: own code only

additive_drop_loss = False
weight_sharing = True  # decoder is a strict superset of encoder (same self-attn+mlp, decoder adds
# real cross-attn conditioning; encoder = same blocks with every cross-attn rung getting the
# learned sink no-op) -- sharing follows directly from that, not just a memory-saving shortcut
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
precision = "bf16"
remat_level = True

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 16
val_batch_size = 16
level_steps = (int(2e3), int(5e4))
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 2000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(5e4)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
