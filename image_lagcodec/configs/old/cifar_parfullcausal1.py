"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar_parfullcausal1.py
"""
# Fork of cifar_par1.py: "full causal parallel decode" -- ncodes_window=-1 (stream_chunks=0), i.e.
# pardec's own documented "causal unbounded" mode (see run_lagcodec.py's own warning at the
# ncodes_window==-1 branch): every group still decodes as its own independent batched row (pardec
# stays parallel), but each group's context window Wg is the FULL causal prefix up to its own
# position instead of a small fixed slice -- later groups get strictly more real context than
# earlier ones, all still computed in one batched call.
#
# Batching-efficiency tradeoff (straight from that same warning): Wg is a FIXED tensor width
# (=n_blocks_p) for every row regardless of position, so cost is O(n_groups * n_blocks) -- a small
# decoder_ncodes (many groups) blows this up badly. The warning's own rule of thumb is
# decoder_ncodes >= n_blocks/8; picked exactly at that threshold here for 8 groups at both levels
# (level0: n_blocks=1024/4=256 -> decoder_ncodes=32; level1: n_blocks=256/4=64 -> decoder_ncodes=8),
# the smallest/most-parallel group count the codebase's own guidance still calls safe.

# --- model ---
img_size = 32

# d_model = (512, 512)
# n_layers = (4, 4)
# n_heads = (2, 2)
# n_kv_heads = (None, None)

d_model = (256, 256)
n_layers = (4, 4)  # cheap encoder
n_heads = (2, 2)
n_kv_heads = (None, None)
decoder_d_model = (512, 512)
decoder_n_layers = (8, 8)
decoder_n_heads = (8, 8)
decoder_n_kv_heads = (8, 8) 

code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (256, 256)
mlp_mult = 4
rope_base = 10000.0



ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 1.0

# Kspan = decoder_ncodes × stride[level]  # = G × K
# Pp    = level_refine_window × Kspan # = level_refine_window × decoder_ncodes × stride[level]

strides = (4, 4)
decoder_ncodes = (32, 8)  # was 16 (both) -- smart-guess threshold for ncodes_window=-1's
# O(n_groups*n_blocks) cost: n_blocks/8 per level (256/8=32, 64/8=8), giving 8 parallel groups at
# both levels without the "small decoder_ncodes + unbounded window" blowup the codebase warns about
ncodes_window = -1  # was 0 -- unbounded causal context per group (the "full causal parallel decode" idea)
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True
level_refine_window = 0  # was 4 -- refine disabled (level_refine_passes=1): isolate ncodes_window=-1's
# own effect, not conflated with refine's separate contribution
level_refine_passes = 1  # was 2
# cond_depth = (2, 1)  # disabled -- isolate ncodes_window=-1's own effect, not conflated with
# multi-level conditioning's separate contribution (matches refine being disabled above)
# cond_window = (4, -1)  # no effect with cond_depth back at its default (1,1)
# cond_drop = 0.5  # no effect with cond_depth back at its default (1,1)

additive_drop_loss = False
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
precision = "bf16"

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e4), int(5e4))
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
