"""
uv run python -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_ar1.py
"""
# Fork of imagenet64_par1.py, switched from pardec+refine to a true flat interleave_decode AR chain
# (decoder_ncodes=1, no refine passes), same shape as cifar_ar2_bigger.py's own interleave setup.
# Keeps imagenet64_par1's proven bigdec architecture (cheap encoder, separate bigger decoder,
# halved to fit HBM). cond_depth=(2,1): level0 also conditions on level1's coarser code (matches
# imagenet64_par1). mtp head ar/horizon=4, mtp_weight=0.1 (active aux loss this time, not dormant).
# decode_future=0 -- interleave_decode doesn't support it yet (WIP, see run_lagcodec.py:938 TODO).
# level_epochs=(1,10): 1 epoch level0, 10 epochs level1.

# --- model ---
multihost = True
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
img_size = 64
d_model = (256, 256)
n_layers = (2, 2)
n_heads = (4, 4)
n_kv_heads = (None, None)
decoder_d_model = (512, 512)
decoder_n_layers = (4, 4)  # halved from 8 -- proven fit for imagenet64_par1 at batch_size=8
decoder_n_heads = (4, 4)
decoder_n_kv_heads = (4, 4)  # plain MHA, avoids relying on encoder-ratio auto-GQA resolution
code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (256, 256)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0  # was 0.1 -- mtp_mode="ar" (nested inner transformer per position) too slow, disabled
mse_weight = 0.0
entropy_weight = 1.0

# Kspan = decoder_ncodes × stride[level]  # = G × K
interleave_decode = True
attn_window = 1024
strides = (4, 4)
decoder_ncodes = (4, 1)  # level0=4 (was 1) -- must be a multiple of level1's cumulative stride (4)
# for cond_depth=2's coarser reveal to align with level0's own group boundary (no lag); level1
# stays 1 (top level, no coarser conditioning, cond_depth=1 there)
ncodes_window = 0
attn_lookahead = 0
decode_past = 0
decode_future = 0  # interleave_decode doesn't support this yet (WIP, run_lagcodec.py:938 TODO)
level_refine_window = 1
cond_depth = (2, 1)

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
remat = False
remat_level = True

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 2
# mtp_mode = "ar"
# mtp_horizon = 4
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 16  # was 8 -- HBM at only ~9/30.75G (~29%) at bsz=8, doubled for headroom
val_batch_size = 16
level_epochs = (1, 10)
seed = 0
train_subset_n = None
val_subset_n = 512
gen_eval_every_epoch = 0.25
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-3)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 2000
ckpt_keep = 1
