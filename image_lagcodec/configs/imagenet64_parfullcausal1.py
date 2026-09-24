"""
uv run python -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_parfullcausal1.py
"""
# Adapts cifar_parfullcausal1.py's "full causal parallel decode" idea (ncodes_window=-1: every
# pardec group still batches independently/parallel, but its context window Wg is the FULL causal
# prefix instead of a small fixed slice -- later groups get strictly more real context, which gave
# cifar noticeably better reconstruction despite refine being OFF) to imagenet64. Model/decode
# params copied AS-IS from cifar_parfullcausal1.py (not rescaled to imagenet64's larger n_blocks --
# if in doubt, keep it identical rather than guess a new derivation). Only the dataloader/training
# hyperparameters are adapted, following imagenet64_ar1.py's style (multihost, dataset paths,
# level_epochs instead of level_steps, gen_eval_every_epoch, val_subset_n).
#
# THROUGHPUT RISK (unverified until measured against imagenet64_ar1's it/s): pardec's own
# dense_self_attention_pardec never uses the splash kernel (interleave_decode does), and
# decoder_ncodes=(32,8) at imagenet64's n_blocks=(1024,256) gives n_groups=(32,32) -- notably more
# groups than cifar_parfullcausal1 had at its own scale (8,8) -- so this may need decoder_ncodes
# raised (fewer, bigger groups) if it's slower than imagenet64_ar1.

# --- model ---
multihost = True
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
img_size = 64

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
pq_dim = (64, 64)  # was 256 -- OOM relief; 64 dims is architecturally fine for 256-way byte
# prediction (linear 64->256 isn't capacity-limited by hidden_dim>=n_classes), watch val_dec_acc
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 1.0

# Kspan = decoder_ncodes × stride[level]  # = G × K
# attn_window = 1024  # was added as an OOM-relief lever at HBM ~92% usage/near-zero headroom;
# no longer needed now that ncodes_window is properly bounded -- HBM at ~60% usage with real
# headroom now, disabled to give the encoder's own self-attention its natural unbounded window back
strides = (4, 4)
decoder_ncodes = (64, 64)  # uniform -- level0 n_blocks=1024 -> n_groups=16; level1 n_blocks=256
# -> n_groups=4. Power of 4 (square patches under zorder traversal). Kspan=decoder_ncodes*stride=256
# per level (dense-processed-per-group cost on top of Wg, on top of the earlier (256,64)/(32,8) history)
ncodes_window = 4  # NOT 256 -- ncodes_window is counted in GROUPS, not raw blocks (Wg =
# ncodes_window*G internally); 256 was >= n_groups at both levels, i.e. still effectively
# unbounded (same as -1, per the config's own warning). 4 = "4 groups' worth of context"
# (Wg = 4*64=256 blocks either way, but expressed in the field's real unit). Was -1 (unbounded),
# which OOM'd repeatedly: unbounded Wg=n_blocks scales with n_groups regardless of decoder_ncodes,
# the actual driver. Bounding decouples memory from n_blocks while still giving each group more
# context than its own span, keeping some of the "more real context" property that made
# cifar_parfullcausal1 reconstruct well
attn_lookahead = 0
decode_past = 0
decode_future = 4  # pardec supports this (unlike interleave_decode)
remat_level = True
level_refine_window = 0  # refine off -- isolate ncodes_window=-1's own effect, matching cifar_parfullcausal1
level_refine_passes = 1

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

byte_group = 3
token_head_type = "ar"
token_dim = (64, 64)  # was 256 -- OOM relief, same reasoning as pq_dim above
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training (adapted to imagenet64_ar1.py's dataloader/epoch style) ---
batch_size = 4  # was 8 -- OOM'd (51.50G needed vs 30.75G available, ~1.68x over). pardec batches
# B2=batch_size*n_groups rows (n_groups=4 per level at decoder_ncodes=(256,64)), more prone to OOM
# than imagenet64_ar1's effectively-flat interleave sequence
val_batch_size = 4
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
