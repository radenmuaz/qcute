"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/imagenet64_res_4.py
"""
# Fork of imagenet64_res_3.py -- uses the "ar" + upsampler_rollout alternative instead of
# pardec_token_head="linear"'s sidestep: keeps the digit-AR head but trains it self-fed
# (token_ar_rollout) instead of always teacher-forced, matching how pardec_generate actually uses it
# at inference (see upsampler_rollout comment below for the full rationale). Needs
# upsampler_ncodes==1 (was 4 in imagenet64_res_3.py -- more sequential decode groups, real
# throughput tradeoff), and pardec_token_head="ar" re-enables downsampler_rollout too.
# --- data ---
img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
multihost = True
# multihost = False

# --- model ---
# remat_level = True
share_across_levels = False
# share_downsampler_upsampler_lm = True
remat = True
codelm_d_model = 512
codelm_n_layers = 2
codelm_n_heads = 2
codelm_n_kv_heads = 2

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 4
rope_base = 10000.0

additive_drop_loss = False
ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 0.0
label_reg_weight = 1.0
# pardec_token_head="ar" below makes downsampler_rollout compatible again (needs pardec_token_head="ar")
downsampler_rollout = True
downsampler_rollout_prob = 0.5
# downsampler_rollout = False
# downsampler_rollout_prob = 1.0

label_fn = "rgb_label_fn_jax"
# bos_rate_mode = "relative"
bos_rate_mode = "absolute"
strides = (4, 4, 4, 4)
# attn_window = 1024
attn_window = 4096
# upsampler_rollout needs upsampler_ncodes==1 at every level (asserted in Config.__post_init__) --
# imagenet64_res_3.py used upsampler_ncodes=4 (batched/parallel groups); this is the real throughput
# tradeoff for training the digit-AR head self-fed instead of sidestepping it with linear heads.
upsampler_ncodes = 1
# upsampler_ncodes = 4
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0   # was 4: aux-loss off-by-one made the last token of each group unpredictable (fixed in code, kept off)


downsampler_d_model = 128
downsampler_n_layers = 1
downsampler_n_heads = 1
downsampler_n_kv_heads = 1
downsampler_window = 1
# downsampler_remat = True

# upsampler_d_model = downsampler_d_model
# upsampler_n_layers = downsampler_n_layers
# upsampler_n_heads = downsampler_n_heads
# upsampler_n_kv_heads = downsampler_n_kv_heads
# upsampler_window = downsampler_window
# upsampler_remat = downsampler_remat

# downsampler_d_model = 1024
# downsampler_n_layers = 4
# downsampler_n_heads = 8
# downsampler_n_kv_heads = 8
# downsampler_window = 2
# downsampler_remat = True
upsampler_d_model = 1024
upsampler_n_layers = 2
upsampler_n_heads = 8
upsampler_n_kv_heads = 8
upsampler_window = 4
# upsampler_remat = True

use_codelm_bos = True
codelm_bos_prob = 1.0
curriculum_mode = "no_freeze"
quantize_mode = "zgr"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 1.0
quantize_drop = 0.0

# added 2026-10-02 (session audit): level_forward's decode-cascade ctx leaks gradient across levels
# by default (a level's loss trains the level ABOVE's upsampler via pseudo_ctx, on top of its own
# loss) -- unlike level 0's target (raw bytes), which was never differentiable. True cuts that path.
ctx_stop_gradient = True
# mixes in a cheap "parallel scheduled sampling" ctx (one teacher-forced pass's detached argmax)
# instead of level_gt_drop's real/pseudo mixture, 30% of decode steps. Requires ctx_stop_gradient=True.
decoder_scheduled_sampling_prob = 0.3
# upsampler_rollout: trains the upsampler's own digit-AR head self-fed (token_ar_rollout) instead of
# always teacher-forced -- a checkpoint_level_eval.py probe on imagenet64_res_3's own checkpoint
# found teacher-forced reconstruction stays near-flat in MSE across cascade depth while full
# self-generation degrades up to 30x worse, pointing at digit-AR self-feeding (never exercised
# during training) as the dominant compounding-error source. Alternative to imagenet64_res_3.py's
# pardec_token_head="linear" sidestep -- this keeps the AR head and trains around the mismatch
# instead of removing it.
upsampler_rollout = True
upsampler_rollout_prob = 1.0
# upsampler_rollout = False
# upsampler_rollout_prob = 1.0

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
# token_head_type = "linears"   # alternative: see imagenet64_res_3.py's sidestep approach
codelm_token_head = "ar"   # CodeLM NTP/free-run head: autoregressive digits
# codelm_token_head = "linear"    # alternative: all digits in one parallel matmul
pardec_token_head = "ar"    # downsampler/upsampler digit head: autoregressive digits
# pardec_token_head = "linear"    # alternative: all digits in one parallel matmul
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 4       # per DEVICE; tpu34 = 2 hosts x 4 devices -> global batch 64 (16 OOMs: 20.4G program vs 16G free)
val_batch_size = 2
# level_epochs = (1, 1, 5)
# Approximate step equivalents (tpu34 v4-16: batch_size 8 x 4 local devices x 2 hosts = global batch 64;
# ImageNet64 train = 1,281,167 imgs -> 1 epoch = 1,281,167 / 64 = 20,018 steps):
#   (1, 1, 5) epochs  ~=  (20018, 20018, 100090) steps  (140,126 total)
# To schedule by steps instead, comment out level_epochs above and uncomment (level_epochs and
# level_steps are mutually exclusive; steps-per-epoch scales with batch_size / n devices / n hosts):
# level_steps = (100_000, 100_000, 100_000, 1_000_000)
level_steps = (0,0,0,1_000_000)
seed = 0
train_subset_n = None
val_subset_n = 512
# gen_eval_every_epoch = 0.25   # ~= every 5,005 steps at global batch 64
gen_eval_every_step = 20_000   # uncomment (and comment gen_eval_every_epoch) to eval by steps
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
# lr_min_step = int(100e3)
# lr_min_step = int(200e3)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
