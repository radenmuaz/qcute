"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/imagenet64_res_1.py
"""
# ImageNet64 fork of cifar_res_1_lreg.py: same model/loss architecture (label_reg_weight=1.0,
# entropy 0, absolute bos, d=1024 downsampler/upsampler, gumbel settings, ...) but with
# codelm_token_head="ar" and pardec_token_head="ar". Data, dataloader, eval cadence and lr setup
# come from imagenet64_res1.py. strides=(4,4,4) covers 64px in 3 levels (top level = 64 codes, same
# top-level size as cifar's (4,4) on 32px), upsampler_ncodes=(16,16,16) like imagenet64_res1.

# --- data ---
img_size = 64
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
multihost = True

# --- model ---
codelm_d_model = 512
codelm_n_layers = 8
codelm_n_heads = 4
codelm_n_kv_heads = 4

code_vocab = 256
pq_chunks = 3
pq_dim = 64
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mse_weight = 0.0
entropy_weight = 1.0
label_reg_weight = 1.0

label_fn = "rgb_label_fn_jax"
bos_rate_mode = "relative"
# bos_rate_mode = "absolute"
strides = (4, 4, 4)
attn_window = 1024
upsampler_ncodes = (16, 16, 16)
attn_lookahead = 0
upsampler_decode_past = 0
upsampler_decode_future = 0   # was 4: aux-loss off-by-one made the last token of each group unpredictable (fixed in code, kept off)
remat_level = True

additive_drop_loss = False
downsampler_d_model = 512
downsampler_n_layers = 4
downsampler_n_heads = 4
downsampler_n_kv_heads = 4
downsampler_window = 4
downsampler_remat = True
upsampler_d_model = 1024
upsampler_n_layers = 8
upsampler_n_heads = 8
upsampler_n_kv_heads = 8
upsampler_window = 4
upsampler_remat = True

use_codelm_bos = False   # ON for this ablation (was False in cifar_res_full1.py)
# use_codelm_bos = True   # ON for this ablation (was False in cifar_res_full1.py)
# codelm_bos_prob = 1.0   # always substitute -- no probabilistic drop back to real content
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 0.1
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.5
quantize_drop = 0.5

init_scheme = "llama"
use_xsa = True
use_sink = True
precision = "bf16"

byte_group = 3
token_head_type = "ar"
codelm_token_head = "ar"   # CodeLM NTP/free-run head: autoregressive digits
pardec_token_head = "ar"    # downsampler/upsampler digit head: autoregressive digits
token_dim = 64
token_n_heads = 2
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8       # per DEVICE; tpu34 = 2 hosts x 4 devices -> global batch 64 (16 OOMs: 20.4G program vs 16G free)
val_batch_size = 8
# level_epochs = (1, 1, 5)
# Approximate step equivalents (tpu34 v4-16: batch_size 8 x 4 local devices x 2 hosts = global batch 64;
# ImageNet64 train = 1,281,167 imgs -> 1 epoch = 1,281,167 / 64 = 20,018 steps):
#   (1, 1, 5) epochs  ~=  (20018, 20018, 100090) steps  (140,126 total)
# To schedule by steps instead, comment out level_epochs above and uncomment (level_epochs and
# level_steps are mutually exclusive; steps-per-epoch scales with batch_size / n devices / n hosts):
level_steps = (20000, 20000, 100000)
seed = 0
train_subset_n = None
val_subset_n = 512
# gen_eval_every_epoch = 0.25   # ~= every 5,005 steps at global batch 64
gen_eval_every_step = 5000   # uncomment (and comment gen_eval_every_epoch) to eval by steps
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
