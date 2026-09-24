"""
uv run python -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_par1.py
"""
# Fork of imagenet64_ar.py: bigdec-styled arch (encoder halved: d_model 512->256, n_layers 4->2; decoder
# separate, bigger: d_model=1024, n_layers=8) + pardec (interleave_decode dropped -- cond_window is
# pardec-only and meaningless under interleave_decode, so this config intentionally uses pardec, unlike
# imagenet64_ar) + a multi-pass refine loop (level_refine_passes=3, window=16), matching run7.py's refine
# pattern. decoder_ncodes=4 (was 1), cond_window widened to 16, attn_window widened to 1024, use_xsa/use_sink
# re-enabled. batch_size=8 first -- reduce if OOM (bigger decoder + wide attn_window may not fit).

# --- model ---
multihost = True
dataset = "imagenet64"
data_root = "/dev/shm/imagenet64"
img_size = 64
d_model = (256, 256)  # was (512, 512) -- halved encoder dim
n_layers = (2, 2)  # was (4, 4) -- halved encoder depth
n_heads = (4, 4)
n_kv_heads = (None, None)
decoder_d_model = (512, 512)  # was 1024 -- OOM'd twice at 613G vs 30.75G, halved per explicit fallback
decoder_n_layers = (4, 4)
decoder_n_heads = (4, 4)  # was 16 -- halved with d_model to keep head_dim=64
decoder_n_kv_heads = (4, 4)  # plain MHA, avoids relying on encoder-ratio auto-GQA resolution
code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (256, 256)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 1.0

strides = (4, 4)
decoder_ncodes = 16
ncodes_window = 4
decode_future = 4
attn_lookahead = 0
attn_window = (1024, 1024)  # was (1024, 1024) -- OOM'd at batch_size=4 (25.81G needed vs 24.77G free, ~1G over);
# symmetric base (2026-09-23 encoder_attn_window/decoder_attn_window feature) bounds both encoder AND decoder/
# refine-pass attention now, so halving this (unlike the old encoder-only attn_window, proven ineffective on
# imagenet64_pardec1) should actually reduce memory this time. decoder_ncodes left at 4 (deliberate contrast
# vs imagenet64_pardec1's 32 -- not touched).
cond_depth = (2, 1)
cond_window = 4  # was (4, -1) -- widened; level1's cond_window has no effect anyway (cond_depth=1 there)
cond_drop = 0.5

level_refine_window = 1  # was 4 -- 5th OOM was identical (26.18G/24.88G) after cutting attn_window, proving
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_gt_drop = 0.8
level_refine_drop = 0.5  # stop before each extra pass w.p. 0.5 -> 1..level_refine_passes passes per step
level_refine_passes = 2  # was 3

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.8
quantize_drop = 0.8

init_scheme = "llama"
use_xsa = True  # was False
use_sink = True  # was False
precision = "bf16"
remat = False
remat_level = True

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 4  # was 8 -- OOM'd by only 40MB (30.79G vs 30.75G), tiny margin; halved to stay TPU-shard-clean (divisible by 4 local devices)
val_batch_size = 4
level_epochs = (0, 2)
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
