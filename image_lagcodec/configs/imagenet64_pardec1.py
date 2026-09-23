"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/imagenet64_pardec1.py
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
n_heads = (8, 8)
n_kv_heads = (None, None)
decoder_d_model = (512, 512)  # was 1024 -- OOM'd twice at 613G vs 30.75G, halved per explicit fallback
decoder_n_layers = (8, 8)
decoder_n_heads = (8, 8)  # was 16 -- halved with d_model to keep head_dim=64
decoder_n_kv_heads = (8, 8)  # plain MHA, avoids relying on encoder-ratio auto-GQA resolution
code_vocab = (256, 256)
pq_chunks = (3, 3)
pq_dim = (128, 128)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.1

strides = (4, 4)
decoder_ncodes = 32  # was 4 -- OOM'd twice (613G, 380G vs 30.75G); fewer/bigger pardec groups = far fewer
# batched rows (n_groups=1024/32=32, B2=8*32=256, was B2=2048) -- the actual driver of the blowup
# interleave_decode NOT set (pardec instead) -- cond_window is pardec-only, meaningless under interleave_decode
attn_lookahead = 0
attn_window = (1024, 1024)  # symmetric base (new 2026-09-23 encoder_attn_window/decoder_attn_window feature) --
# now bounds BOTH encoder and decoder self-attention by default, incl. the decoder's own refine-pass
# attention (previously unbounded -- likely the real culprit, not the encoder-only cut that had zero effect)
cond_depth = (2, 1)
cond_window = 16  # was (4, -1) -- widened; level1's cond_window has no effect anyway (cond_depth=1 there)
cond_drop = 0.5

level_refine_window = 1  # was 4 -- 5th OOM was identical (26.18G/24.88G) after cutting attn_window, proving
# it wasn't the bottleneck; Pp=level_refine_window*decoder_ncodes*stride=1*32*4=128 (was 512) -- much smaller
# per-row redraft length in the decoder's own (unbounded dec_attn_window) refine-pass attention
level_refine_gumbel = True
level_refine_temperature = 1.0
level_refine_gt_drop = 0.8
level_refine_drop = 0.5  # stop before each extra pass w.p. 0.5 -> 1..level_refine_passes passes per step
level_refine_passes = 2  # was 3
# refine_quantize_drop = 0.5  # disabled -- multipass_detach=False (needed for it) keeps more graph alive
# through the refine passes for backward, a real memory contributor at this scale; back to run7.py's own
# (commented-out) pattern, multipass_detach stays at its True default

additive_drop_loss = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
encode_temperature = 1.0
gumbel_at_inference = False
mse_softmax_tau = 1.0
level_gt_drop = 0.5
quantize_drop = 0.5

init_scheme = "llama"
use_xsa = True  # was False
use_sink = True  # was False
precision = "bf16"
remat = False
remat_level = True

byte_group = 3
token_head_type = "ar"
token_dim = (256, 256)
token_n_heads = 4
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = False

# --- training ---
batch_size = 2  # was 8 -- 3rd OOM (106G vs 30.75G, ~3.4x over) after decoder/refine/ncodes cuts; batch scales ~linearly
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
