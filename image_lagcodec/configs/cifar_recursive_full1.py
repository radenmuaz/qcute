"""
uv run python3 -m image_lagcodec.run_lagcodec_recursive --config image_lagcodec/configs/cifar_recursive_full1.py
"""
# Full (not probe-scale) training run combining the two verified pieces of this session's
# CodeLM/downsampler/upsampler redesign: recursive_shared_depth (3 levels, 32->16->8->4 stride=4,
# level0 independent, levels 1&2 share ONE EncDecLevel -- verified: levels[1] is levels[2],
# gradient norm scales correctly with number of uses, checkpoint round-trips) AND
# use_pardec_downsampler (encode_pardec_downsampler: CodeLM forward -> downsampler PardecLM
# teacher-forced against rgb_label_fn_jax's real downsampled-image target, basic
# context_group_size=K/output_group_size=1 case -- one code per K-block, genuinely autoregressive
# over context, unlike the naive pick/CodePoolAttention it replaces). Both verified together via
# smoke test (3-level recursive + pardec downsampler, phase=3 forward+backward, finite loss/grad).
# Uses the STANDARD (non-multires) level_forward/train loop -- level_forward_multires exists and is
# smoke-tested standalone but not yet wired into main().
# Carries over cifar_probe_labelreg1's two confirmed findings: entropy_weight=0 (was fighting
# label_reg_weight, caused label_mse to climb steadily instead of converging) and rgb_label_fn_jax
# (default_label_fn_jax degenerates to a red-only target for pq_chunks=3/code_vocab=256).
# All per-level tuples uniform across levels[1:] (required by recursive_shared_depth's assertions,
# now also enforced for downsampler_* fields) -- kept uniform across ALL 3 levels for simplicity.
# TODO not yet done: >1 code per downsampler group (context_group_size>K case, e.g. 16-in/4-out),
# upsampler-side PardecLM wiring (still uses the old dec_blocks/ctx_embed path), multi-res training
# wiring into main(). downsampler_window MUST stay bounded (not -1) -- unbounded OOM'd even at tiny
# CPU test sizes given many small groups at output_group_size=1.

# --- model ---
img_size = 32

d_model = (256, 256, 256)
n_layers = (4, 4, 4)
n_heads = (2, 2, 2)
n_kv_heads = (None, None, None)
decoder_d_model = (1024, 1024, 1024)
decoder_n_layers = (4, 4, 4)
decoder_n_heads = (8, 8, 8)
decoder_n_kv_heads = (8, 8, 8)

code_vocab = (256, 256, 256)
pq_chunks = (3, 3, 3)
pq_dim = (64, 64, 64)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.0  # was 1.0 in earlier probes -- fights label_reg_weight, confirmed via
# cifar_probe_labelreg1 (label_mse climbed 54->772 over 15k steps with entropy_weight=1.0 on,
# converged and stayed low once set to 0)
label_reg_weight = 1.0

from image_lagcodec.run_lagcodec_recursive import rgb_label_fn_jax
label_fn = rgb_label_fn_jax  # default_label_fn_jax degenerates to red-only for pq_chunks=3/
# code_vocab=256 (grayscale + bit-pack collapses 2 of 3 chunks to always-0); confirmed via
# scripts/sanity_target_downsample.py

strides = (4, 4, 4)
attn_window = (256, 256, 256)
decoder_ncodes = (16, 16, 16)
ncodes_window = (4, 4, 4)
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True
level_refine_window = 0
level_refine_passes = 1

additive_drop_loss = False
weight_sharing = (False, False, False)
recursive_shared_depth = True  # levels 1,2 share one EncDecLevel
use_pardec_downsampler = True  # the other half of the redesign -- see module docstring
downsampler_d_model = (512, 512, 512)
downsampler_n_layers = (4, 4, 4)
downsampler_n_heads = (8, 8, 8)
downsampler_n_kv_heads = (8, 8, 8)
downsampler_window = (4, 4, 4)  # bounded, NOT -1 -- see module docstring
curriculum_mode = "no_freeze"  # required by recursive_shared_depth
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
code_head_attn_pool = False  # naive position-(K-1) pick, not the cross-attn pool -- keep the
# baseline path for this run; code_head_attn_pool is available but not chosen as default yet

byte_group = 3
token_head_type = "ar"
token_dim = (64, 64, 64)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True


# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e4), int(1e4), int(5e4))  # phase1=level0 only, phase2=+shared@1,
# phase3=+shared@2 (full cascade, longest phase) -- 70k steps total
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 2000
epoch_verbose = False

grad_clip = 1.0
lr = 5e-4
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(7e4)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-5)

# --- logging ---
log_every = 100
ckpt_every_step = 5000
ckpt_keep = 1
