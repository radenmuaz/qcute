"""
uv run python3 -m image_lagcodec.run_lagcodec_recursive --config image_lagcodec/configs/cifar_recursive_full1.py
"""
# Full (not probe-scale) training run combining this session's CodeLM/downsampler/upsampler
# redesign, now structurally separated: CodeLM is its own top-level eqx.Module (own weights, own
# encode()), EncDecLevel composes it via backward-compat properties; downsampler and upsampler are
# both independent PardecLM instances (own dedicated weights each) sharing the exact same
# pardec_score/pardec_generate machinery -- downsampler contracts (context_group_size=K,
# output_group_size=1, output_expansion=1), upsampler expands (context_group_size=
# output_group_size=decoder_ncodes, output_expansion=K). Also: recursive_shared_depth (3 levels,
# 32->16->8->4 stride=4, level0 independent, levels 1&2 share ONE EncDecLevel -- verified:
# levels[1] is levels[2], gradient norm scales correctly with number of uses, checkpoint
# round-trips). All three (CodeLM separation, pardec upsampler, pardec downsampler) verified
# together via CPU smoke test: 3-level recursive, phase=3 forward+backward (finite loss/grad) AND
# a full generate_from_prompt cascade (finite output), both on the old dec_blocks path (regression
# check) and this config's use_pardec_upsampler=True path.
# Uses the STANDARD (non-multires) level_forward/train loop -- level_forward_multires exists and is
# smoke-tested standalone but not yet wired into main().
# Carries over cifar_probe_labelreg1's two confirmed findings: entropy_weight=0 (was fighting
# label_reg_weight, caused label_mse to climb steadily instead of converging) and rgb_label_fn_jax
# (default_label_fn_jax degenerates to a red-only target for pq_chunks=3/code_vocab=256).
# All per-level tuples uniform across levels[1:] (required by recursive_shared_depth's assertions,
# now also enforced for downsampler_*/upsampler_* fields) -- kept uniform across ALL 3 levels for
# simplicity.
# TODO not yet done: >1 code per downsampler group (context_group_size>K case, e.g. 16-in/4-out),
# multi-res training wiring into main(). downsampler_window/upsampler_window MUST stay bounded
# (not -1) -- unbounded OOM'd even at tiny CPU test sizes given many small groups. upsampler's
# decode_past is scoring-only (pardec_generate doesn't implement it at generation, matching its own
# documented limitation) -- moot here since decode_past=0 in this config.

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

additive_drop_loss = False
weight_sharing = (False, False, False)
recursive_shared_depth = True  # levels 1,2 share one EncDecLevel
use_pardec_downsampler = True  # the other half of the redesign -- see module docstring
downsampler_d_model = (512, 512, 512)
downsampler_n_layers = (4, 4, 4)
downsampler_n_heads = (8, 8, 8)
downsampler_n_kv_heads = (8, 8, 8)
downsampler_window = (4, 4, 4)  # bounded, NOT -1 -- see module docstring
use_pardec_upsampler = True  # PardecLM-based decode, shares pardec_score/pardec_generate with the
# downsampler -- replaces the old hand-rolled dec_blocks/ctx_embed pardec decode path
upsampler_d_model = (512, 512, 512)
upsampler_n_layers = (4, 4, 4)
upsampler_n_heads = (8, 8, 8)
upsampler_n_kv_heads = (8, 8, 8)
upsampler_window = (4, 4, 4)  # bounded, NOT -1 -- same OOM lesson as downsampler_window
# use_codelm_bos left at default False for this run -- no scale/anchor-code training yet
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

byte_group = 3
token_head_type = "ar"
token_dim = (64, 64, 64)
token_n_heads = 2
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
