"""
uv run python3 -m image_lagcodec.run_lagcodec_res --config image_lagcodec/configs/cifar_res_full1.py
"""
# Full training run on the genuinely modular rewrite (image_lagcodec/run_lagcodec_res.py):
# CodeLM, Downsampler, Upsampler are each a SINGLETON module -- exactly one of each for the whole
# model, shared across all 3 levels, not one per level. "Which level" is communicated into each
# shared module via rate_id (a bos_embed row indexed by level), not via separate weights.
# recursive_shared_depth/weight_sharing/EncDecLevel are GONE (obsolete: singleton sharing is now
# the only mode, and the old hand-rolled dec_blocks decode path -- the only consumer of
# weight_sharing -- is deleted, PardecLM-based decode is the only decode path). CodeLM/Downsampler/
# Upsampler each get their OWN fully independent Config fields with their own defaults (codelm_*,
# downsampler_*, upsampler_*/upsampler_ctx_*) -- no shared-base-with-override entanglement (the old
# d_model/encoder_*/decoder_* override-chain is gone). pq_chunks/code_vocab/pq_dim ARE deliberately
# single shared fields (not per-module) since all three modules must speak the same categorical
# vocab by construction. Every architecture field MUST be uniform across all 3 levels (enforced in
# Config.__post_init__) since there's one shared module, not per-level ones -- this config already
# keeps everything uniform, same values as cifar_recursive_full1.py (the predecessor config for the
# old run_lagcodec_recursive.py module).
# PardecLM (downsampler/upsampler) now support genuine call-time stride modulation
# (output_expansion override + rate_id -> bos_embed row) instead of a construction-time-fixed
# stride, though this config still uses one uniform stride=4 everywhere.
# byte_pq_fn (default rgb_byte_pq_fn): converts raw bytes into CodeLM's own (pq_chunks, code_vocab)
# representation -- identity for the standard RGB case (byte_group==pq_chunks, code_vocab==256),
# user-settable (e.g. byte_to_pq_idx_jax) for other factorizations (binary, hex, ...).

# --- model ---
img_size = 32

codelm_d_model = (256, 256, 256)
codelm_n_layers = (4, 4, 4)
codelm_n_heads = (2, 2, 2)
codelm_n_kv_heads = (None, None, None)

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

label_fn = "rgb_label_fn_jax"  # string, no import -- resolved via main()'s label_fn_registry.
# default_label_fn_jax degenerates to red-only for pq_chunks=3/code_vocab=256 (grayscale +
# bit-pack collapses 2 of 3 chunks to always-0)

strides = (4, 4, 4)
attn_window = (256, 256, 256)
decoder_ncodes = (16, 16, 16)
ncodes_window = (4, 4, 4)  # dead (only fed the removed old dec_blocks path), kept declared/inert
attn_lookahead = 0
decode_past = 0
decode_future = 4
remat_level = True

additive_drop_loss = False
use_pardec_downsampler = True
downsampler_d_model = (512, 512, 512)
downsampler_n_layers = (4, 4, 4)
downsampler_n_heads = (8, 8, 8)
downsampler_n_kv_heads = (8, 8, 8)
downsampler_window = (4, 4, 4)  # bounded, NOT -1 -- unbounded OOM'd even at tiny CPU test sizes
use_pardec_upsampler = True
upsampler_d_model = (512, 512, 512)
upsampler_n_layers = (4, 4, 4)
upsampler_n_heads = (8, 8, 8)
upsampler_n_kv_heads = (8, 8, 8)
upsampler_window = (4, 4, 4)  # bounded, NOT -1 -- same OOM lesson as downsampler_window
# use_codelm_bos left at default False for this run -- no scale/anchor-code training yet
curriculum_mode = "no_freeze"  # asserted unconditionally regardless of this config
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
