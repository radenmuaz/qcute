"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/d2_lazy_bigdec.py
"""
# Fork of d2_lazy_nl4.py: asymmetric encoder/decoder capacity (new 2026-09-22 decoder_*/encoder_* override
# feature). Encoder stays cheap (n_layers=2, d_model=256, the base symmetric fields); decoder gets its own
# separate, much bigger stack (d_model=512, n_layers=8, n_heads=8 plain MHA) via decoder_d_model/
# decoder_n_layers/decoder_n_heads/decoder_n_kv_heads. weight_sharing=False required (already was, for the
# nl4 lineage) -- decoder has always been a separate stack here, just same-dimensioned until now.
# remat_level=True since the decoder stack got much bigger (8 layers @ 512d).

# --- model ---
img_size = 32
d_model = (256, 256)
n_layers = (2, 2)  # cheap encoder (was 4 in d2_lazy_nl4)
n_heads = (2, 2)
n_kv_heads = (None, None)
decoder_d_model = (512, 512)
decoder_n_layers = (8, 8)
decoder_n_heads = (8, 8)
decoder_n_kv_heads = (8, 8)  # plain MHA (head_dim=64), avoids relying on encoder-ratio auto-GQA resolution
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
decoder_ncodes = (256, 64)  # = n_blocks per level -- single group, waits for/sees the whole level, no padding
ncodes_window = -1  # no effect either way: decoder_ncodes=n_blocks (single group) already sees everything
attn_lookahead = 0
decode_past = 0
decode_future = 4
level_refine_window = 1
level_refine_gumbel = True
level_refine_temperature = 1.0
# level_refine_passes = 2
cycle_refine_passes = 1
cond_depth = (1, 1)  # disabled -- single group already sees its whole own level
# cond_drop = 0.5  # no effect with cond_depth<=1

additive_drop_loss = False
weight_sharing = False
remat_level = True
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
precision = "fp32"

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 8
val_batch_size = 8
level_steps = (int(1e3), int(1e5))
seed = 0
train_subset_n = None
val_subset_n = None
gen_eval_every_step = 1000
epoch_verbose = False

grad_clip = 1.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e5)
warmup_steps = 1000
optimizer = "adamw"
optimizer_kwargs = dict(weight_decay=1e-3)

wa_mode = "none"

# --- logging ---
log_every = 100
ckpt_every_step = 1000
ckpt_keep = 1
