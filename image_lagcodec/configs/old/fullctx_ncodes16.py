"""
KIV -- not launched yet (chat 2026-09-15). Written to illustrate the "grouped fullctx" point in
the spectrum, spot #5 of the axes table.
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/fullctx_ncodes16.py

GROUPED FULLCTX preset: like fullctx_ncodes1.py (streaming=False + ncodes_window=-1 -- every group
sees the SAME full real ctx directly, no padding/masking), but decoder_ncodes=16 instead of 1 --
coarser grouping, so fewer/larger parallel groups (n_groups=16 at level0 instead of 256). Cheaper
per-group attention (smaller B2=batch*n_groups) at the cost of more AR steps per group (16 instead
of 1). Sits between fullctx_ncodes1.py (finest, max parallel) and fullctx_ncodesfull.py (coarsest,
fully sequential) on the grouping axis, while staying non-causal fullctx on the streaming axis.
"""


# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, )
n_layers = (4, 4, 4, 4,)
n_heads = (2, 2, 2, 2,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (256, 256, 256, 256,)
pq_chunks = (3, 3, 3, 3,)
mlp_mult = 2
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.1
mse_weight = 1.0
entropy_weight = 0.01

decoder_ncodes = 16
ncodes_window = -1   # "all" -- combined with streaming=False, grouped fullctx: every 16-chunk
# sees the SAME full real ctx directly, no padding
streaming = False
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_drop = 0.5
quantize_drop = 0.5
init_scheme = "llama"
use_xsa = True
pq_dim = (128, 128, 128, 128)

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128, 128, 128)
token_n_heads = 2
mtp_horizon = 1
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 16
val_batch_size = 16
phase_epochs = (100, 100, 100, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 80
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "none"

# --- logging ---
log_every = 100
gen_eval_every_epoch = 100
ckpt_every_epoch = 100
ckpt_keep = 1
qual_gen_n = 16
