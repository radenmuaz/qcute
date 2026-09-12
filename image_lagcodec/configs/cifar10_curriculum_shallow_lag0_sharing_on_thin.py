"""Curriculum-trained hierarchical codec, weight_sharing=True only (chat 2026-09-11 -- dropped
the sharing_off ablation for now, focusing on sharing_on). Ablates against
cifar10_curriculum_shallow_lag0_sharing_on_fat.py: "thin" = d_model scaled per level like
cifar10_stack_fair1024_a/c.py (32 for stride=3, 256 for stride=16), "fat" = same shape x2 per
level (64/512). code_vocab/pq_chunks per-level (8,8,4,4)/(5,5,5,5) -> eff_vocab 32768,32768,
1024,1024 (chat 2026-09-11): level0's code alone can't represent the full per-pixel 3x2^8 byte
space at 1024 (autoregressive decode fills SOME of that gap but 1024 was judged too thin), so
level0/level1 get bumped to 8**5=32768; level2/level3 (coarser, more autoregressive help
available from all lower levels) stay at 4**5=1024, matching fair1024_a/c's level2 value.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curriculum_shallow_lag0_sharing_on_thin.py
"""

run_name = "cifar10_curriculum_shallow_lag0_sharing_on_thin"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256)   # level0 bumped 32->256 (chat 2026-09-11, matches other levels)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (8, 8, 4, 4)
pq_chunks = (5, 5, 5, 5)   # eff_vocab: 32768, 32768, 1024, 1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = True

# --- training ---
batch_size = 16
epochs_per_phase = 3000
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
# sinkgd (previous optimizer here, chat 2026-09-11) -- do not delete, comment/uncomment to swap:
# lr = 1e-2
# weight_decay = 0
# optimizer = "sinkgd"
# optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
