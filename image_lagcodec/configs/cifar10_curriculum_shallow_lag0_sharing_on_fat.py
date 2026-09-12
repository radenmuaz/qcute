"""Curriculum-trained hierarchical codec, weight_sharing=True only (chat 2026-09-11). Ablates
against cifar10_curriculum_shallow_lag0_sharing_on_thin.py: "fat" = every level's d_model x2 vs
thin (64 for stride=3, 512 for stride=16). Everything else identical to thin (code_vocab/
pq_chunks per-level (8,8,4,4)/(5,5,5,5) -> eff_vocab 32768,32768,1024,1024 -- see thin's docstring
for why level0/level1 are bumped above 1024; same strides/lag/training/curriculum schedule) --
isolates capacity (d_model only) as the one variable between the two runs.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curriculum_shallow_lag0_sharing_on_fat.py
"""

run_name = "cifar10_curriculum_shallow_lag0_sharing_on_fat"

# --- model ---
img_size = 32
d_model = (256, 512, 512, 512)   # level0 bumped 64->256 (chat 2026-09-11)
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
