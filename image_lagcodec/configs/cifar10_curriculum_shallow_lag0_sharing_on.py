"""Curriculum-trained hierarchical codec (chat 2026-09-10), weight_sharing=True: encoderlevel_i
= decoderlevel_i (same transformer weights, two forward modes) for every level with a decoder;
the topmost level (own code never consumed) stays encoder-only. Rolling curriculum: level i
trains for 2 phases (lead-in encode-only, then completing encode+decode, then frozen forever),
except level0 (1 phase, no lead-in) and the topmost level (1 phase, lead-in-only). Same
architecture as cifar10_stagelag_hier_overfit100_lag0.py (strides=(3,16,16,-1)) for direct
comparability, EXCEPT d_model now scaled per level like cifar10_stack_fair1024_a/c.py (32 for
stride=3, 256 for stride=16 -- was under capacity at uniform 256/128) and code_vocab/pq_chunks
now per-level tuples (chat 2026-09-11 port from run_lagcodec.py) -- kept UNIFORM (4,5 -> eff 1024)
at every level here, unlike fair1024_a/c's split budget: curriculum freezes each level once
fully trained, so each level's own code should independently carry a full 1024-way code, not a
fraction of a shared cumulative budget. Pair with cifar10_curriculum_shallow_lag0_sharing_off.py
for the ablation.

uv run python3 -m image_lagcodec.run_lagcodec_curriculum --config image_lagcodec/configs/cifar10_curriculum_shallow_lag0_sharing_on.py
"""

run_name = "cifar10_curriculum_shallow_lag0_sharing_on"

# --- model ---
img_size = 32
d_model = (32, 256, 256, 256)
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (4, 4, 4, 4)
pq_chunks = (5, 5, 5, 5)   # effective vocab = code_vocab**pq_chunks = 4**5 = 1024 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = True

# --- training ---
batch_size = 16
epochs_per_phase = 3000
lr = 1e-2
warmup_steps = 100
weight_decay = 0
optimizer = "sinkgd"
optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
