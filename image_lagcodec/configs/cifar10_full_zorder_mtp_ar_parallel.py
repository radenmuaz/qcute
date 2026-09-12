"""KIV (chat 2026-09-12) -- not launched for now, per user request; kept, not deleted. Superseded
for this launch by cifar10_full_zorder_mtp_ar_ar.py (mtp_mode="ar" instead of "parallel").

Full CIFAR-10 (chat 2026-09-12) -- first run this session NOT using the 100-image overfit-
sanity subset (train_subset_n=None -> the real 50,000-image training set). Curriculum ENABLED
(phase-by-phase, not --no_curriculum): epochs_per_phase=(1,1,1,200) -- phases 1-3 (levels 0,1,2
alone) get just 1 epoch each (~3125 steps at batch_size=16 on the full set), phase 4 (all 4
levels jointly, no_freeze) gets 200 epochs (~625,000 steps) -- a dramatically larger budget than
anything else run this session. traversal="zorder", byte_group=3, token_head_type="ar"
everywhere, token_dim sized per-level. TRUE MTP enabled: mtp_horizon=4 (max, capped at each
level's stride=4), mtp_mode="parallel" -- "duplicate ar heads" (chat 2026-09-12): 4 FULLY
INDEPENDENT copies of the token-ar mechanism, each applied to the same hidden state, no chaining
across the 4 future timesteps. Trained via the auxiliary mtp_weight-scaled loss; inference stays
plain one-step decode (no mtp at generation time, per user request 2026-09-12).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_full_zorder_mtp_ar_parallel.py
"""

run_name = "cifar10_full_zorder_mtp_ar_parallel"

# --- model ---
img_size = 32
d_model = (512, 512, 512, 512, 512)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)
mlp_mult = 2
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5

byte_group = 3
token_head_type = "ar"
token_dim = (128, 32, 32, 32, 32)
token_n_heads = 4
mtp_horizon = 4
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 8   # reduced from 16 (chat 2026-09-12): OOM'd at 16 (33.64G required vs 30.75G
# available) -- the 4 duplicated ar heads add meaningfully more HBM per example than a single
# token head.
epochs_per_phase = (1, 1, 1, 200)
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
grad_clip = 1.0
seed = 0
train_subset_n = None

# --- logging ---
log_every = 10
qual_gen_n = 8
