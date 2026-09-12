"""Full CIFAR-10 (chat 2026-09-12) -- first run this session NOT using the 100-image overfit-
sanity subset (train_subset_n=None -> the real 50,000-image training set). Curriculum ENABLED
(phase-by-phase, not --no_curriculum): epochs_per_phase=(20,20,20,100) -- phases 1-3 (levels
0,1,2 alone) get 20 epochs each (~62,500 steps at batch_size=16 on the full set), phase 4 (all 4
levels jointly, no_freeze) gets 100 epochs (~312,500 steps). traversal="zorder", byte_group=3, token_head_type="ar"
everywhere, token_dim sized per-level. TRUE MTP enabled: mtp_horizon=4 (max, capped at each
level's stride=4), mtp_mode="ar" -- "two nested ar transformers" (chat 2026-09-12): an OUTER
causal chain over the 4 future timesteps, each step's output fed into the SAME inner token-ar
chain to predict that timestep's own R/G/B members. Previously OOM'd (33-50G required vs 30.75G
available) because every causal attention call -- both the outer chain and each of the 4 inner
calls -- routed through splash_attention, which pads any sequence to 128 regardless of its real
length (3-4 here); fixed by switching these calls to dense (non-Pallas, unpadded) causal
attention (see dense_self_attention in run_lagcodec_zorder.py) -- confirmed working at
batch_size=16 again (was reduced to 8 before the fix). Trained via the auxiliary mtp_weight-
scaled loss; inference stays plain one-step decode (no mtp at generation time, per user request
2026-09-12).

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_full_zorder_mtp_ar_ar.py
"""

run_name = "cifar10_full_zorder_mtp_ar_ar"

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
mtp_mode = "ar"
traversal = "zorder"

# --- training ---
batch_size = 16   # back to 16 (chat 2026-09-12): the OOM was from splash_attention's 128-padding
# on tiny causal sequences, not the nesting itself -- fixed via dense_self_attention.
epochs_per_phase = (20, 20, 20, 100)
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
