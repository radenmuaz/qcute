"""Fork of cifar10_curr_off_s4_sampler_p05.py (chat 2026-09-12) -- best sampler result of the
session (final CASCADE_acc=0.379) -- run via run_lagcodec_zorder (NOT run_lagcodec_sampler),
adding: byte_group=3 (level0 now predicts one PIXEL (R,G,B) per position, jointly, instead of
one byte at a time) and traversal="zorder" (pixels visited in Morton-curve order instead of
row-major, RGB still contiguous per pixel). group_head_type="ar" everywhere -- a tiny
DeepSeek-MTP-style causal chain jointly predicts a group's members (R->G->B), each member
conditioned on the real values of earlier members via teacher forcing, one self-attention +
residual + untied linear head (no mlp, no weight tying -- chat 2026-09-12). mtp_dim is now
PER-LEVEL (128 for level0's large byte alphabet, 32 for levels1-3's small PQ-code alphabet) --
audited/sized (chat 2026-09-12) to land at ~0.83x/~0.81x of "linears"' dec_head param count per
level respectively, rather than a single global mtp_dim that was 0.5x level0 and 5x levels1-3.
Same architecture/optimizer/cascade_rollout_prob/epochs_per_phase as p05 otherwise.

uv run python3 -m image_lagcodec.run_lagcodec_zorder --config image_lagcodec/configs/cifar10_zorder_s4_sampler_p05_ar.py
"""

run_name = "cifar10_zorder_s4_sampler_p05_ar"

# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, 256)
n_layers = (2, 2, 2, 2, 2)
n_heads = (4, 4, 4, 4, 4)
n_kv_heads = (None, None, None, None, None)
strides = (4, 4, 4, 4, -1)
code_vocab = (16, 16, 16, 16, 16)
pq_chunks = (4, 4, 4, 4, 4)   # eff_vocab: 65536 per level
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
lag = 0
weight_sharing = False
curriculum_mode = "no_freeze"
quantize_mode = "argmax"
cascade_rollout_prob = 0.5

byte_group = 3                         # level0: one pixel (R,G,B) per position
group_head_type = ("ar", "ar", "ar", "ar", "ar")
mtp_dim = (128, 32, 32, 32, 32)        # sized to match "linears" dec_head param count per level
mtp_n_heads = (4, 4, 4, 4, 4)
traversal = "zorder"

# --- training ---
batch_size = 16
epochs_per_phase = 1000
lr = 5e-4
warmup_steps = 100
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging ---
log_every = 10
qual_gen_n = 8
