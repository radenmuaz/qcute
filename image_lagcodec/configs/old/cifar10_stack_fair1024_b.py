"""Fair-capacity ablation B (chat 2026-09-11): keeps candidate C's per-level eff_vocab split
(16, 4, 16 -- C beat A in a direct comparison, giving level2 more capacity than level1 despite
same raw stride, since level2 actually summarizes a 16x16=256-pixel patch vs level1's 16 pixels)
but uses SINGLE-CHUNK codes (pq_chunks=1, bigger vocab) instead of C's multi-chunk PQ at level0
(vocab=4,pq=2) -- tests PQ chunking granularity as an orthogonal variable while holding per-level
capacity fixed, AND avoids code_vocab=2 (degenerate binary quantization, explicitly excluded --
that was candidate A's choice). Same d_model/lag/train_last_encoder/optimizer as A/C.

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_fair1024_b.py
"""

run_name = "cifar10_stack_fair1024_b"

# --- model ---
img_size = 32
d_model = (32, 256, 256, 0)   # d_model[-1]=0: don't-care, topmost level never built
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (16, 4, 16, 16)   # level3 (topmost, index -1) don't-care: train_last_encoder=False
pq_chunks = (1, 1, 1, 1)       # eff_vocab: 16, 4, 16 -> cumulative 16*4*16=1024 (single-chunk codes)
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 3   # max -- see cifar10_stack_fair1024_a.py docstring
train_last_encoder = False

# --- training ---
batch_size = 16
n_devices = None
epochs = 200
lr = 5e-4
lr_schedule = "warmup_cosine"
warmup_steps = 1000
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}
# sinkgd (previous optimizer here, chat 2026-09-11) -- do not delete, comment/uncomment to swap:
# lr = 0.01
# weight_decay = 0
# optimizer = "sinkgd"
# optimizer_kwargs = {"sinkhorn_iters": 1, "weight_decay": 0}
seed = 0
train_subset_n = None

# --- logging / eval ---
log_every = 1000
eval_every_epochs = 10
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
