"""Fair-capacity ablation A (chat 2026-09-11): per-level code_vocab/pq_chunks instead of the
same 1024-effective-vocab at every level (which over-provisions each of the 3 used levels
identically -- combined capacity ~1024^3). Here eff_vocab = code_vocab**pq_chunks per level,
CUMULATIVE product across the 3 used levels (strides=(3,16,16), train_last_encoder=False so the
topmost level[3] is skipped) = 16*8*8 = 1024, matching a single VQ/FSQ code's ~10-bit budget for
the same 16x16 patch. d_model scaled with each level's stride/responsibility: 32 for stride=3
(near-raw RGB), 256 for stride=16 (coarser, more to summarize -- bumped from 128, was under
capacity). d_model[-1]=0: topmost level is never built (train_last_encoder=False), 0 is the
don't-care marker (mirrors strides' own -1-for-unused convention). lag=3 (max: lag_bytes=
(lag+1)*prod(strides[:-1])=(3+1)*768=3072=SEQ_LEN -- whole image's codes visible before the byte
decoder starts, per StackDecoder's lag convention).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/cifar10_stack_fair1024_a.py
"""

run_name = "cifar10_stack_fair1024_a"

# --- model ---
img_size = 32
d_model = (32, 256, 256, 0)   # d_model[-1]=0: don't-care, topmost level never built
n_layers = (2, 2, 2, 2)
n_heads = (4, 4, 4, 4)
n_kv_heads = (None, None, None, None)
strides = (3, 16, 16, -1)
code_vocab = (2, 2, 2, 2)      # level3 (topmost, index -1) don't-care: train_last_encoder=False
pq_chunks = (4, 3, 3, 3)       # eff_vocab: 16, 8, 8 -> cumulative 16*8*8=1024
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "stack"
lag = 3   # max -- see docstring
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
