"""New decoder_type="self_attn_local_track1" -- single decoder for level0's bytes, NO per-level
cascade of stages (unlike self_attn/cross_attn/self_attn_local). Requires n_levels>=3:
strides=(2,2,1) -- level2 (topmost) needs its own stride to define its own granularity, but
stride=1 means no further pooling of level1's codes (nothing above level2 to pass a coarser
code to).

Track0 (own code = level1's code): block-diagonal self-attn, same StackDecoderLocal-inspired
mechanism as self_attn_local (proven mechanically correct but capacity-starved alone: shallow2
ablation plateaued at ~7% teacher-forced acc). Track1 (level2's code) adds global causal
cross-attention on top -- more conditioning information without falling back to the
self_attn/cross_attn architectures' raw-cross-block-byte-copying shortcut.

n_units: L0=1536 (level1's code count), L1=768 (level2's code count, stride[1]=2),
L2=768 (level2's own output, unused -- one level above the top, hard-excluded per reference).

uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/lagcodec_overfit1000_selfattn_local_track1.py
"""

run_name = "cifar_lagcodec_overfit1000_selfattn_local_track1"

# --- model ---
img_size = 32
d_model = (256,) * 3
n_layers = (2,) * 3
n_heads = (4,) * 3
n_kv_heads = (None,) * 3
strides = (2, 2, 1)
code_vocab = 8
pq_chunks = 4
mlp_mult = 4
rope_base = 10000.0
ntp_weight = 1.0
decoder_type = "self_attn_local_track1"

# --- training ---
batch_size = 16
n_devices = None
epochs = 3000
lr = 1e-2
warmup_steps = 100
weight_decay = 1e-5
optimizer = "sinkgd"
optimizer_kwargs = {}
seed = 0
train_subset_n = 100

# --- logging / eval ---
log_every = 200
eval_every_epochs = 1000
qual_gen_n = 8
qual_gen_greedy = True
qual_gen_temperature = 1.0
