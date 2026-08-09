"""qcute.qcute_refine_v4_2 config: full-width (`bit_inner_downsample=1`)
`bit_head_class="conv"`, `bit_conv_impl="depthwise"` (new this session).

Session ask: "consider making bitpredictconv more efficient, last time
huge compute, maybe try group conv or depthwise." The existing "conv1d"/
"matmul" impls are FULLY DENSE across channels (every output channel
reads every input channel at every window position) — at full width
(`d_model=256`, `kernel_size=dq=8`) that's 525,313 params / 8,392,704
FLOPs for this ONE head (measured via FlopCounterMode), the "huge
compute" being referenced (this is why every prior `conv`-family config
this session used `bit_inner_downsample>1`, never full width). New
`conv_impl="depthwise"` gives each channel its own private K-tap filter
(no cross-channel mixing), matching `nn.Conv1d(...,groups=d_inner)`'s
classic depthwise-separable structure but implemented via plain einsum
(not `nn.Conv1d`, to keep the loop-overhead-avoidance property "matmul"
was built for) — 3,073 params / 36,864 FLOPs at the same full width: a
171x/228x reduction. Cheap enough that this config finally tests `conv`
at FULL width, something no prior conv config in this session's family
ever attempted. Verified via fixed/loop-consistency (`torch.allclose`,
exact match, max diff 0.0) + gradient checks + full-model smoke test +
`validate_generation` parity, same as every other head change this
session.

`code_embed_mode="pq_table"` carried over — same fix that resolved
`attn_id4`'s original divergence (see docs/kv_contribution.md §11).

Everything else identical to this session's other `k32_narrow` chain-head
configs: Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="chain", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_conv_depthwise.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_conv_depthwise
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
code_head_mode = "chain"
bit_head_class = "conv"
bit_conv_impl = "depthwise"
bit_inner_downsample = 1
code_embed_mode = "pq_table"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
