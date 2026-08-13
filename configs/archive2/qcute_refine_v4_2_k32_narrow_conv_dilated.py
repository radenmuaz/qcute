"""qcute.qcute_refine_v4_2 config: full-width (`bit_inner_downsample=1`)
`bit_head_class="conv_dilated"`, `conv_dilated_mode="depthwise"` (new
this session).

Session ask: "do this stacked small kernel, then check memory usage vs
single large conv" (params/FLOPs analysis), then "check ar gen conv
code, then train this" (generation support added, now training).
`BitPredictHeadConvDilated`: a WaveNet-style dilated depthwise-separable
causal conv STACK (kernel=`conv_dilated_base=2`, dilation doubling each
layer, `L=3` layers for `dq=8`, receptive field exactly 8) replacing
`BitPredictHeadConv`'s single big `K=dq` kernel. PURELY LINEAR across
layers (no activation — session: "i mean for memory and param save even
though linear"), a real expressivity cost (composition of linear filters
stays linear, representationally a subset of the single-kernel version)
traded for params/FLOPs: at this `dq=8` scale the params saving is a
near-wash vs. `conv_depthwise` (bias terms per extra layer roughly cancel
the tap savings), but FLOPs are modestly better (~22% fewer) and, after
fixing an `nn.Conv1d`-overhead bug found along the way (see docs/
kv_contribution.md §17), wallclock is actually the FASTEST chain-head
variant measured this session (0.53ms/fwd vs. `conv_depthwise`'s own
0.96ms). Verified: fixed/loop consistency (exact/near-exact match),
gradients, greedy-decode smoke test, full-model forward+backward,
`validate_generation` parity (`generate_no_cache` vs. `generate_kv_cache`
exact match) — all clean.

`code_embed_mode="pq_table"` carried over — same fix that resolved
`attn_id4`'s original divergence.

Everything else identical to `qcute_refine_v4_2_k32_narrow_conv_
depthwise.py`'s own family: Ks=(32,32), dq=8, d_model=256, n_layers=1,
context_len=1024, attn_window=(32,32), fuse_encoder_levels=True,
fuse_use_null_kv=False, code_head_mode="chain", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_conv_dilated.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_conv_dilated
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
bit_head_class = "conv_dilated"
conv_dilated_base = 2
conv_dilated_mode = "depthwise"
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
