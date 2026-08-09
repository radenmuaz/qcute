"""qcute.qcute_refine_v4_2 config: `bit_head_class="ssm"`, `bit_inner_
downsample=4`, `bit_downsample_h=False` (default — h stays full d_model).

Session: "queue more experiments to test this hypothesis, repr loss
because of downsample h." Pairs directly with `qcute_refine_v4_2_
k32_narrow_ssm_id4_hds.py` (identical except `bit_downsample_h=True`) —
same downsample ratio, same everything else, isolating whether
downsampling `h` itself (not just the state machinery) causes a real
quality loss. `bit_per_position_head=True` (default, unchanged from
`BitPredictHeadSSM`'s existing revamp — this class's own per-position
design was never found to regress, unlike Attn's).

`code_embed_mode="pq_table"` carried over — same fix that resolved
`attn_id4`'s original divergence.

Everything else identical to this session's other `k32_narrow` chain-head
configs: Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="chain", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_ssm_id4_hfull.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_ssm_id4_hfull
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
bit_head_class = "ssm"
bit_inner_downsample = 4
bit_downsample_h = False
bit_per_position_head = True
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
