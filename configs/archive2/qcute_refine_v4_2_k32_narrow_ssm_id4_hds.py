"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_ssm_id4_hfull.py, ONE change: `bit_downsample_h=True` instead
of False — h is ALSO projected down to d_inner via in_proj (the
pre-this-session behavior), instead of staying at full d_model.

Session: "queue more experiments to test this hypothesis, repr loss
because of downsample h." Direct A/B pair with `ssm_id4_hfull.py` — same
downsample ratio (4), same everything else, isolating whether
downsampling h itself (not just the state machinery) causes a real
quality loss.

Everything else identical to ssm_id4_hfull.py: Ks=(32,32), dq=8,
d_model=256, n_layers=1, context_len=1024, attn_window=(32,32),
fuse_encoder_levels=True, fuse_use_null_kv=False, code_head_mode="chain",
bit_per_position_head=True, code_embed_mode="pq_table", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_ssm_id4_hds.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_ssm_id4_hds
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
bit_downsample_h = True
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
