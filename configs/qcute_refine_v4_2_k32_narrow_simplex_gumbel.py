"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_simplex.py, ONE change: `use_gumbel_noise=True` instead of the
default `False`.

Session rationale: `qcute_refine_v4_2_k32_narrow_attn_id4` (the `bsq`+
`chain`-head family's own `bit_inner_downsample=4` config) diverged
outright (`val_bpb` rising from 5.34 to 5.66 between steps 800-900,
killed early — see docs/status.md/docs/kv_contribution.md §11). Session
ask: "logs/qcute_refine_v4_2_k32_narrow_attn_id4 diverging, retry but
enable gumbel." `bit_head_class="attn"`/`code_head_mode="chain"` (what
`attn_id4` actually uses) has no "gumbel" concept at all — the only
Gumbel-Softmax lever in this file is `quant_type="simplex"`'s
`use_gumbel_noise` flag, a DIFFERENT quantization axis entirely (BSQ's
per-dim sign+STE hypercube vs. simplex's whole-code softmax+STE
category) — `attn_id4`'s own divergence can't be literally "retried with
gumbel enabled" in place, since the two mechanisms don't compose (simplex
mode bypasses `code_head_mode`/`bit_head_class` entirely, see `LevelLM.
__init__`'s own dispatch order). Closest faithful test available: does
adding actual STOCHASTIC exploration to the quantization step (genuine
Gumbel-noise sampling every forward call, replacing `quant_type=
"simplex"`'s own cheap deterministic softmax+argmax+STE default) help
avoid the kind of divergence/instability seen elsewhere in this session's
shared-pool family, as an alternative lever to `code_embed_mode=
"pq_table"`'s already-confirmed fix for the `bsq` family.

Everything else identical to qcute_refine_v4_2_k32_narrow_simplex.py:
Ks=(32,32), d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_bits=8 (default), gumbel_tau=1.0 (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_simplex_gumbel.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_simplex_gumbel
"""
from pathlib import Path

Ks = (32, 32)
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
quant_type = "simplex"
use_gumbel_noise = True

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
