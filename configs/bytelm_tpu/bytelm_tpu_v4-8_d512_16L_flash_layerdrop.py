"""qcute.bytelm_tpu config: sweep chip 2 of 3 on tpu5 -- layer_drop only (no resid_dropout),
otherwise identical to bytelm_tpu_v4-8_d512_16L_flash_singlechip.py (d512x16, ~67M params,
context=8192, flash-attention, no_zero_kv_sink, batch_size=8). Isolates layer_drop's effect on
val_bpb overfitting at fixed model size/steps, against the resid_dropout-only and both-together
sibling configs run on tpu5's other two idle chips (1/2/3).

24h step budget: same as the resid_dropout sibling config -- 1.22s/it measured on chip0
(identical arch/batch/context, flash-attention), 86400s / 1.22s/it ~ 70800 steps -> 70000.

layer_drop=0.1: stochastic depth, each block randomly zeroed (inverted-dropout-scaled) per
forward call during training -- see qcute/bytelm_tpu.py's ByteLM.forward. Implemented as an
on-device multiplicative gate rather than host-side layer skipping, so it doesn't change the
traced graph structure step to step and stays compile-safe on XLA.

    TPU_VISIBLE_CHIPS=2 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_flash_layerdrop.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_flash_layerdrop
"""
from pathlib import Path

preset = "d512x16"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
resid_dropout = 0.0
layer_drop = 0.1
steps = 70000
batch_size = 8
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 9000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
