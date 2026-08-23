"""qcute.bytelm_tpu config: sweep chip 3 of 3 on tpu5 -- resid_dropout AND layer_drop together,
otherwise identical to bytelm_tpu_v4-8_d512_16L_flash_singlechip.py (d512x16, ~67M params,
context=8192, flash-attention, no_zero_kv_sink, batch_size=8). Tests whether the two
regularizers combine additively/synergistically vs. either alone (the resid_dropout-only and
layer_drop-only sibling configs on tpu5's other two idle chips).

24h step budget: same as both siblings -- 1.22s/it measured on chip0 (identical
arch/batch/context, flash-attention), 86400s / 1.22s/it ~ 70800 steps -> 70000.

    TPU_VISIBLE_CHIPS=3 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_flash_bothdrop.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_flash_bothdrop
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
resid_dropout = 0.1
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
