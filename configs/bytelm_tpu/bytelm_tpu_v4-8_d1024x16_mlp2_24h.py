"""qcute.bytelm_tpu config: model-size sweep, chip 1 of 4 on tpu4 -- d1024x16_mlp2 preset
(~168.3M params, d_model=1024, n_layers=16, mlp_mult=2 -- cheaper SwiGLU MLP than the mlp_mult=4
sibling at the same depth/width), context=8192, flash-attention, no_zero_kv_sink, batch_size=4.
Compares model size/shape against the other three sibling configs on tpu4's chips 0/2/3
(d768x12_mlp4 113.7M, d1024x16_mlp4 269M, d1024x24_mlp4 403M).

batch_size=4: bench-probed on this node (2026-08-23) -- bs=4 fine, bs=8 OOM'd. 24h step budget,
corrected: the initial estimate (~13s/it from a noisy 5-step probe window, giving steps=6600)
was badly wrong -- the real observed steady-state rate once actually running was 1.25s/it, ~10x
faster. 86400s / 1.25s/it ~ 69120 steps, rounded to 69000. eval_every raised from an initial 900
to 5000 (~14 evals) to avoid burning hours on eval overhead at the corrected step count.
warmup_steps/constant_steps restored to the resid_dropout sweep's 1000/5000.

**Relaunched 2026-08-24 with resid_dropout=0.1, layer_drop=0.1 added** after the unregularized
first attempt (preserved at logs/bytelm_tpu_v4-8_d1024x16_mlp2_24h_noreg,
checkpoints/bytelm_tpu_v4-8_d1024x16_mlp2_24h_noreg) started overfitting: val_bpb bottomed at
step 15000 (1.239) then rose 2 evals in a row (20000: 1.321, 25000: 1.427) -- same combo
(resid_dropout+layer_drop) winning the parallel tpu5 regularization sweep at the time.

    TPU_VISIBLE_CHIPS=1 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d1024x16_mlp2_24h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d1024x16_mlp2_24h
"""
from pathlib import Path

preset = "d1024x16_mlp2"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
resid_dropout = 0.1
layer_drop = 0.1
steps = 69000
batch_size = 4
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 5000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
