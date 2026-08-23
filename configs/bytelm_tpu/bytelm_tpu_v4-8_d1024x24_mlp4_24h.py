"""qcute.bytelm_tpu config: model-size sweep, chip 3 of 4 on tpu4 -- d1024x24_mlp4 preset
(~403.2M params, d_model=1024, n_layers=24, mlp_mult=4 -- the deepest/biggest config in this
sweep, above the ~250-350M anchor for published near-1.0-bpc enwik8 results), context=8192,
flash-attention, no_zero_kv_sink, batch_size=2 (the largest model here needs the smallest batch
to fit). Compares against the other three sibling configs on tpu4's chips 0-2 (d768x12_mlp4
113.7M, d1024x16_mlp2 168M, d1024x16_mlp4 269M).

batch_size=2: bench-probed on this node (2026-08-23) -- bs=2 fine, bs=4/8 both OOM'd (this is
the only config in the sweep that needed to drop to batch=2). 24h step budget, corrected: the
initial estimate (~22.5s/it from a noisy 5-step probe window, giving steps=3800, only ~0.7
epochs -- flagged at the time as a likely undertraining risk) was badly wrong -- the real
observed steady-state rate once actually running was 1.14s/it, ~20x faster, in fact the fastest
of all four sweep configs despite being the largest model. 86400s / 1.14s/it ~ 75789 steps,
rounded to 75000 -- at batch_size=2, seq_len=8193 this covers ~1.23B bytes, ~13.7 epochs, well
above the other three siblings' 2-2.9 epochs. eval_every raised from an initial 900 to 5000
(~15 evals). warmup_steps/constant_steps restored to the resid_dropout sweep's 1000/5000.

    TPU_VISIBLE_CHIPS=3 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d1024x24_mlp4_24h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d1024x24_mlp4_24h
"""
from pathlib import Path

preset = "d1024x24_mlp4"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
steps = 75000
batch_size = 2
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
