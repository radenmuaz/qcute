"""qcute.bytelm_tpu config: model-size sweep, chip 0 of 4 on tpu4 -- d768x12_mlp4 preset
(~113.7M params, actual GPT-2-small shape: d_model=768, n_layers=12, not a power-of-2 width like
this repo's other presets), context=8192, flash-attention, no_zero_kv_sink, batch_size=4.
Compares model *size/shape* against the other three sibling configs on tpu4's chips 1-3
(d1024x16_mlp2 168M, d1024x16_mlp4 269M, d1024x24_mlp4 403M) -- all aimed at closing the gap
toward 1.0 bpb (the ~250-350M anchor from published near-1.0-bpc results, Transformer-XL-large
277M / Perceiver-AR 358M, is well above this one's 113.7M -- included as the small/cheap end of
the sweep, not expected to be the winner on its own).

batch_size=4: bench-probed on this node (2026-08-23) -- bs=4 fine, bs=8 OOM'd (out of 2/4/8
tried). 24h step budget, corrected: the initial estimate (~11s/it from a noisy 5-step probe
window, giving steps=7800) was badly wrong -- the real observed steady-state rate once actually
running was 0.75s/it (1.33 it/s), ~15x faster. 86400s / 0.75s/it ~ 115200 steps, rounded to
115000. eval_every raised from an initial 900 (each full val+test pass measured ~123s at
step 900 -- at 115000 steps that would be ~127 evals, ~4.3h of pure eval overhead) to 5000
(~23 evals, ~47min overhead). warmup_steps/constant_steps restored to the resid_dropout sweep's
1000/5000 (appropriate again now that steps is back in that ~60-115k range).

    TPU_VISIBLE_CHIPS=0 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d768x12_mlp4_24h.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d768x12_mlp4_24h
"""
from pathlib import Path

preset = "d768x12_mlp4"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
steps = 115000
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
