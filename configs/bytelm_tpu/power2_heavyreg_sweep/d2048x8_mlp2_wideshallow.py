"""qcute.bytelm_tpu config: power-of-2 model-size + heavier-regularization sweep, config 3 of 4
-- see d512x16_heavyreg.py's docstring for the full sweep rationale.

Config 3 (this file): NEW preset d2048x8_mlp2 (336.1M, wide/shallow -- 8 layers at d_model=2048,
mlp_mult=2). Bench-probed on tpu4 (2026-08-24): batch_size 1/2/4 all confirmed safe, no OOM, but
all three landed at a surprisingly uniform ~8.7-8.9s/it post-compile (didn't scale down with
smaller batch as expected -- likely bandwidth/dispatch-bound at this width rather than
compute-bound; not investigated further, out of scope here).

--grad_accum_steps=4 at batch_size=4 gives effective_batch=16 (per user's explicit request to
simulate a bigger batch via grad accumulation). Real launch initially OOM'd (75.72G required vs
31.75G HBM, ~4x a single microbatch's memory) -- root cause: torch_xla's lazy execution piled all
4 microbatches' forward+backward graphs into one unexecuted HLO graph with no intermediate sync;
fixed by adding a mark_step() after each microbatch's backward() in the training loop (see
bytelm_tpu.py). After the fix, real measured rate is ~3.1s/it per optimizer step (much faster
than the original 8.8s/it-based 35.2s/step estimate, which was itself misleading -- likely
dominated by post-compile warmup in the short probe, not true steady state). steps=27000
corrected from 86400s/3.1s/it 2026-08-24.

resid_dropout=0.15, layer_drop=0.15, label_smoothing=0.1, beta1=0.9, beta2=0.98 -- same
heavier-reg/tuned-AdamW settings as configs 1-2.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/power2_heavyreg_sweep/d2048x8_mlp2_wideshallow.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d2048x8_mlp2_wideshallow
"""
from pathlib import Path

preset = "d2048x8_mlp2"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
mtp_heads = 8
resid_dropout = 0.15
layer_drop = 0.15
label_smoothing = 0.1
beta1 = 0.9
beta2 = 0.98
steps = 54000
batch_size = 4
grad_accum_steps = 4
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 3000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
