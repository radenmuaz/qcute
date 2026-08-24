"""qcute.bytelm_tpu config: power-of-2 model-size + heavier-regularization sweep, config 4 of 4
-- see d512x16_heavyreg.py's docstring for the full sweep rationale.

Config 4 (this file): NEW preset d1024x24_mlp2 (252.2M, narrow/deep -- 24 layers at
d_model=1024, mlp_mult=2). This is a substitute: the originally-planned narrow/deep sibling at
matched params to config 3, d1024x32_mlp2 (32 layers), is CONFIRMED UNUSABLE (2026-08-24,
bench-probed on tpu4) -- batch_size 1/2 both stalled at ~66s/it (killed by a 120s probe timeout
before completing even 2 steps), batch_size=4 hit a genuine OOM (50.87G required vs 31.75G HBM).
This is a real, disproportionate per-step-depth slowdown, not explained by FLOPs alone (the
d1024x24_mlp4 sibling in the old model-size sweep, 24 layers with 2x this config's MLP width,
ran at a normal ~1.1-1.3s/it on the same node) -- not investigated further, out of scope here,
kept as a documented negative result in PRESETS's own comment.

d1024x24_mlp2 (this file, 24 layers instead of 32) shows a milder version of the same anomaly:
bench-probed batch_size=2 works (confirmed via run.log: clean completion, first_step_compile_s
42.5s, step 8 finished normally) but at only ~11.4s/it post-compile -- much slower than
config 2's similarly-sized d1024x16_mlp4 (16 layers, more FLOPs/layer via mlp_mult=4) at
~1.36s/it. batch_size=4 OOMs (36.09G required vs 31.75G HBM).

Real launch (grad_accum_steps=2, effective_batch=4) initially OOM'd (41.69G required vs 31.75G
HBM, ~2x a single microbatch's memory) -- same root cause as config 3's d2048x8_mlp2_wideshallow:
torch_xla's lazy execution piled all microbatches' forward+backward graphs into one unexecuted
HLO graph with no intermediate sync; fixed by adding a mark_step() after each microbatch's
backward() (see bytelm_tpu.py). After the fix, real measured rate is ~1.83s/it per
grad_accum_steps=2 optimizer step (~0.9s/microbatch) -- much faster than the pre-fix probe's
~11.4s/it, which was apparently dominated by post-compile warmup in that short 8-step run, not
true steady state. With this corrected rate, the originally-requested effective_batch=16 is
comfortably reachable (unlike what the misleading probe suggested) -- bumped grad_accum_steps
2->8 (batch_size=2 * grad_accum_steps=8 = effective_batch=16, matching the other 3 configs).
steps=11800 corrected from 86400s / (~0.9s/microbatch * 8 microbatches/step) 2026-08-24.

layer_drop=0.25, resid_dropout=0.1 (heavier layer_drop / lighter resid_dropout than configs 1-3's
uniform 0.15/0.15, per user's explicit instruction to scale layer_drop up with depth for this,
the deepest config in the sweep). label_smoothing=0.1, beta1=0.9, beta2=0.98 unchanged from the
other 3.

    TPU_VISIBLE_CHIPS=<i> uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/power2_heavyreg_sweep/d1024x24_mlp2_deep.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/d1024x24_mlp2_deep
"""
from pathlib import Path

preset = "d1024x24_mlp2"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
mtp_heads = 8
resid_dropout = 0.1
layer_drop = 0.25
label_smoothing = 0.1
beta1 = 0.9
beta2 = 0.98
steps = 23600
batch_size = 2
grad_accum_steps = 8
lr_peak = 1e-4
warmup_steps = 300
cosine_decay = True
constant_steps = 1500
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 1300
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1
