"""qcute.bytelm_tpu config: sweep chip 1 of 3 on tpu5 -- resid_dropout only (no layer_drop),
otherwise identical to bytelm_tpu_v4-8_d512_16L_flash_singlechip.py (d512x16, ~67M params,
context=8192, flash-attention, no_zero_kv_sink, batch_size=8). Isolates resid_dropout's effect
on val_bpb overfitting at fixed model size/steps, against the layer_drop-only and
both-together sibling configs run on tpu5's other two idle chips (1/2/3).

24h step budget: chip0's sibling run (same arch/batch/context) measured steady-state 1.22s/it
-- 86400s / 1.22s/it ~ 70800 steps, rounded down to 70000. warmup/constant_steps kept the same
absolute size as the 100k-step config (proportionally shorter cosine-decay tail here).

resid_dropout=0.1: dropout applied to attn/mlp output before the residual add (see
qcute/bytelm_tpu.py's Block.forward) -- kernel-agnostic (unlike SDPA's own attention-weight
dropout, which only applies on the non-flash path and would have forced this run onto the
~18.6s/it non-flash rate measured on tpu4, ~15x slower -- not used for that reason).

    TPU_VISIBLE_CHIPS=1 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_flash_resid_dropout.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_flash_resid_dropout
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
layer_drop = 0.0
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
