"""qcute.qcute_refine_v2 config: layerwise curriculum ablation — QUEUED,
not yet run. Same architecture/optimizer budget as
configs/qcute_refine_v2_3level.py, plus `layer_warmup_steps=(1000, 1000)`:
level 0 (byte LM) trains ALONE for the first 1000 steps (levels 1/2 and
both TokenizerLevels are entirely absent from the forward pass — not
zero-weighted, genuinely not run, no wasted compute); level 1 activates
at step 1000 and trains alongside level 0 for 1000 more steps; level 2
activates at step 2000, after which all 3 levels train together for the
remaining 6000 steps (steps stays 8000 total, matching the baseline
step-budget convention every other config here uses).

Reason (session): let the lower LM's own BSQ codes become stable BEFORE
handing them to the level above as ITS training target — feeding a
still-collapsing/shifting code upward immediately trains the upper level
on a moving target. Each newly-activated level's own parameters get a
FRESH, independent warmup (same shared `lr_at` function every other run
here uses, just reset to 0 at that level's own activation step, via
per-stage optimizer param groups — see qcute_refine_v2.py's
`build_param_groups`/`RefineLM.activation_steps`) rather than jumping
straight to the already-warmed-up peak LR.

Compare against configs/qcute_refine_v2_3level.py (no curriculum, all
levels active from step 0) to isolate whether this actually helps —
watch level1_ntp_acc/level2_ntp_acc and pair0/pair1_tok_acc specifically
around steps 1000/2000 for a visible "does the newly-activated level
start from a better or worse place" signal.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_3level_curriculum.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_3level_curriculum
"""
from pathlib import Path

Ks = (2, 2, 2)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
tier_n_layers = (1, 1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = 128

tok_d_model = 96
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

layer_warmup_steps = (1000, 1000)

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
