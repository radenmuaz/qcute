"""qcute.qcute_refine_v2 config: FORK of configs/v1_rope.py — new 3-level
architecture (replaces the removed configs/qcute_refine_v2_3level_curriculum.py),
combining the layerwise curriculum with a fresh arch sized against bytelm.

Ks=(2,2,2), context_len=1024: 3 levels, seq_lens=[1024, 512, 256] — same
effective 1024-byte raw context as v1_rope's own Ks=(4,4), just reached
via 3 gentler K=2 compressions instead of 2 K=4 ones. attn_window=
(256,128,64): each level genuinely windowed (window strictly below that
level's own sequence length — no coincidental dense fallback, same
per-level-window convention established this session).

layer_warmup_steps=(500,500): level 0 trains alone for 500 steps, level 1
joins at step 500, level 2 joins at step 1000 — half the (1000,1000) used
in the removed 3level_curriculum config, since this run's own step budget
is also halved (4000, not 8000) — same proportion of the run spent in
each curriculum phase.

Sizing (session decision — asked, answered "match bytelm params"):
tier_d_models=(224,224,224) — chosen so this arch's own param count lands
almost exactly on bytelm_xs_mtp4_ctx1024's 3.412M (this config: 3.414M,
ratio 1.000). FLOPs land at ~5.3 GFLOPs vs bytelm's 11.27 GFLOPs (ratio
0.47) — a real, acknowledged tradeoff: this architecture is more FLOP-
efficient per parameter than bytelm's dense design, so no single width
matches both simultaneously (session finding: matching bytelm's FLOPs
instead would require d~352, landing at 8.25M params, 2.4x bytelm's
count). Params was the dimension chosen to match here.

Verified this session (CPU): forward/backward clean at step=0 (n_active=1,
level 0 alone, matching layer_warmup_steps), generate_no_cache runs
correctly. Note generate_kv_cache is NOT usable with this config (or with
v1_rope, or any windowed-level-0 config in this family) — its own
dense-attention-only assertion is a pre-existing limitation, not
introduced here.

QUEUED — do not launch until the "v1" baseline
(qcute_refine_v2_byte4_code256_simple) finishes; do not touch that
config or its run, or configs/v1_rope.py.

    uv run python -m qcute.qcute_refine_v2 --config configs/v1_rope_3level_curriculum.py

    # plot after training:
    uv run python scripts/plot_run.py logs/v1_rope_3level_curriculum
"""
from pathlib import Path

Ks = (2, 2, 2)
dqs = (8, 8, 8)
tier_d_models = (224, 224, 224)
tier_n_layers = (1, 1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 128, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
layer_warmup_steps = (500, 500)

tok_d_model = 224
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

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
