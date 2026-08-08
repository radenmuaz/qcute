"""qcute.bytelm config: CLONE of configs/bytelm_xs1_ctx8.py, ONE change:
`context=32` instead of `context=8`.

Session rationale: same sanity-check logic as `bytelm_xs1_ctx8.py` (see
its own docstring), this time targeting the `qcute_refine_v4_k32_narrow*`
family specifically — those configs crippled level 0's receptive field to
`attn_window=32` (Ks=(32,32), narrow-window family, best result so far
2.4926-2.4961 depending on fuse_position/null_kv — docs/status.md,
docs/kv_contribution.md §7-10), well short of every matched baseline.
This isolates the same question for THAT family: how much of the K=32
family's own gap to baselines is just "any model, however simple,
struggling with only 32 bytes of context" versus something specific to
qcute_refine's hierarchical/fusion structure? `bytelm.py` has no window
concept (always dense), so `context=32` is the only way to cripple its
receptive field the same way — every training example is 32 bytes long,
no prediction ever sees more than 31 bytes of history.

Everything else identical to bytelm_xs1_ctx8.py: preset="xs" (d_model=256,
n_heads=4, mtp_heads=4), n_layers=1, steps=4000.

    uv run python -m qcute.bytelm --config configs/bytelm_xs1_ctx32.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs1_ctx32
"""
from pathlib import Path

preset = "xs"
context = 32
n_layers = 1
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 4000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
